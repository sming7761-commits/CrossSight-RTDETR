#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A640-GLF Native Validator
=========================

Purpose
-------
Evaluate the A innovation point with Ultralytics native validation metrics:
    full image RT-DETR prediction
  + 640x640 local slicing prediction
  + coordinate remapping
  + global-local prediction fusion

This file is designed for final paper evaluation, because it reuses Ultralytics'
DetMetrics / AP / PR-curve / F1-curve / confusion-matrix logic instead of the
lightweight custom evaluator used by val_a640_glf.py.

Important
---------
- It does NOT retrain the model.
- It does NOT use GT boxes to crop at test time.
- It does NOT use ROI re-detection, density ROI, or adaptive ROI selection.
- The only A method is: full image + 640 grid slices + prediction fusion.
"""

import argparse
import gc
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from prettytable import PrettyTable

from ultralytics import RTDETR
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.torch_utils import de_parallel, model_info, select_device, smart_inference_mode
from ultralytics.utils.ops import Profile, xywh2xyxy


def make_tiles_fixed(w: int, h: int, tile: int = 640, overlap: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """Conventional fixed-stride 640 sliding-window slicing."""
    if tile <= 0 or (w <= tile and h <= tile):
        return [(0, 0, w, h)]
    step = max(1, int(round(tile * (1.0 - overlap))))

    def starts(length: int) -> List[int]:
        if length <= tile:
            return [0]
        xs = list(range(0, max(length - tile, 0) + 1, step))
        if xs[-1] != length - tile:
            xs.append(length - tile)
        return xs

    tiles = [(x, y, min(x + tile, w), min(y + tile, h)) for y in starts(h) for x in starts(w)]
    return list(dict.fromkeys(tiles))


def make_tiles_adaptive_640(w: int, h: int, tile: int = 640, overlap: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """Adaptive 640 grid slicing.

    The local crop size remains 640. Only the number of crops and actual stride
    are adapted to the image size. No GT boxes, predicted ROIs, or density maps
    are used.
    """
    S = int(tile)
    if S <= 0 or (w <= S and h <= S):
        return [(0, 0, w, h)]

    def adaptive_starts(length: int) -> List[int]:
        if length <= S:
            return [0]
        base_stride = max(1.0, S * (1.0 - overlap))
        n = int(np.ceil((length - S) / base_stride)) + 1
        n = max(2, n)
        stride = (length - S) / float(n - 1)
        starts = [int(round(i * stride)) for i in range(n)]
        starts = [max(0, min(x, length - S)) for x in starts]
        starts[0] = 0
        starts[-1] = length - S
        return list(dict.fromkeys(starts))

    xs = adaptive_starts(w)
    ys = adaptive_starts(h)
    tiles = [(x, y, min(x + S, w), min(y + S, h)) for y in ys for x in xs]
    return list(dict.fromkeys(tiles))


def torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Small dependency-free class-wise NMS."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[rest] - inter + 1e-9
        iou = inter / union
        order = rest[iou <= iou_thr]
    return torch.stack(keep).long() if keep else torch.zeros((0,), dtype=torch.long, device=boxes.device)


def merge_classwise_nms(dets: torch.Tensor, iou_thr: float, max_det: int, conf_thr: float = 0.001) -> torch.Tensor:
    """Merge full-image and local-slice predictions in native image coordinates."""
    if dets is None or dets.numel() == 0:
        return torch.zeros((0, 6), device=dets.device if isinstance(dets, torch.Tensor) else 'cpu')
    dets = dets[dets[:, 4] >= conf_thr]
    if dets.numel() == 0:
        return torch.zeros((0, 6), device=dets.device)
    out = []
    for c in dets[:, 5].unique():
        idx = (dets[:, 5] == c).nonzero(as_tuple=False).squeeze(1)
        keep = torch_nms(dets[idx, :4], dets[idx, 4], iou_thr)
        if keep.numel():
            out.append(dets[idx[keep]])
    if not out:
        return torch.zeros((0, 6), device=dets.device)
    merged = torch.cat(out, 0)
    order = merged[:, 4].argsort(descending=True)
    return merged[order[:max_det]]


class A640RTDETRNativeValidator(RTDETRValidator):
    """RT-DETR validator that keeps Ultralytics metrics but replaces inference with A640-GLF."""

    def __init__(self, *args, a640_mode='adaptive', tile=640, overlap=0.20, merge_iou=0.55,
                 view_batch=8, **kwargs):
        super().__init__(*args, **kwargs)
        assert a640_mode in {'no_slice', 'fixed', 'adaptive', 'slice_only'}
        self.a640_mode = a640_mode
        self.tile = int(tile)
        self.overlap = float(overlap)
        self.merge_iou = float(merge_iou)
        self.view_batch = int(view_batch)
        self.strategy_counter = Counter()
        self.total_local_tiles = 0
        self.total_views = 0
        self.model_ref = None

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Validation loop. Only standalone validation is supported for A640-GLF."""
        if trainer is not None:
            raise RuntimeError('A640 native validation is intended for standalone validation, not in-training validation.')

        self.training = False
        augment = self.args.augment
        callbacks.add_integration_callbacks(self)
        model = AutoBackend(model or self.args.model,
                            device=select_device(self.args.device, self.args.batch),
                            dnn=self.args.dnn,
                            data=self.args.data,
                            fp16=self.args.half)
        self.model_ref = model
        self.model = model
        self.device = model.device
        self.args.half = model.fp16
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        self.args.imgsz = imgsz
        if engine:
            self.args.batch = model.batch_size
        elif not pt and not jit:
            self.args.batch = 1
            LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        self.data = check_det_dataset(self.args.data)
        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0
        if not pt:
            self.args.rect = False
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        model.eval()
        model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))

        self.run_callbacks('on_val_start')
        dt = Profile(), Profile(), Profile(), Profile()
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []

        for batch_i, batch in enumerate(bar):
            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(batch)
            with dt[1]:
                preds = self.a640_predict_batch(model, batch, augment=augment)
            with dt[2]:
                pass
            with dt[3]:
                pass

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            self.run_callbacks('on_val_batch_end')

        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks('on_val_end')

        LOGGER.info('Speed: %.1fms preprocess, %.1fms A640-inference, %.1fms loss, %.1fms postprocess per image' %
                    tuple(self.speed.values()))
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                LOGGER.info(f'Saving {f.name}...')
                json.dump(self.jdict, f)
            stats = self.eval_json(stats)
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats

    def _postprocess_rtdetr(self, raw_preds):
        """RT-DETR raw output -> list of [xyxy, conf, cls] in square imgsz coordinates."""
        return super().postprocess(raw_preds)

    def _square_to_native(self, pred: torch.Tensor, shape_hw: Tuple[int, int]) -> torch.Tensor:
        """Map RT-DETR square-input coordinates to original image coordinates."""
        if pred is None or pred.numel() == 0:
            return torch.zeros((0, 6), device=self.device)
        h, w = int(shape_hw[0]), int(shape_hw[1])
        out = pred.clone()
        out[:, [0, 2]] *= w / float(self.args.imgsz)
        out[:, [1, 3]] *= h / float(self.args.imgsz)
        out[:, [0, 2]] = out[:, [0, 2]].clamp(0, w)
        out[:, [1, 3]] = out[:, [1, 3]].clamp(0, h)
        return out

    def _crop_to_tensor(self, im0_bgr: np.ndarray, tile: Tuple[int, int, int, int]) -> torch.Tensor:
        x1, y1, x2, y2 = tile
        crop = im0_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.args.imgsz, self.args.imgsz), interpolation=cv2.INTER_LINEAR)
        crop = np.ascontiguousarray(crop.transpose(2, 0, 1)[::-1])  # BGR HWC -> RGB CHW
        return torch.from_numpy(crop)

    def _local_predictions(self, model, batch, augment=False) -> List[List[torch.Tensor]]:
        """Run local 640 tiles and return per-image native-coordinate predictions."""
        local_per_image: List[List[torch.Tensor]] = [[] for _ in batch['im_file']]
        tensor_views = []
        metas = []  # (batch_index, x1, y1, crop_w, crop_h, W, H)

        for si, im_file in enumerate(batch['im_file']):
            im0 = cv2.imread(str(im_file))
            if im0 is None:
                LOGGER.warning(f'WARNING ⚠️ cannot read image: {im_file}')
                continue
            H, W = im0.shape[:2]
            if self.a640_mode == 'fixed':
                tiles = make_tiles_fixed(W, H, self.tile, self.overlap)
                strategy = f'full_plus_fixed_{self.tile}_o{self.overlap}'
            elif self.a640_mode == 'adaptive':
                tiles = make_tiles_adaptive_640(W, H, self.tile, self.overlap)
                strategy = f'adaptive_{self.tile}_grid_o{self.overlap}'
            elif self.a640_mode == 'slice_only':
                tiles = make_tiles_adaptive_640(W, H, self.tile, self.overlap)
                strategy = f'slice_only_adaptive_{self.tile}'
            else:
                tiles = []
                strategy = 'no_slice'

            if self.a640_mode in {'fixed', 'adaptive'}:
                tiles = [t for t in tiles if t != (0, 0, W, H)]
            self.strategy_counter[strategy] += 1
            self.total_local_tiles += len(tiles)
            self.total_views += len(tiles) + (0 if self.a640_mode == 'slice_only' else 1)

            for t in tiles:
                x1, y1, x2, y2 = t
                tensor_views.append(self._crop_to_tensor(im0, t))
                metas.append((si, x1, y1, x2 - x1, y2 - y1, W, H))

        if not tensor_views:
            return local_per_image

        for st in range(0, len(tensor_views), max(1, self.view_batch)):
            chunk = torch.stack(tensor_views[st:st + self.view_batch], 0).to(self.device, non_blocking=True)
            chunk = (chunk.half() if self.args.half else chunk.float()) / 255.0
            raw = model(chunk, augment=augment)
            preds = self._postprocess_rtdetr(raw)
            for pred, meta in zip(preds, metas[st:st + self.view_batch]):
                si, ox, oy, cw, ch, W, H = meta
                if pred is None or pred.numel() == 0:
                    continue
                p = pred.clone()
                p[:, [0, 2]] *= cw / float(self.args.imgsz)
                p[:, [1, 3]] *= ch / float(self.args.imgsz)
                p[:, [0, 2]] += ox
                p[:, [1, 3]] += oy
                p[:, [0, 2]] = p[:, [0, 2]].clamp(0, W)
                p[:, [1, 3]] = p[:, [1, 3]].clamp(0, H)
                area = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
                p = p[area > 1.0]
                if p.numel():
                    local_per_image[si].append(p)
        return local_per_image

    def a640_predict_batch(self, model, batch, augment=False):
        """Return final fused predictions in original image coordinates for each image in a batch."""
        bs = batch['img'].shape[0]
        full_native: List[torch.Tensor] = [torch.zeros((0, 6), device=self.device) for _ in range(bs)]

        if self.a640_mode != 'slice_only':
            raw_full = model(batch['img'], augment=augment)
            full_square = self._postprocess_rtdetr(raw_full)
            for si, pred in enumerate(full_square):
                full_native[si] = self._square_to_native(pred, batch['ori_shape'][si])
        else:
            # still count views in local function; no full-image prediction is used
            pass

        if self.a640_mode == 'no_slice':
            # keep native full-image predictions; no fusion with slices
            self.strategy_counter['no_slice'] += bs
            self.total_views += bs
            return [p[p[:, 4] >= self.args.conf] if p.numel() else p for p in full_native]

        local_per_image = self._local_predictions(model, batch, augment=augment)
        final = []
        for si in range(bs):
            parts = []
            if self.a640_mode != 'slice_only' and full_native[si].numel():
                parts.append(full_native[si])
            if local_per_image[si]:
                parts.extend(local_per_image[si])
            if parts:
                det = torch.cat(parts, 0)
                det = merge_classwise_nms(det, self.merge_iou, self.args.max_det, conf_thr=self.args.conf)
            else:
                det = torch.zeros((0, 6), device=self.device)
            final.append(det)
        return final

    def update_metrics(self, preds, batch):
        """Ultralytics native metric update, but predictions are already in native image coordinates."""
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()  # already native-space xyxy

            if nl:
                tbox = xywh2xyxy(bbox)
                tbox[..., [0, 2]] *= int(shape[1])
                tbox[..., [1, 3]] *= int(shape[0])
                labelsn = torch.cat((cls, tbox), 1)
                correct_bboxes = self._process_batch(predn.float(), labelsn)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, predn[:, 4], predn[:, 5], cls.squeeze(-1)))

            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def plot_predictions(self, batch, preds, ni):
        """Plot predictions on the stretched validation batch image."""
        h_img, w_img = batch['img'].shape[2:]
        plot_preds = []
        for si, pred in enumerate(preds):
            p = pred.clone()
            if p.numel():
                H, W = batch['ori_shape'][si]
                p[:, [0, 2]] *= w_img / float(W)
                p[:, [1, 3]] *= h_img / float(H)
            plot_preds.append(p)
        plot_images(batch['img'],
                    *output_to_target(plot_preds, max_det=self.args.max_det),
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names,
                    on_plot=self.on_plot)


def get_weight_size(path):
    return f'{os.stat(path).st_size / 1024 / 1024:.1f}'


def parse_args():
    p = argparse.ArgumentParser(description='A640-GLF native Ultralytics validation')
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--data', type=str, default='dataset/data.yaml')
    p.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--batch', type=int, default=8, help='dataset batch size')
    p.add_argument('--view-batch', type=int, default=8, help='local-slice inference batch size')
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--max-det', type=int, default=1000)
    p.add_argument('--mode', type=str, default='adaptive', choices=['no_slice', 'fixed', 'adaptive', 'slice_only'])
    p.add_argument('--tile', type=int, default=640)
    p.add_argument('--overlap', type=float, default=0.20)
    p.add_argument('--merge-iou', type=float, default=0.55)
    p.add_argument('--project', type=str, default='runs/val_a640_native')
    p.add_argument('--name', type=str, default='A640_GLF_adaptive_native')
    p.add_argument('--plots', action='store_true', default=True)
    p.add_argument('--save-txt', action='store_true')
    p.add_argument('--save-conf', action='store_true')
    p.add_argument('--save-json', action='store_true')
    p.add_argument('--half', action='store_true')
    p.add_argument('--force-exit', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.tile != 640:
        LOGGER.warning(f'WARNING ⚠️ A640-GLF is designed for 640x640 local views, but tile={args.tile} was given.')

    validator_args = dict(
        model=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        plots=args.plots,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_json=args.save_json,
        half=args.half,
        exist_ok=True,
        verbose=True,
        rect=False,
    )

    validator = A640RTDETRNativeValidator(
        args=validator_args,
        a640_mode=args.mode,
        tile=args.tile,
        overlap=args.overlap,
        merge_iou=args.merge_iou,
        view_batch=args.view_batch,
    )
    stats = validator()

    # Prepare paper-style summary consistent with your original val.py.
    try:
        n_l, n_p, n_g, flops = model_info(validator.model_ref.model)
    except Exception:
        tmp_model = RTDETR(args.weights)
        n_l, n_p, n_g, flops = model_info(tmp_model.model)

    metrics = validator.metrics
    model_names = list(metrics.names.values()) if isinstance(metrics.names, dict) else list(metrics.names)
    length = metrics.box.p.size
    preprocess = validator.speed.get('preprocess', 0.0)
    inference = validator.speed.get('inference', 0.0)
    postprocess = validator.speed.get('postprocess', 0.0)
    all_time = preprocess + inference + postprocess

    model_info_table = PrettyTable()
    model_info_table.title = 'Model Info - Native A640 Validator'
    model_info_table.field_names = [
        'GFLOPs', 'Parameters', '前处理时间/一张图', 'A640推理时间/一张图',
        '后处理时间/一张图', 'FPS(前处理+A640推理+后处理)', 'Model File Size'
    ]
    model_info_table.add_row([
        f'{flops:.1f}', f'{n_p:,}', f'{preprocess / 1000:.6f}s', f'{inference / 1000:.6f}s',
        f'{postprocess / 1000:.6f}s', f'{1000 / all_time:.2f}' if all_time > 0 else '0.00',
        f'{get_weight_size(args.weights)}MB'
    ])

    metric_table = PrettyTable()
    metric_table.title = 'Model Metric - Ultralytics Native Metrics'
    metric_table.field_names = ['Class Name', 'Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP75', 'mAP50-95']
    for idx in range(length):
        metric_table.add_row([
            model_names[idx] if idx < len(model_names) else str(idx),
            f'{metrics.box.p[idx]:.4f}',
            f'{metrics.box.r[idx]:.4f}',
            f'{metrics.box.f1[idx]:.4f}',
            f'{metrics.box.ap50[idx]:.4f}',
            f'{metrics.box.all_ap[idx, 5]:.4f}',
            f'{metrics.box.ap[idx]:.4f}',
        ])

    precision = stats.get('metrics/precision(B)', 0.0)
    recall = stats.get('metrics/recall(B)', 0.0)
    map50 = stats.get('metrics/mAP50(B)', 0.0)
    map5095 = stats.get('metrics/mAP50-95(B)', 0.0)
    f1 = float(np.mean(metrics.box.f1[:length])) if length else 0.0
    map75 = float(np.mean(metrics.box.all_ap[:length, 5])) if length else 0.0
    metric_table.add_row(['all(平均数据)', f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}', f'{map50:.4f}', f'{map75:.4f}', f'{map5095:.4f}'])

    strategy_table = PrettyTable()
    strategy_table.title = 'A640-GLF Strategy Info'
    strategy_table.field_names = ['策略', '图片数量/数值']
    for k, v in validator.strategy_counter.items():
        strategy_table.add_row([k, v])
    n_img = max(validator.seen or 1, 1)
    strategy_table.add_row(['平均局部切片数/图', f'{validator.total_local_tiles / n_img:.2f}'])
    strategy_table.add_row(['平均总视图数/图', f'{validator.total_views / n_img:.2f}'])

    paper = PrettyTable()
    paper.title = 'Paper Summary - Ultralytics Native Metrics'
    paper.field_names = ['Model', 'Precision (%)', 'Recall (%)', 'mAP50 (%)', 'mAP50:95 (%)', 'Params (M)', 'GFLOPs', 'FPS']
    paper.add_row([
        args.name,
        f'{precision * 100:.2f}',
        f'{recall * 100:.2f}',
        f'{map50 * 100:.2f}',
        f'{map5095 * 100:.2f}',
        f'{n_p / 1_000_000:.2f}',
        f'{flops:.1f}',
        f'{1000 / all_time:.2f}' if all_time > 0 else '0.00',
    ])

    print('-' * 20 + ' 论文数据：A640 使用 Ultralytics 原生指标口径 ' + '-' * 20)
    print(model_info_table)
    print(strategy_table)
    print(metric_table)
    print(paper)

    save_dir = validator.save_dir
    with open(save_dir / 'paper_data.txt', 'w', encoding='utf-8', errors='ignore') as f:
        f.write(str(model_info_table) + '\n\n')
        f.write(str(strategy_table) + '\n\n')
        f.write(str(metric_table) + '\n\n')
        f.write(str(paper) + '\n')

    with open(save_dir / 'metrics_summary_native.json', 'w', encoding='utf-8') as f:
        json.dump({
            'name': args.name,
            'mode': args.mode,
            'tile': args.tile,
            'target_overlap': args.overlap,
            'merge_iou': args.merge_iou,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'map50': float(map50),
            'map75': float(map75),
            'map50_95': float(map5095),
            'fps': float(1000 / all_time) if all_time > 0 else 0.0,
            'params_m': float(n_p / 1_000_000),
            'gflops': float(flops),
            'avg_local_tiles_per_image': float(validator.total_local_tiles / n_img),
            'avg_views_per_image': float(validator.total_views / n_img),
            'strategy_counter': dict(validator.strategy_counter),
        }, f, ensure_ascii=False, indent=2)

    print('-' * 20, f'结果已保存至 {save_dir / "paper_data.txt"}', '-' * 20)

    if args.force_exit:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            os._exit(0)


if __name__ == '__main__':
    main()
