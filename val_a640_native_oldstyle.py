#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
val_a640_native_oldstyle.py

使用 Ultralytics / RT-DETR 原生验证器的指标体系，只替换推理阶段：
- no_slice：原生整图推理，作为校验项，应与 val.py 基本一致
- fixed：整图 + 固定切片推理 + 融合，再交给原生 metrics 计算 mAP
- slice_only：只用固定切片推理
- adaptive：沿用旧版 tensor-space 切片流程，生成自适应 640 网格 + 整图融合，再交给原生 metrics 计算 mAP

注意：本脚本不训练、不改权重、不改标签匹配和 AP 计算。
"""

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable

from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, model_info, select_device, smart_inference_mode
from ultralytics.data.utils import check_cls_dataset, check_det_dataset


def get_weight_size(path):
    return f"{os.stat(path).st_size / 1024 / 1024:.1f}"


def make_tiles(w, h, tile, overlap):
    tile = int(tile)
    if tile <= 0 or (w <= tile and h <= tile):
        return [(0, 0, w, h)]
    step = max(1, int(tile * (1.0 - float(overlap))))

    def starts(length):
        if length <= tile:
            return [0]
        xs = list(range(0, max(length - tile, 0) + 1, step))
        if xs[-1] != length - tile:
            xs.append(length - tile)
        return xs

    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in starts(h) for x in starts(w)]



def make_tiles_adaptive_input(w, h, tile, overlap):
    """
    旧版兼容的自适应切片：仍然在 Ultralytics 预处理后的网络输入张量上切片，
    不直接裁剪原图像素，避免和旧版 full+640 的尺度口径不一致。
    对 960 输入和 640 tile，通常会得到与旧 fixed 相同的 2x2 局部视图；
    对非标准输入，则自适应调整步长，让边界覆盖更均匀。
    """
    tile = int(tile)
    if tile <= 0 or (w <= tile and h <= tile):
        return [(0, 0, w, h)]

    def starts(length):
        if length <= tile:
            return [0]
        base_stride = max(1.0, tile * (1.0 - float(overlap)))
        n = int(np.ceil((length - tile) / base_stride)) + 1
        n = max(2, n)
        stride = (length - tile) / float(n - 1)
        xs = [int(round(i * stride)) for i in range(n)]
        xs = [max(0, min(x, length - tile)) for x in xs]
        xs[0] = 0
        xs[-1] = length - tile
        return list(dict.fromkeys(xs))

    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in starts(h) for x in starts(w)]

def dedup_tiles(tiles):
    out, seen = [], set()
    for t in tiles:
        t = tuple(int(v) for v in t)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out




def apply_bcs_score(p, x1, y1, x2, y2, W, H, cfg):
    """
    BCS：Boundary Consistency Score，边界一致性得分调制。
    作用：对切片边界附近的预测框进行软置信度抑制，减少切片边缘伪框。
    注意：不删框，只调分，尽量不伤召回。
    """
    if not getattr(cfg, "bcs", False):
        return p
    if p is None or p.numel() == 0:
        return p

    # 整图不做 BCS，只对切片结果做
    if int(x1) == 0 and int(y1) == 0 and int(x2) >= int(W) and int(y2) >= int(H):
        return p

    margin = float(getattr(cfg, "bcs_margin", 0.08))
    penalty = float(getattr(cfg, "bcs_penalty", 0.85))
    min_conf = float(getattr(cfg, "bcs_min_conf", 0.0))

    if margin <= 0 or penalty >= 1.0:
        return p

    tile_w = max(float(x2 - x1), 1.0)
    tile_h = max(float(y2 - y1), 1.0)
    base = max(min(tile_w, tile_h), 1.0)

    cx = (p[:, 0] + p[:, 2]) / 2.0
    cy = (p[:, 1] + p[:, 3]) / 2.0

    d_left = cx - float(x1)
    d_right = float(x2) - cx
    d_top = cy - float(y1)
    d_bottom = float(y2) - cy

    dmin = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_top, d_bottom)) / base
    border_mask = dmin < margin

    if border_mask.any():
        # 越靠近切片边界，置信度越低；靠近内部则几乎不变
        factor = penalty + (1.0 - penalty) * (dmin[border_mask] / margin).clamp(0.0, 1.0)
        p[border_mask, 4] = p[border_mask, 4] * factor

    if min_conf > 0:
        p = p[p[:, 4] >= min_conf]

    return p


def torch_nms(boxes, scores, iou_thr):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


def merge_dets_torch(dets, merge_iou=0.55, max_det=1000):
    if dets is None or dets.numel() == 0:
        device = dets.device if dets is not None else torch.device('cpu')
        return torch.zeros((0, 6), device=device)
    dets = dets[dets[:, 4].argsort(descending=True)]
    outs = []
    for c in dets[:, 5].unique():
        m = dets[:, 5] == c
        dc = dets[m]
        keep = torch_nms(dc[:, :4], dc[:, 4], float(merge_iou))
        if keep.numel():
            outs.append(dc[keep])
    if not outs:
        return dets[:0]
    out = torch.cat(outs, dim=0)
    out = out[out[:, 4].argsort(descending=True)]
    return out[:int(max_det)]



def apply_sbf_score(p, meta, cfg, W, H):
    """
    SBF: Scale-aware Boundary Fusion（尺度/边界感知融合）
    只改融合前的置信度排序，不改模型结构、不改GT、不改AP计算。
    - 切片中心区域的小目标：轻微加权
    - 靠近切片边界的框：轻微降权，降低截断框/重复框影响
    """
    if p is None or p.numel() == 0 or getattr(cfg, "fusion", "nms") != "sbf":
        return p

    bi, x1, y1, x2, y2 = meta

    # 整图检测结果不处理，只处理切片检测结果
    if int(x1) == 0 and int(y1) == 0 and int(x2) == int(W) and int(y2) == int(H):
        return p

    cx = (p[:, 0] + p[:, 2]) * 0.5
    cy = (p[:, 1] + p[:, 3]) * 0.5

    dist = torch.minimum(
        torch.minimum(cx - float(x1), float(x2) - cx),
        torch.minimum(cy - float(y1), float(y2) - cy)
    )

    boundary = dist < float(getattr(cfg, "sbf_margin", 48))

    area = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    area_ratio = area / max(float(W * H), 1.0)

    small = area_ratio <= float(getattr(cfg, "sbf_small_thr", 0.006))
    center_ok = ~boundary

    factor = torch.ones_like(p[:, 4])
    factor = torch.where(small & center_ok, factor * float(getattr(cfg, "sbf_boost", 1.06)), factor)
    factor = torch.where(boundary, factor * float(getattr(cfg, "sbf_decay", 0.92)), factor)

    p[:, 4] = (p[:, 4] * factor).clamp(0.0, 0.999)
    return p


def density_score_from_dets(dets, w, h, cfg):
    if dets is None or dets.numel() == 0:
        return 0.0, {"total_det": 0, "small_det": 0, "weak_small_det": 0, "grid_peak": 0.0}
    d = dets.detach()
    area_ratio = ((d[:, 2] - d[:, 0]).clamp(min=0) * (d[:, 3] - d[:, 1]).clamp(min=0)) / max(float(w * h), 1.0)
    valid = d[:, 4] >= float(cfg.density_conf)
    small = valid & (area_ratio <= float(cfg.small_area_thr))
    small_d = d[small]
    weak_ids = torch.tensor([0, 1, 2, 6, 7, 9], device=d.device)
    weak = (d[:, 5:6].int() == weak_ids.view(1, -1)).any(dim=1) & small
    grid_peak = 0.0
    if small_d.shape[0] > 0:
        cx = (small_d[:, 0] + small_d[:, 2]) / 2.0
        cy = (small_d[:, 1] + small_d[:, 3]) / 2.0
        gx = torch.clamp((cx / max(w, 1) * int(cfg.grid)).long(), 0, int(cfg.grid) - 1)
        gy = torch.clamp((cy / max(h, 1) * int(cfg.grid)).long(), 0, int(cfg.grid) - 1)
        hist = torch.zeros((int(cfg.grid), int(cfg.grid)), device=d.device)
        for x, y in zip(gx, gy):
            hist[y, x] += 1.0
        grid_peak = float((hist.max() / max(int(small_d.shape[0]), 1)).detach().cpu())
    small_count = int(small.sum().detach().cpu())
    weak_count = int(weak.sum().detach().cpu())
    score = 0.45 * min(small_count / 15.0, 1.0) + 0.30 * min(weak_count / 8.0, 1.0) + 0.25 * grid_peak
    return float(score), {
        "total_det": int(valid.sum().detach().cpu()),
        "small_det": small_count,
        "weak_small_det": weak_count,
        "grid_peak": grid_peak,
    }


def make_density_rois(dets, w, h, cfg):
    if dets is None or dets.numel() == 0 or int(cfg.max_rois) <= 0:
        return []
    d = dets.detach()
    area_ratio = ((d[:, 2] - d[:, 0]).clamp(min=0) * (d[:, 3] - d[:, 1]).clamp(min=0)) / max(float(w * h), 1.0)
    keep = (d[:, 4] >= float(cfg.density_conf)) & (area_ratio <= float(cfg.small_area_thr))
    small = d[keep]
    if small.shape[0] == 0:
        return []
    small = small[small[:, 4].argsort(descending=True)[:int(cfg.max_rois)]]
    tile = min(int(cfg.roi_tile), max(w, h))
    rois = []
    for box in small:
        cx = float(((box[0] + box[2]) / 2.0).detach().cpu())
        cy = float(((box[1] + box[3]) / 2.0).detach().cpu())
        x1 = int(round(cx - tile / 2.0))
        y1 = int(round(cy - tile / 2.0))
        x1 = max(0, min(x1, max(0, w - tile)))
        y1 = max(0, min(y1, max(0, h - tile)))
        rois.append((x1, y1, min(w, x1 + tile), min(h, y1 + tile)))
    return dedup_tiles(rois)


def decide_das_tiles(score, full_det, w, h, cfg):
    if cfg.das_policy == "accuracy":
        tiles = make_tiles(w, h, int(cfg.tile), float(cfg.overlap))
        if bool(cfg.roi_slicing) and score >= float(cfg.density_high):
            tiles = dedup_tiles(tiles + make_density_rois(full_det, w, h, cfg))
        return tiles
    if cfg.das_policy == "balanced":
        if score >= float(cfg.density_high):
            return make_tiles(w, h, 768, 0.25)
        if score >= float(cfg.density_low):
            return make_tiles(w, h, 800, 0.25)
        return make_tiles(w, h, 800, 0.20)
    # speed
    if score < float(cfg.density_low):
        return []
    if score >= float(cfg.density_high):
        return make_tiles(w, h, 800, 0.25)
    return make_tiles(w, h, 800, 0.20)


class RTDETRA640OldStyleValidator(RTDETRValidator):
    def __init__(self, *args, slice_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_cfg = slice_cfg or SimpleNamespace(mode="no_slice")
        self.das_details = []

    def _post_raw(self, raw):
        return RTDETRValidator.postprocess(self, raw)

    def _run_crops(self, model, img_batch, crop_items):
        """crop_items: list of (image_index, x1, y1, x2, y2), coords in network input space."""
        cfg = self.slice_cfg
        if not crop_items:
            return [torch.zeros((0, 6), device=img_batch.device) for _ in range(img_batch.shape[0])]
        outs = [[] for _ in range(img_batch.shape[0])]
        crop_tensors, metas = [], []
        H, W = int(img_batch.shape[2]), int(img_batch.shape[3])
        for item in crop_items:
            bi, x1, y1, x2, y2 = item
            crop = img_batch[bi:bi + 1, :, y1:y2, x1:x2]
            if crop.shape[2] != int(self.args.imgsz) or crop.shape[3] != int(self.args.imgsz):
                crop = F.interpolate(crop, size=(int(self.args.imgsz), int(self.args.imgsz)), mode="bilinear", align_corners=False)
            crop_tensors.append(crop)
            metas.append((bi, x1, y1, x2, y2))

        sb = max(1, int(getattr(cfg, "slice_batch", self.args.batch)))
        for st in range(0, len(crop_tensors), sb):
            x = torch.cat(crop_tensors[st:st + sb], dim=0)
            raw = model(x, augment=False)
            preds = self._post_raw(raw)
            for pred, meta in zip(preds, metas[st:st + sb]):
                bi, x1, y1, x2, y2 = meta
                if pred is None or pred.numel() == 0:
                    continue
                p = pred.clone()
                sx = float(x2 - x1) / float(self.args.imgsz)
                sy = float(y2 - y1) / float(self.args.imgsz)
                p[:, [0, 2]] = p[:, [0, 2]] * sx + float(x1)
                p[:, [1, 3]] = p[:, [1, 3]] * sy + float(y1)
                p[:, [0, 2]] = p[:, [0, 2]].clamp(0, W)
                p[:, [1, 3]] = p[:, [1, 3]].clamp(0, H)
                area = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
                p = p[area > 1.0]
                if p.numel():
                    p = apply_sbf_score(p, meta, cfg, W, H)
                    outs[bi].append(p)

        merged = []
        for parts in outs:
            if not parts:
                merged.append(torch.zeros((0, 6), device=img_batch.device))
            else:
                merged.append(merge_dets_torch(torch.cat(parts, dim=0), cfg.merge_iou, self.args.max_det))
        return merged

    def _sliced_inference(self, model, img_batch):
        cfg = self.slice_cfg
        B, _, H, W = img_batch.shape
        H, W = int(H), int(W)
        full_tile = (0, 0, W, H)

        if cfg.mode == "no_slice":
            raw = model(img_batch, augment=False)
            return self._post_raw(raw)

        if cfg.mode in ("fixed", "adaptive", "slice_only"):
            base_tiles = make_tiles_adaptive_input(W, H, int(cfg.tile), float(cfg.overlap)) if cfg.mode == "adaptive" else make_tiles(W, H, int(cfg.tile), float(cfg.overlap))
            crop_items = []
            for bi in range(B):
                tiles = list(base_tiles)
                if cfg.mode in ("fixed", "adaptive"):
                    tiles = dedup_tiles([full_tile] + tiles)
                else:
                    tiles = dedup_tiles(tiles)
                for t in tiles:
                    crop_items.append((bi, *t))
                self.das_details.append({"mode": cfg.mode, "tiles": len(tiles)})
            return self._run_crops(model, img_batch, crop_items)

        # DAS: 先整图粗检，再根据密度决定局部切片。
        raw_full = model(img_batch, augment=False)
        full_preds = self._post_raw(raw_full)
        all_items = []
        per_image_full = [[] for _ in range(B)]
        for bi in range(B):
            full_det = full_preds[bi]
            per_image_full[bi].append(full_det)
            score, info = density_score_from_dets(full_det, W, H, cfg)
            tiles = decide_das_tiles(score, full_det, W, H, cfg)
            tiles = [t for t in dedup_tiles(tiles) if t != full_tile]
            for t in tiles:
                all_items.append((bi, *t))
            info.update({"mode": "das", "score": score, "tiles": len(tiles)})
            self.das_details.append(info)

        local_preds = self._run_crops(model, img_batch, all_items) if all_items else [torch.zeros((0, 6), device=img_batch.device) for _ in range(B)]
        out = []
        for bi in range(B):
            parts = []
            if per_image_full[bi][0] is not None and per_image_full[bi][0].numel():
                parts.append(per_image_full[bi][0])
            if local_preds[bi] is not None and local_preds[bi].numel():
                parts.append(local_preds[bi])
            if not parts:
                out.append(torch.zeros((0, 6), device=img_batch.device))
            else:
                out.append(merge_dets_torch(torch.cat(parts, dim=0), cfg.merge_iou, self.args.max_det))
        return out

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            raise RuntimeError("val_das_native.py 只用于独立验证，不用于训练中验证。")

        callbacks.add_integration_callbacks(self)
        model = AutoBackend(model or self.args.model,
                            device=select_device(self.args.device, self.args.batch),
                            dnn=self.args.dnn,
                            data=self.args.data,
                            fp16=self.args.half)
        self.device = model.device
        self.args.half = model.fp16
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        elif not pt and not jit:
            self.args.batch = 1
            LOGGER.info(f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        if isinstance(self.args.data, str) and self.args.data.split('.')[-1] in ('yaml', 'yml'):
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == 'classify':
            self.data = check_cls_dataset(self.args.data, split=self.args.split)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

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
                if self.slice_cfg.mode == "no_slice":
                    preds_raw = model(batch['img'], augment=augment)
                else:
                    preds = self._sliced_inference(model, batch['img'])
                    preds_raw = None
            with dt[2]:
                pass
            with dt[3]:
                if self.slice_cfg.mode == "no_slice":
                    preds = self.postprocess(preds_raw)

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
        LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' % tuple(self.speed.values()))
        if self.args.save_json and self.jdict:
            import json
            with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                LOGGER.info(f'Saving {f.name}...')
                json.dump(self.jdict, f)
            stats = self.eval_json(stats)
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--data', type=str, default='dataset/data.yaml')
    p.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    p.add_argument('--name', type=str, default='A640_oldstyle')
    p.add_argument('--project', type=str, default='runs/val_a640_oldstyle')
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--plots', action='store_true', default=True)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--max-det', type=int, default=1000)
    p.add_argument('--half', action='store_true', default=False)
    p.add_argument('--save-txt', action='store_true', default=False)
    p.add_argument('--save-conf', action='store_true', default=False)
    p.add_argument('--save-json', action='store_true', default=False)

    p.add_argument('--mode', type=str, default='no_slice', choices=['no_slice', 'fixed', 'adaptive', 'slice_only', 'das'])
    p.add_argument('--tile', type=int, default=640)
    p.add_argument('--overlap', type=float, default=0.20)
    p.add_argument('--slice-batch', type=int, default=8)
    p.add_argument('--merge-iou', type=float, default=0.55)
    p.add_argument('--bcs', action='store_true', help='开启 BCS 边界一致性得分调制')
    p.add_argument('--bcs-margin', type=float, default=0.08)
    p.add_argument('--bcs-penalty', type=float, default=0.85)
    p.add_argument('--bcs-min-conf', type=float, default=0.0)
    p.add_argument('--fusion', type=str, default='nms', choices=['nms', 'sbf'])
    p.add_argument('--sbf-boost', type=float, default=1.06)
    p.add_argument('--sbf-decay', type=float, default=0.92)
    p.add_argument('--sbf-margin', type=float, default=48)
    p.add_argument('--sbf-small-thr', type=float, default=0.006)
    p.add_argument('--das-policy', type=str, default='accuracy', choices=['accuracy', 'balanced', 'speed'])
    p.add_argument('--density-conf', type=float, default=0.05)
    p.add_argument('--small-area-thr', type=float, default=0.006)
    p.add_argument('--density-low', type=float, default=0.15)
    p.add_argument('--density-high', type=float, default=0.45)
    p.add_argument('--grid', type=int, default=3)
    p.add_argument('--roi-slicing', action='store_true')
    p.add_argument('--roi-tile', type=int, default=640)
    p.add_argument('--max-rois', type=int, default=8)
    p.add_argument('--force-exit', action='store_true')
    return p.parse_args()


def write_paper_data(model, result, args, slice_cfg):
    length = result.box.p.size
    model_names = list(result.names.values())
    mode_label = slice_cfg.mode + ('+BCS' if getattr(slice_cfg, 'bcs', False) else '')
    preprocess_time = result.speed['preprocess']
    inference_time = result.speed['inference']
    postprocess_time = result.speed['postprocess']
    all_time = preprocess_time + inference_time + postprocess_time
    _, n_p, _, flops = model_info(model.model)

    model_info_table = PrettyTable()
    model_info_table.title = 'Model Info'
    model_info_table.field_names = ['GFLOPs', 'Parameters', '前处理时间/一张图', '推理时间/一张图', '后处理时间/一张图', 'FPS(总流程)', 'FPS(推理)', 'Model File Size']
    model_info_table.add_row([
        f'{flops:.1f}', f'{n_p:,}', f'{preprocess_time/1000:.6f}s', f'{inference_time/1000:.6f}s', f'{postprocess_time/1000:.6f}s',
        f'{1000/all_time:.2f}' if all_time > 0 else '0.00',
        f'{1000/inference_time:.2f}' if inference_time > 0 else '0.00',
        f'{get_weight_size(args.weights)}MB'
    ])

    model_metric_table = PrettyTable()
    model_metric_table.title = f'Model Metric - Native Evaluator / mode={mode_label}'
    model_metric_table.field_names = ['Class Name', 'Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP75', 'mAP50-95']
    for idx in range(length):
        model_metric_table.add_row([
            model_names[idx], f'{result.box.p[idx]:.4f}', f'{result.box.r[idx]:.4f}', f'{result.box.f1[idx]:.4f}',
            f'{result.box.ap50[idx]:.4f}', f'{result.box.all_ap[idx, 5]:.4f}', f'{result.box.ap[idx]:.4f}'
        ])

    precision = result.results_dict['metrics/precision(B)']
    recall = result.results_dict['metrics/recall(B)']
    map50 = result.results_dict['metrics/mAP50(B)']
    map5095 = result.results_dict['metrics/mAP50-95(B)']
    f1 = np.mean(result.box.f1[:length])
    map75 = np.mean(result.box.all_ap[:length, 5])
    model_metric_table.add_row(['all(平均数据)', f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}', f'{map50:.4f}', f'{map75:.4f}', f'{map5095:.4f}'])

    summary = PrettyTable()
    summary.title = 'Paper Summary - Native Evaluator'
    summary.field_names = ['Model', 'Mode', 'Precision (%)', 'Recall (%)', 'mAP50 (%)', 'mAP50:95 (%)', 'Params (M)', 'GFLOPs', 'FPS']
    summary.add_row([args.name, mode_label, f'{precision*100:.2f}', f'{recall*100:.2f}', f'{map50*100:.2f}', f'{map5095*100:.2f}', f'{n_p/1_000_000:.2f}', f'{flops:.1f}', f'{1000/all_time:.2f}' if all_time > 0 else '0.00'])

    print('-' * 20 + ' 论文数据：原生 val 指标体系 ' + '-' * 20)
    print(model_info_table)
    print(model_metric_table)
    print(summary)
    save_path = result.save_dir / 'paper_data.txt'
    with open(save_path, 'w+', encoding='utf-8', errors='ignore') as f:
        f.write(str(model_info_table) + '\n\n')
        f.write(str(model_metric_table) + '\n\n')
        f.write(str(summary) + '\n')
    print('-' * 20, f'结果已保存至 {save_path}', '-' * 20)


def main():
    args = parse_args()
    slice_cfg = SimpleNamespace(**vars(args))
    model = RTDETR(args.weights)

    def ValidatorFactory(*v_args, **v_kwargs):
        return RTDETRA640OldStyleValidator(*v_args, slice_cfg=slice_cfg, **v_kwargs)

    # 注意：这里传给 model.val 的都是 Ultralytics 原生参数；DAS 参数通过 slice_cfg 给自定义 validator，不污染原生 cfg。
    result = model.val(
        validator=ValidatorFactory,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=args.plots,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        workers=args.workers,
        half=args.half,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_json=args.save_json,
        exist_ok=True,
    )
    write_paper_data(model, result, args, slice_cfg)
    if args.force_exit:
        os._exit(0)


if __name__ == '__main__':
    main()
