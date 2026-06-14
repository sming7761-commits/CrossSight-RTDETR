#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A640-GLF validation script
==========================

A innovation point used in this project:
    Full image RT-DETR prediction
  + Adaptive 640x640 grid slicing prediction
  + Coordinate remapping
  + Global-local prediction fusion

This script does not train the model and does not change RT-DETR weights.
It is a plug-and-play inference/validation module for UAV small-object detection.

Adaptive 640 grid formula:
    S = 640
    Nw = ceil((W - S) / (S * (1 - rho))) + 1
    Nh = ceil((H - S) / (S * (1 - rho))) + 1
    stride_w = (W - S) / (Nw - 1)
    stride_h = (H - S) / (Nh - 1)
    x_i = round(i * stride_w), y_j = round(j * stride_h)

Compared with a fixed sliding-window stride, the adaptive grid keeps the local
view size fixed at 640 while adapting the number of tiles and the actual stride
to each image size. This avoids uncovered borders and reduces uneven boundary
coverage without using GT boxes, density maps, predicted ROIs, or second-stage
ROI re-detection.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from prettytable import PrettyTable
from ultralytics import RTDETR, YOLO
from ultralytics.utils.torch_utils import model_info

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="A640-GLF: full image + adaptive 640 grid slicing + prediction fusion evaluator.")
    p.add_argument("--weights", type=str, required=True, help="model weights path, e.g. runs/train/baseline_200_clean/weights/best.pt")
    p.add_argument("--data", type=str, default="dataset/data.yaml", help="dataset yaml")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=960, help="inference image size used by RT-DETR")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--batch", type=int, default=8, help="tile batch size for predict")
    p.add_argument("--conf", type=float, default=0.01, help="confidence threshold used for inference and AP computation")
    p.add_argument("--pred-iou", type=float, default=0.70, help="Ultralytics internal postprocess IoU")
    p.add_argument("--max-det", type=int, default=1000)

    p.add_argument("--mode", type=str, default="adaptive", choices=["no_slice", "fixed", "adaptive", "slice_only"],
                   help="no_slice: full image only; fixed: full+fixed 640 sliding window; adaptive: full+adaptive 640 grid; slice_only: adaptive 640 grid only")
    p.add_argument("--tile", type=int, default=640, help="local view size. For A640-GLF keep this as 640")
    p.add_argument("--overlap", type=float, default=0.20, help="target overlap ratio rho for fixed/adaptive slicing")
    p.add_argument("--merge-iou", type=float, default=0.55, help="class-wise IoU threshold after merging global/local predictions")
    p.add_argument("--fusion", type=str, default="nms", choices=["nms", "wbf"], help="prediction fusion method")

    # Backward-compatible convenience flags.
    p.add_argument("--no-slice", action="store_true", help="same as --mode no_slice")
    p.add_argument("--fixed", action="store_true", help="same as --mode fixed")
    p.add_argument("--adaptive", action="store_true", help="same as --mode adaptive")
    p.add_argument("--slice-only", action="store_true", help="same as --mode slice_only")

    p.add_argument("--project", type=str, default="runs/val_a640")
    p.add_argument("--name", type=str, default="A640_GLF_adaptive")
    p.add_argument("--save-preds", action="store_true", help="save final predictions txt for inspection")
    p.add_argument("--force-exit", action="store_true", help="force process exit after saving results to avoid terminal hanging")
    return p.parse_args()


def normalize_mode(args):
    if args.no_slice:
        args.mode = "no_slice"
    if args.fixed:
        args.mode = "fixed"
    if args.adaptive:
        args.mode = "adaptive"
    if args.slice_only:
        args.mode = "slice_only"
    if args.tile != 640:
        print(f"[WARN] A640-GLF is designed for 640x640 local views, but --tile={args.tile} was given.", flush=True)
    if not (0.0 <= args.overlap < 0.9):
        raise ValueError("--overlap must be in [0, 0.9).")
    return args


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_images(data_yaml: str, split: str) -> Tuple[List[Path], List[str]]:
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", "")).expanduser()
    if not root.is_absolute():
        root = (Path(data_yaml).resolve().parent / root).resolve()

    names = cfg.get("names", None)
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, list):
        names = names
    else:
        nc = int(cfg.get("nc", 0))
        names = [str(i) for i in range(nc)]

    split_value = cfg.get(split, None)
    if split_value is None:
        raise FileNotFoundError(f"split '{split}' not found in {data_yaml}")

    def resolve_one(v):
        p = Path(str(v)).expanduser()
        if not p.is_absolute():
            p = root / p
        return p

    paths = [resolve_one(x) for x in split_value] if isinstance(split_value, (list, tuple)) else [resolve_one(split_value)]
    image_files: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    q = Path(line.strip()).expanduser()
                    if not q.is_absolute():
                        q = root / q
                    image_files.append(q.resolve())
        elif p.is_dir():
            image_files.extend([x.resolve() for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS])
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            image_files.append(p.resolve())
        else:
            raise FileNotFoundError(f"Cannot resolve split path: {p}")

    image_files = sorted(list(dict.fromkeys(image_files)))
    if not image_files:
        raise FileNotFoundError(f"No images found for split={split}")
    return image_files, names


def label_path_from_image(img_path: Path) -> Path:
    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def load_yolo_labels(img_path: Path, w: int, h: int) -> np.ndarray:
    lp = label_path_from_image(img_path)
    if not lp.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(lp, "r", encoding="utf-8") as f:
        for line in f:
            ss = line.strip().split()
            if len(ss) < 5:
                continue
            c = int(float(ss[0]))
            x, y, bw, bh = map(float, ss[1:5])
            x1 = (x - bw / 2.0) * w
            y1 = (y - bh / 2.0) * h
            x2 = (x + bw / 2.0) * w
            y2 = (y + bh / 2.0) * h
            rows.append([c, x1, y1, x2, y2])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0, w)
    arr[:, [2, 4]] = np.clip(arr[:, [2, 4]], 0, h)
    return arr


def _dedup_tiles(tiles: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    seen = set()
    out = []
    for t in tiles:
        key = tuple(int(v) for v in t)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def make_tiles_fixed(w: int, h: int, tile: int = 640, overlap: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """Conventional fixed-stride sliding window."""
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
    return _dedup_tiles(tiles)


def make_tiles_adaptive_640(w: int, h: int, tile: int = 640, overlap: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """Adaptive 640 grid slicing.

    The local view size remains fixed, but the number of tiles and real stride are
    adapted to the image size. No GT boxes or predicted ROIs are used.
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
    return _dedup_tiles(tiles)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return np.asarray(keep, dtype=np.int64)


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    area_b = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def wbf_classwise(dets: np.ndarray, merge_iou: float, max_det: int) -> np.ndarray:
    if dets.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    fused_all = []
    for c in np.unique(dets[:, 5].astype(np.int32)):
        cls_dets = dets[dets[:, 5].astype(np.int32) == c]
        order = cls_dets[:, 4].argsort()[::-1]
        cls_dets = cls_dets[order]
        used = np.zeros(len(cls_dets), dtype=bool)
        for i in range(len(cls_dets)):
            if used[i]:
                continue
            ious = box_iou_matrix(cls_dets[i:i + 1, :4], cls_dets[:, :4])[0]
            cluster_idx = np.where((ious >= merge_iou) & (~used))[0]
            if len(cluster_idx) == 0:
                cluster_idx = np.array([i])
            used[cluster_idx] = True
            cluster = cls_dets[cluster_idx]
            weights = np.maximum(cluster[:, 4:5], 1e-6)
            fused_box = (cluster[:, :4] * weights).sum(axis=0) / weights.sum()
            fused_score = float(cluster[:, 4].max())
            fused_all.append(np.array([*fused_box, fused_score, float(c)], dtype=np.float32))
    if not fused_all:
        return np.zeros((0, 6), dtype=np.float32)
    out = np.stack(fused_all, axis=0)
    order = out[:, 4].argsort()[::-1]
    return out[order[:max_det]].astype(np.float32)


def merge_detections(dets: np.ndarray, merge_iou: float, max_det: int, fusion: str = "nms") -> np.ndarray:
    if dets.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if fusion == "wbf":
        return wbf_classwise(dets, merge_iou, max_det)
    out = []
    for c in np.unique(dets[:, 5].astype(np.int32)):
        idx = np.where(dets[:, 5].astype(np.int32) == c)[0]
        keep = nms_numpy(dets[idx, :4], dets[idx, 4], merge_iou)
        out.append(dets[idx[keep]])
    if not out:
        return np.zeros((0, 6), dtype=np.float32)
    merged = np.concatenate(out, axis=0)
    order = merged[:, 4].argsort()[::-1]
    return merged[order[:max_det]].astype(np.float32)


def predict_crops(model, crops: List[np.ndarray], offsets: List[Tuple[int, int]], w: int, h: int, args) -> np.ndarray:
    if not crops:
        return np.zeros((0, 6), dtype=np.float32)
    all_dets = []
    for st in range(0, len(crops), max(1, args.batch)):
        batch = crops[st:st + args.batch]
        batch_offsets = offsets[st:st + args.batch]
        results = model.predict(
            source=batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.pred_iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            stream=False,
        )
        for r, (ox, oy) in zip(results, batch_offsets):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            conf = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
            cls = r.boxes.cls.detach().cpu().numpy().astype(np.float32)
            xyxy[:, [0, 2]] += ox
            xyxy[:, [1, 3]] += oy
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
            det = np.concatenate([xyxy, conf[:, None], cls[:, None]], axis=1)
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det = det[area > 1.0]
            if det.size:
                all_dets.append(det)
    if not all_dets:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(all_dets, axis=0).astype(np.float32)


def crops_from_tiles(img: Image.Image, tiles: List[Tuple[int, int, int, int]]) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    crops, offsets = [], []
    for x1, y1, x2, y2 in tiles:
        crops.append(np.asarray(img.crop((x1, y1, x2, y2)).convert("RGB")))
        offsets.append((x1, y1))
    return crops, offsets


def predict_image(model, img: Image.Image, args) -> Tuple[np.ndarray, dict]:
    w, h = img.size
    rgb = np.asarray(img.convert("RGB"))
    details = {"mode": args.mode, "strategy": args.mode, "num_tiles": 0, "num_views": 1}

    if args.mode == "no_slice":
        det = predict_crops(model, [rgb], [(0, 0)], w, h, args)
        return merge_detections(det, args.merge_iou, args.max_det, args.fusion), details

    if args.mode == "fixed":
        local_tiles = make_tiles_fixed(w, h, args.tile, args.overlap)
        strategy = f"full_plus_fixed_{args.tile}_o{args.overlap}"
    else:
        local_tiles = make_tiles_adaptive_640(w, h, args.tile, args.overlap)
        strategy = f"adaptive_{args.tile}_grid_o{args.overlap}"

    if args.mode in {"fixed", "adaptive"}:
        # Full image + local views. Remove duplicate full-image tile if the image itself is no larger than the tile.
        local_tiles = [t for t in local_tiles if t != (0, 0, w, h)]
        crops = [rgb]
        offsets = [(0, 0)]
        tile_crops, tile_offsets = crops_from_tiles(img, local_tiles)
        crops.extend(tile_crops)
        offsets.extend(tile_offsets)
        details.update({"strategy": strategy, "num_tiles": len(local_tiles), "num_views": len(crops)})
        det = predict_crops(model, crops, offsets, w, h, args)
        return merge_detections(det, args.merge_iou, args.max_det, args.fusion), details

    # slice_only: adaptive local views only, useful for ablation. No full-image result is used.
    local_tiles = [t for t in local_tiles]
    crops, offsets = crops_from_tiles(img, local_tiles)
    details.update({"strategy": "slice_only_adaptive_640", "num_tiles": len(local_tiles), "num_views": len(crops)})
    det = predict_crops(model, crops, offsets, w, h, args)
    return merge_detections(det, args.merge_iou, args.max_det, args.fusion), details


def ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    if rec.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    recall_points = np.linspace(0, 1, 101)
    return float(np.mean([mpre[mrec >= t].max() if np.any(mrec >= t) else 0 for t in recall_points]))


def evaluate_predictions(preds: Dict[int, np.ndarray], gts: Dict[int, np.ndarray], nc: int):
    iou_thrs = np.arange(0.50, 0.96, 0.05)
    ap = np.zeros((nc, len(iou_thrs)), dtype=np.float32)
    best_p = np.zeros(nc, dtype=np.float32)
    best_r = np.zeros(nc, dtype=np.float32)
    best_f1 = np.zeros(nc, dtype=np.float32)
    gt_counts = np.zeros(nc, dtype=np.int64)

    for c in range(nc):
        gt_by_img = {}
        n_gt = 0
        for img_id, gt in gts.items():
            gc = gt[gt[:, 0].astype(np.int32) == c, 1:5]
            gt_by_img[img_id] = gc
            n_gt += len(gc)
        gt_counts[c] = n_gt
        if n_gt == 0:
            continue

        rows = []
        for img_id, det in preds.items():
            if det.size == 0:
                continue
            dc = det[det[:, 5].astype(np.int32) == c]
            for d in dc:
                rows.append([img_id, d[4], d[0], d[1], d[2], d[3]])
        if not rows:
            continue
        rows = np.asarray(rows, dtype=np.float32)
        rows = rows[rows[:, 1].argsort()[::-1]]

        pr_for_05 = None
        for ti, thr in enumerate(iou_thrs):
            used = {img_id: np.zeros(len(gt_by_img[img_id]), dtype=bool) for img_id in gt_by_img.keys()}
            tp = np.zeros(len(rows), dtype=np.float32)
            fp = np.zeros(len(rows), dtype=np.float32)
            for i, row in enumerate(rows):
                img_id = int(row[0])
                gt_boxes = gt_by_img.get(img_id, np.zeros((0, 4), dtype=np.float32))
                if len(gt_boxes) == 0:
                    fp[i] = 1
                    continue
                ious = box_iou_matrix(row[2:6][None, :], gt_boxes)[0]
                j = int(np.argmax(ious)) if ious.size else -1
                if j >= 0 and ious[j] >= thr and not used[img_id][j]:
                    tp[i] = 1
                    used[img_id][j] = True
                else:
                    fp[i] = 1
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / (n_gt + 1e-9)
            prec = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap[c, ti] = ap_from_pr(rec, prec)
            if abs(thr - 0.50) < 1e-6:
                pr_for_05 = (prec, rec)

        if pr_for_05 is not None:
            prec, rec = pr_for_05
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            bi = int(np.argmax(f1)) if f1.size else 0
            best_p[c] = prec[bi] if prec.size else 0
            best_r[c] = rec[bi] if rec.size else 0
            best_f1[c] = f1[bi] if f1.size else 0

    return best_p, best_r, best_f1, ap, gt_counts


def get_weight_size(path):
    return f"{os.stat(path).st_size / 1024 / 1024:.1f}"


def save_csv(save_dir: Path, names: List[str], p, r, f1, ap50, ap75, ap5095):
    with open(save_dir / "class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "map50", "map75", "map50_95"])
        for i, name in enumerate(names):
            writer.writerow([name, float(p[i]), float(r[i]), float(f1[i]), float(ap50[i]), float(ap75[i]), float(ap5095[i])])


def main():
    args = normalize_mode(parse_args())
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    img_files, names = resolve_split_images(args.data, args.split)
    nc = len(names)
    print(f"Images: {len(img_files)} | Classes: {nc} | mode={args.mode}", flush=True)
    print(f"weights={args.weights}, imgsz={args.imgsz}, tile={args.tile}, target_overlap={args.overlap}, conf={args.conf}, fusion={args.fusion}", flush=True)

    try:
        model = RTDETR(args.weights)
    except Exception:
        model = YOLO(args.weights)

    preds: Dict[int, np.ndarray] = {}
    gts: Dict[int, np.ndarray] = {}
    strategy_counter = Counter()
    total_tiles = 0
    total_views = 0

    t0 = time.time()
    preprocess_t = 0.0
    infer_t = 0.0

    for i, img_path in enumerate(img_files):
        t_img0 = time.time()
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        gts[i] = load_yolo_labels(img_path, w, h)
        preprocess_t += time.time() - t_img0

        t_inf0 = time.time()
        det, detail = predict_image(model, img, args)
        infer_t += time.time() - t_inf0
        preds[i] = det
        strategy_counter[detail.get("strategy", args.mode)] += 1
        total_tiles += int(detail.get("num_tiles", 0))
        total_views += int(detail.get("num_views", 1))

        if args.save_preds:
            out = save_dir / "preds_txt"
            out.mkdir(exist_ok=True)
            with open(out / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
                for d in det:
                    f.write(f"{int(d[5])} {d[4]:.6f} {d[0]:.2f} {d[1]:.2f} {d[2]:.2f} {d[3]:.2f}\n")

        if (i + 1) % 20 == 0 or (i + 1) == len(img_files):
            print(f"[{i+1}/{len(img_files)}] elapsed={(time.time()-t0)/60:.1f} min", flush=True)

    p, r, f1, ap, gt_counts = evaluate_predictions(preds, gts, nc)
    ap50 = ap[:, 0]
    ap75 = ap[:, 5] if ap.shape[1] > 5 else np.zeros(nc)
    ap5095 = ap.mean(axis=1)
    valid = gt_counts > 0
    if not np.any(valid):
        valid = np.ones(nc, dtype=bool)

    mean_p = float(np.mean(p[valid]))
    mean_r = float(np.mean(r[valid]))
    mean_f1 = float(np.mean(f1[valid]))
    mean_ap50 = float(np.mean(ap50[valid]))
    mean_ap75 = float(np.mean(ap75[valid]))
    mean_ap5095 = float(np.mean(ap5095[valid]))

    try:
        _, n_p, _, flops = model_info(model.model)
    except Exception:
        n_p, flops = 0, 0.0

    total_images = max(len(img_files), 1)
    total_time = time.time() - t0
    total_per_img_ms = total_time * 1000 / total_images

    model_info_table = PrettyTable()
    model_info_table.title = "Model Info"
    model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理+切片时间/一张图", "FPS(总流程)", "Model File Size"]
    model_info_table.add_row([
        f"{flops:.1f}", f"{n_p:,}", f"{(preprocess_t / total_images):.6f}s", f"{(infer_t / total_images):.6f}s",
        f"{1000 / total_per_img_ms:.2f}", f"{get_weight_size(args.weights)}MB"
    ])

    strategy_table = PrettyTable()
    strategy_table.title = "A640-GLF Strategy Info"
    strategy_table.field_names = ["策略", "图片数量"]
    for k, v in strategy_counter.items():
        strategy_table.add_row([k, v])
    strategy_table.add_row(["平均局部切片数/图", f"{total_tiles / total_images:.2f}"])
    strategy_table.add_row(["平均总视图数/图", f"{total_views / total_images:.2f}"])

    metric_table = PrettyTable()
    metric_table.title = "Model Metric - Same Evaluator"
    metric_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
    for c, name in enumerate(names):
        metric_table.add_row([name, f"{p[c]:.4f}", f"{r[c]:.4f}", f"{f1[c]:.4f}", f"{ap50[c]:.4f}", f"{ap75[c]:.4f}", f"{ap5095[c]:.4f}"])
    metric_table.add_row(["all(平均数据)", f"{mean_p:.4f}", f"{mean_r:.4f}", f"{mean_f1:.4f}", f"{mean_ap50:.4f}", f"{mean_ap75:.4f}", f"{mean_ap5095:.4f}"])

    paper = PrettyTable()
    paper.title = "Paper Summary - Same Evaluator"
    paper.field_names = ["Model", "Precision (%)", "Recall (%)", "mAP50 (%)", "mAP50:95 (%)", "Params (M)", "GFLOPs", "FPS"]
    paper.add_row([args.name, f"{mean_p*100:.2f}", f"{mean_r*100:.2f}", f"{mean_ap50*100:.2f}", f"{mean_ap5095*100:.2f}", f"{n_p/1_000_000:.2f}", f"{flops:.1f}", f"{1000/total_per_img_ms:.2f}"])

    print("-" * 20 + " 论文数据：务必和 no-slice / fixed 使用同一脚本比较 " + "-" * 20)
    print(model_info_table)
    print(strategy_table)
    print(metric_table)
    print(paper)

    out_file = save_dir / "paper_data.txt"
    with open(out_file, "w", encoding="utf-8", errors="ignore") as f:
        f.write(str(model_info_table) + "\n\n")
        f.write(str(strategy_table) + "\n\n")
        f.write(str(metric_table) + "\n\n")
        f.write(str(paper) + "\n")

    save_csv(save_dir, names, p, r, f1, ap50, ap75, ap5095)
    with open(save_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "name": args.name,
            "mode": args.mode,
            "tile": args.tile,
            "target_overlap": args.overlap,
            "precision": mean_p,
            "recall": mean_r,
            "f1": mean_f1,
            "map50": mean_ap50,
            "map75": mean_ap75,
            "map50_95": mean_ap5095,
            "fps": 1000 / total_per_img_ms,
            "params_m": n_p / 1_000_000,
            "gflops": flops,
            "strategy_counter": dict(strategy_counter),
            "avg_local_tiles_per_image": total_tiles / total_images,
            "avg_views_per_image": total_views / total_images,
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved to {out_file}", flush=True)

    if args.force_exit:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            os._exit(0)


if __name__ == "__main__":
    main()
