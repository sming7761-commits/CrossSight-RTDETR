import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import RTDETR
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

from val_a640_native_oldstyle import (
    make_tiles_adaptive_input,
    dedup_tiles,
    merge_dets_torch,
)


# ===================== 路径配置 =====================
BASELINE_WEIGHTS = r"C:\Users\symin\Desktop\知识蒸馏\泛化\人\runs\train\tinyperson_rtdetr_r18_baseline_e200_b4\weights\best.pt"

BC_WEIGHTS = r"C:\Users\symin\Desktop\知识蒸馏\泛化\人\runs\train\tinyperson_hfgmf_iswiou_e200_b4\weights\best.pt"

SOURCE_DIR = r"C:\Users\symin\Desktop\FIG\per"

OUT_DIR = r"C:\Users\symin\Desktop\FIG\vis_out_TinyPerson_A640"

IMG_SIZE = 960
TILE_SIZE = 640
OVERLAP = 0.20
CONF = 0.35#阈值
MERGE_IOU = 0.55
MAX_DET = 1000

# RTX 3050 Laptop 如果爆显存，把 cuda:0 改成 cpu
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# ===================================================


def preprocess_to_input_space(img_bgr):
    letterbox = LetterBox((IMG_SIZE, IMG_SIZE), auto=False, scaleFill=True)
    im = letterbox(image=img_bgr)

    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    return im.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def postprocess_rtdetr(raw, conf=0.25):
    """
    仿照 ultralytics/models/rtdetr/val.py 的 postprocess。
    输出坐标在 960×960 输入空间内。
    """
    if isinstance(raw, (list, tuple)):
        pred = raw[0]
    else:
        pred = raw

    # pred: [B, 300, 4 + num_classes]
    bs, _, nd = pred.shape
    bboxes, scores = pred.split((4, nd - 4), dim=-1)

    bboxes = bboxes * IMG_SIZE
    outputs = []

    for i in range(bs):
        bbox = ops.xywh2xyxy(bboxes[i])
        score, cls = scores[i].max(-1)

        keep = score > conf
        bbox = bbox[keep]
        score_i = score[keep]
        cls_i = cls[keep].float()

        if bbox.numel() == 0:
            outputs.append(torch.zeros((0, 6), device=pred.device))
        else:
            det = torch.cat([bbox, score_i[:, None], cls_i[:, None]], dim=-1)
            det = det[det[:, 4].argsort(descending=True)]
            outputs.append(det)

    return outputs


@torch.no_grad()
def predict_baseline_native(model_obj, img_bgr):
    """
    baseline 普通整图预测。
    """
    results = model_obj.predict(
        source=img_bgr,
        imgsz=IMG_SIZE,
        conf=CONF,
        device=DEVICE,
        verbose=False,
        save=False,
    )

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return torch.zeros((0, 6))

    xyxy = r.boxes.xyxy.detach().cpu()
    conf = r.boxes.conf.detach().cpu()
    cls = r.boxes.cls.detach().cpu()
    return torch.cat([xyxy, conf[:, None], cls[:, None]], dim=1)


@torch.no_grad()
def predict_abc_a640(model_obj, img_bgr):
    """
    真正的 A+B+C：
    B+C 权重 + A640 输入空间局部视图 + 融合。
    """
    net = model_obj.model.to(DEVICE).eval()

    orig_h, orig_w = img_bgr.shape[:2]

    # 1. 原图先进入 RT-DETR native preprocessing，得到统一 960×960 input-space
    input_tensor = preprocess_to_input_space(img_bgr)
    _, _, H, W = input_tensor.shape

    full_tile = (0, 0, W, H)

    # 2. 构造 A640 局部视图；这里用 oldstyle adaptive，对 960/640 等价于固定 2×2 覆盖
    local_tiles = make_tiles_adaptive_input(W, H, TILE_SIZE, OVERLAP)
    tiles = dedup_tiles([full_tile] + local_tiles)

    all_parts = []

    for x1, y1, x2, y2 in tiles:
        crop = input_tensor[:, :, y1:y2, x1:x2]

        # 每个 local view resize 回 960 输入给同一个 RT-DETR
        if crop.shape[2] != IMG_SIZE or crop.shape[3] != IMG_SIZE:
            crop = F.interpolate(
                crop,
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        raw = net(crop, augment=False)
        preds = postprocess_rtdetr(raw, conf=CONF)
        p = preds[0]

        if p is None or p.numel() == 0:
            continue

        # 3. local-view 坐标恢复到全局 960×960 input-space
        sx = float(x2 - x1) / float(IMG_SIZE)
        sy = float(y2 - y1) / float(IMG_SIZE)

        p = p.clone()
        p[:, [0, 2]] = p[:, [0, 2]] * sx + float(x1)
        p[:, [1, 3]] = p[:, [1, 3]] * sy + float(y1)

        p[:, [0, 2]] = p[:, [0, 2]].clamp(0, W)
        p[:, [1, 3]] = p[:, [1, 3]].clamp(0, H)

        area = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
        p = p[area > 1.0]

        if p.numel():
            all_parts.append(p)

    if not all_parts:
        return torch.zeros((0, 6))

    # 4. 融合 full-view + local-view predictions
    merged = merge_dets_torch(
        torch.cat(all_parts, dim=0),
        merge_iou=MERGE_IOU,
        max_det=MAX_DET,
    )

    # 5. 960×960 input-space 坐标恢复到原图尺寸，方便画图
    merged[:, [0, 2]] = merged[:, [0, 2]] / float(W) * float(orig_w)
    merged[:, [1, 3]] = merged[:, [1, 3]] / float(H) * float(orig_h)

    return merged.detach().cpu()


def draw_boxes(img_bgr, dets, color=(0, 255, 0), line_width=2):
    out = img_bgr.copy()

    for det in dets:
        x1, y1, x2, y2, score, cls = det.tolist()

        x1 = int(max(0, min(x1, out.shape[1] - 1)))
        y1 = int(max(0, min(y1, out.shape[0] - 1)))
        x2 = int(max(0, min(x2, out.shape[1] - 1)))
        y2 = int(max(0, min(y2, out.shape[0] - 1)))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

    return out


def concat_compare(left, right):
    h = max(left.shape[0], right.shape[0])

    def resize_h(img, target_h):
        if img.shape[0] == target_h:
            return img
        scale = target_h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, target_h), interpolation=cv2.INTER_LINEAR)

    left = resize_h(left, h)
    right = resize_h(right, h)

    gap = np.ones((h, 12, 3), dtype=np.uint8) * 255
    return np.concatenate([left, gap, right], axis=1)


def main():
    print("DEVICE:", DEVICE)
    print("Baseline weights:", BASELINE_WEIGHTS)
    print("BC weights used with A640:", BC_WEIGHTS)

    source_dir = Path(SOURCE_DIR)
    out_dir = Path(OUT_DIR)

    out_base = out_dir / "baseline_boxonly"
    out_abc = out_dir / "abc_a640_boxonly"
    out_cmp = out_dir / "compare_boxonly"

    out_base.mkdir(parents=True, exist_ok=True)
    out_abc.mkdir(parents=True, exist_ok=True)
    out_cmp.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".jipg"}
    images = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in exts])

    if len(images) == 0:
        raise FileNotFoundError(f"没有找到图片：{SOURCE_DIR}")

    print("Loading baseline model...")
    baseline_model = RTDETR(BASELINE_WEIGHTS)

    print("Loading BC model for ABC=A+B+C inference...")
    bc_model = RTDETR(BC_WEIGHTS)

    for img_path in images:
        print("Processing:", img_path.name)

        img = cv2.imread(str(img_path))
        if img is None:
            print("OpenCV 读取失败，跳过：", img_path)
            continue

        base_det = predict_baseline_native(baseline_model, img)

        # 关键：这里不是 detect.py，而是 BC 权重 + A640 推理
        abc_det = predict_abc_a640(bc_model, img)

        base_vis = draw_boxes(img, base_det, color=(0, 255, 0), line_width=2)
        abc_vis = draw_boxes(img, abc_det, color=(0, 255, 0), line_width=2)

        compare = concat_compare(base_vis, abc_vis)

        stem = img_path.stem
        cv2.imwrite(str(out_base / f"{stem}_baseline.jpg"), base_vis)
        cv2.imwrite(str(out_abc / f"{stem}_ABC_A640.jpg"), abc_vis)
        cv2.imwrite(str(out_cmp / f"{stem}_compare.jpg"), compare)

    print("\n完成。输出位置：")
    print(out_base)
    print(out_abc)
    print(out_cmp)


if __name__ == "__main__":
    main()