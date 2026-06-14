import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

from val_a640_native_oldstyle import (
    make_tiles_adaptive_input,
    dedup_tiles,
    merge_dets_torch,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline-weights", type=str, required=True)
    parser.add_argument("--bc-weights", type=str, required=True)
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="Dataset")

    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--merge-iou", type=float, default=0.55)
    parser.add_argument("--max-det", type=int, default=1000)

    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--conf-base", type=float, default=None)
    parser.add_argument("--conf-abc", type=float, default=None)

    parser.add_argument("--device", type=str, default="auto")

    # 如果不指定，就默认取文件夹里排序后的前 4 张
    # 如果要指定，写法：--image-list a.jpg,b.jpg,c.jpg,d.jpg
    parser.add_argument("--image-list", type=str, default="")

    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--cell-w", type=int, default=520)
    parser.add_argument("--cell-h", type=int, default=360)

    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_arg


def preprocess_to_input_space(img_bgr, imgsz, device):
    letterbox = LetterBox((imgsz, imgsz), auto=False, scaleFill=True)
    im = letterbox(image=img_bgr)

    im = im[:, :, ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    return im.unsqueeze(0).to(device)


@torch.no_grad()
def postprocess_rtdetr(raw, imgsz, conf=0.25):
    if isinstance(raw, (list, tuple)):
        pred = raw[0]
    else:
        pred = raw

    bs, _, nd = pred.shape
    bboxes, scores = pred.split((4, nd - 4), dim=-1)

    bboxes = bboxes * imgsz
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
def predict_baseline_native(model_obj, img_bgr, imgsz, conf, device):
    results = model_obj.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        save=False,
    )

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return torch.zeros((0, 6))

    xyxy = r.boxes.xyxy.detach().cpu()
    conf_v = r.boxes.conf.detach().cpu()
    cls = r.boxes.cls.detach().cpu()
    return torch.cat([xyxy, conf_v[:, None], cls[:, None]], dim=1)


@torch.no_grad()
def predict_abc_a640(model_obj, img_bgr, imgsz, tile_size, overlap, conf, merge_iou, max_det, device):
    net = model_obj.model.to(device).eval()

    orig_h, orig_w = img_bgr.shape[:2]

    input_tensor = preprocess_to_input_space(img_bgr, imgsz, device)
    _, _, H, W = input_tensor.shape

    full_tile = (0, 0, W, H)

    # 这里沿用你原脚本的 oldstyle input-space A640 流程
    local_tiles = make_tiles_adaptive_input(W, H, tile_size, overlap)
    tiles = dedup_tiles([full_tile] + local_tiles)

    all_parts = []

    for x1, y1, x2, y2 in tiles:
        crop = input_tensor[:, :, y1:y2, x1:x2]

        if crop.shape[2] != imgsz or crop.shape[3] != imgsz:
            crop = F.interpolate(
                crop,
                size=(imgsz, imgsz),
                mode="bilinear",
                align_corners=False,
            )

        raw = net(crop, augment=False)
        preds = postprocess_rtdetr(raw, imgsz=imgsz, conf=conf)
        p = preds[0]

        if p is None or p.numel() == 0:
            continue

        sx = float(x2 - x1) / float(imgsz)
        sy = float(y2 - y1) / float(imgsz)

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

    merged = merge_dets_torch(
        torch.cat(all_parts, dim=0),
        merge_iou=merge_iou,
        max_det=max_det,
    )

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


def resize_pad(img_bgr, cell_w, cell_h):
    h, w = img_bgr.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 255
    x0 = (cell_w - new_w) // 2
    y0 = (cell_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def make_grid(baseline_imgs, abc_imgs, names, dataset_name, out_png, out_pdf, cell_w, cell_h):
    n = len(baseline_imgs)
    assert n == 4, "当前脚本固定输出 4 张图：上排 baseline，下排 ours"

    title_h = 42
    left_w = 110
    gap = 12

    grid_w = left_w + n * cell_w + (n - 1) * gap
    grid_h = title_h + 2 * cell_h + gap

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # 大标题
    cv2.putText(
        canvas,
        dataset_name,
        (left_w, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # 行标题
    cv2.putText(
        canvas,
        "Baseline",
        (8, title_h + cell_h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Ours",
        (8, title_h + cell_h + gap + cell_h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    for i in range(n):
        x = left_w + i * (cell_w + gap)
        y1 = title_h
        y2 = title_h + cell_h + gap

        b = resize_pad(baseline_imgs[i], cell_w, cell_h)
        o = resize_pad(abc_imgs[i], cell_w, cell_h)

        canvas[y1:y1 + cell_h, x:x + cell_w] = b
        canvas[y2:y2 + cell_h, x:x + cell_w] = o

        # 每列编号，避免标题太长
        cv2.putText(
            canvas,
            f"Image {i + 1}",
            (x + 8, title_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_png), canvas)

    # 保存 PDF
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    fig_w = grid_w / 180
    fig_h = grid_h / 180

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_pdf), bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    args = parse_args()

    device = get_device(args.device)
    conf_base = args.conf if args.conf_base is None else args.conf_base
    conf_abc = args.conf if args.conf_abc is None else args.conf_abc

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)

    out_base = out_dir / "baseline_boxonly"
    out_abc = out_dir / "abc_a640_boxonly"
    out_grid = out_dir / "grid"

    out_base.mkdir(parents=True, exist_ok=True)
    out_abc.mkdir(parents=True, exist_ok=True)
    out_grid.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    if args.image_list.strip():
        image_names = [x.strip() for x in args.image_list.split(",") if x.strip()]
        images = [source_dir / name for name in image_names]
    else:
        images = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in exts])[:4]

    if len(images) != 4:
        raise RuntimeError(f"需要正好 4 张图片，现在找到 {len(images)} 张：{images}")

    for p in images:
        if not p.exists():
            raise FileNotFoundError(f"图片不存在：{p}")

    print("DEVICE:", device)
    print("Dataset:", args.dataset_name)
    print("Baseline weights:", args.baseline_weights)
    print("BC weights used for ABC:", args.bc_weights)
    print("Source images:")
    for p in images:
        print("  ", p)

    print("Loading baseline model...")
    baseline_model = RTDETR(args.baseline_weights)

    print("Loading BC model for ABC=A+B+C inference...")
    bc_model = RTDETR(args.bc_weights)

    baseline_vis_list = []
    abc_vis_list = []
    names = []

    for img_path in images:
        print("Processing:", img_path.name)

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"OpenCV 读取失败：{img_path}")

        base_det = predict_baseline_native(
            baseline_model,
            img,
            imgsz=args.imgsz,
            conf=conf_base,
            device=device,
        )

        abc_det = predict_abc_a640(
            bc_model,
            img,
            imgsz=args.imgsz,
            tile_size=args.tile,
            overlap=args.overlap,
            conf=conf_abc,
            merge_iou=args.merge_iou,
            max_det=args.max_det,
            device=device,
        )

        base_vis = draw_boxes(img, base_det, color=(0, 255, 0), line_width=args.line_width)
        abc_vis = draw_boxes(img, abc_det, color=(0, 255, 0), line_width=args.line_width)

        stem = img_path.stem
        cv2.imwrite(str(out_base / f"{stem}_baseline.jpg"), base_vis)
        cv2.imwrite(str(out_abc / f"{stem}_ABC_A640.jpg"), abc_vis)

        baseline_vis_list.append(base_vis)
        abc_vis_list.append(abc_vis)
        names.append(stem)

        print(f"  baseline boxes: {len(base_det)}")
        print(f"  ABC boxes:      {len(abc_det)}")

    out_png = out_grid / f"{args.dataset_name}_baseline_vs_ours_2x4.png"
    out_pdf = out_grid / f"{args.dataset_name}_baseline_vs_ours_2x4.pdf"

    make_grid(
        baseline_imgs=baseline_vis_list,
        abc_imgs=abc_vis_list,
        names=names,
        dataset_name=args.dataset_name,
        out_png=out_png,
        out_pdf=out_pdf,
        cell_w=args.cell_w,
        cell_h=args.cell_h,
    )

    print("\n完成。输出位置：")
    print("Baseline single images:", out_base)
    print("ABC single images:", out_abc)
    print("Grid PNG:", out_png)
    print("Grid PDF:", out_pdf)


if __name__ == "__main__":
    main()