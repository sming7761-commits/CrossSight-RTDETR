import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

# 必须使用 fixed oldstyle A640
from val_a640_native_oldstyle import (
    make_tiles,
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

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--conf-base", type=float, default=None)
    parser.add_argument("--conf-abc", type=float, default=None)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--line-width", type=int, default=2)

    # 可选：只跑前 N 张。不填就是全部跑
    parser.add_argument("--max-images", type=int, default=0)

    # 可选：递归读取子文件夹图片
    parser.add_argument("--recursive", action="store_true")

    # 可选：指定图片名，逗号分隔，例如 a.jpg,b.jpg,c.jpg
    parser.add_argument("--image-list", type=str, default="")

    # 输出格式：png 或 jpg
    parser.add_argument("--save-ext", type=str, default="png", choices=["png", "jpg", "jpeg"])

    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_arg


def collect_images(source_dir, recursive=False, image_list="", max_images=0):
    source_dir = Path(source_dir)
    
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".pgm", ".pnm"}
    if image_list.strip():
        names = [x.strip() for x in image_list.split(",") if x.strip()]
        images = [source_dir / name for name in names]
    else:
        if recursive:
            images = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in exts])
        else:
            images = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in exts])

    if max_images and max_images > 0:
        images = images[:max_images]

    if len(images) == 0:
        raise RuntimeError(f"没有找到图片：{source_dir}")

    for p in images:
        if not p.exists():
            raise FileNotFoundError(f"图片不存在：{p}")

    return images


def preprocess_to_input_space(img_bgr, imgsz, device):
    # 与 RT-DETR val oldstyle 保持一致：输入空间 960×960
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

    # pred: [B, N, 4 + C]
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
    # 上排/基线：普通整图推理
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
    # 下排/改进：BC 权重 + fixed A640 oldstyle 输入空间融合
    net = model_obj.model.to(device).eval()

    orig_h, orig_w = img_bgr.shape[:2]

    input_tensor = preprocess_to_input_space(img_bgr, imgsz, device)
    _, _, H, W = input_tensor.shape

    full_tile = (0, 0, W, H)

    # fixed A640 oldstyle：不是 adaptive
    local_tiles = make_tiles(W, H, tile_size, overlap)
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

        # local-to-global in input space
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

    # input-space 960 坐标映射回原图坐标
    merged[:, [0, 2]] = merged[:, [0, 2]] / float(W) * float(orig_w)
    merged[:, [1, 3]] = merged[:, [1, 3]] / float(H) * float(orig_h)

    return merged.detach().cpu()


def draw_boxes(img_bgr, dets, color=(0, 0, 255), line_width=2):
    out = img_bgr.copy()

    for det in dets:
        x1, y1, x2, y2, score, cls = det.tolist()

        x1 = int(max(0, min(x1, out.shape[1] - 1)))
        y1 = int(max(0, min(y1, out.shape[0] - 1)))
        x2 = int(max(0, min(x2, out.shape[1] - 1)))
        y2 = int(max(0, min(y2, out.shape[0] - 1)))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

    return out


def safe_name(path):
    # 防止递归子目录重名
    name = path.stem
    parent = path.parent.name
    if parent:
        return f"{parent}_{name}"
    return name


def main():
    args = parse_args()

    device = get_device(args.device)
    conf_base = args.conf if args.conf_base is None else args.conf_base
    conf_abc = args.conf if args.conf_abc is None else args.conf_abc

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)

    out_base = out_dir / "baseline_boxonly"
    out_abc = out_dir / "abc_a640_boxonly"
    out_txt = out_dir / "box_counts.txt"

    out_base.mkdir(parents=True, exist_ok=True)
    out_abc.mkdir(parents=True, exist_ok=True)

    images = collect_images(
        source_dir=source_dir,
        recursive=args.recursive,
        image_list=args.image_list,
        max_images=args.max_images,
    )

    print("=" * 80)
    print("Dataset:", args.dataset_name)
    print("Device:", device)
    print("Images:", len(images))
    print("Baseline weights:", args.baseline_weights)
    print("BC weights for ABC:", args.bc_weights)
    print("Source dir:", source_dir)
    print("Out dir:", out_dir)
    print("conf_base:", conf_base)
    print("conf_abc:", conf_abc)
    print("A640:", f"imgsz={args.imgsz}, tile={args.tile}, overlap={args.overlap}, merge_iou={args.merge_iou}")
    print("=" * 80)

    print("Loading baseline model...")
    baseline_model = RTDETR(args.baseline_weights)

    print("Loading BC model for ABC inference...")
    bc_model = RTDETR(args.bc_weights)

    count_lines = []
    count_lines.append("image,baseline_boxes,abc_boxes\n")

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Processing: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] OpenCV 读取失败，跳过：{img_path}")
            continue

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

        base_vis = draw_boxes(
            img,
            base_det,
            color=(0, 0, 255),
            line_width=args.line_width,
        )

        abc_vis = draw_boxes(
            img,
            abc_det,
            color=(0, 0, 255),
            line_width=args.line_width,
        )

        stem = safe_name(img_path)
        ext = args.save_ext.lower()

        base_out = out_base / f"{stem}_baseline.{ext}"
        abc_out = out_abc / f"{stem}_ABC_A640.{ext}"

        cv2.imwrite(str(base_out), base_vis)
        cv2.imwrite(str(abc_out), abc_vis)

        print(f"    baseline boxes: {len(base_det)}")
        print(f"    ABC boxes:      {len(abc_det)}")
        print(f"    saved baseline: {base_out}")
        print(f"    saved ABC:      {abc_out}")

        count_lines.append(f"{img_path.name},{len(base_det)},{len(abc_det)}\n")

    out_txt.write_text("".join(count_lines), encoding="utf-8")

    print("\n完成。")
    print("Baseline 输出：", out_base)
    print("ABC 输出：", out_abc)
    print("框数量记录：", out_txt)


if __name__ == "__main__":
    main()