import argparse
import csv
import math
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_path(p, yaml_dir, root):
    p = Path(p)
    if p.is_absolute():
        return p
    if root:
        root = Path(root)
        if not root.is_absolute():
            root = yaml_dir / root
        return root / p
    return yaml_dir / p


def load_images(data_yaml, split):
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    yaml_dir = data_yaml.parent
    root = data.get("path", "")

    split_value = data.get(split)
    if split_value is None:
        raise ValueError(f"Cannot find split '{split}' in {data_yaml}")

    split_path = resolve_path(split_value, yaml_dir, root)

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        images = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = split_path.parent / p
            images.append(p)
        return images

    if split_path.is_dir():
        images = []
        for ext in IMG_EXTS:
            images.extend(split_path.rglob(f"*{ext}"))
        return sorted(images)

    raise FileNotFoundError(f"Cannot resolve split path: {split_path}")


def image_to_label_path(img_path):
    img_path = Path(img_path)
    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def parse_yolo(path, w, h, is_pred=False):
    path = Path(path)
    boxes = []
    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        vals = line.strip().split()
        if len(vals) < 5:
            continue

        cls = int(float(vals[0]))
        xc = float(vals[1]) * w
        yc = float(vals[2]) * h
        bw = float(vals[3]) * w
        bh = float(vals[4]) * h
        conf = float(vals[5]) if is_pred and len(vals) >= 6 else 1.0

        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        boxes.append({
            "cls": cls,
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "conf": conf,
            "area": max(0.0, bw) * max(0.0, bh),
        })

    return boxes


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def center_error(a, b):
    ca = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2], dtype=np.float32)
    cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=np.float32)
    return float(np.linalg.norm(ca - cb))


def collect_for_method(images, pred_root, method, small_area_ratio, iou_thr):
    pred_root = Path(pred_root)
    label_dir = pred_root / "labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Prediction labels not found: {label_dir}")

    rows = []
    total_small_gt = 0
    matched_small_gt = 0

    for img_path in images:
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue

        gt_path = image_to_label_path(img_path)
        pred_path = label_dir / f"{img_path.stem}.txt"

        gts = parse_yolo(gt_path, w, h, is_pred=False)
        preds = parse_yolo(pred_path, w, h, is_pred=True)
        preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

        small_gts = [
            g for g in gts
            if g["area"] / float(w * h + 1e-9) <= small_area_ratio
        ]
        total_small_gt += len(small_gts)

        used = set()
        for gt in small_gts:
            best_iou = 0.0
            best_idx = -1

            for i, pred in enumerate(preds):
                if i in used:
                    continue
                if pred["cls"] != gt["cls"]:
                    continue

                v = iou(gt["box"], pred["box"])
                if v > best_iou:
                    best_iou = v
                    best_idx = i

            if best_idx >= 0 and best_iou >= iou_thr:
                used.add(best_idx)
                pred = preds[best_idx]

                err = center_error(gt["box"], pred["box"])
                norm = err / math.sqrt(w * w + h * h)

                rows.append({
                    "method": method,
                    "image": str(img_path),
                    "cls": gt["cls"],
                    "iou": best_iou,
                    "center_error_px": err,
                    "normalized_center_error": norm,
                    "gt_area_ratio": gt["area"] / float(w * h + 1e-9),
                    "conf": pred["conf"],
                })
                matched_small_gt += 1

    return rows, total_small_gt, matched_small_gt


def summarize(rows, total, matched):
    if not rows:
        return {
            "small_gt": total,
            "matched": matched,
            "match_rate": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p90": None,
            "norm": None,
        }

    errs = np.array([r["center_error_px"] for r in rows], dtype=np.float32)
    norms = np.array([r["normalized_center_error"] for r in rows], dtype=np.float32)

    return {
        "small_gt": total,
        "matched": matched,
        "match_rate": matched / max(total, 1),
        "mean": float(np.mean(errs)),
        "median": float(np.median(errs)),
        "std": float(np.std(errs)),
        "p90": float(np.percentile(errs, 90)),
        "norm": float(np.mean(norms)),
    }


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_cdf(path, rows_by_method):
    plt.figure()
    for method, rows in rows_by_method.items():
        errs = np.array([r["center_error_px"] for r in rows], dtype=np.float32)
        if len(errs) == 0:
            continue
        x = np.sort(errs)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=method)

    plt.xlabel("Center error (pixels)")
    plt.ylabel("Cumulative probability")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_box(path, rows_by_method):
    data, labels = [], []
    for method, rows in rows_by_method.items():
        errs = [r["center_error_px"] for r in rows]
        if errs:
            data.append(errs)
            labels.append(method)

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Center error (pixels)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-space-pred", required=True)
    parser.add_argument("--input-space-pred", required=True)
    parser.add_argument("--out", default="runs/offset_analysis")
    parser.add_argument("--small-area-ratio", type=float, default=0.01)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = load_images(args.data, args.split)
    print(f"[INFO] images: {len(images)}")

    configs = {
        "Image-space GLF": args.image_space_pred,
        "IS-A640-GLF": args.input_space_pred,
    }

    all_rows = []
    rows_by_method = {}
    summary_rows = []

    for method, pred in configs.items():
        rows, total, matched = collect_for_method(
            images, pred, method, args.small_area_ratio, args.iou_thr
        )
        rows_by_method[method] = rows
        all_rows.extend(rows)

        s = summarize(rows, total, matched)
        summary_rows.append({
            "method": method,
            "small_gt": s["small_gt"],
            "matched": s["matched"],
            "match_rate": round(s["match_rate"], 4),
            "mean_center_error_px": None if s["mean"] is None else round(s["mean"], 4),
            "median_center_error_px": None if s["median"] is None else round(s["median"], 4),
            "std_center_error_px": None if s["std"] is None else round(s["std"], 4),
            "p90_center_error_px": None if s["p90"] is None else round(s["p90"], 4),
            "mean_normalized_error": None if s["norm"] is None else round(s["norm"], 6),
        })

    write_csv(
        out / "center_offset_matches.csv",
        all_rows,
        ["method", "image", "cls", "iou", "center_error_px",
         "normalized_center_error", "gt_area_ratio", "conf"]
    )

    write_csv(
        out / "center_offset_summary.csv",
        summary_rows,
        ["method", "small_gt", "matched", "match_rate",
         "mean_center_error_px", "median_center_error_px",
         "std_center_error_px", "p90_center_error_px",
         "mean_normalized_error"]
    )

    plot_cdf(out / "center_offset_cdf.png", rows_by_method)
    plot_box(out / "center_offset_boxplot.png", rows_by_method)

    print("\n===== Center Offset Summary =====")
    for r in summary_rows:
        print(r)

    print(f"\nSaved to {out}")
    print(f"Summary: {out / 'center_offset_summary.csv'}")
    print(f"CDF: {out / 'center_offset_cdf.png'}")
    print(f"Boxplot: {out / 'center_offset_boxplot.png'}")


if __name__ == "__main__":
    main()
