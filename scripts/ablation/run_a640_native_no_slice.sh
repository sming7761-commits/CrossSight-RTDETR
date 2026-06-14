#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:?用法: bash run_a640_native_no_slice.sh /path/to/best.pt [split] [实验名]}
SPLIT=${2:-test}
NAME=${3:-baseline_no_slice_native_${SPLIT}}

# 这个脚本主要用来检查 val_a640_native.py 的 no_slice 口径是否接近原生 val.py。
# 论文主表 baseline 仍建议用你原来的 run_val_native_plots.sh / val.py。
python val_a640_native.py \
  --weights "$WEIGHTS" \
  --data dataset/data.yaml \
  --split "$SPLIT" \
  --imgsz 960 \
  --batch 8 \
  --view-batch 8 \
  --device 0 \
  --mode no_slice \
  --tile 640 \
  --overlap 0.20 \
  --merge-iou 0.55 \
  --project runs/val_a640_native \
  --name "$NAME" \
  --plots
