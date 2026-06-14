#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:?用法: bash run_a640_slice_only.sh /path/to/best.pt [split] [实验名]}
SPLIT=${2:-test}
NAME=${3:-A640_slice_only_${SPLIT}}

# 消融：只看自适应 640 切片，不融合整图结果。
python val_a640_glf.py \
  --weights "$WEIGHTS" \
  --data dataset/data.yaml \
  --split "$SPLIT" \
  --imgsz 960 \
  --mode slice_only \
  --tile 640 \
  --overlap 0.20 \
  --conf 0.01 \
  --pred-iou 0.70 \
  --merge-iou 0.55 \
  --batch 8 \
  --device 0 \
  --project runs/val_a640 \
  --name "$NAME" \
  --force-exit
