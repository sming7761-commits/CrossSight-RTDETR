#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:?用法: bash run_a640_native_adaptive.sh /path/to/best.pt [split] [实验名]}
SPLIT=${2:-test}
NAME=${3:-A640_GLF_adaptive_native_${SPLIT}}

python val_a640_native.py \
  --weights "$WEIGHTS" \
  --data dataset/data.yaml \
  --split "$SPLIT" \
  --imgsz 960 \
  --batch 8 \
  --view-batch 8 \
  --device 0 \
  --mode adaptive \
  --tile 640 \
  --overlap 0.20 \
  --merge-iou 0.55 \
  --project runs/val_a640_native \
  --name "$NAME" \
  --plots
