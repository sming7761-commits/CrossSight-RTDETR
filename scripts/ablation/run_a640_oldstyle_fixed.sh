#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:-/root/best.pt}
SPLIT=${2:-test}
NAME=${3:-A640_oldstyle_fixed_${SPLIT}}
python val_a640_native_oldstyle.py \
  --weights "$WEIGHTS" \
  --data dataset/data.yaml \
  --split "$SPLIT" \
  --mode fixed \
  --tile 640 \
  --overlap 0.20 \
  --conf 0.001 \
  --iou 0.70 \
  --max-det 1000 \
  --merge-iou 0.55 \
  --batch 4 \
  --slice-batch 8 \
  --device 0 \
  --project runs/val_a640_oldstyle \
  --name "$NAME" \
  --plots
