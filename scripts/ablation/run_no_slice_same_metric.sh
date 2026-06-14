#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:?用法: bash run_no_slice_same_metric.sh /path/to/best.pt [split] [实验名]}
SPLIT=${2:-test}
NAME=${3:-baseline_no_slice_${SPLIT}}

# 使用 A640-GLF 脚本里的同一套评估器跑整图基线，和 A640 方法公平比较。
python val_a640_glf.py \
  --weights "$WEIGHTS" \
  --data dataset/data.yaml \
  --split "$SPLIT" \
  --imgsz 960 \
  --mode no_slice \
  --conf 0.01 \
  --pred-iou 0.70 \
  --merge-iou 0.55 \
  --batch 8 \
  --device 0 \
  --project runs/val_a640 \
  --name "$NAME" \
  --force-exit
