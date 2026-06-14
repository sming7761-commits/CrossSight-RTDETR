#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:?用法: bash run_val_native_plots.sh /path/to/best.pt [split] [实验名]}
SPLIT=${2:-test}
NAME=${3:-native_val_${SPLIT}}

# Ultralytics 原生整图验证/画图。论文主表建议使用 val_a640_glf.py 的同一评估器结果。
python val.py \
  --weights "$WEIGHTS" \
  --data dataset/data.yaml \
  --split "$SPLIT" \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --project runs/val_native \
  --name "$NAME" \
  --plots
