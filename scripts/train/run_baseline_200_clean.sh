#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p logs

# 干净基线：不加载任何权重，从 rtdetr-r18.yaml 从零训练 200 轮。
python train.py \
  --model ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml \
  --data dataset/data.yaml \
  --imgsz 960 \
  --epochs 200 \
  --batch 8 \
  --workers 12 \
  --device 0 \
  --project runs/train \
  --name baseline_200_clean
