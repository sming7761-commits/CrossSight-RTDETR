#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:?用法: bash run_a640_adaptive.sh /path/to/best.pt [split] [实验名]}
SPLIT=${2:-test}
NAME=${3:-A640_GLF_adaptive_${SPLIT}}

# A 创新点：整图 + 自适应 640x640 网格切片 + 坐标映射 + 预测结果融合。
# 不使用 ROI 复检、不使用密度图、不使用自适应区域二次检测。
python val_a640_glf.py \
  --weights "$WEIGHTS" \
  --data dataset/data.yaml \
  --split "$SPLIT" \
  --imgsz 960 \
  --mode adaptive \
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
