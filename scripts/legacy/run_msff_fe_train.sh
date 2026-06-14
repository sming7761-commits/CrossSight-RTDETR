#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS_OR_PRETRAIN="${1:-}"
EPOCHS="${2:-200}"
BATCH="${3:-4}"
NAME="${4:-rtdetr_r18_msff_fe_200}"
NO_AMP="${5:-0}"
CMD=(python train.py \
  --model ultralytics/cfg/models/rt-detr/rtdetr-r18-msff-fe.yaml \
  --data dataset/data.yaml \
  --name "$NAME" \
  --imgsz 960 \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --workers 4 \
  --device 0 \
  --project runs/train)

if [[ "$NO_AMP" == "1" ]]; then
  CMD+=(--no-amp)
fi

# Optional: pass /root/best.pt for warm-start debugging only. For formal paper training, leave the first arg empty.
if [[ -n "$WEIGHTS_OR_PRETRAIN" && "$WEIGHTS_OR_PRETRAIN" != "none" ]]; then
  CMD+=(--pretrain "$WEIGHTS_OR_PRETRAIN")
fi

"${CMD[@]}"
