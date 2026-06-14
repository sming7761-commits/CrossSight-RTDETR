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
NAME="${4:-rtdetr_r18_hfgmf_200}"
NO_AMP="${5:-0}"
CMD=(python train.py \
  --model ultralytics/cfg/models/rt-detr/rtdetr-r18-hfgmf.yaml \
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
else
  mkdir -p ultralytics/assets
  if [ ! -f ultralytics/assets/bus.jpg ]; then
    python - <<'PYBUS'
from pathlib import Path
import numpy as np
out = Path('ultralytics/assets/bus.jpg')
out.parent.mkdir(parents=True, exist_ok=True)
img = np.full((640, 640, 3), 127, dtype=np.uint8)
try:
    from PIL import Image
    Image.fromarray(img).save(out)
except Exception:
    import cv2
    cv2.imwrite(str(out), img)
print(out.resolve())
PYBUS
  fi
  CMD+=(--amp)
fi

# Optional warm start. For formal paper training from scratch, pass none as the first argument.
if [[ -n "$WEIGHTS_OR_PRETRAIN" && "$WEIGHTS_OR_PRETRAIN" != "none" ]]; then
  CMD+=(--pretrain "$WEIGHTS_OR_PRETRAIN")
fi

"${CMD[@]}"
