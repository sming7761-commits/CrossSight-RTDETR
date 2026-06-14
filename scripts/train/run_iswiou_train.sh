#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

PRETRAIN=${1:-none}
EPOCHS=${2:-200}
BATCH=${3:-8}
NAME=${4:-rtdetr_r18_iswiou_200}
MIX=${5:-0.70}
USE_AMP=${6:-1}
TAU=${7:-0.10}
TEMP=${8:-0.04}
RATIO=${9:-1.0}

ARGS="--model ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml --data dataset/data.yaml --imgsz 960 --epochs ${EPOCHS} --batch ${BATCH} --workers 4 --device 0 --project runs/train --name ${NAME} --iswiou --iswiou-mix ${MIX} --iswiou-tau ${TAU} --iswiou-temp ${TEMP} --iswiou-ratio ${RATIO}"
if [ "${PRETRAIN}" != "none" ] && [ -n "${PRETRAIN}" ]; then
  ARGS="${ARGS} --pretrain ${PRETRAIN}"
fi
if [ "${USE_AMP}" = "1" ]; then
  mkdir -p ultralytics/assets
  if [ ! -f ultralytics/assets/bus.jpg ]; then
    python - <<'PY'
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
PY
  fi
  ARGS="${ARGS} --amp"
fi
# Keep cuDNN acceleration but disable the v8 frontend that may trigger GET engine errors.
TORCH_CUDNN_V8_API_DISABLED=${TORCH_CUDNN_V8_API_DISABLED:-1} python train.py ${ARGS}
