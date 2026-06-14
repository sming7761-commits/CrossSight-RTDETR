#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS="${1:?Usage: bash run_msff_fe_ab_oldstyle.sh /path/to/best.pt [split] [name]}"
SPLIT="${2:-test}"
NAME="${3:-AB_MSFF_FE_IS_A640_GLF_${SPLIT}}"
python val_a640_native_oldstyle.py \
  --weights "$WEIGHTS" \
  --data dataset/data.yaml \
  --split "$SPLIT" \
  --mode adaptive \
  --tile 640 \
  --overlap 0.20 \
  --conf 0.001 \
  --max-det 1000 \
  --merge-iou 0.55 \
  --project runs/val_ab_msff_a640_oldstyle \
  --name "$NAME" \
  --device 0 \
  --plots
