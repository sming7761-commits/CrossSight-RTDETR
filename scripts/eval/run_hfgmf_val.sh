#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS="${1:?Usage: bash run_hfgmf_val.sh /path/to/best.pt [split] [name]}"
SPLIT="${2:-test}"
NAME="${3:-B_HFGMF_native_${SPLIT}}"
python val.py \
  --weights "$WEIGHTS" \
  --data dataset/data.yaml \
  --split "$SPLIT" \
  --name "$NAME" \
  --imgsz 960 \
  --batch 4 \
  --device 0 \
  --project runs/val_hfgmf \
  --plots
