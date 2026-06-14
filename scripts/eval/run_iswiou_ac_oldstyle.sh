#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:?weights path required}
SPLIT=${2:-test}
NAME=${3:-AC_ISWIoU_IS_A640_GLF_test}
# A + C: use C-trained weights and final A oldstyle input-space 640 GLF validator.
python val_a640_native_oldstyle.py --weights "${WEIGHTS}" --data dataset/data.yaml --split "${SPLIT}" --mode fixed --tile 640 --overlap 0.20 --project runs/val_ac_iswiou_a640_oldstyle --name "${NAME}" --device 0 --plots
