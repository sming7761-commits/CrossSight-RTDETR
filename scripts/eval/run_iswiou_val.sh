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
NAME=${3:-C_ISWIoU_native_test}
python val.py --weights "${WEIGHTS}" --data dataset/data.yaml --split "${SPLIT}" --project runs/val_iswiou --name "${NAME}" --device 0 --plots
