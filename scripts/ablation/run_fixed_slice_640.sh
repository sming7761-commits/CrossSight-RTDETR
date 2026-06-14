#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:?用法: bash run_fixed_slice_640.sh /path/to/best.pt [split] [实验名]}
SPLIT=${2:-test}
NAME=${3:-A640_fixed_${SPLIT}}

# 兼容旧文件名：整图 + 固定 640x640 切片 + 预测融合。
bash run_a640_fixed.sh "$WEIGHTS" "$SPLIT" "$NAME"
