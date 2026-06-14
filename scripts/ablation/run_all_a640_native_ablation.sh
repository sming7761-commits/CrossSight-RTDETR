#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:?用法: bash run_all_a640_native_ablation.sh /path/to/best.pt [val|test]}
SPLIT=${2:-test}

# 1) 原生整图 baseline：完全走你原来的 Ultralytics val.py 口径。
bash run_val_native_plots.sh "$WEIGHTS" "$SPLIT" "native_baseline_${SPLIT}"

# 2) A 模块消融：走 val_a640_native.py，使用 Ultralytics 原生 DetMetrics/AP/绘图逻辑。
bash run_a640_native_slice_only.sh "$WEIGHTS" "$SPLIT" "A640_slice_only_native_${SPLIT}"
bash run_a640_native_fixed.sh "$WEIGHTS" "$SPLIT" "A640_fixed_native_${SPLIT}"
bash run_a640_native_adaptive.sh "$WEIGHTS" "$SPLIT" "A640_GLF_adaptive_native_${SPLIT}"

echo "完成。结果目录："
echo "  原生 baseline: runs/val_native/native_baseline_${SPLIT}"
echo "  A640 native:   runs/val_a640_native/"
echo "查看 A640 最终结果："
echo "  cat runs/val_a640_native/A640_GLF_adaptive_native_${SPLIT}/paper_data.txt"
