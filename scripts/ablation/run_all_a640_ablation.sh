#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

WEIGHTS=${1:-runs/train/baseline_200_clean/weights/best.pt}
SPLIT=${2:-test}
mkdir -p logs

bash run_no_slice_same_metric.sh "$WEIGHTS" "$SPLIT" "baseline_no_slice_${SPLIT}" | tee "logs/a640_no_slice_${SPLIT}.log"
bash run_a640_slice_only.sh "$WEIGHTS" "$SPLIT" "A640_slice_only_${SPLIT}" | tee "logs/a640_slice_only_${SPLIT}.log"
bash run_a640_fixed.sh "$WEIGHTS" "$SPLIT" "A640_fixed_${SPLIT}" | tee "logs/a640_fixed_${SPLIT}.log"
bash run_a640_adaptive.sh "$WEIGHTS" "$SPLIT" "A640_GLF_adaptive_${SPLIT}" | tee "logs/a640_adaptive_${SPLIT}.log"
python collect_paper_results.py --root runs/val_a640 --out "runs/val_a640/summary_${SPLIT}.csv"
