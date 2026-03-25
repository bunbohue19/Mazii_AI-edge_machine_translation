#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run FLORES-200 + wmt22-comet-da benchmark for ALL supported language pairs.
#
# Usage:
#   bash run.sh                    # all pairs, 200 samples each
#   bash run.sh --n-samples 1012   # full devtest split
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="$SCRIPT_DIR/results"

TARGET_LANGS=(
    "Vietnamese"
    "Traditional Chinese"
    "Indonesian"
)

FAILED=()

for TGT in "${TARGET_LANGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Benchmarking: Japanese → ${TGT}"
    echo "============================================================"

    python "$SCRIPT_DIR/benchmark.py" \
        --server-url   http://localhost:8501 \
        --src-lang     Japanese \
        --tgt-lang     "$TGT" \
        --flores-split devtest \
        --n-samples    200 \
        --temperature  0.20 \
        --concurrency  8 \
        --comet-batch  8 \
        --gpus         1 \
        --output-dir   "$OUTPUT_DIR" \
        "$@" || FAILED+=("$TGT")
done

echo ""
echo "============================================================"
echo "  All benchmarks complete."
echo "  Results saved in: $OUTPUT_DIR"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED pairs: ${FAILED[*]}"
fi
echo "============================================================"
