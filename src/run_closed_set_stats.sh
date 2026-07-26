#!/usr/bin/env bash
# Run the four canonical closed-set XGBoost evaluations sequentially.
#
# Multi-seed macro-F1 is enabled by default in trace_analyzer.py. When no
# explicit --classifier-seeds are supplied, each experiment generates and
# records 10 random seeds in its closed_set_macro_f1.json.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRACES_DIR="${TRACES_DIR:-./traces}"
FILTER="${1:-}"

# Keep CPU-side preprocessing and BLAS bounded. XGBoost training uses the GPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

AGENTS_ALL=(
    gpt_5_4
    claude_opus_4_6
    gemma-4-31B-it
    gemma_4_26B_A4B_it
    glm_4.6v
    glm_4.6v_flash
    qwen3vl_8b
    qwen3vl_30b_a3b
    qwen3_5_27b
    qwen3_5_9b
    uitars_7b
    gemini_3_1
    gemini_3_flash
    seed_2_lite
)

run_experiment() {
    local dataset="$1"
    shift
    if [[ -n "$FILTER" && "$dataset" != *"$FILTER"* ]]; then
        return
    fi

    local tag="closed_set_stats/${dataset}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Closed set: ${dataset}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "$PYTHON_BIN" trace_analyzer.py \
        --traces-dir "$TRACES_DIR" \
        --tag "$tag" \
        --classifiers XGBoost \
        --agents "${AGENTS_ALL[@]}" \
        "$@"
    echo "Saved → $TRACES_DIR/classifiers/$tag/closed_set_macro_f1.json"
    echo
}

run_experiment wiki \
    --train-datasets 2wikimultihop

run_experiment frames \
    --train-datasets frames \
    --resplit-datasets frames \
    --resplit-n-per-agent 300

run_experiment webshop \
    --train-datasets webshop

run_experiment deepshop \
    --train-datasets deepshop \
    --resplit-datasets deepshop \
    --resplit-n-per-agent 150

