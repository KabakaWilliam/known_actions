#!/usr/bin/env bash
# Naming convention:
#   <train_dataset>           — in-domain only (no OOD datasets; test split is the in-domain result)
#   <train_dataset>_x_<ood>   — single OOD pair
#   <train_dataset>_ood_all   — train on one dataset, OOD on all others
#
# Each experiment produces traces/classifiers/<tag>/results.json with:
#   - test_report      (in-domain test accuracy)
#   - ood_reports      (one entry per OOD dataset)
#   - classifier.pkl / lstm_model.pt
#
# Trains classifiers on timing-corrupted traces to measure robustness against
# an adversary that inserts random pauses between actions.
#
# Usage:
#   bash train_delayed_xgb_classifier.sh           # run all uncommented experiments
#   bash train_delayed_xgb_classifier.sh wiki      # run only experiments whose tag matches 'wiki'

set -euo pipefail
trap 'kill 0' EXIT
cd "$(dirname "$0")"


TRACES_DIR=./traces
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/VData/linna4335/.apptainer_cache}"
export APPTAINER_CACHEDIR
FILTER="${1:-}"

run_experiment() {
    local tag="$1"; shift
    if [[ -z "$FILTER" || "$tag" == *"$FILTER"* ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Experiment: $tag"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python trace_analyzer.py --traces-dir "$TRACES_DIR" --tag "$tag" "$@"
        echo ""
        echo "  Saved → $TRACES_DIR/classifiers/$tag/"
    fi
}

# Full agent roster (14 agents with traces across all datasets):
#   webgames excludes seed_2_lite (no webgames traces)
AGENTS_ALL="gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash seed_2_lite"
AGENTS_NO_SEED="gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash"

# ══════════════════════════════════════════════════════════════════════════════
# RED LINE — Training poisoning: train AND test on corrupted data.
# The classifier re-learns on noisy timing; shows how much retraining helps.
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment wiki_delayed_xgb_500ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents $AGENTS_ALL \
#     --classifiers XGBoost \
#     --add-random-delay 500

# run_experiment wiki_delayed_xgb_1000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents $AGENTS_ALL \
#     --classifiers XGBoost \
#     --add-random-delay 1000

# run_experiment wiki_delayed_xgb_2000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents $AGENTS_ALL \
#     --classifiers XGBoost \
#     --add-random-delay 2000

# run_experiment wiki_delayed_xgb_5000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents $AGENTS_ALL \
#     --classifiers XGBoost \
#     --add-random-delay 5000

# ══════════════════════════════════════════════════════════════════════════════
# BLUE LINE — Test-time jitter: train on clean data, corrupt test split only.
# Simulates an agent inserting pauses against a deployed (non-retrained) classifier.
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment wiki_jitter_test_500ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents $AGENTS_ALL \
#     --classifiers XGBoost \
#     --test-random-delay 500

run_experiment wiki_jitter_test_1000ms \
    --train-datasets 2wikimultihop \
    --ood-datasets frames \
    --agents $AGENTS_ALL \
    --classifiers XGBoost \
    --test-random-delay 1000

# run_experiment wiki_jitter_test_2000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents $AGENTS_ALL \
#     --classifiers XGBoost \
#     --test-random-delay 2000

# run_experiment wiki_jitter_test_5000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents $AGENTS_ALL \
#     --classifiers XGBoost \
#     --test-random-delay 5000
