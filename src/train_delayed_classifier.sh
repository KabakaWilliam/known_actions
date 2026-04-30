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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
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

# Load a pre-trained classifier from SRC_TAG and evaluate with --test-random-delay.
# Skips training entirely — use for jitter_test variants of already-trained experiments.
run_jitter_experiment() {
    local tag="$1"; local src_tag="$2"; shift 2
    run_experiment "$tag" --load-classifier "$TRACES_DIR/classifiers/$src_tag" "$@"
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
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --add-random-delay 500

# run_experiment wiki_delayed_xgb_1000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --add-random-delay 1000

# run_experiment wiki_delayed_xgb_2000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --add-random-delay 2000

# run_experiment wiki_delayed_xgb_5000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --add-random-delay 5000

# ══════════════════════════════════════════════════════════════════════════════
# BLUE LINE — Test-time jitter: train on clean data, corrupt test split only.
# Simulates an agent inserting pauses against a deployed (non-retrained) classifier.
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment wiki_jitter_test_500ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 500

# run_experiment wiki_jitter_test_1000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 1000

# run_experiment wiki_jitter_test_2000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 2000

# run_experiment wiki_jitter_test_5000ms \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 5000

### FRAMES
# ══════════════════════════════════════════════════════════════════════════════
# RED LINE — Training poisoning: train AND test on corrupted data.
# The classifier re-learns on noisy timing; shows how much retraining helps.
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment frames_delayed_xgb_500ms \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --add-random-delay 500

# run_experiment frames_delayed_xgb_1000ms \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --add-random-delay 1000

# run_experiment frames_delayed_xgb_2000ms \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --add-random-delay 2000

run_experiment frames_delayed_xgb_5000ms \
    --train-datasets frames \
    --agents ${AGENTS_ALL} \
    --resplit-datasets frames \
    --resplit-n-per-agent 300 \
    --classifiers XGBoost \
    --add-random-delay 5000

# ══════════════════════════════════════════════════════════════════════════════
# BLUE LINE — Test-time jitter: train on clean data, corrupt test split only.
# Simulates an agent inserting pauses against a deployed (non-retrained) classifier.
# ══════════════════════════════════════════════════════════════════════════════
# run_experiment frames_jitter_test_500ms \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --test-random-delay 500

# run_experiment frames_jitter_test_1000ms \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --test-random-delay 1000

# run_experiment frames_jitter_test_2000ms \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --test-random-delay 2000

# run_experiment frames_jitter_test_5000ms \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --test-random-delay 5000



### webshop
# ══════════════════════════════════════════════════════════════════════════════
# RED LINE — Training poisoning: train AND test on corrupted data.
# The classifier re-learns on noisy timing; shows how much retraining helps.
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment webshop_delayed_xgb_500ms \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --add-random-delay 500

# run_experiment webshop_delayed_xgb_1000ms \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --add-random-delay 1000

# run_experiment webshop_delayed_xgb_2000ms \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --add-random-delay 2000

# run_experiment webshop_delayed_xgb_5000ms \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --add-random-delay 5000

# ══════════════════════════════════════════════════════════════════════════════
# BLUE LINE — Test-time jitter: train on clean data, corrupt test split only.
# Simulates an agent inserting pauses against a deployed (non-retrained) classifier.
# ══════════════════════════════════════════════════════════════════════════════
# run_experiment webshop_jitter_test_500ms \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 500

# run_experiment webshop_jitter_test_1000ms \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 1000

# run_experiment webshop_jitter_test_2000ms \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 2000

# run_experiment webshop_jitter_test_5000ms \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 5000


### DeepShop
# ══════════════════════════════════════════════════════════════════════════════
# RED LINE — Training poisoning: train AND test on corrupted data.
# The classifier re-learns on noisy timing; shows how much retraining helps.
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment deepshop_delayed_xgb_500ms \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --add-random-delay 500

# run_experiment deepshop_delayed_xgb_1000ms \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --add-random-delay 1000

# run_experiment deepshop_delayed_xgb_2000ms \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --add-random-delay 2000

# run_experiment deepshop_delayed_xgb_5000ms \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --add-random-delay 5000

# ══════════════════════════════════════════════════════════════════════════════
# BLUE LINE — Test-time jitter: train on clean data, corrupt test split only.
# Simulates an agent inserting pauses against a deployed (non-retrained) classifier.
# ══════════════════════════════════════════════════════════════════════════════
# run_experiment deepshop_jitter_test_500ms \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --test-random-delay 500

# run_experiment deepshop_jitter_test_1000ms \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --test-random-delay 1000

# run_experiment deepshop_jitter_test_2000ms \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --test-random-delay 2000

# run_experiment deepshop_jitter_test_5000ms \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --test-random-delay 5000


# ══════════════════════════════════════════════════════════════════════════════
# JITTER-ONLY EVAL — load best trained classifiers, apply test-time delay only.
# Source: the *_ood_all classifiers trained on clean data.
# Results saved to traces/classifiers/<tag>/ (same structure as full runs).
# ══════════════════════════════════════════════════════════════════════════════

### frames_ood_all → jitter
# run_jitter_experiment frames_ood_all_jitter_500ms  frames_ood_all \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --test-random-delay 500

# run_jitter_experiment frames_ood_all_jitter_1000ms frames_ood_all \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --test-random-delay 1000

# run_jitter_experiment frames_ood_all_jitter_2000ms frames_ood_all \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --test-random-delay 2000

# run_jitter_experiment frames_ood_all_jitter_5000ms frames_ood_all \
#     --train-datasets frames \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --classifiers XGBoost \
#     --test-random-delay 5000

# ### wiki_ood_all → jitter
# run_jitter_experiment wiki_ood_all_jitter_500ms  wiki_ood_all \
#     --train-datasets 2wikimultihop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 500

# run_jitter_experiment wiki_ood_all_jitter_1000ms wiki_ood_all \
#     --train-datasets 2wikimultihop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 1000

# run_jitter_experiment wiki_ood_all_jitter_2000ms wiki_ood_all \
#     --train-datasets 2wikimultihop \
#     --ood-datasets deepshop frames webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 2000

# run_jitter_experiment wiki_ood_all_jitter_5000ms wiki_ood_all \
#     --train-datasets 2wikimultihop \
#     --ood-datasets deepshop frames webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 5000

# ### deepshop_ood_all → jitter
# run_jitter_experiment deepshop_ood_all_jitter_500ms  deepshop_ood_all \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --test-random-delay 500

# run_jitter_experiment deepshop_ood_all_jitter_1000ms deepshop_ood_all \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --test-random-delay 1000

# run_jitter_experiment deepshop_ood_all_jitter_2000ms deepshop_ood_all \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --test-random-delay 2000

# run_jitter_experiment deepshop_ood_all_jitter_5000ms deepshop_ood_all \
#     --train-datasets deepshop \
#     --agents ${AGENTS_ALL} \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --classifiers XGBoost \
#     --test-random-delay 5000

# ### webshop_ood_all → jitter
# run_jitter_experiment webshop_ood_all_jitter_500ms  webshop_ood_all \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 500

# run_jitter_experiment webshop_ood_all_jitter_1000ms webshop_ood_all \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 1000

# run_jitter_experiment webshop_ood_all_jitter_2000ms webshop_ood_all \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 2000

# run_jitter_experiment webshop_ood_all_jitter_5000ms webshop_ood_all \
#     --train-datasets webshop \
#     --agents ${AGENTS_ALL} \
#     --classifiers XGBoost \
#     --test-random-delay 5000