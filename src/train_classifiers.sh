#!/usr/bin/env bash
# train_classifiers_new.sh — canonical per-dataset experiments (individual checkpoint labels)
#
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
# Usage:
#   bash train_classifiers_new.sh              # run all uncommented experiments
#   bash train_classifiers_new.sh wiki         # run only experiments whose tag matches 'wiki'
#   bash train_classifiers_new.sh frames_ood   # run all frames OOD experiments

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
# 2WIKIMULTIHOP
# ══════════════════════════════════════════════════════════════════════════════

# ── In-domain ─────────────────────────────────────────────────────────────────
run_experiment wiki_xgb_ood_frames \
    --train-datasets 2wikimultihop \
    --ood-datasets frames\
    --agents $AGENTS_ALL \
    --classifiers XGBoost

# # ── OOD: all others ───────────────────────────────────────────────────────────
# run_experiment wiki_ood_all \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames webshop deepshop \
#     --agents $AGENTS_ALL

# # ── OOD: individual pairs ─────────────────────────────────────────────────────
# # run_experiment wiki_x_frames \
# #     --train-datasets 2wikimultihop \
# #     --ood-datasets frames \
# #     --agents $AGENTS_ALL

# # run_experiment wiki_x_webshop \
# #     --train-datasets 2wikimultihop \
# #     --ood-datasets webshop \
# #     --agents $AGENTS_ALL

# # run_experiment wiki_x_deepshop \
# #     --train-datasets 2wikimultihop \
# #     --ood-datasets deepshop \
# #     --agents $AGENTS_ALL

# # run_experiment wiki_x_webgames \
# #     --train-datasets 2wikimultihop \
# #     --ood-datasets webgames \
# #     --agents $AGENTS_NO_SEED


# # ══════════════════════════════════════════════════════════════════════════════
# # FRAMES
# # frames only has a _test dir — pool and resplit (300 → ~150/75/75 per agent)
# # ══════════════════════════════════════════════════════════════════════════════

# # ── In-domain ─────────────────────────────────────────────────────────────────
# # run_experiment frames \
# #     --train-datasets frames \
# #     --resplit-datasets frames \
# #     --resplit-n-per-agent 300 \
# #     --agents $AGENTS_ALL

# # ── OOD: all others ───────────────────────────────────────────────────────────
# run_experiment frames_ood_all \
#     --train-datasets frames \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --ood-datasets 2wikimultihop webshop deepshop \
#     --agents $AGENTS_ALL

# # ── OOD: individual pairs ─────────────────────────────────────────────────────
# # run_experiment frames_x_wiki \
# #     --train-datasets frames \
# #     --resplit-datasets frames \
# #     --resplit-n-per-agent 300 \
# #     --ood-datasets 2wikimultihop \
# #     --agents $AGENTS_ALL

# # run_experiment frames_x_webshop \
# #     --train-datasets frames \
# #     --resplit-datasets frames \
# #     --resplit-n-per-agent 300 \
# #     --ood-datasets webshop \
# #     --agents $AGENTS_ALL

# # run_experiment frames_x_deepshop \
# #     --train-datasets frames \
# #     --resplit-datasets frames \
# #     --resplit-n-per-agent 300 \
# #     --ood-datasets deepshop \
# #     --agents $AGENTS_ALL


# # ══════════════════════════════════════════════════════════════════════════════
# # WEBSHOP
# # ══════════════════════════════════════════════════════════════════════════════

# # ── In-domain ─────────────────────────────────────────────────────────────────
# # run_experiment webshop \
# #     --train-datasets webshop \
# #     --agents $AGENTS_ALL

# # ── OOD: all others ───────────────────────────────────────────────────────────
# run_experiment webshop_ood_all \
#     --train-datasets webshop \
#     --ood-datasets 2wikimultihop frames deepshop webgames \
#     --agents $AGENTS_ALL

# # ── OOD: individual pairs ─────────────────────────────────────────────────────
# # run_experiment webshop_x_deepshop \
# #     --train-datasets webshop \
# #     --ood-datasets deepshop \
# #     --agents $AGENTS_ALL

# # run_experiment webshop_x_wiki \
# #     --train-datasets webshop \
# #     --ood-datasets 2wikimultihop \
# #     --agents $AGENTS_ALL

# # run_experiment webshop_x_frames \
# #     --train-datasets webshop \
# #     --ood-datasets frames \
# #     --agents $AGENTS_ALL

# # run_experiment webshop_x_webgames \
# #     --train-datasets webshop \
# #     --ood-datasets webgames \
# #     --agents $AGENTS_NO_SEED


# # ══════════════════════════════════════════════════════════════════════════════
# # DEEPSHOP
# # deepshop only has a _ood split — pool and resplit (150 → ~75/37/37 per agent)
# # ══════════════════════════════════════════════════════════════════════════════

# # ── In-domain ─────────────────────────────────────────────────────────────────
# # run_experiment deepshop \
# #     --train-datasets deepshop \
# #     --resplit-datasets deepshop \
# #     --resplit-n-per-agent 150 \
# #     --agents $AGENTS_ALL

# # ── OOD: all others ───────────────────────────────────────────────────────────
# run_experiment deepshop_ood_all \
#     --train-datasets deepshop \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --ood-datasets 2wikimultihop frames webshop webgames \
#     --agents $AGENTS_ALL

# # ── OOD: individual pairs ─────────────────────────────────────────────────────
# # run_experiment deepshop_x_webshop \
# #     --train-datasets deepshop \
# #     --resplit-datasets deepshop \
# #     --resplit-n-per-agent 150 \
# #     --ood-datasets webshop \
# #     --agents $AGENTS_ALL

# # run_experiment deepshop_x_wiki \
# #     --train-datasets deepshop \
# #     --resplit-datasets deepshop \
# #     --resplit-n-per-agent 150 \
# #     --ood-datasets 2wikimultihop \
# #     --agents $AGENTS_ALL


# # ══════════════════════════════════════════════════════════════════════════════
# # WEBGAMES
# # seed_2_lite excluded — no traces yet
# # ══════════════════════════════════════════════════════════════════════════════

# # ── In-domain ─────────────────────────────────────────────────────────────────
# # run_experiment webgames \
# #     --train-datasets webgames \
# #     --agents $AGENTS_NO_SEED

# # ── OOD: all others ───────────────────────────────────────────────────────────
# run_experiment webgames_ood_all \
#     --train-datasets webgames \
#     --ood-datasets 2wikimultihop frames webshop deepshop \
#     --agents $AGENTS_NO_SEED

# ── OOD: individual pairs ─────────────────────────────────────────────────────
# run_experiment webgames_x_wiki \
#     --train-datasets webgames \
#     --ood-datasets 2wikimultihop \
#     --agents $AGENTS_NO_SEED

# run_experiment webgames_x_webshop \
#     --train-datasets webgames \
#     --ood-datasets webshop \
#     --agents $AGENTS_NO_SEED

# run_experiment webgames_x_deepshop \
#     --train-datasets webgames \
#     --ood-datasets deepshop \
#     --agents $AGENTS_NO_SEED


echo ""
echo "All experiments complete."
