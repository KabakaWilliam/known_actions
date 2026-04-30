#!/usr/bin/env bash
# train_learning_curve.sh — Sample-efficiency experiment: training traces per agent vs. F1.
#
# Trains classifiers at increasing per-agent training budgets (val/test unchanged)
# to answer: how many labeled traces per agent are needed for a performant classifier?
#
# Four experiment directions (two dataset pairs):
#   wiki_2_frames   — train: 2WikiMultiHop, OOD: FRAMES
#   frames_2_wiki   — train: FRAMES,        OOD: 2WikiMultiHop
#   webshop_2_deepshop — train: WebShop,    OOD: DeepShop
#   deepshop_2_webshop — train: DeepShop,   OOD: WebShop
#
# Output:
#   traces/classifiers/learning_curve/{tag}_n{N}/results.json
#
# Usage:
#   bash train_learning_curve.sh              # all four directions
#   bash train_learning_curve.sh wiki         # only wiki→frames
#   bash train_learning_curve.sh webshop      # only webshop→deepshop

set -euo pipefail
trap 'kill 0' EXIT
cd "$(dirname "$0")"

TRACES_DIR=./traces
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/VData/linna4335/.apptainer_cache}"
export APPTAINER_CACHEDIR
FILTER="${1:-}"

# AGENTS=(gpt_5_4 gemma_4_26B_A4B_it glm_4.6v_flash qwen3_5_27b qwen3vl_8b qwen3vl_30b_a3b uitars_7b gemini_3_1)
AGENTS=(gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash seed_2_lite)
SIZES=(5 10 20 50 100 150 all)

run_size() {
    local base_tag="$1"; shift
    local n="$1";         shift   # integer or "all"
    local extra=("$@")

    local tag="learning_curve/${base_tag}_n${n}"

    if [[ -n "$FILTER" && "$base_tag" != *"$FILTER"* ]]; then
        return
    fi

    local n_arg=()
    [[ "$n" != "all" ]] && n_arg=(--n-train-per-agent "$n")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment : $base_tag   n_train_per_agent=${n}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python trace_analyzer.py \
        --traces-dir "$TRACES_DIR" \
        --tag        "$tag" \
        "${extra[@]}" \
        "${n_arg[@]}" \
        --agents "${AGENTS[@]}"
    echo "  Saved → $TRACES_DIR/classifiers/$tag/results.json"
    echo ""
}

# ── wiki → frames ──────────────────────────────────────────────────────────────
# for n in "${SIZES[@]}"; do
#     run_size wiki_2_frames "$n" \
#         --train-datasets 2wikimultihop \
#         --ood-datasets   frames
# done

# ── frames → wiki ──────────────────────────────────────────────────────────────
# frames has no pre-split dirs so we resplit; cap at 300/agent to match 2wiki budget.
for n in "${SIZES[@]}"; do
    run_size frames_2_wiki "$n" \
        --train-datasets   frames \
        --resplit-datasets frames \
        --resplit-n-per-agent 300 \
        --ood-datasets     2wikimultihop
done

# ── webshop → deepshop ─────────────────────────────────────────────────────────
for n in "${SIZES[@]}"; do
    run_size webshop_2_deepshop "$n" \
        --train-datasets webshop \
        --ood-datasets   deepshop 
done

# ── deepshop → webshop ─────────────────────────────────────────────────────────
# deepshop has no pre-split dirs; resplit without cap (naturally ~150/agent).
# for n in "${SIZES[@]}"; do
#     run_size deepshop_2_webshop "$n" \
#         --train-datasets   deepshop \
#         --resplit-datasets deepshop \
#         --resplit-n-per-agent 150 \
#         --ood-datasets     webshop
# done

echo "All learning-curve experiments complete."
echo "Plot: python plot_learning_curve.py"
echo "      python plot_learning_curve.py --tags webshop_2_deepshop deepshop_2_webshop"
