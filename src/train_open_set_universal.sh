#!/usr/bin/env bash
# train_open_set_universal.sh — LOO open-set experiments on multi-dataset settings.
#
# Two experiment blocks:
#   wiki+frames   — trains on 2WikiMultihopQA + FRAMES (resplit 50/25/25, cap 300/agent)
#   ws+deepshop   — trains on WebShop + DeepShop (deepshop resplit, no cap needed)
#
# For each block, one LOO run per agent: train on the other 13, evaluate whether
# the held-out agent's traces are detectable as unknown (AUROC, FPR95).
#
# Output directories:
#   traces/classifiers/wiki_frames_open_set/open_set_loo_<agent>/results.json
#   traces/classifiers/ws_deepshop_open_set/open_set_loo_<agent>/results.json
#
# Usage:
#   bash train_open_set_universal.sh               # all experiments
#   bash train_open_set_universal.sh gpt_5_4       # filter by agent name
#   bash train_open_set_universal.sh wiki           # filter by tag substring

# # default (unchanged behaviour)
# python plot_open_set_summary.py --traces-dir ./traces --plot scatter

# # multi-dataset results
# python plot_open_set_summary.py --loo-subdir wiki_frames_open_set
# python plot_open_set_summary.py --loo-subdir ws_deepshop_open_set \
#     --closed-set-tag webshop_ood_all


set -euo pipefail
trap 'kill 0' EXIT
cd "$(dirname "$0")"

TRACES_DIR=./traces
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/VData/linna4335/.apptainer_cache}"
export APPTAINER_CACHEDIR
FILTER="${1:-}"

ALL_AGENTS=(
    gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it
    glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b
    qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash seed_2_lite
)

run_loo() {
    local dataset_tag="$1"; shift   # e.g. wiki_frames_open_set
    local held_out="$1";    shift   # agent being held out
    local extra_args=("$@")         # --train-datasets ... --resplit-datasets ...

    local tag="${dataset_tag}/open_set_loo_${held_out}"

    # Skip if filter doesn't match tag or agent name
    if [[ -n "$FILTER" && "$tag" != *"$FILTER"* && "$held_out" != *"$FILTER"* ]]; then
        return
    fi

    local known_agents=()
    for agent in "${ALL_AGENTS[@]}"; do
        [[ "$agent" != "$held_out" ]] && known_agents+=("$agent")
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment : $dataset_tag"
    echo "  Held-out   : $held_out"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python trace_analyzer.py \
        --traces-dir "$TRACES_DIR" \
        --tag        "$tag" \
        "${extra_args[@]}" \
        --agents          "${known_agents[@]}" \
        --open-set-agents "$held_out"

    echo ""
    echo "  Saved → $TRACES_DIR/classifiers/$tag/results.json"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# Block 1 — wiki + frames
# frames only has frames_test/ → resplit 50/25/25, cap at 300/agent to match
# the 2wiki budget (~150/75/75 per agent).
# ══════════════════════════════════════════════════════════════════════════════
# for held_out in "${ALL_AGENTS[@]}"; do
#     run_loo wiki_frames_open_set "$held_out" \
#         --train-datasets   2wikimultihop frames \
#         --resplit-datasets frames \
#         --resplit-n-per-agent 300
# done

# ══════════════════════════════════════════════════════════════════════════════
# Block 2 — webshop + deepshop
# deepshop only has deepshop_ood/ → resplit without cap (naturally ~150/agent,
# split gives ~75/37/37, comparable to the webshop budget).
# ══════════════════════════════════════════════════════════════════════════════
for held_out in "${ALL_AGENTS[@]}"; do
    run_loo ws_deepshop_open_set "$held_out" \
        --train-datasets   webshop deepshop \
        --resplit-datasets deepshop
done

echo "All open-set LOO experiments complete."
echo "Plot: python plot_open_set_summary.py --loo-subdir wiki_frames_open_set"
echo "      python plot_open_set_summary.py --loo-subdir ws_deepshop_open_set --closed-set-tag webshop_ood_all"



