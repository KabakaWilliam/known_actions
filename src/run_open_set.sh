#!/usr/bin/env bash
# run_open_set.sh — Leave-one-agent-out open-set recognition experiments.
#
# For each of the 6 agents, trains classifiers on the other 5, then evaluates
# whether the held-out agent's traces can be detected as "unknown" using
# max-confidence thresholding (AUROC, FPR95).
#
# Question answered: "Is this trace from one of the known agents, or from
# an agent outside my known set?"
#
# Usage:
#   bash run_open_set.sh                      # all 6 LOO experiments (2wikimultihop)
#   bash run_open_set.sh gpt_5_4              # only the gpt_5_4-held-out run
#   DATASET=frames bash run_open_set.sh       # use FRAMES instead of 2wikimultihop

set -euo pipefail
trap 'kill 0' EXIT
cd "$(dirname "$0")"

PYTHON=/opt/anaconda/envs/dispatch/bin/python
TRACES_DIR=./traces
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/VData/linna4335/.apptainer_cache}"
export APPTAINER_CACHEDIR
FILTER="${1:-}"

# ALL_AGENTS=(gpt_5_4 gemma_4_26B_A4B_it glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b uitars_7b)
ALL_AGENTS=(gpt_5_4 gemma_4_26B_A4B_it glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b uitars_7b gemini_3_1 qwen3_5_27b claude_opus_4_6)
DATASET="${DATASET:-2wikimultihop}"
# DATASET="${DATASET:-webgames}"

run_loo() {
    local held_out="$1"
    local tag="${DATASET}_open_set/open_set_loo_${held_out}"
    if [[ -n "$FILTER" && "$held_out" != *"$FILTER"* ]]; then
        return
    fi
    # Build the list of known agents (all except the held-out one)
    local known_agents=()
    for agent in "${ALL_AGENTS[@]}"; do
        [[ "$agent" != "$held_out" ]] && known_agents+=("$agent")
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Open-set LOO: held-out = $held_out"
    echo "  Known agents: ${known_agents[*]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "$PYTHON" trace_analyzer.py \
        --traces-dir "$TRACES_DIR" \
        --tag        "$tag" \
        --train-datasets "$DATASET" \
        --agents     "${known_agents[@]}" \
        --open-set-agents "$held_out"
    echo ""
    echo "  Saved → $TRACES_DIR/models/${DATASET}_open_set/open_set_loo_${held_out}/results.json"
    echo ""
}

for held_out in "${ALL_AGENTS[@]}"; do
    run_loo "$held_out"
done

echo "All open-set LOO experiments complete."
echo "Run: python plot_open_set.py  to generate visualisations."



# run_open_set.sh — leave-one-out loop over all 6 agents. For each held-out agent, trains on the other 5 and runs open-set eval:


# bash run_open_set.sh              # all 6 LOO experiments
# bash run_open_set.sh gpt_5_4     # single agent
# DATASET=frames bash run_open_set.sh
# plot_open_set.py — three figures saved to traces/models/open_set/:

# open_set_auroc_summary.png — grouped bars (AUROC per held-out agent × classifier)
# open_set_fpr95_summary.png — FPR95 bars (lower = better detection)
# open_set_roc_curves.png — 6-subplot ROC grid, one per held-out agent, FPR95 operating point marked