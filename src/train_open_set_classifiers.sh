#!/usr/bin/env bash
# train_open_set_classifiers.sh — Leave-one-agent-out open-set recognition experiments.
#
# Trains classifiers on N-1 agents and evaluates whether the held-out agent's
# traces can be detected as "unknown" (AUROC, FPR95).
#
# Usage:
#   bash train_open_set_classifiers.sh                          # all agents, 2wikimultihop
#   bash train_open_set_classifiers.sh gpt_5_4                 # single held-out agent
#   DATASET=frames bash train_open_set_classifiers.sh          # FRAMES dataset
#   DATASET=webshop bash train_open_set_classifiers.sh         # WebShop dataset
#   DATASET=deepshop bash train_open_set_classifiers.sh        # DeepShop dataset
#   CLASSIFIERS="RandomForest XGBoost" DATASET=frames \
#       bash train_open_set_classifiers.sh                     # RF + XGB only
#
# Resplit behaviour (mirrors train_open_set_universal.sh):
#   frames   — resplit 50/25/25, cap 300/agent
#   deepshop — resplit 50/25/25, no cap
#   others   — use existing train/val/test splits

set -euo pipefail
trap 'kill 0' EXIT
cd "$(dirname "$0")"

TRACES_DIR=./traces
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/VData/linna4335/.apptainer_cache}"
export APPTAINER_CACHEDIR
FILTER="${1:-}"

ALL_AGENTS=(gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash seed_2_lite)
DATASET="${DATASET:-2wikimultihop}"

# Optional: space-separated list of classifiers to train.
# Leave unset (or empty) to train all classifiers.
# Example: CLASSIFIERS="RandomForest XGBoost"
CLASSIFIERS="${CLASSIFIERS:-}"

# Per-dataset resplit args (frames/deepshop have no natural train/val/test splits)
case "$DATASET" in
    frames)
        RESPLIT_ARGS=(--resplit-datasets frames --resplit-n-per-agent 300)
        ;;
    deepshop)
        RESPLIT_ARGS=(--resplit-datasets deepshop)
        ;;
    *)
        RESPLIT_ARGS=()
        ;;
esac

run_loo() {
    local held_out="$1"
    local tag="${DATASET}_open_set/open_set_loo_${held_out}"

    if [[ -n "$FILTER" && "$held_out" != *"$FILTER"* ]]; then
        return
    fi

    local known_agents=()
    for agent in "${ALL_AGENTS[@]}"; do
        [[ "$agent" != "$held_out" ]] && known_agents+=("$agent")
    done

    local clf_args=()
    [[ -n "$CLASSIFIERS" ]] && clf_args=(--classifiers $CLASSIFIERS)

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Dataset    : $DATASET"
    echo "  Held-out   : $held_out"
    [[ -n "$CLASSIFIERS" ]] && echo "  Classifiers: $CLASSIFIERS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python trace_analyzer.py \
        --traces-dir     "$TRACES_DIR" \
        --tag            "$tag" \
        --train-datasets "$DATASET" \
        "${RESPLIT_ARGS[@]}" \
        --agents         "${known_agents[@]}" \
        --open-set-agents "$held_out" \
        "${clf_args[@]}"

    echo ""
    echo "  Saved → $TRACES_DIR/classifiers/$tag/results.json"
    echo ""
}

for held_out in "${ALL_AGENTS[@]}"; do
    run_loo "$held_out"
done

echo "All open-set LOO experiments complete."
echo "Plot: python plot_open_set_summary.py --loo-subdir ${DATASET}_open_set"
