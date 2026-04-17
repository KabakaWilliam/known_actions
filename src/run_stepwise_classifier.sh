#!/usr/bin/env bash
# run_stepwise_classifier.sh — Wiki↔frames experiments with stepwise prefix evaluation.
#
# Trains on full traces as normal, then evaluates at truncated prefixes
# (first N DOM events and first T ms) to answer: how quickly can each
# agent be identified from a partial trace?
#
# Overwrites results.json in-place (adds 'prefix_curve' key alongside
# existing test/OOD reports).
#
# Usage:
#   bash run_stepwise_classifier.sh              # both experiments
#   bash run_stepwise_classifier.sh wiki         # only wiki_2_frames
#   bash run_stepwise_classifier.sh frames       # only frames_2_wiki

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

run_experiment() {
    local tag="$1"; shift
    if [[ -z "$FILTER" || "$tag" == *"$FILTER"* ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Early-ID experiment: $tag"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        "$PYTHON" trace_analyzer.py --traces-dir "$TRACES_DIR" --tag "$tag" --prefix-eval "$@"
        echo ""
        echo "  Saved → $TRACES_DIR/models/$tag/results.json"
        echo ""
    fi
}

# ── Easy → Hard: train on 2wikimultihop, OOD test on FRAMES ──────────────────
run_experiment wiki_2_frames_early \
    --train-datasets 2wikimultihop \
    --ood-datasets frames \
    --agents gpt_5_4 gemma_4_26B_A4B_it glm_4.6v_flash qwen3_5_27b qwen3vl_8b qwen3vl_30b_a3b uitars_7b gemini_3_1

# ── Hard → Easy: train on FRAMES, OOD test on 2wikimultihop ──────────────────
# frames only has a _test split, so --resplit-datasets re-splits 50/25/25 by agent.
# --resplit-n-per-agent 300 matches 2wiki's ~150/75/75 per-class budget.
run_experiment frames_2_wiki_early \
    --train-datasets frames \
    --resplit-datasets frames \
    --resplit-n-per-agent 300 \
    --ood-datasets 2wikimultihop \
    --agents gpt_5_4 gemma_4_26B_A4B_it glm_4.6v_flash qwen3_5_27b qwen3vl_8b qwen3vl_30b_a3b uitars_7b gemini_3_1

echo "All stepwise-classifier experiments complete."
echo "Run: python plot_early_id.py  to generate visualisations."

# python plot_early_id.py                              # n_events (default), both experiments
# python plot_early_id.py --mode t_ms                  # time-based axis
# python plot_early_id.py --tags wiki_2_frames_early   # single experiment
