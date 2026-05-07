#!/usr/bin/env bash
# identification_speed.sh — Early identification: macro F1 vs. DOM events observed at test time.
#
# Trains classifiers on full traces (or loads pre-trained ones), then runs
# the prefix-curve sweep to measure how F1 grows as more DOM events are revealed.
#
# Results saved to: traces/classifiers/identification_speed/{tag}/results.json
#
# Usage:
#   bash identification_speed.sh                                 # train RF+XGBoost, all 4 directions
#   bash identification_speed.sh --load-dir traces/classifiers   # load pre-trained classifiers
#   bash identification_speed.sh --classifiers "XGBoost RandomForest"
#   bash identification_speed.sh wiki                            # only wiki/frames directions
#   bash identification_speed.sh webshop                         # only webshop/deepshop directions
#
# --load-dir DIR:
#   Expects DIR/{tag}/classifier.pkl for each direction (e.g. DIR/wiki_2_frames/classifier.pkl).
#   Falls back to training from scratch if a bundle is missing.

set -euo pipefail
trap 'kill 0' EXIT
cd "$(dirname "$0")"

TRACES_DIR=./traces
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES

LOAD_DIR=""
CLASSIFIERS="XGBoost RandomForest"
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --load-dir)    LOAD_DIR="$2";    shift 2 ;;
        --classifiers) CLASSIFIERS="$2"; shift 2 ;;
        *)             FILTER="$1";      shift ;;
    esac
done

# AGENTS=(gpt_5_4 gemma_4_26B_A4B_it glm_4.6v_flash qwen3_5_27b qwen3vl_8b qwen3vl_30b_a3b uitars_7b gemini_3_1)
AGENTS=(gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash seed_2_lite)

run_speed() {
    local base_tag="$1"; shift

    if [[ -n "$FILTER" && "$base_tag" != *"$FILTER"* ]]; then
        return
    fi

    # Parse --load-dir from per-call args so it doesn't get forwarded to trace_analyzer.py.
    # A per-direction --load-dir overrides the global LOAD_DIR for this run only.
    local per_dir_load=""
    local extra=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --load-dir) per_dir_load="$2"; shift 2 ;;
            *)          extra+=("$1");     shift ;;
        esac
    done
    local effective_load="${per_dir_load:-$LOAD_DIR}"
    # Resolve bare names (e.g. "wiki_ood_all") against TRACES_DIR/classifiers/
    if [[ -n "$effective_load" && ! -d "$effective_load" ]]; then
        effective_load="${TRACES_DIR}/classifiers/${effective_load}"
    fi

    local out_tag="identification_speed/${base_tag}"
    local load_args=()

    if [[ -n "$effective_load" ]]; then
        if [[ -f "${effective_load}/classifier.pkl" ]]; then
            load_args=(--load-classifier "$effective_load")
        elif [[ -f "${effective_load}/${base_tag}/classifier.pkl" ]]; then
            load_args=(--load-classifier "${effective_load}/${base_tag}")
        else
            echo "  [WARN] No classifier.pkl at ${effective_load} or ${effective_load}/${base_tag} — training from scratch."
        fi
    fi

    # Split CLASSIFIERS string into separate args
    read -ra clf_arr <<< "$CLASSIFIERS"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment : $base_tag   classifiers=${CLASSIFIERS}"
    [[ ${#load_args[@]} -gt 0 ]] && echo "  Loading from: ${load_args[1]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python trace_analyzer.py \
        --traces-dir "$TRACES_DIR" \
        --tag        "$out_tag" \
        "${extra[@]}" \
        "${load_args[@]}" \
        --classifiers "${clf_arr[@]}" \
        --prefix-eval \
        --agents "${AGENTS[@]}"

    echo "  Saved → $TRACES_DIR/classifiers/$out_tag/results.json"
    echo ""
}

# ── wiki → frames ──────────────────────────────────────────────────────────────
# run_speed wiki_2_frames \
#     --train-datasets 2wikimultihop \
#     --load-dir wiki_ood_all \
#     --ood-datasets   frames

# ── frames → wiki ──────────────────────────────────────────────────────────────
# run_speed frames_2_wiki \
#     --train-datasets   frames \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --load-dir frames_ood_all \
#     --ood-datasets     2wikimultihop

# ── webshop → deepshop ─────────────────────────────────────────────────────────
run_speed webshop_2_deepshop \
    --train-datasets webshop \
    --load-dir       webshop_ood_all \
    --ood-datasets   deepshop

# ── deepshop → webshop ─────────────────────────────────────────────────────────
run_speed deepshop_2_webshop \
    --train-datasets   deepshop \
    --load-dir deepshop_ood_all \
    --resplit-datasets deepshop \
    --resplit-n-per-agent 150 \
    --ood-datasets     webshop

echo "All identification-speed experiments complete."
echo "Plot: python plot_identification_speed.py"
echo "      python plot_identification_speed.py --tags webshop_2_deepshop deepshop_2_webshop"
