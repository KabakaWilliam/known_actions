#!/usr/bin/env bash
# train_universal_classifiers.sh — multi-dataset universal classifiers
#
# Each experiment trains on 3–4 datasets and tests OOD on the remainder.
#
# Split sources:
#   2wikimultihop  — existing _train/_val/_test dirs (~150/75/75 per agent)
#   webshop        — existing _train/_val/_test dirs (~150/75/75 per agent)
#   webgames       — existing _train/_val/_test dirs (~50/25/25 per agent)
#   frames         — only frames_test exists; --resplit-datasets frames pools
#                    all traces and re-stratifies 50/25/25.
#                    --resplit-n-per-agent 300 → ~150/75/75 per agent.
#   deepshop       — only deepshop_ood exists; --resplit-datasets deepshop
#                    pools and re-stratifies. Natural cap ~150/agent so
#                    --resplit-n-per-agent 300 passes through at ~75/37/37.
#
# OOD evaluation:
#   ALL traces for any dataset in --ood-datasets go to the OOD bucket,
#   regardless of directory suffix.  frames OOD → full ~800+ traces.
#   deepshop OOD → full ~150 traces.
#
# seed_2_lite has no webgames traces.
#   → Excluded (AGENTS_NO_SEED) when webgames is a training dataset
#     (its class would have zero training examples from that split).
#   → Included (AGENTS_ALL) when webgames is OOD-only; it simply shows
#     "--" in the webgames OOD column (support = 0).
#
# Usage:
#   bash train_universal_classifiers.sh              # run all experiments
#   bash train_universal_classifiers.sh wiki_ws      # filter by tag substring

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
        echo "  Universal experiment: $tag"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python trace_analyzer.py --traces-dir "$TRACES_DIR" --tag "$tag" "$@"
        echo ""
        echo "  Saved → $TRACES_DIR/classifiers/$tag/"
    fi
}

AGENTS_ALL="gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash seed_2_lite"
AGENTS_NO_SEED="gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash"

# ══════════════════════════════════════════════════════════════════════════════
# 3-DATASET COMBOS — each leaves 2 datasets as OOD
# ══════════════════════════════════════════════════════════════════════════════

# Train: wiki + webshop + frames  →  OOD: deepshop, webgames
# seed_2_lite included (webgames is OOD-only; seed shows -- in that column)
# run_experiment universal_wiki_ws_frames \
#     --train-datasets 2wikimultihop webshop frames \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --ood-datasets deepshop webgames \
#     --agents $AGENTS_ALL

# # Train: wiki + webshop + webgames  →  OOD: frames, deepshop
# # seed_2_lite excluded (webgames in training; seed has no webgames traces)
# run_experiment universal_wiki_ws_webgames \
#     --train-datasets 2wikimultihop webshop webgames \
#     --ood-datasets frames deepshop \
#     --agents $AGENTS_ALL

# # Train: wiki + frames + webgames  →  OOD: webshop, deepshop
# # seed_2_lite excluded (webgames in training)
# run_experiment universal_wiki_frames_webgames \
#     --train-datasets 2wikimultihop frames webgames \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --ood-datasets webshop deepshop \
#     --agents $AGENTS_ALL

# # Train: webshop + frames + webgames  →  OOD: wiki, deepshop
# # seed_2_lite excluded (webgames in training)
# run_experiment universal_ws_frames_webgames \
#     --train-datasets webshop frames webgames \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --ood-datasets 2wikimultihop deepshop \
#     --agents $AGENTS_ALL

# # ══════════════════════════════════════════════════════════════════════════════
# # 4-DATASET COMBO — maximum coverage, single OOD holdout
# # ══════════════════════════════════════════════════════════════════════════════

# # Train: wiki + webshop + frames + deepshop  →  OOD: webgames
# # deepshop resplit alongside frames; deepshop naturally caps at ~150/agent
# # (below the 300 ceiling) → contributes ~75/37/37 vs frames' ~150/75/75.
# # seed_2_lite included (webgames is OOD-only)
# run_experiment universal_4ds_ood_webgames \
#     --train-datasets 2wikimultihop webshop frames deepshop \
#     --resplit-datasets frames deepshop \
#     --resplit-n-per-agent 300 \
#     --ood-datasets webgames \
#     --agents $AGENTS_ALL

# # Train: wiki + webshop + frames + webgames  →  OOD: deepshop
# # seed_2_lite excluded (webgames in training)
run_experiment universal_4ds_ood_deepshop \
    --train-datasets 2wikimultihop webshop frames webgames \
    --resplit-datasets frames \
    --resplit-n-per-agent 300 \
    --ood-datasets deepshop \
    --agents $AGENTS_ALL


# Train: wiki + deepshop + webshop + webgames  →  OOD: deepshop
# seed_2_lite excluded (webgames in training)
# run_experiment universal_4ds_ood_frames \
#     --train-datasets 2wikimultihop webshop deepshop webgames \
#     --ood-datasets frames \
#     --agents $AGENTS_ALL

echo ""
echo "All universal experiments complete."
