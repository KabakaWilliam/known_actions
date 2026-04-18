#!/usr/bin/env bash
# train_classifiers_family.sh — family-level classification experiments
#
# Labels collapse individual checkpoints to provider families:
#   gpt, claude, gemini, gemma, glm, qwen3vl, qwen35, uitars, seed
#
# Tags mirror train_classifiers_new.sh but with a _family suffix so results
# land in separate dirs (traces/models/wiki_family/, etc.) and don't overwrite
# checkpoint-level results.
#
# Usage:
#   bash train_classifiers_family.sh              # run all uncommented experiments
#   bash train_classifiers_family.sh wiki         # match tag filter

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
        echo "  Family experiment: $tag"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        "$PYTHON" trace_analyzer.py --traces-dir "$TRACES_DIR" --tag "$tag" --label-by family "$@"
        echo ""
        echo "  Saved → $TRACES_DIR/models/$tag/"
    fi
}

# All 14 agents with traces — both checkpoints per family where available,
# so family classifier has more training data per class.
# seed_2_lite excluded from webgames (no webgames traces).
AGENTS_ALL="gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash seed_2_lite"
AGENTS_NO_SEED="gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash"

# ══════════════════════════════════════════════════════════════════════════════
# 2WIKIMULTIHOP
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment wiki_family \
#     --train-datasets 2wikimultihop \
#     --agents $AGENTS_ALL

run_experiment wiki_family_ood_all \
    --train-datasets 2wikimultihop \
    --ood-datasets frames webshop deepshop  \
    --agents $AGENTS_ALL

# run_experiment wiki_family_ood_all \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames webshop deepshop webgames \
#     --agents $AGENTS_NO_SEED

# run_experiment wiki_family_x_frames \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents $AGENTS_ALL

# run_experiment wiki_family_x_webshop \
#     --train-datasets 2wikimultihop \
#     --ood-datasets webshop \
#     --agents $AGENTS_ALL

# run_experiment wiki_family_x_deepshop \
#     --train-datasets 2wikimultihop \
#     --ood-datasets deepshop \
#     --agents $AGENTS_ALL


# ══════════════════════════════════════════════════════════════════════════════
# FRAMES
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment frames_family \
#     --train-datasets frames \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --agents $AGENTS_ALL

run_experiment frames_family_ood_all \
    --train-datasets frames \
    --resplit-datasets frames \
    --resplit-n-per-agent 300 \
    --ood-datasets 2wikimultihop webshop deepshop  \
    --agents $AGENTS_ALL

# run_experiment frames_family_x_wiki \
#     --train-datasets frames \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --ood-datasets 2wikimultihop \
#     --agents $AGENTS_ALL


# ══════════════════════════════════════════════════════════════════════════════
# WEBSHOP
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment webshop_family \
#     --train-datasets webshop \
#     --agents $AGENTS_ALL

run_experiment webshop_family_ood_all \
    --train-datasets webshop \
    --ood-datasets 2wikimultihop frames deepshop  \
    --agents $AGENTS_ALL

# run_experiment webshop_family_x_deepshop \
#     --train-datasets webshop \
#     --ood-datasets deepshop \
#     --agents $AGENTS_ALL

# run_experiment webshop_family_x_wiki \
#     --train-datasets webshop \
#     --ood-datasets 2wikimultihop \
#     --agents $AGENTS_ALL


# ══════════════════════════════════════════════════════════════════════════════
# DEEPSHOP
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment deepshop_family \
#     --train-datasets deepshop \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --agents $AGENTS_ALL

run_experiment deepshop_family_ood_all \
    --train-datasets deepshop \
    --resplit-datasets deepshop \
    --resplit-n-per-agent 150 \
    --ood-datasets 2wikimultihop frames webshop  \
    --agents $AGENTS_ALL

# run_experiment deepshop_family_x_webshop \
#     --train-datasets deepshop \
#     --resplit-datasets deepshop \
#     --resplit-n-per-agent 150 \
#     --ood-datasets webshop \
#     --agents $AGENTS_ALL


# ══════════════════════════════════════════════════════════════════════════════
# WEBGAMES  (seed_2_lite excluded — no traces)
# ══════════════════════════════════════════════════════════════════════════════

# run_experiment webgames_family \
#     --train-datasets webgames \
#     --agents $AGENTS_NO_SEED

run_experiment webgames_family_ood_all \
    --train-datasets webgames \
    --ood-datasets 2wikimultihop frames webshop deepshop \
    --agents $AGENTS_NO_SEED


echo ""
echo "All family experiments complete."
