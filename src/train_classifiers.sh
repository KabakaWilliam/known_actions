#!/usr/bin/env bash
# train_classifiers.sh — run trace_analyzer.py for common experiment configurations
#
# Usage:
#   bash train_classifiers.sh              # run all uncommented experiments
#   bash train_classifiers.sh wiki         # run only the 'wiki' experiment (matches tag)
#
# Uncomment experiment blocks below to activate them.

set -euo pipefail
trap 'kill 0' EXIT  # kill all child processes (incl. orphaned joblib workers) on exit
cd "$(dirname "$0")"  # always run from src/

PYTHON=/opt/anaconda/envs/dispatch/bin/python
TRACES_DIR=./traces
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # override: CUDA_VISIBLE_DEVICES=2 bash train_classifiers.sh
export CUDA_VISIBLE_DEVICES
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/VData/linna4335/.apptainer_cache}"
export APPTAINER_CACHEDIR
FILTER="${1:-}"  # optional positional arg: only run experiments whose tag matches

run_experiment() {
    local tag="$1"; shift
    if [[ -z "$FILTER" || "$tag" == *"$FILTER"* ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Experiment: $tag"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        "$PYTHON" trace_analyzer.py --traces-dir "$TRACES_DIR" --tag "$tag" "$@"
        echo ""
        echo "  Saved → $TRACES_DIR/models/$tag/"
        echo "           ├── classifier.pkl"
        echo "           ├── lstm_model.pt"
        echo "           └── results.json"
    fi
}

# ── Wikipedia in-domain ────────────────────────────────────────────────────────
# run_experiment wiki \
#     --train-datasets 2wikimultihop

# # ── Wikipedia train → Amazon OOD ──────────────────────────────────────────────
# run_experiment wiki_ood_amazon_deep \
#     --train-datasets 2wikimultihop \
#     --ood-datasets webshop deepshop

# # ── Amazon in-domain ──────────────────────────────────────────────────────────
# run_experiment webshop \
#     --train-datasets webshop

# ── Amazon train → Wiki + DeepShop OOD ───────────────────────────────────────────────
# run_experiment webshop_ood_deepshop_wiki \
#     --train-datasets webshop \
#     --ood-datasets deepshop 2wikimultihop

# ── Amazon train → Wikipedia OOD ──────────────────────────────────────────────
# run_experiment amazon_ood_wiki \
#     --train-datasets webshop \
#     --ood-datasets 2wikimultihop

# ── Agent subset examples (uncomment and edit as needed)────────────────────── ##qwen3vl_8b
# run_experiment wiki_no_uitars \
#     --train-datasets 2wikimultihop \
#     --agents gpt_5_4 qwen3vl_30b_a3b

# run_experiment wiki_small_models\
#     --train-datasets 2wikimultihop \
#     --agents qwen3vl_8b uitars_7b

# run_experiment wiki_gpt_5_4_only \
#     --train-datasets 2wikimultihop \
#     --agents gpt_5_4

# # ── WebGames in-domain ────────────────────────────────────────────────────────
# run_experiment webgames \
#     --train-datasets webgames

run_experiment webgames_all_ood \
    --train-datasets webgames \
    --ood-datasets 2wikimultihop webshop deepshop \
    --agents gpt_5_4 claude_opus_4_6 gemini_3_1 glm_4.6v_flash uitars_7b qwen3vl_8b
    # --agents gpt_5_4 claude_opus_4_6 gemini_3_1 glm_4.6v_flash qwen3vl_30b_a3b uitars_7b


# # ── WebGames in-domain proprietery only ────────────────────────────────────────────────────────
# run_experiment webgames_proprietary \
#     --train-datasets webgames \
#     --agents gpt_5_4 claude_opus_4_6 gemini_3_1 \
#     --ood-datasets 2wikimultihop webshop deepshop

# ── WebGames train → 2wikimultihop + webshop OOD ─────────────────────────────
# run_experiment webgames_ood_wiki_webshop \
#     --train-datasets webgames \
#     --ood-datasets 2wikimultihop webshop

# ── 2wikimultihop + webshop train → WebGames OOD ─────────────────────────────
# run_experiment wiki_webshop_ood_webgames \
#     --train-datasets 2wikimultihop webshop \
#     --ood-datasets webgames

echo ""
echo "All experiments complete."



# Possible extra featuers to add:
# time to first action
# frequency between keypresses (typing speed)
# lag between mouse movement
# number of mouse movements
# number of clicks
# area of screen covered by mouse movements
# time spent on each page (if applicable)