#!/usr/bin/env bash
# train_classifiers.sh — run trace_analyzer.py for common experiment configurations
#
# Usage:
#   bash train_classifiers.sh              # run all uncommented experiments
#   bash train_classifiers.sh wiki         # run only the 'wiki' experiment (matches tag)
#
# Uncomment experiment blocks below to activate them.

set -euo pipefail
cd "$(dirname "$0")"  # always run from src/

PYTHON=/opt/anaconda/envs/dispatch/bin/python
TRACES_DIR=./traces
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # override: CUDA_VISIBLE_DEVICES=2 bash train_classifiers.sh
export CUDA_VISIBLE_DEVICES
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
run_experiment webshop_ood_deepshop_wiki \
    --train-datasets webshop \
    --ood-datasets deepshop 2wikimultihop

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

# ── WebGames in-domain ────────────────────────────────────────────────────────
# run_experiment webgames \
#     --train-datasets webgames

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