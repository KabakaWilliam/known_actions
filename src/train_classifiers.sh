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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # override: CUDA_VISIBLE_DEVICES=2 bash train_classifiers.sh
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
        echo "  Saved → $TRACES_DIR/classifiers/$tag/"
        echo "           ├── classifier.pkl"
        echo "           ├── lstm_model.pt"
        echo "           └── results.json"
    fi
}

# ── Wikipedia in-domain ────────────────────────────────────────────────────────
# run_experiment wiki \
#     --train-datasets 2wikimultihop
#     --agents gpt_5_4 claude_opus_4_6 gemini_3_1 glm_4.6v_flash uitars_7b qwen3vl_8b

# # ── Wikipedia train → Amazon OOD ──────────────────────────────────────────────
# run_experiment wiki_ood_amazon_deep \
#     --train-datasets 2wikimultihop \
#     --ood-datasets webshop deepshop

# ── Wikipedia in-domain_selective ────────────────────────────────────────────────────────
# run_experiment wiki_2_frames \
#     --train-datasets 2wikimultihop \
#     --ood-datasets frames \
#     --agents gpt_5_4 gemma_4_26B_A4B_it gemini_3_1 glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b uitars_7b qwen3_5_27b

# ── FRAMES (hard) → 2wikimultihop OOD (easy) ─────────────────────────────────
# Hypothesis: harder compositional tasks elicit more reliable agent fingerprints
# that generalise to simpler same-site tasks.
# --resplit-datasets: frames only has a _test dir, so pool all traces and split
# --resplit-n-per-agent 300: cap to 150/75/75 per agent to match 2wiki's budget
# run_experiment frames_2_wiki \
#     --train-datasets frames \
#     --resplit-datasets frames \
#     --resplit-n-per-agent 300 \
#     --ood-datasets 2wikimultihop \
#     --agents gpt_5_4 gemma_4_26B_A4B_it gemini_3_1 glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b uitars_7b qwen3_5_27b

# # ── Amazon in-domain ──────────────────────────────────────────────────────────
# run_experiment webshop \
#     --train-datasets webshop

# ── Amazon train → Wiki + DeepShop OOD ───────────────────────────────────────────────
# run_experiment webshop_ood_deepshop_wiki \
#     --train-datasets webshop \
#     --ood-datasets deepshop 2wikimultihop

# ── Webshop train → DeepShop OOD ───────────────────────────────────────────────
# run_experiment webshop_ood_deepshop \
#     --train-datasets webshop \
#     --ood-datasets deepshop \
#     --agents gpt_5_4 claude_opus_4_6 gemma_4_26B_A4B_it glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b uitars_7b gemini_3_1 qwen3_5_27b

# ── DeepShop train → Webshop OOD ───────────────────────────────────────────────
# deepshop only has a _ood split, so --resplit-datasets pools all traces and
# stratified-splits 50/25/25 by agent. --resplit-n-per-agent 150 → ~150/75/75
# per agent, matching 2wiki's budget.
# Note: other experiments that use --ood-datasets deepshop are unaffected —
# --resplit-datasets is only applied when explicitly passed.
run_experiment deepshop_2_webshop \
    --train-datasets deepshop \
    --resplit-datasets deepshop \
    --resplit-n-per-agent 150 \
    --ood-datasets webshop \
    --agents gpt_5_4 claude_opus_4_6 gemma_4_26B_A4B_it glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b uitars_7b gemini_3_1 qwen3_5_27b



# ── Amazon train → Wikipedia OOD ──────────────────────────────────────────────
# run_experiment amazon_ood_wiki \
#     --train-datasets webshop \
#     --ood-datasets 2wikimultihop

# ── Family-level classification (model family as label, not individual checkpoint) ──
# run_experiment wiki_family \
#     --train-datasets 2wikimultihop \
#     --label-by family \
#     --agents gpt_5_4 claude_opus_4_6 gemma_4_26B_A4B_it glm_4.6v_flash \
#              qwen3vl_8b qwen3vl_30b_a3b uitars_7b gemini_3_1 qwen3_5_27b qwen3_5_9b

# run_experiment webshop_family \
#     --train-datasets webshop \
#     --ood-datasets deepshop \
#     --label-by family \
#     --agents gpt_5_4 claude_opus_4_6 gemma_4_26B_A4B_it glm_4.6v_flash \
#              qwen3vl_8b qwen3vl_30b_a3b uitars_7b gemini_3_1 qwen3_5_27b qwen3_5_9b

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

# run_experiment webgames_all_ood \
#     --train-datasets webgames \
#     --ood-datasets 2wikimultihop webshop deepshop \


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


# bash train_classifiers.sh wiki_family