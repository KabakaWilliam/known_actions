#!/usr/bin/env bash
# eval_universal_wiki_frames.sh — evaluate the pre-trained universal_wiki_frames
# XGBoost model on each constituent dataset separately (no retraining).
#
# Outputs:
#   traces/classifiers/eval_universal_wiki_frames_on_wiki/results.json
#   traces/classifiers/eval_universal_wiki_frames_on_frames/results.json

set -euo pipefail
cd "$(dirname "$0")"

TRACES_DIR=./traces
PYTHON=/opt/anaconda/envs/dispatch/bin/python
LOAD_CLF="$TRACES_DIR/classifiers/universal_wiki_frames"

AGENTS_ALL="gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash seed_2_lite"

run_eval() {
    local tag="$1"; shift
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Eval: $tag"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    $PYTHON trace_analyzer.py \
        --traces-dir "$TRACES_DIR" \
        --tag "$tag" \
        --load-classifier "$LOAD_CLF" \
        "$@"
    echo ""
    echo "  Saved → $TRACES_DIR/classifiers/$tag/"
}

# 2WikiMultiHopQA: natural _train/_val/_test splits → test split is clean held-out data
run_eval eval_universal_wiki_frames_on_wiki \
    --train-datasets 2wikimultihop \
    --agents $AGENTS_ALL

# FRAMES: no natural splits → resplit with same seed=42 and cap=300 as training run
run_eval eval_universal_wiki_frames_on_frames \
    --train-datasets frames \
    --resplit-datasets frames \
    --resplit-n-per-agent 300 \
    --agents $AGENTS_ALL

echo "Done. Verify with:"
echo "  python3 -c \""
echo "  import json"
echo "  for tag in ['eval_universal_wiki_frames_on_wiki', 'eval_universal_wiki_frames_on_frames']:"
echo "      r = json.load(open(f'traces/classifiers/{tag}/results.json'))"
echo "      macro = r['models']['XGBoost']['test_report']['macro avg']['f1-score']"
echo "      print(f'{tag}: macro F1 = {macro:.3f}')"
echo "  \""
