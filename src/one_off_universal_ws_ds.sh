#!/usr/bin/env bash
# one_off_universal_ws_ds.sh
#
# 1. Train XGBoost-only universal classifier on WebShop + DeepShop (with proper
#    resplit for DeepShop, which has no natural _train/_val/_test dirs).
# 2. Evaluate the freshly trained model on each dataset individually.
#
# Outputs:
#   traces/classifiers/universal_ws_ds/              ← trained model
#   traces/classifiers/eval_universal_ws_ds_on_webshop/results.json
#   traces/classifiers/eval_universal_ws_ds_on_deepshop/results.json

set -euo pipefail
cd "$(dirname "$0")"

TRACES_DIR=./traces
PYTHON=/opt/anaconda/envs/dispatch/bin/python
TAG=universal_ws_ds

AGENTS_ALL="gpt_5_4 claude_opus_4_6 gemma-4-31B-it gemma_4_26B_A4B_it glm_4.6v glm_4.6v_flash qwen3vl_8b qwen3vl_30b_a3b qwen3_5_27b qwen3_5_9b uitars_7b gemini_3_1 gemini_3_flash seed_2_lite"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 1: Train universal_ws_ds (XGBoost only)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON trace_analyzer.py \
    --traces-dir "$TRACES_DIR" \
    --tag "$TAG" \
    --train-datasets webshop deepshop \
    --resplit-datasets deepshop \
    --resplit-n-per-agent 300 \
    --ood-datasets 2wikimultihop frames webgames \
    --agents $AGENTS_ALL \
    --classifiers XGBoost

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2: Eval on WebShop solo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON trace_analyzer.py \
    --traces-dir "$TRACES_DIR" \
    --tag eval_universal_ws_ds_on_webshop \
    --train-datasets webshop \
    --agents $AGENTS_ALL \
    --load-classifier "$TRACES_DIR/classifiers/$TAG" \
    --classifiers XGBoost

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 3: Eval on DeepShop solo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON trace_analyzer.py \
    --traces-dir "$TRACES_DIR" \
    --tag eval_universal_ws_ds_on_deepshop \
    --train-datasets deepshop \
    --resplit-datasets deepshop \
    --resplit-n-per-agent 300 \
    --agents $AGENTS_ALL \
    --load-classifier "$TRACES_DIR/classifiers/$TAG" \
    --classifiers XGBoost

echo ""
echo "All done. Verify with:"
echo "  python3 -c \""
echo "  import json"
echo "  for tag in ['eval_universal_ws_ds_on_webshop', 'eval_universal_ws_ds_on_deepshop']:"
echo "      r = json.load(open(f'traces/classifiers/{tag}/results.json'))"
echo "      macro = r['models']['XGBoost']['test_report']['macro avg']['f1-score']"
echo "      print(f'{tag}: macro F1 = {macro:.3f}')"
echo "  \""
