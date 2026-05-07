#!/usr/bin/env bash
set -euo pipefail


/opt/anaconda/envs/dispatch/bin/python plot_shap_comparison.py \
    --dir-a wiki_xgb_ood_frames \
    --dir-b wiki_delayed_xgb_5000ms \
    --label-a "original" \
    --label-b "delayed (5 s)" \
    --legend-title "2WikiMultiHopQA: original vs. delayed (5 s)" \
    "$@"