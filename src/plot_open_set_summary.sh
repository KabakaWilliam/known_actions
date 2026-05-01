#!/usr/bin/env bash
# Produce the 2-panel open-set classification summary figure.
#
# Panel A: Per-agent AUROC (RF + XGB) from leave-one-out experiments on 2WikiMultihopQA.
# Panel B: Closed-set Macro F1 vs open-set AUROC scatter.
#
# Usage:
#   bash plot_open_set_summary.sh
#   bash plot_open_set_summary.sh --out my_fig.png

set -euo pipefail
cd "$(dirname "$0")"

TRACES_DIR=./traces
# CLASSIFIER_DIR="2wikimultihop_open_set"
CLASSIFIER_DIR="wiki_frames_open_set"
# CLASSIFIER_DIR="ws_deepshop_open_set"

python plot_open_set_summary.py \
    --traces-dir "$TRACES_DIR" \
    --plot scatter \
    --loo-subdir "$CLASSIFIER_DIR" \
    "$@"

python plot_open_set_summary.py \
    --traces-dir "$TRACES_DIR" \
    --plot cross_dataset \
    "$@"
