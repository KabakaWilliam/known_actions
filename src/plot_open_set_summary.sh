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

python plot_open_set_summary.py \
    --traces-dir "$TRACES_DIR" \
    --plot scatter \
    "$@"
