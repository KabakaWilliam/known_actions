#!/usr/bin/env bash
# plot_openset_bar_chart.sh — XGBoost open-set AUROC strip chart across datasets.
#
# Each point is a held-out agent (unique marker + colour); IQR box overlaid.
# Saves to src/figures/ by default.
#
# Usage:
#   bash plot_openset_bar_chart.sh                        # PNG, all 4 single-dataset dirs
#   bash plot_openset_bar_chart.sh --format pdf           # PDF output
#   bash plot_openset_bar_chart.sh --out /tmp/fig.png     # custom path
#   bash plot_openset_bar_chart.sh \
#       --loo-subdirs 2wikimultihop_open_set frames_open_set   # subset of datasets

set -euo pipefail
cd "$(dirname "$0")"

TRACES_DIR=./traces

python plot_open_set_summary.py \
    --traces-dir "$TRACES_DIR" \
    --plot xgb_strip \
    "$@"
