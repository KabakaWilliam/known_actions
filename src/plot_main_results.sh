#!/usr/bin/env bash
# plot_main_results.sh — Closed-set Macro F1 + open-set AUROC main results figure.
#
# Usage:
#   bash plot_main_results.sh                          # PNG (default)
#   bash plot_main_results.sh --format pdf             # PDF
#   bash plot_main_results.sh --classifier RandomForest
#   bash plot_main_results.sh --out /tmp/fig.png
#   bash plot_main_results.sh --panel closed --format pdf
#   bash plot_main_results.sh --panel closed --format pdf

set -euo pipefail
cd "$(dirname "$0")"

# python3 plot_main_results.py "$@"
/opt/anaconda/envs/dispatch/bin/python plot_main_results.py "$@"
