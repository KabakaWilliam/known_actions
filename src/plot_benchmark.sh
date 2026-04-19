#!/usr/bin/env bash
# plot_benchmark.sh — generate the per-agent task accuracy heatmap
#
# Usage:
#   bash plot_benchmark.sh
#   bash plot_benchmark.sh --split test
#   bash plot_benchmark.sh --out figures/benchmark.pdf

set -euo pipefail
cd "$(dirname "$0")"


TRACES_DIR="./traces"
OUT="./figures/benchmark.png"

python plot_benchmark.py \
    --traces-dir "$TRACES_DIR" \
    --out        "$OUT" \
    "$@"
