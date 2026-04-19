#!/usr/bin/env bash
# plot_capability_vs_identifiability.sh — capability vs identifiability scatter
#
# Usage:
#   bash plot_capability_vs_identifiability.sh
#   bash plot_capability_vs_identifiability.sh --results-json traces/classifiers/wiki_ood_all/results.json
#   bash plot_capability_vs_identifiability.sh --out figures/cap_vs_id.pdf

set -euo pipefail
cd "$(dirname "$0")"


TRACES_DIR="./traces"
RESULTS_JSON="./traces/classifiers/wiki_ood_all/results.json"
OUT="./figures/capability_vs_identifiability.png"

python capability_vs_identifiability.py \
    --results-json "$RESULTS_JSON" \
    --traces-dir   "$TRACES_DIR" \
    --out          "$OUT" \
    "$@"
