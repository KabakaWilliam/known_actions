#!/usr/bin/env bash
# Produce feature-importance × timing-attack degradation figures for XGBoost.
#
# Usage:
#   bash plot_importance_x_degradation.sh             # combined + individual panels
#   bash plot_importance_x_degradation.sh importance  # importance panel only
#   bash plot_importance_x_degradation.sh degradation # degradation panel only

set -euo pipefail
cd "$(dirname "$0")"

TRACES_DIR=./traces
PLOT="${1:-both}"
[[ $# -gt 0 ]] && shift
EXTRA_ARGS=("$@")

BASELINE_TAG="wiki_xgb_ood_frames"

# RED line — train and test both corrupted (poisoning attack)
POISON_TAGS=(
    wiki_delayed_xgb_500ms
    wiki_delayed_xgb_1000ms
    wiki_delayed_xgb_2000ms
    wiki_delayed_xgb_5000ms
)

# BLUE line — train clean, test corrupted (test-time evasion)
JITTER_TAGS=(
    wiki_jitter_test_500ms
    wiki_jitter_test_1000ms
    wiki_jitter_test_2000ms
    wiki_jitter_test_5000ms
)

python plot_importance_x_degradation.py \
    --baseline-tag "$BASELINE_TAG" \
    --poison-tags  "${POISON_TAGS[@]}" \
    --jitter-tags  "${JITTER_TAGS[@]}" \
    --traces-dir   "$TRACES_DIR" \
    --top-n        8 \
    --plot         "$PLOT" \
    "${EXTRA_ARGS[@]}"
