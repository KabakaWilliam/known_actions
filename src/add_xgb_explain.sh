#!/usr/bin/env bash
# Compute and append SHAP importances for XGBoost across all (or filtered) experiments.
#
# Usage:
#   bash add_xgb_explain.sh              # all experiments
#   bash add_xgb_explain.sh deepshop     # only tags containing "deepshop"
#   bash add_xgb_explain.sh "" --overwrite  # all, overwriting existing SHAP results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFIERS_DIR="$SCRIPT_DIR/traces/classifiers"
FILTER="${1:-}"
EXTRA_ARGS="${@:2}"

if [[ ! -d "$CLASSIFIERS_DIR" ]]; then
    echo "ERROR: classifiers directory not found: $CLASSIFIERS_DIR"
    exit 1
fi

found=0
for tag_dir in "$CLASSIFIERS_DIR"/*/; do
    tag="$(basename "$tag_dir")"
    [[ -n "$FILTER" && "$tag" != *"$FILTER"* ]] && continue
    [[ ! -f "$tag_dir/classifier.pkl" ]] && continue
    echo "=== $tag ==="
    python "$SCRIPT_DIR/add_xgb_explain.py" --tag "$tag" $EXTRA_ARGS || true
    echo ""
    found=$((found + 1))
done

if [[ $found -eq 0 ]]; then
    echo "No experiments found${FILTER:+ matching '$FILTER'}."
fi
