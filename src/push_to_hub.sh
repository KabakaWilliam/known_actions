#!/usr/bin/env bash
# push_to_hub.sh — reverify all traces then upload to HuggingFace
#
# Usage:
#   bash push_to_hub.sh               # reverify + upload (private by default)
#   bash push_to_hub.sh --dry-run     # reverify + show what would be uploaded
#   bash push_to_hub.sh --public      # upload and make the repo public
#   bash push_to_hub.sh --private     # upload and keep the repo private (default)
#
# Requires HF_TOKEN to be set in the environment or in a .env file at the repo root.

set -euo pipefail
cd "$(dirname "$0")"

HF_REPO="CoffeeGitta/known-actions-traces"
TRACES_DIR="./traces"


# Load .env from repo root if present (one directory up from src/)
if [[ -f ".env" ]]; then
    set -o allexport
    source ".env"
    set +o allexport
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it or add HF_TOKEN=hf_xxx to .env"
    exit 1
fi

DRY_RUN=""
VISIBILITY="--public"
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN="--dry-run"
    [[ "$arg" == "--public"  ]] && VISIBILITY="--public"
    [[ "$arg" == "--private" ]] && VISIBILITY="--private"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 1: Re-run verification on all traces"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -n "$DRY_RUN" ]]; then
    python reverify.py --traces-dir "$TRACES_DIR"
else
    python reverify.py --traces-dir "$TRACES_DIR" --apply
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2: Upload traces to HuggingFace ($HF_REPO)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python upload_to_hub.py \
    --traces-dir "$TRACES_DIR" \
    --repo-id    "$HF_REPO" \
    --token      "$HF_TOKEN" \
    $VISIBILITY \
    $DRY_RUN
