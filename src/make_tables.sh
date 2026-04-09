#!/usr/bin/env bash
# make_tables.sh — generate LaTeX table(s) from trace_analyzer results
#
# Usage:
#   bash make_tables.sh              # run all uncommented table configs
#   bash make_tables.sh all          # run only configs whose tag matches 'all'
#
# Uncomment table blocks below to activate them.

set -euo pipefail
cd "$(dirname "$0")"  # always run from src/

PYTHON=/opt/anaconda/envs/dispatch/bin/python
TRACES_DIR=./traces
FILTER="${1:-}"

WIKI_TAG="${WIKI_TAG:-wiki_ood_amazon_deep}"
AMAZON_TAG="${AMAZON_TAG:-webshop_ood_deepshop_wiki}"

make_table() {
    local tag="$1"; shift
    if [[ -z "$FILTER" || "$tag" == *"$FILTER"* ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Table: $tag  (wiki=$WIKI_TAG  amazon=$AMAZON_TAG)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        "$PYTHON" make_tables.py "$TRACES_DIR" \
            --wiki-tag "$WIKI_TAG" \
            --amazon-tag "$AMAZON_TAG" \
            "$@"
        # Rename output so multiple runs don't overwrite each other
        local src="$TRACES_DIR/models/table_main.tex"
        local dst="$TRACES_DIR/models/table_${tag}.tex"
        [[ "$src" != "$dst" ]] && mv "$src" "$dst"
        echo ""
        echo "  Saved → $dst"
    fi
}

# ── 6-model mix used in paper ─────────────────────────────────────────────────
make_table paper_6 \
    --agents gpt_5_4 claude_opus_4_6 gemini_3_1 qwen3vl_30b_a3b uitars_7b glm_4.6v_flash

# ── All agents with results ────────────────────────────────────────────────────
# make_table all

# ── Proprietary only ──────────────────────────────────────────────────────────
# make_table proprietary \
#     --agents gpt_5_4 claude_opus_4_6 gemini_3_1

# ── Open-source only ──────────────────────────────────────────────────────────
# make_table open_source \
#     --agents qwen3vl_8b uitars_7b glm_4.6v_flash

# ── Small models only ─────────────────────────────────────────────────────────
# make_table small_models \
#     --agents qwen3vl_8b uitars_7b glm_4.6v_flash qwen25vl_7b

echo ""
echo "All tables complete."
