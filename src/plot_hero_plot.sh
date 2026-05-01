#!/usr/bin/env bash
# plot_hero_plot.sh — Agent identifiability bar chart (hero figure).
#
# Usage:
#   bash plot_hero_plot.sh                  # PNG
#   bash plot_hero_plot.sh --format pdf     # PDF
#   bash plot_hero_plot.sh --out fig.pdf --format pdf

set -euo pipefail
cd "$(dirname "$0")"

# # Identity, pick classifier automatically
# python plot_hero_plot.py \
#     --test-set-source wiki_ood_all webshop_ood_all frames_ood_all deepshop_ood_all

# # Family, explicit classifier
# python plot_hero_plot.py --mode family --classifier RandomForest \
#     --test-set-source wiki_family_ood_all webshop_family_ood_all \
#                       frames_family_ood_all deepshop_family_ood_all

# # Fewer than 4 panels also works (e.g. 2)
# python plot_hero_plot.py \
#     --test-set-source wiki_ood_all frames_ood_all

python3 plot_hero_plot.py \
    --test-set-source wiki_ood_all frames_ood_all deepshop_ood_all webshop_ood_all \
    "$@"
