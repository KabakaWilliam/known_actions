#!/usr/bin/env bash
# make_tables.sh — generate LaTeX table(s) from trace_analyzer results
#
# Usage:


# python make_tables.py --in-domain-only
python make_tables.py --family --in-domain-only --indomain wiki frames webshop deepshop


# python make_tables.py --ood-only --ood-pairs wiki:frames frames:wiki  deepshop:webshop webshop:deepshop