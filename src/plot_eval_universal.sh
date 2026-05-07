#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda/envs/dispatch/bin/python plot_eval_universal.py "$@"
