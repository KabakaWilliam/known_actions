#!/usr/bin/env bash
# Run the multi-seed open-set AUROC postprocessor with bounded CPU threading.
#
# Examples:
#   ./run_open_set_stats.sh
#   ./run_open_set_stats.sh --datasets wiki frames
#   PYTHON_BIN=python3 ./run_open_set_stats.sh --classifier-seeds 1 2 3 4 5

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

exec "$PYTHON_BIN" run_open_set_stats.py "$@"
