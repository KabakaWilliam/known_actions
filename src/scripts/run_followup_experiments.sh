#!/usr/bin/env bash
#
# Human-readable launcher for reviewer follow-up experiments.
#
# Edit only SETTINGS: choose GPUs and comment/uncomment experiment names.
# Independent families run in parallel; commands within one family are serial.

set -Eeuo pipefail

# =============================================================================
# SETTINGS
# =============================================================================

MODEL_ARRIVAL_GPU=0
CLOSED_SET_SCALING_GPU=0
OPEN_SET_SCALING_GPU=0
INFERENCE_ENGINE_GPU=1
PREPARE_INPUTS=false

SELECTED_EXPERIMENTS=(
  # Simulate each of 14 MidScene models arriving as a new closed-set label.
  # Trains update curves with 1/2/5/10/20/50/100/all new labeled traces.
  # model_arrival_full

  # Grow the original MidScene closed set from 2 to 14 models using five
  # randomized nested model orders; report F1, fit time, and tree complexity.
  closed_set_scaling_full

  # Leave 1, 2, 3, or 4 of the original 14 MidScene models entirely out of
  # training/validation; identify known models and reject unseen ones.
  # open_set_scaling_full

  # Priority inference-engine result: train on vLLM, test SGLang, plus all
  # within-engine, reverse-transfer, and mixed-engine controls at seed 42.
  # inference_engine_full_seed42

  # Timing-only and non-timing versions of the complete engine matrix, seed 42.
  # inference_engine_ablation_seed42

  # Final five-seed confirmation of the full-feature inference-engine matrix.
  # Enable after reviewing seed 42; seed 42 is reused rather than rerun.
  # inference_engine_full_five_seeds
)

# =============================================================================
# IMPLEMENTATION
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${SRC_DIR}"

PYTHON_BIN="/opt/anaconda/envs/dispatch/bin/python"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_ROOT="artifacts/experiment_runs/followup_${RUN_STAMP}"
MPLCONFIGDIR="${SRC_DIR}/../.tmp/matplotlib"
export MPLCONFIGDIR
mkdir -p "${LOG_ROOT}"
mkdir -p "${MPLCONFIGDIR}"

ARRIVAL_CONFIG="experiments/model_arrival/configs/midscene_14model.yaml"
SCALING_CONFIG="experiments/closed_set_scaling/configs/midscene_14model_class_count.yaml"
OPEN_SET_CONFIG="experiments/open_set_scaling/configs/midscene_14model_leave_p_out.yaml"
ENGINE_CONFIG="experiments/inference_engine/configs/webshop_sglang_analysis.yaml"

selected() {
  local wanted="$1"
  local value
  for value in "${SELECTED_EXPERIMENTS[@]}"; do
    [[ "${value}" == "${wanted}" ]] && return 0
  done
  return 1
}

run_logged() {
  local name="$1"
  shift
  echo "[$(date -u +%FT%TZ)] START ${name}"
  echo "[$(date -u +%FT%TZ)] CMD   $*"
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
  echo "[$(date -u +%FT%TZ)] DONE  ${name}"
}

arrival_selected() {
  selected model_arrival_full
}

scaling_selected() {
  selected closed_set_scaling_full
}

open_set_selected() {
  selected open_set_scaling_full
}

engine_selected() {
  selected inference_engine_full_seed42 \
    || selected inference_engine_ablation_seed42 \
    || selected inference_engine_full_five_seeds
}

prepare_inputs() {
  if arrival_selected; then
    run_logged model_arrival_audit \
      "${PYTHON_BIN}" -m experiments.model_arrival.pipeline \
      --config "${ARRIVAL_CONFIG}" audit
    run_logged model_arrival_prepare \
      "${PYTHON_BIN}" -m experiments.model_arrival.pipeline \
      --config "${ARRIVAL_CONFIG}" prepare
  fi
  if scaling_selected; then
    run_logged closed_set_scaling_audit \
      "${PYTHON_BIN}" -m experiments.closed_set_scaling.pipeline \
      --config "${SCALING_CONFIG}" audit
    run_logged closed_set_scaling_prepare \
      "${PYTHON_BIN}" -m experiments.closed_set_scaling.pipeline \
      --config "${SCALING_CONFIG}" prepare
  fi
  if open_set_selected; then
    run_logged open_set_scaling_audit \
      "${PYTHON_BIN}" -m experiments.open_set_scaling.pipeline \
      --config "${OPEN_SET_CONFIG}" audit
    run_logged open_set_scaling_prepare \
      "${PYTHON_BIN}" -m experiments.open_set_scaling.pipeline \
      --config "${OPEN_SET_CONFIG}" prepare
  fi
  if engine_selected; then
    run_logged inference_engine_audit \
      "${PYTHON_BIN}" -m experiments.inference_engine.pipeline \
      --config "${ENGINE_CONFIG}" audit
    run_logged inference_engine_prepare \
      "${PYTHON_BIN}" -m experiments.inference_engine.pipeline \
      --config "${ENGINE_CONFIG}" prepare
  fi
}

run_arrival_queue() {
  if selected model_arrival_full; then
    run_logged model_arrival_full \
      env CUDA_VISIBLE_DEVICES="${MODEL_ARRIVAL_GPU}" \
      "${PYTHON_BIN}" -m experiments.model_arrival.pipeline \
      --config "${ARRIVAL_CONFIG}" run-grid \
      --xgb-device cuda
  fi
  if selected closed_set_scaling_full; then
    run_logged closed_set_scaling_full \
      env CUDA_VISIBLE_DEVICES="${CLOSED_SET_SCALING_GPU}" \
      "${PYTHON_BIN}" -m experiments.closed_set_scaling.pipeline \
      --config "${SCALING_CONFIG}" run-grid \
      --xgb-device cuda
  fi
  if selected open_set_scaling_full; then
    run_logged open_set_scaling_full \
      env CUDA_VISIBLE_DEVICES="${OPEN_SET_SCALING_GPU}" \
      "${PYTHON_BIN}" -m experiments.open_set_scaling.pipeline \
      --config "${OPEN_SET_CONFIG}" run-grid \
      --xgb-device cuda
  fi
}

run_engine_queue() {
  if selected inference_engine_full_seed42; then
    run_logged inference_engine_full_seed42 \
      env CUDA_VISIBLE_DEVICES="${INFERENCE_ENGINE_GPU}" \
      "${PYTHON_BIN}" -m experiments.inference_engine.pipeline \
      --config "${ENGINE_CONFIG}" run-grid \
      --feature-groups full \
      --seeds 42 \
      --xgb-device cuda
  fi
  if selected inference_engine_ablation_seed42; then
    run_logged inference_engine_ablation_seed42 \
      env CUDA_VISIBLE_DEVICES="${INFERENCE_ENGINE_GPU}" \
      "${PYTHON_BIN}" -m experiments.inference_engine.pipeline \
      --config "${ENGINE_CONFIG}" run-grid \
      --feature-groups timing_only non_timing \
      --seeds 42 \
      --xgb-device cuda
  fi
  if selected inference_engine_full_five_seeds; then
    run_logged inference_engine_full_five_seeds \
      env CUDA_VISIBLE_DEVICES="${INFERENCE_ENGINE_GPU}" \
      "${PYTHON_BIN}" -m experiments.inference_engine.pipeline \
      --config "${ENGINE_CONFIG}" run-grid \
      --feature-groups full \
      --seeds 40 41 42 43 44 \
      --xgb-device cuda
  fi
}

echo "Selected experiments:"
printf '  - %s\n' "${SELECTED_EXPERIMENTS[@]}"
echo "Logs: ${SRC_DIR}/${LOG_ROOT}"

if [[ "${#SELECTED_EXPERIMENTS[@]}" -eq 0 ]]; then
  echo "Nothing selected. Uncomment at least one experiment." >&2
  exit 2
fi

if [[ "${PREPARE_INPUTS}" == "true" ]]; then
  prepare_inputs
elif [[ "${PREPARE_INPUTS}" != "false" ]]; then
  echo "PREPARE_INPUTS must be true or false." >&2
  exit 2
fi

declare -a PIDS=()
declare -a NAMES=()
if arrival_selected || scaling_selected || open_set_selected; then
  run_arrival_queue > >(tee "${LOG_ROOT}/model_arrival.queue.log") 2>&1 &
  PIDS+=("$!")
  NAMES+=("midscene_scaling")
fi
if engine_selected; then
  run_engine_queue > >(tee "${LOG_ROOT}/inference_engine.queue.log") 2>&1 &
  PIDS+=("$!")
  NAMES+=("inference_engine")
fi

FAILED=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "Completed queue: ${NAMES[$index]}"
  else
    echo "Failed queue: ${NAMES[$index]}" >&2
    FAILED=1
  fi
done
exit "${FAILED}"
