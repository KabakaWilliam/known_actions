#!/usr/bin/env bash
#
# Human-readable experiment launcher.
#
# Workflow:
#   1. Edit only the SETTINGS section below.
#   2. Comment/uncomment experiment names in SELECTED_EXPERIMENTS.
#   3. Run: bash scripts/run_experiments_simple.sh
#
# Experiments assigned to the same GPU run sequentially. The cross-harness and
# policy queues run in parallel when both families are selected.

set -Eeuo pipefail

# =============================================================================
# SETTINGS — edit this section
# =============================================================================

# Physical GPU numbers. These are currently the two GPUs not used by the
# Qwen/Gemma SGLang collectors.
CROSS_HARNESS_GPU=2
POLICY_NORMALIZATION_GPU=3

# Set to true on a fresh artifact directory. It runs read-only audits and
# creates frozen manifests before launching classifiers.
# It is false right now because preparation already completed successfully.
PREPARE_INPUTS=false

# Comment out a line to skip that experiment. Uncomment it to run it.
SELECTED_EXPERIMENTS=(
  # Six-class model identification: train/test within each harness, transfer
  # between MidScene and browser-use, and train on a balanced harness mixture.
  # cross_main

  # Repeat model identification using timing-only and non-timing features to
  # measure which feature family carries the identity signal.
  # cross_ablation

  # Binary harness classification with one model held out at a time (LOMO).
  # This tests whether harness identity transfers to a previously unseen model.
  # harness_detector

  # Four-class WebShop defense study: canonical vs normalized-policy traces,
  # including fixed-attacker, defense-aware, and mixed-policy training.
  # policy_normalization

  # Five-seed confirmation of the full-feature cross-harness identity grid.
  cross_five_seeds

  # Five-seed confirmation of the full-feature policy-normalization study.
  # policy_five_seeds
)

# =============================================================================
# IMPLEMENTATION — normally no editing below this line
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${SRC_DIR}"

PYTHON_BIN="python"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_ROOT="artifacts/experiment_runs/simple_${RUN_STAMP}"

CROSS_CONFIG="experiments/cross_harness/configs/final_6model.yaml"
POLICY_CONFIG="experiments/policy_normalization/configs/webshop_full_analysis.yaml"

mkdir -p "${LOG_ROOT}"

selected() {
  local requested="$1"
  local experiment
  for experiment in "${SELECTED_EXPERIMENTS[@]}"; do
    if [[ "${experiment}" == "${requested}" ]]; then
      return 0
    fi
  done
  return 1
}

run_logged() {
  local label="$1"
  shift
  echo
  echo "[$(date -u +%FT%TZ)] START ${label}"
  echo "[$(date -u +%FT%TZ)] CMD   $*"
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${LOG_ROOT}/${label}.log"
  echo "[$(date -u +%FT%TZ)] DONE  ${label}"
}

cross_selected() {
  selected cross_main \
    || selected cross_ablation \
    || selected harness_detector \
    || selected cross_five_seeds
}

policy_selected() {
  selected policy_normalization || selected policy_five_seeds
}

prepare_selected_inputs() {
  if cross_selected; then
    run_logged cross_harness_audit \
      "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
      --config "${CROSS_CONFIG}" audit

    run_logged cross_harness_prepare \
      "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
      --config "${CROSS_CONFIG}" prepare
  fi

  if policy_selected; then
    run_logged policy_normalization_audit \
      "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
      --config "${POLICY_CONFIG}" audit

    run_logged policy_normalization_prepare \
      "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
      --config "${POLICY_CONFIG}" prepare
  fi
}

run_cross_queue() {
  if selected cross_main; then
    # Experiment: identify the six models within harness, across harnesses,
    # and after balanced mixed-harness training.
    run_logged cross_main_full_seed42 \
      env CUDA_VISIBLE_DEVICES="${CROSS_HARNESS_GPU}" \
      "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
      --config "${CROSS_CONFIG}" run-grid \
      --classifier XGBoost \
      --feature-group full \
      --seeds 42 \
      --xgb-device cuda
  fi

  if selected cross_ablation; then
    # Experiment: rerun model identification with timing isolated from all
    # website-visible non-timing features.
    run_logged cross_ablation_seed42 \
      env CUDA_VISIBLE_DEVICES="${CROSS_HARNESS_GPU}" \
      "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
      --config "${CROSS_CONFIG}" run-ablation \
      --classifier XGBoost \
      --feature-groups timing_only non_timing \
      --seeds 42 \
      --xgb-device cuda
  fi

  if selected harness_detector; then
    # Experiment: predict MidScene vs browser-use while holding out every
    # model in turn; run separate full/timing/non-timing feature views.
    local feature_group
    for feature_group in full timing_only non_timing; do
      run_logged "harness_detector_${feature_group}_seed42" \
        env CUDA_VISIBLE_DEVICES="${CROSS_HARNESS_GPU}" \
        "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
        --config "${CROSS_CONFIG}" harness-detector \
        --classifier XGBoost \
        --feature-group "${feature_group}" \
        --seed 42 \
        --xgb-device cuda
    done
  fi

  if selected cross_five_seeds; then
    # Confirmation: repeat the headline full-feature model-identity grid over
    # five classifier RNG seeds.
    run_logged cross_main_full_five_seeds \
      env CUDA_VISIBLE_DEVICES="${CROSS_HARNESS_GPU}" \
      "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
      --config "${CROSS_CONFIG}" run-grid \
      --classifier XGBoost \
      --feature-group full \
      --seeds 40 41 42 43 44 \
      --xgb-device cuda
  fi

  run_logged cross_harness_summarize \
    "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
    --config "${CROSS_CONFIG}" summarize
}

run_policy_queue() {
  if selected policy_normalization; then
    # Experiment: quantify fixed-attacker degradation and adaptive recovery
    # under the normalized browser-action policy.
    run_logged policy_grid_seed42 \
      env CUDA_VISIBLE_DEVICES="${POLICY_NORMALIZATION_GPU}" \
      "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
      --config "${POLICY_CONFIG}" run-grid \
      --feature-groups full timing_only non_timing \
      --seeds 42 \
      --xgb-device cuda
  fi

  if selected policy_five_seeds; then
    # Confirmation: repeat the policy experiment's full-feature view over five
    # classifier RNG seeds.
    run_logged policy_full_five_seeds \
      env CUDA_VISIBLE_DEVICES="${POLICY_NORMALIZATION_GPU}" \
      "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
      --config "${POLICY_CONFIG}" run-grid \
      --feature-groups full \
      --seeds 40 41 42 43 44 \
      --xgb-device cuda
  fi

  run_logged policy_normalization_summarize \
    "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
    --config "${POLICY_CONFIG}" summarize
}

echo "Selected experiments:"
printf '  - %s\n' "${SELECTED_EXPERIMENTS[@]}"
echo "Cross-harness GPU: ${CROSS_HARNESS_GPU}"
echo "Policy-normalization GPU: ${POLICY_NORMALIZATION_GPU}"
echo "Logs: ${SRC_DIR}/${LOG_ROOT}"

if [[ "${#SELECTED_EXPERIMENTS[@]}" -eq 0 ]]; then
  echo "Nothing selected. Uncomment at least one experiment." >&2
  exit 2
fi

if [[ "${PREPARE_INPUTS}" == "true" ]]; then
  prepare_selected_inputs
elif [[ "${PREPARE_INPUTS}" != "false" ]]; then
  echo "PREPARE_INPUTS must be true or false." >&2
  exit 2
else
  echo "Preparation skipped (PREPARE_INPUTS=false)."
fi

declare -a PIDS=()
declare -a NAMES=()

if cross_selected; then
  run_cross_queue > >(tee "${LOG_ROOT}/cross_queue.log") 2>&1 &
  PIDS+=("$!")
  NAMES+=("cross")
  echo "Launched cross-harness queue as PID ${PIDS[-1]}"
fi

if policy_selected; then
  run_policy_queue > >(tee "${LOG_ROOT}/policy_queue.log") 2>&1 &
  PIDS+=("$!")
  NAMES+=("policy")
  echo "Launched policy-normalization queue as PID ${PIDS[-1]}"
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
