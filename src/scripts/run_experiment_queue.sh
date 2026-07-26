#!/usr/bin/env bash
#
# Run the classifier experiments that are independent of the in-progress
# SGLang trace collection.
#
# Editing workflow:
#   - Comment out any `run_logged ...` line inside a queue to skip that
#     experiment.
#   - Comment out either `launch ...` line at the bottom to skip that entire
#     GPU queue.
#   - Keep preparation enabled unless its frozen manifests already exist.
#
# GPU layout while Qwen/Gemma trace collection occupies GPUs 0 and 1:
#   GPU 2: final six-model cross-harness queue
#   GPU 3: four-model policy-normalization queue

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${SRC_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CROSS_GPU="${CROSS_GPU:-2}"
POLICY_GPU="${POLICY_GPU:-3}"
EXPERIMENT_QUEUE="${EXPERIMENT_QUEUE:-both}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-artifacts/experiment_runs/${RUN_STAMP}}"

CROSS_CONFIG="experiments/cross_harness/configs/final_6model.yaml"
POLICY_CONFIG="experiments/policy_normalization/configs/webshop_full_analysis.yaml"

mkdir -p "${LOG_ROOT}"

run_logged() {
  local label="$1"
  shift
  echo
  echo "[$(date -u +%FT%TZ)] START ${label}"
  echo "[$(date -u +%FT%TZ)] CMD   $*"
  "$@" 2>&1 | tee "${LOG_ROOT}/${label}.log"
  echo "[$(date -u +%FT%TZ)] DONE  ${label}"
}

prepare_inputs() {
  run_logged cross_harness_audit \
    "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
    --config "${CROSS_CONFIG}" audit

  run_logged cross_harness_prepare \
    "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
    --config "${CROSS_CONFIG}" prepare

  run_logged policy_normalization_audit \
    "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
    --config "${POLICY_CONFIG}" audit

  run_logged policy_normalization_prepare \
    "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
    --config "${POLICY_CONFIG}" prepare
}

run_cross_harness_queue() {
  # Experiment 1 — six-class model identity:
  # train/test within each harness, transfer between MidScene and browser-use,
  # and train on a balanced mixture before testing each harness separately.
  run_logged cross_main_full_seed42 \
    env CUDA_VISIBLE_DEVICES="${CROSS_GPU}" \
    "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
    --config "${CROSS_CONFIG}" run-grid \
    --classifier XGBoost \
    --feature-group full \
    --seeds 42 \
    --xgb-device cuda

  # Experiment 2 — feature ablation:
  # repeat model identification with timing-only and non-timing features.
  # `full` is omitted because Experiment 1 already produces it.
  run_logged cross_ablation_seed42 \
    env CUDA_VISIBLE_DEVICES="${CROSS_GPU}" \
    "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
    --config "${CROSS_CONFIG}" run-ablation \
    --classifier XGBoost \
    --feature-groups timing_only non_timing \
    --seeds 42 \
    --xgb-device cuda

  # Experiment 3 — binary harness classification:
  # predict MidScene vs browser-use while holding one model out of training,
  # then test transfer on that unseen model (LOMO), one feature view at a time.
  run_logged harness_detector_full_seed42 \
    env CUDA_VISIBLE_DEVICES="${CROSS_GPU}" \
    "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
    --config "${CROSS_CONFIG}" harness-detector \
    --classifier XGBoost \
    --feature-group full \
    --seed 42 \
    --xgb-device cuda

  run_logged harness_detector_timing_seed42 \
    env CUDA_VISIBLE_DEVICES="${CROSS_GPU}" \
    "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
    --config "${CROSS_CONFIG}" harness-detector \
    --classifier XGBoost \
    --feature-group timing_only \
    --seed 42 \
    --xgb-device cuda

  run_logged harness_detector_non_timing_seed42 \
    env CUDA_VISIBLE_DEVICES="${CROSS_GPU}" \
    "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
    --config "${CROSS_CONFIG}" harness-detector \
    --classifier XGBoost \
    --feature-group non_timing \
    --seed 42 \
    --xgb-device cuda

  run_logged cross_harness_summarize \
    "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
    --config "${CROSS_CONFIG}" summarize

  # Final five-seed confirmation: leave disabled until seed-42 outputs have
  # been reviewed. The current pipeline performs tuning per seed, so this is
  # intentionally not part of the fast queue.
  #
  # run_logged cross_main_full_five_seeds \
  #   env CUDA_VISIBLE_DEVICES="${CROSS_GPU}" \
  #   "${PYTHON_BIN}" -m experiments.cross_harness.pipeline \
  #   --config "${CROSS_CONFIG}" run-grid \
  #   --classifier XGBoost \
  #   --feature-group full \
  #   --seeds 40 41 42 43 44 \
  #   --xgb-device cuda
}

run_policy_normalization_queue() {
  # Experiment 4 — behavioral-policy defense:
  # compare canonical and normalized-policy WebShop traces under a fixed
  # canonical attacker, a defense-aware attacker, and mixed-policy training.
  run_logged policy_grid_seed42 \
    env CUDA_VISIBLE_DEVICES="${POLICY_GPU}" \
    "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
    --config "${POLICY_CONFIG}" run-grid \
    --feature-groups full timing_only non_timing \
    --seeds 42 \
    --xgb-device cuda

  run_logged policy_normalization_summarize \
    "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
    --config "${POLICY_CONFIG}" summarize

  # Final five-seed confirmation: enable after reviewing seed 42.
  #
  # run_logged policy_full_five_seeds \
  #   env CUDA_VISIBLE_DEVICES="${POLICY_GPU}" \
  #   "${PYTHON_BIN}" -m experiments.policy_normalization.pipeline \
  #   --config "${POLICY_CONFIG}" run-grid \
  #   --feature-groups full \
  #   --seeds 40 41 42 43 44 \
  #   --xgb-device cuda
}

declare -a JOB_PIDS=()
declare -a JOB_NAMES=()

launch() {
  local name="$1"
  shift
  "$@" > >(tee "${LOG_ROOT}/${name}.queue.log") 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("${name}")
  echo "Launched ${name} as PID ${JOB_PIDS[-1]}"
}

wait_for_jobs() {
  local failed=0
  local index
  for index in "${!JOB_PIDS[@]}"; do
    if wait "${JOB_PIDS[$index]}"; then
      echo "Queue completed: ${JOB_NAMES[$index]}"
    else
      echo "Queue failed: ${JOB_NAMES[$index]}" >&2
      failed=1
    fi
  done
  return "${failed}"
}

echo "Experiment logs: ${SRC_DIR}/${LOG_ROOT}"
echo "Cross-harness GPU: ${CROSS_GPU}"
echo "Policy GPU: ${POLICY_GPU}"
echo "Queue selection: ${EXPERIMENT_QUEUE}"

# Preparation is serial to avoid concurrent writes to frozen manifests.
if [[ "${SKIP_PREPARE}" != "1" ]]; then
  prepare_inputs
else
  echo "Skipping preparation because SKIP_PREPARE=1"
fi

case "${EXPERIMENT_QUEUE}" in
  both)
    launch cross_harness_queue run_cross_harness_queue
    launch policy_normalization_queue run_policy_normalization_queue
    ;;
  cross)
    launch cross_harness_queue run_cross_harness_queue
    ;;
  policy)
    launch policy_normalization_queue run_policy_normalization_queue
    ;;
  *)
    echo "EXPERIMENT_QUEUE must be one of: both, cross, policy" >&2
    exit 2
    ;;
esac

wait_for_jobs
