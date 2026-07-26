#!/usr/bin/env bash
#
# Human-readable launcher for the MidScene future-wave experiment.
#
# Edit only SETTINGS: comment/uncomment operations, then run:
#   bash scripts/run_temporal_generalization.sh
#
# Operations run in the order listed. Collection and classifier fits resume.

set -Eeuo pipefail

# =============================================================================
# SETTINGS
# =============================================================================

CLASSIFIER_GPU=0

SELECTED_OPERATIONS=(
  # Collect Qwen on GPU 0 and Gemma on GPU 1 concurrently. Each campaign owns
  # its vLLM server, trace namespace, log, and resume state.
  # collect_qwen_and_gemma_parallel 

  # Collect GLM after Qwen/Gemma finish. GLM uses GPUs 0-3 and therefore must
  # not overlap either single-GPU local collection.
  # collect_glm

  # Collect the same 75 test tasks with Gemini 3.1 Pro through OpenRouter.
  # collect_gemini

  # Audit traces and freeze matched old/future task manifests. Run only after
  # every model above has completed collection.
  # prepare_analysis

  # Fast headline: old-wave classifier tested on old and future test waves.
  # temporal_full_seed42

  # Determine whether temporal robustness differs for timing/non-timing signal.
  # temporal_ablations_seed42

  # Five-seed confirmation. Seed 42 is resumed rather than recomputed.
  temporal_full_five_seeds
)

# =============================================================================
# IMPLEMENTATION
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${SRC_DIR}"

PYTHON_BIN="/opt/anaconda/envs/dispatch/bin/python"
CAMPAIGN_CONFIG="experiments/temporal_generalization/configs/webshop_future_midscene_campaign.yaml"
ANALYSIS_CONFIG="experiments/temporal_generalization/configs/webshop_future_midscene_analysis.yaml"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_ROOT="artifacts/experiment_runs/temporal_generalization_${RUN_STAMP}"
MPLCONFIGDIR="${SRC_DIR}/../.tmp/matplotlib"
export MPLCONFIGDIR
mkdir -p "${LOG_ROOT}" "${MPLCONFIGDIR}"

selected() {
  local wanted="$1"
  local value
  for value in "${SELECTED_OPERATIONS[@]}"; do
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

if [[ "${#SELECTED_OPERATIONS[@]}" -eq 0 ]]; then
  echo "Nothing selected. Uncomment at least one operation in SETTINGS." >&2
  exit 2
fi

echo "Selected temporal-generalization operations:"
printf '  - %s\n' "${SELECTED_OPERATIONS[@]}"
echo "Logs: ${SRC_DIR}/${LOG_ROOT}"

if selected collect_qwen_and_gemma_parallel; then
  run_logged collect_qwen \
    "${PYTHON_BIN}" browser_use_campaign.py \
    --config "${CAMPAIGN_CONFIG}" \
    --only qwen3_5_27b \
    --skip-openrouter &
  qwen_pid=$!
  run_logged collect_gemma \
    "${PYTHON_BIN}" browser_use_campaign.py \
    --config "${CAMPAIGN_CONFIG}" \
    --only gemma_4_26B_A4B_it \
    --skip-openrouter &
  gemma_pid=$!
  collection_failed=0
  wait "${qwen_pid}" || collection_failed=1
  wait "${gemma_pid}" || collection_failed=1
  if [[ "${collection_failed}" -ne 0 ]]; then
    echo "Qwen/Gemma parallel collection failed; inspect logs before resuming." >&2
    exit 1
  fi
fi

if selected collect_glm; then
  run_logged collect_glm \
    "${PYTHON_BIN}" browser_use_campaign.py \
    --config "${CAMPAIGN_CONFIG}" \
    --only glm_4.6v \
    --skip-openrouter
fi

if selected collect_gemini; then
  run_logged collect_gemini \
    "${PYTHON_BIN}" browser_use_campaign.py \
    --config "${CAMPAIGN_CONFIG}" \
    --only gemini_3_1 \
    --skip-local
fi

if selected prepare_analysis; then
  run_logged temporal_audit \
    "${PYTHON_BIN}" -m experiments.temporal_generalization.pipeline \
    --config "${ANALYSIS_CONFIG}" audit
  run_logged temporal_prepare \
    "${PYTHON_BIN}" -m experiments.temporal_generalization.pipeline \
    --config "${ANALYSIS_CONFIG}" prepare
fi

if selected temporal_full_seed42; then
  run_logged temporal_full_seed42 \
    env CUDA_VISIBLE_DEVICES="${CLASSIFIER_GPU}" \
    "${PYTHON_BIN}" -m experiments.temporal_generalization.pipeline \
    --config "${ANALYSIS_CONFIG}" run-grid \
    --feature-groups full \
    --seeds 42 \
    --xgb-device cuda
fi

if selected temporal_ablations_seed42; then
  run_logged temporal_ablations_seed42 \
    env CUDA_VISIBLE_DEVICES="${CLASSIFIER_GPU}" \
    "${PYTHON_BIN}" -m experiments.temporal_generalization.pipeline \
    --config "${ANALYSIS_CONFIG}" run-grid \
    --feature-groups timing_only non_timing \
    --seeds 42 \
    --xgb-device cuda
fi

if selected temporal_full_five_seeds; then
  run_logged temporal_full_five_seeds \
    env CUDA_VISIBLE_DEVICES="${CLASSIFIER_GPU}" \
    "${PYTHON_BIN}" -m experiments.temporal_generalization.pipeline \
    --config "${ANALYSIS_CONFIG}" run-grid \
    --feature-groups full \
    --seeds 40 41 42 43 44 \
    --xgb-device cuda
fi
