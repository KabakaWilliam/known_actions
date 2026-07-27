#!/usr/bin/env bash
set -euo pipefail

# Browser Use inference-engine experiment for the single-GPU checkpoints:
#   Qwen/Qwen3-VL-8B-Instruct
#   zai-org/GLM-4.6V-Flash
#
# Edit only this SETTINGS block. Comment out operations you do not want, then:
#   bash scripts/run_small_vlm_engine_experiment.sh

PYTHON_BIN="/opt/anaconda/envs/dispatch/bin/python"
ANALYSIS_GPU="2"

SELECTED_OPERATIONS=(
  # Matched Browser Use controls. Required for a clean engine comparison.
  # collect_vllm

  # SGLang treatment. Each checkpoint is served on one GPU.
  # collect_sglang

  # Inspect matched coverage without writing frozen manifests.
  # audit

  # Freeze the exact common train/validation/test tasks.
  # prepare

  # Fast first result: full features, classifier seed 42.
  # train_full_seed42

  # Timing-only and non-timing views, seed 42.
  # train_ablations_seed42

  # Publication confirmation. Seed 42 resumes; it is not recomputed.
  # train_full_five_seeds

  # Regenerate CSV summaries and the human-readable Markdown report.
  # summarize
)

VLLM_CAMPAIGN="experiments/inference_engine/configs/webshop_small_vlm_vllm_campaign.yaml"
SGLANG_CAMPAIGN="experiments/inference_engine/configs/webshop_small_vlm_sglang_campaign.yaml"
ANALYSIS_CONFIG="experiments/inference_engine/configs/webshop_sglang_small_vlm_4model_analysis.yaml"

selected() {
  local wanted="$1"
  local operation
  for operation in "${SELECTED_OPERATIONS[@]}"; do
    [[ "$operation" == "$wanted" ]] && return 0
  done
  return 1
}

run_collection_pair() {
  local config="$1"
  local label="$2"
  local qwen_log="$LOG_ROOT/${label}_qwen3vl8.log"
  local glm_log="$LOG_ROOT/${label}_glm_flash.log"

  echo "Starting $label Qwen3-VL-8B and GLM-4.6V-Flash collections in parallel."
  "$PYTHON_BIN" browser_use_campaign.py \
    --config "$config" --only qwen3vl_8b --skip-openrouter \
    2>&1 | tee "$qwen_log" &
  local qwen_pid=$!
  "$PYTHON_BIN" browser_use_campaign.py \
    --config "$config" --only glm_4.6v_flash --skip-openrouter \
    2>&1 | tee "$glm_log" &
  local glm_pid=$!

  local status=0
  wait "$qwen_pid" || status=$?
  wait "$glm_pid" || status=$?
  return "$status"
}

run_analysis() {
  CUDA_VISIBLE_DEVICES="$ANALYSIS_GPU" "$PYTHON_BIN" \
    -m experiments.inference_engine.pipeline \
    --config "$ANALYSIS_CONFIG" "$@"
}

if [[ ${#SELECTED_OPERATIONS[@]} -eq 0 ]]; then
  echo "No operations selected. Uncomment entries in SELECTED_OPERATIONS."
  exit 0
fi

LOG_ROOT="artifacts/experiment_runs/small_vlm_engine_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
echo "Logs: $LOG_ROOT"

if selected collect_vllm; then
  run_collection_pair "$VLLM_CAMPAIGN" "vllm"
fi

if selected collect_sglang; then
  run_collection_pair "$SGLANG_CAMPAIGN" "sglang"
fi

if selected audit; then
  run_analysis audit | tee "$LOG_ROOT/audit.log"
fi

if selected prepare; then
  run_analysis prepare | tee "$LOG_ROOT/prepare.log"
fi

if selected train_full_seed42; then
  run_analysis run-grid \
    --feature-groups full --seeds 42 --xgb-device cuda \
    | tee "$LOG_ROOT/train_full_seed42.log"
fi

if selected train_ablations_seed42; then
  run_analysis run-grid \
    --feature-groups timing_only non_timing --seeds 42 --xgb-device cuda \
    | tee "$LOG_ROOT/train_ablations_seed42.log"
fi

if selected train_full_five_seeds; then
  run_analysis run-grid \
    --feature-groups full --seeds 40 41 42 43 44 --xgb-device cuda \
    | tee "$LOG_ROOT/train_full_five_seeds.log"
fi

if selected summarize; then
  run_analysis summarize | tee "$LOG_ROOT/summarize.log"
fi
