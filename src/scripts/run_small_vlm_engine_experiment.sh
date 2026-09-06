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
COLLECTION_GPUS=(0 1 2 3)
export PYTHONUNBUFFERED=1

SELECTED_OPERATIONS=(
  # Resource-aware queue over GPUs 0-3: complete both GLM engine
  # conditions first, then both Qwen conditions. It waits for randomly
  # available GPUs and resumes existing traces.
  collect_dynamic_engine_queue

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

ANALYSIS_CONFIG="experiments/inference_engine/configs/webshop_sglang_small_vlm_4model_analysis.yaml"

selected() {
  local wanted="$1"
  local operation
  for operation in "${SELECTED_OPERATIONS[@]}"; do
    [[ "$operation" == "$wanted" ]] && return 0
  done
  return 1
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

if selected collect_dynamic_engine_queue; then
  "$PYTHON_BIN" scripts/schedule_small_vlm_collection.py \
    --gpus "${COLLECTION_GPUS[@]}" \
    --maximum-used-mib 2048 --maximum-utilization 10 --poll-seconds 30 \
    | tee "$LOG_ROOT/dynamic_collection_queue.log"
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
