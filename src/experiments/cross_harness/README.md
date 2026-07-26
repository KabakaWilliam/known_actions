# Cross-harness experiments

This pipeline covers two related questions:

1. Model identity under MidScene, browser-use, and a task-balanced 50/50 mix.
2. Binary harness detection with leave-one-model-out evaluation.

## Frozen main-experiment roster

The final cross-harness experiment contains exactly these six models:

| Agent ID | Canonical model | Access |
|---|---|---|
| `qwen3_5_27b` | `Qwen/Qwen3.5-27B` | local vLLM |
| `glm_4.6v` | `zai-org/GLM-4.6V` (served as `glm-4.6v`) | local vLLM |
| `gemma_4_26B_A4B_it` | `google/gemma-4-26B-A4B-it` | local vLLM |
| `gpt_5_4` | GPT-5.4 | API |
| `gemini_3_1` | Gemini 3.1 Pro Preview | API |
| `claude_opus_4_6` | Claude Opus 4.6 | API |

Qwen3-VL is not part of the main six-model experiment. It remains a model in
the separate policy-defense experiment, where its normalized-policy treatment
uses the same two-GPU tensor-parallel topology as its canonical controls.

## Artifact naming

New experiment outputs use:

```text
artifacts/experiments/<study>/<cohort-version>/
├── frozen_task_splits/
├── model_identity/
├── model_identity_summaries/
└── harness_detection_lomo/
```

The final roster therefore writes to
`artifacts/experiments/cross_harness/main_6model_3local_3api_v2/`. Directory
names describe the scientific question rather than only an implementation
object such as `models` or an opaque version such as `v1`.

The existing `artifacts/classifiers/multi_harness_identity/v1/` directory is a
preserved legacy pilot using Qwen3-VL instead of Gemma. It is not renamed
because its JSON records contain paths to that frozen layout.

`configs/final_6model.yaml` records the provider-qualified aliases observed in
the historical MidScene and browser-use traces. Inventory scans reject traces
with a model-name/agent-ID mismatch.

The completed browser-use launch manifests record GLM-4.6V with tensor
parallelism 4. The separate policy-defense configuration keeps Qwen3-VL at
tensor parallelism 2 on GPUs 0 and 3. These are recorded experimental
conditions, not minimum hardware requirements. Experiments that reuse those
traces must match the serving topology or explicitly treat a topology change
as another experimental variable.

It reads frozen JSONL manifests and never mutates raw traces. Feature ablations
are views constructed during extraction:

- `full`: all 41 tabular trace features.
- `timing_only`: the 15 duration/inter-event/dwell features.
- `non_timing`: the remaining 26 browser-visible behavioral features.

The LSTM is intentionally restricted to `full`: its event sequence embeds
timing, so labeling an LSTM run `non_timing` would be misleading.

From `src/`, audit and freeze the final six-model intersection:

```bash
python -m experiments.cross_harness.pipeline audit
python -m experiments.cross_harness.pipeline prepare
```

Run the primary identity grid and feature ablation:

```bash
python -m experiments.cross_harness.pipeline run-grid \
  --classifier XGBoost --seeds 40 41 42 43 44

python -m experiments.cross_harness.pipeline run-ablation \
  --classifier XGBoost --seeds 40 41 42 43 44 --xgb-device cpu
```

Each evaluation includes a stratified percentile bootstrap over test traces.
`summaries/seed_aggregates.csv` reports the mean across classifier seeds and a
bootstrap interval obtained by resampling the aligned test traces and averaging
macro-F1 across seeds within each replicate.

The summarizer also writes class-level metrics without printing them:

- `summaries/per_model_metrics.csv`: precision, recall, F1, and support for
  every evaluation seed and model.
- `summaries/per_model_seed_aggregates.csv`: the per-model mean and standard
  deviation across classifier seeds.

`macro_f1` is the unweighted arithmetic mean of the model-level F1 scores.
`weighted_f1` is recorded separately and weights each model by test support.

Run harness detection for one feature view:

```bash
python -m experiments.cross_harness.pipeline harness-detector \
  --classifier XGBoost --feature-group non_timing --xgb-device cpu
```

Use `--config experiments/cross_harness/configs/provisional_5model.yaml` for
the isolated provisional experiment. The old `cross_harness_pipeline.py`
entry point remains as a compatibility shim.
