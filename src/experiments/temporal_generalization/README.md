# Future-wave temporal generalization

This experiment asks whether a classifier fitted on the original MidScene
collection still identifies models in a genuinely later collection wave.

## Protocol

- Benchmark: WebShop.
- Models: Qwen3.5-27B, GLM-4.6V, Gemma-4-26B-A4B, and Gemini 3.1 Pro.
- New collection: the 75 original WebShop test tasks only.
- Classifier fitting and model selection: original-wave train and validation
  traces only.
- Evaluation: original and future traces for exactly the same test task IDs.
- Primary metric: four-class macro-F1.
- Temporal effect: future minus original macro-F1 with a paired task-clustered
  bootstrap confidence interval.

This is a realistic temporal-drift test. It does not claim to isolate time from
all operational changes to the browser, harness dependencies, serving engine,
hardware load, or API-provider infrastructure.

## Collection

Run all four models sequentially:

```bash
python browser_use_campaign.py \
  --config experiments/temporal_generalization/configs/webshop_future_midscene_campaign.yaml
```

Local-only and Gemini-only runs can be launched independently:

```bash
python browser_use_campaign.py \
  --config experiments/temporal_generalization/configs/webshop_future_midscene_campaign.yaml \
  --only qwen3_5_27b glm_4.6v gemma_4_26B_A4B_it \
  --skip-openrouter

python browser_use_campaign.py \
  --config experiments/temporal_generalization/configs/webshop_future_midscene_campaign.yaml \
  --only gemini_3_1 \
  --skip-local
```

The human launcher exposes the faster safe schedule: Qwen on GPU 0 and Gemma
on GPU 1 concurrently, followed by GLM on all four GPUs. Do not overlap GLM
with either single-GPU collector.

Future traces are written under
`traces_experiments/temporal_generalization_webshop_midscene_future_v1/`.
Original traces remain untouched.

## Analysis

After all four models are complete:

```bash
python -m experiments.temporal_generalization.pipeline audit
python -m experiments.temporal_generalization.pipeline prepare

CUDA_VISIBLE_DEVICES=0 \
python -m experiments.temporal_generalization.pipeline run-grid \
  --feature-groups full \
  --seeds 42 \
  --xgb-device cuda
```

Run the remaining feature views and five classifier seeds after inspecting the
seed-42 headline:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m experiments.temporal_generalization.pipeline run-grid \
  --feature-groups timing_only non_timing \
  --seeds 42 \
  --xgb-device cuda

CUDA_VISIBLE_DEVICES=0 \
python -m experiments.temporal_generalization.pipeline run-grid \
  --feature-groups full \
  --seeds 40 41 42 43 44 \
  --xgb-device cuda
```

The generated human-readable result is
`artifacts/experiments/temporal_generalization/webshop_midscene_future_wave_4model_v1/REPORT.md`.
