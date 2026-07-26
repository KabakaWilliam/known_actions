# Experiment index

The experiment directories are the canonical place to find a research
question, frozen configuration, command, and output namespace. Raw traces are
inputs and are never rewritten by analysis code.

| Experiment | Entry point | Config | Output |
|---|---|---|---|
| Cross-harness identity | `experiments.cross_harness.pipeline` | `cross_harness/configs/final_6model.yaml` | `artifacts/experiments/cross_harness/main_6model_3local_3api_v2/` |
| Provisional five-model identity | same | `cross_harness/configs/provisional_5model.yaml` | `artifacts/classifiers/multi_harness_identity/provisional_5model/` |
| Harness detection (LOMO) | same, `harness-detector` | either config above | `<artifact_root>/harness_detection_lomo/` |
| Timing feature ablation | same, `run-ablation` | either config above | `<artifact_root>/model_identity/**/features=*/` |
| Full-WebShop policy normalization | `experiments.policy_normalization.pipeline` | `policy_normalization/configs/webshop_full_analysis.yaml` | `artifacts/experiments/defenses/webshop_full_policy_normalization_4model_v1/` |
| Closed-set model arrival | `experiments.model_arrival.pipeline` | `model_arrival/configs/midscene_14model.yaml` | `artifacts/experiments/model_arrival/midscene_14model_incremental_update_v1/` |
| SGLang engine intervention (full WebShop) | `browser_use_campaign.py` | `inference_engine/configs/webshop_sglang_smoke_campaign.yaml`, then `webshop_sglang_full_campaign.yaml` | `traces_experiments/inference_engine_webshop_sglang_{smoke,full}_v1/` |
| vLLM↔SGLang analysis | `experiments.inference_engine.pipeline` | `inference_engine/configs/webshop_sglang_analysis.yaml` | `artifacts/experiments/inference_engine/webshop_sglang_full_v1/` |
| MidScene future-wave temporal generalization | `experiments.temporal_generalization.pipeline` | `temporal_generalization/configs/webshop_future_midscene_analysis.yaml` | `artifacts/experiments/temporal_generalization/webshop_midscene_future_wave_4model_v1/` |

Historical single-harness, open-set, learning-curve, timing-jitter, and
publication workflows remain at the top level during collection. They are
indexed in `../scripts/README.md`; moving their inputs and outputs is deferred
until no collector is active.

## Human-readable result contract

Every maintained analysis artifact root must contain a generated `REPORT.md`
that can be read without opening JSON or remembering command-line terminology.
Each report must state:

1. the prediction task and experimental unit;
2. what every train/test condition means;
3. headline metrics with seed counts and uncertainty semantics;
4. feature ablations separately from headline results;
5. per-model and utility caveats where applicable;
6. incomplete or partial experiment status;
7. links to machine-readable CSV/JSON and individual evaluations.

`summarize` regenerates the report from machine-readable outputs; the report is
never the source of truth. The current maintained reports are:

- `artifacts/experiments/cross_harness/main_6model_3local_3api_v2/REPORT.md`
- `artifacts/experiments/defenses/webshop_full_policy_normalization_4model_v1/REPORT.md`
- `artifacts/experiments/model_arrival/midscene_14model_incremental_update_v1/REPORT.md`
- `artifacts/experiments/inference_engine/webshop_sglang_full_v1/REPORT.md`
- `artifacts/experiments/temporal_generalization/webshop_midscene_future_wave_4model_v1/REPORT.md`

Collection-only campaign logs are not a results report.
