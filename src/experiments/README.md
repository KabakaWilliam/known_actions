# Experiment index

The experiment directories are the canonical place to find a research
question, frozen configuration, command, and output namespace. Raw traces are
inputs and are never rewritten by analysis code.

| Experiment | Entry point | Config | Output |
|---|---|---|---|
| Cross-harness identity | `experiments.cross_harness.pipeline` | `cross_harness/configs/final_6model.yaml` | `artifacts/classifiers/multi_harness_identity/v1/` |
| Provisional five-model identity | same | `cross_harness/configs/provisional_5model.yaml` | `artifacts/classifiers/multi_harness_identity/provisional_5model/` |
| Harness detection (LOMO) | same, `harness-detector` | either config above | `<artifact_root>/harness_detector/` |
| Timing feature ablation | same, `run-ablation` | either config above | `<artifact_root>/models/**/features=*/` |
| Full-WebShop policy normalization | `experiments.policy_normalization.pipeline` | `policy_normalization/configs/webshop_full_analysis.yaml` | `artifacts/experiments/defenses/webshop_full_policy_normalization_4model_v1/` |
| SGLang engine intervention (full WebShop) | `browser_use_campaign.py` | `inference_engine/configs/webshop_sglang_smoke_campaign.yaml`, then `webshop_sglang_full_campaign.yaml` | `traces_experiments/inference_engine_webshop_sglang_{smoke,full}_v1/` |

Historical single-harness, open-set, learning-curve, timing-jitter, and
publication workflows remain at the top level during collection. They are
indexed in `../scripts/README.md`; moving their inputs and outputs is deferred
until no collector is active.
