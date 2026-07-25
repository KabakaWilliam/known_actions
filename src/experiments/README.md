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

Historical single-harness, open-set, learning-curve, timing-jitter, and
publication workflows remain at the top level during collection. They are
indexed in `../scripts/README.md`; moving their inputs and outputs is deferred
until no collector is active.
