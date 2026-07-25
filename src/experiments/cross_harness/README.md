# Cross-harness experiments

This pipeline covers two related questions:

1. Model identity under MidScene, browser-use, and a task-balanced 50/50 mix.
2. Binary harness detection with leave-one-model-out evaluation.

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
  --classifier XGBoost

python -m experiments.cross_harness.pipeline run-ablation \
  --classifier XGBoost --xgb-device cpu
```

Run harness detection for one feature view:

```bash
python -m experiments.cross_harness.pipeline harness-detector \
  --classifier XGBoost --feature-group non_timing --xgb-device cpu
```

Use `--config experiments/cross_harness/configs/provisional_5model.yaml` for
the isolated provisional experiment. The old `cross_harness_pipeline.py`
entry point remains as a compatibility shim.
