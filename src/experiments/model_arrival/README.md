# Closed-set model-arrival update experiment

This experiment directly tests the claim that a closed-set model-identity
classifier can be cheaply updated when a new model is released. It uses only
the original MidScene traces and does not require new collection.

Each of the 14 models takes one turn as the simulated new release. For each
fold, the classifier receives the complete training set for the other 13
models and an increasing labeled-trace budget for the arriving model:
1, 2, 5, 10, 20, 50, 100, and all available traces. The classifier is then
refit with frozen XGBoost hyperparameters and evaluated on the unchanged
14-model test split.

The primary quantities are:

- F1 for the arriving model;
- overall 14-class macro-F1;
- macro-F1 over the 13 established models, measuring retention;
- fraction of the full-update arriving-model F1 recovered;
- refit wall-clock time and number of newly labeled traces.

Class-balanced sample weights prevent a small arriving-model class from being
ignored merely because the 13 established classes have more traces.
Hyperparameters remain fixed throughout the arrival curve, so the experiment
measures the cost of adding labels and refitting rather than repeating model
selection at every budget.

From `src/`:

```bash
python -m experiments.model_arrival.pipeline audit
python -m experiments.model_arrival.pipeline prepare
python -m experiments.model_arrival.pipeline run-grid --xgb-device cuda
python -m experiments.model_arrival.pipeline summarize
```

The grid uses seeds 40–44. Outputs and the generated human-readable report are
written to:

```text
artifacts/experiments/model_arrival/midscene_14model_incremental_update_v1/
```
