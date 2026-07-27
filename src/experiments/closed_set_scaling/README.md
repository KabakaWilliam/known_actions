# Closed-set class-count scaling

This experiment measures how MidScene model-identification performance changes
as the closed set grows from 2 to all 14 models. It complements the
model-arrival experiment:

- model arrival asks how many examples are needed to add one new label;
- class-count scaling asks how identification difficulty and compute change as
  the number of candidate labels increases.

The experiment reuses the frozen, task-matched train/test manifests from the
original MidScene campaign. For each of five seeds, the 14 models are placed in
a deterministic random order. The two-class set contains the first two models,
the three-class set adds the third, and so on. These nested sets ensure that
each point adds exactly one model, while using five different orders prevents
one arbitrary ordering from determining the result.

XGBoost hyperparameters remain fixed. The experiment reports:

- macro-F1 and accuracy;
- the balanced uniform-random macro-F1 reference, `1 / class_count`;
- fit time and prediction milliseconds per trace;
- serialized booster size;
- number of trees, nodes, and leaves.

The runtime figure shows all five repetitions as faint trajectories, their
median as a thick line, and their interquartile range as a colored band,
exposing system-load or warm-up outliers directly.
Complexity curves use the median and interquartile range across model subsets.
The IQR is a dispersion summary, not a confidence interval.

Tree and model complexity are reported instead of FLOPs because FLOPs are not a
standard or particularly meaningful cost measure for tree traversal.

From `src/`:

```bash
python -m experiments.closed_set_scaling.pipeline audit
python -m experiments.closed_set_scaling.pipeline prepare
python -m experiments.closed_set_scaling.pipeline run-grid --xgb-device cuda
```

Outputs, PNG/PDF figures, CSV summaries, and `REPORT.md` are written to:

```text
artifacts/experiments/closed_set_scaling/midscene_14model_class_count_scaling_v1/
```
