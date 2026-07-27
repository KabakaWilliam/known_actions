# Leave-multiple-models-out open-set identification

This experiment addresses whether identification remains reliable when several
agents are absent from classifier training. It uses the original frozen,
task-matched MidScene manifests for all 14 models and both datasets.

For each outer fold, `p` models are removed completely from training and
validation, for `p = 1, 2, 3, 4`. XGBoost learns the remaining `14-p` identities.
At test time it must identify traces from known models and reject traces from
all held-out models as `UNKNOWN`.

The confidence threshold is learned from known validation traces only and
targets 95% known validation acceptance. The held-out models never participate
in fitting or calibration. Headline metrics are:

- known-versus-unknown AUROC and average precision;
- OSCR (correct known identification versus unknown false acceptance);
- known-model macro-F1;
- unknown recall at the known-validation threshold;
- balanced open-set macro-F1 over the known labels plus `UNKNOWN`.

All 14 one-model and 91 two-model holdouts are evaluated. For the 364 and 1,001
possible three- and four-model holdouts, respectively, the frozen design uses
100 deterministic model-balanced sets each. Five classifier seeds are averaged
within each holdout set before aggregating across model sets.

Extracted tabular features are cached under the experiment artifact directory,
keyed by the frozen source-manifest hash. This makes interrupted grids cheap to
resume without altering or normalizing the original traces.

From `src/`:

```bash
python -m experiments.open_set_scaling.pipeline audit
python -m experiments.open_set_scaling.pipeline prepare
CUDA_VISIBLE_DEVICES=0 python -m experiments.open_set_scaling.pipeline \
  run-grid --xgb-device cuda
```

For a resumable one-fold-per-condition smoke run:

```bash
python -m experiments.open_set_scaling.pipeline run-grid \
  --datasets webshop --unknown-counts 1 2 3 4 --seeds 42 \
  --limit-holdouts 1 --xgb-device cpu
```

Outputs live under:

```text
artifacts/experiments/open_set_scaling/midscene_14model_leave_p_out_v1/
```

`REPORT.md` is the human-readable entry point. The pipeline never changes raw
traces or the source manifests.
