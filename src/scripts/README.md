# Workflow index

The shell entry points remain in `src/` until trace collection and campaign
recovery are finished. This index makes their roles explicit without changing
working-directory assumptions mid-campaign.

## Collection and operations

- `collect_traces.sh`, `orchestrator.py`: original collection.
- `browser_use_campaign.py`, `browser_use_campaign.yaml`: resumable
  browser-use campaign.
- `check_latest_vllm_campaign.sh`, `check_open_router_balance.sh`,
  `check_trace_covereage.sh`: campaign diagnostics.
- `serve_vllm.sh`: manual local model serving.
- `prep_datasets.py`, `reverify.py`, `normalize_browser_use_traces.py`:
  explicit data maintenance. None run automatically at startup.

## Classifier experiments

- `trace_analyzer.py`, `train_classifiers*.sh`: historical closed-set runs.
- `train_open_set*.sh`: open-set runs.
- `train_universal_classifiers.sh`, `eval_universal*.sh`: universal/OOD runs.
- `train_learning_curve.sh`, `identification_speed.sh`: sample-efficiency
  experiments.
- `train_delayed_classifier.sh`: timing-jitter defense.
- `experiments/cross_harness/`: canonical two-harness identity, harness
  detection, and timing/non-timing ablations.

## Analysis and publication

- `plot_*.py`, `plot_*.sh`: figures.
- `make_tables.py`, `make_tables.sh`: tables.
- `add_xgb_explain.py`, `plot_shap_comparison.py`: feature attribution.
- `upload_to_hub.py`, `push_to_hub.sh`: explicit publication.

## Deferred physical moves

After all collectors are stopped and recovery state is no longer needed:

1. Move raw datasets/traces under a stable `data/` namespace.
2. Move campaign state and classifier outputs under `outputs/`.
3. Move SIF/definition files under `containers/`.
4. Move plotting code under `scripts/publication/`.
5. Update absolute paths embedded in frozen manifests or regenerate those
   manifests deliberately.

There will be no automatic stale-directory sweep; cleanup remains an explicit,
targeted operation.
