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
- `scripts/run_experiments_simple.sh`: preferred human-readable launcher. Set
  GPU numbers in its SETTINGS block, then comment/uncomment named experiments;
  every selection is documented with its train/test protocol and purpose. It
  includes cross-harness, policy-normalization, and leave-p-models-out open-set
  analysis.
- `scripts/run_experiment_queue.sh`: commentable, two-GPU queue for the final
  six-model XGBoost grid/ablations/harness detector and the four-model
  policy-normalization analysis. It excludes incomplete SGLang analysis.
  Select one family without editing using `EXPERIMENT_QUEUE=cross` or
  `EXPERIMENT_QUEUE=policy`; use `SKIP_PREPARE=1` only after a successful
  preparation pass.
- `scripts/run_followup_experiments.sh`: commentable launcher for the
  14-model MidScene model-arrival update curve and the matched vLLM↔SGLang
  WebShop grid. It also contains the 2-to-14-model closed-set scaling curve and
  the leave-one-to-four-models-out open-set experiment. Each selection explains
  the reviewer question it answers.
- `scripts/run_temporal_generalization.sh`: commentable launcher for the
  test-only MidScene future-wave collection and frozen old-to-new WebShop
  temporal-generalization analysis.

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
