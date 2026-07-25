# Source workspace

Start with one of these entry points:

- Browser-use collection: `browser_use_campaign.py` with
  `browser_use_campaign.yaml`.
- Original collection: `orchestrator.py` with `config.yaml` or
  `multi_harness_config.yaml`.
- Two-harness experiments and ablations:
  `experiments/cross_harness/README.md`.
- Historical workflow index: `scripts/README.md`.
- Automated checks: `tests/`.

Data and generated state currently remain in `datasets/`, `traces/`,
`campaign_runs/`, and `artifacts/`. They are intentionally not moved while
collection/recovery may still depend on their paths. The post-collection move
order is documented in `scripts/README.md`.
