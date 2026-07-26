# Full-WebShop behavioral-policy normalization defense

This experiment applies the normalized browsing policy to every task in the
official WebShop train, validation, and test files for four locally served
models:

- `qwen3_5_27b`
- `qwen3vl_30b_a3b`
- `gemma_4_26B_A4B_it`
- `glm_4.6v`

Each model contributes 150 train, 75 validation, and 75 test traces per
condition. The normalized condition therefore contains 1,200 traces, matched
task-for-task to 1,200 existing canonical browser-use controls.

The defended condition gives every model the same fixed browsing procedure:
search concisely, inspect results in order, verify constraints, avoid
unnecessary revisits, take one action at a time, and stop after carting the
best match. Treatment traces are isolated under
`traces_experiments/webshop_full_policy_normalization_4model_v1/`; canonical
controls remain untouched under `traces/`.

From `src/`, inspect the campaign without starting a server:

```bash
python browser_use_campaign.py \
  --config experiments/policy_normalization/configs/webshop_full_campaign.yaml \
  --skip-openrouter --dry-run
```

Collect sequentially through the configured local models:

```bash
python browser_use_campaign.py \
  --config experiments/policy_normalization/configs/webshop_full_campaign.yaml \
  --skip-openrouter
```

Collection is resumable by task identity. Traces collected by the retired
30/15/15 pilot are members of these complete dataset files and count toward
the 150/75/75 targets when their prompt and metadata validate.

GLM-4.6V runs last with tensor parallelism 4 on GPUs 0, 1, 2, and 3. The
campaign skips the three already-complete models before performing GPU checks.

Once all conditions are complete, freeze the matched task intersection:

```bash
python -m experiments.policy_normalization.pipeline audit
python -m experiments.policy_normalization.pipeline prepare
```

Run the XGBoost matrix with five seeds, bootstrap confidence intervals, and
all three feature views:

```bash
python -m experiments.policy_normalization.pipeline run-grid \
  --xgb-device cpu
```

The matrix reports canonical→canonical, canonical→normalized-policy,
normalized-policy→normalized-policy, and mixed50→each condition. Outputs are
isolated under
`artifacts/experiments/defenses/webshop_full_policy_normalization_4model_v1/`.

Task-success rates must be reported beside identification performance so that
classifier degradation is not mistaken for a useful defense when task utility
has collapsed.

Every `run-grid` and `summarize` invocation regenerates the top-level
`REPORT.md`. Start there: it labels the fixed-attacker, defense-aware, and
mixed-attacker comparisons; separates five-seed headline results from
single-seed ablations; includes per-model behavior and task utility; and links
back to every machine-readable summary.
