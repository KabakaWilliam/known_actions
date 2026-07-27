# Known Actions: Agent Traces As SideChannels

Records Wikipedia and Amazon browsing traces from LLM agents and trains classifiers to predict which agent produced each trace from behavioral patterns alone — without looking at the content of what was searched or read.

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│  orchestrator.py --config custom_config.yaml                    │
│  loads registry (config.yaml) + experiment spec                 │
│  → spawns one Apptainer subprocess per (agent × question × rep) │
└───────────────────┬─────────────────────────────────────────────┘
                    │ apptainer exec --bind src:/app/workspace
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  agent.sif  (Playwright base image + Node deps + Chromium)      │
│                                                                 │
│  agent_runner.ts                                                │
│  ├── launches headless Chromium via Playwright                  │
│  ├── exposeFunction(__pushTraceEvent) — events pushed from page │
│  │     to Node.js over CDP in real-time (no polling)           │
│  ├── registers framenavigated listener (HTTP navigation capture)│
│  ├── injects page_tracer.js into every page load                │
│  │     records: clicks · scrolls · keypresses · focus          │
│  │               navigations · beforeunload (scroll at exit)   │
│  │     (no search terms, no content — behavior only)           │
│  ├── navigates to START_URL (default: en.wikipedia.org)         │
│  │     waitUntil: 'load' — compatible with ad-heavy sites      │
│  ├── calls MidScene aiAct() with the task prompt                │
│  │     the LLM browses freely, following links across pages     │
│  │     on failure: partial trace saved with error field set     │
│  ├── single backstop harvest() for beforeunload edge cases      │
│  └── writes  traces/{agent_id}/{dataset}/{timestamp}/{id}.json  │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  trace_analyzer.py                                              │
│  ├── Random Forest  on 29 hand-crafted behavioral features      │
│  ├── Gradient Boosting on the same 29 features                  │
│  └── LSTM on raw event-token sequences (7 token types)          │
│       + RF features injected at final hidden state              │
│       + 4 per-event continuous scalars (timing, position)       │
│  evaluated on held-out test split + optional OOD domain         │
│  outputs: traces/models/{tag}/classifier.pkl · lstm_model.pt    │
└─────────────────────────────────────────────────────────────────┘
```

## Results

XGBoost classifier performance across 14 agents using behavioral features alone, without access to question text or page content.

| Train | Test | Website | Setting | Macro F1 |
|---|---|---|---|---|
| 2WikiMultiHopQA | 2WikiMultiHopQA | Wikipedia | in-domain | 79.4 |
| FRAMES | FRAMES | Wikipedia | in-domain | 75.3 |
| WebShop | WebShop | Amazon | in-domain | 74.3 |
| DeepShop | DeepShop | Amazon | in-domain | 72.6 |
| 2WikiMultiHopQA | FRAMES | Wikipedia | cross-task | 41.1 |
| FRAMES | 2WikiMultiHopQA | Wikipedia | cross-task | 49.8 |
| 2WikiMultiHopQA + FRAMES | 2WikiMultiHopQA test | Wikipedia | pooled-site | 81.3 |
| 2WikiMultiHopQA + FRAMES | FRAMES test | Wikipedia | pooled-site | 77.2 |
| WebShop | DeepShop | Amazon | cross-benchmark | 52.7 |
| DeepShop | WebShop | Amazon | cross-benchmark | 55.9 |
| WebShop + DeepShop | WebShop test | Amazon | pooled-site | 78.8 |
| WebShop + DeepShop | DeepShop test | Amazon | pooled-site | 70.9 |
| Wikipedia pooled | Amazon test | cross-site | cross-site | 29.7 |
| Amazon pooled | Wikipedia test | cross-site | cross-site | 26.0 |

Single-task transfer across tasks on the same site is substantially weaker than in-domain attribution, but pooling multiple tasks from the same website recovers strong performance. Cross-site transfer remains weak, suggesting that behavioural fingerprints are site-conditioned rather than universal.

### What is an episode?

An **episode** is one complete browsing session: one agent, one question/task, navigating from scratch until complete. `episodes_per_combo` controls how many times the same *(agent × question)* pair is repeated — because LLM browsing is stochastic, multiple reps capture behavioral variance and give the classifier more training data.

Each episode produces one `.json` trace file:

| Field | Contents |
|---|---|
| `meta` | model name, agent id, dataset name, question, start_url, task_type, timestamp |
| `result` | the agent's answer, confidence, sources cited — `null` if `aiAct` failed |
| `error` | error message string if `aiAct` threw, otherwise `null` |
| `midscene_log` | high-level MidScene action log — what the LLM decided to do |
| `dom_trace` | low-level DOM events — clicks, scrolls, keypresses, navigations |

The classifiers see only `dom_trace` events — they never see the question text, search terms, or page content.

## DOM event types

`page_tracer.js` is injected into every page load as an `addInitScript`. It records only what a website's own JS analytics (or server logs) could observe — no content leakage.

| Event type | Source | Key fields |
|---|---|---|
| `navigate` `trigger:"http"` | Playwright `framenavigated` (Node.js side) | `url`, `t_episode` |
| `navigate` `trigger:"pushState"` | `history.pushState` patch | `url`, `to`, `t_episode` |
| `navigate` `trigger:"popstate"` | `window popstate` listener | `url`, `to`, `t_episode` |
| `click` | `document click` (capture phase) | `x`, `y`, `target_tag`, `target_text` (≤100 chars), `target_id`, `target_class`, `href` |
| `focus` | `document focus` (capture phase), inputs only | `target_tag`, `target_id`, `target_name` |
| `keydown` | `document keydown` (capture phase) | `key` — structural keys verbatim (`Enter`, `ArrowDown`, …); printable chars → `"char"` |
| `scroll` | `document scroll`, debounced 200ms | `scrollY`, `docHeight`, `pct` (0–100) |
| `beforeunload` | `window beforeunload` | `scrollY`, `docHeight`, `pct`, `leaving_href` |

### Timeline anchoring

Each page reload reinitialises the in-page timer (`t` restarts at 0). `agent_runner.ts` stamps `t_episode = Date.now() - episodeStart` on every event when it arrives via the `__pushTraceEvent` CDP bridge — giving a single monotonic timeline across all pages in a session without any per-page offset arithmetic.

HTTP navigation events are captured directly by Playwright's `framenavigated` listener (not by the in-page script), so they are never lost to page reloads.

## Behavioral features

`trace_analyzer.py` extracts 29 features per episode for the Random Forest / Gradient Boosting classifiers. The LSTM operates on the raw token sequence (7 token types, including `focus`) augmented with per-event continuous scalars and RF features injected at the final hidden state.

### Volume

| Feature | Description |
|---|---|
| `n_clicks` | Total click events |
| `n_scrolls` | Total scroll events (debounced, one per 200ms window) |
| `n_navigations` | Total navigate events (HTTP + pushState + popstate) |
| `n_keydowns` | Total keydown events |
| `n_beforeunload` | Page exit events fired (equals page_count − 1 when capture is complete) |
| `n_focus` | Input/textarea focus events (search box interactions) |
| `n_events_total` | All DOM events combined |
| `n_midscene_actions` | High-level MidScene planning steps (from `midscene_log`) |
| `page_count` | Unique pages visited (counted from HTTP navigate events) |

### Timing

All timing uses `t_episode` — monotonic milliseconds from episode start.

| Feature | Description |
|---|---|
| `total_duration_s` | Wall time from first to last event (seconds) |
| `mean_iei_ms` | Mean inter-event interval across all events |
| `std_iei_ms` | Std dev of inter-event intervals |
| `median_iei_ms` | Median inter-event interval |
| `p10_iei_ms` | 10th-percentile inter-event interval (burst speed) |
| `p90_iei_ms` | 90th-percentile inter-event interval (idle gaps) |

### Scroll

| Feature | Description |
|---|---|
| `max_scroll_pct` | Deepest scroll reached on any page (0–100) |
| `mean_scroll_pct` | Mean scroll depth across all scroll events |
| `n_deep_scrolls` | Events with `pct > 60` (agent read past the fold) |
| `scroll_reversals` | Direction changes in scroll sequence (up/down alternations) |
| `mean_exit_scroll_pct` | Mean scroll depth recorded at `beforeunload` — how deep the agent was when leaving each page |

### Clicks

| Feature | Description |
|---|---|
| `click_x_std` | Std dev of click X positions (targeting spread) |
| `click_y_std` | Std dev of click Y positions |
| `n_link_clicks` | Clicks that had an `href` (outbound link clicks) |
| `link_click_ratio` | `n_link_clicks / n_clicks` |

### Navigation ratios

| Feature | Description |
|---|---|
| `actions_per_page` | `n_events_total / page_count` |
| `nav_to_click_ratio` | `n_navigations / n_clicks` |
| `keydowns_per_page` | `n_keydowns / page_count` (search-box typing frequency) |
| `midscene_per_page` | `n_midscene_actions / page_count` |
| `focus_per_page` | `n_focus / page_count` (how often the agent re-focuses the search box per page) |

### LSTM sequence features

Each event in the sequence includes 4 per-event continuous scalars alongside the token type:

| Scalar | Description |
|---|---|
| `log1p(delta_t)` | Log inter-event gap in ms |
| `log1p(t_episode)` | Log absolute time since episode start |
| `scroll_pct / 100` | Scroll depth normalized to [0, 1] (scroll/beforeunload events; 0 otherwise) |
| `x / 1280` or `y / 800` | Normalized click position (click events; 0 otherwise) |

At the final hidden state, all 29 aggregate RF features (StandardScaler-normalized) are concatenated before the classification head.

## Configuration

The system uses two config files:

| File | Role |
|---|---|
| [`config.yaml`](config.yaml) | **Registry** — defines all available agents and dataset loaders. Edit once when adding a new model or data source. |
| [`custom_config.yaml`](custom_config.yaml) | **Experiment spec** — selects which agents and datasets, sets run parameters. One file per experiment. |

### Running an experiment

```bash
# Prep any non-builtin datasets (one-time per split)
python prep_datasets.py --config custom_config.yaml

# Run the full experiment
python orchestrator.py --config custom_config.yaml
```

That's it. The experiment config is the single source of truth for everything in the run.

### Experiment config format (`custom_config.yaml`)

```yaml
run:
  episodes_per_combo: 3   # reps per (agent × question) pair
  timeout_s: 300          # per-episode timeout
  workers: 5              # parallel episodes; 1 = serial

agents:
  - agent_id: qwen3vl_8b  # resolved from config.yaml registry
  - agent_id: gpt54
    env:                  # optional per-experiment overrides
      MIDSCENE_REPLANNING_CYCLE_LIMIT: "15"

datasets:
  - name: custom
    source: builtin       # qa_dataset.py — no prep needed

  - name: 2wikimultihop_val
    source: datasets/2wikimultihop_val.json
    hf_dataset: 2wikimultihop   # key in config.yaml dataset_loaders
    split: validation
    n_questions: 50
    seed: 42
```

Traces land in `traces/{agent_id}/{dataset_name}/{timestamp}/`.

### Shopping domain config

For Amazon/WebShop-style tasks, add `start_url`, `task_type`, and `task_prompt_template` to the dataset entry:

```yaml
datasets:
  - name: webshop_train
    source: datasets/webshop_train.json
    hf_dataset: webshop_goals
    n_questions: 150
    seed: 42
    offset: 0
    start_url: "https://www.amazon.com"
    task_type: shop
    task_prompt_template: |
      You are a shopping assistant. Search Amazon to find a product matching this description:
      "{question}"
      Browse freely. Search, filter, view product pages, and add promising items to your cart.
      You may remove items from your cart if you find better alternatives.
      When done, your cart should contain only the product(s) that best match the description.
      Do NOT proceed to checkout or log in.
```

`{question}` in the template is interpolated with the episode's question at runtime.

### Dataset splits

Each split is a separate entry with its own `name` — this keeps train/val/test traces in separate directories without any extra code:

```yaml
datasets:
  - name: 2wikimultihop_train
    source: datasets/2wikimultihop_train.json
    hf_dataset: 2wikimultihop
    split: train
    n_questions: 200
    seed: 42

  - name: 2wikimultihop_val
    source: datasets/2wikimultihop_val.json
    hf_dataset: 2wikimultihop
    split: validation
    n_questions: 50
    seed: 42
```

`trace_analyzer.py` assigns traces to train/val/test/OOD buckets based on the dataset name suffix (`_train`, `_val`, `_test`, `_ood`).

### Non-overlapping WebShop splits

WebShop uses a single shuffled pool. The `offset` field ensures no question appears in two splits:

```yaml
# seed=42 shuffles once; offset slices into non-overlapping windows
- name: webshop_train   # offset=0,   n=150 → items 0–149
- name: webshop_val     # offset=150, n=75  → items 150–224
- name: webshop_test    # offset=225, n=75  → items 225–299
```

## Training classifiers

```bash
# Train on Wikipedia traces only
python trace_analyzer.py --datasets 2wikimultihop --tag wiki

# Train on Wikipedia; evaluate Amazon as OOD
python trace_analyzer.py \
    --datasets 2wikimultihop webshop \
    --ood-datasets webshop \
    --tag wiki_ood_amazon

# Train on Amazon; evaluate DeepShop as OOD
python trace_analyzer.py \
    --datasets webshop deepshop \
    --ood-datasets deepshop \
    --tag webshop
```

### CLI reference

| Argument | Description |
|---|---|
| `--traces-dir PATH` | Root traces directory (default: `./traces`) |
| `--datasets NAME [NAME …]` | Dataset base names to include (all suffixes: `_train/_val/_test/_ood`) |
| `--ood-datasets NAME [NAME …]` | Force these base names into the OOD bucket regardless of suffix |
| `--tag TAG` | Output subdirectory name under `traces/models/`; auto-derived from train datasets if omitted |

`--datasets` without `--ood-datasets`: suffix-based split assignment (`_train` → train, `_val` → val, `_test` → test, `_ood` → OOD).

`--ood-datasets webshop`: all webshop traces go to OOD regardless of whether they're named `_train/_val/_test`.

### Train/val/test/OOD workflow

| Split bucket | Used for |
|---|---|
| `_train` | Fitting and sklearn training-fold cross-validation |
| `_val` | LSTM hyperparameter selection and validation reports |
| `_test` | Final held-out in-domain evaluation |
| OOD | Cross-domain evaluation — never seen during training |

Results are saved to `traces/models/{tag}/results.json` and include `test_report` and `ood_report` for each model.

### Closed-set macro-F1 confidence intervals

Closed-set evaluation uses 10 randomly generated classifier seeds by default
and reports seed-averaged macro-F1 with a percentile bootstrap confidence
interval over held-out test traces:

```bash
python trace_analyzer.py \
    --traces-dir ./traces \
    --train-datasets 2wikimultihop \
    --tag wiki_ood_all \
    --classifiers XGBoost \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 2026
```

Ten seeds provide a useful estimate of run-to-run variation without the steep
cost of 20 or more full refits. The exact generated seeds are recorded in the
output. To supply fixed seeds instead, use
`--classifier-seeds 42 43 44 45 46`; at least five unique seeds are required.
Use `--classifier-seed-count N` to change the generated count, or
`--no-closed-set-stats` to disable the multi-seed evaluation.
`--load-classifier` remains evaluation-only and therefore skips multi-seed
refits; run without it to produce closed-set seed statistics.

Hyperparameters are selected once using cross-validation within the training
split, then held fixed while the classifier is refit for each seed. The
validation split remains a separate evaluation. Each bootstrap replicate
resamples the held-out traces with replacement, uses the same sampled trace
indices for all classifier seeds, computes per-model F1 and macro-F1 for each
seed, and averages across seeds. The output reports the seed-averaged estimate,
trace-bootstrap interval, per-seed scores, and seed standard deviation for each
agent/model class under `macro_f1.per_class`, as well as the overall macro-F1
summary.

These statistics are written separately to
`traces/classifiers/{tag}/closed_set_macro_f1.json`; the existing
`results.json` format is unchanged.

Run all four canonical XGBoost evaluations sequentially with
`src/run_closed_set_stats.sh`. The aggregate output from the completed run is
saved as `src/closed_set_macro_f1_results.json`.

Pass that aggregate file to `plot_main_results.py` or `plot_hero_plot.py` with
`--closed-set-stats` to render one asymmetric bootstrap error bar for each
agent/model rather than a dataset-level interval band.

### Open-set AUROC confidence intervals

The leave-one-model-out open-set bars use the same 10-seed default. For each
held-out model, `run_open_set_stats.py` refits XGBoost once per seed using the
hyperparameters already selected in that run's `results.json`. It therefore
avoids repeating hyperparameter search and leaves every existing
`results.json` unchanged:

```bash
./run_open_set_stats.sh \
    --traces-dir ./traces \
    --datasets wiki frames webshop deepshop \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 2026 \
    --aggregate-output open_set_auroc_results.json
```

If `--classifier-seeds` is omitted, 10 unique seeds are generated with system
randomness once and shared across the entire batch. The exact seeds are
persisted in `traces/classifiers/open_set_auroc_batch.json` for safe resume and
recorded in every output; pass at least five explicit values to reproduce a
particular seed set. Ten is the recommended default because it captures useful
run-to-run variation while keeping the 56 leave-one-model-out evaluations
tractable.

Each bootstrap replicate independently resamples the known test traces and the
held-out model's unknown traces with replacement. The same pair of resamples is
then used for all classifier seeds, AUROC is computed separately for every
seed, and those AUROCs are averaged. This is a stratified paired bootstrap over
evaluation traces; classifier-seed standard deviation is reported separately.
The legacy open-set population is preserved: known samples come from the test
split, while the unknown population contains all available traces for the
held-out model.

Per-run statistics are written separately to
`traces/classifiers/{open_set_tag}/open_set_loo_{model}/open_set_auroc.json`.
The combined result is `open_set_auroc_results.json`.

Use the aggregate to render the figure variants with one interval per held-out
model:

```bash
python plot_main_results.py \
    --traces-dir ./traces \
    --closed-set-stats closed_set_macro_f1_results.json \
    --open-set-stats open_set_auroc_results.json

python plot_open_set_summary.py \
    --traces-dir ./traces \
    --plot xgb_strip \
    --open-set-stats open_set_auroc_results.json
```

### Progressive pooled open-set holdouts

The progressive experiment reruns the one-model holdouts and adds two- and
three-model holdouts under a corrected, directly comparable protocol. For each
subset, only valid test traces are used for evaluation: the held-in models form
the known population and all held-out models are pooled into one binary
unknown class. The classifier is trained on held-in training traces, and each
subset selects among the same eight fixed XGBoost candidates using macro-F1 on
held-in validation traces only; unknown and test traces never affect tuning.

2WikiMultiHop and WebShop retain their recorded directory splits. FRAMES and
DeepShop instead use a fixed SHA-256 assignment of exact question groups to
50/25/25 train/validation/test splits, so every model's traces for a question
remain in the same split (FRAMES is capped at 150/75/75 traces per model).
Because the evaluation population differs from the earlier leave-one-out
analysis, `k=1` is rerun alongside `k=2` and `k=3`.

With 14 models there are `C(14, k)` possible subsets. The default design
evaluates all 14 singletons and all 91 pairs, then takes a deterministic,
model-balanced sample of 100 of the 364 triples. Increase
`--max-subsets-per-size` to evaluate more triples. Ten classifier seeds are
generated once, saved in the batch manifest, and shared across every dataset
and subset.

The resource-safe runner defaults numerical libraries to one thread,
checkpoints every subset atomically, and resumes completed subsets when the
same command is rerun. From the repository root, run:

```bash
src/run_open_set_multi_stats.sh \
    --traces-dir ./traces \
    --model-universe-stats src/open_set_auroc_results.json \
    --datasets wiki frames webshop deepshop \
    --holdout-sizes 1 2 3 \
    --max-subsets-per-size 100 \
    --subset-seed 2026 \
    --classifier-seed-count 10 \
    --tuning-candidates 8 \
    --tuning-seed 42 \
    --bootstrap-replicates 10000 \
    --bootstrap-confidence 0.95 \
    --bootstrap-seed 2026 \
    --device cuda \
    --n-jobs 1 \
    --work-dir src/open_set_multi_checkpoints \
    --aggregate-output src/open_set_multi_holdout_auroc_results.json
```

Each held-out subset receives its own pointwise 95% percentile CI by
resampling known and pooled-unknown test traces within their strata, with the
same bootstrap draws paired across classifier seeds. These are intervals for
the individual subsets, not for the mean across subsets. The combined result
is `src/open_set_multi_holdout_auroc_results.json`.

Generate the progression and ranked-subset figure variants with:

```bash
python src/plot_open_set_holdout_progression.py \
    --stats src/open_set_multi_holdout_auroc_results.json \
    --out-dir src/figures \
    --format both
```

This writes
`src/figures/open_set_holdout_progression_bootstrap_ci.{png,pdf}` and
`src/figures/open_set_holdout_ranked_bootstrap_ci.{png,pdf}`. Every plotted
whisker is an individual subset's confidence interval; the progression plot's
median and IQR summarize subsets descriptively and are not confidence bounds.

## Generating LaTeX tables

```bash
python make_tables.py                    # reads ./traces/models/
python make_tables.py /path/to/traces    # custom traces dir
```

Reads `wiki_ood_amazon/results.json` and `webshop/results.json` and writes:
- `traces/models/table_main.tex` — accuracy + macro F1 across all four conditions
- `traces/models/table_per_class.tex` — per-agent F1 breakdown by model and condition

## Trace directory layout

```
traces/
├── qwen3vl_8b/
│   ├── 2wikimultihop_train/
│   │   └── 20260404_090000/
│   │       ├── qwen3vl_a1b2c3d4.json
│   │       └── ...
│   ├── 2wikimultihop_val/
│   │   └── ...
│   └── webshop_train/
│       └── ...
├── gpt54/
│   └── ...
└── models/
    ├── wiki_ood_amazon/
    │   ├── classifier.pkl     (RF + GB + LabelEncoder + feature names + StandardScaler)
    │   ├── lstm_model.pt
    │   └── results.json
    └── webshop/
        ├── classifier.pkl
        ├── lstm_model.pt
        └── results.json
```

## Setup

### 1. Build the container (once, ~5–10 min)

```bash
apptainer build agent.sif agent.def
```

### 2. Install Python dependencies

```bash
/opt/anaconda/envs/dispatch/bin/pip install -r requirements.txt
```

### 3. Configure API keys in `.env`

```bash
VLLM_API_KEY=test_away      # any non-empty string; vLLM accepts anything
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## Quick-start

```bash
# 1. (If using external datasets) Download and save them
python prep_datasets.py --config custom_config.yaml

# 2. Preview — print commands without executing
python orchestrator.py --config custom_config.yaml --dry-run

# 3. Run
python orchestrator.py --config custom_config.yaml

# 4. Train classifiers once enough traces exist (15+ recommended)
python trace_analyzer.py --datasets 2wikimultihop --tag wiki

# 5. Generate LaTeX tables
python make_tables.py
```

## Adding a new agent

1. Add an entry to the `agents:` section in [`config.yaml`](config.yaml) with the model's env vars.
2. Reference it by `agent_id` in your experiment config's `agents:` list.
3. If it's a local model, start the vLLM server first.

## Adding a new dataset

1. Add a loader entry to `dataset_loaders:` in [`config.yaml`](config.yaml):
   ```yaml
   dataset_loaders:
     hotpotqa:
       hf_repo: "hotpot_qa"
   ```
2. Add a loader function to `LOADERS` in [`prep_datasets.py`](prep_datasets.py) if the data format differs from the default (question/answer fields, filtering logic).
3. Reference it in your experiment config:
   ```yaml
   datasets:
     - name: hotpotqa_val
       source: datasets/hotpotqa_val.json
       hf_dataset: hotpotqa
       split: validation
       n_questions: 50
       seed: 42
   ```

## Serving local models

Local models are served via vLLM. Apptainer shares the host network namespace, so `http://127.0.0.1:<port>` is reachable from inside the container.

```bash
bash serve_vllm.sh   # Qwen3-VL-8B on GPU 2, port 3030
```

Verify the model name matches what vLLM reports — this must match `MIDSCENE_MODEL_NAME` in the registry:

```bash
curl http://127.0.0.1:3030/v1/models | python3 -m json.tool
```

### Planned local models

| agent_id | Model | Port | MidScene family |
|---|---|---|---|
| `qwen3vl_8b` | Qwen3-VL-8B-Instruct | 3030 | `qwen3-vl` |
| `qwen25vl_7b` | Qwen2.5-VL-7B-Instruct | 3031 | `qwen2.5-vl` |
| `qwen35_7b` | Qwen3.5-VL-7B-Instruct | 3032 | `qwen3.5` |
| `uitars_7b` | UI-TARS-7B-SFT | 3033 | `vlm-ui-tars` |

## Tuning MidScene behavior

Set in `config.yaml` (`midscene_defaults:`) or override per-agent in your experiment config's `env:` block:

| Setting | Effect |
|---|---|
| `MIDSCENE_REPLANNING_CYCLE_LIMIT` | Max planning cycles before giving up. Default: 40 |

## File layout

```
src/
├── config.yaml             Registry: agents, dataset loaders, MidScene defaults
├── custom_config.yaml      Experiment spec (edit this to define a run)
├── agent.def               Apptainer build spec
├── agent.sif               Built container image (generated)
├── package.json            Node deps baked into the image
├── page_tracer.js          IIFE injected into every page; records DOM events
├── agent_runner.ts         One agent episode end-to-end (TypeScript)
├── orchestrator.py         Drives all episodes via subprocess
├── prep_datasets.py        Download + standardize HuggingFace/local datasets
├── qa_dataset.py           10 built-in curated multi-hop questions
├── trace_analyzer.py       Feature extraction + RF + Gradient Boosting + LSTM
├── make_tables.py          Generate NeurIPS-ready LaTeX tables from results.json
├── serve_vllm.sh           Launch Qwen3-VL-8B on port 3030
├── requirements.txt        Python deps
├── web_shop_goals.json     ~12k WebShop shopping goals (local copy)
├── .env                    API keys (never committed)
├── datasets/               Prepared question files (generated by prep_datasets.py)
│   ├── 2wikimultihop_val.json
│   ├── webshop_train.json
│   └── ...
└── traces/
    ├── {agent_id}/
    │   └── {dataset_name}/
    │       └── {YYYYMMDD_HHMMSS}/
    │           └── {episode_id}.json
    └── models/
        └── {tag}/
            ├── classifier.pkl   (RF, GB, LabelEncoder, feature names, StandardScaler)
            ├── lstm_model.pt
            └── results.json
```

<!-- grep -rl "Error: failed to call AI model service" src/traces/ | awk -F'/' '{print $3"/"$4}' | sort | uniq -c | sort -nr -->
<!-- to delete: grep -rl "Error: failed to call AI model service" src/traces/claude_opus_4_6/frames_test/ | xargs rm -->
<!-- tar --exclude='*.pt' --exclude='*.pkl' -cf - traces/ | xz -9e > traces_under_100mb.tar.xz -->
