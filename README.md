# Known Actions: Agent Traces As SideChannels

Records Wikipedia and Amazon browsing traces from LLM agents and trains classifiers to predict which agent produced each trace from behavioral patterns alone — without looking at the content of what was searched or read.

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│  orchestrator.py --config custom_config.yaml                    │
│  loads registry (config.yaml) + experiment spec                 │
│  → spawns one subprocess per (agent × harness × question × rep) │
└───────────────────┬─────────────────────────────────────────────┘
                    │ apptainer exec --bind src:/app/workspace
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  agent.sif  (Playwright base image + Node deps + Chromium)      │
│                                                                 │
│  agent_runner.ts (MidScene) or browser_use_runner.py            │
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
│  ├── calls the selected harness with the task prompt            │
│  │     the LLM browses freely, following links across pages     │
│  │     on failure: partial trace saved with error field set     │
│  ├── single backstop harvest() for beforeunload edge cases      │
│  └── writes traces/{agent}/{dataset}/{harness}/{time}/{id}.json │
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
| `midscene_log` / `browser_use_log` | harness-specific high-level action log; never used by the behavioral classifiers |
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

New traces land in
`traces/{agent_id}/{dataset_name}/{harness}/{timestamp}/`. Existing
pre-harness traces in `traces/{agent_id}/{dataset_name}/{timestamp}/` remain
supported and are treated as MidScene episodes.

### Browser-use comparison traces

Use the browser-use harness to collect a second trace corpus without rerunning
MidScene:

```yaml
harnesses:
  - browser_use
```

[`src/multi_harness_config.yaml`](src/multi_harness_config.yaml) defines the
six-model browser-use collection over 2WikiMultiHopQA and WebShop. Existing
MidScene traces and trained classifiers are left untouched. The config selects
a separate image so the existing MidScene-only image is not replaced:

```bash
cd src
apptainer build agent.multi.sif agent.def
python orchestrator.py --config multi_harness_config.yaml --dry-run
python orchestrator.py --config multi_harness_config.yaml
```

Use `--agents qwen3_5_27b` to collect one model at a time when local vLLM
servers share a port. The harness filter is optional because this config
contains only `browser_use`. Resume state remains tracked independently by
harness, so existing MidScene traces are neither skipped nor overwritten.

The browser-use runner connects a passive Playwright observer to the same
Chromium process over CDP and injects the existing `page_tracer.js`. Thus both
harnesses produce the same website-visible event vocabulary. A task that
finishes incorrectly, reaches its step limit, or reports `task_success: false`
is still retained. Only traces without browser events, corrupt files, and
configured fatal API/service failures are excluded by the analyzer.

For labeled tasks, `task_success` is the ground-truth verification result;
browser-use's own completion judgement is stored separately as
`browser_use_log.agent_reported_success`. The configured 300-second task
budget stops agent actions, while `cleanup_grace_s` only allows the runner to
harvest events, save a partial trace, and close Chromium.

### Automated browser-use campaigns

`browser_use_campaign.py` manages the model-level lifecycle above the episode
orchestrator. It skips complete models, starts and health-checks owned vLLM
servers, resumes missing traces, and stops each owned server. Local models run
in the configured order Qwen3-VL-30B, Qwen3.5, Gemma, then GLM-4.6V;
OpenRouter models then run in the order GPT, Gemini, and Opus:

```bash
cd src
python browser_use_campaign.py --config browser_use_campaign.yaml --dry-run
python browser_use_campaign.py --config browser_use_campaign.yaml
```

Use `--only <agent_id> ...`, `--skip-local`, or `--skip-openrouter` to select
campaign phases. A resource-blocked local model is recorded in the campaign
manifest while later cloud phases continue. Exit code 2 means collection
stopped on a fatal API/credit error; exit code 3 means cloud work completed but
one or more local models remain resource-blocked. Rerunning the same command
resumes from valid traces already collected.

Each future browser-use trace records its prompt, cached-prompt, completion,
and total token counts under `browser_use_log.usage`. Campaign-level totals and
OpenRouter key-balance snapshots are written to the ignored
`campaign_runs/<campaign-id>/manifest.json`. Use a dedicated OpenRouter key so
balance deltas represent only this campaign.

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

### Cross-harness identity experiments

The six-model MidScene/browser-use experiment is managed separately from the
historical flat classifier runs:

```text
qwen3vl_30b_a3b  qwen3_5_27b  glm_4.6v
gpt_5_4          gemini_3_1    claude_opus_4_6
```

It uses frozen trace manifests so that within-harness, cross-harness, and
balanced mixed-harness evaluations use the same task universe. Legacy traces
without a harness directory are treated as MidScene; raw trace JSON is never
rewritten.

Audit collection coverage at any time (safe while collection is running):

```bash
cd src
python cross_harness_pipeline.py --config cross_harness_config.yaml audit
```

Once all six browser-use collections are complete, freeze the common task
universe:

```bash
python cross_harness_pipeline.py --config cross_harness_config.yaml prepare
```

`prepare` requires the configured minimum number of tasks that have a valid
trace for every agent under both harnesses. It refuses to silently replace a
different frozen manifest. Generated manifests and models live under
`src/artifacts/classifiers/multi_harness_identity/v1/`.

XGBoost is the primary classifier:

```bash
python cross_harness_pipeline.py --config cross_harness_config.yaml run-grid \
  --classifier XGBoost
```

The grid fits three policies per dataset (`midscene`, `browser_use`,
`mixed50`) and evaluates:

```text
midscene    → midscene, browser_use
browser_use → browser_use, midscene
mixed50     → mixed50, midscene, browser_use
```

The binary harness detector uses the same frozen paired task universe. Each
leave-one-model-out fold trains on both harnesses from the remaining models
and tests on both harnesses from the entirely unseen held-out model:

```bash
python cross_harness_pipeline.py --config cross_harness_config.yaml \
  harness-detector --classifier XGBoost
```

While a final model is still collecting, the explicitly provisional
five-model config can exercise the complete CPU path without writing into the
final six-model artifact namespace:

```bash
python cross_harness_pipeline.py \
  --config cross_harness_config.provisional.yaml prepare

python cross_harness_pipeline.py \
  --config cross_harness_config.provisional.yaml run-grid \
  --classifier XGBoost --quick --xgb-device cpu

python cross_harness_pipeline.py \
  --config cross_harness_config.provisional.yaml harness-detector \
  --classifier XGBoost --quick --xgb-device cpu
```

For pipeline validation without taking a GPU from trace collection, use the
small CPU Random Forest path on one cell:

```bash
python cross_harness_pipeline.py --config cross_harness_config.yaml train \
  --dataset 2wikimultihop --train-policy midscene \
  --classifier RandomForest --quick

python cross_harness_pipeline.py --config cross_harness_config.yaml evaluate \
  --dataset 2wikimultihop --train-policy midscene \
  --eval-policy browser_use --classifier RandomForest
```

LSTM is supported as the next-priority classifier. `--quick` keeps its smoke
fit and evaluation on CPU; full LSTM and the default XGBoost configuration use
a GPU and should wait until collection releases one.

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
| `_train` | Fitting RF, GB, LSTM |
| `_val` | Early stopping (LSTM), hyperparameter selection |
| `_test` | Final held-out in-domain evaluation |
| OOD | Cross-domain evaluation — never seen during training |

Results are saved to `traces/models/{tag}/results.json` and include `test_report` and `ood_report` for each model.

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
│   │   ├── midscene/
│   │   │   └── 20260404_090000/
│   │   │       └── qwen3vl_midscene_a1b2c3d4.json
│   │   └── browser_use/
│   │       └── 20260404_090000/
│   │           └── qwen3vl_browser_use_e5f6a7b8.json
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
OPEN_ROUTER_API_KEY=sk-or-...
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
├── multi_harness_config.yaml  Balanced MidScene + browser-use experiment
├── agent.def               Apptainer build spec
├── agent.sif               Built container image (generated)
├── package.json            Node deps baked into the image
├── page_tracer.js          IIFE injected into every page; records DOM events
├── agent_runner.ts         One agent episode end-to-end (TypeScript)
├── browser_use_runner.py   One browser-use episode + passive CDP trace observer
├── browser_use_campaign.py Model server lifecycle + resumable campaign runner
├── browser_use_campaign.yaml Local/OpenRouter campaign and GPU configuration
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
