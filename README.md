# Known Actions: Agent Trace Recorder

Records Wikipedia browsing traces from LLM agents and trains classifiers to predict which agent produced each trace from behavioral patterns alone — without looking at the content of what was searched or read.

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
│  ├── registers framenavigated listener (HTTP navigation capture)│
│  ├── injects page_tracer.js into every page load                │
│  │     records: clicks · scrolls · keypresses · navigations    │
│  │               beforeunload (scroll depth at page exit)       │
│  │     (no search terms, no content — behavior only)           │
│  ├── navigates to en.wikipedia.org                              │
│  ├── calls MidScene aiAct() with the question                   │
│  │     the LLM browses freely, following links across pages     │
│  ├── polls __agentTrace every 100ms; adds t_episode (monotonic) │
│  └── writes  traces/{agent_id}/{dataset}/{timestamp}/{id}.json  │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  trace_analyzer.py                                              │
│  ├── Random Forest  on 27 hand-crafted behavioral features      │
│  └── LSTM           on raw event-token sequences (6 token types)│
│  both evaluated with 5-fold stratified cross-validation         │
│  outputs: traces/models/classifier.pkl · lstm_model.pt          │
└─────────────────────────────────────────────────────────────────┘
```

### What is an episode?

An **episode** is one complete browsing session: one agent, one question, navigating Wikipedia from scratch until it has an answer. `episodes_per_combo` controls how many times the same *(agent × question)* pair is repeated — because LLM browsing is stochastic, multiple reps capture behavioral variance and give the classifier more training data.

Each episode produces one `.json` trace file:

| Field | Contents |
|---|---|
| `meta` | model name, agent id, dataset name, question, timestamp |
| `result` | the agent's answer, confidence, Wikipedia sources cited |
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
| `keydown` | `document keydown` (capture phase) | `key` — structural keys verbatim (`Enter`, `ArrowDown`, …); printable chars → `"char"` |
| `scroll` | `document scroll`, debounced 200ms | `scrollY`, `docHeight`, `pct` (0–100) |
| `beforeunload` | `window beforeunload` | `scrollY`, `docHeight`, `pct`, `leaving_href` |

### Timeline anchoring

Each Wikipedia page reload reinitializes the in-page timer. `page_tracer.js` records `startTimeWall` (absolute wall-clock at page init). `agent_runner.ts` reads this on each navigation to compute a per-page offset, then adds `t_episode = pageEpochOffset + e.t` to every event — giving a single monotonic timeline across all pages in a session.

HTTP navigation events are captured directly by Playwright's `framenavigated` listener (not by the in-page script), so they are never lost to page reloads.

## Behavioral features

`trace_analyzer.py` extracts 27 features per episode for the Random Forest / Gradient Boosting classifiers. The LSTM operates on the raw token sequence (6 token types).

### Volume

| Feature | Description |
|---|---|
| `n_clicks` | Total click events |
| `n_scrolls` | Total scroll events (debounced, one per 200ms window) |
| `n_navigations` | Total navigate events (HTTP + pushState + popstate) |
| `n_keydowns` | Total keydown events |
| `n_beforeunload` | Page exit events fired (equals page_count − 1 when capture is complete) |
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

agents:
  - agent_id: qwen3vl     # resolved from config.yaml registry
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

`trace_analyzer.py` recurses the whole `traces/` tree, so all splits are pooled for classifier training automatically. If you want to evaluate on one split only, filter by path.

## Trace directory layout

```
traces/
├── qwen3vl/
│   ├── custom/
│   │   └── 20260404_090000/
│   │       ├── qwen3vl_a1b2c3d4.json
│   │       └── ...
│   └── 2wikimultihop_val/
│       └── 20260404_093000/
│           ├── qwen3vl_f1a2b3c4.json
│           └── ...
├── gpt54/
│   └── 2wikimultihop_val/
│       └── ...
└── models/
    ├── classifier.pkl
    └── lstm_model.pt
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
python trace_analyzer.py
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
| `qwen3vl` | Qwen3-VL-8B-Instruct | 3030 | `qwen3-vl` |
| `qwen25vl` | Qwen2.5-VL-7B-Instruct | 3031 | `qwen2.5-vl` |
| `qwen35` | Qwen3.5-VL-7B-Instruct | 3032 | `qwen3.5` |
| `uitars` | UI-TARS-7B-SFT | 3033 | `vlm-ui-tars` |

## Tuning MidScene behavior

Set in `config.yaml` (`midscene_defaults:`) or override per-agent in your experiment config's `env:` block:

| Setting | Effect |
|---|---|
| `MIDSCENE_REPLANNING_CYCLE_LIMIT` | Max planning cycles before giving up. Smaller models (8B) need more — default 20, local models use 30. |

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
├── prep_datasets.py        Download + standardize HuggingFace datasets
├── qa_dataset.py           10 built-in curated multi-hop questions
├── trace_analyzer.py       Feature extraction + Random Forest + LSTM
├── serve_vllm.sh           Launch Qwen3-VL-8B on port 3030
├── requirements.txt        Python deps
├── .env                    API keys (never committed)
├── datasets/               Prepared question files (generated by prep_datasets.py)
│   ├── 2wikimultihop_val.json
│   └── ...
└── traces/
    ├── {agent_id}/
    │   └── {dataset_name}/
    │       └── {YYYYMMDD_HHMMSS}/
    │           └── {episode_id}.json
    └── models/
        ├── classifier.pkl
        └── lstm_model.pt
```
