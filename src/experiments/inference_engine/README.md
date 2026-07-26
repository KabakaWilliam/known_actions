# Inference-engine pilot

This is intentionally lower priority than the final XGBoost, feature-ablation,
and behavioral-policy normalization experiments.

## Environment

SGLang is installed separately from vLLM in the project-local
`w_serve_sglang_0512` environment:

```bash
export CONDA_ENVS_PATH=/VData/linna4335/known_actions/.conda-runtime/envs
export CONDA_PKGS_DIRS=/VData/linna4335/known_actions/.conda-runtime/pkgs
export CONDA_NO_PLUGINS=true
conda activate w_serve_sglang_0512
```

The environment and package cache live on `/VData`, are gitignored, and use
Python 3.11.15 to match `known_ag`. The campaign can invoke its Python directly,
so activation is not required for collection.

The frozen serving stack is Python 3.11.15, SGLang 0.5.12.post1,
`sglang-kernel` 0.4.2.post2, PyTorch 2.11.0 with CUDA 13.0, and Transformers
5.6.0. Hugging Face `kernels` is explicitly pinned to 0.12.3: SGLang's
dependency is unbounded, but Transformers 5.6.0 declares the compatible range
as `kernels>=0.12.0,<0.13`; allowing pip to resolve 0.16.0 breaks SGLang during
import. The environment was installed with pip temporary storage on `/VData`
because the host root filesystem (and therefore `/tmp`) has insufficient free
space:

```bash
mkdir -p /VData/linna4335/known_actions/.tmp
export TMPDIR=/VData/linna4335/known_actions/.tmp
/VData/linna4335/known_actions/.conda-runtime/envs/w_serve_sglang_0512/bin/python \
  -m pip install --no-cache-dir 'sglang[all]==0.5.12.post1'
/VData/linna4335/known_actions/.conda-runtime/envs/w_serve_sglang_0512/bin/python \
  -m pip install --no-cache-dir 'kernels>=0.12.0,<0.13'
```

The host driver is 610.43.02 and reports CUDA UMD 13.3, which is new enough
for the CUDA 13.0 runtime bundled with PyTorch. Package inspection confirms
SGLang entry classes for Qwen3.5 MoE, GLM-4V MoE, Gemma-4 multimodal, and
Qwen3-VL MoE. That is only a package-level support check: each exact checkpoint
must still pass model-loading, text, image, and browser-use smoke tests on the
host GPUs before trace collection.

Each `local_models` item configures exactly one server:

```yaml
local_models:
  - agent_id: qwen3_5_27b
    sglang:
      model: Qwen/Qwen3.5-27B
      served_model_name: Qwen/Qwen3.5-27B
      gpus: [0]
      port: 3031
      startup_timeout_s: 3600
      args:
        - --tp-size
        - "1"
        - --tool-call-parser
        - qwen
        - --reasoning-parser
        - qwen3
```

Use a mapping after `sglang:` as shown above, not a YAML list. Existing
`vllm:` entries continue to work without changes. A model entry containing
both engine keys is rejected. `served_model_name` is passed to SGLang
automatically so that its `/v1/models` response and the agent registry agree.

For the four-GPU GLM entry, the engine block is:

```yaml
- agent_id: glm_4.6v
  sglang:
    model: zai-org/GLM-4.6V
    served_model_name: glm-4.6v
    gpus: [0, 1, 2, 3]
    port: 3031
    startup_timeout_s: 3600
    args:
      - --tp-size
      - "4"
      - --tool-call-parser
      - glm
      - --reasoning-parser
      - glm45
```

The intervention uses the complete frozen WebShop split: 150 train, 75
validation, and 75 test tasks. Its required models are the three open-weight
members of the final identity roster:
GLM-4.6V, Qwen3.5-27B, and Gemma-4-26B-A4B. Qwen3-VL remains a supplemental
model: it is included in smoke coverage but has a separate optional 75-task
campaign so it cannot alter the final six-model experiment.

SGLang is the candidate second engine, subject to these gates:

1. Install it in a separate environment; do not alter the working vLLM
   environment.
2. Confirm the exact model architecture and vision/tool-calling path are
   supported.
3. Start an OpenAI-compatible endpoint and pass `/v1/models`.
4. Run two browser-use episodes and verify actual browser activity, valid tool
   calls, trace writing, and task-success semantics.
5. Match model revision, tokenizer, precision, decoding parameters, context
   length, worker count, timeout, browser image, and prompt.

Only after the smoke gate passes should the 75-task traces be collected. Never
place SGLang traces in baseline `traces/`.

## Output layout

The smoke namespace is:

- traces:
  `traces_experiments/inference_engine_webshop_sglang_smoke_v1/`
- campaign manifests and logs:
  `campaign_runs/experiments/inference_engine_webshop_sglang_smoke_v1/<campaign-id>/`

The full WebShop intervention namespace is:

- traces:
  `traces_experiments/inference_engine_webshop_sglang_full_v1/`
- campaign manifests and logs:
  `campaign_runs/experiments/inference_engine_webshop_sglang_full_v1/<campaign-id>/`
- reserved analysis output:
  `artifacts/experiments/inference_engine/webshop_sglang_full_v1/`

The analysis pipeline writes
`artifacts/experiments/inference_engine/webshop_sglang_full_v1/REPORT.md`
following the repository-wide reporting contract in `experiments/README.md`.
It distinguishes collection completeness from experimental results and labels
vLLM→SGLang, within-engine, reverse-transfer, and mixed-engine comparisons
explicitly.

Each new campaign invocation creates a new timestamp/UUID campaign directory,
but trace resume scans the fixed trace root. Restarting therefore preserves
the prior logs and skips already valid traces.

## Commands

Inspect all three generated server commands without loading weights:

```bash
cd /VData/linna4335/known_actions/src
python browser_use_campaign.py \
  --config experiments/inference_engine/configs/webshop_sglang_smoke_campaign.yaml \
  --skip-openrouter \
  --dry-run
```

Run the two-episode-per-model GPU smoke for the required trio plus supplemental
Qwen3-VL:

```bash
cd /VData/linna4335/known_actions/src
python browser_use_campaign.py \
  --config experiments/inference_engine/configs/webshop_sglang_smoke_campaign.yaml \
  --skip-openrouter
```

For the fastest schedule, collect all 300 GLM traces first:

```bash
cd /VData/linna4335/known_actions/src
python browser_use_campaign.py \
  --config experiments/inference_engine/configs/webshop_sglang_full_campaign.yaml \
  --only glm_4.6v \
  --skip-openrouter
```

After GLM finishes and releases all four GPUs, launch these concurrently in
two terminals:

```bash
# Terminal 1: Qwen3.5 on GPU 0
cd /VData/linna4335/known_actions/src
python browser_use_campaign.py \
  --config experiments/inference_engine/configs/webshop_sglang_full_campaign.yaml \
  --only qwen3_5_27b \
  --skip-openrouter
```

```bash
# Terminal 2: Gemma on GPU 1
cd /VData/linna4335/known_actions/src
python browser_use_campaign.py \
  --config experiments/inference_engine/configs/webshop_sglang_full_campaign.yaml \
  --only gemma_4_26B_A4B_it \
  --skip-openrouter
```

The two concurrent campaigns use different GPUs, ports, model directories, and
timestamp/UUID campaign directories. Each uses ten browser workers, for 20
browser workers in aggregate. Startup timeout is one hour per model, task
timeout is 300 seconds, and up to three collection rounds recover failed
episodes.

Optionally collect Qwen3-VL on the 75 WebShop test tasks into the same trace
root, with campaign state kept in a supplemental namespace:

```bash
cd /VData/linna4335/known_actions/src
python browser_use_campaign.py \
  --config experiments/inference_engine/configs/webshop_sglang_qwen3vl_supplemental_campaign.yaml \
  --skip-openrouter
```

The evaluation matrix is:

- vLLM train/validation → vLLM test (control)
- vLLM train/validation → SGLang test (priority engine-generalization result)
- SGLang train/validation → SGLang test
- SGLang train/validation → vLLM test
- balanced mixed-engine train/validation → each engine's test split

Run XGBoost with full, timing-only, and non-timing features. A different number
of H100s is a serving-topology change, not evidence of robustness to a
different GPU architecture; report those claims separately.

The complete collection has now passed the matched-task audit at 150/75/75
tasks for all three models under both engines. Freeze and run the analysis:

```bash
cd /VData/linna4335/known_actions/src
python -m experiments.inference_engine.pipeline audit
python -m experiments.inference_engine.pipeline prepare

# Fast first pass
CUDA_VISIBLE_DEVICES=1 python -m experiments.inference_engine.pipeline run-grid \
  --feature-groups full \
  --seeds 42 \
  --xgb-device cuda

# Timing/non-timing views
CUDA_VISIBLE_DEVICES=1 python -m experiments.inference_engine.pipeline run-grid \
  --feature-groups timing_only non_timing \
  --seeds 42 \
  --xgb-device cuda
```

The commentable launcher `scripts/run_followup_experiments.sh` contains these
commands plus the five-seed confirmation.
