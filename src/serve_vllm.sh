# CUDA_VISIBLE_DEVICES=1,3 vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
#     --port 3030 \
#     --limit-mm-per-prompt.video 0 \
#     --tensor-parallel-size 2 \
#     --max_model_len 163840 \
#     --gpu-memory-utilization 0.75 \
#     --async-scheduling 

# CUDA_VISIBLE_DEVICES=1,3 vllm serve google/gemma-4-31B-it \
#   --port 3030 \
#   --tensor-parallel-size 2 \
#   --max-model-len 32768 \
#   --gpu-memory-utilization 0.90

# CUDA_VISIBLE_DEVICES=3 vllm serve google/gemma-4-26B-A4B-it \
#     --port 3030 \
#     --max-model-len 32768 \


# CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-27B \
#     --port 3031 \
#     --limit-mm-per-prompt.video 0 \
#     --reasoning-parser qwen3 \
#     --enable-prefix-caching \
#     --max_model_len 163840 

# CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-9B \
#     --port 3031 \
#     --limit-mm-per-prompt.video 0 \
#     --reasoning-parser qwen3 \
#     --enable-prefix-caching \
#     --max_model_len 163840 

# CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-VL-8B-Instruct \
#     --port 3030 \
#     --gpu-memory-utilization 0.9 \
#     --limit-mm-per-prompt.video 0 \
#     --max_model_len 163840

# CUDA_VISIBLE_DEVICES=2 vllm serve zai-org/GLM-4.6V-Flash \
#      --trust-remote-code \
#      --tool-call-parser glm45 \
#      --reasoning-parser glm45 \
#      --enable-auto-tool-choice \
#      --served-model-name glm-4.6v \
#      --allowed-local-media-path / \
#      --mm-encoder-tp-mode data \
#      --mm_processor_cache_type shm \
#      --port 3031 


vllm serve zai-org/GLM-4.6V \
     --tensor-parallel-size 4 \
     --tool-call-parser glm45 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.6v \
     --enable-expert-parallel \
     --allowed-local-media-path / \
     --mm-encoder-tp-mode data \
     --mm_processor_cache_type shm \
     --port 3031 

# CUDA_VISIBLE_DEVICES=2 vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
#     --port 3031 \
#     --gpu-memory-utilization 0.9 \
#     --max_model_len 128000

# curl -s http://127.0.0.1:3030/v1/models | jq



# fix for vllm for glm issue: try using with vLLM 0.18.0 and transformers>=5.3.0 https://huggingface.co/zai-org/GLM-4.6V-Flash/discussions/81



# Planned models:


# claude-haiku-4-5-20251001
# gpt-5.4-mini


# pip install -U vllm --pre \
#   --extra-index-url https://wheels.vllm.ai/nightly/cu124 \
#   --extra-index-url https://download.pytorch.org/whl/cu124

# pip install transformers==5.5.0
