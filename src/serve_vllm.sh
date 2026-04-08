# CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
#     --port 3030 \
#     --limit-mm-per-prompt.video 0 \
#     --tensor-parallel-size 2 \
#     --max_model_len 163840 \
#     --async-scheduling 

# CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-VL-8B-Instruct \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --reasoning-parser deepseek_r1 \
#     --port 3030 \
#     --gpu-memory-utilization 0.9 \
#     --limit-mm-per-prompt.video 0 \
#     --max_model_len 163840

CUDA_VISIBLE_DEVICES=2 vllm serve zai-org/GLM-4.6V-Flash \
     --trust-remote-code \
     --tool-call-parser glm45 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.6v \
     --allowed-local-media-path / \
     --mm-encoder-tp-mode data \
     --mm_processor_cache_type shm \
     --port 3031 

# CUDA_VISIBLE_DEVICES=3 vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
#     --port 3031 \
#     --gpu-memory-utilization 0.9 \
#     --max_model_len 128000

# curl -s http://127.0.0.1:3030/v1/models | jq



# fix for vllm for glm issue: try using with vLLM 0.18.0 and transformers>=5.3.0 https://huggingface.co/zai-org/GLM-4.6V-Flash/discussions/8

