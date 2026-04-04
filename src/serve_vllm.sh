CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-VL-8B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 --port 3030 --gpu-memory-utilization 0.9 --max_model_len 163840


# curl -s http://127.0.0.1:3030/v1/models | jq