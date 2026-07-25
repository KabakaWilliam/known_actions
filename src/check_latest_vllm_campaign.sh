#!/bin/bash

run_dir=$(ls -td campaign_runs/* | head -n 1)
tail -F "$run_dir/logs/qwen3vl_30b_a3b.vllm.log"