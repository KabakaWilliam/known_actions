#!/bin/bash

run_dir=$(ls -td campaign_runs/* | head -n 1)
tail -F "$run_dir/logs/glm_4.6v.vllm.log"