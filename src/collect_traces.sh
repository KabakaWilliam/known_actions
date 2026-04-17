#!/bin/bash

python3 orchestrator.py --config custom_config.yaml

# get error counts by agent/dataset
# grep -rl "Error: failed to call AI model service" src/traces/ | awk -F'/' '{print $3"/"$4}' | sort | uniq -c | sort -nr