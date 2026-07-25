#!/bin/bash

curl -sS https://openrouter.ai/api/v1/key \
  -H "Authorization: Bearer $OPEN_ROUTER_API_KEY" |
  jq '.data'