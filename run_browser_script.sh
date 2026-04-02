#!/bin/bash
# Usage: ./run_script.sh tasks/open_browser.ts
# Usage: ./run_script.sh tasks/search_google.ts

set -e

WORKDIR=/VData/linna4335/known_actions
SCRIPT=${1:-"tasks/open_browser.ts"}

if [ ! -f "$WORKDIR/.env" ]; then
  echo "ERROR: .env file not found at $WORKDIR/.env"
  exit 1
fi

# Load .env into current shell for apptainer --env flags
set -a
source "$WORKDIR/.env"
set +a

echo "[*] Running script: $SCRIPT"
echo "[*] Model: $MIDSCENE_MODEL_NAME"
echo ""

apptainer exec \
  --no-home \
  --writable-tmpfs \
  --home "$WORKDIR/container_home":/home/headless \
  --bind "$WORKDIR/scripts":/scripts \
  --bind "$WORKDIR/.env":/scripts/.env \
  --env DISPLAY=:10 \
  --env MIDSCENE_MODEL_BASE_URL="$MIDSCENE_MODEL_BASE_URL" \
  --env MIDSCENE_MODEL_API_KEY="$MIDSCENE_MODEL_API_KEY" \
  --env MIDSCENE_MODEL_NAME="$MIDSCENE_MODEL_NAME" \
  --env MIDSCENE_MODEL_FAMILY="$MIDSCENE_MODEL_FAMILY" \
  --env MIDSCENE_SERVICE_URL="http://localhost:3333" \
  "$WORKDIR/midscene-desktop.sif" \
  bash -c "cd /scripts && npx tsx $SCRIPT"
