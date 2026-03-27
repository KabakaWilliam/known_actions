#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$SCRIPT_DIR/apptainer_tmp}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$SCRIPT_DIR/apptainer_cache}
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

SANDBOX_DIR=${PLAYWRIGHT_SANDBOX_DIR:-$SCRIPT_DIR/playwright-sandbox}
IMAGE=${PLAYWRIGHT_IMAGE:-docker://mcr.microsoft.com/playwright/python:v1.58.0-noble}

mkdir -p "$(dirname "$SANDBOX_DIR")"

if [ ! -d "$SANDBOX_DIR" ]; then
  echo "Building Playwright sandbox at $SANDBOX_DIR from $IMAGE ..."
  apptainer build --sandbox "$SANDBOX_DIR" "$IMAGE"
else
  echo "Sandbox already exists at $SANDBOX_DIR, skipping build."
fi

echo "Installing Python dependencies into sandbox ..."
apptainer exec \
  --writable \
  --cleanenv \
  "$SANDBOX_DIR" \
  /bin/bash -lc '
    set -euo pipefail
    python -m pip install --upgrade pip
    python -m pip install openai python-dotenv pytest-playwright
    playwright install
    python - <<"PY"
import openai
import dotenv
print("Installed:", openai.__version__, dotenv.__version__ if hasattr(dotenv, "__version__") else "python-dotenv")

from playwright.sync_api import sync_playwright
print("playwright import ok")
with sync_playwright() as p:
    print("chromium object ok:", p.chromium.name)
PY
  '

echo "Smoke test:"
apptainer exec --cleanenv "$SANDBOX_DIR" /bin/bash -lc '
  set -euo pipefail
  python --version
  python - <<"PY"
import openai
import dotenv
from playwright.sync_api import sync_playwright

print("openai ok")
print("python-dotenv ok")
print("playwright import ok")
with sync_playwright() as p:
    print("chromium object ok:", p.chromium.name)
PY
  command -v xvfb-run >/dev/null && echo "xvfb-run present"
'
