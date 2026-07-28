#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U uv
uv pip install -U vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly

source "$PROJECT_DIR/scripts/cuda_env.sh"
python -c 'import vllm; print("vLLM installed:", vllm.__version__)'
