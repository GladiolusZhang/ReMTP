#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
MODEL_REVISION="${MODEL_REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen3.5-4B}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf is not installed. Activate .venv after running ./scripts/install.sh" >&2
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "[ReMTP] downloading $MODEL_ID@$MODEL_REVISION"
echo "[ReMTP] endpoint: $HF_ENDPOINT"
echo "[ReMTP] local directory: $MODEL_PATH"

hf download "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_PATH" \
  --max-workers "$HF_MAX_WORKERS"

PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m remtp.checkpoint "$MODEL_PATH"
echo "[ReMTP] model ready: $MODEL_PATH"
