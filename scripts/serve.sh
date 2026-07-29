#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/cuda_env.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
MTP_METHOD="${MTP_METHOD:-mtp}"
MTP_TOKENS="${MTP_TOKENS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed. Run: ./scripts/install.sh" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
if ! python -m remtp.checkpoint "$MODEL_PATH"; then
  echo "Download or resume it first: ./scripts/download_model.sh" >&2
  exit 1
fi

export REMTP_TRACE=1
export REMTP_MODEL="$MODEL_PATH"

echo "[ReMTP] model=$MODEL_PATH method=$MTP_METHOD draft_tokens=$MTP_TOKENS"

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --worker-cls remtp.worker.ReMTPWorker \
  --language-model-only \
  --enforce-eager \
  --speculative-config \
  "{\"method\":\"${MTP_METHOD}\",\"num_speculative_tokens\":${MTP_TOKENS}}"
