#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/cuda_env.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/MiMo-7B-RL-0530}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-XiaomiMiMo/MiMo-7B-RL-0530}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
PORT="${PORT:-8000}"

echo "[ReMTP][MiMo target] model=$MODEL_PATH speculative_decoding=off"
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --no-async-scheduling \
  --enforce-eager \
  --port "$PORT"
