#!/usr/bin/env bash
# Serve TencentBAC/FastMTP with native (strict) MTP verification
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/FastMTP}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-TencentBAC/FastMTP}"
PORT="${PORT:-8000}"
MTP_TOKENS="${MTP_TOKENS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
NO_ASYNC_SCHEDULING="${NO_ASYNC_SCHEDULING:-1}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Error: FastMTP model not found at $MODEL_PATH" >&2
  echo "Download it with: ./scripts/download_fastmtp.sh" >&2
  exit 2
fi

WORKER_CLS="remtp.fastmtp_worker.FastMTPProbabilisticWorker"

EXTRA_ARGS=()
[[ "$ENFORCE_EAGER" == "1" ]] && EXTRA_ARGS+=(--enforce-eager)
[[ "$NO_ASYNC_SCHEDULING" == "1" ]] && EXTRA_ARGS+=(--no-async-scheduling)

# FastMTP uses custom code (modeling_mimo.py)
EXTRA_ARGS+=(--trust-remote-code)

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --worker-cls "$WORKER_CLS" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}" \
  "${EXTRA_ARGS[@]}"
