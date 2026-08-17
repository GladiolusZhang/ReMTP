#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/cuda_env.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/MiMo-7B-Base-MTP3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-XiaomiMiMo/MiMo-7B-Base}"
METHOD="${METHOD:-native}"
MTP_TOKENS="${MTP_TOKENS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
NO_ASYNC_SCHEDULING="${NO_ASYNC_SCHEDULING:-0}"
MIMO_MTP_LAYER_MODE="${MIMO_MTP_LAYER_MODE:-physical}"
PORT="${PORT:-8000}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export REMTP_MIMO_MTP_LAYER_MODE="$MIMO_MTP_LAYER_MODE"
if [[ "$MIMO_MTP_LAYER_MODE" == "physical" ]]; then
  export REMTP_ALLOW_EXPERIMENTAL_MIMO_MTP3=1
  python -m remtp.mimo_checkpoint check "$MODEL_PATH"
else
  python -m remtp.mimo_checkpoint check "$MODEL_PATH" --allow-single-layer
fi

case "$METHOD" in
  native)
    worker="remtp.mimo_worker.MiMoProbabilisticMTPWorker"
    ;;
  cactus)
    worker="remtp.mimo_worker.MiMoCactusMTPWorker"
    ;;
  spec_cascade|token_v3)
    worker="remtp.mimo_worker.MiMoSpecCascadeMTPWorker"
    export REMTP_CASCADE_RULE="${REMTP_CASCADE_RULE:-token_v3}"
    ;;
  block_verification)
    worker="remtp.mimo_worker.MiMoBlockVerificationMTPWorker"
    ;;
  cactus_block)
    worker="remtp.mimo_worker.MiMoCactusBlockVerificationMTPWorker"
    ;;
  proposal_calibrated)
    worker="remtp.mimo_worker.MiMoProposalCalibratedMTPWorker"
    ;;
  remtp)
    worker="remtp.mimo_worker.MiMoReMTPWorker"
    export REMTP_RISK_SCHEME="${REMTP_RISK_SCHEME:-scheme12_joint}"
    ;;
  *)
    echo "Unknown METHOD=$METHOD" >&2
    echo "Choose: native cactus spec_cascade block_verification cactus_block proposal_calibrated remtp" >&2
    exit 2
    ;;
esac

speculative_config="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
args=(
  serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --tensor-parallel-size 1
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs 1
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --worker-cls "$worker"
  --speculative-config "$speculative_config"
  --port "$PORT"
)
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  args+=(--enforce-eager)
fi
if [[ "$NO_ASYNC_SCHEDULING" == "1" ]]; then
  args+=(--no-async-scheduling)
fi

echo "[ReMTP][MiMo] method=$METHOD model=$MODEL_PATH mtp_tokens=$MTP_TOKENS physical_route=$MIMO_MTP_LAYER_MODE"
vllm "${args[@]}"
