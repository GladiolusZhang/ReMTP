#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

export REMTP_TRACE=0
export ENFORCE_EAGER=0
export MTP_TOKENS="${MTP_TOKENS:-4}"
export MTP_REJECTION_SAMPLE_METHOD=probabilistic
export REMTP_WORKER_CLS=remtp.worker.BlockFeatureMTPWorker

variant="${BLOCK_FEATURE_VARIANT:-full}"
case "$variant" in
  block_budget)
    default_prefix=1
    default_future=0
    default_feature=0
    ;;
  lookahead)
    default_prefix=1
    default_future=1
    default_feature=0
    ;;
  full)
    default_prefix=1
    default_future=1
    default_feature=1
    ;;
  *)
    echo "Unknown BLOCK_FEATURE_VARIANT=$variant" >&2
    echo "Expected one of: block_budget, lookahead, full" >&2
    exit 2
    ;;
esac

export REMTP_BLOCK_KL_BUDGET="${BLOCK_KL_BUDGET:-1.2}"
export REMTP_BLOCK_EXPECTED_DRAFT_TOKENS="$MTP_TOKENS"
export REMTP_BLOCK_USE_PREFIX_VALUE="${BLOCK_USE_PREFIX_VALUE:-$default_prefix}"
export REMTP_BLOCK_USE_FUTURE_SUPPORT="${BLOCK_USE_FUTURE_SUPPORT:-$default_future}"
export REMTP_BLOCK_USE_FEATURE_CONSISTENCY="${BLOCK_USE_FEATURE_CONSISTENCY:-$default_feature}"
export REMTP_BLOCK_FUTURE_DECAY="${BLOCK_FUTURE_DECAY:-0.7}"
export REMTP_BLOCK_LOCAL_WEIGHT="${BLOCK_LOCAL_WEIGHT:-1.0}"
export REMTP_BLOCK_FUTURE_WEIGHT="${BLOCK_FUTURE_WEIGHT:-1.0}"
export REMTP_BLOCK_CONSISTENCY_WEIGHT="${BLOCK_CONSISTENCY_WEIGHT:-1.0}"
export REMTP_BLOCK_RESCUE_WEIGHT="${BLOCK_RESCUE_WEIGHT:-2.0}"
export REMTP_BLOCK_MAX_NORMALIZED_SURPRISAL="${BLOCK_MAX_NORMALIZED_SURPRISAL:-12.0}"
export REMTP_BLOCK_MIN_FEATURE_CONSISTENCY="${BLOCK_MIN_FEATURE_CONSISTENCY:-0.02}"
export REMTP_BLOCK_MIN_PREFIX_REACH="${BLOCK_MIN_PREFIX_REACH:-0.01}"
export REMTP_BLOCK_BISECTION_STEPS="${BLOCK_BISECTION_STEPS:-20}"

# Qwen3.5's GDN Triton kernel cannot be captured safely in this pinned stack.
if [[ -z "${REMTP_COMPILATION_CONFIG:-}" ]]; then
  export REMTP_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
fi

echo "[ReMTP][BlockFeature] variant=$variant mtp_tokens=$MTP_TOKENS"
exec "$PROJECT_DIR/scripts/serve.sh"
