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
  token_only)
    default_prefix=1
    default_current_distribution=0
    default_future_distribution=0
    ;;
  distribution)
    default_prefix=1
    default_current_distribution=1
    default_future_distribution=0
    ;;
  full)
    default_prefix=1
    default_current_distribution=1
    default_future_distribution=1
    ;;
  *)
    echo "Unknown BLOCK_FEATURE_VARIANT=$variant" >&2
    echo "Expected one of: token_only, distribution, full" >&2
    exit 2
    ;;
esac

export REMTP_BLOCK_KL_BUDGET="${BLOCK_KL_BUDGET:-1.2}"
export REMTP_BLOCK_EXPECTED_DRAFT_TOKENS="$MTP_TOKENS"
export REMTP_BLOCK_USE_PREFIX_VALUE="${BLOCK_USE_PREFIX_VALUE:-$default_prefix}"
export REMTP_BLOCK_USE_CURRENT_DISTRIBUTION="${BLOCK_USE_CURRENT_DISTRIBUTION:-$default_current_distribution}"
export REMTP_BLOCK_USE_FUTURE_DISTRIBUTION="${BLOCK_USE_FUTURE_DISTRIBUTION:-$default_future_distribution}"
export REMTP_BLOCK_FUTURE_DECAY="${BLOCK_FUTURE_DECAY:-0.7}"
export REMTP_BLOCK_DISTRIBUTION_TOP_K="${BLOCK_DISTRIBUTION_TOP_K:-8}"
export REMTP_BLOCK_TOKEN_RANK_SCALE="${BLOCK_TOKEN_RANK_SCALE:-4.0}"
export REMTP_BLOCK_TOKEN_LOG_GAP_SCALE="${BLOCK_TOKEN_LOG_GAP_SCALE:-2.0}"
export REMTP_BLOCK_OVERLAP_WEIGHT="${BLOCK_OVERLAP_WEIGHT:-0.4}"
export REMTP_BLOCK_PROBABILITY_COSINE_WEIGHT="${BLOCK_PROBABILITY_COSINE_WEIGHT:-0.4}"
export REMTP_BLOCK_ENTROPY_WEIGHT="${BLOCK_ENTROPY_WEIGHT:-0.2}"
export REMTP_BLOCK_TOKEN_SIGNAL_WEIGHT="${BLOCK_TOKEN_SIGNAL_WEIGHT:-1.0}"
export REMTP_BLOCK_CURRENT_DISTRIBUTION_WEIGHT="${BLOCK_CURRENT_DISTRIBUTION_WEIGHT:-1.0}"
export REMTP_BLOCK_FUTURE_DISTRIBUTION_WEIGHT="${BLOCK_FUTURE_DISTRIBUTION_WEIGHT:-1.0}"
export REMTP_BLOCK_RELIABILITY_POWER="${BLOCK_RELIABILITY_POWER:-2.0}"
export REMTP_BLOCK_MIN_RELIABILITY="${BLOCK_MIN_RELIABILITY:-0.2}"
export REMTP_BLOCK_MAX_NORMALIZED_SURPRISAL="${BLOCK_MAX_NORMALIZED_SURPRISAL:-12.0}"
export REMTP_BLOCK_MIN_PREFIX_REACH="${BLOCK_MIN_PREFIX_REACH:-0.01}"
export REMTP_BLOCK_BISECTION_STEPS="${BLOCK_BISECTION_STEPS:-20}"

# Qwen3.5's GDN Triton kernel cannot be captured safely in this pinned stack.
if [[ -z "${REMTP_COMPILATION_CONFIG:-}" ]]; then
  export REMTP_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
fi

echo "[ReMTP][BlockFeature] variant=$variant mtp_tokens=$MTP_TOKENS"
exec "$PROJECT_DIR/scripts/serve.sh"
