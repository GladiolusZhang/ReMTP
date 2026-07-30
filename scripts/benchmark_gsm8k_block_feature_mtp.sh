#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

variant="${BLOCK_FEATURE_VARIANT:-full}"
budget="${BLOCK_DELTA_BUDGET:-${BLOCK_KL_BUDGET:-4.0}}"
top_k="${BLOCK_DISTRIBUTION_TOP_K:-8}"

RUN_NAME="block_feature_fast_${variant}_delta${budget}_topk${top_k}" \
MTP_TOKENS="${MTP_TOKENS:-4}" \
exec "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" "$@"
