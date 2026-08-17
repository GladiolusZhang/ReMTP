#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MTP_TOKENS="${MTP_TOKENS:-6}"
export REMTP_WORKER_CLS=remtp.worker.AdaptiveChainMTPWorker
# vLLM's async EAGLE/MTP path keeps a fixed-width GPU draft buffer and does
# not expose per-request valid draft counts. Disable it only for Scheme 3 so
# the scheduler can consume the dynamically shortened draft list correctly.
export ASYNC_SCHEDULING=0
export REMTP_ADAPTIVE_MAX_DEPTH="$MTP_TOKENS"
export REMTP_ADAPTIVE_MEDIUM_DEPTH="${ADAPTIVE_MEDIUM_DEPTH:-4}"
export REMTP_ADAPTIVE_SHORT_DEPTH="${ADAPTIVE_SHORT_DEPTH:-2}"
export REMTP_ADAPTIVE_TOP_M="${ADAPTIVE_TOP_M:-4}"
export REMTP_ADAPTIVE_HIGH_MARGIN="${ADAPTIVE_HIGH_MARGIN:-1.50}"
export REMTP_ADAPTIVE_MEDIUM_MARGIN="${ADAPTIVE_MEDIUM_MARGIN:-0.50}"
export REMTP_ADAPTIVE_LOW_ENTROPY="${ADAPTIVE_LOW_ENTROPY:-0.45}"
export REMTP_ADAPTIVE_HIGH_ENTROPY="${ADAPTIVE_HIGH_ENTROPY:-0.75}"
export REMTP_ADAPTIVE_HIGH_MTP_CONF="${ADAPTIVE_HIGH_MTP_CONF:-0.30}"
export REMTP_ADAPTIVE_MEDIUM_MTP_CONF="${ADAPTIVE_MEDIUM_MTP_CONF:-0.10}"
export REMTP_ADAPTIVE_DIAGNOSTICS="${ADAPTIVE_DIAGNOSTICS:-0}"

exec "$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh"
