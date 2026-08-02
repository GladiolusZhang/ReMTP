#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MTP_TOKENS="${MTP_TOKENS:-6}"
export REMTP_WORKER_CLS=remtp.worker.PrefixTrimBlockMTPWorker
export REMTP_PREFIX_TRIM_DIAGNOSTICS="${PREFIX_TRIM_DIAGNOSTICS:-0}"
export REMTP_BLOCK_VERIFY_DIAGNOSTICS="${BLOCK_VERIFY_DIAGNOSTICS:-0}"
export REMTP_BLOCK_VERIFY_AUDIT_INTERVAL="${BLOCK_VERIFY_AUDIT_INTERVAL:-0}"

exec "$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh"
