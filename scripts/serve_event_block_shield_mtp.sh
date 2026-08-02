#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MTP_TOKENS="${MTP_TOKENS:-6}"
export TARGET_ANCHORED_VARIANT=tv_event_shield
export REMTP_WORKER_CLS=remtp.worker.TargetAnchoredBlockMTPWorker
export BLOCK_SHIELD_CACTUS_MIX="${BLOCK_SHIELD_CACTUS_MIX:-0.30}"
export REMTP_BLOCK_VERIFY_DIAGNOSTICS="${BLOCK_VERIFY_DIAGNOSTICS:-0}"
export REMTP_BLOCK_VERIFY_AUDIT_INTERVAL="${BLOCK_VERIFY_AUDIT_INTERVAL:-0}"

exec "$PROJECT_DIR/scripts/serve_target_anchored_mtp.sh"
