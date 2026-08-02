#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MTP_TOKENS="${MTP_TOKENS:-6}"
export TARGET_ANCHORED_VARIANT=tv_target_surplus
export SURPLUS_MAX_LOG_GAP="${SURPLUS_MAX_LOG_GAP:-2.0}"
export TARGET_ANCHORED_AUDIT_INTERVAL="${TARGET_ANCHORED_AUDIT_INTERVAL:-0}"

exec "$PROJECT_DIR/scripts/serve_target_anchored_mtp.sh"
