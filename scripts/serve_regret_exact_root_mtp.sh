#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MTP_TOKENS="${MTP_TOKENS:-6}"
export REGRET_DIRECTION=expected_root_exact
export REGRET_ROOT_ANCHOR_REFERENCE="${REGRET_ROOT_ANCHOR_REFERENCE:-0.05}"

exec "$PROJECT_DIR/scripts/serve_regret_feedback_mtp.sh"
