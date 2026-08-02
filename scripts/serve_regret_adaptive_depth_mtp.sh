#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MTP_TOKENS="${MTP_TOKENS:-6}"
export REGRET_DIRECTION=adaptive_depth
export REGRET_ADAPTIVE_MIN_DEPTH="${REGRET_ADAPTIVE_MIN_DEPTH:-4}"
export REGRET_ADAPTIVE_DEPTH_BLOCKS="${REGRET_ADAPTIVE_DEPTH_BLOCKS:-2}"
export REGRET_ADAPTIVE_DEPTH_TRIGGER="${REGRET_ADAPTIVE_DEPTH_TRIGGER:-0.25}"

exec "$PROJECT_DIR/scripts/serve_regret_feedback_mtp.sh"
