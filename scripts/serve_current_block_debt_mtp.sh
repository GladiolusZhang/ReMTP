#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MTP_TOKENS="${MTP_TOKENS:-6}"
export TARGET_ANCHORED_VARIANT=tv_debt_control
export DEBT_POSITION_LIMIT="${DEBT_POSITION_LIMIT:-0.35}"
export DEBT_BLOCK_LIMIT="${DEBT_BLOCK_LIMIT:-1.20}"
export DEBT_SOFT_LOG_GAP="${DEBT_SOFT_LOG_GAP:-2.0}"
export DEBT_HARD_LOG_GAP="${DEBT_HARD_LOG_GAP:-6.0}"
export DEBT_MAX_POSITION_TV="${DEBT_MAX_POSITION_TV:-0.15}"
export DEBT_MAX_CACTUS_RATIO="${DEBT_MAX_CACTUS_RATIO:-2.0}"
export DEBT_FALLBACK="${DEBT_FALLBACK:-strict}"

exec "$PROJECT_DIR/scripts/serve_target_anchored_mtp.sh"
