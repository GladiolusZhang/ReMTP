#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MTP_TOKENS="${MTP_TOKENS:-6}"
export TARGET_ANCHORED_VARIANT=tv_debt_control
export REGRET_DIRECTION=expected_top1_bias
export REGRET_TOP1_BIAS_INITIAL="${REGRET_TOP1_BIAS_INITIAL:-0.50}"
export REGRET_TOP1_BIAS_LEARNING_RATE="${REGRET_TOP1_BIAS_LEARNING_RATE:-0.25}"
export REGRET_TOP1_BIAS_DECAY="${REGRET_TOP1_BIAS_DECAY:-0.90}"
export REGRET_TOP1_BIAS_CLIP="${REGRET_TOP1_BIAS_CLIP:-0.75}"

exec "$PROJECT_DIR/scripts/serve_regret_feedback_mtp.sh"
