#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# This preset matched Cactus draft acceptance within 0.56 percentage points
# on the fixed 100-sample GSM8K tuning subset. It intentionally trades more
# target-distribution fidelity for acceptance and is not the default preset.
export GATED_KL_EPSILON_0="${GATED_KL_EPSILON_0:-0.8}"
export GATED_KL_DEPTH_DECAY="${GATED_KL_DEPTH_DECAY:-1.0}"
export GATED_KL_MAX_LOG_GAP="${GATED_KL_MAX_LOG_GAP:-10.0}"
export GATED_KL_MAX_RANK="${GATED_KL_MAX_RANK:-128}"
export GATED_KL_MIN_TARGET_PROB="${GATED_KL_MIN_TARGET_PROB:-0.000001}"
export GATED_KL_MAX_PROB_RATIO="${GATED_KL_MAX_PROB_RATIO:-32.0}"

exec "$PROJECT_DIR/scripts/serve_gated_depth_kl_mtp.sh"
