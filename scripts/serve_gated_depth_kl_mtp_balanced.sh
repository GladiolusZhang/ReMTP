#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Quality-preserving point from the fixed 100-sample GSM8K tuning subset.
export GATED_KL_EPSILON_0="${GATED_KL_EPSILON_0:-0.32}"
export GATED_KL_DEPTH_DECAY="${GATED_KL_DEPTH_DECAY:-1.0}"
export GATED_KL_MAX_LOG_GAP="${GATED_KL_MAX_LOG_GAP:-6.0}"
export GATED_KL_MAX_RANK="${GATED_KL_MAX_RANK:-32}"
export GATED_KL_MIN_TARGET_PROB="${GATED_KL_MIN_TARGET_PROB:-0.0005}"
export GATED_KL_MAX_PROB_RATIO="${GATED_KL_MAX_PROB_RATIO:-8.0}"

exec "$PROJECT_DIR/scripts/serve_gated_depth_kl_mtp.sh"
