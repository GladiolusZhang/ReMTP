#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

export REMTP_TRACE=0
export ENFORCE_EAGER=0
export MTP_TOKENS="${MTP_TOKENS:-2}"
export MTP_REJECTION_SAMPLE_METHOD=probabilistic
export REMTP_WORKER_CLS=remtp.worker.GatedDepthKLMTPWorker

export REMTP_GATED_KL_EPSILON_0="${GATED_KL_EPSILON_0:-0.02}"
export REMTP_GATED_KL_DEPTH_DECAY="${GATED_KL_DEPTH_DECAY:-0.7}"
export REMTP_GATED_KL_MAX_RANK="${GATED_KL_MAX_RANK:-8}"
export REMTP_GATED_KL_MIN_TARGET_PROB="${GATED_KL_MIN_TARGET_PROB:-0.005}"
export REMTP_GATED_KL_MAX_LOG_GAP="${GATED_KL_MAX_LOG_GAP:-1.5}"
export REMTP_GATED_KL_MAX_PROB_RATIO="${GATED_KL_MAX_PROB_RATIO:-4.0}"
export REMTP_GATED_KL_BISECTION_STEPS="${GATED_KL_BISECTION_STEPS:-20}"

if [[ -z "${REMTP_COMPILATION_CONFIG:-}" ]]; then
  export REMTP_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
fi

exec "$PROJECT_DIR/scripts/serve.sh"
