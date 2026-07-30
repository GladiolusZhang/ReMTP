#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

epsilon_0="${GATED_KL_EPSILON_0:-0.02}"
depth_decay="${GATED_KL_DEPTH_DECAY:-0.7}"
max_log_gap="${GATED_KL_MAX_LOG_GAP:-1.5}"

RUN_NAME="gated_depth_kl_e${epsilon_0}_r${depth_decay}_tau${max_log_gap}" \
exec "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" "$@"
