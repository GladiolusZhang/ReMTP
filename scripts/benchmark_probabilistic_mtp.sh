#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

RUN_NAME="probabilistic_mtp" \
TEMPERATURE="${TEMPERATURE:-0.7}" \
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-20}" \
SEED="${SEED:-42}" \
MAX_TOKENS="${MAX_TOKENS:-128}" \
MTP_TOKENS="${MTP_TOKENS:-2}" \
exec "$PROJECT_DIR/scripts/benchmark_specbench.sh" "$@"
