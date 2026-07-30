#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

alpha="${REGRET_ALPHA:-0.03}"
decay="${REGRET_TOKEN_DECAY:-0.90}"

RUN_NAME="budget_induced_regret_a${alpha}_decay${decay}" \
MTP_TOKENS="${MTP_TOKENS:-6}" \
exec "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" "$@"
