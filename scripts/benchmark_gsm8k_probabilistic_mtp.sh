#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

RUN_NAME="probabilistic_mtp" \
exec "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" "$@"
