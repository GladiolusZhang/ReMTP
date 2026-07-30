#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

rule="${CASCADE_RULE:-token_v3}"
alpha="${CASCADE_ALPHA:-0.5}"

RUN_NAME="spec_cascade_${rule}_a${alpha}" \
exec "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" "$@"
