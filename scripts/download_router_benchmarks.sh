#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

declare -a EXTRA_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi
if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--check-only)
fi

python -m remtp.router_benchmark_download \
  --output-dir "${ROUTER_BENCHMARK_DIR:-data/regret_router/benchmarks}" \
  --timeout "${DOWNLOAD_TIMEOUT:-120}" \
  --retries "${DOWNLOAD_RETRIES:-3}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
