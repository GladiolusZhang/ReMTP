#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

export REMTP_TRACE=0
export ENFORCE_EAGER=0
export MTP_TOKENS="${MTP_TOKENS:-2}"
export MTP_REJECTION_SAMPLE_METHOD=probabilistic
export REMTP_WORKER_CLS=remtp.worker.SpecCascadeMTPWorker
export REMTP_CASCADE_RULE="${CASCADE_RULE:-token_v3}"
export REMTP_CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"

# Same stable compiled mode used by the native-MTP benchmark.
if [[ -z "${REMTP_COMPILATION_CONFIG:-}" ]]; then
  export REMTP_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
fi

exec "$PROJECT_DIR/scripts/serve.sh"
