#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Native performance mode: no Python trace hook or per-round GPU synchronization,
# and allow vLLM to use torch.compile/CUDA graphs.
export REMTP_TRACE=0
export ENFORCE_EAGER=0
export MTP_TOKENS="${MTP_TOKENS:-2}"

exec "$PROJECT_DIR/scripts/serve.sh"
