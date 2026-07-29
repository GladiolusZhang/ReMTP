#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

# Native performance mode: no Python trace hook or per-round GPU synchronization.
export REMTP_TRACE=0
export ENFORCE_EAGER=0
export MTP_TOKENS="${MTP_TOKENS:-2}"

# Qwen3.5's GDN/causal-conv Triton path is not CUDA Graph compatible with the
# pinned vLLM/CUDA stack. Keep torch.compile enabled, but disable graph capture.
if [[ -z "${REMTP_COMPILATION_CONFIG:-}" ]]; then
  export REMTP_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
fi

exec "$PROJECT_DIR/scripts/serve.sh"
