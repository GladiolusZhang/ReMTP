#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

export REMTP_TRACE=0
export ENFORCE_EAGER=0
export MTP_TOKENS="${MTP_TOKENS:-4}"
export MTP_REJECTION_SAMPLE_METHOD=probabilistic
export REMTP_WORKER_CLS="${REMTP_WORKER_CLS:-remtp.worker.ProbabilisticMTPWorker}"

# Qwen3.5's GDN Triton kernel is incompatible with CUDA Graph capture in the
# pinned stack. Keep torch.compile but disable only CUDA Graph.
if [[ -z "${REMTP_COMPILATION_CONFIG:-}" ]]; then
  export REMTP_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
fi

exec "$PROJECT_DIR/scripts/serve.sh"
