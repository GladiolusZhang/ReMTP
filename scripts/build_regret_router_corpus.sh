#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

if ! python -c 'import datasets, pyarrow' >/dev/null 2>&1; then
  echo "Missing corpus dependencies." >&2
  echo "Install with: uv pip install -U datasets pyarrow datasketch 'huggingface-hub>=0.34,<1.0'" >&2
  exit 2
fi

python -m remtp.router_corpus \
  --profile "${ROUTER_CORPUS_PROFILE:-pilot}" \
  --output "${ROUTER_CORPUS_RAW:-data/regret_router/router_corpus_raw.jsonl}" \
  --seed "${ROUTER_CORPUS_SEED:-20260803}" \
  --instruction-source "${ROUTER_INSTRUCTION_SOURCE:-no_robots}" \
  "$@"
