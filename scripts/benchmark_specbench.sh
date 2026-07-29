#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

python -m remtp.benchmark \
  --data "${SPEC_BENCH_DATA:-data/spec_bench/question.jsonl}" \
  --tasks translation summarization math_reasoning rag \
  --samples-per-task "${SAMPLES_PER_TASK:-20}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --generation-seed "${SEED:-42}" \
  --max-tokens "${MAX_TOKENS:-128}" \
  --mtp-tokens "${MTP_TOKENS:-2}" \
  "$@"
