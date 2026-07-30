#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

python -m remtp.gsm8k_benchmark \
  --data "${GSM8K_DATA:-data/gsm8k/test.jsonl}" \
  --samples "${SAMPLES:-100}" \
  --sample-seed "${SAMPLE_SEED:-20260730}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --generation-seed "${SEED:-42}" \
  --max-tokens "${MAX_TOKENS:-384}" \
  --mtp-tokens "${MTP_TOKENS:-4}" \
  --run-name "${RUN_NAME:-native_mtp}" \
  "$@"
