#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

python -m remtp.humaneval_benchmark \
  --data "${HUMANEVAL_DATA:-data/humaneval/HumanEval.jsonl.gz}" \
  --samples "${SAMPLES:-164}" \
  --sample-seed "${SAMPLE_SEED:-20260802}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --generation-seed "${SEED:-42}" \
  --max-tokens "${MAX_TOKENS:-512}" \
  --mtp-tokens "${MTP_TOKENS:-6}" \
  --run-name "${RUN_NAME:-native_mtp}" \
  "$@"
