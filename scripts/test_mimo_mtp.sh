#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python -m compileall -q \
  remtp/mimo_checkpoint.py \
  remtp/mimo_mtp.py \
  remtp/mimo_worker.py \
  remtp/binary_mtp_tree.py \
  remtp/dynamic_mtp_tree.py \
  remtp/dynamic_tree_vllm.py \
  remtp/dynamic_tree_humaneval_report.py \
  remtp/mimo_tree.py \
  remtp/mimo_tree_vllm.py \
  remtp/tree_audit.py \
  remtp/mimo_tree_report.py
pytest -q \
  tests/test_mimo_checkpoint.py \
  tests/test_mimo_mtp.py \
  tests/test_mimo_tree.py \
  tests/test_binary_mtp_tree.py \
  tests/test_dynamic_mtp_tree.py \
  tests/test_dynamic_tree_humaneval_report.py \
  tests/test_tree_audit.py \
  tests/test_mimo_tree_report.py \
  tests/test_fixed6_microtree.py

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/MiMo-7B-Base-MTP3}"
if [[ -d "$MODEL_PATH" ]]; then
  python -m remtp.mimo_checkpoint check "$MODEL_PATH"
else
  echo "[MiMo test] checkpoint not present; GPU/model-loading smoke skipped: $MODEL_PATH"
fi
