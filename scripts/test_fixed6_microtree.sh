#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

python -m py_compile \
  remtp/fixed6_microtree.py \
  remtp/fixed6_vllm.py \
  remtp/fixed6_validation.py
python -m unittest \
  tests.test_fixed6_microtree \
  tests.test_fixed6_validation \
  -q
