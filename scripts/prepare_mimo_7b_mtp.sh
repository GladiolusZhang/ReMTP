#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

MODEL_ROOT="${MODEL_ROOT:-$PROJECT_DIR/models}"
BASE_DIR="${BASE_DIR:-$MODEL_ROOT/MiMo-7B-Base}"
MTP_DIR="${MTP_DIR:-$MODEL_ROOT/MiMo-7B-MTPs}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_ROOT/MiMo-7B-Base-MTP3}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output exists; refusing to overwrite: $OUTPUT_DIR" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -m remtp.mimo_checkpoint build \
  --base "$BASE_DIR" \
  --mtp "$MTP_DIR" \
  --output "$OUTPUT_DIR"
