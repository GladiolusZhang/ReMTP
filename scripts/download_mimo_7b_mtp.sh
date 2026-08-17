#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

# Always use the official Hugging Face endpoint by default.  Do not inherit a
# stale HF_ENDPOINT (for example hf-mirror.com) from the parent shell.  An
# alternative endpoint must be requested explicitly through REMTP_HF_ENDPOINT.
export HF_ENDPOINT="${REMTP_HF_ENDPOINT:-https://huggingface.co}"

MODEL_ROOT="${MODEL_ROOT:-$PROJECT_DIR/models}"
BASE_DIR="${BASE_DIR:-$MODEL_ROOT/MiMo-7B-Base}"
MTP_DIR="${MTP_DIR:-$MODEL_ROOT/MiMo-7B-MTPs}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_ROOT/MiMo-7B-Base-MTP3}"
BASE_REVISION="${BASE_REVISION:-c72df4586cb8bdeebd65f36929cd3385a6566fbe}"
MTP_REVISION="${MTP_REVISION:-791db75c0da873570fba05c82a6295ae6d72dec1}"

if ! command -v hf >/dev/null 2>&1; then
  python -m pip install -U 'huggingface_hub[cli]'
fi

mkdir -p "$MODEL_ROOT"

echo "[MiMo] downloading base -> $BASE_DIR"
echo "[MiMo] Hugging Face endpoint: $HF_ENDPOINT"
hf download XiaomiMiMo/MiMo-7B-Base \
  --revision "$BASE_REVISION" \
  --local-dir "$BASE_DIR"

echo "[MiMo] downloading extra MTP layers -> $MTP_DIR"
hf download XiaomiMiMo/MiMo-7B-MTPs \
  --revision "$MTP_REVISION" \
  --local-dir "$MTP_DIR"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Overlay already exists: $OUTPUT_DIR" >&2
  echo "Validate it with: python -m remtp.mimo_checkpoint check '$OUTPUT_DIR'" >&2
  exit 2
fi

PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
python -m remtp.mimo_checkpoint build \
  --base "$BASE_DIR" \
  --mtp "$MTP_DIR" \
  --output "$OUTPUT_DIR"

echo "[MiMo] ready: $OUTPUT_DIR"
echo "[MiMo] large weights remain in the two download directories; do not move them independently."
