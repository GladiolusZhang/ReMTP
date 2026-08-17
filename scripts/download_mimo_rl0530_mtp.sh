#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Download immutable MiMo-7B-RL-0530 and MiMo-7B-MTPs revisions, then build a
symlink-based local MTP3 overlay. The default endpoint is official Hugging Face.

Usage:
  ./scripts/download_mimo_rl0530_mtp.sh

The default endpoint is https://huggingface.co.
Interrupted downloads are resumed by huggingface_hub.
EOF
  exit 0
fi
if (( $# > 0 )); then
  echo "This script accepts no positional arguments. Use --help." >&2
  exit 2
fi

# The endpoint and proxy policy apply only to this process and its children;
# they do not modify shell startup files or the user's persistent environment.
if [[ "${CLEAR_DOWNLOAD_PROXY:-0}" == "1" ]]; then
  unset http_proxy https_proxy all_proxy no_proxy
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
fi
export HF_ENDPOINT="${HF_DOWNLOAD_ENDPOINT:-https://huggingface.co}"
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

MODEL_ROOT="${MODEL_ROOT:-$PROJECT_DIR/models}"
TARGET_DIR="${TARGET_DIR:-$MODEL_ROOT/MiMo-7B-RL-0530}"
MTP_DIR="${MTP_DIR:-$MODEL_ROOT/MiMo-7B-MTPs}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_ROOT/MiMo-7B-RL-0530-MTP3}"
TARGET_REVISION="${TARGET_REVISION:-323400599af3903adc2a536d6340a23fee88d2e0}"
MTP_REVISION="${MTP_REVISION:-791db75c0da873570fba05c82a6295ae6d72dec1}"

if ! command -v hf >/dev/null 2>&1; then
  python -m pip install -U 'huggingface_hub[cli]'
fi

mkdir -p "$MODEL_ROOT"

echo "[MiMo RL-0530] endpoint: $HF_ENDPOINT"

direct_file() {
  local repo="$1"
  local revision="$2"
  local destination_root="$3"
  local filename="$4"
  local expected_sha="${5:-}"
  local destination="$destination_root/$filename"
  local partial="$destination.part"
  local url="${HF_ENDPOINT%/}/${repo}/resolve/${revision}/${filename}?download=true"
  if [[ -s "$destination" ]]; then
    return
  fi
  mkdir -p "$(dirname "$destination")"

  # Reuse bytes left by a failed huggingface_hub download when its LFS SHA is
  # known. cp --reflink is instant on supporting filesystems and harmless
  # otherwise; the original Hub cache remains untouched.
  if [[ ! -s "$partial" && -n "$expected_sha" ]]; then
    local cached=""
    cached="$(find "$destination_root/.cache/huggingface/download" \
      -maxdepth 1 -type f -name "*.${expected_sha}.incomplete" \
      -size +0c -print -quit 2>/dev/null || true)"
    if [[ -n "$cached" ]]; then
      echo "[MiMo direct] reusing $(stat -c %s "$cached") cached bytes for $filename"
      cp --reflink=auto "$cached" "$partial"
    fi
  fi

  echo "[MiMo direct] $repo/$filename"
  local resume_args=()
  if [[ -s "$partial" ]]; then
    resume_args=(--continue-at -)
  fi
  curl --location --fail --show-error \
    --retry 30 --retry-delay 2 --retry-all-errors \
    "${resume_args[@]}" --output "$partial" "$url"
  mv "$partial" "$destination"
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  local stamp="$path.sha256-${expected}.ok"
  if [[ -f "$stamp" ]]; then
    return
  fi
  echo "[MiMo direct] verifying sha256: $(basename "$path")"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch for $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual  : $actual" >&2
    echo "Move the bad file aside and rerun the downloader." >&2
    exit 1
  fi
  : > "$stamp"
}

download_target_direct() {
  local repo="XiaomiMiMo/MiMo-7B-RL-0530"
  local small_files=(
    .gitattributes README.md added_tokens.json config.json
    configuration_mimo.py generation_config.json merges.txt
    model.safetensors.index.json modeling_mimo.py special_tokens_map.json
    tokenizer.json tokenizer_config.json vocab.json
  )
  for filename in "${small_files[@]}"; do
    direct_file "$repo" "$TARGET_REVISION" "$TARGET_DIR" "$filename"
  done
  direct_file "$repo" "$TARGET_REVISION" "$TARGET_DIR" \
    model-00001-of-00004.safetensors \
    9b4d1566074fe7c7afaa396ba1eb95a34890b48a8433e89a19fa99c4b4c951db
  direct_file "$repo" "$TARGET_REVISION" "$TARGET_DIR" \
    model-00002-of-00004.safetensors \
    06dde056e1e06ce66ab72b70da852f2a92ad6e2da4c99f5860b8273ad3dd0e0b
  direct_file "$repo" "$TARGET_REVISION" "$TARGET_DIR" \
    model-00003-of-00004.safetensors \
    445637e4aa2cba64070751acaf06017ae29cd5cd57dfb479a3b1a5ee2c9897c5
  direct_file "$repo" "$TARGET_REVISION" "$TARGET_DIR" \
    model-00004-of-00004.safetensors \
    1ea475d9ab1093ce5ceebf8852b828e1b7c9829351cfc911d24e0f1e4d778056
  verify_sha256 "$TARGET_DIR/model-00001-of-00004.safetensors" \
    9b4d1566074fe7c7afaa396ba1eb95a34890b48a8433e89a19fa99c4b4c951db
  verify_sha256 "$TARGET_DIR/model-00002-of-00004.safetensors" \
    06dde056e1e06ce66ab72b70da852f2a92ad6e2da4c99f5860b8273ad3dd0e0b
  verify_sha256 "$TARGET_DIR/model-00003-of-00004.safetensors" \
    445637e4aa2cba64070751acaf06017ae29cd5cd57dfb479a3b1a5ee2c9897c5
  verify_sha256 "$TARGET_DIR/model-00004-of-00004.safetensors" \
    1ea475d9ab1093ce5ceebf8852b828e1b7c9829351cfc911d24e0f1e4d778056
}

download_mtp_direct() {
  local repo="XiaomiMiMo/MiMo-7B-MTPs"
  local files=(
    .gitattributes README.md config.json configuration_mimo.py
    generation_config.json merges.txt modeling_mimo.py tokenizer.json
    tokenizer_config.json vocab.json
  )
  for filename in "${files[@]}"; do
    direct_file "$repo" "$MTP_REVISION" "$MTP_DIR" "$filename"
  done
  direct_file "$repo" "$MTP_REVISION" "$MTP_DIR" model.safetensors \
    ac1f4a0260d41f60ddd273118268c89270304de2283848ae831cc4738d34c91f
  verify_sha256 "$MTP_DIR/model.safetensors" \
    ac1f4a0260d41f60ddd273118268c89270304de2283848ae831cc4738d34c91f
}

echo "[MiMo RL-0530] downloading target -> $TARGET_DIR"
if [[ "${REMTP_DIRECT_HF_DOWNLOAD:-0}" == "1" ]]; then
  download_target_direct
else
  hf download XiaomiMiMo/MiMo-7B-RL-0530 \
    --revision "$TARGET_REVISION" \
    --local-dir "$TARGET_DIR"
fi

echo "[MiMo RL-0530] downloading/reusing extra MTP layers -> $MTP_DIR"
if [[ "${REMTP_DIRECT_HF_DOWNLOAD:-0}" == "1" ]]; then
  download_mtp_direct
else
  hf download XiaomiMiMo/MiMo-7B-MTPs \
    --revision "$MTP_REVISION" \
    --local-dir "$MTP_DIR"
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[MiMo RL-0530] overlay exists; validating it: $OUTPUT_DIR"
  python -m remtp.mimo_checkpoint check "$OUTPUT_DIR"
else
  python -m remtp.mimo_checkpoint build \
    --base "$TARGET_DIR" \
    --mtp "$MTP_DIR" \
    --output "$OUTPUT_DIR"
fi

# Prove that all runtime-critical checkpoint files can be resolved locally.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m remtp.mimo_checkpoint check "$OUTPUT_DIR" >/dev/null

echo
echo "[MiMo RL-0530] ready"
echo "  target-only : $TARGET_DIR"
echo "  MTP3 overlay: $OUTPUT_DIR"
echo "The overlay uses symlinks; do not move its source directories independently."
