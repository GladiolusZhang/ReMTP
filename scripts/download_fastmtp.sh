#!/usr/bin/env bash
# Download TencentBAC/FastMTP model
# FastMTP is a Qwen2-based 8B model with trained MTP layers (1 layer, 3 speculative steps)
# Paper: https://arxiv.org/abs/2509.18362
# Performance: 2.03x speedup, 82% better than vanilla MTP

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODEL_NAME="TencentBAC/FastMTP"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/models/FastMTP}"
REVISION="${REVISION:-main}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

echo "=== Downloading TencentBAC/FastMTP ==="
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Revision: $REVISION"
echo "Endpoint: $HF_ENDPOINT"
echo

if [[ -d "$OUTPUT_DIR" && -f "$OUTPUT_DIR/config.json" ]]; then
  echo "Model directory already exists: $OUTPUT_DIR"
  echo "To re-download, remove the directory first:"
  echo "  rm -rf $OUTPUT_DIR"
  exit 1
fi

if ! command -v huggingface-cli &>/dev/null; then
  echo "Error: huggingface-cli not found"
  echo "Install it with: pip install -U huggingface-hub"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Downloading model files..."
echo "This will download ~15.7 GB (4 safetensors shards + config + tokenizer)"
echo

# Download with huggingface-cli
HF_ENDPOINT="$HF_ENDPOINT" \
HF_HUB_DISABLE_XET=1 \
  huggingface-cli download "$MODEL_NAME" \
  --revision "$REVISION" \
  --local-dir "$OUTPUT_DIR" \
  --local-dir-use-symlinks False \
  --max-workers 4

echo
echo "=== Download Complete ==="
echo

# Verify critical files
critical_files=(
  "config.json"
  "modeling_mimo.py"
  "configuration_mimo.py"
  "tokenizer.json"
  "model.safetensors.index.json"
)

missing_files=()
for file in "${critical_files[@]}"; do
  if [[ ! -f "$OUTPUT_DIR/$file" ]]; then
    missing_files+=("$file")
  fi
done

if (( ${#missing_files[@]} > 0 )); then
  echo "Error: Missing critical files:"
  printf '  - %s\n' "${missing_files[@]}"
  exit 1
fi

echo "✓ All critical files present"
echo

# Display model info
echo "=== Model Information ==="
python3 -c "
import json
config = json.load(open('$OUTPUT_DIR/config.json'))
print(f\"Architecture: {config['architectures'][0]}\")
print(f\"Model Type: {config['model_type']}\")
print(f\"Hidden Size: {config['hidden_size']}\")
print(f\"Num Layers: {config['num_hidden_layers']}\")
print(f\"Vocab Size: {config['vocab_size']}\")
print(f\"MTP Layers: {config['num_nextn_predict_layers']}\")
print(f\"Speculative Steps: {config['num_speculative_steps']}\")
print(f\"Max Position Embeddings: {config['max_position_embeddings']}\")
print(f\"Tokenizer: {config.get('tokenizer_class', ['Unknown'])[0]}\")
"
echo

echo "Model ready at: $OUTPUT_DIR"
echo
echo "Key characteristics:"
echo "  - Base: MiMo-7B-RL checkpoint (Qwen2-based, 8B parameters)"
echo "  - Architecture: MiMo (Qwen2 + MTP support)"
echo "  - MTP: 1 trained layer with position-shared weights"
echo "  - Performance: 2.03x speedup, lossless quality"
echo "  - Paper: https://arxiv.org/abs/2509.18362"
echo
echo "Note: FastMTP is trained on MiMo-7B-RL (Xiaomi's Qwen2-based model)"
echo "      MiMo architecture inherits from Qwen2 but adds MTP layers."
