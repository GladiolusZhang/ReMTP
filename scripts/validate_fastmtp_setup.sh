#!/usr/bin/env bash
# Quick validation for FastMTP integration

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== FastMTP Integration Validation ==="
echo

# Check Python environment
echo "1. Checking Python environment..."
if [[ ! -d ".venv" ]]; then
  echo "  ❌ Virtual environment not found"
  exit 1
fi
source .venv/bin/activate
echo "  ✓ Virtual environment activated"

# Check new modules
echo
echo "2. Checking FastMTP modules..."
for module in remtp.fastmtp_worker remtp.fastmtp_three_way_report; do
  if python -c "import ${module%.*}" 2>/dev/null; then
    echo "  ✓ $module"
  else
    echo "  ❌ $module not found"
  fi
done

# Check scripts
echo
echo "3. Checking FastMTP scripts..."
for script in scripts/download_fastmtp.sh \
              scripts/serve_fastmtp_native.sh \
              scripts/serve_fastmtp_cactus.sh \
              scripts/serve_fastmtp_spec_cascade.sh \
              scripts/run_fastmtp_three_way_50.sh; do
  if [[ -x "$script" ]]; then
    echo "  ✓ $script (executable)"
  elif [[ -f "$script" ]]; then
    echo "  ⚠️  $script (not executable)"
    chmod +x "$script"
    echo "     ✓ Fixed"
  else
    echo "  ❌ $script missing"
  fi
done

# Check if FastMTP model exists
echo
echo "4. Checking FastMTP model..."
FASTMTP_MODEL="models/FastMTP"
if [[ -d "$FASTMTP_MODEL" && -f "$FASTMTP_MODEL/config.json" ]]; then
  echo "  ✓ FastMTP model found at $FASTMTP_MODEL"
  python3 -c "
import json
config = json.load(open('$FASTMTP_MODEL/config.json'))
print(f\"    Model: {config['architectures'][0]}\")
print(f\"    MTP Layers: {config['num_nextn_predict_layers']}\")
print(f\"    Speculative Steps: {config['num_speculative_steps']}\")
"
else
  echo "  ⚠️  FastMTP model not found"
  echo "     Download with: ./scripts/download_fastmtp.sh"
fi

# Check data files
echo
echo "5. Checking data files..."
for file in data/gsm8k/test.jsonl data/humaneval/HumanEval.jsonl.gz; do
  if [[ -f "$file" ]]; then
    echo "  ✓ $file"
  else
    echo "  ❌ $file missing"
  fi
done

# Check Docker
echo
echo "6. Checking Docker..."
if docker info >/dev/null 2>&1; then
  echo "  ✓ Docker available"
  if docker image inspect python:3-slim >/dev/null 2>&1; then
    echo "  ✓ python:3-slim image available"
  else
    echo "  ⚠️  python:3-slim missing (run: docker pull python:3-slim)"
  fi
else
  echo "  ❌ Docker unavailable"
fi

# Check port
echo
echo "7. Checking port availability..."
if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "  ⚠️  Port 8000 in use (stop with: pkill -f 'vllm serve')"
else
  echo "  ✓ Port 8000 available"
fi

echo
echo "=== Validation Complete ==="
echo
echo "Next steps:"
echo "  1. Download FastMTP model (~15.7 GB, one-time):"
echo "     ./scripts/download_fastmtp.sh"
echo
echo "  2. Run three-way comparison (50 samples, ~1.5-2 hours):"
echo "     RUN_TAG=fastmtp_three_way_50_\$(date +%Y%m%d) \\"
echo "       ./scripts/run_fastmtp_three_way_50.sh"
echo
echo "Expected improvements with FastMTP:"
echo "  - Much higher baseline MAL (trained MTP vs untrained)"
echo "  - Better acceptance rates (paper: 2.03x speedup)"
echo "  - Clear relaxation effects on quality baseline"
