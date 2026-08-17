#!/usr/bin/env bash
# Quick validation script to check if all components are ready

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== ReMTP Four-Way Comparison Setup Validation ==="
echo

# Check Python environment
echo "1. Checking Python environment..."
if [[ ! -d ".venv" ]]; then
  echo "  ❌ Virtual environment not found at .venv"
  exit 1
fi
source .venv/bin/activate
echo "  ✓ Virtual environment activated"

# Check required Python modules
echo
echo "2. Checking Python modules..."
for module in remtp.gsm8k_benchmark remtp.humaneval_benchmark remtp.humaneval_evaluator \
              remtp.mimo_checkpoint remtp.tree_benchmark_metrics remtp.mimo_four_way_report; do
  if python -c "import ${module%.*}" 2>/dev/null; then
    echo "  ✓ $module"
  else
    echo "  ❌ $module not found"
  fi
done

# Check data files
echo
echo "3. Checking data files..."
for file in data/gsm8k/test.jsonl data/humaneval/HumanEval.jsonl.gz; do
  if [[ -f "$file" ]]; then
    echo "  ✓ $file"
  else
    echo "  ❌ $file missing"
  fi
done

# Check MiMo model
echo
echo "4. Checking MiMo model..."
MTP_MODEL="models/MiMo-7B-Base-MTP3"
if [[ -d "$MTP_MODEL" && -f "$MTP_MODEL/config.json" ]]; then
  echo "  ✓ $MTP_MODEL found"
  python -m remtp.mimo_checkpoint check "$MTP_MODEL" 2>&1 | grep -q "valid" && echo "  ✓ Model checkpoint valid"
else
  echo "  ❌ $MTP_MODEL not found"
  echo "     Run: ./scripts/download_mimo_7b_mtp.sh"
fi

# Check Docker
echo
echo "5. Checking Docker..."
if docker info >/dev/null 2>&1; then
  echo "  ✓ Docker available"
  if docker image inspect python:3-slim >/dev/null 2>&1; then
    echo "  ✓ python:3-slim image available"
  else
    echo "  ⚠️  python:3-slim image missing"
    echo "     Run: docker pull python:3-slim"
  fi
else
  echo "  ❌ Docker unavailable"
fi

# Check if port 8000 is available
echo
echo "6. Checking port availability..."
if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "  ⚠️  Port 8000 is already in use"
  echo "     Stop the running server first"
else
  echo "  ✓ Port 8000 available"
fi

# Check scripts
echo
echo "7. Checking scripts..."
for script in scripts/run_mimo_four_way_comparison_50.sh \
              scripts/serve_mimo_native.sh \
              scripts/serve_mimo_cactus.sh \
              scripts/serve_mimo_spec_cascade.sh \
              scripts/serve_mimo_dynamic_tree.sh; do
  if [[ -x "$script" ]]; then
    echo "  ✓ $script (executable)"
  elif [[ -f "$script" ]]; then
    echo "  ⚠️  $script (not executable, fixing...)"
    chmod +x "$script"
    echo "     ✓ Fixed"
  else
    echo "  ❌ $script missing"
  fi
done

echo
echo "=== Validation Complete ==="
echo
echo "To run the four-way comparison (50 samples):"
echo "  cd $PROJECT_DIR"
echo "  source .venv/bin/activate"
echo "  RUN_TAG=mimo_four_way_50_\$(date +%Y%m%d) \\"
echo "    ./scripts/run_mimo_four_way_comparison_50.sh"
echo
echo "Results will be saved to: results/mimo_four_way_50_*/comparison.md"
