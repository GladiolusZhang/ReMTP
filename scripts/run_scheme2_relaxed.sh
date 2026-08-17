#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"

case "$MODE" in
  gsm8k)
    exec "$PROJECT_DIR/scripts/run_gsm8k_scheme2_relaxed.sh"
    ;;
  humaneval)
    exec "$PROJECT_DIR/scripts/run_humaneval_scheme2_relaxed.sh"
    ;;
  all)
    "$PROJECT_DIR/scripts/run_gsm8k_scheme2_relaxed.sh"
    "$PROJECT_DIR/scripts/run_humaneval_scheme2_relaxed.sh"
    ;;
  *)
    echo "Usage: $0 [gsm8k|humaneval|all]" >&2
    exit 2
    ;;
esac
