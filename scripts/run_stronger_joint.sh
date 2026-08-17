#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"

case "$MODE" in
  gsm8k)
    exec "$PROJECT_DIR/scripts/run_gsm8k_stronger_joint.sh"
    ;;
  humaneval)
    exec "$PROJECT_DIR/scripts/run_humaneval_stronger_joint.sh"
    ;;
  all)
    "$PROJECT_DIR/scripts/run_gsm8k_stronger_joint.sh"
    "$PROJECT_DIR/scripts/run_humaneval_stronger_joint.sh"
    ;;
  *)
    echo "Usage: $0 [gsm8k|humaneval|all]" >&2
    exit 2
    ;;
esac
