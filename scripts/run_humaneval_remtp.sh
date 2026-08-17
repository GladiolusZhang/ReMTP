#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Run HumanEval-164: Native MTP, Cactus, SpecCascade and ReMTP."
  echo "Usage: $0"
  exit 0
fi
if (( $# > 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi

# Frozen comparison: the main ReMTP method and the same three baselines used
# by GSM8K.  Only ReMTP is newly generated/evaluated.
export SAMPLES="${SAMPLES:-164}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export SEED="${SEED:-42}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export MTP_TOKENS="${MTP_TOKENS:-6}"
export PROFILES="native_mtp cactus spec_cascade remtp"
export REUSE_PROFILES="native_mtp cactus spec_cascade"
export REUSE_RESULT_ROOTS="${HUMANEVAL_REFERENCE_ROOT:-$PROJECT_DIR/results/humaneval_comparison_ultra_anchored_20260804_224028}"
export REUSE_REQUIRED=1
export RUN_TAG="${RUN_TAG:-remtp_$(date +%Y%m%d_%H%M%S)}"

exec "$PROJECT_DIR/scripts/run_humaneval_comparison.sh"
