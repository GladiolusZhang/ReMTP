#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Screen Target-Band Rescue on a shared HumanEval subset.

This compares Native MTP, Cactus, SpecCascade, the best exact-top-1 setting,
and four increasingly broad Target-Band settings. Model runs remain serial;
isolated Docker evaluation is parallelized only after each server stops.

Usage:
  ./scripts/run_humaneval_target_band_screen.sh

Defaults:
  SAMPLES=40 TEMPERATURE=0.7 SEED=42 MTP_TOKENS=6 EVAL_WORKERS=4

For the full official split:
  SAMPLES=164 ./scripts/run_humaneval_target_band_screen.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

export SAMPLES="${SAMPLES:-40}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export SEED="${SEED:-42}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export MTP_TOKENS=6
export EVAL_WORKERS="${EVAL_WORKERS:-4}"
export RUN_TAG="${RUN_TAG:-target_band_screen_$(date +%Y%m%d_%H%M%S)}"
export PROFILES="native_mtp cactus spec_cascade target_mode_top1_m010 target_band_r2_g025 target_band_r4_g050 target_band_r4_g075 target_band_r8_g100"

"$PROJECT_DIR/scripts/run_humaneval_comparison.sh"

RESULT_DIR="$PROJECT_DIR/results/humaneval_comparison_${RUN_TAG}"
echo
echo "Target-Band screen complete: $RESULT_DIR/comparison.md"
