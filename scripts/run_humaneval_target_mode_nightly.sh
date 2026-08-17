#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Run the complete MTP=6 Target-Mode overnight ablation on HumanEval.

The suite generates and officially evaluates these groups on the same tasks:
  1. Native probabilistic MTP, Cactus, and SpecCascade TokenV3 references.
  2. Four p>=0.5 safety-baseline parameter settings.
  3. Four exact target-top-1 settings with log-prob margins 0/.10/.25/.50.

Model profiles run serially on one GPU for fair throughput measurements.
After each model server stops, isolated HumanEval Docker jobs run in parallel.

Usage:
  ./scripts/run_humaneval_target_mode_nightly.sh

Useful overrides:
  SAMPLES=164
  SAMPLE_SEED=20260802
  TEMPERATURE=0.7
  SEED=42
  MAX_TOKENS=512
  EVAL_WORKERS=4
  RUN_TAG=target_mode_nightly_YYYYMMDD
  NIGHTLY_PROFILES="native_mtp cactus target_mode_top1_m025"  # debug only

Quick end-to-end smoke (still includes all 11 profiles):
  SAMPLES=2 MAX_TOKENS=64 EVAL_WORKERS=2 \
    ./scripts/run_humaneval_target_mode_nightly.sh
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

export SAMPLES="${SAMPLES:-164}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export SEED="${SEED:-42}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export MTP_TOKENS=6
export EVAL_WORKERS="${EVAL_WORKERS:-4}"
export RUN_TAG="${RUN_TAG:-target_mode_nightly_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_NIGHTLY_PROFILES="native_mtp cactus spec_cascade target_mode_p05_b060 target_mode_p05_b075 target_mode_p05_b090 target_mode_p05_b095 target_mode_top1_m000 target_mode_top1_m010 target_mode_top1_m025 target_mode_top1_m050"
export PROFILES="${NIGHTLY_PROFILES:-$DEFAULT_NIGHTLY_PROFILES}"

echo "Nightly Target-Mode suite: $RUN_TAG"
echo "Model runs are serial; Docker evaluation workers=$EVAL_WORKERS"
echo "Profiles: $PROFILES"

"$PROJECT_DIR/scripts/run_humaneval_comparison.sh"

RESULT_DIR="$PROJECT_DIR/results/humaneval_comparison_${RUN_TAG}"
echo
echo "Nightly suite complete. Open:"
echo "  $RESULT_DIR/comparison.md"
echo "Machine-readable results:"
echo "  $RESULT_DIR/comparison.json"
echo "  $RESULT_DIR/comparison.csv"
