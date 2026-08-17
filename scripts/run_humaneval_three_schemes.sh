#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run HumanEval for Scheme 1, Scheme 2, Scheme 3 adaptive-chain fallback and
Scheme 1+2, while reusing protocol-compatible Native/Cactus/SpecCascade rows.

Usage:
  ./scripts/run_humaneval_four_schemes.sh

Useful overrides:
  SAMPLES=164 TEMPERATURE=0.7 SEED=42 MAX_TOKENS=512 MTP_TOKENS=6
  HUMANEVAL_REFERENCE_ROOT=results/humaneval_comparison_target_band_full_01
  RUN_TAG=<name>
EOF
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown argument: $1" >&2
  exit 2
fi

cat <<'EOF'
HumanEval comparison:
  historical rows: Native MTP, Cactus, SpecCascade TokenV3
  new rows: Scheme 1, Scheme 2, Scheme 3 adaptive-chain fallback, Scheme 1+2

Only the four new profiles are generated. Historical rows are copied after a
strict protocol/data/container fingerprint check.
EOF

export SAMPLES="${SAMPLES:-164}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export SEED="${SEED:-42}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export MTP_TOKENS="${MTP_TOKENS:-6}"
export PROFILES="native_mtp cactus spec_cascade scheme1 scheme2 scheme3 scheme12"
export REUSE_PROFILES="native_mtp cactus spec_cascade"
export REUSE_RESULT_ROOTS="${HUMANEVAL_REFERENCE_ROOT:-$PROJECT_DIR/results/humaneval_comparison_target_band_full_01}"
export REUSE_REQUIRED=1
export RUN_TAG="${RUN_TAG:-three_schemes_$(date +%Y%m%d_%H%M%S)}"

exec "$PROJECT_DIR/scripts/run_humaneval_comparison.sh"
