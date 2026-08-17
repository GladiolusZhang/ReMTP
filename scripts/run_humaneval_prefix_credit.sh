#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Compare Prefix-Credit native MTP against reused, locked HumanEval references.

The main comparison keeps the same target top-1 confidence mask and varies
only the native-MTP block allocator:
  - target_mode_top1_m010: accurate token-wise relaxed baseline;
  - prefix_credit_token_cap: same mask + joint verification, old h<=q cap;
  - prefix_credit_atomic: over-Q credit only for fully repairable prefixes;
  - prefix_credit_g025/g050/g100: ordered-Q prefix credit with increasing
    minimum joint-prefix gain per unit TV.

Default full run:
  ./scripts/run_humaneval_prefix_credit.sh

Useful overrides:
  SAMPLES=164
  SAMPLE_SEED=20260802
  TEMPERATURE=0.7
  SEED=42
  MAX_TOKENS=512
  EVAL_WORKERS=4
  RUN_TAG=prefix_credit_YYYYMMDD
  PREFIX_CREDIT_NEW_PROFILES="prefix_credit_token_cap prefix_credit_g050"
  PREFIX_CREDIT_REFERENCE_ROOT=results/humaneval_comparison_target_band_full_01

The historical Native MTP, Cactus, SpecCascade and target-top1 rows are
protocol-checked and reused. Only PREFIX_CREDIT_NEW_PROFILES are executed.
The default reference cache is the locked 164-task run, so this wrapper keeps
SAMPLES=164 unless a different protocol-compatible reference root is supplied.
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
export PREFIX_CREDIT_AUDIT_INTERVAL="${PREFIX_CREDIT_AUDIT_INTERVAL:-100}"
export RUN_TAG="${RUN_TAG:-prefix_credit_$(date +%Y%m%d_%H%M%S)}"
REFERENCE_PROFILES="native_mtp cactus spec_cascade target_mode_top1_m010"
DEFAULT_NEW_PROFILES="prefix_credit_token_cap prefix_credit_atomic prefix_credit_g025 prefix_credit_g050 prefix_credit_g100"
NEW_PROFILES="${PREFIX_CREDIT_NEW_PROFILES:-$DEFAULT_NEW_PROFILES}"
export PROFILES="$REFERENCE_PROFILES $NEW_PROFILES"
export REUSE_PROFILES="$REFERENCE_PROFILES"
export REUSE_RESULT_ROOTS="${PREFIX_CREDIT_REFERENCE_ROOT:-$PROJECT_DIR/results/humaneval_comparison_target_band_full_01}"
export REUSE_REQUIRED=1

echo "Prefix-Credit HumanEval suite: $RUN_TAG"
echo "Historical rows (not rerun): $REFERENCE_PROFILES"
echo "New methods to run: $NEW_PROFILES"
echo "MTP=6 temperature=$TEMPERATURE samples=$SAMPLES"

"$PROJECT_DIR/scripts/run_humaneval_comparison.sh"

RESULT_DIR="$PROJECT_DIR/results/humaneval_comparison_${RUN_TAG}"
echo
echo "Suite complete. Open:"
echo "  $RESULT_DIR/comparison.md"
echo "Mechanism audit signals are in each prefix-credit server log when"
echo "PREFIX_CREDIT_AUDIT_INTERVAL is set."
