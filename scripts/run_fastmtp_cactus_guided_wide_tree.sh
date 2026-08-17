#!/usr/bin/env bash
# Wider target-dominant tree with bounded Cactus dead-frontier rescue.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the wider Cactus-guided D=3 dynamic tree on the audited N=100 tasks while
reusing unchanged Native/Cactus/SpecCascade baselines.

  RUN_TAG=fastmtp_cactus_guided_wide_n100_v3 \
    ./scripts/run_fastmtp_cactus_guided_wide_tree.sh

Normal tree survival remains target-relative. Cactus is consulted only when a
surviving parent has no normally surviving child. Compared with the v2 guided
profile, this experiment lowers the target/evidence floors and permits at most
two dead-frontier rescues along one path. It remains an approximate relaxed
decoder and is not guaranteed to match Chain Cactus MAL.
EOF
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown arguments; use --help." >&2
  exit 2
fi

export GSM8K_SAMPLES="${GSM8K_SAMPLES:-100}"
export HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-100}"
export BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_DIR/results/fastmtp_balanced_wide_n100}"
export RUN_TAG="${RUN_TAG:-fastmtp_cactus_guided_wide_n100_$(date +%Y%m%d_%H%M%S)}"

export DYNAMIC_SUPPORT_MODE=cactus_guided
export DYNAMIC_CACTUS_DELTA="${DYNAMIC_CACTUS_DELTA:-1.0}"
export DYNAMIC_CACTUS_TARGET_WEIGHT="${DYNAMIC_CACTUS_TARGET_WEIGHT:-0.50}"
export DYNAMIC_MAX_GUIDED_RESCUES="${DYNAMIC_MAX_GUIDED_RESCUES:-2}"
export DYNAMIC_MIN_COVERAGE="${DYNAMIC_MIN_COVERAGE:-0.01}"
export DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.25}"
export DYNAMIC_CONFIRMATION_MIN_RELATIVE="${DYNAMIC_CONFIRMATION_MIN_RELATIVE:-0.01}"
export DYNAMIC_PROPOSAL_SUPPORT_RATIO="${DYNAMIC_PROPOSAL_SUPPORT_RATIO:-0.25}"
export DYNAMIC_BETA="${DYNAMIC_BETA:-2.0}"
export DYNAMIC_FRONTIER_RESCUE=0
export DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.001}"

cat <<EOF
FastMTP wider target-dominant Cactus-guided tree
  logical depth          : 3
  normal survival        : target-relative >= $DYNAMIC_TAU_RELAX
  minimum tree coverage  : $DYNAMIC_MIN_COVERAGE
  dead-frontier prior    : Cactus delta=$DYNAMIC_CACTUS_DELTA
  guided target weight   : $DYNAMIC_CACTUS_TARGET_WEIGHT
  guided evidence        : relative >= $DYNAMIC_CONFIRMATION_MIN_RELATIVE plus
                           P/Q >= $DYNAMIC_PROPOSAL_SUPPORT_RATIO,
                           retained future top-1, or medium target support
  path risk control      : at most $DYNAMIC_MAX_GUIDED_RESCUES guided rescues/path
  minimum target P(y)    : $DYNAMIC_RESCUE_MIN_TARGET_PROB
  length reward beta     : $DYNAMIC_BETA
  proposal tree          : Q-driven soft-reach, maximum 10 nodes, no hard quota
  target forward/round   : 1
  output                 : $PROJECT_DIR/results/$RUN_TAG/comparison.md
EOF

exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh"
