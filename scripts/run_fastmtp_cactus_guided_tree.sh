#!/usr/bin/env bash
# Dynamic tree whose target verifier uses Cactus only to calibrate a dead frontier.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the target-dominant Cactus-guided D=3 dynamic tree on the audited N=100
tasks while reusing the unchanged Native/Cactus/SpecCascade baselines.

  RUN_TAG=fastmtp_cactus_guided_tree_n100_v2 \
    ./scripts/run_fastmtp_cactus_guided_tree.sh

This is not Cactus applied independently to every tree node. Normal survival
still uses target-relative support. Cactus is consulted only when a surviving
parent has no normally surviving child, and only one tree-supported candidate
may be rescued per path. Proposal construction remains Q-driven soft-reach,
with no per-layer or per-parent node quota.
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
export RUN_TAG="${RUN_TAG:-fastmtp_cactus_guided_tree_n100_$(date +%Y%m%d_%H%M%S)}"

export DYNAMIC_SUPPORT_MODE=cactus_guided
export DYNAMIC_CACTUS_DELTA="${DYNAMIC_CACTUS_DELTA:-1.0}"
export DYNAMIC_CACTUS_TARGET_WEIGHT="${DYNAMIC_CACTUS_TARGET_WEIGHT:-0.65}"
export DYNAMIC_MIN_COVERAGE="${DYNAMIC_MIN_COVERAGE:-0.05}"
export DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.35}"
export DYNAMIC_CONFIRMATION_MIN_RELATIVE="${DYNAMIC_CONFIRMATION_MIN_RELATIVE:-0.05}"
export DYNAMIC_PROPOSAL_SUPPORT_RATIO="${DYNAMIC_PROPOSAL_SUPPORT_RATIO:-1.0}"
export DYNAMIC_BETA="${DYNAMIC_BETA:-1.5}"
export DYNAMIC_FRONTIER_RESCUE=0
export DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.001}"

cat <<EOF
FastMTP target-dominant Cactus-guided tree
  logical depth          : 3
  normal survival        : target-relative >= $DYNAMIC_TAU_RELAX
  dead-frontier prior    : Cactus delta=$DYNAMIC_CACTUS_DELTA
  guided target weight   : $DYNAMIC_CACTUS_TARGET_WEIGHT
  guided evidence floor  : relative >= $DYNAMIC_CONFIRMATION_MIN_RELATIVE plus
                           P/Q, retained future top-1, or medium target support
  path risk control      : at most one guided rescue per path
  length reward beta     : $DYNAMIC_BETA
  proposal tree          : Q-driven soft-reach, maximum 10 nodes, no hard quota
  target forward/round   : 1
  output                 : $PROJECT_DIR/results/$RUN_TAG/comparison.md
EOF

exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh"
