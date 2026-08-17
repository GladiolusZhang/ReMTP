#!/usr/bin/env bash
# Recommended target-confirmed relaxation after rejected-token audit.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the recommended D=3 soft-reach + target-confirmed relaxation.

  RUN_TAG=fastmtp_confirmed_support_d3_full \
    ./scripts/run_fastmtp_confirmed_support_tree.sh

Default support rule:
  normal: P(y)/P(top1) >= 0.35
  extra : P(y)/P(top1) >= 0.05 AND
          [P(y)/Q(y) >= 1 OR a retained child is target next-step top-1]

Coverage and EOS protection remain unchanged. The script reuses completed
Native/Cactus/SpecCascade baselines and runs only the new tree method.
EOF
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown arguments; use --help." >&2
  exit 2
fi

export DYNAMIC_SUPPORT_MODE="${DYNAMIC_SUPPORT_MODE:-confirmed}"
export DYNAMIC_PROPOSAL_SUPPORT_RATIO="${DYNAMIC_PROPOSAL_SUPPORT_RATIO:-1.0}"
export DYNAMIC_CONFIRMATION_MIN_RELATIVE="${DYNAMIC_CONFIRMATION_MIN_RELATIVE:-0.05}"
export DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.35}"

exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh" "$@"
