#!/usr/bin/env bash
# Aggressive target-anchored dual-support D=3 experiment.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the aggressive D=3 soft-reach + dual-support experiment.

  RUN_TAG=fastmtp_dual_support_d3_full \
    ./scripts/run_fastmtp_dual_support_tree.sh

Default support rule:
  coverage/EOS safety AND (P(y)/P(top1) >= 0.10 OR P(y)/Q(y) >= 1.0)

The script reuses the completed Native/Cactus/SpecCascade baselines and runs
only the new dynamic-tree method.
EOF
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown arguments; use --help." >&2
  exit 2
fi

export DYNAMIC_SUPPORT_MODE="${DYNAMIC_SUPPORT_MODE:-dual}"
export DYNAMIC_PROPOSAL_SUPPORT_RATIO="${DYNAMIC_PROPOSAL_SUPPORT_RATIO:-1.0}"
export DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.10}"

exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh" "$@"
