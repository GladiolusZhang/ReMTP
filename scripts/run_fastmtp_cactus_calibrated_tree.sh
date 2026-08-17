#!/usr/bin/env bash
# D=3 tree whose node survival uses the same local Cactus delta as the chain.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run only the new Cactus-calibrated D=3 tree on the audited N=100 tasks.

  RUN_TAG=fastmtp_cactus_calibrated_tree_n100_v1 \
    ./scripts/run_fastmtp_cactus_calibrated_tree.sh

Each tree node uses the chain Cactus local rule:
  h(y) = min(1, p(y) + sqrt(2 * delta * p(y) * (1-p(y))))
  A(y) = min(1, h(y) / q(y))

The candidate survives with probability A(y). Among surviving prefixes, the
path score still combines target support, Cactus acceptance and length; it
does not blindly choose the longest path. Native/Cactus/SpecCascade N=100
baselines are linked and are not rerun.
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
export RUN_TAG="${RUN_TAG:-fastmtp_cactus_calibrated_tree_n100_$(date +%Y%m%d_%H%M%S)}"

export DYNAMIC_SUPPORT_MODE=cactus
export DYNAMIC_CACTUS_DELTA="${DYNAMIC_CACTUS_DELTA:-1.0}"
export DYNAMIC_CACTUS_TARGET_WEIGHT="${DYNAMIC_CACTUS_TARGET_WEIGHT:-0.25}"
export DYNAMIC_MIN_COVERAGE=0.0
export DYNAMIC_BETA="${DYNAMIC_BETA:-1.5}"
export DYNAMIC_FRONTIER_RESCUE=0

cat <<EOF
FastMTP Cactus-calibrated relaxed tree
  logical depth          : 3
  Cactus delta           : $DYNAMIC_CACTUS_DELTA
  target score weight    : $DYNAMIC_CACTUS_TARGET_WEIGHT
  length reward beta     : $DYNAMIC_BETA
  proposal tree          : unchanged soft-reach, maximum 10 nodes
  target forward/round   : 1
  output                 : $PROJECT_DIR/results/$RUN_TAG/comparison.md
EOF

exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh"
