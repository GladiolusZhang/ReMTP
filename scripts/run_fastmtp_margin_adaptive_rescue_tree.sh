#!/usr/bin/env bash
# Target-margin adaptive prefix-reopen rescue on the frozen D=3 FastMTP tree.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the target-margin adaptive prefix-reopen pilot and reuse audited baselines.

Default 20-task pilot:
  ./scripts/run_fastmtp_margin_adaptive_rescue_tree.sh

Explicit output tag:
  RUN_TAG=fastmtp_margin_adaptive_rescue_pilot \
    ./scripts/run_fastmtp_margin_adaptive_rescue_tree.sh

The rule is task-agnostic:
  - uncertain target (small top1/top2 log gap): rescue more than v1;
  - decisive target (large log gap): rescue less than v1;
  - one rescue debt per path, with ordinary continuation afterwards.
EOF
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown arguments; use --help." >&2
  exit 2
fi

export GSM8K_SAMPLES="${GSM8K_SAMPLES:-20}"
export HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-20}"
export BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_DIR/results/fastmtp_cactus_trunk_rescue_tempfix_pilot}"
export RUN_TAG="${RUN_TAG:-fastmtp_margin_adaptive_rescue_$(date +%Y%m%d_%H%M%S)}"

export DYNAMIC_MAX_DEPTH=3
export DYNAMIC_MAX_NODES="${DYNAMIC_MAX_NODES:-9}"
export DYNAMIC_MAX_CHILDREN="${DYNAMIC_MAX_CHILDREN:-3}"
export DYNAMIC_MIN_SIBLING_RATIO="${DYNAMIC_MIN_SIBLING_RATIO:-0.01}"
export DYNAMIC_TAU_MIN="${DYNAMIC_TAU_MIN:-0.001}"
export DYNAMIC_KAPPA="${DYNAMIC_KAPPA:-1.5}"
export DYNAMIC_MU="${DYNAMIC_MU:-0.5}"
export DYNAMIC_ETA=0
export DYNAMIC_ALLOCATION=soft_reach
export DYNAMIC_RANK_PENALTY="${DYNAMIC_RANK_PENALTY:-1.5}"

export DYNAMIC_SUPPORT_MODE=prefix_reopen_rescue
export DYNAMIC_COVERAGE_MODE=coverage_gate
export DYNAMIC_MIN_COVERAGE=0
export DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.20}"
export DYNAMIC_CACTUS_DELTA="${DYNAMIC_CACTUS_DELTA:-1.0}"
export DYNAMIC_CACTUS_TARGET_WEIGHT="${DYNAMIC_CACTUS_TARGET_WEIGHT:-0.65}"
export DYNAMIC_MAX_GUIDED_RESCUES=1

# Base thresholds are deliberately wider than the previous 0.08/0.12/0.16
# only when the target is uncertain.  A decisive target can double them.
export DYNAMIC_RESCUE_SCORE_THRESHOLD="${DYNAMIC_RESCUE_SCORE_THRESHOLD:-0.06}"
export DYNAMIC_RESCUE_DEPTH_PENALTY="${DYNAMIC_RESCUE_DEPTH_PENALTY:-0.35}"
export DYNAMIC_RESCUE_MARGIN_REFERENCE="${DYNAMIC_RESCUE_MARGIN_REFERENCE:-1.5}"
export DYNAMIC_RESCUE_MARGIN_PENALTY="${DYNAMIC_RESCUE_MARGIN_PENALTY:-1.0}"
export DYNAMIC_RESCUE_MIN_RELATIVE="${DYNAMIC_RESCUE_MIN_RELATIVE:-0.015}"
export DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.0001}"

export DYNAMIC_FRONTIER_RESCUE=0
export DYNAMIC_PATH_SELECTION=longest
export DYNAMIC_PATH_TEMPERATURE=0
export DYNAMIC_BETA="${DYNAMIC_BETA:-0.75}"
export FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-1}"

cat <<EOF
FastMTP margin-adaptive prefix rescue
  logical depth       : 3
  node/children caps  : $DYNAMIC_MAX_NODES / $DYNAMIC_MAX_CHILDREN
  ordinary threshold : $DYNAMIC_TAU_RELAX
  rescue base/depth   : $DYNAMIC_RESCUE_SCORE_THRESHOLD / $DYNAMIC_RESCUE_DEPTH_PENALTY
  margin ref/penalty  : $DYNAMIC_RESCUE_MARGIN_REFERENCE / $DYNAMIC_RESCUE_MARGIN_PENALTY
  target floors       : relative=$DYNAMIC_RESCUE_MIN_RELATIVE probability=$DYNAMIC_RESCUE_MIN_TARGET_PROB
  output              : $PROJECT_DIR/results/$RUN_TAG/comparison.md
EOF

exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh"
