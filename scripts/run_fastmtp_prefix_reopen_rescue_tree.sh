#!/usr/bin/env bash
# Front-loaded multi-path rescue: one rescue debt per path, then ordinary continuation.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the D=3 prefix-reopen rescue pilot against reused audited baselines.

Default 20-task pilot:
  ./scripts/run_fastmtp_prefix_reopen_rescue_tree.sh

Selected audited configuration:
  ./scripts/run_fastmtp_selected_tree_20.sh

Explicit tag or sample count:
  RUN_TAG=fastmtp_prefix_reopen_rescue_pilot \
  GSM8K_SAMPLES=20 HUMANEVAL_SAMPLES=20 \
    ./scripts/run_fastmtp_prefix_reopen_rescue_tree.sh

Semantics:
  1. Every node first uses ordinary P(y)/max(P) relaxed verification.
  2. Every reachable path may spend at most one deterministic rescue.
  3. All threshold-qualified failed siblings survive, not just one parent/path.
  4. A rescued branch may continue through ordinary verification, but cannot
     spend another rescue.
  5. Rescue thresholds are front-loaded: 0.08, 0.12, 0.16 at depths 1,2,3.
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
export RUN_TAG="${RUN_TAG:-fastmtp_prefix_reopen_rescue_$(date +%Y%m%d_%H%M%S)}"

export DYNAMIC_MAX_DEPTH=3
export DYNAMIC_MAX_NODES="${DYNAMIC_MAX_NODES:-9}"
export DYNAMIC_MAX_CHILDREN="${DYNAMIC_MAX_CHILDREN:-3}"
export DYNAMIC_MIN_SIBLING_RATIO="${DYNAMIC_MIN_SIBLING_RATIO:-0.01}"
export DYNAMIC_TAU_MIN="${DYNAMIC_TAU_MIN:-0.001}"
export DYNAMIC_KAPPA="${DYNAMIC_KAPPA:-1.5}"
export DYNAMIC_MU="${DYNAMIC_MU:-0.5}"
export DYNAMIC_ETA=0
export DYNAMIC_ALLOCATION="${DYNAMIC_ALLOCATION:-soft_reach}"
export DYNAMIC_RANK_PENALTY="${DYNAMIC_RANK_PENALTY:-1.5}"

export DYNAMIC_SUPPORT_MODE=prefix_reopen_rescue
export DYNAMIC_COVERAGE_MODE=coverage_gate
export DYNAMIC_MIN_COVERAGE=0
export DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.20}"
export DYNAMIC_CACTUS_DELTA="${DYNAMIC_CACTUS_DELTA:-1.0}"
export DYNAMIC_CACTUS_TARGET_WEIGHT="${DYNAMIC_CACTUS_TARGET_WEIGHT:-0.65}"
export DYNAMIC_MAX_GUIDED_RESCUES=1
export DYNAMIC_RESCUE_SCORE_THRESHOLD="${DYNAMIC_RESCUE_SCORE_THRESHOLD:-0.08}"
export DYNAMIC_RESCUE_DEPTH_PENALTY="${DYNAMIC_RESCUE_DEPTH_PENALTY:-0.5}"
export DYNAMIC_RESCUE_MIN_RELATIVE="${DYNAMIC_RESCUE_MIN_RELATIVE:-0.02}"
export DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.0001}"
export DYNAMIC_FRONTIER_RESCUE=0
export DYNAMIC_PATH_SELECTION="${DYNAMIC_PATH_SELECTION:-longest}"
export DYNAMIC_PATH_TEMPERATURE=0
export DYNAMIC_BETA="${DYNAMIC_BETA:-0.75}"
export FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-1}"

cat <<EOF
FastMTP prefix-reopen rescue
  logical depth       : 3
  node/children caps  : $DYNAMIC_MAX_NODES / $DYNAMIC_MAX_CHILDREN
  ordinary threshold : $DYNAMIC_TAU_RELAX
  rescue thresholds  : base=$DYNAMIC_RESCUE_SCORE_THRESHOLD depth_penalty=$DYNAMIC_RESCUE_DEPTH_PENALTY
  allocation          : $DYNAMIC_ALLOCATION
  continuation rescue : discount=${DYNAMIC_RESCUE_CONTINUATION_DISCOUNT:-0.0} min_depth=${DYNAMIC_RESCUE_CONTINUATION_MIN_DEPTH:-2}
  rescue rule         : all eligible paths, at most one rescue per path,
                        ordinary continuation after rescue
  output              : $PROJECT_DIR/results/$RUN_TAG/comparison.md
EOF

exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh"
