#!/usr/bin/env bash
# Log-calibrated relaxed-tree profiles.  The default is deliberately MAL-first.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Run one log-calibrated D=3 dynamic-tree profile on the existing N=100 task set.

Default MAL-first stress test:
  ./scripts/run_fastmtp_empirical_relaxation.sh

Profiles:
  PROFILE=plausible  target coverage >= 0.01, relative support >= 0.01,
                     confidence/length beta=3.0
  PROFILE=mal_first  target coverage >= 0.001, relative support >= 0.001,
                     confidence/length beta=5.0 (default)
  PROFILE=ceiling    retain every non-EOS candidate and use beta=5.0;
                     diagnostic upper-bound only, quality may collapse

Examples:
  PROFILE=mal_first RUN_TAG=fastmtp_mal_first_n100_v1 \
    ./scripts/run_fastmtp_empirical_relaxation.sh

  PROFILE=plausible RUN_TAG=fastmtp_plausible_mass_n100_v1 \
    ./scripts/run_fastmtp_empirical_relaxation.sh

Only the new tree method is generated. Completed Native/Cactus/SpecCascade
N=100 baselines are linked into the result directory and are not rerun.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if (( $# > 0 )); then usage >&2; exit 2; fi

PROFILE="${PROFILE:-mal_first}"
case "$PROFILE" in
  plausible)
    min_coverage=0.01
    min_relative=0.01
    length_reward=3.0
    ;;
  mal_first)
    min_coverage=0.001
    min_relative=0.001
    length_reward=5.0
    ;;
  ceiling)
    min_coverage=0.0
    min_relative=0.0
    length_reward=5.0
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE (expected plausible, mal_first or ceiling)." >&2
    exit 2
    ;;
esac

export GSM8K_SAMPLES="${GSM8K_SAMPLES:-100}"
export HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-100}"
export BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_DIR/results/fastmtp_balanced_wide_n100}"
export RUN_TAG="${RUN_TAG:-fastmtp_empirical_${PROFILE}_n100_$(date +%Y%m%d_%H%M%S)}"

# Keep the proposal tree identical to the audited soft-reach tree.  This run
# changes only target-side survival and the confidence--length trade-off.
export DYNAMIC_SUPPORT_MODE=relative
export DYNAMIC_MIN_COVERAGE="${DYNAMIC_MIN_COVERAGE:-$min_coverage}"
export DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-$min_relative}"
export DYNAMIC_BETA="${DYNAMIC_BETA:-$length_reward}"
export DYNAMIC_FRONTIER_RESCUE=0

cat <<EOF
FastMTP empirical relaxation
  profile                  : $PROFILE
  minimum target coverage  : $DYNAMIC_MIN_COVERAGE
  minimum relative support : $DYNAMIC_TAU_RELAX
  balanced path beta       : $DYNAMIC_BETA
  frontier rescue          : disabled (direct survival is already broader)
  samples                  : GSM8K=$GSM8K_SAMPLES HumanEval=$HUMANEVAL_SAMPLES
  output                   : $PROJECT_DIR/results/$RUN_TAG/comparison.md
EOF

exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh"
