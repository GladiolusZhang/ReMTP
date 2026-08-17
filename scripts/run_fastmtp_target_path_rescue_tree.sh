#!/usr/bin/env bash
# Target-supported longest paths with one post-selection rescue extension.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the target-supported longest-path tree with a one-token rescue extension.

Default 20-task pilot (reuses matching audited baseline summaries):
  ./scripts/run_fastmtp_target_path_rescue_tree.sh

Explicit 100-task run after the pilot is acceptable:
  GSM8K_SAMPLES=100 HUMANEVAL_SAMPLES=100 \
  BASELINE_ROOT=results/fastmtp_cactus_guided_wide_n100_v3 \
  RUN_TAG=fastmtp_target_path_rescue_n100 \
    ./scripts/run_fastmtp_target_path_rescue_tree.sh

Main quality/length controls:
  DYNAMIC_TAU_RELAX=0.20              ordinary node P(y)/max(P) threshold
  DYNAMIC_RESCUE_SCORE_THRESHOLD=0.08 geometric target/Cactus rescue threshold
  DYNAMIC_RESCUE_MIN_RELATIVE=0.02    absolute relative-target rescue floor

First, every node is subject to ordinary relaxed target verification and the
base path is selected by maximum surviving depth, with target confidence used
only to break ties. Second, if that path has not reached maximum depth, exactly
one rejected child on its frontier may be appended as a rescue extension.
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
export RUN_TAG="${RUN_TAG:-fastmtp_target_path_rescue_$(date +%Y%m%d_%H%M%S)}"

# These are maximums, not quotas. The MTP entropy/probability rule can retain
# fewer nodes when extra branches have negligible proposal probability.
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

export DYNAMIC_SUPPORT_MODE=target_path_rescue
export DYNAMIC_COVERAGE_MODE=coverage_gate
export DYNAMIC_MIN_COVERAGE=0
export DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.20}"
export DYNAMIC_CACTUS_DELTA="${DYNAMIC_CACTUS_DELTA:-1.0}"
export DYNAMIC_CACTUS_TARGET_WEIGHT="${DYNAMIC_CACTUS_TARGET_WEIGHT:-0.65}"
export DYNAMIC_MAX_GUIDED_RESCUES="${DYNAMIC_MAX_GUIDED_RESCUES:-1}"
export DYNAMIC_RESCUE_SCORE_THRESHOLD="${DYNAMIC_RESCUE_SCORE_THRESHOLD:-0.08}"
export DYNAMIC_RESCUE_MIN_RELATIVE="${DYNAMIC_RESCUE_MIN_RELATIVE:-0.02}"
export DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.0001}"
export DYNAMIC_FRONTIER_RESCUE=0
export DYNAMIC_PATH_SELECTION=longest
export DYNAMIC_PATH_TEMPERATURE=0
export FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-1}"

cat <<EOF
FastMTP target-supported longest-path tree
  logical depth       : $DYNAMIC_MAX_DEPTH
  node/children caps  : $DYNAMIC_MAX_NODES / $DYNAMIC_MAX_CHILDREN (not force-filled)
  ordinary survival  : P(y)/max(P) >= $DYNAMIC_TAU_RELAX
  rescue extension   : score >= $DYNAMIC_RESCUE_SCORE_THRESHOLD, relative >= $DYNAMIC_RESCUE_MIN_RELATIVE
  rescue score        : relative^$DYNAMIC_CACTUS_TARGET_WEIGHT * CactusAccept^(1-$DYNAMIC_CACTUS_TARGET_WEIGHT)
  two-stage selector  : ordinary relaxed longest path, then at most +1 rescue token
  baseline summaries  : $BASELINE_ROOT
  output              : $PROJECT_DIR/results/$RUN_TAG/comparison.md
EOF

if python - "$BASELINE_ROOT" "$GSM8K_SAMPLES" "$HUMANEVAL_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"gsm8k": int(sys.argv[2]), "humaneval": int(sys.argv[3])}
for dataset, samples in expected.items():
    for method in ("native", "cactus", "spec_cascade"):
        path = root / dataset / method / "summary.json"
        if not path.is_file():
            raise SystemExit(1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results", [])
        if len(rows) != 1 or int(rows[0].get("samples", -1)) != samples:
            raise SystemExit(1)
PY
then
  echo "Baseline sample counts match; only the new tree method will run."
  exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh"
fi

echo "Baseline sample counts do not match; generating all four fair rows."
export METHODS_CSV=native,cactus,spec_cascade,dynamic_tree
export INCLUDE_TARGET=0
export MTP_TOKENS=3
exec "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"
