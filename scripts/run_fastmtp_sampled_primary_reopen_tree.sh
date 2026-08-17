#!/usr/bin/env bash
# Cactus-equivalent sampled primary path with full-subtree rejection recovery.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the sampled-primary tree-recovery experiment.

Quick 20-task validation (reuses matching audited baselines):
  RUN_TAG=fastmtp_sampled_primary_reopen_pilot \
    ./scripts/run_fastmtp_sampled_primary_reopen_tree.sh

Larger run after the pilot is acceptable:
  GSM8K_SAMPLES=100 HUMANEVAL_SAMPLES=100 \
  BASELINE_ROOT=results/<matching-100-task-baseline> \
  RUN_TAG=fastmtp_sampled_primary_reopen_n100 \
    ./scripts/run_fastmtp_sampled_primary_reopen_tree.sh

Semantics:
  - rank-0 at every depth is sampled from the same MTP Q as chain Cactus;
  - the sampled primary path uses exact candidate-wise Cactus verification;
  - backups never replace an accepted primary token;
  - at the first primary rejection, one residual-supported sibling may reopen
    its already verified subtree; descendants must pass target-relative P;
  - failure to recover uses the ordinary Cactus residual correction.
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
export RUN_TAG="${RUN_TAG:-fastmtp_sampled_primary_reopen_$(date +%Y%m%d_%H%M%S)}"

# Three full root alternatives fit in nine nodes: sampled primary plus two
# backups, followed by one reach-first continuation per retained path.
export DYNAMIC_MAX_DEPTH=3
export DYNAMIC_MAX_NODES="${DYNAMIC_MAX_NODES:-9}"
export DYNAMIC_MAX_CHILDREN="${DYNAMIC_MAX_CHILDREN:-3}"
export DYNAMIC_MIN_SIBLING_RATIO="${DYNAMIC_MIN_SIBLING_RATIO:-0.01}"
export DYNAMIC_TAU_MIN="${DYNAMIC_TAU_MIN:-0.001}"
export DYNAMIC_KAPPA="${DYNAMIC_KAPPA:-1.5}"
export DYNAMIC_MU="${DYNAMIC_MU:-0.5}"
export DYNAMIC_ETA=0
export DYNAMIC_ALLOCATION=reach_first

export DYNAMIC_SUPPORT_MODE=sampled_primary_reopen
export DYNAMIC_CACTUS_DELTA="${DYNAMIC_CACTUS_DELTA:-1.0}"
export DYNAMIC_RESCUE_DELTA="${DYNAMIC_RESCUE_DELTA:-1.0}"
export DYNAMIC_CACTUS_TARGET_WEIGHT="${DYNAMIC_CACTUS_TARGET_WEIGHT:-0.5}"
export DYNAMIC_RESCUE_MIN_RELATIVE="${DYNAMIC_RESCUE_MIN_RELATIVE:-0.01}"
export DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.0001}"
export DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.20}"
export DYNAMIC_FRONTIER_RESCUE=0
export DYNAMIC_PATH_SELECTION=longest
export DYNAMIC_PATH_TEMPERATURE=0
export FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-1}"

cat <<EOF
FastMTP sampled-primary tree recovery
  logical depth/nodes : 3 / cap $DYNAMIC_MAX_NODES
  proposal            : sampled Q primary + at most $((DYNAMIC_MAX_CHILDREN - 1)) root backups
  allocation          : reach-first, budget spread across remaining depths
  primary verifier    : Cactus delta=$DYNAMIC_CACTUS_DELTA
  rejection recovery  : residual-Cactus delta=$DYNAMIC_RESCUE_DELTA
  continuation        : target relative >= $DYNAMIC_TAU_RELAX, no second rescue
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
        rows = json.loads(path.read_text(encoding="utf-8")).get("results", [])
        if len(rows) != 1 or int(rows[0].get("samples", -1)) != samples:
            raise SystemExit(1)
PY
then
  echo "Baseline sample counts match; only the new dynamic-tree row will run."
  exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh"
fi

echo "Baseline sample counts do not match; generating all four fair rows."
export METHODS_CSV=native,cactus,spec_cascade,dynamic_tree
export INCLUDE_TARGET=0
export MTP_TOKENS=3
exec "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"
