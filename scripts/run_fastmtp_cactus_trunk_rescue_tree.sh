#!/usr/bin/env bash
# Sampled-Q Cactus trunk with rejection-only tree rescue.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the sampled-Q Cactus trunk + rejection-only tree rescue experiment.

Quick trial (recommended first):
  GSM8K_SAMPLES=20 HUMANEVAL_SAMPLES=20 \
    RUN_TAG=fastmtp_cactus_trunk_rescue_pilot \
    ./scripts/run_fastmtp_cactus_trunk_rescue_tree.sh

Audited 100-task comparison:
  GSM8K_SAMPLES=100 HUMANEVAL_SAMPLES=100 \
    RUN_TAG=fastmtp_cactus_trunk_rescue_n100 \
    ./scripts/run_fastmtp_cactus_trunk_rescue_tree.sh

The unchanged Native/Cactus/SpecCascade summaries are linked from
BASELINE_ROOT. Only the new dynamic-tree row is generated. The sampled trunk
uses the same Q and Cactus delta as Chain Cactus; deterministic siblings are
consulted only at the first rejected trunk token.
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
export RUN_TAG="${RUN_TAG:-fastmtp_cactus_trunk_rescue_$(date +%Y%m%d_%H%M%S)}"

# Three sampled trunk nodes plus up to two rejection backups at each depth.
# Nine is a cap, not a force-fill target; the existing Q floors can retain
# fewer siblings when FastMTP is sufficiently confident.
export DYNAMIC_MAX_DEPTH=3
export DYNAMIC_MAX_NODES="${DYNAMIC_MAX_NODES:-9}"
export DYNAMIC_MAX_CHILDREN="${DYNAMIC_MAX_CHILDREN:-3}"
export DYNAMIC_MIN_SIBLING_RATIO="${DYNAMIC_MIN_SIBLING_RATIO:-0.02}"
export DYNAMIC_TAU_MIN="${DYNAMIC_TAU_MIN:-0.001}"
export DYNAMIC_KAPPA="${DYNAMIC_KAPPA:-1.25}"
export DYNAMIC_MU="${DYNAMIC_MU:-0.5}"
export DYNAMIC_ETA=0
export DYNAMIC_ALLOCATION=soft_reach

export DYNAMIC_SUPPORT_MODE=cactus_trunk_rescue
export DYNAMIC_CACTUS_DELTA="${DYNAMIC_CACTUS_DELTA:-1.0}"
export DYNAMIC_RESCUE_DELTA="${DYNAMIC_RESCUE_DELTA:-1.0}"
export DYNAMIC_CACTUS_TARGET_WEIGHT="${DYNAMIC_CACTUS_TARGET_WEIGHT:-0.5}"
export DYNAMIC_RESCUE_MIN_RELATIVE="${DYNAMIC_RESCUE_MIN_RELATIVE:-0.01}"
export DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.0001}"
export DYNAMIC_FRONTIER_RESCUE=0
export DYNAMIC_PATH_SELECTION=longest
export DYNAMIC_PATH_TEMPERATURE=0
export FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-1}"

cat <<EOF
FastMTP sampled-Q Cactus trunk + rejection-only tree rescue
  logical trunk depth : 3 (fair with chain baselines)
  node cap            : $DYNAMIC_MAX_NODES
  backups per depth   : at most $((DYNAMIC_MAX_CHILDREN - 1))
  trunk verifier      : Cactus delta=$DYNAMIC_CACTUS_DELTA
  rescue verifier     : residual-Cactus delta=$DYNAMIC_RESCUE_DELTA
  rescue floors       : relative=$DYNAMIC_RESCUE_MIN_RELATIVE P=$DYNAMIC_RESCUE_MIN_TARGET_PROB
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
  echo "Baseline sample counts match; only the new method will run."
  exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh"
fi

echo "Baseline sample counts do not match this run; generating all four fair rows."
export METHODS_CSV=native,cactus,spec_cascade,dynamic_tree
export INCLUDE_TARGET=0
export MTP_TOKENS=3
exec "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"
