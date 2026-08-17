#!/usr/bin/env bash
# Run only the new balanced/wider dynamic tree and reuse frozen N=100 baselines.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SAMPLES="${SAMPLES:-100}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-$PROJECT_DIR/results/fastmtp_live_n100}"
RUN_TAG="${RUN_TAG:-fastmtp_balanced_wide_n${SAMPLES}_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"

if [[ ! -f "$BASELINE_RUN_ROOT/comparison.md" ]]; then
  echo "Frozen baseline run is missing: $BASELINE_RUN_ROOT/comparison.md" >&2
  exit 2
fi

python - "$BASELINE_RUN_ROOT" "$SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
for dataset in ("gsm8k", "humaneval"):
    for method in ("native", "cactus", "spec_cascade"):
        path = root / dataset / method / "summary.json"
        if not path.is_file():
            raise SystemExit(f"missing frozen baseline summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results") or []
        count = rows[0].get("samples") if len(rows) == 1 else None
        if count is None:
            count = payload.get("num_samples")
        if count is None:
            raise SystemExit(f"cannot determine baseline sample count: {path}")
        if int(count) != expected:
            raise SystemExit(
                f"baseline sample mismatch for {dataset}/{method}: "
                f"expected {expected}, found {count}"
            )
PY

mkdir -p "$RUN_ROOT/gsm8k" "$RUN_ROOT/humaneval"
for dataset in gsm8k humaneval; do
  for method in native cactus spec_cascade; do
    source_dir="$(realpath "$BASELINE_RUN_ROOT/$dataset/$method")"
    destination="$RUN_ROOT/$dataset/$method"
    if [[ -e "$destination" || -L "$destination" ]]; then
      if [[ ! -L "$destination" || "$(realpath "$destination")" != "$source_dir" ]]; then
        echo "Refusing to replace existing baseline path: $destination" >&2
        exit 2
      fi
    else
      ln -s "$source_dir" "$destination"
    fi
  done
done

echo "Balanced wider-tree experiment"
echo "  samples          : $SAMPLES per dataset"
echo "  reused baselines : $BASELINE_RUN_ROOT"
echo "  new method only  : dynamic_tree"
echo "  tree             : D=3 N_max=8 children_max=2 sibling_ratio=0.20"
echo "  path score       : geometric target support * length^0.75"
echo "  result           : $RUN_ROOT/comparison.md"

SAMPLES="$SAMPLES" \
RUN_TAG="$RUN_TAG" \
METHODS_CSV=dynamic_tree \
INCLUDE_TARGET=0 \
PROGRESS_EVERY="${PROGRESS_EVERY:-1}" \
FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}" \
DYNAMIC_MAX_DEPTH=3 \
DYNAMIC_MAX_NODES=8 \
DYNAMIC_MAX_CHILDREN=2 \
DYNAMIC_MIN_SIBLING_RATIO=0.20 \
DYNAMIC_PATH_SELECTION=balanced \
DYNAMIC_BETA=0.75 \
DYNAMIC_PATH_TEMPERATURE=0 \
DYNAMIC_FRONTIER_RESCUE=1 \
DYNAMIC_RESCUE_DELTA=0.5 \
DYNAMIC_RESCUE_MIN_RELATIVE=0.1 \
DYNAMIC_RESCUE_MIN_TARGET_PROB=0.001 \
  "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"
