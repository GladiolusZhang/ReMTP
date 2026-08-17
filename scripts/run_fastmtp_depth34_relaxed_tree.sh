#!/usr/bin/env bash
# Full D=3 versus D=4 relaxed-tree experiment with one shared baseline run.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

GSM8K_SAMPLES="${GSM8K_SAMPLES:-500}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-164}"
RUN_TAG="${RUN_TAG:-fastmtp_depth34_relaxed_$(date +%Y%m%d_%H%M%S)}"
STUDY_ROOT="$PROJECT_DIR/results/$RUN_TAG"
D3_TAG="$RUN_TAG/d3"
D4_TAG="$RUN_TAG/d4"
D3_ROOT="$PROJECT_DIR/results/$D3_TAG"
D4_ROOT="$PROJECT_DIR/results/$D4_TAG"

# The cap is intentionally not a minimum. Candidates enter only after passing
# Q-based absolute/relative guards, then compete under the global node budget.
NODE_CAP="${NODE_CAP:-10}"
MAX_CHILDREN="${MAX_CHILDREN:-3}"
MIN_SIBLING_RATIO="${MIN_SIBLING_RATIO:-0.02}"
MIN_DRAFT_PROB="${MIN_DRAFT_PROB:-0.002}"
KAPPA="${KAPPA:-1.25}"

run_depth() {
  local depth="$1" tag="$2" methods="$3"
  GSM8K_SAMPLES="$GSM8K_SAMPLES" \
  HUMANEVAL_SAMPLES="$HUMANEVAL_SAMPLES" \
  RUN_TAG="$tag" \
  METHODS_CSV="$methods" \
  INCLUDE_TARGET=0 \
  PROGRESS_EVERY="${PROGRESS_EVERY:-1}" \
  FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}" \
  MTP_TOKENS=3 \
  DYNAMIC_MAX_DEPTH="$depth" \
  DYNAMIC_MAX_NODES="$NODE_CAP" \
  DYNAMIC_MAX_CHILDREN="$MAX_CHILDREN" \
  DYNAMIC_MIN_SIBLING_RATIO="$MIN_SIBLING_RATIO" \
  DYNAMIC_TAU_MIN="$MIN_DRAFT_PROB" \
  DYNAMIC_KAPPA="$KAPPA" \
  DYNAMIC_MU="${DYNAMIC_MU:-0.5}" \
  DYNAMIC_ETA="${DYNAMIC_ETA:-0.25}" \
  DYNAMIC_COVERAGE_MODE=coverage_gate \
  DYNAMIC_MIN_COVERAGE="${DYNAMIC_MIN_COVERAGE:-0.05}" \
  DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.50}" \
  DYNAMIC_PATH_SELECTION=balanced \
  DYNAMIC_BETA="${DYNAMIC_BETA:-0.75}" \
  DYNAMIC_PATH_TEMPERATURE=0 \
  DYNAMIC_FRONTIER_RESCUE=1 \
  DYNAMIC_RESCUE_DELTA="${DYNAMIC_RESCUE_DELTA:-0.5}" \
  DYNAMIC_RESCUE_MIN_RELATIVE="${DYNAMIC_RESCUE_MIN_RELATIVE:-0.1}" \
  DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.001}" \
    "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"
}

link_shared_baselines() {
  mkdir -p "$D4_ROOT/gsm8k" "$D4_ROOT/humaneval"
  for dataset in gsm8k humaneval; do
    for method in native cactus spec_cascade; do
      local source_dir destination
      source_dir="$(realpath "$D3_ROOT/$dataset/$method")"
      destination="$D4_ROOT/$dataset/$method"
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
}

cat <<EOF
FastMTP relaxed-tree D=3/D=4 study
  GSM8K           : $GSM8K_SAMPLES samples
  HumanEval       : $HUMANEVAL_SAMPLES samples
  node cap        : $NODE_CAP (maximum only; never force-filled)
  children/parent : <= $MAX_CHILDREN
  proposal guards : q >= $MIN_DRAFT_PROB and q/q_top1 >= $MIN_SIBLING_RATIO
  kappa           : $KAPPA
  verifier        : target-relative >= ${DYNAMIC_TAU_RELAX:-0.50}, balanced beta=${DYNAMIC_BETA:-0.75}
  output          : $STUDY_ROOT/depth34_comparison.md
EOF

echo "===== phase 1/3: shared baselines + relaxed tree D=3 ====="
run_depth 3 "$D3_TAG" "native,cactus,spec_cascade,dynamic_tree"

echo "===== phase 2/3: relaxed tree D=4 (baselines reused) ====="
link_shared_baselines
run_depth 4 "$D4_TAG" "dynamic_tree"

echo "===== phase 3/3: combined report ====="
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -m remtp.fastmtp_depth34_report \
  --d3-root "$D3_ROOT" --d4-root "$D4_ROOT" --output-dir "$STUDY_ROOT"
echo "Complete: $STUDY_ROOT/depth34_comparison.md"
sed -n '1,240p' "$STUDY_ROOT/depth34_comparison.md"
