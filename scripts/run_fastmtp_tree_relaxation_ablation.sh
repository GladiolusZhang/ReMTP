#!/usr/bin/env bash
# Run direct Cactus+tree and the target-dominant Cactus-guided tree sequentially.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the two D=3 tree-relaxation variants sequentially and write one report:

  1. Direct Cactus + Dynamic Tree (Cactus at every node; ablation)
  2. Cactus-guided Dynamic Tree (ours; Cactus only at a dead frontier)

Default N=100 run:
  RUN_TAG=fastmtp_tree_relaxation_n100 \
    ./scripts/run_fastmtp_tree_relaxation_ablation.sh

Reuse the already completed direct-Cactus run and execute only the new guided
variant when it is missing:
  RUN_TAG=fastmtp_tree_relaxation_n100 \
  DIRECT_RUN_TAG=fastmtp_cactus_calibrated_tree_n100_v1 \
    ./scripts/run_fastmtp_tree_relaxation_ablation.sh

The script prints every sample, resumes completed runs, reuses the frozen
Native/Cactus/SpecCascade baselines, and writes:
  results/$RUN_TAG/comparison.md
EOF
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown arguments; use --help." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-fastmtp_tree_relaxation_$(date +%Y%m%d_%H%M%S)}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-${SAMPLES:-100}}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-${SAMPLES:-100}}"
BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_DIR/results/fastmtp_balanced_wide_n100}"
DIRECT_RUN_TAG="${DIRECT_RUN_TAG:-${RUN_TAG}_direct_cactus_tree}"
GUIDED_RUN_TAG="${GUIDED_RUN_TAG:-${RUN_TAG}_cactus_guided_tree}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
OUTPUT_ROOT="$PROJECT_DIR/results/$RUN_TAG"

if [[ "$DIRECT_RUN_TAG" == "$GUIDED_RUN_TAG" ]]; then
  echo "DIRECT_RUN_TAG and GUIDED_RUN_TAG must be different." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

record_verified_variant() {
  local run_tag="$1" support_mode="$2" server_log result_root
  server_log="$PROJECT_DIR/logs/$run_tag/dynamic_tree_server.log"
  result_root="$PROJECT_DIR/results/$run_tag"
  [[ -f "$server_log" ]] || {
    echo "Missing dynamic-tree server log: $server_log" >&2
    exit 2
  }
  grep -Fq "support_mode=$support_mode " "$server_log" || {
    echo "Tree support mode was not verified as $support_mode: $server_log" >&2
    exit 2
  }
  python - "$result_root/tree_variant.json" "$support_mode" "$server_log" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "support_mode": sys.argv[2],
            "server_log": str(Path(sys.argv[3]).resolve()),
            "marker_verified": True,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

cat <<EOF
FastMTP tree relaxation ablation
  GSM8K / HumanEval : $GSM8K_SAMPLES / $HUMANEVAL_SAMPLES
  frozen baselines  : $BASELINE_ROOT
  direct run        : results/$DIRECT_RUN_TAG
  guided run        : results/$GUIDED_RUN_TAG
  progress          : every $PROGRESS_EVERY sample(s)
  combined report   : $OUTPUT_ROOT/comparison.md
EOF

echo "===== 1/2 Direct Cactus + Dynamic Tree ====="
GSM8K_SAMPLES="$GSM8K_SAMPLES" \
HUMANEVAL_SAMPLES="$HUMANEVAL_SAMPLES" \
BASELINE_ROOT="$BASELINE_ROOT" \
PROGRESS_EVERY="$PROGRESS_EVERY" \
RUN_TAG="$DIRECT_RUN_TAG" \
  "$PROJECT_DIR/scripts/run_fastmtp_cactus_calibrated_tree.sh"
record_verified_variant "$DIRECT_RUN_TAG" cactus

echo "===== 2/2 Cactus-guided Dynamic Tree (ours) ====="
GSM8K_SAMPLES="$GSM8K_SAMPLES" \
HUMANEVAL_SAMPLES="$HUMANEVAL_SAMPLES" \
BASELINE_ROOT="$BASELINE_ROOT" \
PROGRESS_EVERY="$PROGRESS_EVERY" \
RUN_TAG="$GUIDED_RUN_TAG" \
  "$PROJECT_DIR/scripts/run_fastmtp_cactus_guided_tree.sh"
record_verified_variant "$GUIDED_RUN_TAG" cactus_guided

python -m remtp.fastmtp_tree_relaxation_report \
  --direct-root "$PROJECT_DIR/results/$DIRECT_RUN_TAG" \
  --guided-root "$PROJECT_DIR/results/$GUIDED_RUN_TAG" \
  --output-root "$OUTPUT_ROOT"

echo "Complete: $OUTPUT_ROOT/comparison.md"
sed -n '1,240p' "$OUTPUT_ROOT/comparison.md"
