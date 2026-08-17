#!/usr/bin/env bash
# Larger MAL/quality comparison for the bounded frontier-rescue tree.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SAMPLES="${SAMPLES:-50}"
RUN_TAG="${RUN_TAG:-fastmtp_frontier_rescue_n${SAMPLES}_$(date +%Y%m%d_%H%M%S)}"

# Keep the established target-dominant tree unchanged.  Only a dead frontier
# may receive one Cactus-style rescue per eventual committed path.
export SAMPLES RUN_TAG
export INCLUDE_TARGET=0
export PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
export FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}"
export DYNAMIC_FRONTIER_RESCUE="${DYNAMIC_FRONTIER_RESCUE:-1}"
export DYNAMIC_RESCUE_DELTA="${DYNAMIC_RESCUE_DELTA:-0.5}"
export DYNAMIC_RESCUE_MIN_RELATIVE="${DYNAMIC_RESCUE_MIN_RELATIVE:-0.1}"
export DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.001}"

echo "FastMTP MAL/quality comparison"
echo "  samples/dataset       : $SAMPLES"
echo "  methods               : native, cactus, spec_cascade, dynamic_tree"
echo "  live progress         : every $PROGRESS_EVERY sample(s)"
echo "  dynamic frontier rescue: enabled=$DYNAMIC_FRONTIER_RESCUE delta=$DYNAMIC_RESCUE_DELTA"
echo "  target support floor  : relative=$DYNAMIC_RESCUE_MIN_RELATIVE absolute=$DYNAMIC_RESCUE_MIN_TARGET_PROB"
echo "  no-thinking template  : $FAST_MTP_NO_THINK"
echo "  output                : results/$RUN_TAG/comparison.md"

exec "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"
