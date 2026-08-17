#!/usr/bin/env bash
# Selected 20-task FastMTP tree+relaxation pilot from the 2026-08-12 audit.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export GSM8K_SAMPLES="${GSM8K_SAMPLES:-20}"
export HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-20}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
export RUN_TAG="${RUN_TAG:-fastmtp_selected_tree_20_$(date +%Y%m%d_%H%M%S)}"

export DYNAMIC_MAX_NODES=12
export DYNAMIC_ADAPTIVE_BASE_NODES=0
export DYNAMIC_ALLOCATION=reach_first
export DYNAMIC_TAU_RELAX=0.18
export DYNAMIC_RESCUE_SCORE_THRESHOLD=0.08
export DYNAMIC_RESCUE_DEPTH_PENALTY=0.5
export DYNAMIC_RESCUE_CONTINUATION_DISCOUNT=0.05
export DYNAMIC_RESCUE_CONTINUATION_MIN_DEPTH=2
export DYNAMIC_PATH_SELECTION=longest
export DYNAMIC_BETA=0.75

exec "$PROJECT_DIR/scripts/run_fastmtp_prefix_reopen_rescue_tree.sh"
