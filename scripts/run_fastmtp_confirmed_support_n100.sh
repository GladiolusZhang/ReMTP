#!/usr/bin/env bash
# 100-task screen reusing the existing audited N=100 baselines.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export GSM8K_SAMPLES="${GSM8K_SAMPLES:-100}"
export HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-100}"
export BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_DIR/results/fastmtp_balanced_wide_n100}"
export RUN_TAG="${RUN_TAG:-fastmtp_confirmed_support_d3_n100}"

exec "$PROJECT_DIR/scripts/run_fastmtp_confirmed_support_tree.sh" "$@"
