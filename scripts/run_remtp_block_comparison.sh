#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"

case "$MODE" in
  gsm8k)
    exec "$PROJECT_DIR/scripts/run_gsm8k_remtp_block.sh"
    ;;
  humaneval)
    exec "$PROJECT_DIR/scripts/run_humaneval_remtp_block.sh"
    ;;
  all)
    shared_tag="${RUN_TAG:-remtp_block_$(date +%Y%m%d_%H%M%S)}"
    RUN_TAG="$shared_tag" "$PROJECT_DIR/scripts/run_gsm8k_remtp_block.sh"
    RUN_TAG="$shared_tag" "$PROJECT_DIR/scripts/run_humaneval_remtp_block.sh"
    gsm_root="$PROJECT_DIR/results/gsm8k_three_schemes_${shared_tag}"
    human_root="$PROJECT_DIR/results/humaneval_comparison_${shared_tag}"
    combined_report="$PROJECT_DIR/results/combined_${shared_tag}.md"
    "$PROJECT_DIR/.venv/bin/python" -m remtp.combined_benchmark_report \
      --gsm-root "$gsm_root" \
      --humaneval-root "$human_root" \
      --output "$combined_report" \
      --focus remtp_block
    echo "Combined report: $combined_report"
    ;;
  *)
    echo "Usage: $0 [gsm8k|humaneval|all]" >&2
    exit 2
    ;;
esac
