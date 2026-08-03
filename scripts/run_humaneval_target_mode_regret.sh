#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Native MTP and the two reference-paper methods are mandatory context. Keep
# Exact-TV as the strongest high-throughput method from the previous phase.
export PROFILES="${PROFILES:-native_mtp target_mode_identity cactus spec_cascade exact_tv target_mode_regret}"
export SAMPLES="${SAMPLES:-164}"
export MTP_TOKENS="${MTP_TOKENS:-6}"

exec "$PROJECT_DIR/scripts/run_humaneval_comparison.sh"
