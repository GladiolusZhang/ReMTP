#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export SAMPLES="${SAMPLES:-200}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260730}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export SEED="${SEED:-42}"
export MAX_TOKENS="${MAX_TOKENS:-384}"
export MTP_TOKENS="${MTP_TOKENS:-6}"
export PROFILES="native_mtp cactus spec_cascade remtp_block"
export REUSE_PROFILES="native_mtp cactus spec_cascade"
export ELIGIBLE_PROFILES="remtp_block"
export REUSE_REFERENCES=1
export GSM8K_REFERENCE_ROOT="${GSM8K_REFERENCE_ROOT:-$PROJECT_DIR/results/gsm8k_three_schemes_ultra_anchored_20260804_224028}"
export RUN_TAG="${RUN_TAG:-remtp_block_$(date +%Y%m%d_%H%M%S)}"

exec "$PROJECT_DIR/scripts/run_gsm8k_three_schemes.sh"
