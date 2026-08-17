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
export PROFILES="native_mtp cactus spec_cascade scheme1 scheme2 scheme3 scheme12 scheme2_relaxed scheme2_strong scheme12_joint scheme2_ultra scheme12_anchored"
export REUSE_PROFILES="native_mtp cactus spec_cascade scheme1 scheme2 scheme3 scheme12 scheme2_relaxed scheme2_strong scheme12_joint"
export ELIGIBLE_PROFILES="scheme2_ultra scheme12_anchored"
export REUSE_REFERENCES=1
export GSM8K_REFERENCE_ROOT="${GSM8K_REFERENCE_ROOT:-$PROJECT_DIR/results/gsm8k_three_schemes_stronger_joint_20260804_212705}"
export RUN_TAG="${RUN_TAG:-ultra_anchored_$(date +%Y%m%d_%H%M%S)}"

exec "$PROJECT_DIR/scripts/run_gsm8k_three_schemes.sh"
