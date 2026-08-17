#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Frozen main method: target-margin calibrated native-MTP relaxation.
#
# Compared with the exploratory Scheme 2 Ultra point, this profile spends a
# slightly larger TV budget only where the target is locally ambiguous.  It
# protects decisive target positions with a tighter gap floor and requires a
# stronger same-round sentinel for broad-rank candidates.  Saturated/unused TV
# is water-filled toward high-value prefix positions on the GPU.
export RISK_VARIANT=remtp
export RISK_MAX_RANK="${RISK_MAX_RANK:-8}"
export RISK_SCHEME2_RANK="${RISK_SCHEME2_RANK:-8}"
export RISK_BASE_GAP="${RISK_BASE_GAP:-2.15}"
export REMTP_FINAL_GAP_FLOOR="${REMTP_FINAL_GAP_FLOOR:-1.15}"
export REMTP_FINAL_MARGIN_TEMPERATURE="${REMTP_FINAL_MARGIN_TEMPERATURE:-0.75}"
export REMTP_FINAL_RISKY_RANK="${REMTP_FINAL_RISKY_RANK:-4}"
export REMTP_FINAL_RISKY_GAP="${REMTP_FINAL_RISKY_GAP:-1.20}"
export REMTP_FINAL_RISKY_MIN_CHECKS="${REMTP_FINAL_RISKY_MIN_CHECKS:-2}"
export REMTP_FINAL_PREFIX_WEIGHT="${REMTP_FINAL_PREFIX_WEIGHT:-0.75}"
export REMTP_FINAL_SUPPORT_TEMPERATURE="${REMTP_FINAL_SUPPORT_TEMPERATURE:-1.00}"

export CACTUS_DELTA="${CACTUS_DELTA:-1.25}"
export RISK_BUDGET="${RISK_BUDGET:-5.50}"
export RISK_PER_TOKEN_TV="${RISK_PER_TOKEN_TV:-0.65}"
export RISK_BLOCK_TV="${RISK_BLOCK_TV:-2.10}"

export SENTINEL_MAX_GAP="${SENTINEL_MAX_GAP:-2.50}"
export SENTINEL_MTP_FLOOR="${SENTINEL_MTP_FLOOR:-0.002}"
export SENTINEL_HEAD_MASS_FLOOR="${SENTINEL_HEAD_MASS_FLOOR:-0.025}"
export SENTINEL_MIN_CHECKS="${SENTINEL_MIN_CHECKS:-1}"
export SENTINEL_SOFT_FLOOR="${SENTINEL_SOFT_FLOOR:-0.70}"
export SENTINEL_LAST_GAP_SCALE="${SENTINEL_LAST_GAP_SCALE:-1.00}"

export RISK_DEBT_DECAY="${RISK_DEBT_DECAY:-0.80}"
# The published ReMTP result was produced on vLLM's async path, where the old
# output-length boundary heuristic reset debt every round. Keep that behavior
# explicit while the audit measures whether reach-weighted debt is beneficial.
export RISK_DEBT_SCALE="${RISK_DEBT_SCALE:-0.00}"
export RISK_ADAPTIVE_DRAFT="${RISK_ADAPTIVE_DRAFT:-0}"

exec "$PROJECT_DIR/scripts/serve_risk_entropy_mtp.sh"
