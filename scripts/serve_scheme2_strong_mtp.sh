#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A deliberately stronger point on the same Scheme 2 quality/speed curve.
export RISK_VARIANT=scheme2_relaxed
export RISK_MAX_RANK="${RISK_MAX_RANK:-8}"
export RISK_SCHEME2_RANK="${RISK_SCHEME2_RANK:-8}"
export RISK_BASE_GAP="${RISK_BASE_GAP:-1.70}"
export RISK_BUDGET="${RISK_BUDGET:-4.50}"
export RISK_PER_TOKEN_TV="${RISK_PER_TOKEN_TV:-0.45}"
export RISK_BLOCK_TV="${RISK_BLOCK_TV:-1.55}"
export SENTINEL_MAX_GAP="${SENTINEL_MAX_GAP:-2.00}"
export SENTINEL_MTP_FLOOR="${SENTINEL_MTP_FLOOR:-0.005}"
export SENTINEL_HEAD_MASS_FLOOR="${SENTINEL_HEAD_MASS_FLOOR:-0.05}"
export SENTINEL_MIN_CHECKS="${SENTINEL_MIN_CHECKS:-1}"
export SENTINEL_SOFT_FLOOR="${SENTINEL_SOFT_FLOOR:-0.55}"
export SENTINEL_LAST_GAP_SCALE="${SENTINEL_LAST_GAP_SCALE:-1.00}"
export RISK_DEBT_SCALE="${RISK_DEBT_SCALE:-0.30}"
export RISK_ADAPTIVE_DRAFT="${RISK_ADAPTIVE_DRAFT:-0}"

exec "$PROJECT_DIR/scripts/serve_risk_entropy_mtp.sh"
