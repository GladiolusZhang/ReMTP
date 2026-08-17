#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Stronger Scheme 2 profile.  The old scheme2 script is left unchanged so its
# published local results remain reproducible.
export RISK_VARIANT=scheme2_relaxed
export RISK_MAX_RANK="${RISK_MAX_RANK:-8}"
export RISK_SCHEME2_RANK="${RISK_SCHEME2_RANK:-8}"
export RISK_BASE_GAP="${RISK_BASE_GAP:-1.35}"
export RISK_BUDGET="${RISK_BUDGET:-3.60}"
export RISK_PER_TOKEN_TV="${RISK_PER_TOKEN_TV:-0.35}"
export RISK_BLOCK_TV="${RISK_BLOCK_TV:-1.20}"
export SENTINEL_MAX_GAP="${SENTINEL_MAX_GAP:-1.50}"
export SENTINEL_MTP_FLOOR="${SENTINEL_MTP_FLOOR:-0.01}"
export SENTINEL_HEAD_MASS_FLOOR="${SENTINEL_HEAD_MASS_FLOOR:-0.10}"
export SENTINEL_MIN_CHECKS="${SENTINEL_MIN_CHECKS:-1}"
export SENTINEL_SOFT_FLOOR="${SENTINEL_SOFT_FLOOR:-0.35}"
export SENTINEL_LAST_GAP_SCALE="${SENTINEL_LAST_GAP_SCALE:-1.00}"
export RISK_DEBT_SCALE="${RISK_DEBT_SCALE:-0.45}"

# Debt continues to regulate the next block's target band and TV budget.  It
# no longer shortens the MTP proposal, so vLLM can keep asynchronous scheduling
# and every round retains the full six-token opportunity.
export RISK_ADAPTIVE_DRAFT="${RISK_ADAPTIVE_DRAFT:-0}"

exec "$PROJECT_DIR/scripts/serve_risk_entropy_mtp.sh"
