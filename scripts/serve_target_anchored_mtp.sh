#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

export REMTP_TRACE=0
export ENFORCE_EAGER=0
export MTP_TOKENS="${MTP_TOKENS:-6}"
export MTP_REJECTION_SAMPLE_METHOD=probabilistic
export REMTP_WORKER_CLS="${REMTP_WORKER_CLS:-remtp.worker.TargetAnchoredMTPWorker}"

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "Target-anchored block verification requires MTP_TOKENS=6." >&2
  exit 2
fi

variant="${TARGET_ANCHORED_VARIANT:-tv_hidden_veto}"
case "$variant" in
  cactus_cap|tv_head|tv_hidden_veto|tv_debt_control|tv_top1_surplus|tv_target_surplus|tv_risk_swap|tv_block_shield|tv_event_shield) ;;
  *)
    echo "Unknown TARGET_ANCHORED_VARIANT=$variant" >&2
    echo "Expected: cactus_cap, tv_head, tv_hidden_veto, tv_debt_control, tv_top1_surplus, tv_target_surplus, tv_risk_swap, tv_block_shield, or tv_event_shield" >&2
    exit 2
    ;;
esac

export REMTP_TA_VARIANT="$variant"
export REMTP_TA_EXPECTED_DRAFT_TOKENS="$MTP_TOKENS"
export REMTP_CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
export REMTP_TA_HEAD_RELIABILITY="${HEAD_RELIABILITY:-1.0,0.85,0.70,0.55,0.40,0.30}"
export REMTP_TA_TARGET_LOG_GAP_SCALE="${TARGET_LOG_GAP_SCALE:-2.0}"
export REMTP_TA_MAX_TARGET_LOG_GAP="${MAX_TARGET_LOG_GAP:-8.0}"
export REMTP_TA_FUTURE_VETO_FLOOR="${FUTURE_VETO_FLOOR:-0.20}"
export REMTP_TA_HIDDEN_RELIABILITY_FLOOR="${HIDDEN_RELIABILITY_FLOOR:-0.25}"
export REMTP_TA_USE_PREFIX_VALUE="${USE_PREFIX_VALUE:-1}"
export REMTP_TA_DEBT_POSITION_LIMIT="${DEBT_POSITION_LIMIT:-0.35}"
export REMTP_TA_DEBT_BLOCK_LIMIT="${DEBT_BLOCK_LIMIT:-1.20}"
export REMTP_TA_DEBT_SOFT_LOG_GAP="${DEBT_SOFT_LOG_GAP:-2.0}"
export REMTP_TA_DEBT_HARD_LOG_GAP="${DEBT_HARD_LOG_GAP:-6.0}"
export REMTP_TA_DEBT_MAX_POSITION_TV="${DEBT_MAX_POSITION_TV:-0.15}"
export REMTP_TA_DEBT_MAX_CACTUS_RATIO="${DEBT_MAX_CACTUS_RATIO:-2.0}"
export REMTP_TA_DEBT_FALLBACK="${DEBT_FALLBACK:-strict}"
export REMTP_TA_SURPLUS_MAX_LOG_GAP="${SURPLUS_MAX_LOG_GAP:-2.0}"
export REMTP_TA_RISK_SWAP_SOFT_LOG_GAP="${RISK_SWAP_SOFT_LOG_GAP:-4.0}"
export REMTP_TA_RISK_SWAP_HARD_LOG_GAP="${RISK_SWAP_HARD_LOG_GAP:-10.0}"
export REMTP_TA_RISK_SWAP_DESTINATION_LOG_GAP="${RISK_SWAP_DESTINATION_LOG_GAP:-2.0}"
export REMTP_TA_BLOCK_SHIELD_CACTUS_MIX="${BLOCK_SHIELD_CACTUS_MIX:-0.30}"
export REMTP_TA_RECOVERY_MODE="${TARGET_ANCHORED_RECOVERY_MODE:-residual}"
export REMTP_TA_AUDIT_INTERVAL="${TARGET_ANCHORED_AUDIT_INTERVAL:-0}"
export REMTP_TA_DIAGNOSTICS="${TARGET_ANCHORED_DIAGNOSTICS:-0}"
export REMTP_TA_COMPILE="${TARGET_ANCHORED_COMPILE:-1}"

# Qwen3.5's GDN Triton kernel cannot be captured safely in this pinned stack.
if [[ -z "${REMTP_COMPILATION_CONFIG:-}" ]]; then
  export REMTP_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
fi

if [[ "$variant" == "tv_risk_swap" || "$variant" == "tv_block_shield" || "$variant" == "tv_event_shield" ]]; then
  echo "[ReMTP][TargetAnchored] variant=$variant mtp_tokens=$MTP_TOKENS risk_swap_gap=$REMTP_TA_RISK_SWAP_SOFT_LOG_GAP:$REMTP_TA_RISK_SWAP_HARD_LOG_GAP destination_gap=$REMTP_TA_RISK_SWAP_DESTINATION_LOG_GAP"
elif [[ "$variant" == "tv_top1_surplus" || "$variant" == "tv_target_surplus" ]]; then
  echo "[ReMTP][TargetAnchored] variant=$variant mtp_tokens=$MTP_TOKENS surplus_max_log_gap=$REMTP_TA_SURPLUS_MAX_LOG_GAP"
else
  echo "[ReMTP][TargetAnchored] variant=$variant mtp_tokens=$MTP_TOKENS debt_position=$REMTP_TA_DEBT_POSITION_LIMIT debt_block=$REMTP_TA_DEBT_BLOCK_LIMIT debt_gap=$REMTP_TA_DEBT_SOFT_LOG_GAP:$REMTP_TA_DEBT_HARD_LOG_GAP debt_max_tv=$REMTP_TA_DEBT_MAX_POSITION_TV debt_ratio=$REMTP_TA_DEBT_MAX_CACTUS_RATIO debt_fallback=$REMTP_TA_DEBT_FALLBACK"
fi
exec "$PROJECT_DIR/scripts/serve.sh"
