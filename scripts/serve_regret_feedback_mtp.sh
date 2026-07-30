#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

export REMTP_TRACE=0
export ENFORCE_EAGER=0
export MTP_TOKENS="${MTP_TOKENS:-6}"
export MTP_REJECTION_SAMPLE_METHOD=probabilistic
export REMTP_WORKER_CLS=remtp.worker.RegretFeedbackMTPWorker

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "Budget-Induced Regret Feedback requires MTP_TOKENS=6." >&2
  exit 2
fi

# Keep the current target-anchored verifier exactly unchanged.
export REMTP_TA_VARIANT=tv_hidden_veto
export REMTP_TA_EXPECTED_DRAFT_TOKENS="$MTP_TOKENS"
export REMTP_CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
export REMTP_TA_HEAD_RELIABILITY="${HEAD_RELIABILITY:-1.0,0.85,0.70,0.55,0.40,0.30}"
export REMTP_TA_TARGET_LOG_GAP_SCALE="${TARGET_LOG_GAP_SCALE:-2.0}"
export REMTP_TA_MAX_TARGET_LOG_GAP="${MAX_TARGET_LOG_GAP:-8.0}"
export REMTP_TA_FUTURE_VETO_FLOOR="${FUTURE_VETO_FLOOR:-0.20}"
export REMTP_TA_HIDDEN_RELIABILITY_FLOOR="${HIDDEN_RELIABILITY_FLOOR:-0.25}"
export REMTP_TA_USE_PREFIX_VALUE="${USE_PREFIX_VALUE:-1}"
export REMTP_TA_AUDIT_INTERVAL="${TARGET_ANCHORED_AUDIT_INTERVAL:-0}"
export REMTP_TA_DIAGNOSTICS="${TARGET_ANCHORED_DIAGNOSTICS:-0}"
export REMTP_TA_COMPILE="${TARGET_ANCHORED_COMPILE:-1}"

# Cross-block regret feedback. Only the copied root condition of the next MTP
# block is changed; target logits/KV and target-anchored verification are not.
export REMTP_REGRET_EXPECTED_DRAFT_TOKENS="$MTP_TOKENS"
export REMTP_REGRET_TOP_K="${REGRET_TOP_K:-16}"
export REMTP_REGRET_COMPATIBILITY_TOP_K="${REGRET_COMPATIBILITY_TOP_K:-32}"
export REMTP_REGRET_ALPHA="${REGRET_ALPHA:-0.03}"
export REMTP_REGRET_TOKEN_DECAY="${REGRET_TOKEN_DECAY:-0.90}"
export REMTP_REGRET_REJECTION_RESET="${REGRET_REJECTION_RESET:-0.25}"
export REMTP_REGRET_MAX_IDLE_BLOCKS="${REGRET_MAX_IDLE_BLOCKS:-2}"
export REMTP_REGRET_INCOMPATIBLE_DECAY="${REGRET_INCOMPATIBLE_DECAY:-0.50}"
export REMTP_REGRET_HEAD_RELIABILITY="${HEAD_RELIABILITY:-1.0,0.85,0.70,0.55,0.40,0.30}"
export REMTP_REGRET_AUDIT_INTERVAL="${REGRET_AUDIT_INTERVAL:-100}"
export REMTP_REGRET_DIAGNOSTICS="${REGRET_DIAGNOSTICS:-0}"

# Qwen3.5's GDN Triton kernel cannot be captured safely in this pinned stack.
if [[ -z "${REMTP_COMPILATION_CONFIG:-}" ]]; then
  export REMTP_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}'
fi

echo "[ReMTP][Regret] mtp_tokens=$MTP_TOKENS alpha=$REMTP_REGRET_ALPHA decay=$REMTP_REGRET_TOKEN_DECAY"
exec "$PROJECT_DIR/scripts/serve.sh"
