#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

DATASET="${1:-gsm8k}"
if [[ "$DATASET" == "-h" || "$DATASET" == "--help" ]]; then
  cat <<'EOF'
Run a few requests and render every MTP verification decision.

Usage:
  ./scripts/run_proposal_decision_trace.sh [gsm8k|humaneval]

Examples:
  SAMPLES=3 MTP_TOKENS=6 ./scripts/run_proposal_decision_trace.sh gsm8k

  METHOD=remtp RISK_BLOCK_TV=2.4 RISK_BASE_GAP=2.3 SAMPLES=3 \
    ./scripts/run_proposal_decision_trace.sh gsm8k

  PROPOSAL_CONFIG=configs/my_proposal.json SAMPLES=2 MAX_TOKENS=128 \
    ./scripts/run_proposal_decision_trace.sh humaneval

METHOD may be `proposal_calibrated` (strict verification) or `remtp` (relaxed
verification). The generated decision_trace.md contains proposal/target top-k candidates,
the first rejection in each round, recovery/bonus tokens, and final outputs.
This is a synchronization-heavy debugging run and is invalid for speed tests.
EOF
  exit 0
fi
case "$DATASET" in gsm8k|humaneval) ;; *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;; esac

SAMPLES="${SAMPLES:-3}"
METHOD="${METHOD:-proposal_calibrated}"
MTP_TOKENS="${MTP_TOKENS:-6}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-96}"
TRACE_TOPK="${TRACE_TOPK:-5}"
MAX_ROUNDS_PER_REQUEST="${MAX_ROUNDS_PER_REQUEST:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-240}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen3.5-4B}"
PROPOSAL_CONFIG="${PROPOSAL_CONFIG:-$PROJECT_DIR/configs/proposal_calibration_static.json}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/${METHOD}_decision_trace_${DATASET}_${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/${METHOD}_decision_trace_${DATASET}_${RUN_TAG}}"

case "$METHOD" in proposal_calibrated|remtp) ;; *)
  echo "METHOD must be proposal_calibrated or remtp" >&2
  exit 2
  ;;
esac

if [[ "$DATASET" == "gsm8k" ]]; then
  SAMPLE_SEED="${SAMPLE_SEED:-20260730}"
  DATA_PATH="${GSM8K_DATA:-$PROJECT_DIR/data/gsm8k/test.jsonl}"
else
  SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
  DATA_PATH="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
fi

[[ -f "$DATA_PATH" ]] || { echo "Missing dataset: $DATA_PATH" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 2; }
if [[ "$METHOD" == "proposal_calibrated" ]]; then
  [[ -f "$PROPOSAL_CONFIG" ]] || {
    echo "Missing config: $PROPOSAL_CONFIG" >&2
    exit 2
  }
fi
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
TRACE_PATH="$RUN_ROOT/decisions.jsonl"
GENERATION_DIR="$RUN_ROOT/generation"
SERVER_LOG="$LOG_ROOT/server.log"
BENCHMARK_LOG="$LOG_ROOT/benchmark.log"
rm -f "$TRACE_PATH"

server_pid=""
cleanup_server() {
  if [[ -z "$server_pid" ]]; then return; fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -INT -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
      sleep 1
    done
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    sleep 2
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -KILL -- "-$server_pid" 2>/dev/null || true
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
trap cleanup_server EXIT
trap 'exit 130' INT TERM

wait_for_server() {
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -n 160 "$SERVER_LOG" >&2
      return 1
    fi
    sleep 2
  done
  tail -n 160 "$SERVER_LOG" >&2
  return 1
}

echo "[trace] method=$METHOD dataset=$DATASET samples=$SAMPLES mtp=$MTP_TOKENS"
if [[ "$METHOD" == "proposal_calibrated" ]]; then
  SERVER_SCRIPT="$PROJECT_DIR/scripts/serve_proposal_calibrated_mtp.sh"
  TRACE_ENV=(
    REMTP_PC_CONFIG="$PROPOSAL_CONFIG"
    REMTP_PC_TRACE_PATH="$TRACE_PATH"
    REMTP_PC_TRACE_MODE=decisions
    REMTP_PC_TRACE_TOPK="$TRACE_TOPK"
    REMTP_PC_TRACE_DATASET="$DATASET"
  )
else
  SERVER_SCRIPT="$PROJECT_DIR/scripts/serve_remtp_mtp.sh"
  TRACE_ENV=(
    RISK_AUDIT_PATH="$TRACE_PATH"
    RISK_AUDIT_DATASET="$DATASET"
    REMTP_ORACLE_TOPK="$TRACE_TOPK"
  )
fi
setsid env MODEL_PATH="$MODEL_PATH" MTP_TOKENS="$MTP_TOKENS" \
  "${TRACE_ENV[@]}" "$SERVER_SCRIPT" >"$SERVER_LOG" 2>&1 &
server_pid=$!
wait_for_server

if [[ "$DATASET" == "gsm8k" ]]; then
  RUN_NAME="${METHOD}_decision_trace" \
  GSM8K_DATA="$DATA_PATH" SAMPLES="$SAMPLES" SAMPLE_SEED="$SAMPLE_SEED" \
  TEMPERATURE="$TEMPERATURE" SEED="$SEED" MAX_TOKENS="$MAX_TOKENS" \
  MTP_TOKENS="$MTP_TOKENS" \
    "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" \
    --base-url "$BASE_URL" --output-dir "$GENERATION_DIR" \
    --progress-every 1 --skip-warmup 2>&1 | tee "$BENCHMARK_LOG"
else
  RUN_NAME="${METHOD}_decision_trace" \
  HUMANEVAL_DATA="$DATA_PATH" SAMPLES="$SAMPLES" SAMPLE_SEED="$SAMPLE_SEED" \
  TEMPERATURE="$TEMPERATURE" SEED="$SEED" MAX_TOKENS="$MAX_TOKENS" \
  MTP_TOKENS="$MTP_TOKENS" \
    "$PROJECT_DIR/scripts/benchmark_humaneval.sh" \
    --base-url "$BASE_URL" --output-dir "$GENERATION_DIR" \
    --progress-every 1 --skip-warmup 2>&1 | tee "$BENCHMARK_LOG"
fi

cleanup_server
[[ -s "$TRACE_PATH" ]] || { echo "No trace was written: $TRACE_PATH" >&2; exit 1; }
[[ -s "$GENERATION_DIR/requests.jsonl" ]] || {
  echo "No request records were written: $GENERATION_DIR/requests.jsonl" >&2
  exit 1
}

python -m remtp.proposal_decision_report \
  --trace "$TRACE_PATH" \
  --requests "$GENERATION_DIR/requests.jsonl" \
  --tokenizer "$MODEL_PATH" \
  --dataset "$DATASET" \
  --data "$DATA_PATH" \
  --output-md "$RUN_ROOT/decision_trace.md" \
  --output-json "$RUN_ROOT/decision_trace.json" \
  --max-rounds-per-request "$MAX_ROUNDS_PER_REQUEST"

echo
echo "Open: $RUN_ROOT/decision_trace.md"
echo "Raw trace: $TRACE_PATH"
echo "Reminder: do not report speed from this traced run."
