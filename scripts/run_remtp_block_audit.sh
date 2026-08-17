#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

MODE="${1:-all}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  echo "Usage: $0 [gsm8k|humaneval|all]"
  echo "Defaults: GSM8K_SAMPLES=50 HUMANEVAL_SAMPLES=50 MAX_TOKENS=256"
  exit 0
fi
if [[ "$MODE" != "gsm8k" && "$MODE" != "humaneval" && "$MODE" != "all" ]]; then
  echo "Usage: $0 [gsm8k|humaneval|all]" >&2
  exit 2
fi

MTP_TOKENS="${MTP_TOKENS:-6}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-256}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-50}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-50}"
GSM8K_SAMPLE_SEED="${GSM8K_SAMPLE_SEED:-20260730}"
HUMANEVAL_SAMPLE_SEED="${HUMANEVAL_SAMPLE_SEED:-20260802}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
AUDIT_SERVER_SCRIPT="${AUDIT_SERVER_SCRIPT:-$PROJECT_DIR/scripts/serve_remtp_mtp.sh}"
RUN_ROOT="$PROJECT_DIR/results/remtp_block_audit_${RUN_TAG}"
LOG_ROOT="$PROJECT_DIR/logs/remtp_block_audit_${RUN_TAG}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "The ReMTP block audit requires MTP_TOKENS=6." >&2
  exit 2
fi
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi

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
  local log_file="$1"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -n 120 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  tail -n 120 "$log_file" >&2
  return 1
}

collect_one() {
  local dataset="$1"
  local samples="$2"
  local sample_seed="$3"
  local audit_file="$RUN_ROOT/${dataset}_rounds.jsonl"
  local output_dir="$RUN_ROOT/${dataset}_generation"
  local server_log="$LOG_ROOT/${dataset}_server.log"
  local benchmark_log="$LOG_ROOT/${dataset}_benchmark.log"

  echo "===== ReMTP block audit: $dataset ($samples samples) ====="
  setsid env \
    MTP_TOKENS="$MTP_TOKENS" \
    RISK_AUDIT_PATH="$audit_file" \
    RISK_AUDIT_DATASET="$dataset" \
    "$AUDIT_SERVER_SCRIPT" >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server "$server_log"

  if [[ "$dataset" == "gsm8k" ]]; then
    RUN_NAME=remtp_audit \
    SAMPLES="$samples" \
    SAMPLE_SEED="$sample_seed" \
    TEMPERATURE="$TEMPERATURE" \
    SEED="$SEED" \
    MAX_TOKENS="$MAX_TOKENS" \
    MTP_TOKENS="$MTP_TOKENS" \
    "$PROJECT_DIR/scripts/benchmark_gsm8k_risk_entropy.sh" \
      --base-url "$BASE_URL" \
      --output-dir "$output_dir" \
      --progress-every 10 2>&1 | tee "$benchmark_log"
  else
    RUN_NAME=remtp_audit \
    SAMPLES="$samples" \
    SAMPLE_SEED="$sample_seed" \
    TEMPERATURE="$TEMPERATURE" \
    SEED="$SEED" \
    MAX_TOKENS="$MAX_TOKENS" \
    MTP_TOKENS="$MTP_TOKENS" \
    "$PROJECT_DIR/scripts/benchmark_humaneval.sh" \
      --base-url "$BASE_URL" \
      --output-dir "$output_dir" \
      --progress-every 10 2>&1 | tee "$benchmark_log"
  fi
  cleanup_server

  python -m remtp.block_oracle \
    --input "$audit_file" \
    --skip-requests 1 \
    --output-json "$RUN_ROOT/${dataset}_oracle.json" \
    --output-md "$RUN_ROOT/${dataset}_oracle.md"
}

if [[ "$MODE" == "gsm8k" || "$MODE" == "all" ]]; then
  collect_one gsm8k "$GSM8K_SAMPLES" "$GSM8K_SAMPLE_SEED"
fi
if [[ "$MODE" == "humaneval" || "$MODE" == "all" ]]; then
  collect_one humaneval "$HUMANEVAL_SAMPLES" "$HUMANEVAL_SAMPLE_SEED"
fi

echo "Audit results: $RUN_ROOT"
