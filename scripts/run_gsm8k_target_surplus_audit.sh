#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

SAMPLES="${SAMPLES:-10}"
SAMPLE_SEED="${SAMPLE_SEED:-20260804}"
MAX_TOKENS="${MAX_TOKENS:-128}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/gsm8k_target_surplus_audit_${RUN_TAG}"
LOG_ROOT="$PROJECT_DIR/logs/gsm8k_target_surplus_audit_${RUN_TAG}"
server_pid=""

cleanup() {
  if [[ -z "$server_pid" ]]; then
    return
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -INT -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        break
      fi
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
trap cleanup EXIT INT TERM

if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"

setsid env \
  MTP_TOKENS=6 \
  SURPLUS_MAX_LOG_GAP="${SURPLUS_MAX_LOG_GAP:-2.0}" \
  TARGET_ANCHORED_AUDIT_INTERVAL=1 \
  TARGET_ANCHORED_DIAGNOSTICS=1 \
  "$PROJECT_DIR/scripts/serve_target_surplus_mtp.sh" \
  >"$LOG_ROOT/server.log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + ${SERVER_START_TIMEOUT:-180}))
while ! curl -fsS "$BASE_URL/health" >/dev/null 2>&1; do
  if ! kill -0 "$server_pid" 2>/dev/null || (( SECONDS >= deadline )); then
    tail -n 100 "$LOG_ROOT/server.log" >&2
    exit 1
  fi
  sleep 2
done

SAMPLES="$SAMPLES" \
SAMPLE_SEED="$SAMPLE_SEED" \
TEMPERATURE="${TEMPERATURE:-0.7}" \
SEED="${SEED:-42}" \
MAX_TOKENS="$MAX_TOKENS" \
MTP_TOKENS=6 \
RUN_NAME=target_surplus_audit \
"$PROJECT_DIR/scripts/benchmark_gsm8k.sh" \
  --base-url "$BASE_URL" \
  --output-dir "$RUN_ROOT" \
  --progress-every 1

cleanup
server_pid=""
echo "Audit results: $RUN_ROOT"
echo "Audit log: $LOG_ROOT/server.log"
