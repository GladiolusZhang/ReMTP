#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

if [[ -z "${ROUTER_CORPUS:-}" ]]; then
  echo "Set ROUTER_CORPUS to a JSON/JSONL corpus." >&2
  echo "Rows may contain messages, prompt, instruction, or text." >&2
  exit 2
fi
if curl -fsS "${BASE_URL:-http://127.0.0.1:8000}/health" >/dev/null 2>&1; then
  echo "A server is already responding; stop it first." >&2
  exit 2
fi

COLLECT_DIR="${REGRET_ROUTER_COLLECT_DIR:-$PROJECT_DIR/data/regret_router/traces}"
SERVER_LOG="${REGRET_ROUTER_SERVER_LOG:-$PROJECT_DIR/logs/regret_router_collect_server.log}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
mkdir -p "$COLLECT_DIR" "$(dirname "$SERVER_LOG")"

server_pid=""
cleanup() {
  if [[ -z "$server_pid" ]]; then
    return
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -INT -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
trap cleanup EXIT INT TERM

setsid env \
  REGRET_ROUTER_MODE=collect \
  REGRET_ROUTER_COLLECT_DIR="$COLLECT_DIR" \
  MTP_TOKENS=6 \
  "$PROJECT_DIR/scripts/serve_exact_tv_regret_router.sh" \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!

deadline=$((SECONDS + SERVER_START_TIMEOUT))
until curl -fsS "$BASE_URL/health" >/dev/null 2>&1; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 100 "$SERVER_LOG" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Router collection server startup timed out." >&2
    tail -n 100 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 2
done

python -m remtp.regret_router_collect \
  --data "$ROUTER_CORPUS" \
  --samples "${SAMPLES:-1000}" \
  --sample-seed "${SAMPLE_SEED:-20260803}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --generation-seed "${SEED:-42}" \
  --max-tokens "${MAX_TOKENS:-256}" \
  --base-url "$BASE_URL" \
  --progress-every "${PROGRESS_EVERY:-10}"

cleanup
echo "Router traces: $COLLECT_DIR"
