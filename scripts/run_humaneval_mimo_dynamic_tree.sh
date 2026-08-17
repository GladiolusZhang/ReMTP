#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

SAMPLES="${SAMPLES:-50}"
SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-512}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
RUN_QUALITY_EVAL="${RUN_QUALITY_EVAL:-0}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
HUMANEVAL_DATA="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
RUN_TAG="${RUN_TAG:-humaneval_dynamic_tree_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"
LOG_ROOT="$PROJECT_DIR/logs/$RUN_TAG"
TREE_RUN_DIR="$RUN_ROOT/tree"
HUMANEVAL_RUN_DIR="$RUN_ROOT/humaneval"
AUDIT="$TREE_RUN_DIR/rounds.jsonl"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$TREE_RUN_DIR"

if [[ ! -f "$HUMANEVAL_DATA" ]]; then
  echo "Missing HumanEval: $HUMANEVAL_DATA" >&2
  echo "Run ./scripts/download_humaneval.sh first." >&2
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
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
trap cleanup_server EXIT
trap 'exit 130' INT TERM

server_log="$LOG_ROOT/server.log"
setsid env \
  PORT="$PORT" \
  RUN_TAG="$RUN_TAG" \
  TREE_RUN_DIR="$TREE_RUN_DIR" \
  REMTP_TREE_AUDIT_JSONL="$AUDIT" \
  REMTP_TREE_AUDIT_DETAIL="${REMTP_TREE_AUDIT_DETAIL:-0}" \
  REMTP_TREE_TRACE="${REMTP_TREE_TRACE:-0}" \
  REMTP_TREE_TRACE_LOG="$TREE_RUN_DIR/tree_trace.log" \
  "$PROJECT_DIR/scripts/serve_mimo_dynamic_tree.sh" \
  >"$server_log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + SERVER_START_TIMEOUT))
while (( SECONDS < deadline )); do
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Dynamic-tree server exited early:" >&2
    tail -n 120 "$server_log" >&2
    exit 1
  fi
  sleep 2
done
if ! curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "Server startup timed out." >&2
  tail -n 120 "$server_log" >&2
  exit 1
fi

echo "Warmup (excluded from HumanEval and tree audit)..."
BASE_URL="$BASE_URL" MAX_TOKENS=32 TEMPERATURE="$TEMPERATURE" SEED="$SEED" \
  "$PROJECT_DIR/scripts/request_mimo.sh" >/dev/null
: > "$AUDIT"
: > "$TREE_RUN_DIR/tree_trace.log"

echo "Running HumanEval: samples=$SAMPLES temperature=$TEMPERATURE"
python -m remtp.humaneval_benchmark \
  --data "$HUMANEVAL_DATA" \
  --samples "$SAMPLES" \
  --sample-seed "$SAMPLE_SEED" \
  --temperature "$TEMPERATURE" \
  --generation-seed "$SEED" \
  --max-tokens "$MAX_TOKENS" \
  --mtp-tokens "${REMTP_DYNAMIC_TREE_MAX_NODES:-32}" \
  --base-url "$BASE_URL" \
  --model XiaomiMiMo/MiMo-7B-Base \
  --run-name mimo_dynamic_tree \
  --output-dir "$HUMANEVAL_RUN_DIR" \
  --progress-every "$PROGRESS_EVERY" \
  --skip-warmup

cleanup_server

if [[ "$RUN_QUALITY_EVAL" == "1" ]]; then
  python -m remtp.humaneval_evaluator "$HUMANEVAL_RUN_DIR" \
    --data "$HUMANEVAL_DATA" \
    --workers "$EVAL_WORKERS" \
    --progress-every "$PROGRESS_EVERY"
fi

python -m remtp.dynamic_tree_humaneval_report \
  --run-dir "$HUMANEVAL_RUN_DIR" \
  --audit "$AUDIT" \
  --output "$RUN_ROOT/comparison.md" \
  --json-output "$RUN_ROOT/comparison.json"

echo
echo "Complete. Open:"
echo "  $RUN_ROOT/comparison.md"
echo "  $TREE_RUN_DIR/tree_summary.md"
echo "  $TREE_RUN_DIR/tree_trace.log"
echo
sed -n '1,80p' "$RUN_ROOT/comparison.md"
