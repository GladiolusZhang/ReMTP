#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Run only Budget-Induced Regret Feedback and append it to a completed MTP=6
five-way GSM8K table. The previous result directory is never modified.

Usage:
  BASELINE_RUN_ROOT=results/gsm8k_mtp6_five_way_20260730_202927 \
    ./scripts/run_gsm8k_mtp6_regret_append.sh

If BASELINE_RUN_ROOT is omitted, the newest complete non-sanity five-way run
is selected.

Experiment variables (defaults match the existing 200-sample table):
  SAMPLES=200
  SAMPLE_SEED=20260730
  TEMPERATURE=0.7
  SEED=42
  MAX_TOKENS=384
  MTP_TOKENS=6

Regret variables:
  REGRET_ALPHA=0.03
  REGRET_TOKEN_DECAY=0.90
  REGRET_TOP_K=16
  REGRET_COMPATIBILITY_TOP_K=32
  REGRET_REJECTION_RESET=0.25
  REGRET_MAX_IDLE_BLOCKS=2
  REGRET_INCOMPATIBLE_DECAY=0.50
  REGRET_AUDIT_INTERVAL=100

All generated tables, audit files, and logs stay under results/ and logs/.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

SAMPLES="${SAMPLES:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-20260730}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MTP_TOKENS="${MTP_TOKENS:-6}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "This experiment requires MTP_TOKENS=6." >&2
  exit 2
fi
if [[ ! -f "$PROJECT_DIR/data/gsm8k/test.jsonl" ]]; then
  echo "Missing data/gsm8k/test.jsonl. Download GSM8K first." >&2
  exit 2
fi
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi

if [[ -z "${BASELINE_RUN_ROOT:-}" ]]; then
  BASELINE_RUN_ROOT="$(
    find "$PROJECT_DIR/results" -maxdepth 1 -type d \
      -name 'gsm8k_mtp6_five_way_*' ! -name '*sanity*' \
      -printf '%T@ %p\n' \
      | sort -nr \
      | awk 'NR==1 {print $2}'
  )"
fi
if [[ -z "$BASELINE_RUN_ROOT" ]]; then
  echo "No completed five-way baseline was found." >&2
  exit 2
fi
BASELINE_RUN_ROOT="$(realpath "$BASELINE_RUN_ROOT")"
if [[ ! -f "$BASELINE_RUN_ROOT/comparison.json" ]]; then
  echo "Missing baseline comparison: $BASELINE_RUN_ROOT/comparison.json" >&2
  exit 2
fi

RUN_ROOT="$PROJECT_DIR/results/gsm8k_mtp6_regret_${RUN_TAG}"
LOG_ROOT="$PROJECT_DIR/logs/gsm8k_mtp6_regret_${RUN_TAG}"
SERVER_LOG="$LOG_ROOT/regret_feedback_server.log"
BENCHMARK_LOG="$LOG_ROOT/regret_feedback_benchmark.log"
server_pid=""

cleanup_server() {
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

trap cleanup_server EXIT
trap 'exit 130' INT TERM

wait_for_server() {
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before becoming healthy. Last log lines:" >&2
      tail -n 100 "$SERVER_LOG" >&2
      return 1
    fi
    sleep 2
  done
  echo "Server startup timed out after ${SERVER_START_TIMEOUT}s." >&2
  tail -n 100 "$SERVER_LOG" >&2
  return 1
}

mkdir -p "$LOG_ROOT"
mkdir "$RUN_ROOT"

echo "GSM8K MTP=6 Budget-Induced Regret Feedback"
echo "baseline=$BASELINE_RUN_ROOT"
echo "samples=$SAMPLES sample_seed=$SAMPLE_SEED temperature=$TEMPERATURE"
echo "alpha=${REGRET_ALPHA:-0.03} decay=${REGRET_TOKEN_DECAY:-0.90}"
echo "local_results=$RUN_ROOT"

setsid "$PROJECT_DIR/scripts/serve_regret_feedback_mtp.sh" \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!
wait_for_server

SAMPLES="$SAMPLES" \
SAMPLE_SEED="$SAMPLE_SEED" \
TEMPERATURE="$TEMPERATURE" \
SEED="$SEED" \
MAX_TOKENS="$MAX_TOKENS" \
MTP_TOKENS="$MTP_TOKENS" \
"$PROJECT_DIR/scripts/benchmark_gsm8k_regret_feedback_mtp.sh" \
  --base-url "$BASE_URL" \
  --output-dir "$RUN_ROOT/regret_feedback" \
  --progress-every "$PROGRESS_EVERY" \
  2>&1 | tee "$BENCHMARK_LOG"

cleanup_server

if ! cmp -s \
  "$BASELINE_RUN_ROOT/cactus/sample_manifest.json" \
  "$RUN_ROOT/regret_feedback/sample_manifest.json"; then
  echo "The regret run did not use the same GSM8K subset as the baseline." >&2
  exit 2
fi

python -m remtp.regret_compare \
  "$BASELINE_RUN_ROOT" \
  "$RUN_ROOT/regret_feedback/summary.json" \
  "$SERVER_LOG" \
  "$RUN_ROOT"

echo "Combined local table: $RUN_ROOT/comparison.md"
echo "Mechanism audit: $RUN_ROOT/regret_mechanism.json"
