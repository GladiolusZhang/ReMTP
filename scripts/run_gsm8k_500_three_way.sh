#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Run the three GSM8K methods sequentially on one GPU.

Usage:
  ./scripts/run_gsm8k_500_three_way.sh

Optional environment variables:
  SAMPLES=500                 Number of shared GSM8K samples
  SAMPLE_SEED=20260730        Fixed subset seed
  TEMPERATURE=0.7             Sampling temperature
  SEED=42                     Generation seed
  MAX_TOKENS=384              Per-answer token limit
  MTP_TOKENS=2                MTP draft tokens per round
  CASCADE_ALPHA=0.5           TokenV3 alpha
  CACTUS_DELTA=1.0            Cactus delta
  PROGRESS_EVERY=25           Print one sample row every N requests
  SERVER_START_TIMEOUT=180    Startup timeout in seconds
  RUN_TAG=<timestamp>         Optional local output suffix

Results and comparison files are written only under results/.
Server and benchmark logs are written only under logs/.
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

SAMPLES="${SAMPLES:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-20260730}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MTP_TOKENS="${MTP_TOKENS:-2}"
CASCADE_RULE="${CASCADE_RULE:-token_v3}"
CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"
CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

RUN_ROOT="$PROJECT_DIR/results/gsm8k_${SAMPLES}_three_way_${RUN_TAG}"
LOG_ROOT="$PROJECT_DIR/logs/gsm8k_${SAMPLES}_three_way_${RUN_TAG}"
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
  local log_file="$1"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before becoming healthy. Last log lines:" >&2
      tail -n 80 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "Server health check timed out after ${SERVER_START_TIMEOUT}s." >&2
  tail -n 80 "$log_file" >&2
  return 1
}

run_method() {
  local label="$1"
  local result_name="$2"
  local serve_script="$3"
  local benchmark_script="$4"
  local server_log="$LOG_ROOT/${result_name}_server.log"
  local benchmark_log="$LOG_ROOT/${result_name}_benchmark.log"

  echo
  echo "===== ${label} ====="
  echo "Starting server; log: $server_log"
  setsid "$serve_script" >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server "$server_log"

  echo "Server ready. Running ${SAMPLES} GSM8K samples."
  SAMPLES="$SAMPLES" \
  SAMPLE_SEED="$SAMPLE_SEED" \
  TEMPERATURE="$TEMPERATURE" \
  SEED="$SEED" \
  MAX_TOKENS="$MAX_TOKENS" \
  MTP_TOKENS="$MTP_TOKENS" \
  CASCADE_RULE="$CASCADE_RULE" \
  CASCADE_ALPHA="$CASCADE_ALPHA" \
  CACTUS_DELTA="$CACTUS_DELTA" \
  "$benchmark_script" \
    --base-url "$BASE_URL" \
    --output-dir "$RUN_ROOT/$result_name" \
    --progress-every "$PROGRESS_EVERY" \
    2>&1 | tee "$benchmark_log"

  cleanup_server
}

if [[ ! -f "$PROJECT_DIR/data/gsm8k/test.jsonl" ]]; then
  echo "Missing data/gsm8k/test.jsonl. Download GSM8K first." >&2
  exit 2
fi
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
mkdir "$RUN_ROOT"

export MTP_TOKENS CASCADE_RULE CASCADE_ALPHA CACTUS_DELTA

echo "GSM8K three-way run"
echo "samples=$SAMPLES sample_seed=$SAMPLE_SEED temperature=$TEMPERATURE"
echo "generation_seed=$SEED max_tokens=$MAX_TOKENS mtp_tokens=$MTP_TOKENS"
echo "local results=$RUN_ROOT"

run_method \
  "Standard probabilistic MTP" \
  "standard_mtp" \
  "$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh" \
  "$PROJECT_DIR/scripts/benchmark_gsm8k_probabilistic_mtp.sh"

run_method \
  "SpecCascade [TokenV3]" \
  "tokenv3" \
  "$PROJECT_DIR/scripts/serve_spec_cascade.sh" \
  "$PROJECT_DIR/scripts/benchmark_gsm8k_spec_cascade.sh"

run_method \
  "Cactus + MTP" \
  "cactus" \
  "$PROJECT_DIR/scripts/serve_cactus_mtp.sh" \
  "$PROJECT_DIR/scripts/benchmark_gsm8k_cactus_mtp.sh"

python -m remtp.gsm8k_compare "$RUN_ROOT"
