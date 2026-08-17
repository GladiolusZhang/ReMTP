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
MTP_TOKENS="${MTP_TOKENS:-3}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
HUMANEVAL_DATA="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
DYNAMIC_RESULT="${DYNAMIC_RESULT:-$PROJECT_DIR/results/humaneval_dynamic_tree_20260807_154416/comparison.json}"
RUN_TAG="${RUN_TAG:-mimo_chain_baselines_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"
LOG_ROOT="$PROJECT_DIR/logs/$RUN_TAG"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

if [[ ! -f "$HUMANEVAL_DATA" || ! -f "$DYNAMIC_RESULT" ]]; then
  echo "Missing HumanEval data or dynamic result." >&2
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

for method in native cactus; do
  run_dir="$RUN_ROOT/$method"
  log_file="$LOG_ROOT/$method.log"
  serve_script="$PROJECT_DIR/scripts/serve_mimo_${method}.sh"
  extra_env=()
  if [[ "$method" == "cactus" ]]; then
    extra_env+=(REMTP_CACTUS_DIAGNOSTICS=0 REMTP_CACTUS_DELTA="${REMTP_CACTUS_DELTA:-1.0}")
  fi
  echo "Starting $method..."
  setsid env PORT="$PORT" MTP_TOKENS="$MTP_TOKENS" \
    MIMO_MTP_LAYER_MODE=physical ENFORCE_EAGER=1 NO_ASYNC_SCHEDULING=1 \
    REMTP_MIMO_PREFILL_ALL_LAYERS="${REMTP_MIMO_PREFILL_ALL_LAYERS:-1}" \
    "${extra_env[@]}" "$serve_script" >"$log_file" 2>&1 &
  server_pid=$!
  deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -n 120 "$log_file" >&2
      exit 1
    fi
    sleep 2
  done
  curl -fsS "$BASE_URL/health" >/dev/null

  BASE_URL="$BASE_URL" MAX_TOKENS=32 TEMPERATURE="$TEMPERATURE" SEED="$SEED" \
    "$PROJECT_DIR/scripts/request_mimo.sh" >/dev/null
  python -m remtp.humaneval_benchmark \
    --data "$HUMANEVAL_DATA" --samples "$SAMPLES" \
    --sample-seed "$SAMPLE_SEED" --temperature "$TEMPERATURE" \
    --generation-seed "$SEED" --max-tokens "$MAX_TOKENS" \
    --mtp-tokens "$MTP_TOKENS" --base-url "$BASE_URL" \
    --model XiaomiMiMo/MiMo-7B-Base --run-name "$method" \
    --output-dir "$run_dir" --progress-every "$PROGRESS_EVERY" --skip-warmup
  cleanup_server
  python -m remtp.humaneval_evaluator "$run_dir" \
    --data "$HUMANEVAL_DATA" --workers "$EVAL_WORKERS" \
    --progress-every "$PROGRESS_EVERY"
done

python -m remtp.mimo_tree_comparison_report \
  --native "$RUN_ROOT/native" --cactus "$RUN_ROOT/cactus" \
  --dynamic "$DYNAMIC_RESULT" --output "$RUN_ROOT/comparison.md" \
  --json-output "$RUN_ROOT/comparison.json"
sed -n '1,100p' "$RUN_ROOT/comparison.md"
