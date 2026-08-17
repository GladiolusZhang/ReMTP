#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

usage() {
  cat <<'EOF'
Three-way comparison on TencentBAC/FastMTP (Qwen2-based with trained MTP):
  1. Native probabilistic MTP (baseline with trained MTP)
  2. Cactus + MTP
  3. SpecCascade TokenV3 + MTP

Uses 50 samples from GSM8K and 50 from HumanEval.

FastMTP advantages over MiMo:
  - Based on Qwen2 (not MiMo architecture)
  - 1 TRAINED MTP layer (not pretrained-only)
  - 2.03x speedup, 82% better than vanilla MTP
  - Position-shared weights for better long-range dependencies

Usage:
  # First download the model
  ./scripts/download_fastmtp.sh

  # Then run comparison
  RUN_TAG=fastmtp_three_way_50_$(date +%Y%m%d) \
    ./scripts/run_fastmtp_three_way_50.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SAMPLES=50
GSM8K_SAMPLE_SEED=20260730
HUMANEVAL_SAMPLE_SEED=20260802
TEMPERATURE=0.7
SEED=42
GSM8K_MAX_TOKENS=2048
HUMANEVAL_MAX_TOKENS=2048
MTP_TOKENS=3
MAX_MODEL_LEN=8192
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:$PORT"
SERVER_START_TIMEOUT=360
PROGRESS_EVERY=10
EVAL_WORKERS=4
HUMANEVAL_IMAGE="python:3-slim"
HUMANEVAL_EVAL_TIMEOUT=8

GSM8K_DATA="$PROJECT_DIR/data/gsm8k/test.jsonl"
HUMANEVAL_DATA="$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/FastMTP}"
MODEL_NAME="TencentBAC/FastMTP"
RUN_TAG="${RUN_TAG:-fastmtp_three_way_50_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"
LOG_ROOT="$PROJECT_DIR/logs/$RUN_TAG"

METHODS=(native_mtp cactus spec_cascade)

CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
CASCADE_RULE="${CASCADE_RULE:-token_v3}"
CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"

# Validate prerequisites
for path in "$MODEL_PATH/config.json" "$GSM8K_DATA" "$HUMANEVAL_DATA"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    [[ "$path" == "$MODEL_PATH/config.json" ]] && \
      echo "Download FastMTP with: ./scripts/download_fastmtp.sh" >&2
    exit 2
  fi
done

if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable; HumanEval cannot be evaluated safely." >&2
  exit 2
fi

if ! docker image inspect "$HUMANEVAL_IMAGE" >/dev/null 2>&1; then
  echo "Missing Docker image. Run: docker pull $HUMANEVAL_IMAGE" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/gsm8k" "$RUN_ROOT/humaneval" "$LOG_ROOT"
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
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then return; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before becoming healthy:" >&2
      tail -n 160 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "Server startup timed out:" >&2
  tail -n 160 "$log_file" >&2
  return 1
}

humaneval_complete() {
  [[ -f "$1/summary.json" ]] || return 1
  python - "$1/summary.json" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1], encoding="utf-8")).get("results") or []
raise SystemExit(0 if len(rows) == 1 and rows[0].get("evaluation_status") == "complete" else 1)
PY
}

refuse_partial() {
  if [[ -d "$1" && ! -f "$1/summary.json" ]]; then
    echo "Partial output directory found: $1" >&2
    echo "Move it aside or choose another RUN_TAG." >&2
    exit 2
  fi
}

start_server() {
  local method="$1"
  local log_file="$2"

  case "$method" in
    native_mtp)
      setsid env PORT="$PORT" MODEL_PATH="$MODEL_PATH" \
        SERVED_MODEL_NAME="$MODEL_NAME" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MTP_TOKENS="$MTP_TOKENS" ENFORCE_EAGER=1 NO_ASYNC_SCHEDULING=1 \
        "$PROJECT_DIR/scripts/serve_fastmtp_native.sh" >"$log_file" 2>&1 &
      ;;
    cactus)
      setsid env PORT="$PORT" MODEL_PATH="$MODEL_PATH" \
        SERVED_MODEL_NAME="$MODEL_NAME" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MTP_TOKENS="$MTP_TOKENS" ENFORCE_EAGER=1 NO_ASYNC_SCHEDULING=1 \
        REMTP_CACTUS_DELTA="$CACTUS_DELTA" REMTP_CACTUS_DIAGNOSTICS=0 \
        WORKER_CLS="remtp.fastmtp_worker.FastMTPCactusWorker" \
        "$PROJECT_DIR/scripts/serve_fastmtp_native.sh" >"$log_file" 2>&1 &
      ;;
    spec_cascade)
      setsid env PORT="$PORT" MODEL_PATH="$MODEL_PATH" \
        SERVED_MODEL_NAME="$MODEL_NAME" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MTP_TOKENS="$MTP_TOKENS" ENFORCE_EAGER=1 NO_ASYNC_SCHEDULING=1 \
        CASCADE_RULE="$CASCADE_RULE" CASCADE_ALPHA="$CASCADE_ALPHA" \
        WORKER_CLS="remtp.fastmtp_worker.FastMTPSpecCascadeWorker" \
        "$PROJECT_DIR/scripts/serve_fastmtp_native.sh" >"$log_file" 2>&1 &
      ;;
    *) echo "Unknown method: $method" >&2; exit 2 ;;
  esac
  server_pid=$!
  wait_for_server "$log_file"
}

run_method() {
  local method="$1"
  local gsm_dir="$RUN_ROOT/gsm8k/$method"
  local human_dir="$RUN_ROOT/humaneval/$method"
  local server_log="$LOG_ROOT/${method}_server.log"
  local need_gsm=1 need_human=1

  refuse_partial "$gsm_dir"
  refuse_partial "$human_dir"
  [[ -f "$gsm_dir/summary.json" ]] && need_gsm=0
  [[ -f "$human_dir/summary.json" ]] && need_human=0

  if (( need_gsm == 1 || need_human == 1 )); then
    echo "===== $method ====="
    start_server "$method" "$server_log"
    MODEL="$MODEL_NAME" BASE_URL="$BASE_URL" TEMPERATURE="$TEMPERATURE" \
      MAX_TOKENS=32 SEED="$SEED" "$PROJECT_DIR/scripts/request_mimo.sh" >/dev/null

    if (( need_gsm == 1 )); then
      python -m remtp.gsm8k_benchmark \
        --data "$GSM8K_DATA" --samples "$SAMPLES" \
        --sample-seed "$GSM8K_SAMPLE_SEED" --temperature "$TEMPERATURE" \
        --generation-seed "$SEED" --max-tokens "$GSM8K_MAX_TOKENS" \
        --mtp-tokens "$MTP_TOKENS" --base-url "$BASE_URL" --model "$MODEL_NAME" \
        --run-name "$method" --output-dir "$gsm_dir" \
        --progress-every "$PROGRESS_EVERY" --skip-warmup \
        2>&1 | tee "$LOG_ROOT/${method}_gsm8k.log"
    fi

    if (( need_human == 1 )); then
      python -m remtp.humaneval_benchmark \
        --data "$HUMANEVAL_DATA" --samples "$SAMPLES" \
        --sample-seed "$HUMANEVAL_SAMPLE_SEED" --temperature "$TEMPERATURE" \
        --generation-seed "$SEED" --max-tokens "$HUMANEVAL_MAX_TOKENS" \
        --mtp-tokens "$MTP_TOKENS" --base-url "$BASE_URL" --model "$MODEL_NAME" \
        --run-name "$method" --output-dir "$human_dir" \
        --progress-every "$PROGRESS_EVERY" --skip-warmup \
        2>&1 | tee "$LOG_ROOT/${method}_humaneval.log"
    fi
    cleanup_server
  else
    echo "[resume] generation complete: $method"
  fi

  if ! humaneval_complete "$human_dir"; then
    python -m remtp.humaneval_evaluator "$human_dir" \
      --data "$HUMANEVAL_DATA" --image "$HUMANEVAL_IMAGE" \
      --timeout "$HUMANEVAL_EVAL_TIMEOUT" --workers "$EVAL_WORKERS" \
      --progress-every "$PROGRESS_EVERY" \
      2>&1 | tee "$LOG_ROOT/${method}_humaneval_eval.log"
  fi
}

cat <<EOF
FastMTP Three-Way Comparison (50 samples)
  samples          : $SAMPLES per dataset
  model            : TencentBAC/FastMTP (Qwen2 + trained MTP)
  temperature      : $TEMPERATURE
  max tokens       : GSM8K=$GSM8K_MAX_TOKENS HumanEval=$HUMANEVAL_MAX_TOKENS
  methods          : native_mtp, cactus, spec_cascade
  output           : $RUN_ROOT
EOF

for method in "${METHODS[@]}"; do
  run_method "$method"
done

python -m remtp.fastmtp_three_way_report --run-root "$RUN_ROOT"
echo
echo "Complete: $RUN_ROOT/comparison.md"
sed -n '1,200p' "$RUN_ROOT/comparison.md"
