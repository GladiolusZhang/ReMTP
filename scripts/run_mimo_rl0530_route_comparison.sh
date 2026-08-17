#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

# Every model/tokenizer path used below is local. Keeping the benchmark offline
# prevents an accidental Hub or mirror lookup from changing the protocol.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

usage() {
  cat <<'EOF'
Compare MiMo-7B-RL-0530 under one quality protocol:

  target_only  : no speculative decoding (quality baseline)
  native_012   : strict MTP using physical layers 0,1,2
  native_000   : strict MTP reusing physical layer 0
  tree_012     : relaxed dynamic tree using 0,1,2 plus EOS protection
  tree_000     : relaxed dynamic tree reusing 0 plus EOS protection

Quick pilot:
  SAMPLES=20 RUN_TAG=mimo_rl0530_routes_pilot \
    ./scripts/run_mimo_rl0530_route_comparison.sh

Formal run:
  SAMPLES=100 RUN_TAG=mimo_rl0530_routes_n100 \
    ./scripts/run_mimo_rl0530_route_comparison.sh

Rerun the same command to resume completed methods. A partial method directory
is not overwritten; move it aside or use a new RUN_TAG.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 0 )); then
  usage >&2
  exit 2
fi

SAMPLES="${SAMPLES:-100}"
GSM8K_SAMPLE_SEED="${GSM8K_SAMPLE_SEED:-20260730}"
HUMANEVAL_SAMPLE_SEED="${HUMANEVAL_SAMPLE_SEED:-20260802}"
TEMPERATURE="${TEMPERATURE:-0.6}"
SEED="${SEED:-42}"
GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-2048}"
HUMANEVAL_MAX_TOKENS="${HUMANEVAL_MAX_TOKENS:-2048}"
MTP_TOKENS=3
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-360}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
HUMANEVAL_IMAGE="${HUMANEVAL_IMAGE:-python:3-slim}"
HUMANEVAL_EVAL_TIMEOUT="${HUMANEVAL_EVAL_TIMEOUT:-8}"
GSM8K_DATA="${GSM8K_DATA:-$PROJECT_DIR/data/gsm8k/test.jsonl}"
HUMANEVAL_DATA="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-$PROJECT_DIR/models/MiMo-7B-RL-0530}"
MTP_MODEL_PATH="${MTP_MODEL_PATH:-$PROJECT_DIR/models/MiMo-7B-RL-0530-MTP3}"
MODEL_NAME="${MODEL_NAME:-XiaomiMiMo/MiMo-7B-RL-0530}"
RUN_TAG="${RUN_TAG:-mimo_rl0530_routes_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"
LOG_ROOT="$PROJECT_DIR/logs/$RUN_TAG"
METHODS=(target_only native_012 native_000 tree_012 tree_000)

DYNAMIC_MAX_DEPTH="${DYNAMIC_MAX_DEPTH:-3}"
DYNAMIC_MAX_NODES="${DYNAMIC_MAX_NODES:-32}"
DYNAMIC_TAU_MIN="${DYNAMIC_TAU_MIN:-0.02}"
DYNAMIC_KAPPA="${DYNAMIC_KAPPA:-1.0}"
DYNAMIC_MU="${DYNAMIC_MU:-0.5}"
DYNAMIC_ETA="${DYNAMIC_ETA:-0.25}"
DYNAMIC_ALPHA="${DYNAMIC_ALPHA:-0.5}"
DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.7}"
DYNAMIC_BETA="${DYNAMIC_BETA:-0.5}"
DYNAMIC_PATH_TEMPERATURE="${DYNAMIC_PATH_TEMPERATURE:-0.3}"
EOS_THRESHOLD="${EOS_THRESHOLD:-0.5}"

if [[ "$TEMPERATURE" != "0.6" ]]; then
  echo "This quality protocol fixes TEMPERATURE=0.6 (received $TEMPERATURE)." >&2
  exit 2
fi
for path in "$TARGET_MODEL_PATH/config.json" "$MTP_MODEL_PATH/config.json" \
            "$GSM8K_DATA" "$HUMANEVAL_DATA"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    echo "Download the model first with ./scripts/download_mimo_rl0530_mtp.sh" >&2
    exit 2
  fi
done
python -m remtp.mimo_checkpoint check "$MTP_MODEL_PATH"
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

write_tree_metrics() {
  local run_dir="$1"
  if [[ ! -s "$run_dir/tree_rounds.jsonl" ]]; then
    echo "Missing tree audit: $run_dir/tree_rounds.jsonl" >&2
    exit 2
  fi
  python -m remtp.tree_benchmark_metrics \
    --run-dir "$run_dir" --audit "$run_dir/tree_rounds.jsonl"
}

start_server() {
  local method="$1"
  local log_file="$2"
  local audit_file="$3"
  local tree_dir="$4"
  local route=""
  case "$method" in
    target_only)
      setsid env PORT="$PORT" MODEL_PATH="$TARGET_MODEL_PATH" \
        SERVED_MODEL_NAME="$MODEL_NAME" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        "$PROJECT_DIR/scripts/serve_mimo_target.sh" >"$log_file" 2>&1 &
      ;;
    native_012|native_000)
      if [[ "$method" == "native_012" ]]; then route=physical; else route=layer0; fi
      setsid env PORT="$PORT" MODEL_PATH="$MTP_MODEL_PATH" \
        SERVED_MODEL_NAME="$MODEL_NAME" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MTP_TOKENS="$MTP_TOKENS" MIMO_MTP_LAYER_MODE="$route" \
        ENFORCE_EAGER=1 NO_ASYNC_SCHEDULING=1 \
        REMTP_MIMO_PREFILL_ALL_LAYERS=1 \
        "$PROJECT_DIR/scripts/serve_mimo_native.sh" >"$log_file" 2>&1 &
      ;;
    tree_012|tree_000)
      if [[ "$method" == "tree_012" ]]; then
        route=physical_012
      else
        route=repeat_000
      fi
      mkdir -p "$tree_dir"
      : > "$audit_file"
      setsid env PORT="$PORT" MODEL_PATH="$MTP_MODEL_PATH" \
        SERVED_MODEL_NAME="$MODEL_NAME" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        TREE_RUN_DIR="$tree_dir" REMTP_TREE_AUDIT_JSONL="$audit_file" \
        REMTP_TREE_AUDIT_DETAIL=0 REMTP_TREE_TRACE=0 \
        REMTP_MIMO_PREFILL_ALL_LAYERS=1 \
        REMTP_DYNAMIC_TREE_MTP_ROUTE="$route" \
        REMTP_DYNAMIC_TREE_MAX_DEPTH="$DYNAMIC_MAX_DEPTH" \
        REMTP_DYNAMIC_TREE_MAX_NODES="$DYNAMIC_MAX_NODES" \
        REMTP_DYNAMIC_TREE_TAU_MIN="$DYNAMIC_TAU_MIN" \
        REMTP_DYNAMIC_TREE_KAPPA="$DYNAMIC_KAPPA" \
        REMTP_DYNAMIC_TREE_MU="$DYNAMIC_MU" \
        REMTP_DYNAMIC_TREE_ETA="$DYNAMIC_ETA" \
        REMTP_DYNAMIC_TREE_ALPHA="$DYNAMIC_ALPHA" \
        REMTP_DYNAMIC_TREE_TAU_RELAX="$DYNAMIC_TAU_RELAX" \
        REMTP_DYNAMIC_TREE_BETA="$DYNAMIC_BETA" \
        REMTP_DYNAMIC_TREE_PATH_TEMPERATURE="$DYNAMIC_PATH_TEMPERATURE" \
        REMTP_DYNAMIC_TREE_EOS_THRESHOLD="$EOS_THRESHOLD" \
        "$PROJECT_DIR/scripts/serve_mimo_dynamic_tree.sh" >"$log_file" 2>&1 &
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
  local audit_active="$LOG_ROOT/${method}_active_rounds.jsonl"
  local tree_dir="$LOG_ROOT/${method}_tree_server"
  local mtp_value="$MTP_TOKENS"
  local need_gsm=1 need_human=1
  [[ "$method" == "target_only" ]] && mtp_value=0
  refuse_partial "$gsm_dir"
  refuse_partial "$human_dir"
  [[ -f "$gsm_dir/summary.json" ]] && need_gsm=0
  [[ -f "$human_dir/summary.json" ]] && need_human=0
  if [[ "$method" == tree_* ]]; then
    if (( need_gsm == 0 )) && [[ ! -f "$gsm_dir/tree_metrics.json" ]]; then
      write_tree_metrics "$gsm_dir"
    fi
    if (( need_human == 0 )) && [[ ! -f "$human_dir/tree_metrics.json" ]]; then
      write_tree_metrics "$human_dir"
    fi
  fi

  if (( need_gsm == 1 || need_human == 1 )); then
    echo "===== $method ====="
    start_server "$method" "$server_log" "$audit_active" "$tree_dir"
    MODEL="$MODEL_NAME" BASE_URL="$BASE_URL" TEMPERATURE="$TEMPERATURE" \
      MAX_TOKENS=32 SEED="$SEED" "$PROJECT_DIR/scripts/request_mimo.sh" >/dev/null
    [[ "$method" == tree_* ]] && : > "$audit_active"

    if (( need_gsm == 1 )); then
      python -m remtp.gsm8k_benchmark \
        --data "$GSM8K_DATA" --samples "$SAMPLES" \
        --sample-seed "$GSM8K_SAMPLE_SEED" --temperature "$TEMPERATURE" \
        --generation-seed "$SEED" --max-tokens "$GSM8K_MAX_TOKENS" \
        --mtp-tokens "$mtp_value" --base-url "$BASE_URL" --model "$MODEL_NAME" \
        --run-name "$method" --output-dir "$gsm_dir" \
        --progress-every "$PROGRESS_EVERY" --skip-warmup --empty-system-prompt \
        2>&1 | tee "$LOG_ROOT/${method}_gsm8k.log"
      if [[ "$method" == tree_* ]]; then
        cp "$audit_active" "$gsm_dir/tree_rounds.jsonl"
        write_tree_metrics "$gsm_dir"
        : > "$audit_active"
      fi
    elif [[ "$method" == tree_* ]]; then
      : > "$audit_active"
    fi

    if (( need_human == 1 )); then
      python -m remtp.humaneval_benchmark \
        --data "$HUMANEVAL_DATA" --samples "$SAMPLES" \
        --sample-seed "$HUMANEVAL_SAMPLE_SEED" --temperature "$TEMPERATURE" \
        --generation-seed "$SEED" --max-tokens "$HUMANEVAL_MAX_TOKENS" \
        --mtp-tokens "$mtp_value" --base-url "$BASE_URL" --model "$MODEL_NAME" \
        --run-name "$method" --output-dir "$human_dir" \
        --progress-every "$PROGRESS_EVERY" --skip-warmup --empty-system-prompt \
        2>&1 | tee "$LOG_ROOT/${method}_humaneval.log"
      if [[ "$method" == tree_* ]]; then
        cp "$audit_active" "$human_dir/tree_rounds.jsonl"
        write_tree_metrics "$human_dir"
      fi
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
MiMo RL-0530 route comparison
  samples          : $SAMPLES per dataset
  system prompt    : explicit empty string
  temperature      : $TEMPERATURE
  max tokens       : GSM8K=$GSM8K_MAX_TOKENS HumanEval=$HUMANEVAL_MAX_TOKENS
  MTP route tests  : physical_012, repeat_000
  EOS protection   : target top-1 EOS and P(EOS) >= $EOS_THRESHOLD
  output           : $RUN_ROOT
EOF

for method in "${METHODS[@]}"; do
  run_method "$method"
done

python -m remtp.mimo_rl_route_report --run-root "$RUN_ROOT"
echo
echo "Complete: $RUN_ROOT/comparison.md"
sed -n '1,200p' "$RUN_ROOT/comparison.md"
