#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Run the MiMo algorithm-correctness comparison on the same 100 GSM8K and
100 HumanEval tasks:

  1. Native probabilistic MTP
  2. Cactus + MTP
  3. SpecCascade TokenV3 + MTP
  4. Dynamic MTP Tree + target-dominant relaxation

The primary comparison contains task quality and mean acceptance length.
Throughput is still retained in each raw summary.json, but is deliberately
excluded from the final decision table.

Usage:
  ./scripts/run_mimo_correctness_100.sh

Useful overrides:
  RUN_TAG=mimo_correctness_100_my_run
  TEMPERATURE=0.7 SEED=42
  GSM8K_MAX_TOKENS=384 HUMANEVAL_MAX_TOKENS=512
  EVAL_WORKERS=4 PORT=8000

Resume:
  Rerun with the same RUN_TAG. Completed generations and HumanEval evaluations
  are skipped. An output directory that exists without summary.json is treated
  as an interrupted write and is never overwritten automatically.
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
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-384}"
HUMANEVAL_MAX_TOKENS="${HUMANEVAL_MAX_TOKENS:-512}"
MTP_TOKENS="${MTP_TOKENS:-3}"
CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
CASCADE_RULE="${CASCADE_RULE:-token_v3}"
CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-300}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
HUMANEVAL_IMAGE="${HUMANEVAL_IMAGE:-python:3-slim}"
HUMANEVAL_EVAL_TIMEOUT="${HUMANEVAL_EVAL_TIMEOUT:-8}"
GSM8K_DATA="${GSM8K_DATA:-$PROJECT_DIR/data/gsm8k/test.jsonl}"
HUMANEVAL_DATA="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
RUN_TAG="${RUN_TAG:-mimo_correctness_100_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"
LOG_ROOT="$PROJECT_DIR/logs/$RUN_TAG"
MODEL_NAME="${MODEL_NAME:-XiaomiMiMo/MiMo-7B-Base}"
METHODS=(native cactus spec_cascade dynamic_tree)

# Freeze the current dynamic-tree method explicitly so an unrelated shell
# export cannot silently change the formal comparison.
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

if [[ "$SAMPLES" != "100" ]]; then
  echo "This formal script requires SAMPLES=100 (received $SAMPLES)." >&2
  exit 2
fi
if [[ "$MTP_TOKENS" != "3" ]]; then
  echo "MiMo-7B-Base-MTP3 requires MTP_TOKENS=3 (received $MTP_TOKENS)." >&2
  exit 2
fi
for data_file in "$GSM8K_DATA" "$HUMANEVAL_DATA"; do
  if [[ ! -f "$data_file" ]]; then
    echo "Missing benchmark data: $data_file" >&2
    exit 2
  fi
done
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is unavailable; HumanEval cannot be evaluated safely." >&2
  exit 2
fi
if ! docker image inspect "$HUMANEVAL_IMAGE" >/dev/null 2>&1; then
  echo "Missing HumanEval Docker image: $HUMANEVAL_IMAGE" >&2
  echo "Install it first with: docker pull $HUMANEVAL_IMAGE" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/gsm8k" "$RUN_ROOT/humaneval" "$LOG_ROOT"
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
      echo "Server exited before becoming healthy:" >&2
      tail -n 160 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "Server startup timed out after ${SERVER_START_TIMEOUT}s:" >&2
  tail -n 160 "$log_file" >&2
  return 1
}

has_summary() {
  [[ -f "$1/summary.json" ]]
}

humaneval_complete() {
  local run_dir="$1"
  [[ -f "$run_dir/summary.json" ]] || return 1
  python - "$run_dir/summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload.get("results") or []
raise SystemExit(0 if len(rows) == 1 and rows[0].get("evaluation_status") == "complete" else 1)
PY
}

refuse_partial_directory() {
  local run_dir="$1"
  if [[ -d "$run_dir" && ! -f "$run_dir/summary.json" ]]; then
    echo "Interrupted output directory found: $run_dir" >&2
    echo "Move it aside or choose a new RUN_TAG; it will not be overwritten." >&2
    exit 2
  fi
}

write_tree_metrics() {
  local run_dir="$1"
  local audit="$run_dir/tree_rounds.jsonl"
  if [[ ! -s "$audit" ]]; then
    echo "Dynamic-tree audit is missing or empty: $audit" >&2
    exit 2
  fi
  python -m remtp.tree_benchmark_metrics \
    --run-dir "$run_dir" --audit "$audit"
}

run_method() {
  local method="$1"
  local gsm_dir="$RUN_ROOT/gsm8k/$method"
  local human_dir="$RUN_ROOT/humaneval/$method"
  local server_log="$LOG_ROOT/${method}_server.log"
  local audit_active="$LOG_ROOT/${method}_active_rounds.jsonl"
  local tree_server_dir="$LOG_ROOT/${method}_tree_server"
  local need_gsm=1
  local need_human=1

  refuse_partial_directory "$gsm_dir"
  refuse_partial_directory "$human_dir"
  if has_summary "$gsm_dir"; then need_gsm=0; fi
  if has_summary "$human_dir"; then need_human=0; fi

  if [[ "$method" == "dynamic_tree" ]]; then
    if (( need_gsm == 0 )) && [[ ! -f "$gsm_dir/tree_metrics.json" ]]; then
      write_tree_metrics "$gsm_dir"
    fi
    if (( need_human == 0 )) && [[ ! -f "$human_dir/tree_metrics.json" ]]; then
      write_tree_metrics "$human_dir"
    fi
  fi

  if (( need_gsm == 1 || need_human == 1 )); then
    local serve_script="$PROJECT_DIR/scripts/serve_mimo_${method}.sh"
    echo
    echo "===== Starting $method ====="
    if [[ "$method" == "dynamic_tree" ]]; then
      mkdir -p "$tree_server_dir"
      : > "$audit_active"
      setsid env \
        PORT="$PORT" \
        TREE_RUN_DIR="$tree_server_dir" \
        REMTP_TREE_AUDIT_JSONL="$audit_active" \
        REMTP_TREE_AUDIT_DETAIL=0 \
        REMTP_TREE_TRACE=0 \
        REMTP_MIMO_PREFILL_ALL_LAYERS=1 \
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
        "$serve_script" >"$server_log" 2>&1 &
    else
      setsid env \
        PORT="$PORT" \
        MTP_TOKENS="$MTP_TOKENS" \
        MIMO_MTP_LAYER_MODE=physical \
        ENFORCE_EAGER=1 \
        NO_ASYNC_SCHEDULING=1 \
        REMTP_MIMO_PREFILL_ALL_LAYERS=1 \
        REMTP_CACTUS_DIAGNOSTICS=0 \
        REMTP_CACTUS_DELTA="$CACTUS_DELTA" \
        REMTP_CASCADE_RULE="$CASCADE_RULE" \
        REMTP_CASCADE_ALPHA="$CASCADE_ALPHA" \
        "$serve_script" >"$server_log" 2>&1 &
    fi
    server_pid=$!
    wait_for_server "$server_log"

    echo "Warmup for $method (excluded)..."
    BASE_URL="$BASE_URL" MAX_TOKENS=32 TEMPERATURE="$TEMPERATURE" SEED="$SEED" \
      "$PROJECT_DIR/scripts/request_mimo.sh" >/dev/null
    if [[ "$method" == "dynamic_tree" ]]; then
      : > "$audit_active"
    fi

    if (( need_gsm == 1 )); then
      echo "Running GSM8K 100: $method"
      python -m remtp.gsm8k_benchmark \
        --data "$GSM8K_DATA" \
        --samples "$SAMPLES" \
        --sample-seed "$GSM8K_SAMPLE_SEED" \
        --temperature "$TEMPERATURE" \
        --generation-seed "$SEED" \
        --max-tokens "$GSM8K_MAX_TOKENS" \
        --mtp-tokens "$MTP_TOKENS" \
        --base-url "$BASE_URL" \
        --model "$MODEL_NAME" \
        --run-name "$method" \
        --output-dir "$gsm_dir" \
        --progress-every "$PROGRESS_EVERY" \
        --skip-warmup \
        2>&1 | tee "$LOG_ROOT/${method}_gsm8k.log"
      if [[ "$method" == "dynamic_tree" ]]; then
        cp "$audit_active" "$gsm_dir/tree_rounds.jsonl"
        write_tree_metrics "$gsm_dir"
        : > "$audit_active"
      fi
    elif [[ "$method" == "dynamic_tree" ]]; then
      : > "$audit_active"
    fi

    if (( need_human == 1 )); then
      echo "Running HumanEval 100 generation: $method"
      python -m remtp.humaneval_benchmark \
        --data "$HUMANEVAL_DATA" \
        --samples "$SAMPLES" \
        --sample-seed "$HUMANEVAL_SAMPLE_SEED" \
        --temperature "$TEMPERATURE" \
        --generation-seed "$SEED" \
        --max-tokens "$HUMANEVAL_MAX_TOKENS" \
        --mtp-tokens "$MTP_TOKENS" \
        --base-url "$BASE_URL" \
        --model "$MODEL_NAME" \
        --run-name "$method" \
        --output-dir "$human_dir" \
        --progress-every "$PROGRESS_EVERY" \
        --skip-warmup \
        2>&1 | tee "$LOG_ROOT/${method}_humaneval.log"
      if [[ "$method" == "dynamic_tree" ]]; then
        cp "$audit_active" "$human_dir/tree_rounds.jsonl"
        write_tree_metrics "$human_dir"
      fi
    fi
    cleanup_server
  else
    echo "[resume] generation already complete: $method"
  fi

  if ! humaneval_complete "$human_dir"; then
    echo "Evaluating HumanEval in Docker: $method"
    python -m remtp.humaneval_evaluator "$human_dir" \
      --data "$HUMANEVAL_DATA" \
      --image "$HUMANEVAL_IMAGE" \
      --timeout "$HUMANEVAL_EVAL_TIMEOUT" \
      --workers "$EVAL_WORKERS" \
      --progress-every "$PROGRESS_EVERY" \
      2>&1 | tee "$LOG_ROOT/${method}_humaneval_eval.log"
  else
    echo "[resume] HumanEval quality already evaluated: $method"
  fi
}

cat <<EOF
MiMo correctness experiment
  samples/dataset : $SAMPLES
  temperature     : $TEMPERATURE
  generation seed : $SEED
  MTP depth       : $MTP_TOKENS
  Cactus delta    : $CACTUS_DELTA
  cascade         : $CASCADE_RULE (alpha=$CASCADE_ALPHA)
  dynamic tree    : D=$DYNAMIC_MAX_DEPTH N_max=$DYNAMIC_MAX_NODES tau_relax=$DYNAMIC_TAU_RELAX
  results         : $RUN_ROOT
EOF

for method in "${METHODS[@]}"; do
  run_method "$method"
done

python -m remtp.mimo_correctness_report --run-root "$RUN_ROOT"

echo
echo "Complete. Primary report:"
echo "  $RUN_ROOT/comparison.md"
echo
sed -n '1,160p' "$RUN_ROOT/comparison.md"
