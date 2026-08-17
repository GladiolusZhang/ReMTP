#!/usr/bin/env bash
# Verified FastMTP comparison: target control + four requested methods.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

usage() {
  cat <<'EOF'
Audited comparison on the same TencentBAC/FastMTP checkpoint:

  target         optional target-only local control (INCLUDE_TARGET=1)
  native         strict probabilistic FastMTP
  cactus         Cactus verifier over the same full MTP Q
  spec_cascade   SpecCascade TokenV3 over the same full MTP Q
  dynamic_tree   target-dominant relaxed dynamic tree (approximate)

Quick pilot:
  SAMPLES=10 RUN_TAG=fastmtp_verified_pilot \
    ./scripts/run_fastmtp_verified_comparison.sh

Larger run:
  SAMPLES=50 RUN_TAG=fastmtp_verified_n50 \
    ./scripts/run_fastmtp_verified_comparison.sh

Disable MiMo visible reasoning with the compatible template:
  FAST_MTP_NO_THINK=1 SAMPLES=10 RUN_TAG=fastmtp_no_think_pilot \
    ./scripts/run_fastmtp_verified_comparison.sh

The script refuses to reuse port 8000, verifies the active Worker from its
startup marker, uses temperature 0.6 and an explicit empty system message,
evaluates HumanEval in Docker, and writes one comparison.md.
Set INCLUDE_TARGET=0 to skip the slow target-only control. Set
PROGRESS_EVERY=1 to print every generated/evaluated sample immediately.
Set METHODS_CSV=dynamic_tree to run only a selected comma-separated subset;
this is useful when unchanged baseline directories are linked into RUN_ROOT.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if (( $# > 0 )); then usage >&2; exit 2; fi

SAMPLES="${SAMPLES:-20}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-$SAMPLES}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-$SAMPLES}"
GSM8K_SAMPLE_SEED="${GSM8K_SAMPLE_SEED:-20260730}"
HUMANEVAL_SAMPLE_SEED="${HUMANEVAL_SAMPLE_SEED:-20260802}"
TEMPERATURE="${TEMPERATURE:-0.6}"
SEED="${SEED:-42}"
GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-2048}"
HUMANEVAL_MAX_TOKENS="${HUMANEVAL_MAX_TOKENS:-2048}"
MTP_TOKENS="${MTP_TOKENS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-360}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
HUMANEVAL_IMAGE="${HUMANEVAL_IMAGE:-python:3-slim}"
HUMANEVAL_EVAL_TIMEOUT="${HUMANEVAL_EVAL_TIMEOUT:-8}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/FastMTP}"
MODEL_NAME="${MODEL_NAME:-TencentBAC/FastMTP}"
GSM8K_DATA="${GSM8K_DATA:-$PROJECT_DIR/data/gsm8k/test.jsonl}"
HUMANEVAL_DATA="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
RUN_TAG="${RUN_TAG:-fastmtp_verified_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"
LOG_ROOT="$PROJECT_DIR/logs/$RUN_TAG"
INCLUDE_TARGET="${INCLUDE_TARGET:-0}"
RESUME_PARTIAL="${RESUME_PARTIAL:-0}"
METHODS_CSV="${METHODS_CSV:-}"
if [[ -n "$METHODS_CSV" ]]; then
  IFS=',' read -r -a METHODS <<< "$METHODS_CSV"
  declare -A seen_methods=()
  for method in "${METHODS[@]}"; do
    case "$method" in
      target|native|cactus|spec_cascade|dynamic_tree) ;;
      *) echo "Unknown method in METHODS_CSV: $method" >&2; exit 2 ;;
    esac
    if [[ -n "${seen_methods[$method]:-}" ]]; then
      echo "Duplicate method in METHODS_CSV: $method" >&2
      exit 2
    fi
    seen_methods[$method]=1
  done
elif [[ "$INCLUDE_TARGET" == "1" ]]; then
  METHODS=(target native cactus spec_cascade dynamic_tree)
elif [[ "$INCLUDE_TARGET" == "0" ]]; then
  METHODS=(native cactus spec_cascade dynamic_tree)
else
  echo "INCLUDE_TARGET must be 0 or 1; received $INCLUDE_TARGET" >&2
  exit 2
fi
if [[ "$RESUME_PARTIAL" != "0" && "$RESUME_PARTIAL" != "1" ]]; then
  echo "RESUME_PARTIAL must be 0 or 1; received $RESUME_PARTIAL" >&2
  exit 2
fi

CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
CASCADE_RULE="${CASCADE_RULE:-token_v3}"
CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"
DYNAMIC_MAX_DEPTH="${DYNAMIC_MAX_DEPTH:-3}"
DYNAMIC_MAX_NODES="${DYNAMIC_MAX_NODES:-6}"
DYNAMIC_ADAPTIVE_BASE_NODES="${DYNAMIC_ADAPTIVE_BASE_NODES:-0}"
DYNAMIC_ADAPTIVE_ENTROPY_THRESHOLD="${DYNAMIC_ADAPTIVE_ENTROPY_THRESHOLD:-1.0}"
DYNAMIC_MAX_CHILDREN="${DYNAMIC_MAX_CHILDREN:-2}"
DYNAMIC_MIN_SIBLING_RATIO="${DYNAMIC_MIN_SIBLING_RATIO:-0.25}"
DYNAMIC_TAU_MIN="${DYNAMIC_TAU_MIN:-0.02}"
DYNAMIC_KAPPA="${DYNAMIC_KAPPA:-1.0}"
DYNAMIC_MU="${DYNAMIC_MU:-0.5}"
DYNAMIC_ETA="${DYNAMIC_ETA:-0.25}"
DYNAMIC_ALLOCATION="${DYNAMIC_ALLOCATION:-geometric}"
DYNAMIC_RANK_PENALTY="${DYNAMIC_RANK_PENALTY:-2.0}"
DYNAMIC_COVERAGE_MODE="${DYNAMIC_COVERAGE_MODE:-coverage_gate}"
DYNAMIC_MIN_COVERAGE="${DYNAMIC_MIN_COVERAGE:-0.05}"
DYNAMIC_ALPHA="${DYNAMIC_ALPHA:-0.5}"
DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.50}"
DYNAMIC_SUPPORT_MODE="${DYNAMIC_SUPPORT_MODE:-relative}"
DYNAMIC_PROPOSAL_SUPPORT_RATIO="${DYNAMIC_PROPOSAL_SUPPORT_RATIO:-1.0}"
DYNAMIC_CONFIRMATION_MIN_RELATIVE="${DYNAMIC_CONFIRMATION_MIN_RELATIVE:-0.05}"
DYNAMIC_CACTUS_DELTA="${DYNAMIC_CACTUS_DELTA:-$CACTUS_DELTA}"
DYNAMIC_CACTUS_TARGET_WEIGHT="${DYNAMIC_CACTUS_TARGET_WEIGHT:-0.25}"
DYNAMIC_MAX_GUIDED_RESCUES="${DYNAMIC_MAX_GUIDED_RESCUES:-1}"
DYNAMIC_RESCUE_SCORE_THRESHOLD="${DYNAMIC_RESCUE_SCORE_THRESHOLD:-0.35}"
DYNAMIC_RESCUE_DEPTH_PENALTY="${DYNAMIC_RESCUE_DEPTH_PENALTY:-0.0}"
DYNAMIC_RESCUE_CONTINUATION_DISCOUNT="${DYNAMIC_RESCUE_CONTINUATION_DISCOUNT:-0.0}"
DYNAMIC_RESCUE_CONTINUATION_MIN_DEPTH="${DYNAMIC_RESCUE_CONTINUATION_MIN_DEPTH:-2}"
DYNAMIC_RESCUE_MARGIN_REFERENCE="${DYNAMIC_RESCUE_MARGIN_REFERENCE:-0.0}"
DYNAMIC_RESCUE_MARGIN_PENALTY="${DYNAMIC_RESCUE_MARGIN_PENALTY:-0.0}"
DYNAMIC_BETA="${DYNAMIC_BETA:-0.25}"
DYNAMIC_PATH_TEMPERATURE="${DYNAMIC_PATH_TEMPERATURE:-0.10}"
DYNAMIC_PATH_SELECTION="${DYNAMIC_PATH_SELECTION:-longest}"
DYNAMIC_FRONTIER_RESCUE="${DYNAMIC_FRONTIER_RESCUE:-1}"
DYNAMIC_RESCUE_DELTA="${DYNAMIC_RESCUE_DELTA:-0.5}"
DYNAMIC_RESCUE_MIN_RELATIVE="${DYNAMIC_RESCUE_MIN_RELATIVE:-0.1}"
DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.001}"
EOS_THRESHOLD="${EOS_THRESHOLD:-0.5}"
FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-0}"

if [[ "$TEMPERATURE" != "0.6" ]]; then
  echo "The verified quality protocol fixes TEMPERATURE=0.6; received $TEMPERATURE." >&2
  exit 2
fi
for path in "$MODEL_PATH/config.json" "$GSM8K_DATA" "$HUMANEVAL_DATA"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 2; }
done
python -m remtp.fastmtp_checkpoint check "$MODEL_PATH"
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable; HumanEval cannot be evaluated safely." >&2
  exit 2
fi
if ! docker image inspect "$HUMANEVAL_IMAGE" >/dev/null 2>&1; then
  echo "Missing Docker image; run: docker pull $HUMANEVAL_IMAGE" >&2
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
  local log_file="$1" deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then return; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before becoming healthy:" >&2
      tail -n 180 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "Server startup timed out:" >&2
  tail -n 180 "$log_file" >&2
  return 1
}

humaneval_complete() {
  [[ -f "$1/summary.json" ]] || return 1
  python - "$1/summary.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8")).get("results") or []
raise SystemExit(0 if len(rows) == 1 and rows[0].get("evaluation_status") == "complete" else 1)
PY
}

refuse_partial() {
  if [[ -d "$1" && ! -f "$1/summary.json" ]]; then
    if [[ "$RESUME_PARTIAL" == "1" ]]; then
      echo "[resume] partial output accepted: $1"
      return
    fi
    echo "Partial output directory found: $1" >&2
    echo "Set RESUME_PARTIAL=1, move it aside, or select another RUN_TAG." >&2
    exit 2
  fi
}

prepare_partial_tree_audit() {
  local run_dir="$1" audit_active="$2" stamp destination
  [[ -d "$run_dir" && ! -f "$run_dir/summary.json" ]] || return
  [[ -s "$audit_active" ]] || return
  if [[ -s "$run_dir/requests.checkpoint.jsonl" ]]; then
    cat "$audit_active" >> "$run_dir/tree_rounds.checkpoint.jsonl"
    echo "[resume] preserved tree audit for checkpointed requests: $run_dir"
  else
    stamp="$(date +%Y%m%d_%H%M%S)"
    destination="$run_dir/tree_rounds.unrecoverable_${stamp}.jsonl"
    mv "$audit_active" "$destination"
    echo "[resume] archived uncheckpointed tree audit: $destination"
  fi
  : > "$audit_active"
}

finalize_tree_audit() {
  local run_dir="$1" audit_active="$2"
  if [[ -s "$run_dir/tree_rounds.checkpoint.jsonl" ]]; then
    cat "$run_dir/tree_rounds.checkpoint.jsonl" "$audit_active" \
      > "$run_dir/tree_rounds.jsonl"
  else
    cp "$audit_active" "$run_dir/tree_rounds.jsonl"
  fi
}

write_tree_metrics() {
  local run_dir="$1"
  [[ -s "$run_dir/tree_rounds.jsonl" ]] || {
    echo "Missing tree audit: $run_dir/tree_rounds.jsonl" >&2; exit 2;
  }
  python -m remtp.tree_benchmark_metrics \
    --run-dir "$run_dir" --audit "$run_dir/tree_rounds.jsonl"
}

expected_marker() {
  case "$1" in
    target) echo "[ReMTP][FastMTPVerified] method=target speculative_decoding=off" ;;
    *) echo "[ReMTP][FastMTPVerified] method=$1 worker=active" ;;
  esac
}

expected_adapter_marker() {
  case "$1" in
    target) echo "[ReMTP][FastMTPVerified] method=target speculative_decoding=off" ;;
    native) echo "[ReMTP][ProbMTP][diagnostic]" ;;
    cactus) echo "[ReMTP][Cactus][diagnostic]" ;;
    spec_cascade) echo "[ReMTP][SpecCascade][diagnostic]" ;;
    dynamic_tree) echo "[ReMTP][DynamicTree] installed" ;;
  esac
}

verify_adapter() {
  local method="$1" log_file="$2" marker
  marker="$(expected_adapter_marker "$method")"
  grep -Fq "$marker" "$log_file" || {
    echo "The expected runtime adapter did not execute for $method: $marker" >&2
    tail -n 180 "$log_file" >&2
    exit 2
  }
}

write_runtime() {
  local method="$1" destination="$2" log_file="$3" marker adapter_marker
  marker="$(expected_marker "$method")"
  adapter_marker="$(expected_adapter_marker "$method")"
  grep -Fq "$marker" "$log_file" || {
    echo "Required runtime marker missing for $method: $marker" >&2
    tail -n 180 "$log_file" >&2
    exit 2
  }
  grep -Fq "$adapter_marker" "$log_file" || {
    echo "Required adapter marker missing for $method: $adapter_marker" >&2
    exit 2
  }
  python - "$method" "$marker" "$adapter_marker" "$log_file" "$destination" <<'PY'
import json, sys
from pathlib import Path
method, marker, adapter_marker, log_file, destination = sys.argv[1:]
Path(destination).write_text(json.dumps({
    "method": method,
    "marker": marker,
    "marker_verified": True,
    "adapter_marker": adapter_marker,
    "adapter_verified": True,
    "server_log": str(Path(log_file).resolve()),
}, indent=2) + "\n", encoding="utf-8")
PY
}

start_server() {
  local method="$1" log_file="$2" audit_file="$3"
  if [[ "$RESUME_PARTIAL" == "1" && -s "$log_file" ]]; then
    mv "$log_file" "${log_file%.log}.interrupted_$(date +%Y%m%d_%H%M%S).log"
  fi
  : > "$log_file"
  echo "[$method] starting vLLM; startup log: $log_file"
  local -a environment=(
    METHOD="$method" PORT="$PORT" MODEL_PATH="$MODEL_PATH"
    SERVED_MODEL_NAME="$MODEL_NAME" MAX_MODEL_LEN="$MAX_MODEL_LEN"
    FAST_MTP_NO_THINK="$FAST_MTP_NO_THINK"
    MTP_TOKENS="$MTP_TOKENS" ENFORCE_EAGER=1 NO_ASYNC_SCHEDULING=1
    REMTP_CACTUS_DELTA="$CACTUS_DELTA" REMTP_CACTUS_DIAGNOSTICS=1
    REMTP_CASCADE_RULE="$CASCADE_RULE" REMTP_CASCADE_ALPHA="$CASCADE_ALPHA"
    REMTP_CASCADE_DIAGNOSTICS=1
    REMTP_TREE_AUDIT_JSONL="$audit_file" REMTP_TREE_AUDIT_DETAIL=0
    REMTP_TREE_TRACE=0 REMTP_DYNAMIC_TREE_MAX_DEPTH="$DYNAMIC_MAX_DEPTH"
    REMTP_DYNAMIC_TREE_MAX_NODES="$DYNAMIC_MAX_NODES"
    REMTP_DYNAMIC_TREE_ADAPTIVE_BASE_NODES="$DYNAMIC_ADAPTIVE_BASE_NODES"
    REMTP_DYNAMIC_TREE_ADAPTIVE_ENTROPY_THRESHOLD="$DYNAMIC_ADAPTIVE_ENTROPY_THRESHOLD"
    REMTP_DYNAMIC_TREE_MAX_CHILDREN="$DYNAMIC_MAX_CHILDREN"
    REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO="$DYNAMIC_MIN_SIBLING_RATIO"
    REMTP_DYNAMIC_TREE_TAU_MIN="$DYNAMIC_TAU_MIN"
    REMTP_DYNAMIC_TREE_KAPPA="$DYNAMIC_KAPPA"
    REMTP_DYNAMIC_TREE_MU="$DYNAMIC_MU" REMTP_DYNAMIC_TREE_ETA="$DYNAMIC_ETA"
    REMTP_DYNAMIC_TREE_ALLOCATION="$DYNAMIC_ALLOCATION"
    REMTP_DYNAMIC_TREE_RANK_PENALTY="$DYNAMIC_RANK_PENALTY"
    REMTP_DYNAMIC_TREE_COVERAGE_MODE="$DYNAMIC_COVERAGE_MODE"
    REMTP_DYNAMIC_TREE_MIN_TARGET_COVERAGE="$DYNAMIC_MIN_COVERAGE"
    REMTP_DYNAMIC_TREE_ALPHA="$DYNAMIC_ALPHA"
    REMTP_DYNAMIC_TREE_TAU_RELAX="$DYNAMIC_TAU_RELAX"
    REMTP_DYNAMIC_TREE_SUPPORT_MODE="$DYNAMIC_SUPPORT_MODE"
    REMTP_DYNAMIC_TREE_PROPOSAL_SUPPORT_RATIO="$DYNAMIC_PROPOSAL_SUPPORT_RATIO"
    REMTP_DYNAMIC_TREE_CONFIRMATION_MIN_RELATIVE="$DYNAMIC_CONFIRMATION_MIN_RELATIVE"
    REMTP_DYNAMIC_TREE_CACTUS_DELTA="$DYNAMIC_CACTUS_DELTA"
    REMTP_DYNAMIC_TREE_CACTUS_TARGET_WEIGHT="$DYNAMIC_CACTUS_TARGET_WEIGHT"
    REMTP_DYNAMIC_TREE_MAX_GUIDED_RESCUES="$DYNAMIC_MAX_GUIDED_RESCUES"
    REMTP_DYNAMIC_TREE_RESCUE_SCORE_THRESHOLD="$DYNAMIC_RESCUE_SCORE_THRESHOLD"
    REMTP_DYNAMIC_TREE_RESCUE_DEPTH_PENALTY="$DYNAMIC_RESCUE_DEPTH_PENALTY"
    REMTP_DYNAMIC_TREE_RESCUE_CONTINUATION_DISCOUNT="$DYNAMIC_RESCUE_CONTINUATION_DISCOUNT"
    REMTP_DYNAMIC_TREE_RESCUE_CONTINUATION_MIN_DEPTH="$DYNAMIC_RESCUE_CONTINUATION_MIN_DEPTH"
    REMTP_DYNAMIC_TREE_RESCUE_MARGIN_REFERENCE="$DYNAMIC_RESCUE_MARGIN_REFERENCE"
    REMTP_DYNAMIC_TREE_RESCUE_MARGIN_PENALTY="$DYNAMIC_RESCUE_MARGIN_PENALTY"
    REMTP_DYNAMIC_TREE_BETA="$DYNAMIC_BETA"
    REMTP_DYNAMIC_TREE_PATH_TEMPERATURE="$DYNAMIC_PATH_TEMPERATURE"
    REMTP_DYNAMIC_TREE_PATH_SELECTION="$DYNAMIC_PATH_SELECTION"
    REMTP_DYNAMIC_TREE_FRONTIER_RESCUE="$DYNAMIC_FRONTIER_RESCUE"
    REMTP_DYNAMIC_TREE_RESCUE_DELTA="$DYNAMIC_RESCUE_DELTA"
    REMTP_DYNAMIC_TREE_RESCUE_MIN_RELATIVE="$DYNAMIC_RESCUE_MIN_RELATIVE"
    REMTP_DYNAMIC_TREE_RESCUE_MIN_TARGET_PROB="$DYNAMIC_RESCUE_MIN_TARGET_PROB"
    REMTP_DYNAMIC_TREE_EOS_THRESHOLD="$EOS_THRESHOLD"
  )
  setsid env "${environment[@]}" \
    "$PROJECT_DIR/scripts/serve_fastmtp_verified.sh" >"$log_file" 2>&1 &
  server_pid=$!
  wait_for_server "$log_file"
  echo "[$method] server is healthy"
  local marker
  marker="$(expected_marker "$method")"
  grep -Fq "$marker" "$log_file" || {
    echo "Wrong Worker was loaded for $method." >&2
    tail -n 180 "$log_file" >&2
    exit 2
  }
}

run_method() {
  local method="$1"
  local gsm_dir="$RUN_ROOT/gsm8k/$method"
  local human_dir="$RUN_ROOT/humaneval/$method"
  local server_log="$LOG_ROOT/${method}_server.log"
  local audit_active="$LOG_ROOT/${method}_active_rounds.jsonl"
  local need_gsm=1 need_human=1 mtp_value="$MTP_TOKENS"
  local -a resume_args=()
  [[ "$RESUME_PARTIAL" == "1" ]] && resume_args=(--resume)
  [[ "$method" == target ]] && mtp_value=0
  [[ "$method" == dynamic_tree ]] && mtp_value="$DYNAMIC_MAX_NODES"
  refuse_partial "$gsm_dir"
  refuse_partial "$human_dir"
  [[ -f "$gsm_dir/summary.json" ]] && need_gsm=0
  [[ -f "$human_dir/summary.json" ]] && need_human=0

  if [[ "$method" == dynamic_tree ]]; then
    if (( need_gsm == 0 )) && [[ ! -f "$gsm_dir/tree_metrics.json" ]]; then
      write_tree_metrics "$gsm_dir"
    fi
    if (( need_human == 0 )) && [[ ! -f "$human_dir/tree_metrics.json" ]]; then
      write_tree_metrics "$human_dir"
    fi
  fi

  if (( need_gsm == 1 || need_human == 1 )); then
    echo "===== $method ====="
    if [[ "$method" == dynamic_tree && "$RESUME_PARTIAL" == "1" ]]; then
      if (( need_human == 1 )) && [[ -d "$human_dir" ]]; then
        prepare_partial_tree_audit "$human_dir" "$audit_active"
      elif (( need_gsm == 1 )) && [[ -d "$gsm_dir" ]]; then
        prepare_partial_tree_audit "$gsm_dir" "$audit_active"
      fi
    fi
    : > "$audit_active"
    start_server "$method" "$server_log" "$audit_active"
    MODEL="$MODEL_NAME" BASE_URL="$BASE_URL" TEMPERATURE="$TEMPERATURE" \
      MAX_TOKENS=32 SEED="$SEED" "$PROJECT_DIR/scripts/request_mimo.sh" >/dev/null
    verify_adapter "$method" "$server_log"
    [[ "$method" == dynamic_tree ]] && : > "$audit_active"

    if (( need_gsm == 1 )); then
      echo "[$method][GSM8K] generating $GSM8K_SAMPLES samples (live progress below)"
      python -m remtp.gsm8k_benchmark \
        --data "$GSM8K_DATA" --samples "$GSM8K_SAMPLES" \
        --sample-seed "$GSM8K_SAMPLE_SEED" --temperature "$TEMPERATURE" \
        --generation-seed "$SEED" --max-tokens "$GSM8K_MAX_TOKENS" \
        --mtp-tokens "$mtp_value" --base-url "$BASE_URL" --model "$MODEL_NAME" \
        --run-name "$method" --output-dir "$gsm_dir" \
        --progress-every "$PROGRESS_EVERY" --skip-warmup --empty-system-prompt \
        "${resume_args[@]}" \
        2>&1 | tee -a "$LOG_ROOT/${method}_gsm8k.log"
      write_runtime "$method" "$gsm_dir/method_runtime.json" "$server_log"
      if [[ "$method" == dynamic_tree ]]; then
        finalize_tree_audit "$gsm_dir" "$audit_active"
        write_tree_metrics "$gsm_dir"
        : > "$audit_active"
      fi
    fi

    if (( need_human == 1 )); then
      echo "[$method][HumanEval] generating $HUMANEVAL_SAMPLES samples (live progress below)"
      python -m remtp.humaneval_benchmark \
        --data "$HUMANEVAL_DATA" --samples "$HUMANEVAL_SAMPLES" \
        --sample-seed "$HUMANEVAL_SAMPLE_SEED" --temperature "$TEMPERATURE" \
        --generation-seed "$SEED" --max-tokens "$HUMANEVAL_MAX_TOKENS" \
        --mtp-tokens "$mtp_value" --base-url "$BASE_URL" --model "$MODEL_NAME" \
        --run-name "$method" --output-dir "$human_dir" \
        --progress-every "$PROGRESS_EVERY" --skip-warmup --empty-system-prompt \
        "${resume_args[@]}" \
        2>&1 | tee -a "$LOG_ROOT/${method}_humaneval.log"
      write_runtime "$method" "$human_dir/method_runtime.json" "$server_log"
      if [[ "$method" == dynamic_tree ]]; then
        finalize_tree_audit "$human_dir" "$audit_active"
        write_tree_metrics "$human_dir"
      fi
    fi
    cleanup_server
  else
    echo "[resume] generation complete: $method"
  fi

  if ! humaneval_complete "$human_dir"; then
    echo "[$method][HumanEval] running isolated tests (live progress below)"
    python -m remtp.humaneval_evaluator "$human_dir" \
      --data "$HUMANEVAL_DATA" --image "$HUMANEVAL_IMAGE" \
      --timeout "$HUMANEVAL_EVAL_TIMEOUT" --workers "$EVAL_WORKERS" \
      --progress-every "$PROGRESS_EVERY" \
      2>&1 | tee "$LOG_ROOT/${method}_humaneval_eval.log"
  fi
}

cat <<EOF
Verified FastMTP comparison
  samples       : GSM8K=$GSM8K_SAMPLES HumanEval=$HUMANEVAL_SAMPLES
  temperature   : $TEMPERATURE
  system prompt : explicit empty string
  no thinking   : $FAST_MTP_NO_THINK
  include target: $INCLUDE_TARGET
  methods       : ${METHODS[*]}
  live progress : every $PROGRESS_EVERY sample(s)
  resume partial: $RESUME_PARTIAL
  target/MTP    : same TencentBAC/FastMTP checkpoint; physical MTP heads=1
  tree          : D=$DYNAMIC_MAX_DEPTH N_max=$DYNAMIC_MAX_NODES adaptive_base=$DYNAMIC_ADAPTIVE_BASE_NODES entropy_threshold=$DYNAMIC_ADAPTIVE_ENTROPY_THRESHOLD children_max=$DYNAMIC_MAX_CHILDREN sibling_ratio=$DYNAMIC_MIN_SIBLING_RATIO coverage=$DYNAMIC_COVERAGE_MODE
  allocation    : $DYNAMIC_ALLOCATION rank_penalty=$DYNAMIC_RANK_PENALTY
  support       : $DYNAMIC_SUPPORT_MODE relative_tau=$DYNAMIC_TAU_RELAX proposal_ratio=$DYNAMIC_PROPOSAL_SUPPORT_RATIO confirmation_min=$DYNAMIC_CONFIRMATION_MIN_RELATIVE max_guided_rescues=$DYNAMIC_MAX_GUIDED_RESCUES rescue_score_tau=$DYNAMIC_RESCUE_SCORE_THRESHOLD rescue_depth_penalty=$DYNAMIC_RESCUE_DEPTH_PENALTY continuation_discount=$DYNAMIC_RESCUE_CONTINUATION_DISCOUNT continuation_min_depth=$DYNAMIC_RESCUE_CONTINUATION_MIN_DEPTH rescue_margin_ref=$DYNAMIC_RESCUE_MARGIN_REFERENCE rescue_margin_penalty=$DYNAMIC_RESCUE_MARGIN_PENALTY
  path selector : $DYNAMIC_PATH_SELECTION beta=$DYNAMIC_BETA temperature=$DYNAMIC_PATH_TEMPERATURE
  output        : $RUN_ROOT
  live logs     : $LOG_ROOT
EOF

for method in "${METHODS[@]}"; do run_method "$method"; done
python -m remtp.fastmtp_verified_report --run-root "$RUN_ROOT"
echo "Complete: $RUN_ROOT/comparison.md"
sed -n '1,220p' "$RUN_ROOT/comparison.md"
