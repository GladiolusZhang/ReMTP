#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

MODE="${1:-all}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/run_proposal_calibrated_mtp.sh [all|test|collect|search|pilot]

Recommended unattended run:
  GSM8K_SAMPLES=30 HUMANEVAL_SAMPLES=30 \
  ./scripts/run_proposal_calibrated_mtp.sh all

`all` runs distribution tests, compact native-Q trace collection, offline
head-wise search, and a clean online native-vs-calibrated pilot. It never runs
a formal 200/full benchmark.

To reuse a prior collection:
  RUN_ROOT=results/proposal_calibrated_mtp_<tag> \
  ./scripts/run_proposal_calibrated_mtp.sh search

EOF
  exit 0
fi
case "$MODE" in all|test|collect|search|pilot) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac

MTP_TOKENS="${MTP_TOKENS:-6}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-256}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-30}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-30}"
GSM8K_SAMPLE_SEED="${GSM8K_SAMPLE_SEED:-20260730}"
HUMANEVAL_SAMPLE_SEED="${HUMANEVAL_SAMPLE_SEED:-20260802}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-240}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/proposal_calibrated_mtp_${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/proposal_calibrated_mtp_${RUN_TAG}}"
CONFIG_PATH="${REMTP_PC_CONFIG:-$RUN_ROOT/best_static_config.json}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "Proposal-Calibrated MTP screening is frozen at MTP_TOKENS=6." >&2
  exit 2
fi
if [[ "$MODE" != "test" ]] && curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
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
  if kill -0 "$server_pid" 2>/dev/null; then kill -TERM -- "-$server_pid" 2>/dev/null || true; sleep 2; fi
  if kill -0 "$server_pid" 2>/dev/null; then kill -KILL -- "-$server_pid" 2>/dev/null || true; fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
trap cleanup_server EXIT
trap 'exit 130' INT TERM

wait_for_server() {
  local log_file="$1" deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -n 160 "$log_file" >&2; return 1; fi
    sleep 2
  done
  tail -n 160 "$log_file" >&2
  return 1
}

run_tests() {
  python -m unittest tests.test_proposal_calibration tests.test_probabilistic_mtp -v
}

benchmark_dataset() {
  local dataset="$1" samples="$2" sample_seed="$3" output_dir="$4" log_file="$5"
  if [[ "$dataset" == "gsm8k" ]]; then
    RUN_NAME=proposal_calibration \
    SAMPLES="$samples" SAMPLE_SEED="$sample_seed" TEMPERATURE="$TEMPERATURE" \
    SEED="$SEED" MAX_TOKENS="$MAX_TOKENS" MTP_TOKENS="$MTP_TOKENS" \
      "$PROJECT_DIR/scripts/benchmark_gsm8k_risk_entropy.sh" \
      --base-url "$BASE_URL" --output-dir "$output_dir" --progress-every 10 \
      2>&1 | tee "$log_file"
  else
    RUN_NAME=proposal_calibration \
    SAMPLES="$samples" SAMPLE_SEED="$sample_seed" TEMPERATURE="$TEMPERATURE" \
    SEED="$SEED" MAX_TOKENS="$MAX_TOKENS" MTP_TOKENS="$MTP_TOKENS" \
      "$PROJECT_DIR/scripts/benchmark_humaneval.sh" \
      --base-url "$BASE_URL" --output-dir "$output_dir" --progress-every 10 \
      2>&1 | tee "$log_file"
  fi
}

collect_one() {
  local dataset="$1" samples="$2" sample_seed="$3"
  local trace="$RUN_ROOT/${dataset}_pq_compact.jsonl"
  local server_log="$LOG_ROOT/${dataset}_trace_server.log"
  rm -f "$trace"
  setsid env MTP_TOKENS=6 REMTP_PC_TRACE_PATH="$trace" \
    REMTP_PC_TRACE_DATASET="$dataset" REMTP_PC_CONFIG="" \
    "$PROJECT_DIR/scripts/serve_proposal_calibrated_mtp.sh" >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server "$server_log"
  benchmark_dataset "$dataset" "$samples" "$sample_seed" \
    "$RUN_ROOT/trace_generation/$dataset" "$LOG_ROOT/${dataset}_trace_benchmark.log"
  cleanup_server
  if [[ ! -s "$trace" ]]; then echo "No compact trace written: $trace" >&2; exit 1; fi
}

search_offline() {
  local args=(
    --trace "gsm8k=$RUN_ROOT/gsm8k_pq_compact.jsonl"
    --trace "humaneval=$RUN_ROOT/humaneval_pq_compact.jsonl"
    --output-json "$RUN_ROOT/offline_analysis.json"
    --output-md "$RUN_ROOT/offline_analysis.md"
    --output-config "$CONFIG_PATH"
  )
  local legacy_root="$PROJECT_DIR/results/remtp_block_audit_pilot30_20260805_002409"
  if [[ -s "$legacy_root/gsm8k_rounds.jsonl" ]]; then
    args+=(--legacy "gsm8k=$legacy_root/gsm8k_rounds.jsonl")
  fi
  if [[ -s "$legacy_root/humaneval_rounds.jsonl" ]]; then
    args+=(--legacy "humaneval=$legacy_root/humaneval_rounds.jsonl")
  fi
  python -m remtp.proposal_calibration_offline "${args[@]}"
}

pilot_one() {
  local dataset="$1" samples="$2" sample_seed="$3" method="$4"
  local server_log="$LOG_ROOT/${dataset}_${method}_server.log"
  local server_script="$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh"
  local config=""
  if [[ "$method" == "calibrated" ]]; then
    server_script="$PROJECT_DIR/scripts/serve_proposal_calibrated_mtp.sh"
    config="$CONFIG_PATH"
  fi
  setsid env MTP_TOKENS=6 REMTP_PC_CONFIG="$config" "$server_script" >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server "$server_log"
  benchmark_dataset "$dataset" "$samples" "$sample_seed" \
    "$RUN_ROOT/pilot/$dataset/$method" "$LOG_ROOT/${dataset}_${method}_benchmark.log"
  cleanup_server
}

run_pilot() {
  if [[ ! -s "$CONFIG_PATH" ]]; then echo "Missing config: $CONFIG_PATH" >&2; exit 2; fi
  pilot_one gsm8k "$GSM8K_SAMPLES" "$GSM8K_SAMPLE_SEED" native
  pilot_one gsm8k "$GSM8K_SAMPLES" "$GSM8K_SAMPLE_SEED" calibrated
  pilot_one humaneval "$HUMANEVAL_SAMPLES" "$HUMANEVAL_SAMPLE_SEED" native
  pilot_one humaneval "$HUMANEVAL_SAMPLES" "$HUMANEVAL_SAMPLE_SEED" calibrated
  python -m remtp.proposal_calibration_report --root "$RUN_ROOT"
}

case "$MODE" in
  test) run_tests ;;
  collect)
    collect_one gsm8k "$GSM8K_SAMPLES" "$GSM8K_SAMPLE_SEED"
    collect_one humaneval "$HUMANEVAL_SAMPLES" "$HUMANEVAL_SAMPLE_SEED"
    ;;
  search) search_offline ;;
  pilot) run_pilot ;;
  all)
    run_tests
    collect_one gsm8k "$GSM8K_SAMPLES" "$GSM8K_SAMPLE_SEED"
    collect_one humaneval "$HUMANEVAL_SAMPLES" "$HUMANEVAL_SAMPLE_SEED"
    search_offline
    run_pilot
    ;;
esac

echo "Proposal calibration results: $RUN_ROOT"
