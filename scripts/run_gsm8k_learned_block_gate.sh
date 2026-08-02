#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Calibrate a tiny request-agnostic Cactus/Block-Shield router, then compare it
with native MTP and paper baselines on held-out GSM8K questions.

Usage:
  ./scripts/run_gsm8k_learned_block_gate.sh

Environment:
  CALIBRATION_SAMPLES=100
  SAMPLES=100
  CALIBRATION_SAMPLE_SEED=20260913
  SAMPLE_SEED=20260818
  SECOND_SAMPLE_SEED=20260825
  TEMPERATURE=0.7
  SEED=42
  MAX_TOKENS=384
  MTP_TOKENS=6
  CACTUS_DELTA=1.0
  CASCADE_ALPHA=0.5
  BLOCK_SHIELD_CACTUS_MIX=0.30
  BLOCK_GATE_SURPLUS_SPEND_FRACTION=0.50
  BLOCK_GATE_MAX_FALSE_POSITIVE_RATE=0.05
  BLOCK_GATE_MIN_REMAINING_SURPLUS=0.10
  PROGRESS_EVERY=10
  SERVER_START_TIMEOUT=180
  RUN_TAG=<timestamp>

Calibration, test seed 1, and test seed 2 use disjoint question manifests.
The second test seed runs only if learned_gate passes the first Pareto gate.
Models, results, and logs remain in ignored local directories.
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

CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-100}"
SAMPLES="${SAMPLES:-100}"
CALIBRATION_SAMPLE_SEED="${CALIBRATION_SAMPLE_SEED:-20260913}"
SAMPLE_SEED="${SAMPLE_SEED:-20260818}"
SECOND_SAMPLE_SEED="${SECOND_SAMPLE_SEED:-20260825}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MTP_TOKENS="${MTP_TOKENS:-6}"
CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"
BLOCK_SHIELD_CACTUS_MIX="${BLOCK_SHIELD_CACTUS_MIX:-0.30}"
BLOCK_GATE_SURPLUS_SPEND_FRACTION="${BLOCK_GATE_SURPLUS_SPEND_FRACTION:-0.50}"
BLOCK_GATE_MAX_FALSE_POSITIVE_RATE="${BLOCK_GATE_MAX_FALSE_POSITIVE_RATE:-0.05}"
BLOCK_GATE_MIN_REMAINING_SURPLUS="${BLOCK_GATE_MIN_REMAINING_SURPLUS:-0.10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "This experiment requires MTP_TOKENS=6." >&2
  exit 2
fi
if [[ "$CALIBRATION_SAMPLE_SEED" == "$SAMPLE_SEED" || \
      "$CALIBRATION_SAMPLE_SEED" == "$SECOND_SAMPLE_SEED" || \
      "$SAMPLE_SEED" == "$SECOND_SAMPLE_SEED" ]]; then
  echo "Calibration and test seeds must be distinct." >&2
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

RUN_ROOT="$PROJECT_DIR/results/gsm8k_learned_block_gate_${RUN_TAG}"
LOG_ROOT="$PROJECT_DIR/logs/gsm8k_learned_block_gate_${RUN_TAG}"
CALIBRATION_DATA="$RUN_ROOT/block_gate_calibration.jsonl"
CALIBRATION_RUN="$RUN_ROOT/calibration_cactus_block"
GATE_MODEL="$RUN_ROOT/block_gate_model.json"
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
      tail -n 100 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "Server startup timed out after ${SERVER_START_TIMEOUT}s." >&2
  tail -n 100 "$log_file" >&2
  return 1
}

start_server() {
  local profile="$1"
  local log_file="$2"
  local script
  local -a profile_env=()
  case "$profile" in
    collect)
      script="$PROJECT_DIR/scripts/serve_block_gate_collect.sh"
      profile_env=(
        BLOCK_GATE_DATA="$CALIBRATION_DATA"
        BLOCK_GATE_SURPLUS_SPEND_FRACTION="$BLOCK_GATE_SURPLUS_SPEND_FRACTION"
      )
      ;;
    native_mtp)
      script="$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh"
      ;;
    cactus)
      script="$PROJECT_DIR/scripts/serve_cactus_mtp.sh"
      profile_env=(REMTP_CACTUS_DIAGNOSTICS=0)
      ;;
    spec_cascade)
      script="$PROJECT_DIR/scripts/serve_spec_cascade.sh"
      profile_env=(CASCADE_RULE=token_v3 CASCADE_ALPHA="$CASCADE_ALPHA")
      ;;
    cactus_block)
      script="$PROJECT_DIR/scripts/serve_cactus_block_mtp.sh"
      profile_env=(REMTP_CACTUS_DIAGNOSTICS=0)
      ;;
    block_shield)
      script="$PROJECT_DIR/scripts/serve_block_shield_mtp.sh"
      ;;
    learned_gate)
      script="$PROJECT_DIR/scripts/serve_learned_block_gate.sh"
      profile_env=(BLOCK_GATE_MODEL="$GATE_MODEL")
      ;;
    *)
      echo "Unknown profile: $profile" >&2
      exit 2
      ;;
  esac
  setsid env \
    MTP_TOKENS="$MTP_TOKENS" \
    CACTUS_DELTA="$CACTUS_DELTA" \
    BLOCK_SHIELD_CACTUS_MIX="$BLOCK_SHIELD_CACTUS_MIX" \
    TARGET_ANCHORED_AUDIT_INTERVAL=0 \
    TARGET_ANCHORED_DIAGNOSTICS=0 \
    BLOCK_VERIFY_DIAGNOSTICS=0 \
    REMTP_BLOCK_GATE_DIAGNOSTICS=0 \
    "${profile_env[@]}" \
    "$script" >"$log_file" 2>&1 &
  server_pid=$!
  wait_for_server "$log_file"
}

run_benchmark() {
  local output_dir="$1"
  local run_name="$2"
  local sample_count="$3"
  local sample_seed="$4"
  shift 4
  local -a exclude_args=()
  for manifest in "$@"; do
    exclude_args+=(--exclude-manifest "$manifest")
  done
  RUN_NAME="$run_name" \
  SAMPLES="$sample_count" \
  SAMPLE_SEED="$sample_seed" \
  TEMPERATURE="$TEMPERATURE" \
  SEED="$SEED" \
  MAX_TOKENS="$MAX_TOKENS" \
  MTP_TOKENS="$MTP_TOKENS" \
  "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" \
    --base-url "$BASE_URL" \
    --output-dir "$output_dir" \
    --progress-every "$PROGRESS_EVERY" \
    "${exclude_args[@]}"
}

collect_and_train() {
  local server_log="$LOG_ROOT/calibration_server.log"
  local benchmark_log="$LOG_ROOT/calibration_benchmark.log"
  echo
  echo "===== calibration: Cactus + Block Verification ====="
  start_server collect "$server_log"
  run_benchmark \
    "$CALIBRATION_RUN" calibration_cactus_block \
    "$CALIBRATION_SAMPLES" "$CALIBRATION_SAMPLE_SEED" \
    2>&1 | tee "$benchmark_log"
  cleanup_server
  python -m remtp.train_block_gate \
    --data "$CALIBRATION_DATA" \
    --output "$GATE_MODEL" \
    --seed "$SEED" \
    --drop-first-groups 1 \
    --max-false-positive-rate "$BLOCK_GATE_MAX_FALSE_POSITIVE_RATE" \
    --min-expected-delta -1.0 \
    --min-remaining-surplus "$BLOCK_GATE_MIN_REMAINING_SURPLUS" \
    2>&1 | tee "$LOG_ROOT/gate_training.log"
}

run_method() {
  local stage_root="$1"
  local log_stage="$2"
  local sample_seed="$3"
  local profile="$4"
  shift 4
  local server_log="$log_stage/${profile}_server.log"
  local benchmark_log="$log_stage/${profile}_benchmark.log"
  echo
  echo "===== seed=${sample_seed} profile=${profile} ====="
  start_server "$profile" "$server_log"
  run_benchmark \
    "$stage_root/$profile" "$profile" "$SAMPLES" "$sample_seed" "$@" \
    2>&1 | tee "$benchmark_log"
  cleanup_server
}

run_stage() {
  local sample_seed="$1"
  shift
  local -a exclusions=("$@")
  local stage_root="$RUN_ROOT/seed_${sample_seed}"
  local log_stage="$LOG_ROOT/seed_${sample_seed}"
  local -a profiles=(
    native_mtp spec_cascade cactus_block block_shield learned_gate
  )
  mkdir -p "$stage_root" "$log_stage"
  run_method "$stage_root" "$log_stage" "$sample_seed" cactus \
    "${exclusions[@]}"
  for profile in "${profiles[@]}"; do
    run_method "$stage_root" "$log_stage" "$sample_seed" "$profile" \
      "${exclusions[@]}"
  done
  python -m remtp.current_block_debt_compare \
    "$stage_root" "${profiles[@]}"
}

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

echo "GSM8K calibrated minimal Block Shield gate"
echo "calibration_samples=$CALIBRATION_SAMPLES test_samples=$SAMPLES"
echo "calibration_seed=$CALIBRATION_SAMPLE_SEED test_seeds=$SAMPLE_SEED,$SECOND_SAMPLE_SEED"
echo "temperature=$TEMPERATURE generation_seed=$SEED max_tokens=$MAX_TOKENS"
echo "local_results=$RUN_ROOT"

collect_and_train
calibration_manifest="$CALIBRATION_RUN/sample_manifest.json"

run_stage "$SAMPLE_SEED" "$calibration_manifest"
first_stage="$RUN_ROOT/seed_${SAMPLE_SEED}"
first_pass="$(python -c '
import json, sys
rows = json.load(open(sys.argv[1]))
row = next(x for x in rows if x["directory"] == "learned_gate")
print(int(row["pareto_pass"]))
' "$first_stage/comparison.json")"

if [[ "$first_pass" != "1" ]]; then
  echo
  echo "The calibrated router did not pass the first held-out Pareto gate."
  echo "The second seed was not run: $first_stage/comparison.md"
  exit 0
fi

echo
echo "The calibrated router passed seed $SAMPLE_SEED; confirming."
run_stage \
  "$SECOND_SAMPLE_SEED" \
  "$calibration_manifest" \
  "$first_stage/cactus/sample_manifest.json"
second_stage="$RUN_ROOT/seed_${SECOND_SAMPLE_SEED}"
second_pass="$(python -c '
import json, sys
rows = json.load(open(sys.argv[1]))
row = next(x for x in rows if x["directory"] == "learned_gate")
print(int(row["pareto_pass"]))
' "$second_stage/comparison.json")"

echo
if [[ "$second_pass" == "1" ]]; then
  echo "REPRODUCED: learned gate passed both held-out seeds."
else
  echo "NOT REPRODUCED: learned gate failed the second held-out seed."
fi
echo "Seed 1: $first_stage/comparison.md"
echo "Seed 2: $second_stage/comparison.md"
