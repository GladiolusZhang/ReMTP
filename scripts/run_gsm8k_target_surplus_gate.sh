#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Run native probabilistic MTP, Cactus, the old balanced debt controller, the
old target-surplus ablation, and two target-anchored candidates on one GSM8K
seed. Confirm the best passing candidate on a second seed.

Usage:
  ./scripts/run_gsm8k_target_anchored_gate.sh

Environment:
  SAMPLES=100
  SAMPLE_SEED=20260804
  SECOND_SAMPLE_SEED=20260811
  TEMPERATURE=0.7
  SEED=42
  MAX_TOKENS=384
  MTP_TOKENS=6
  CACTUS_DELTA=1.0
  SURPLUS_MAX_LOG_GAP=2.0
  RISK_SWAP_SOFT_LOG_GAP=4.0
  RISK_SWAP_HARD_LOG_GAP=10.0
  RISK_SWAP_DESTINATION_LOG_GAP=2.0
  PROGRESS_EVERY=10
  SERVER_START_TIMEOUT=180
  RUN_TAG=<timestamp>

Production audit is always disabled for a fair throughput comparison.
Results and logs remain local under ignored results/ and logs/ directories.
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

SAMPLES="${SAMPLES:-100}"
SAMPLE_SEED="${SAMPLE_SEED:-20260804}"
SECOND_SAMPLE_SEED="${SECOND_SAMPLE_SEED:-20260811}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MTP_TOKENS="${MTP_TOKENS:-6}"
CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
SURPLUS_MAX_LOG_GAP="${SURPLUS_MAX_LOG_GAP:-2.0}"
RISK_SWAP_SOFT_LOG_GAP="${RISK_SWAP_SOFT_LOG_GAP:-4.0}"
RISK_SWAP_HARD_LOG_GAP="${RISK_SWAP_HARD_LOG_GAP:-10.0}"
RISK_SWAP_DESTINATION_LOG_GAP="${RISK_SWAP_DESTINATION_LOG_GAP:-2.0}"
HEAD_RELIABILITY="${HEAD_RELIABILITY:-1.0,0.85,0.70,0.55,0.40,0.30}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "This experiment requires MTP_TOKENS=6." >&2
  exit 2
fi
if [[ "$SAMPLE_SEED" == "$SECOND_SAMPLE_SEED" ]]; then
  echo "SAMPLE_SEED and SECOND_SAMPLE_SEED must differ." >&2
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

RUN_ROOT="$PROJECT_DIR/results/gsm8k_target_anchored_gate_${RUN_TAG}"
LOG_ROOT="$PROJECT_DIR/logs/gsm8k_target_anchored_gate_${RUN_TAG}"
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

profile_server() {
  local profile="$1"
  case "$profile" in
    native_mtp)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh"
      PROFILE_ENV=()
      ;;
    cactus)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_cactus_mtp.sh"
      PROFILE_ENV=(REMTP_CACTUS_DIAGNOSTICS=0)
      ;;
    debt_balanced)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_current_block_debt_mtp.sh"
      PROFILE_ENV=(
        DEBT_POSITION_LIMIT=0.35 DEBT_BLOCK_LIMIT=1.20
        DEBT_SOFT_LOG_GAP=2.0 DEBT_HARD_LOG_GAP=6.0
        DEBT_MAX_POSITION_TV=0.15 DEBT_MAX_CACTUS_RATIO=2.0
        DEBT_FALLBACK=strict
      )
      ;;
    target_surplus)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_surplus_mtp.sh"
      PROFILE_ENV=(SURPLUS_MAX_LOG_GAP="$SURPLUS_MAX_LOG_GAP")
      ;;
    risk_swap)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_risk_swap_mtp.sh"
      PROFILE_ENV=(
        RISK_SWAP_SOFT_LOG_GAP="$RISK_SWAP_SOFT_LOG_GAP"
        RISK_SWAP_HARD_LOG_GAP="$RISK_SWAP_HARD_LOG_GAP"
        RISK_SWAP_DESTINATION_LOG_GAP="$RISK_SWAP_DESTINATION_LOG_GAP"
      )
      ;;
    target_recovery)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_recovery_mtp.sh"
      PROFILE_ENV=(SURPLUS_MAX_LOG_GAP="$SURPLUS_MAX_LOG_GAP")
      ;;
    *)
      echo "Unknown profile: $profile" >&2
      exit 2
      ;;
  esac
}

run_method() {
  local stage_root="$1"
  local log_stage="$2"
  local sample_seed="$3"
  local profile="$4"
  local server_log="$log_stage/${profile}_server.log"
  local benchmark_log="$log_stage/${profile}_benchmark.log"

  profile_server "$profile"
  echo
  echo "===== seed=${sample_seed} profile=${profile} ====="
  setsid env \
    MTP_TOKENS="$MTP_TOKENS" \
    CACTUS_DELTA="$CACTUS_DELTA" \
    HEAD_RELIABILITY="$HEAD_RELIABILITY" \
    TARGET_ANCHORED_AUDIT_INTERVAL=0 \
    TARGET_ANCHORED_DIAGNOSTICS=0 \
    "${PROFILE_ENV[@]}" \
    "$PROFILE_SCRIPT" >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server "$server_log"

  RUN_NAME="$profile" \
  SAMPLES="$SAMPLES" \
  SAMPLE_SEED="$sample_seed" \
  TEMPERATURE="$TEMPERATURE" \
  SEED="$SEED" \
  MAX_TOKENS="$MAX_TOKENS" \
  MTP_TOKENS="$MTP_TOKENS" \
  "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" \
    --base-url "$BASE_URL" \
    --output-dir "$stage_root/$profile" \
    --progress-every "$PROGRESS_EVERY" \
    2>&1 | tee "$benchmark_log"
  cleanup_server
}

run_stage() {
  local sample_seed="$1"
  shift
  local profiles=("$@")
  local stage_root="$RUN_ROOT/seed_${sample_seed}"
  local log_stage="$LOG_ROOT/seed_${sample_seed}"
  mkdir -p "$log_stage"
  mkdir "$stage_root"

  run_method "$stage_root" "$log_stage" "$sample_seed" cactus
  for profile in "${profiles[@]}"; do
    run_method "$stage_root" "$log_stage" "$sample_seed" "$profile"
  done
  python -m remtp.current_block_debt_compare \
    "$stage_root" "${profiles[@]}"
}

mkdir -p "$LOG_ROOT"
mkdir "$RUN_ROOT"

echo "GSM8K target-anchored relaxed-MTP gate"
echo "samples=$SAMPLES seeds=$SAMPLE_SEED,$SECOND_SAMPLE_SEED"
echo "temperature=$TEMPERATURE generation_seed=$SEED max_tokens=$MAX_TOKENS"
echo "production_audit=0 local_results=$RUN_ROOT"

run_stage "$SAMPLE_SEED" native_mtp debt_balanced target_surplus risk_swap target_recovery
first_stage="$RUN_ROOT/seed_${SAMPLE_SEED}"
winner="$(python -c '
import json, sys
rows = json.load(open(sys.argv[1]))
rows = [row for row in rows if row["directory"] in {"risk_swap", "target_recovery"} and row["pareto_pass"]]
if not rows:
    print("")
else:
    row = max(rows, key=lambda x: (x["accuracy"], x["mean_acceptance_length"], x["e2e_tok_s"]))
    print(row["directory"])
' "$first_stage/comparison.json")"

if [[ -z "$winner" ]]; then
  echo
  echo "No target-anchored candidate passed the first-seed Pareto gate."
  echo "The second seed was not run: $first_stage/comparison.md"
  exit 0
fi

echo
echo "$winner passed seed $SAMPLE_SEED; confirming on $SECOND_SAMPLE_SEED."
run_stage "$SECOND_SAMPLE_SEED" native_mtp "$winner"
second_stage="$RUN_ROOT/seed_${SECOND_SAMPLE_SEED}"
second_pass="$(python -c '
import json, sys
print(int(json.load(open(sys.argv[1]))["pareto_pass"]))
' "$second_stage/gate_selection.json")"

echo
if [[ "$second_pass" == "1" ]]; then
  echo "REPRODUCED: $winner passed all three metrics on both seeds."
else
  echo "NOT REPRODUCED: $winner failed the second-seed gate."
fi
echo "Seed 1: $first_stage/comparison.md"
echo "Seed 2: $second_stage/comparison.md"
