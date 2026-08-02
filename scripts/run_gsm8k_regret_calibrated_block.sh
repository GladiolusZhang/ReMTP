#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Run the locked GSM8K comparison for Regret-Calibrated Block Relaxation.

The comparison contains native probabilistic MTP, Cactus, SpecCascade
TokenV3, native joint Block Verification, Cactus + Block Verification, and
our unified relaxation plus current-block regret negative feedback method.

Usage:
  ./scripts/run_gsm8k_regret_calibrated_block.sh

Environment:
  SAMPLES=100
  SAMPLE_SEED=20261103
  SECOND_SAMPLE_SEED=20261117
  TEMPERATURE=0.7
  SEED=42
  MAX_TOKENS=384
  MTP_TOKENS=6
  CACTUS_DELTA=1.0
  CASCADE_RULE=token_v3
  CASCADE_ALPHA=0.5
  REGRET_FEEDBACK_SCALE=0.05
  RISK_SWAP_SOFT_LOG_GAP=4.0
  RISK_SWAP_HARD_LOG_GAP=10.0
  RISK_SWAP_DESTINATION_LOG_GAP=2.0
  BLOCK_SHIELD_CACTUS_MIX=0.30
  PROGRESS_EVERY=10
  SERVER_START_TIMEOUT=180
  CONFIRM_ON_SUCCESS=1
  RUN_TAG=<timestamp>

Success requires our unchanged method to have Accuracy > Cactus,
MAL > Cactus, and E2E >= Cactus. A first-seed success is repeated unchanged
on the second seed. There is no method switch or learned fallback.
Results and logs remain in ignored local directories.
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
SAMPLE_SEED="${SAMPLE_SEED:-20261103}"
SECOND_SAMPLE_SEED="${SECOND_SAMPLE_SEED:-20261117}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MTP_TOKENS="${MTP_TOKENS:-6}"
CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
CASCADE_RULE="${CASCADE_RULE:-token_v3}"
CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"
REGRET_FEEDBACK_SCALE="${REGRET_FEEDBACK_SCALE:-0.05}"
RISK_SWAP_SOFT_LOG_GAP="${RISK_SWAP_SOFT_LOG_GAP:-4.0}"
RISK_SWAP_HARD_LOG_GAP="${RISK_SWAP_HARD_LOG_GAP:-10.0}"
RISK_SWAP_DESTINATION_LOG_GAP="${RISK_SWAP_DESTINATION_LOG_GAP:-2.0}"
BLOCK_SHIELD_CACTUS_MIX="${BLOCK_SHIELD_CACTUS_MIX:-0.30}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
CONFIRM_ON_SUCCESS="${CONFIRM_ON_SUCCESS:-1}"
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
if [[ "$CONFIRM_ON_SUCCESS" != "0" && "$CONFIRM_ON_SUCCESS" != "1" ]]; then
  echo "CONFIRM_ON_SUCCESS must be 0 or 1." >&2
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

RUN_ROOT="$PROJECT_DIR/results/gsm8k_regret_calibrated_block_${RUN_TAG}"
LOG_ROOT="$PROJECT_DIR/logs/gsm8k_regret_calibrated_block_${RUN_TAG}"
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
  PROFILE_ENV=()
  case "$profile" in
    native_mtp)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh"
      ;;
    cactus)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_cactus_mtp.sh"
      PROFILE_ENV=(REMTP_CACTUS_DIAGNOSTICS=0)
      ;;
    spec_cascade)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_spec_cascade.sh"
      ;;
    native_block)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_block_mtp.sh"
      ;;
    cactus_block)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_cactus_block_mtp.sh"
      PROFILE_ENV=(REMTP_CACTUS_DIAGNOSTICS=0)
      ;;
    regret_calibrated_block)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_regret_calibrated_block_mtp.sh"
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
    CASCADE_RULE="$CASCADE_RULE" \
    CASCADE_ALPHA="$CASCADE_ALPHA" \
    REGRET_FEEDBACK_SCALE="$REGRET_FEEDBACK_SCALE" \
    RISK_SWAP_SOFT_LOG_GAP="$RISK_SWAP_SOFT_LOG_GAP" \
    RISK_SWAP_HARD_LOG_GAP="$RISK_SWAP_HARD_LOG_GAP" \
    RISK_SWAP_DESTINATION_LOG_GAP="$RISK_SWAP_DESTINATION_LOG_GAP" \
    BLOCK_SHIELD_CACTUS_MIX="$BLOCK_SHIELD_CACTUS_MIX" \
    TARGET_ANCHORED_AUDIT_INTERVAL=0 \
    TARGET_ANCHORED_DIAGNOSTICS=0 \
    BLOCK_VERIFY_DIAGNOSTICS=0 \
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
  local stage_root="$RUN_ROOT/seed_${sample_seed}"
  local log_stage="$LOG_ROOT/seed_${sample_seed}"
  local profiles=(
    native_mtp spec_cascade native_block cactus_block
    regret_calibrated_block
  )
  mkdir -p "$log_stage"
  mkdir "$stage_root"

  run_method "$stage_root" "$log_stage" "$sample_seed" cactus
  for profile in "${profiles[@]}"; do
    run_method "$stage_root" "$log_stage" "$sample_seed" "$profile"
  done
  python -m remtp.current_block_debt_compare \
    "$stage_root" "${profiles[@]}" \
    --eligible regret_calibrated_block
}

mkdir -p "$LOG_ROOT"
mkdir "$RUN_ROOT"

echo "GSM8K Regret-Calibrated Block Relaxation comparison"
echo "samples=$SAMPLES seeds=$SAMPLE_SEED,$SECOND_SAMPLE_SEED"
echo "temperature=$TEMPERATURE generation_seed=$SEED max_tokens=$MAX_TOKENS"
echo "mtp_tokens=$MTP_TOKENS cactus_delta=$CACTUS_DELTA"
echo "cascade_rule=$CASCADE_RULE cascade_alpha=$CASCADE_ALPHA"
echo "regret_scale=$REGRET_FEEDBACK_SCALE"
echo "correction_gap=$RISK_SWAP_SOFT_LOG_GAP:$RISK_SWAP_HARD_LOG_GAP"
echo "destination_gap=$RISK_SWAP_DESTINATION_LOG_GAP cactus_mix=$BLOCK_SHIELD_CACTUS_MIX"
echo "local_results=$RUN_ROOT"

run_stage "$SAMPLE_SEED"
first_stage="$RUN_ROOT/seed_${SAMPLE_SEED}"
first_pass="$(python -c '
import json, sys
rows = {x["directory"]: x for x in json.load(open(sys.argv[1]))}
print(int(rows["regret_calibrated_block"]["pareto_pass"]))
' "$first_stage/comparison.json")"

if [[ "$first_pass" != "1" ]]; then
  echo
  echo "Regret-Calibrated Block Relaxation did not satisfy all criteria."
  echo "The second seed is not run and no alternate method is selected."
  echo "Result: $first_stage/comparison.md"
  exit 0
fi

if [[ "$CONFIRM_ON_SUCCESS" == "0" ]]; then
  echo
  echo "The method satisfied the first-seed criteria."
  echo "Confirmation was disabled for this mechanism screen."
  echo "Result: $first_stage/comparison.md"
  exit 0
fi

echo
echo "The method satisfied seed $SAMPLE_SEED; repeating unchanged."
run_stage "$SECOND_SAMPLE_SEED"
second_stage="$RUN_ROOT/seed_${SECOND_SAMPLE_SEED}"
second_pass="$(python -c '
import json, sys
rows = {x["directory"]: x for x in json.load(open(sys.argv[1]))}
print(int(rows["regret_calibrated_block"]["pareto_pass"]))
' "$second_stage/comparison.json")"

echo
if [[ "$second_pass" == "1" ]]; then
  echo "REPRODUCED: regret_calibrated_block"
else
  echo "NOT REPRODUCED: regret_calibrated_block"
fi
echo "Seed 1: $first_stage/comparison.md"
echo "Seed 2: $second_stage/comparison.md"
