#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Run the four requested profiles on the same 200-example GSM8K subset:
  Scheme 1, Scheme 2, Scheme 3 adaptive-chain fallback, Scheme 1+2.

They are compared with native probabilistic MTP, Cactus and SpecCascade
TokenV3. The protocol-compatible historical Cactus/SpecCascade rows are
reused by default; native MTP and the four new profiles are run.

Usage:
  ./scripts/run_gsm8k_four_schemes.sh

Useful overrides:
  SAMPLES=200 SAMPLE_SEED=20260730 TEMPERATURE=0.7 SEED=42
  MAX_TOKENS=384 MTP_TOKENS=6 REUSE_REFERENCES=1
  PROFILES="native_mtp cactus spec_cascade scheme1 scheme2 scheme3 scheme12"
  REUSE_PROFILES="cactus spec_cascade"
  ELIGIBLE_PROFILES="scheme1 scheme2 scheme3 scheme12"
  GSM8K_REFERENCE_ROOT=results/gsm8k_mtp6_five_way_20260730_202927
  RUN_TAG=<name>

Set REUSE_REFERENCES=0 when intentionally changing the locked protocol; then
native MTP, Cactus and SpecCascade will all be rerun.
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

SAMPLES="${SAMPLES:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-20260730}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MTP_TOKENS="${MTP_TOKENS:-6}"
CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
CASCADE_RULE="${CASCADE_RULE:-token_v3}"
CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="${RUN_TAG:-three_schemes_$(date +%Y%m%d_%H%M%S)}"
REUSE_REFERENCES="${REUSE_REFERENCES:-1}"
REFERENCE_ROOT="${GSM8K_REFERENCE_ROOT:-$PROJECT_DIR/results/gsm8k_mtp6_five_way_20260730_202927}"
DATA="$PROJECT_DIR/data/gsm8k/test.jsonl"
DEFAULT_PROFILES="native_mtp cactus spec_cascade scheme1 scheme2 scheme3 scheme12"
read -r -a profiles <<< "${PROFILES:-$DEFAULT_PROFILES}"
read -r -a reuse_profiles <<< "${REUSE_PROFILES:-cactus spec_cascade}"
read -r -a eligible_profiles <<< "${ELIGIBLE_PROFILES:-scheme1 scheme2 scheme3 scheme12}"

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "This comparison requires MTP_TOKENS=6." >&2
  exit 2
fi
if [[ ! -f "$DATA" ]]; then
  echo "Missing $DATA; download GSM8K first." >&2
  exit 2
fi
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi
if [[ " ${profiles[*]} " != *" cactus "* ]]; then
  echo "PROFILES must include cactus as the comparison reference." >&2
  exit 2
fi
if [[ " ${profiles[*]} " != *" native_mtp "* ]]; then
  echo "PROFILES must include native_mtp." >&2
  exit 2
fi

RUN_ROOT="$PROJECT_DIR/results/gsm8k_three_schemes_${RUN_TAG}"
LOG_ROOT="$PROJECT_DIR/logs/gsm8k_three_schemes_${RUN_TAG}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
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
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -n 100 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  tail -n 100 "$log_file" >&2
  return 1
}

profile_server() {
  local profile="$1"
  case "$profile" in
    native_mtp) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh" ;;
    cactus) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_cactus_mtp.sh" ;;
    spec_cascade) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_spec_cascade.sh" ;;
    scheme1) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme1_mtp.sh" ;;
    scheme2) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme2_mtp.sh" ;;
    scheme2_relaxed) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme2_relaxed_mtp.sh" ;;
    scheme2_strong) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme2_strong_mtp.sh" ;;
    scheme2_ultra) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme2_ultra_mtp.sh" ;;
    scheme3) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme3_adaptive_chain_mtp.sh" ;;
    scheme12) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme12_mtp.sh" ;;
    scheme12_joint) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme12_joint_mtp.sh" ;;
    scheme12_anchored) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme12_anchored_mtp.sh" ;;
    remtp) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_remtp_mtp.sh" ;;
    remtp_block) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_remtp_block_mtp.sh" ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
  esac
}

run_method() {
  local profile="$1"
  if [[ -e "$RUN_ROOT/$profile" || -L "$RUN_ROOT/$profile" ]]; then
    echo "[reuse] skipping $profile"
    return
  fi
  profile_server "$profile"
  local server_log="$LOG_ROOT/${profile}_server.log"
  local benchmark_log="$LOG_ROOT/${profile}_benchmark.log"
  echo
  echo "===== GSM8K profile=$profile ====="
  setsid env \
    MTP_TOKENS="$MTP_TOKENS" \
    CACTUS_DELTA="$CACTUS_DELTA" \
    CASCADE_RULE="$CASCADE_RULE" \
    CASCADE_ALPHA="$CASCADE_ALPHA" \
    "$PROFILE_SCRIPT" >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server "$server_log"

  RUN_NAME="$profile" \
  SAMPLES="$SAMPLES" \
  SAMPLE_SEED="$SAMPLE_SEED" \
  TEMPERATURE="$TEMPERATURE" \
  SEED="$SEED" \
  MAX_TOKENS="$MAX_TOKENS" \
  MTP_TOKENS="$MTP_TOKENS" \
  "$PROJECT_DIR/scripts/benchmark_gsm8k_risk_entropy.sh" \
    --base-url "$BASE_URL" \
    --output-dir "$RUN_ROOT/$profile" \
    --progress-every "$PROGRESS_EVERY" \
    2>&1 | tee "$benchmark_log"
  cleanup_server
}

if [[ "$REUSE_REFERENCES" == "1" && ${#reuse_profiles[@]} -gt 0 ]]; then
  python -m remtp.gsm8k_reuse \
    --destination-root "$RUN_ROOT" \
    --source-root "$REFERENCE_ROOT" \
    --profiles "${reuse_profiles[@]}" \
    --samples "$SAMPLES" \
    --sample-seed "$SAMPLE_SEED" \
    --temperature "$TEMPERATURE" \
    --generation-seed "$SEED" \
    --max-tokens "$MAX_TOKENS" \
    --mtp-tokens "$MTP_TOKENS" \
    --data "$DATA"
elif [[ "$REUSE_REFERENCES" != "0" ]]; then
  echo "REUSE_REFERENCES must be 0 or 1." >&2
  exit 2
fi

for profile in "${profiles[@]}"; do
  run_method "$profile"
done

comparison_profiles=()
for profile in "${profiles[@]}"; do
  if [[ "$profile" != "cactus" ]]; then
    comparison_profiles+=("$profile")
  fi
done
eligible_args=()
for profile in "${eligible_profiles[@]}"; do
  eligible_args+=(--eligible "$profile")
done
python -m remtp.current_block_debt_compare \
  "$RUN_ROOT" "${comparison_profiles[@]}" "${eligible_args[@]}"

echo "Results: $RUN_ROOT/comparison.md"
