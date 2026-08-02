#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Run the protocol-locked HumanEval comparison.

Default methods:
  - native probabilistic MTP
  - Cactus + MTP
  - SpecCascade TokenV3 + MTP
  - native MTP + published Block Verification
  - Cactus + published Block Verification
  - current-block debt (balanced historical variant)
  - Cactus-dominant target surplus
  - Exact-TV + head calibration
  - Regret-Calibrated Block Relaxation (ours)

Generated code is never executed by the vLLM benchmark process. Each solution
is evaluated afterwards in a fresh Docker container with no network, a
read-only root filesystem, dropped Linux capabilities, and resource limits.

Usage:
  ./scripts/run_humaneval_comparison.sh

Useful environment variables:
  SAMPLES=164
  SAMPLE_SEED=20260802
  TEMPERATURE=0.7
  SEED=42
  MAX_TOKENS=512
  MTP_TOKENS=6
  PROFILES="native_mtp cactus spec_cascade native_block cactus_block debt_balanced target_surplus tv_head regret_calibrated_block"
  EVAL_TIMEOUT=8
  HUMANEVAL_DOCKER_IMAGE=python:3-slim
  PROGRESS_EVERY=10
  SERVER_START_TIMEOUT=180
  RUN_TAG=<timestamp>

For a quick smoke test:
  SAMPLES=5 PROFILES="native_mtp cactus regret_calibrated_block" \
    ./scripts/run_humaneval_comparison.sh

Results and logs remain under ignored local directories.
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

SAMPLES="${SAMPLES:-164}"
SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MTP_TOKENS="${MTP_TOKENS:-6}"
CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
CASCADE_RULE="${CASCADE_RULE:-token_v3}"
CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"
REGRET_FEEDBACK_SCALE="${REGRET_FEEDBACK_SCALE:-0.05}"
BLOCK_SHIELD_CACTUS_MIX="${BLOCK_SHIELD_CACTUS_MIX:-0.30}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-8}"
EVAL_MEMORY="${EVAL_MEMORY:-512m}"
EVAL_CPUS="${EVAL_CPUS:-1.0}"
HUMANEVAL_DOCKER_IMAGE="${HUMANEVAL_DOCKER_IMAGE:-python:3-slim}"
HUMANEVAL_DATA="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_PROFILES="native_mtp cactus spec_cascade native_block cactus_block debt_balanced target_surplus tv_head regret_calibrated_block"
read -r -a profiles <<< "${PROFILES:-$DEFAULT_PROFILES}"

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "This comparison requires MTP_TOKENS=6." >&2
  exit 2
fi
if [[ ! -f "$HUMANEVAL_DATA" ]]; then
  echo "Missing HumanEval data: $HUMANEVAL_DATA" >&2
  echo "Run ./scripts/download_humaneval.sh first." >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is unavailable; run ./scripts/setup_humaneval.sh." >&2
  exit 2
fi
if ! docker image inspect "$HUMANEVAL_DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "Missing Docker image $HUMANEVAL_DOCKER_IMAGE." >&2
  echo "Run ./scripts/setup_humaneval.sh first." >&2
  exit 2
fi
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi
if (( ${#profiles[@]} == 0 )); then
  echo "PROFILES must contain at least one method." >&2
  exit 2
fi
if [[ " ${profiles[*]} " != *" cactus "* ]]; then
  echo "PROFILES must include cactus as the reference." >&2
  exit 2
fi
if [[ " ${profiles[*]} " != *" native_mtp "* ]]; then
  echo "PROFILES must include native_mtp." >&2
  exit 2
fi

RUN_ROOT="$PROJECT_DIR/results/humaneval_comparison_${RUN_TAG}"
LOG_ROOT="$PROJECT_DIR/logs/humaneval_comparison_${RUN_TAG}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
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
    debt_balanced)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_current_block_debt_mtp.sh"
      ;;
    target_surplus)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_surplus_mtp.sh"
      ;;
    tv_head)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_anchored_mtp.sh"
      PROFILE_ENV=(TARGET_ANCHORED_VARIANT=tv_head)
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
  local profile="$1"
  local server_log="$LOG_ROOT/${profile}_server.log"
  local generation_log="$LOG_ROOT/${profile}_generation.log"
  local evaluation_log="$LOG_ROOT/${profile}_evaluation.log"
  local output_dir="$RUN_ROOT/$profile"

  profile_server "$profile"
  echo
  echo "===== HumanEval profile=$profile ====="
  setsid env \
    MTP_TOKENS="$MTP_TOKENS" \
    CACTUS_DELTA="$CACTUS_DELTA" \
    CASCADE_RULE="$CASCADE_RULE" \
    CASCADE_ALPHA="$CASCADE_ALPHA" \
    REGRET_FEEDBACK_SCALE="$REGRET_FEEDBACK_SCALE" \
    BLOCK_SHIELD_CACTUS_MIX="$BLOCK_SHIELD_CACTUS_MIX" \
    TARGET_ANCHORED_AUDIT_INTERVAL=0 \
    TARGET_ANCHORED_DIAGNOSTICS=0 \
    BLOCK_VERIFY_DIAGNOSTICS=0 \
    "${PROFILE_ENV[@]}" \
    "$PROFILE_SCRIPT" >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server "$server_log"

  RUN_NAME="$profile" \
  HUMANEVAL_DATA="$HUMANEVAL_DATA" \
  SAMPLES="$SAMPLES" \
  SAMPLE_SEED="$SAMPLE_SEED" \
  TEMPERATURE="$TEMPERATURE" \
  SEED="$SEED" \
  MAX_TOKENS="$MAX_TOKENS" \
  MTP_TOKENS="$MTP_TOKENS" \
  "$PROJECT_DIR/scripts/benchmark_humaneval.sh" \
    --base-url "$BASE_URL" \
    --output-dir "$output_dir" \
    --progress-every "$PROGRESS_EVERY" \
    2>&1 | tee "$generation_log"

  cleanup_server
  python -m remtp.humaneval_evaluator \
    "$output_dir" \
    --data "$HUMANEVAL_DATA" \
    --image "$HUMANEVAL_DOCKER_IMAGE" \
    --timeout "$EVAL_TIMEOUT" \
    --memory "$EVAL_MEMORY" \
    --cpus "$EVAL_CPUS" \
    --progress-every "$PROGRESS_EVERY" \
    2>&1 | tee "$evaluation_log"
}

echo "HumanEval MTP comparison"
echo "samples=$SAMPLES sample_seed=$SAMPLE_SEED"
echo "temperature=$TEMPERATURE generation_seed=$SEED max_tokens=$MAX_TOKENS"
echo "mtp_tokens=$MTP_TOKENS"
echo "profiles=${profiles[*]}"
echo "container=$HUMANEVAL_DOCKER_IMAGE timeout=${EVAL_TIMEOUT}s"
echo "local_results=$RUN_ROOT"

for profile in "${profiles[@]}"; do
  run_method "$profile"
done

python -m remtp.humaneval_compare "$RUN_ROOT" "${profiles[@]}"
echo
echo "HumanEval comparison complete: $RUN_ROOT/comparison.md"
