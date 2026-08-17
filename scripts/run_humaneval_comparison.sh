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
  - Cactus + residual regret feedback
  - SpecCascade TokenV3 + MTP
  - native MTP + published Block Verification
  - Cactus + published Block Verification
  - current-block debt (balanced historical variant)
  - Cactus-dominant target surplus
  - Exact-TV + head calibration
  - Exact-TV + head/hidden + future veto
  - Exact-TV + target-only future veto
  - Exact-TV + learned expected-regret Router
  - Regret-Calibrated Block Relaxation (ours)
  - Target-Mode Rescue + within-block regret (ours)
  - Prefix-Credit native MTP relaxation + joint verification (ours)
  - fused strict-MTP identity control (attribution)

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
  CACTUS_REGRET_ALPHA=0.03
  PROFILES="native_mtp cactus cactus_regret spec_cascade native_block cactus_block debt_balanced target_surplus tv_head tv_hidden_veto exact_tv exact_tv_regret_router regret_calibrated_block target_mode_identity target_mode_regret target_mode_p05_b060 target_mode_p05_b075 target_mode_p05_b090 target_mode_p05_b095 target_mode_top1_m000 target_mode_top1_m010 target_mode_top1_m025 target_mode_top1_m050 target_band_r2_g025 target_band_r4_g050 target_band_r4_g075 target_band_r8_g100 prefix_credit_token_cap prefix_credit_atomic prefix_credit prefix_credit_g025 prefix_credit_g050 prefix_credit_g100 prefix_credit_b060 prefix_credit_b075 prefix_credit_b090"
  REGRET_ROUTER_CHECKPOINT=checkpoints/regret_router.pt
  EVAL_TIMEOUT=8
  EVAL_WORKERS=1
  HUMANEVAL_DOCKER_IMAGE=python:3-slim
  PROGRESS_EVERY=10
  SERVER_START_TIMEOUT=180
  RUN_TAG=<timestamp>
  REUSE_PROFILES="native_mtp cactus spec_cascade"
  REUSE_RESULT_ROOTS="results/old_run_a:results/old_run_b"
  REUSE_REQUIRED=1

For a quick smoke test:
  SAMPLES=5 PROFILES="native_mtp cactus target_mode_regret" \
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
CACTUS_REGRET_ALPHA="${CACTUS_REGRET_ALPHA:-0.03}"
CACTUS_REGRET_TOP_K="${CACTUS_REGRET_TOP_K:-16}"
CACTUS_REGRET_STRENGTH_REFERENCE="${CACTUS_REGRET_STRENGTH_REFERENCE:-0.10}"
CACTUS_REGRET_DEPTH_DECAY="${CACTUS_REGRET_DEPTH_DECAY:-0.90}"
CACTUS_REGRET_RESPONSIBILITY="${CACTUS_REGRET_RESPONSIBILITY:-posterior}"
CACTUS_REGRET_INJECTION_SITE="${CACTUS_REGRET_INJECTION_SITE:-head1}"
CACTUS_REGRET_RESIDUAL_SPACE="${CACTUS_REGRET_RESIDUAL_SPACE:-output}"
CACTUS_REGRET_AUDIT_INTERVAL="${CACTUS_REGRET_AUDIT_INTERVAL:-0}"
CACTUS_REGRET_DIAGNOSTICS="${CACTUS_REGRET_DIAGNOSTICS:-0}"
BLOCK_SHIELD_CACTUS_MIX="${BLOCK_SHIELD_CACTUS_MIX:-0.30}"
REGRET_ROUTER_CHECKPOINT="${REGRET_ROUTER_CHECKPOINT:-$PROJECT_DIR/checkpoints/regret_router.pt}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-8}"
EVAL_MEMORY="${EVAL_MEMORY:-512m}"
EVAL_CPUS="${EVAL_CPUS:-1.0}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
HUMANEVAL_DOCKER_IMAGE="${HUMANEVAL_DOCKER_IMAGE:-python:3-slim}"
HUMANEVAL_DATA="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REUSE_PROFILES="${REUSE_PROFILES:-}"
REUSE_RESULT_ROOTS="${REUSE_RESULT_ROOTS:-}"
REUSE_REQUIRED="${REUSE_REQUIRED:-1}"
DEFAULT_PROFILES="native_mtp target_mode_identity cactus spec_cascade exact_tv target_mode_regret"
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

if [[ -n "$REUSE_PROFILES" ]]; then
  if [[ -z "$REUSE_RESULT_ROOTS" ]]; then
    echo "REUSE_RESULT_ROOTS is required when REUSE_PROFILES is set." >&2
    exit 2
  fi
  read -r -a reuse_profiles <<< "$REUSE_PROFILES"
  IFS=':' read -r -a reuse_roots <<< "$REUSE_RESULT_ROOTS"
  reuse_args=()
  for root in "${reuse_roots[@]}"; do
    reuse_args+=(--source-root "$root")
  done
  image_id="$(docker image inspect --format '{{.Id}}' "$HUMANEVAL_DOCKER_IMAGE")"
  allow_missing=()
  if [[ "$REUSE_REQUIRED" == "0" ]]; then
    allow_missing=(--allow-missing)
  fi
  python -m remtp.humaneval_reuse \
    --destination-root "$RUN_ROOT" \
    "${reuse_args[@]}" \
    --profiles "${reuse_profiles[@]}" \
    --samples "$SAMPLES" \
    --sample-seed "$SAMPLE_SEED" \
    --temperature "$TEMPERATURE" \
    --generation-seed "$SEED" \
    --max-tokens "$MAX_TOKENS" \
    --mtp-tokens "$MTP_TOKENS" \
    --data "$HUMANEVAL_DATA" \
    --evaluation-image "$HUMANEVAL_DOCKER_IMAGE" \
    --evaluation-image-id "$image_id" \
    --evaluation-timeout "$EVAL_TIMEOUT" \
    "${allow_missing[@]}"
fi

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
    cactus_regret)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_cactus_regret_mtp.sh"
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
    tv_hidden_veto)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_anchored_mtp.sh"
      PROFILE_ENV=(TARGET_ANCHORED_VARIANT=tv_hidden_veto)
      ;;
    exact_tv)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_anchored_mtp.sh"
      PROFILE_ENV=(TARGET_ANCHORED_VARIANT=tv_router)
      ;;
    exact_tv_regret_fixed)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_exact_tv_regret_router.sh"
      PROFILE_ENV=(REGRET_ROUTER_MODE=fixed)
      ;;
    exact_tv_regret_router)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_exact_tv_regret_router.sh"
      PROFILE_ENV=(
        REGRET_ROUTER_MODE=learned
        REGRET_ROUTER_CHECKPOINT="$REGRET_ROUTER_CHECKPOINT"
      )
      ;;
    regret_calibrated_block)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_regret_calibrated_block_mtp.sh"
      ;;
    target_mode_regret)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      ;;
    target_mode_identity)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=probability
        MODE_REGRET_BLOCK_TV=0
        MODE_REGRET_COMPILE=0
      )
      ;;
    target_mode_p05_b060)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=probability
        MODE_REGRET_MIN_TARGET_PROB=0.50
        MODE_REGRET_PER_TOKEN_TV=0.49
        MODE_REGRET_BLOCK_TV=0.60
        MODE_REGRET_DEBT_SLOPE=0.50
        MODE_REGRET_DEPTH_SLOPE=0.00
      )
      ;;
    target_mode_p05_b075)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=probability
        MODE_REGRET_MIN_TARGET_PROB=0.50
        MODE_REGRET_PER_TOKEN_TV=0.49
        MODE_REGRET_BLOCK_TV=0.75
        MODE_REGRET_DEBT_SLOPE=0.35
        MODE_REGRET_DEPTH_SLOPE=0.00
      )
      ;;
    target_mode_p05_b090)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=probability
        MODE_REGRET_MIN_TARGET_PROB=0.50
        MODE_REGRET_PER_TOKEN_TV=0.49
        MODE_REGRET_BLOCK_TV=0.90
        MODE_REGRET_DEBT_SLOPE=0.20
        MODE_REGRET_DEPTH_SLOPE=0.00
      )
      ;;
    target_mode_p05_b095)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=probability
        MODE_REGRET_MIN_TARGET_PROB=0.50
        MODE_REGRET_PER_TOKEN_TV=0.49
        MODE_REGRET_BLOCK_TV=0.95
        MODE_REGRET_DEBT_SLOPE=0.00
        MODE_REGRET_DEPTH_SLOPE=0.00
      )
      ;;
    target_mode_top1_m000|target_mode_top1_m010|target_mode_top1_m025|target_mode_top1_m050)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      case "$profile" in
        target_mode_top1_m000) margin=0.00 ;;
        target_mode_top1_m010) margin=0.10 ;;
        target_mode_top1_m025) margin=0.25 ;;
        target_mode_top1_m050) margin=0.50 ;;
      esac
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=top1_margin
        MODE_REGRET_MIN_TOP1_MARGIN="$margin"
        MODE_REGRET_PER_TOKEN_TV=0.49
        MODE_REGRET_BLOCK_TV=0.75
        MODE_REGRET_MARGIN_DEBT_SLOPE=0.35
        MODE_REGRET_MARGIN_DEPTH_SLOPE=0.00
      )
      ;;
    target_band_r2_g025)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=target_band
        MODE_REGRET_MAX_TARGET_RANK=2
        MODE_REGRET_MAX_CANDIDATE_GAP=0.25
        MODE_REGRET_PER_TOKEN_TV=0.20
        MODE_REGRET_BLOCK_TV=0.60
        MODE_REGRET_BAND_DEBT_SLOPE=0.75
        MODE_REGRET_BAND_DEPTH_SLOPE=0.00
      )
      ;;
    target_band_r4_g050)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=target_band
        MODE_REGRET_MAX_TARGET_RANK=4
        MODE_REGRET_MAX_CANDIDATE_GAP=0.50
        MODE_REGRET_PER_TOKEN_TV=0.25
        MODE_REGRET_BLOCK_TV=0.75
        MODE_REGRET_BAND_DEBT_SLOPE=0.75
        MODE_REGRET_BAND_DEPTH_SLOPE=0.00
      )
      ;;
    target_band_r4_g075)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=target_band
        MODE_REGRET_MAX_TARGET_RANK=4
        MODE_REGRET_MAX_CANDIDATE_GAP=0.75
        MODE_REGRET_PER_TOKEN_TV=0.30
        MODE_REGRET_BLOCK_TV=0.85
        MODE_REGRET_BAND_DEBT_SLOPE=0.50
        MODE_REGRET_BAND_DEPTH_SLOPE=0.00
      )
      ;;
    target_band_r8_g100)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
      PROFILE_ENV=(
        MODE_REGRET_ELIGIBILITY=target_band
        MODE_REGRET_MAX_TARGET_RANK=8
        MODE_REGRET_MAX_CANDIDATE_GAP=1.00
        MODE_REGRET_PER_TOKEN_TV=0.35
        MODE_REGRET_BLOCK_TV=0.95
        MODE_REGRET_BAND_DEBT_SLOPE=0.35
        MODE_REGRET_BAND_DEPTH_SLOPE=0.00
      )
      ;;
    prefix_credit_token_cap)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_prefix_credit_mtp.sh"
      PROFILE_ENV=(
        PREFIX_CREDIT_ALLOCATION=token_cap
        PREFIX_CREDIT_MIN_TOP1_MARGIN=0.10
        PREFIX_CREDIT_PER_TOKEN_TV=0.49
        PREFIX_CREDIT_BLOCK_TV=0.75
      )
      ;;
    prefix_credit_atomic)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_prefix_credit_mtp.sh"
      PROFILE_ENV=(
        PREFIX_CREDIT_ALLOCATION=atomic_credit
        PREFIX_CREDIT_MIN_TOP1_MARGIN=0.10
        PREFIX_CREDIT_PER_TOKEN_TV=0.49
        PREFIX_CREDIT_BLOCK_TV=0.75
      )
      ;;
    prefix_credit|prefix_credit_g050|prefix_credit_b075)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_prefix_credit_mtp.sh"
      PROFILE_ENV=(
        PREFIX_CREDIT_ALLOCATION=prefix_credit
        PREFIX_CREDIT_MIN_TOP1_MARGIN=0.10
        PREFIX_CREDIT_PER_TOKEN_TV=0.49
        PREFIX_CREDIT_BLOCK_TV=0.75
        PREFIX_CREDIT_MIN_GAIN_PER_TV=0.50
      )
      ;;
    prefix_credit_g025)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_prefix_credit_mtp.sh"
      PROFILE_ENV=(
        PREFIX_CREDIT_ALLOCATION=prefix_credit
        PREFIX_CREDIT_MIN_TOP1_MARGIN=0.10
        PREFIX_CREDIT_PER_TOKEN_TV=0.49
        PREFIX_CREDIT_BLOCK_TV=0.75
        PREFIX_CREDIT_MIN_GAIN_PER_TV=0.25
      )
      ;;
    prefix_credit_g100)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_prefix_credit_mtp.sh"
      PROFILE_ENV=(
        PREFIX_CREDIT_ALLOCATION=prefix_credit
        PREFIX_CREDIT_MIN_TOP1_MARGIN=0.10
        PREFIX_CREDIT_PER_TOKEN_TV=0.49
        PREFIX_CREDIT_BLOCK_TV=0.75
        PREFIX_CREDIT_MIN_GAIN_PER_TV=1.00
      )
      ;;
    prefix_credit_b060)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_prefix_credit_mtp.sh"
      PROFILE_ENV=(
        PREFIX_CREDIT_ALLOCATION=prefix_credit
        PREFIX_CREDIT_MIN_TOP1_MARGIN=0.10
        PREFIX_CREDIT_PER_TOKEN_TV=0.49
        PREFIX_CREDIT_BLOCK_TV=0.60
        PREFIX_CREDIT_MIN_GAIN_PER_TV=0.50
      )
      ;;
    prefix_credit_b090)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_prefix_credit_mtp.sh"
      PROFILE_ENV=(
        PREFIX_CREDIT_ALLOCATION=prefix_credit
        PREFIX_CREDIT_MIN_TOP1_MARGIN=0.10
        PREFIX_CREDIT_PER_TOKEN_TV=0.49
        PREFIX_CREDIT_BLOCK_TV=0.90
        PREFIX_CREDIT_MIN_GAIN_PER_TV=0.50
      )
      ;;
    scheme1)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme1_mtp.sh"
      ;;
    scheme2)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme2_mtp.sh"
      ;;
    scheme2_relaxed)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme2_relaxed_mtp.sh"
      ;;
    scheme2_strong)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme2_strong_mtp.sh"
      ;;
    scheme2_ultra)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme2_ultra_mtp.sh"
      ;;
    scheme3)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme3_adaptive_chain_mtp.sh"
      ;;
    scheme12)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme12_mtp.sh"
      ;;
    scheme12_joint)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme12_joint_mtp.sh"
      ;;
    scheme12_anchored)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_scheme12_anchored_mtp.sh"
      ;;
    remtp)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_remtp_mtp.sh"
      ;;
    remtp_block)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_remtp_block_mtp.sh"
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
    CACTUS_REGRET_ALPHA="$CACTUS_REGRET_ALPHA" \
    CACTUS_REGRET_TOP_K="$CACTUS_REGRET_TOP_K" \
    CACTUS_REGRET_STRENGTH_REFERENCE="$CACTUS_REGRET_STRENGTH_REFERENCE" \
    CACTUS_REGRET_DEPTH_DECAY="$CACTUS_REGRET_DEPTH_DECAY" \
    CACTUS_REGRET_RESPONSIBILITY="$CACTUS_REGRET_RESPONSIBILITY" \
    CACTUS_REGRET_INJECTION_SITE="$CACTUS_REGRET_INJECTION_SITE" \
    CACTUS_REGRET_RESIDUAL_SPACE="$CACTUS_REGRET_RESIDUAL_SPACE" \
    CACTUS_REGRET_AUDIT_INTERVAL="$CACTUS_REGRET_AUDIT_INTERVAL" \
    CACTUS_REGRET_DIAGNOSTICS="$CACTUS_REGRET_DIAGNOSTICS" \
    BLOCK_SHIELD_CACTUS_MIX="$BLOCK_SHIELD_CACTUS_MIX" \
    REGRET_ROUTER_CHECKPOINT="$REGRET_ROUTER_CHECKPOINT" \
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
    --workers "$EVAL_WORKERS" \
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
  if [[ -L "$RUN_ROOT/$profile" ]]; then
    echo "[reuse] skipping inference and evaluation for $profile"
  else
    run_method "$profile"
  fi
done

python -m remtp.humaneval_compare "$RUN_ROOT" "${profiles[@]}"
echo
echo "HumanEval comparison complete: $RUN_ROOT/comparison.md"
