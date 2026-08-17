#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Formal single-GPU comparison over MTP lengths 2/4/6/8.

Default matrix:
  datasets: GSM8K 500 examples + HumanEval 164 tasks
  methods : native MTP, Cactus, SpecCascade TokenV3, Proposal-Calibrated MTP
  depths  : 2, 4, 6, 8

Recommended overnight run:
  ./scripts/run_formal_proposal_calibration_depths.sh

Useful overrides:
  MTP_LENGTHS="2 4 6 8"
  METHODS="native_mtp cactus spec_cascade proposal_calibrated"
  DATASETS="gsm8k humaneval"
  GSM8K_SAMPLES=500
  HUMANEVAL_SAMPLES=164
  TEMPERATURE=0.7
  SEED=42
  RUN_ROOT=/absolute/or/relative/result/path   # resume the same matrix
  CONTINUE_ON_ERROR=1

Quick infrastructure smoke:
  MTP_LENGTHS="2 8" METHODS="native_mtp proposal_calibrated" \
  GSM8K_SAMPLES=2 HUMANEVAL_SAMPLES=2 \
  ./scripts/run_formal_proposal_calibration_depths.sh

The script regenerates one RUN_ROOT/comparison.md after every completed run.
Existing complete summary files are reused; partial attempt directories are
left intact for diagnosis and never treated as finished results.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if (( $# > 0 )); then echo "Unknown argument: $1" >&2; usage >&2; exit 2; fi

read -r -a depths <<< "${MTP_LENGTHS:-2 4 6 8}"
read -r -a methods <<< "${METHODS:-native_mtp cactus spec_cascade proposal_calibrated}"
read -r -a datasets <<< "${DATASETS:-gsm8k humaneval}"

GSM8K_SAMPLES="${GSM8K_SAMPLES:-500}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-164}"
GSM8K_SAMPLE_SEED="${GSM8K_SAMPLE_SEED:-20260730}"
HUMANEVAL_SAMPLE_SEED="${HUMANEVAL_SAMPLE_SEED:-20260802}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-384}"
HUMANEVAL_MAX_TOKENS="${HUMANEVAL_MAX_TOKENS:-512}"
CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
CASCADE_RULE="${CASCADE_RULE:-token_v3}"
CASCADE_ALPHA="${CASCADE_ALPHA:-0.5}"
PROPOSAL_CONFIG="${PROPOSAL_CONFIG:-$PROJECT_DIR/configs/proposal_calibration_static.json}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-240}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-8}"
EVAL_MEMORY="${EVAL_MEMORY:-512m}"
EVAL_CPUS="${EVAL_CPUS:-1.0}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
HUMANEVAL_DOCKER_IMAGE="${HUMANEVAL_DOCKER_IMAGE:-python:3-slim}"
HUMANEVAL_DATA="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/proposal_calibration_depths_${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/proposal_calibration_depths_${RUN_TAG}}"

contains() {
  local needle="$1"; shift
  local item
  for item in "$@"; do [[ "$item" == "$needle" ]] && return 0; done
  return 1
}

for depth in "${depths[@]}"; do
  [[ "$depth" =~ ^[0-9]+$ ]] && (( depth > 0 )) || {
    echo "Invalid MTP depth: $depth" >&2; exit 2;
  }
done
for method in "${methods[@]}"; do
  contains "$method" native_mtp cactus spec_cascade proposal_calibrated || {
    echo "Unknown method: $method" >&2; exit 2;
  }
done
for dataset in "${datasets[@]}"; do
  contains "$dataset" gsm8k humaneval || {
    echo "Unknown dataset: $dataset" >&2; exit 2;
  }
done

[[ -f "$PROJECT_DIR/data/gsm8k/test.jsonl" ]] || {
  if contains gsm8k "${datasets[@]}"; then echo "Missing data/gsm8k/test.jsonl" >&2; exit 2; fi
}
[[ -f "$PROPOSAL_CONFIG" ]] || { echo "Missing proposal config: $PROPOSAL_CONFIG" >&2; exit 2; }
if contains humaneval "${datasets[@]}"; then
  [[ -f "$HUMANEVAL_DATA" ]] || { echo "Missing HumanEval data: $HUMANEVAL_DATA" >&2; exit 2; }
  docker info >/dev/null 2>&1 || { echo "Docker daemon unavailable; run scripts/setup_humaneval.sh" >&2; exit 2; }
  docker image inspect "$HUMANEVAL_DOCKER_IMAGE" >/dev/null 2>&1 || {
    echo "Missing Docker image $HUMANEVAL_DOCKER_IMAGE; run scripts/setup_humaneval.sh" >&2; exit 2;
  }
fi
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi

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
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before health check: $log_file" >&2
      tail -n 160 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "Server startup timeout: $log_file" >&2
  tail -n 160 "$log_file" >&2
  return 1
}

profile_server() {
  local method="$1"
  PROFILE_ENV=(
    REMTP_TRACE=0
    REMTP_PROB_MTP_DIAGNOSTICS=0
    REMTP_CACTUS_DIAGNOSTICS=0
    REMTP_CASCADE_DIAGNOSTICS=0
  )
  case "$method" in
    native_mtp) PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh" ;;
    cactus)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_cactus_mtp.sh"
      PROFILE_ENV+=(CACTUS_DELTA="$CACTUS_DELTA")
      ;;
    spec_cascade)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_spec_cascade.sh"
      PROFILE_ENV+=(CASCADE_RULE="$CASCADE_RULE" CASCADE_ALPHA="$CASCADE_ALPHA")
      ;;
    proposal_calibrated)
      PROFILE_SCRIPT="$PROJECT_DIR/scripts/serve_proposal_calibrated_mtp.sh"
      PROFILE_ENV+=(REMTP_PC_CONFIG="$PROPOSAL_CONFIG")
      ;;
  esac
}

is_generation_complete() {
  [[ -s "$1/summary.json" && -s "$1/requests.jsonl" && -s "$1/sample_manifest.json" ]]
}

is_humaneval_complete() {
  local path="$1/summary.json"
  [[ -s "$path" ]] || return 1
  python - "$path" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
rows=payload.get("results", [])
raise SystemExit(0 if len(rows)==1 and rows[0].get("evaluation_status")=="complete" else 1)
PY
}

run_generation() {
  local dataset="$1" depth="$2" method="$3" final_dir="$4" generation_log="$5"
  local attempt_dir="$final_dir.attempt_${RUN_TAG}_$$"
  if [[ -e "$final_dir" ]]; then
    echo "Incomplete final directory blocks resume: $final_dir" >&2
    return 1
  fi
  if [[ "$dataset" == "gsm8k" ]]; then
    RUN_NAME="$method" SAMPLES="$GSM8K_SAMPLES" SAMPLE_SEED="$GSM8K_SAMPLE_SEED" \
    TEMPERATURE="$TEMPERATURE" SEED="$SEED" MAX_TOKENS="$GSM8K_MAX_TOKENS" \
    MTP_TOKENS="$depth" "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" \
      --base-url "$BASE_URL" --output-dir "$attempt_dir" --progress-every "$PROGRESS_EVERY" \
      2>&1 | tee "$generation_log" || return 1
  else
    RUN_NAME="$method" HUMANEVAL_DATA="$HUMANEVAL_DATA" SAMPLES="$HUMANEVAL_SAMPLES" \
    SAMPLE_SEED="$HUMANEVAL_SAMPLE_SEED" TEMPERATURE="$TEMPERATURE" SEED="$SEED" \
    MAX_TOKENS="$HUMANEVAL_MAX_TOKENS" MTP_TOKENS="$depth" \
      "$PROJECT_DIR/scripts/benchmark_humaneval.sh" \
      --base-url "$BASE_URL" --output-dir "$attempt_dir" --progress-every "$PROGRESS_EVERY" \
      2>&1 | tee "$generation_log" || return 1
  fi
  mv "$attempt_dir" "$final_dir"
}

evaluate_humaneval() {
  local directory="$1" evaluation_log="$2"
  if is_humaneval_complete "$directory"; then
    echo "[resume] HumanEval evaluation complete: $directory"
    return 0
  fi
  python -m remtp.humaneval_evaluator "$directory" \
    --data "$HUMANEVAL_DATA" --image "$HUMANEVAL_DOCKER_IMAGE" \
    --timeout "$EVAL_TIMEOUT" --memory "$EVAL_MEMORY" --cpus "$EVAL_CPUS" \
    --workers "$EVAL_WORKERS" --progress-every "$PROGRESS_EVERY" \
    2>&1 | tee "$evaluation_log"
}

aggregate() {
  python -m remtp.proposal_depth_compare \
    --root "$RUN_ROOT" --depths "${depths[@]}" --methods "${methods[@]}" \
    --datasets "${datasets[@]}"
}

run_combination() {
  local depth="$1" method="$2"
  local method_root="$RUN_ROOT/mtp_${depth}/$method"
  local log_root="$LOG_ROOT/mtp_${depth}/$method"
  local gsm_dir="$method_root/gsm8k" human_dir="$method_root/humaneval"
  mkdir -p "$method_root" "$log_root"

  local need_gsm=0 need_human=0
  if contains gsm8k "${datasets[@]}" && ! is_generation_complete "$gsm_dir"; then need_gsm=1; fi
  if contains humaneval "${datasets[@]}" && ! is_generation_complete "$human_dir"; then need_human=1; fi

  if (( need_gsm || need_human )); then
    profile_server "$method"
    echo
    echo "===== MTP=$depth method=$method ====="
    setsid env MTP_TOKENS="$depth" "${PROFILE_ENV[@]}" "$PROFILE_SCRIPT" \
      >"$log_root/server.log" 2>&1 &
    server_pid=$!
    wait_for_server "$log_root/server.log" || { cleanup_server; return 1; }
    if contains gsm8k "${datasets[@]}"; then
      if (( need_gsm )); then
        run_generation gsm8k "$depth" "$method" "$gsm_dir" "$log_root/gsm8k.log" || {
          cleanup_server; return 1;
        }
      else
        echo "[resume] GSM8K generation complete: $gsm_dir"
      fi
    fi
    if contains humaneval "${datasets[@]}"; then
      if (( need_human )); then
        run_generation humaneval "$depth" "$method" "$human_dir" "$log_root/humaneval_generation.log" || {
          cleanup_server; return 1;
        }
      else
        echo "[resume] HumanEval generation complete: $human_dir"
      fi
    fi
    cleanup_server
  else
    echo "[resume] generation complete: MTP=$depth method=$method"
  fi

  if contains humaneval "${datasets[@]}"; then
    evaluate_humaneval "$human_dir" "$log_root/humaneval_evaluation.log" || return 1
  fi
  aggregate
}

echo "Formal Proposal-Calibrated MTP depth matrix"
echo "depths=${depths[*]}"
echo "methods=${methods[*]}"
echo "datasets=${datasets[*]}"
echo "gsm8k_samples=$GSM8K_SAMPLES humaneval_samples=$HUMANEVAL_SAMPLES"
echo "temperature=$TEMPERATURE seed=$SEED"
echo "proposal_config=$PROPOSAL_CONFIG"
echo "results=$RUN_ROOT"

failures=()
aggregate
for depth in "${depths[@]}"; do
  for method in "${methods[@]}"; do
    if ! run_combination "$depth" "$method"; then
      failures+=("mtp_${depth}/$method")
      echo "[FAILED] MTP=$depth method=$method" >&2
      cleanup_server
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit 1; fi
    fi
  done
done
aggregate

echo
echo "Unified report: $RUN_ROOT/comparison.md"
if (( ${#failures[@]} > 0 )); then
  echo "Failed combinations: ${failures[*]}" >&2
  exit 1
fi
echo "All requested combinations completed."
