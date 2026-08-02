#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

METHOD="${DSE_METHOD:-regret}"
SAMPLES="${SAMPLES:-40}"
SAMPLE_SEED="${SAMPLE_SEED:-20260731}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MTP_TOKENS="${MTP_TOKENS:-6}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-180}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
DSE_TAG="${DSE_TAG:-${METHOD}_$(date +%Y%m%d_%H%M%S)}"

if [[ "$MTP_TOKENS" != "6" ]]; then
  echo "Regret DSE requires MTP_TOKENS=6." >&2
  exit 2
fi
export MTP_TOKENS
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi

case "$METHOD" in
  cactus)
    serve_script="$PROJECT_DIR/scripts/serve_cactus_mtp.sh"
    benchmark_script="$PROJECT_DIR/scripts/benchmark_gsm8k_cactus_mtp.sh"
    ;;
  exact_tv)
    serve_script="$PROJECT_DIR/scripts/serve_target_anchored_mtp.sh"
    benchmark_script="$PROJECT_DIR/scripts/benchmark_gsm8k_target_anchored_mtp.sh"
    export TARGET_ANCHORED_VARIANT=tv_hidden_veto
    ;;
  audit_baseline)
    serve_script="$PROJECT_DIR/scripts/serve_regret_feedback_mtp.sh"
    benchmark_script="$PROJECT_DIR/scripts/benchmark_gsm8k_regret_feedback_mtp.sh"
    export REGRET_DIRECTION=none
    ;;
  regret)
    serve_script="$PROJECT_DIR/scripts/serve_regret_feedback_mtp.sh"
    benchmark_script="$PROJECT_DIR/scripts/benchmark_gsm8k_regret_feedback_mtp.sh"
    ;;
  *)
    echo "DSE_METHOD must be cactus, exact_tv, audit_baseline, or regret." >&2
    exit 2
    ;;
esac

RUN_ROOT="$PROJECT_DIR/dse_results/outputs/$DSE_TAG"
RESULT_ROOT="$RUN_ROOT/result"
SERVER_LOG="$RUN_ROOT/server.log"
BENCHMARK_LOG="$RUN_ROOT/benchmark.log"
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
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -n 100 "$SERVER_LOG" >&2
      return 1
    fi
    sleep 2
  done
  tail -n 100 "$SERVER_LOG" >&2
  return 1
}

mkdir -p "$RUN_ROOT"
setsid "$serve_script" >"$SERVER_LOG" 2>&1 &
server_pid=$!
wait_for_server

SAMPLES="$SAMPLES" \
SAMPLE_SEED="$SAMPLE_SEED" \
TEMPERATURE="$TEMPERATURE" \
SEED="$SEED" \
MAX_TOKENS="$MAX_TOKENS" \
MTP_TOKENS="$MTP_TOKENS" \
"$benchmark_script" \
  --base-url "$BASE_URL" \
  --output-dir "$RESULT_ROOT" \
  --progress-every "$PROGRESS_EVERY" \
  2>&1 | tee "$BENCHMARK_LOG"

cleanup_server

python - "$METHOD" "$DSE_TAG" "$RESULT_ROOT/summary.json" <<'PY'
import json
import sys
from pathlib import Path

method, tag, summary_path = sys.argv[1:]
result = json.loads(Path(summary_path).read_text())["results"][0]
payload = {
    "method": method,
    "tag": tag,
    "accuracy": result["accuracy"],
    "decode_tok_s": result["decode_tok_s"],
    "e2e_tok_s": result["e2e_output_tok_s"],
    "mean_acceptance_length": result["mean_acceptance_length"],
    "draft_acceptance": result["draft_token_acceptance_rate"],
    "truncation_rate": result["truncation_rate"],
    "summary": str(Path(summary_path).resolve()),
}
print("DSE_RESULT=" + json.dumps(payload, sort_keys=True))
PY
