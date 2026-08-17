#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

SAMPLES="${SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-42}"
SAMPLE_SEED="${SAMPLE_SEED:-20260730}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/fixed6_microtree_pilot_$RUN_TAG"
LOG_ROOT="$PROJECT_DIR/logs/fixed6_microtree_pilot_$RUN_TAG"

if (( SAMPLES > 32 || MAX_TOKENS > 256 )); then
  echo "Fixed-6 has not passed its gate; pilot is limited to <=32 samples and <=256 tokens." >&2
  exit 2
fi
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "A server is already responding at $BASE_URL; stop it first." >&2
  exit 2
fi
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap cleanup EXIT INT TERM

wait_server() {
  local log_file="$1"
  for _ in $(seq 1 90); do
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

for topology in 6-chain 4+2 3+2+1; do
  safe_name="${topology//+/_}"
  safe_name="${safe_name//-/_}"
  server_log="$LOG_ROOT/${safe_name}_server.log"
  audit="$RUN_ROOT/${safe_name}_rounds.jsonl"
  echo "===== Fixed-6 pilot topology=$topology ====="
  setsid env \
    REMTP_ALLOW_UNVERIFIED_GDN_TREE=1 \
    TREE_TOPOLOGY="$topology" \
    TREE_AUDIT_JSONL="$audit" \
    "$PROJECT_DIR/scripts/serve_fixed6_microtree.sh" >"$server_log" 2>&1 &
  server_pid=$!
  wait_server "$server_log"

  RUN_NAME="${safe_name}_strict" \
  SAMPLES="$SAMPLES" \
  SAMPLE_SEED="$SAMPLE_SEED" \
  TEMPERATURE="$TEMPERATURE" \
  SEED="$SEED" \
  MAX_TOKENS="$MAX_TOKENS" \
  MTP_TOKENS=6 \
  "$PROJECT_DIR/scripts/benchmark_gsm8k_risk_entropy.sh" \
    --base-url "$BASE_URL" \
    --output-dir "$RUN_ROOT/$safe_name" \
    --progress-every 2
  cleanup
done

python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for name in ("6_chain", "4_2", "3_2_1"):
    payload = json.loads((root / name / "summary.json").read_text())
    result = payload["results"][0]
    rows.append((name, result))
baseline = rows[0][1]
lines = [
    "# Fixed-6 topology pilot",
    "",
    "| topology | MAL | delta vs chain | decode tok/s | E2E tok/s |",
    "|---|---:|---:|---:|---:|",
]
for name, result in rows:
    delta = result["mean_acceptance_length"] - baseline["mean_acceptance_length"]
    lines.append(
        f"| {name} | {result['mean_acceptance_length']:.3f} | {delta:+.3f} | "
        f"{result['decode_tok_s']:.3f} | {result['e2e_output_tok_s']:.3f} |"
    )
passed = all(
    result["mean_acceptance_length"] >= baseline["mean_acceptance_length"] + 0.10
    for _, result in rows[1:]
)
lines.extend(["", f"Strict-tree MAL gate passed: **{passed}**", ""])
(root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Results: {root / 'comparison.md'}")
PY
