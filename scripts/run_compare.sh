#!/usr/bin/env bash
# Run the compared methods one after another on the same HumanEval sample, seeds
# and budget, judge with the Docker evaluator, and print one table (2026-11-15).
#
#   scripts/run_compare.sh                       # AR, Native, Cactus, RAVEN; 100 tasks
#   METHODS="cactus raven" SAMPLES=30 scripts/run_compare.sh
#
# Each method gets its own server (scripts/serve_compare.sh), started and stopped
# here.  Outputs: $OUT/<method>/server.log, $OUT/<method>/humaneval/{summary.json,...}
# and $OUT/compare.md.  Environment: METHODS, SAMPLES, MAX_TOKENS, TEMPERATURE,
# SAMPLE_SEED, GENERATION_SEED, OUT, PORT, SKIP_EVAL=1, plus serve_compare.sh's.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

METHODS="${METHODS:-target native cactus raven}"
SAMPLES="${SAMPLES:-100}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.6}"
SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
GENERATION_SEED="${GENERATION_SEED:-42}"
export PORT="${PORT:-8000}"
OUT="${OUT:-results/compare_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
DATA="data/humaneval/HumanEval.jsonl.gz"
[[ -f "$DATA" ]] || ./scripts/download_humaneval.sh

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 60); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$server_pid" 2>/dev/null || true
  fi
  server_pid=""
  for _ in $(seq 60); do
    (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null || return 0
    sleep 1
  done
}
trap stop_server EXIT

for method in $METHODS; do
  dir="$OUT/$method"
  mkdir -p "$dir"
  run="$dir/humaneval"
  if [[ -f "$run/summary.json" ]]; then
    echo "[compare] $method already done, skipping"
    continue
  fi
  echo "[compare] starting $method server"
  setsid scripts/serve_compare.sh "$method" > "$dir/server.log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 600); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[compare] $method server exited; see $dir/server.log" >&2
      exit 1
    fi
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 3
  done
  [[ "$ready" == "1" ]] || { echo "[compare] $method server not healthy" >&2; exit 1; }
  grep -h "\[ReMTP\]\[Compare\]\|Graph capturing finished\|device-side tree construction\|target_gate_min_p" \
    "$dir/server.log" | sed 's/^/[compare]   /' | cut -c1-200 || true
  python -m remtp.humaneval_benchmark \
    --data "$DATA" --base-url "http://127.0.0.1:$PORT" --model TencentBAC/FastMTP \
    --run-name "$method" --samples "$SAMPLES" --sample-seed "$SAMPLE_SEED" \
    --temperature "$TEMPERATURE" --generation-seed "$GENERATION_SEED" \
    --max-tokens "$MAX_TOKENS" --mtp-tokens 3 --output-dir "$run" --timeout 3600
  stop_server
  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    python -m remtp.humaneval_evaluator "$run" || echo "[compare] evaluation failed for $method" >&2
  fi
done

python - "$OUT" $METHODS <<'PY'
import json, sys
from pathlib import Path
out, methods = Path(sys.argv[1]), sys.argv[2:]
rows, ar = [], None
for m in methods:
    path = out / m / "humaneval" / "summary.json"
    if not path.exists():
        continue
    s = json.loads(path.read_text())["results"][0]
    rounds = s.get("draft_rounds") or s.get("server_output_tokens")
    rows.append((m, s.get("pass_at_1"), s.get("mean_acceptance_length"),
                 s["decode_seconds"] / rounds if rounds else None, s.get("decode_tok_s"), s.get("e2e_output_tok_s"),
                 s.get("truncated_outputs")))
    if m == "target":
        ar = s.get("decode_tok_s")
fmt = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"
lines = ["| method | pass@1 | MAL | s/round | decode tok/s | E2E tok/s | truncated | decode vs AR |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for m, p, mal, spr, dec, e2e, trunc in rows:
    lines.append(f"| {m} | {fmt(p)} | {fmt(mal)} | {fmt(spr, 4)} | {fmt(dec, 2)} | {fmt(e2e, 2)} | {trunc} | "
                 f"{fmt(dec / ar, 2) + 'x' if ar and dec else 'n/a'} |")
text = "\n".join(lines)
(out / "compare.md").write_text(text + "\n")
print(text)
PY
