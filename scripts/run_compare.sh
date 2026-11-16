#!/usr/bin/env bash
# Run the compared methods one after another on the same samples, seeds and
# budget, judge code in Docker, and print one table per dataset (2026-11-15/16).
#
#   scripts/run_compare.sh                                   # AR, Native, Cactus, RAVEN; HumanEval 100
#   DATASETS="humaneval gsm8k mbpp" SAMPLES=50 MAX_TOKENS=2048 ASYNC_SCHED=1 scripts/run_compare.sh
#   METHODS="cactus raven" SAMPLES=30 scripts/run_compare.sh
#
# Each method gets one server (scripts/serve_compare.sh) for all datasets.
# Outputs: $OUT/<method>/server.log, $OUT/<method>/<dataset>/{summary.json,...}
# and $OUT/compare.md.  Environment: METHODS, DATASETS (humaneval gsm8k mbpp),
# SAMPLES, MAX_TOKENS, TEMPERATURE, SAMPLE_SEED (HumanEval), GSM8K_SAMPLE_SEED,
# MBPP_SAMPLE_SEED, GENERATION_SEED, OUT, PORT, SKIP_EVAL=1, plus serve_compare.sh's.
# Re-running the same command resumes: finished datasets are skipped and a
# partial one continues from its checkpoint.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

METHODS="${METHODS:-target native cactus raven}"
DATASETS="${DATASETS:-humaneval}"
SAMPLES="${SAMPLES:-100}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.6}"
SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
GSM8K_SAMPLE_SEED="${GSM8K_SAMPLE_SEED:-20260730}"
MBPP_SAMPLE_SEED="${MBPP_SAMPLE_SEED:-20261116}"
GENERATION_SEED="${GENERATION_SEED:-42}"
export PORT="${PORT:-8000}"
OUT="${OUT:-results/compare_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
HUMANEVAL_DATA="data/humaneval/HumanEval.jsonl.gz"
GSM8K_DATA="data/gsm8k/test.jsonl"
MBPP_DATA="data/regret_router/benchmarks/mbpp.jsonl"
for dataset in $DATASETS; do
  case "$dataset" in
    humaneval) [[ -f "$HUMANEVAL_DATA" ]] || ./scripts/download_humaneval.sh ;;
    gsm8k|mbpp)
      [[ -f "$GSM8K_DATA" && -f "$MBPP_DATA" ]] || python -m remtp.router_benchmark_download \
        --output-dir data/regret_router/benchmarks
      [[ -f "$GSM8K_DATA" ]] || GSM8K_DATA="data/regret_router/benchmarks/gsm8k_test.jsonl" ;;
    *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
  esac
done

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

generate() {  # $1 dataset, $2 method, $3 output dir
  local dataset="$1" method="$2" run="$3"
  local -a resume=()
  [[ -d "$run" ]] && resume=(--resume)
  local -a common=(--base-url "http://127.0.0.1:$PORT" --model TencentBAC/FastMTP --run-name "$method"
                   --samples "$SAMPLES" --temperature "$TEMPERATURE" --generation-seed "$GENERATION_SEED"
                   --max-tokens "$MAX_TOKENS" --mtp-tokens 3 --output-dir "$run" --timeout 3600 "${resume[@]}")
  case "$dataset" in
    humaneval) python -m remtp.humaneval_benchmark --data "$HUMANEVAL_DATA" --sample-seed "$SAMPLE_SEED" "${common[@]}" ;;
    gsm8k) python -m remtp.gsm8k_benchmark --data "$GSM8K_DATA" --sample-seed "$GSM8K_SAMPLE_SEED" "${common[@]}" ;;
    mbpp) python -m remtp.mbpp_benchmark --data "$MBPP_DATA" --sample-seed "$MBPP_SAMPLE_SEED" "${common[@]}" ;;
  esac
}

evaluate() {  # $1 dataset, $2 output dir
  case "$1" in
    humaneval) python -m remtp.humaneval_evaluator "$2" --data "$HUMANEVAL_DATA" ;;
    mbpp) python -m remtp.mbpp_evaluator "$2" --data "$MBPP_DATA" ;;
    gsm8k) ;;   # graded during generation (final #### answer)
  esac
}

for method in $METHODS; do
  dir="$OUT/$method"
  mkdir -p "$dir"
  todo=()
  for dataset in $DATASETS; do
    [[ -f "$dir/$dataset/summary.json" ]] || todo+=("$dataset")
  done
  if (( ${#todo[@]} == 0 )); then
    echo "[compare] $method already done, skipping"
  else
    echo "[compare] starting $method server (datasets: ${todo[*]})"
    setsid scripts/serve_compare.sh "$method" >> "$dir/server.log" 2>&1 &
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
      "$dir/server.log" | sed 's/^/[compare]   /' | cut -c1-200 | tail -4 || true
    for dataset in "${todo[@]}"; do
      echo "[compare] $method / $dataset"
      generate "$dataset" "$method" "$dir/$dataset"
    done
    stop_server
  fi
  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    for dataset in $DATASETS; do
      summary="$dir/$dataset/summary.json"
      [[ -f "$summary" ]] || continue
      grep -q '"evaluation_status": "complete"' "$summary" && continue
      [[ "$dataset" == "gsm8k" ]] && continue
      evaluate "$dataset" "$dir/$dataset" || echo "[compare] evaluation failed for $method/$dataset" >&2
    done
  fi
done

python - "$OUT" "$DATASETS" $METHODS <<'PY'
import json, sys
from pathlib import Path
out, datasets, methods = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3:]
fmt = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"
sections = []
for dataset in datasets:
    rows, ar = [], None
    for m in methods:
        path = out / m / dataset / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())["results"][0]
        rounds = s.get("draft_rounds") or s.get("server_output_tokens")
        quality = s.get("accuracy") if dataset == "gsm8k" else s.get("pass_at_1")
        rows.append((m, quality, s.get("mean_acceptance_length"),
                     s["decode_seconds"] / rounds if rounds else None, s.get("decode_tok_s"),
                     s.get("e2e_output_tok_s"), s.get("truncated_outputs"), s.get("output_tokens")))
        if m == "target":
            ar = s.get("decode_tok_s")
    metric = "accuracy" if dataset == "gsm8k" else "pass@1"
    lines = [f"### {dataset}", "",
             f"| method | {metric} | MAL | s/round | decode tok/s | E2E tok/s | output tokens | truncated | decode vs AR |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m, q, mal, spr, dec, e2e, trunc, toks in rows:
        lines.append(f"| {m} | {fmt(q)} | {fmt(mal)} | {fmt(spr, 4)} | {fmt(dec, 2)} | {fmt(e2e, 2)} | {toks} | "
                     f"{trunc} | {fmt(dec / ar, 2) + 'x' if ar and dec else 'n/a'} |")
    sections.append("\n".join(lines))
text = "\n\n".join(sections)
(out / "compare.md").write_text(text + "\n")
print(text)
PY
