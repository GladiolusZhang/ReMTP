#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

if ! python -c 'import datasketch' >/dev/null 2>&1; then
  echo "Missing datasketch. Install with: uv pip install -U datasketch" >&2
  exit 2
fi

CORPUS="${ROUTER_CORPUS_RAW:-$PROJECT_DIR/data/regret_router/router_corpus_raw.jsonl}"
OUTPUT="${ROUTER_CORPUS:-$PROJECT_DIR/data/regret_router/router_corpus.jsonl}"
REPORT="${ROUTER_DECONTAMINATION_REPORT:-$PROJECT_DIR/data/regret_router/decontamination_report.json}"
ALLOW_PARTIAL="${ALLOW_PARTIAL_DECONTAMINATION:-0}"
BENCHMARK_DIR="${ROUTER_BENCHMARK_DIR:-$PROJECT_DIR/data/regret_router/benchmarks}"

declare -a BENCHMARKS=()
add_benchmark() {
  local name="$1"
  local path="$2"
  local required="$3"
  if [[ -f "$path" ]]; then
    BENCHMARKS+=(--benchmark "$name=$path")
  elif [[ "$required" == "1" && "$ALLOW_PARTIAL" != "1" ]]; then
    echo "Missing required decontamination benchmark: $name ($path)" >&2
    exit 2
  else
    echo "Warning: skipping unavailable benchmark $name ($path)" >&2
  fi
}

add_benchmark humaneval \
  "${HUMANEVAL_DATA:-$BENCHMARK_DIR/HumanEval.jsonl.gz}" 1
add_benchmark humaneval_plus \
  "${HUMANEVAL_PLUS_DATA:-$BENCHMARK_DIR/HumanEvalPlus.jsonl.gz}" 1
add_benchmark mbpp "${MBPP_DATA:-$BENCHMARK_DIR/mbpp.jsonl}" 1
add_benchmark mbpp_plus \
  "${MBPP_PLUS_DATA:-$BENCHMARK_DIR/MbppPlus.jsonl.gz}" 1
add_benchmark gsm8k "${GSM8K_DATA:-$BENCHMARK_DIR/gsm8k_test.jsonl}" 1
add_benchmark ifeval \
  "${IFEVAL_DATA:-$BENCHMARK_DIR/ifeval_input_data.jsonl}" 1

python -m remtp.router_decontaminate \
  --corpus "$CORPUS" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  "${BENCHMARKS[@]}" \
  "$@"
