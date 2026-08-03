#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

RAW_CORPUS="${ROUTER_CORPUS_RAW:-$PROJECT_DIR/data/regret_router/router_corpus_raw.jsonl}"

if [[ ! -f "$RAW_CORPUS" ]]; then
  echo "Raw router corpus is missing; building ${ROUTER_CORPUS_PROFILE:-pilot} corpus."
  ROUTER_CORPUS_RAW="$RAW_CORPUS" \
    ROUTER_CORPUS_PROFILE="${ROUTER_CORPUS_PROFILE:-pilot}" \
    "$PROJECT_DIR/scripts/build_regret_router_corpus.sh"
else
  echo "Using existing raw router corpus: $RAW_CORPUS"
fi

"$PROJECT_DIR/scripts/download_router_benchmarks.sh"

ROUTER_CORPUS_RAW="$RAW_CORPUS" \
  "$PROJECT_DIR/scripts/decontaminate_regret_router_corpus.sh"

echo
echo "Router data is ready:"
echo "  data/regret_router/router_corpus.jsonl"
echo "  data/regret_router/decontamination_report.json"
echo "  data/regret_router/benchmarks/manifest.json"
