#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${HUMANEVAL_DATA:-$PROJECT_DIR/data/humaneval/HumanEval.jsonl.gz}"
URL="${HUMANEVAL_URL:-https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz}"

if [[ -e "$DESTINATION" && "${FORCE:-0}" != "1" ]]; then
  echo "HumanEval already exists: $DESTINATION"
  echo "Set FORCE=1 to replace it."
  exit 0
fi

mkdir -p "$(dirname "$DESTINATION")"
download_tmp="$(mktemp "${DESTINATION}.download.XXXXXX")"
cleanup() {
  rm -f "$download_tmp"
}
trap cleanup EXIT

echo "Downloading official HumanEval data"
echo "source=$URL"
echo "destination=$DESTINATION"
curl --fail --location --retry 5 --retry-delay 2 \
  --output "$download_tmp" "$URL"

PROJECT_DIR="$PROJECT_DIR" DOWNLOAD_TMP="$download_tmp" \
  "$PROJECT_DIR/.venv/bin/python" - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["PROJECT_DIR"])
from remtp.humaneval_benchmark import load_humaneval

path = Path(os.environ["DOWNLOAD_TMP"])
rows = load_humaneval(path)
if len(rows) != 164:
    raise SystemExit(f"expected 164 official tasks, found {len(rows)}")
print(f"Validated {len(rows)} HumanEval tasks")
PY

mv "$download_tmp" "$DESTINATION"
trap - EXIT
echo "Ready: $DESTINATION"
