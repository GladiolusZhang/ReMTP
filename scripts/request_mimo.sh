#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-XiaomiMiMo/MiMo-7B-Base}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-128}"
SEED="${SEED:-42}"
PROMPT="${PROMPT:-Explain speculative decoding in three concise sentences.}"

python - "$BASE_URL" "$MODEL" "$TEMPERATURE" "$MAX_TOKENS" "$SEED" "$PROMPT" <<'PY'
import json
import sys
from urllib.request import Request, urlopen

base_url, model, temperature, max_tokens, seed, prompt = sys.argv[1:]
payload = json.dumps({
    "model": model,
    "prompt": prompt,
    "temperature": float(temperature),
    "max_tokens": int(max_tokens),
    "seed": int(seed),
}).encode()
request = Request(
    base_url.rstrip("/") + "/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=300) as response:
    result = json.load(response)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
