#!/usr/bin/env bash
set -euo pipefail

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-64}"

curl --fail-with-body --silent --show-error \
  http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @- <<JSON
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": "Explain speculative decoding in three short sentences."
    }
  ],
  "chat_template_kwargs": {
    "enable_thinking": false
  },
  "temperature": ${TEMPERATURE},
  "seed": ${SEED},
  "max_tokens": ${MAX_TOKENS}
}
JSON
printf '\n'
