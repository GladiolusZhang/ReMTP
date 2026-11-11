#!/usr/bin/env bash
# Show unattended comparison status without opening the running service.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="${RUN_TAG:-raven_jetson_gsm8k_n100_fp4_2048}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"

systemctl --user show remtp-raven-gsm8k.service \
  -p ActiveState -p SubState -p Result --no-pager
if [[ -f "$RUN_ROOT/protocol.json" ]]; then
  echo "Protocol: $RUN_ROOT/protocol.json"
else
  echo "Protocol not started: $RUN_ROOT"
  exit 0
fi
for method in native cactus spec_cascade uniform adaptive; do
  if [[ -f "$RUN_ROOT/$method/summary.json" ]]; then
    echo "$method: complete"
  elif [[ -f "$RUN_ROOT/$method/requests.checkpoint.jsonl" ]]; then
    count="$(wc -l < "$RUN_ROOT/$method/requests.checkpoint.jsonl")"
    echo "$method: $count requests checkpointed"
  else
    echo "$method: pending"
  fi
done
if [[ -f "$RUN_ROOT/COMPLETE" ]]; then
  echo "Completed: $(cat "$RUN_ROOT/COMPLETE")"
  echo "Report: $RUN_ROOT/comparison.md"
fi
