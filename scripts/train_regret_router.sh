#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

TRACE_DIR="${REGRET_ROUTER_TRACE_DIR:-$PROJECT_DIR/data/regret_router/traces}"
CHECKPOINT="${REGRET_ROUTER_CHECKPOINT:-$PROJECT_DIR/checkpoints/regret_router.pt}"

python -m remtp.regret_router_train \
  --data-dir "$TRACE_DIR" \
  --output "$CHECKPOINT" \
  --device "${DEVICE:-cuda}" \
  --epochs "${EPOCHS:-10}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --learning-rate "${LEARNING_RATE:-3e-4}" \
  --debt-reference "${REGRET_ROUTER_DEBT_REFERENCE:-0.05}" \
  --seed "${SEED:-42}" \
  "$@"
