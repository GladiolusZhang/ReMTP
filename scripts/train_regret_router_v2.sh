#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

TRACE_DIR="${REGRET_ROUTER_TRACE_DIR:-$PROJECT_DIR/data/regret_router/traces_pilot}"
CHECKPOINT="${REGRET_ROUTER_CHECKPOINT:-$PROJECT_DIR/checkpoints/regret_router_v2.pt}"

python -m remtp.regret_router_selective_train \
  --data-dir "$TRACE_DIR" \
  --output "$CHECKPOINT" \
  --device "${DEVICE:-cuda}" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --debt-reference "${REGRET_ROUTER_DEBT_REFERENCE:-0.05}" \
  --oracle-grid-size "${ORACLE_GRID_SIZE:-17}" \
  --min-oracle-gain "${MIN_ORACLE_GAIN:-1e-4}" \
  --min-validation-gain "${MIN_VALIDATION_GAIN:-1e-4}" \
  --min-sign-accuracy "${MIN_SIGN_ACCURACY:-0.55}" \
  --min-action-scale "${MIN_ACTION_SCALE:-0.01}" \
  --patience "${PATIENCE:-6}" \
  --seed "${SEED:-42}" \
  "$@"
