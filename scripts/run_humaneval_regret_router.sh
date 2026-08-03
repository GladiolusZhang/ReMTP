#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MTP_TOKENS=6
export TEMPERATURE="${TEMPERATURE:-0.7}"
export SAMPLES="${SAMPLES:-164}"
export PROFILES="${PROFILES:-native_mtp cactus spec_cascade tv_head tv_hidden_veto exact_tv exact_tv_regret_router}"
export REGRET_ROUTER_CHECKPOINT="${REGRET_ROUTER_CHECKPOINT:-$PROJECT_DIR/checkpoints/regret_router.pt}"

if [[ " $PROFILES " == *" exact_tv_regret_router "* && ! -f "$REGRET_ROUTER_CHECKPOINT" ]]; then
  echo "Missing trained router: $REGRET_ROUTER_CHECKPOINT" >&2
  echo "Run ./scripts/collect_regret_router.sh and ./scripts/train_regret_router.sh first." >&2
  exit 2
fi

exec "$PROJECT_DIR/scripts/run_humaneval_comparison.sh" "$@"
