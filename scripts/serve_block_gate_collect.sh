#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${BLOCK_GATE_DATA:-}" ]]; then
  echo "Set BLOCK_GATE_DATA to a local JSONL output path." >&2
  exit 2
fi

export MTP_TOKENS="${MTP_TOKENS:-6}"
export REMTP_WORKER_CLS=remtp.worker.LearnedBlockGateMTPWorker
export REMTP_BLOCK_GATE_MODE=collect
export REMTP_BLOCK_GATE_DATA="$BLOCK_GATE_DATA"
export REMTP_BLOCK_GATE_COLLECT_ACTION="${BLOCK_GATE_COLLECT_ACTION:-cactus}"
export REMTP_BLOCK_GATE_SURPLUS_SPEND_FRACTION="${BLOCK_GATE_SURPLUS_SPEND_FRACTION:-0.5}"
export REMTP_BLOCK_GATE_DIAGNOSTICS="${BLOCK_GATE_DIAGNOSTICS:-0}"
export REMTP_BLOCK_VERIFY_DIAGNOSTICS=0
export TARGET_ANCHORED_VARIANT=tv_block_shield

exec "$PROJECT_DIR/scripts/serve_target_anchored_mtp.sh"
