#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${BLOCK_GATE_MODEL:-}" || ! -f "$BLOCK_GATE_MODEL" ]]; then
  echo "Set BLOCK_GATE_MODEL to a trained local gate JSON file." >&2
  exit 2
fi

export MTP_TOKENS="${MTP_TOKENS:-6}"
export REMTP_WORKER_CLS=remtp.worker.LearnedBlockGateMTPWorker
export REMTP_BLOCK_GATE_MODE=infer
export REMTP_BLOCK_GATE_MODEL="$BLOCK_GATE_MODEL"
export REMTP_BLOCK_GATE_DIAGNOSTICS="${BLOCK_GATE_DIAGNOSTICS:-0}"
export REMTP_BLOCK_VERIFY_DIAGNOSTICS=0
export TARGET_ANCHORED_VARIANT=tv_block_shield

exec "$PROJECT_DIR/scripts/serve_target_anchored_mtp.sh"
