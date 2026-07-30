#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

variant="${TARGET_ANCHORED_VARIANT:-tv_hidden_veto}"
delta="${CACTUS_DELTA:-1.0}"

RUN_NAME="target_anchored_${variant}_cactus_delta${delta}" \
MTP_TOKENS="${MTP_TOKENS:-4}" \
exec "$PROJECT_DIR/scripts/benchmark_gsm8k.sh" "$@"
