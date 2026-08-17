#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MTP_TOKENS="${MTP_TOKENS:-6}"
export MTP_REJECTION_SAMPLE_METHOD=probabilistic
export REMTP_WORKER_CLS=remtp.worker.ProposalCalibratedMTPWorker
export REMTP_TRACE=0

exec "$PROJECT_DIR/scripts/serve_probabilistic_mtp.sh"
