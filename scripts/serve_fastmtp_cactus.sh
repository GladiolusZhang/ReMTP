#!/usr/bin/env bash
# Serve FastMTP with Cactus relaxation
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_CLS="remtp.fastmtp_worker.FastMTPCactusWorker"
exec "$PROJECT_DIR/scripts/serve_fastmtp_native.sh"
