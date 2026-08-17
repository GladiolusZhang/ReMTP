#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RISK_VARIANT=scheme12
exec "$PROJECT_DIR/scripts/serve_risk_entropy_mtp.sh"
