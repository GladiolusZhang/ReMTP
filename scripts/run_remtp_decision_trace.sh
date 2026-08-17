#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export METHOD=remtp
exec "$PROJECT_DIR/scripts/run_proposal_decision_trace.sh" "$@"
