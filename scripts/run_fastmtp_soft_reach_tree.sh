#!/usr/bin/env bash
# Public name for the quota-free soft-reach D=3 experiment.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$PROJECT_DIR/scripts/run_fastmtp_reach_first_tree.sh" "$@"
