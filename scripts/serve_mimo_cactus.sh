#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD=cactus exec "$PROJECT_DIR/scripts/serve_mimo_method.sh"
