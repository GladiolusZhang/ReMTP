#!/usr/bin/env bash
# Selected exact-residual-hit tree: Cactus primary + exact correction reuse.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the selected FastMTP residual-hit anchor method against shared baselines.

Default formal-size run:
  SAMPLES=100 PROGRESS_EVERY=1 \
    RUN_TAG=fastmtp_residual_anchor_n100 \
    ./scripts/run_fastmtp_selected_residual_anchor.sh

To reuse protocol-identical Native/Cactus/SpecCascade results:
  BASELINE_ROOT=/absolute/path/to/comparison_root \
    SAMPLES=100 RUN_TAG=fastmtp_residual_anchor_n100 \
    ./scripts/run_fastmtp_selected_residual_anchor.sh

Repeat the identical command and RUN_TAG after interruption to resume.
EOF
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown arguments; use --help." >&2
  exit 2
fi

export SAMPLES="${SAMPLES:-100}"
export ROUTES_CSV=anchor
export PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
export RUN_TAG="${RUN_TAG:-fastmtp_residual_anchor_$(date +%Y%m%d_%H%M%S)}"
export RESUME_PARTIAL="${RESUME_PARTIAL:-1}"

exec "$PROJECT_DIR/scripts/run_fastmtp_residual_hit_suite.sh"
