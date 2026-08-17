#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Balanced Target-Band candidate selected for full evaluation. It anchors the
# candidate to the target head, limits each position and the whole block, and
# uses actual preceding TV allocation as within-block regret debt.
export MODE_REGRET_ELIGIBILITY=target_band
export MODE_REGRET_MAX_TARGET_RANK="${MODE_REGRET_MAX_TARGET_RANK:-4}"
export MODE_REGRET_MAX_CANDIDATE_GAP="${MODE_REGRET_MAX_CANDIDATE_GAP:-0.75}"
export MODE_REGRET_PER_TOKEN_TV="${MODE_REGRET_PER_TOKEN_TV:-0.30}"
export MODE_REGRET_BLOCK_TV="${MODE_REGRET_BLOCK_TV:-0.85}"
export MODE_REGRET_BAND_DEBT_SLOPE="${MODE_REGRET_BAND_DEBT_SLOPE:-0.50}"
export MODE_REGRET_BAND_DEPTH_SLOPE="${MODE_REGRET_BAND_DEPTH_SLOPE:-0.00}"

exec "$PROJECT_DIR/scripts/serve_target_mode_regret_mtp.sh"
