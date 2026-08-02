#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A focused comparison: native MTP is the quality anchor, Cactus the speed
# anchor, SpecCascade the paper reference, and residual regret is our method.
export PROFILES="${PROFILES:-native_mtp cactus spec_cascade cactus_regret}"
export SAMPLES="${SAMPLES:-40}"
export SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export SEED="${SEED:-42}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export MTP_TOKENS=6
export CACTUS_DELTA="${CACTUS_DELTA:-1.0}"
export CACTUS_REGRET_ALPHA="${CACTUS_REGRET_ALPHA:-0.03}"
export CACTUS_REGRET_TOP_K="${CACTUS_REGRET_TOP_K:-16}"
export CACTUS_REGRET_STRENGTH_REFERENCE="${CACTUS_REGRET_STRENGTH_REFERENCE:-0.10}"
export CACTUS_REGRET_DEPTH_DECAY="${CACTUS_REGRET_DEPTH_DECAY:-0.90}"
export CACTUS_REGRET_RESPONSIBILITY="${CACTUS_REGRET_RESPONSIBILITY:-posterior}"
export CACTUS_REGRET_INJECTION_SITE="${CACTUS_REGRET_INJECTION_SITE:-head1}"
export CACTUS_REGRET_RESIDUAL_SPACE="${CACTUS_REGRET_RESIDUAL_SPACE:-output}"

exec "$PROJECT_DIR/scripts/run_humaneval_comparison.sh" "$@"
