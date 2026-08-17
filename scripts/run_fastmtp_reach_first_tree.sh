#!/usr/bin/env bash
# Fair D=3 reach-first relaxed-tree run; reuse unchanged audited baselines.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run only the new fair-depth soft-reach tree and reuse audited baselines.

Default full experiment:
  ./scripts/run_fastmtp_soft_reach_tree.sh

Optional output name:
  RUN_TAG=fastmtp_reach_first_d3_full \
    ./scripts/run_fastmtp_soft_reach_tree.sh

The default verifier threshold is 0.35. Override DYNAMIC_TAU_RELAX only for
an explicit follow-up ablation; the proposal depth remains fixed at D=3.
EOF
  exit 0
fi
if (( $# > 0 )); then
  echo "Unknown arguments; use --help." >&2
  exit 2
fi

GSM8K_SAMPLES="${GSM8K_SAMPLES:-500}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-164}"
BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_DIR/results/fastmtp_depth34_relaxed_full/d3}"
RUN_TAG="${RUN_TAG:-fastmtp_reach_first_d3_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"

link_baselines() {
  local dataset method source_dir destination
  for dataset in gsm8k humaneval; do
    mkdir -p "$RUN_ROOT/$dataset"
    for method in native cactus spec_cascade; do
      source_dir="$(realpath "$BASELINE_ROOT/$dataset/$method")"
      [[ -f "$source_dir/summary.json" ]] || {
        echo "Missing audited baseline: $source_dir/summary.json" >&2
        exit 2
      }
      destination="$RUN_ROOT/$dataset/$method"
      if [[ -e "$destination" || -L "$destination" ]]; then
        if [[ ! -L "$destination" || "$(realpath "$destination")" != "$source_dir" ]]; then
          echo "Refusing to replace existing path: $destination" >&2
          exit 2
        fi
      else
        ln -s "$source_dir" "$destination"
      fi
    done
  done
}

link_baselines

case "${DYNAMIC_SUPPORT_MODE:-relative}" in
  cactus_trunk_rescue)
    verifier_summary="sampled-Q Cactus trunk with rejection-only residual rescue"
    ;;
  sampled_primary_reopen)
    verifier_summary="sampled-Q Cactus primary with residual tree reopen; continuation relative >= ${DYNAMIC_TAU_RELAX:-0.20}"
    ;;
  sampled_primary_shadow)
    verifier_summary="sampled-Q Cactus primary; exact residual correction; tree observed only"
    ;;
  residual_hit_anchor)
    verifier_summary="sampled-Q Cactus primary; exact residual hit reuses branch plus target anchor"
    ;;
  residual_hit_strict)
    verifier_summary="sampled-Q Cactus primary; exact residual hit plus strict p/q continuation"
    ;;
  residual_hit_cactus)
    verifier_summary="sampled-Q Cactus primary; exact residual hit plus Cactus continuation"
    ;;
  *)
    verifier_summary="target-relative >= ${DYNAMIC_TAU_RELAX:-0.35}"
    ;;
esac

cat <<EOF
FastMTP reach-aware relaxed tree
  fair logical depth : 3 (same as Native/Cactus/SpecCascade)
  reused baselines   : $BASELINE_ROOT
  samples            : GSM8K=$GSM8K_SAMPLES HumanEval=$HUMANEVAL_SAMPLES
  proposal allocator : ${DYNAMIC_ALLOCATION:-soft_reach}; rank-0 continuations
                       are prioritized when reach_first is selected
  node cap           : ${DYNAMIC_MAX_NODES:-10} (maximum, not force-filled)
  relaxed verifier   : $verifier_summary
  path selector      : ${DYNAMIC_PATH_SELECTION:-balanced} beta=${DYNAMIC_BETA:-0.75}, deterministic
  output             : $RUN_ROOT/comparison.md
EOF

GSM8K_SAMPLES="$GSM8K_SAMPLES" \
HUMANEVAL_SAMPLES="$HUMANEVAL_SAMPLES" \
RUN_TAG="$RUN_TAG" \
METHODS_CSV=dynamic_tree \
INCLUDE_TARGET=0 \
PROGRESS_EVERY="${PROGRESS_EVERY:-1}" \
FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}" \
MTP_TOKENS=3 \
DYNAMIC_MAX_DEPTH=3 \
DYNAMIC_MAX_NODES="${DYNAMIC_MAX_NODES:-10}" \
DYNAMIC_MAX_CHILDREN="${DYNAMIC_MAX_CHILDREN:-3}" \
DYNAMIC_MIN_SIBLING_RATIO="${DYNAMIC_MIN_SIBLING_RATIO:-0.02}" \
DYNAMIC_TAU_MIN="${DYNAMIC_TAU_MIN:-0.002}" \
DYNAMIC_KAPPA="${DYNAMIC_KAPPA:-1.25}" \
DYNAMIC_MU="${DYNAMIC_MU:-0.5}" \
DYNAMIC_ETA="${DYNAMIC_ETA:-0.25}" \
DYNAMIC_ALLOCATION="${DYNAMIC_ALLOCATION:-soft_reach}" \
DYNAMIC_RANK_PENALTY="${DYNAMIC_RANK_PENALTY:-2.0}" \
DYNAMIC_COVERAGE_MODE=coverage_gate \
DYNAMIC_MIN_COVERAGE="${DYNAMIC_MIN_COVERAGE:-0.05}" \
DYNAMIC_TAU_RELAX="${DYNAMIC_TAU_RELAX:-0.35}" \
DYNAMIC_SUPPORT_MODE="${DYNAMIC_SUPPORT_MODE:-relative}" \
DYNAMIC_PROPOSAL_SUPPORT_RATIO="${DYNAMIC_PROPOSAL_SUPPORT_RATIO:-1.0}" \
DYNAMIC_CONFIRMATION_MIN_RELATIVE="${DYNAMIC_CONFIRMATION_MIN_RELATIVE:-0.05}" \
DYNAMIC_PATH_SELECTION="${DYNAMIC_PATH_SELECTION:-balanced}" \
DYNAMIC_BETA="${DYNAMIC_BETA:-0.75}" \
DYNAMIC_PATH_TEMPERATURE=0 \
DYNAMIC_FRONTIER_RESCUE="${DYNAMIC_FRONTIER_RESCUE:-1}" \
DYNAMIC_RESCUE_DELTA="${DYNAMIC_RESCUE_DELTA:-0.5}" \
DYNAMIC_RESCUE_MIN_RELATIVE="${DYNAMIC_RESCUE_MIN_RELATIVE:-0.1}" \
DYNAMIC_RESCUE_MIN_TARGET_PROB="${DYNAMIC_RESCUE_MIN_TARGET_PROB:-0.001}" \
  "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"
