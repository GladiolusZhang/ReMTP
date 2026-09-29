#!/usr/bin/env bash
# Compare uniform RAVEN with saturation-aware adaptive actual-TV allocation.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

usage() {
  cat <<'EOF'
Run the recommended 100-task adaptive-relaxation confirmation:

  SAMPLES=100 PROGRESS_EVERY=1 \
    RUN_TAG=raven_adaptive_n100 \
    ./scripts/run_fastmtp_raven_adaptive_relaxation.sh

The suite runs Native, Cactus, SpecCascade and uniform RAVEN once, then links
the three chain baselines into the adaptive profile and runs adaptive RAVEN.
Generation is resumable: repeat the exact command after an interruption.

Optional reuse of an existing protocol-identical uniform result:

  CONTROL_ROOT=results/<prior-run>/uniform \
  SAMPLES=100 RUN_TAG=raven_adaptive_n100 \
    ./scripts/run_fastmtp_raven_adaptive_relaxation.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if (( $# > 0 )); then usage >&2; exit 2; fi

SAMPLES="${SAMPLES:-100}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-$SAMPLES}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-$SAMPLES}"
RUN_TAG="${RUN_TAG:-raven_adaptive_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="$PROJECT_DIR/results/$RUN_TAG"
CONTROL_ROOT="${CONTROL_ROOT:-}"
RAVEN_BASE_DELTA="${RAVEN_BASE_DELTA:-1.0}"
ADAPTIVE_MIX="${ADAPTIVE_MIX:-0.5}"
ADAPTIVE_MAX_RATIO="${ADAPTIVE_MAX_RATIO:-1.5}"
ADAPTIVE_TARGET_POWER="${ADAPTIVE_TARGET_POWER:-0.25}"
ADAPTIVE_STEPS="${ADAPTIVE_STEPS:-16}"

mkdir -p "$SUITE_ROOT"
python - "$SUITE_ROOT/suite_manifest.json" "$RUN_TAG" \
  "$GSM8K_SAMPLES" "$HUMANEVAL_SAMPLES" "$RAVEN_BASE_DELTA" \
  "$ADAPTIVE_MIX" "$ADAPTIVE_MAX_RATIO" "$ADAPTIVE_TARGET_POWER" \
  "$ADAPTIVE_STEPS" <<'PY'
import json, sys
from pathlib import Path
(
    path, tag, gsm, human, delta, mix, max_ratio, target_power, steps
) = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_tag": tag,
    "profiles": ["uniform", "adaptive"],
    "gsm8k_samples": int(gsm),
    "humaneval_samples": int(human),
    "base_delta": float(delta),
    "adaptive_mix": float(mix),
    "adaptive_max_ratio": float(max_ratio),
    "adaptive_target_power": float(target_power),
    "adaptive_steps": int(steps),
    "protocol": "temperature=0.6, seed=42, empty system, no visible thinking",
}, indent=2) + "\n", encoding="utf-8")
PY

link_methods() {
  local source_root="$1" destination_root="$2"
  shift 2
  local dataset method source destination
  for dataset in gsm8k humaneval; do
    mkdir -p "$destination_root/$dataset"
    for method in "$@"; do
      source="$(realpath "$source_root/$dataset/$method")"
      [[ -f "$source/summary.json" ]] || {
        echo "Missing reusable result: $source/summary.json" >&2
        exit 2
      }
      destination="$destination_root/$dataset/$method"
      if [[ -e "$destination" || -L "$destination" ]]; then
        [[ -L "$destination" && "$(realpath "$destination")" == "$source" ]] || {
          echo "Refusing to replace existing result: $destination" >&2
          exit 2
        }
      else
        ln -s "$source" "$destination"
      fi
    done
  done
}

write_profile_manifest() {
  local root="$1" profile="$2" enabled="$3" label="$4"
  python - "$root/profile_manifest.json" "$profile" "$enabled" "$label" \
    "$RAVEN_BASE_DELTA" "$ADAPTIVE_MIX" "$ADAPTIVE_MAX_RATIO" \
    "$ADAPTIVE_TARGET_POWER" "$ADAPTIVE_STEPS" <<'PY'
import json, sys
from pathlib import Path
path, profile, enabled, label, delta, mix, max_ratio, target_power, steps = sys.argv[1:]
Path(path).write_text(json.dumps({
    "profile": profile,
    "label": label,
    "base_delta": float(delta),
    "depth_weights": [1.0, 1.0, 1.0],
    "adaptive_relaxation": enabled == "1",
    "adaptive_mix": float(mix),
    "adaptive_max_ratio": float(max_ratio),
    "adaptive_target_power": float(target_power),
    "adaptive_steps": int(steps),
    "support_mode": "residual_hit_anchor",
    "max_depth": 3,
    "max_nodes": 9,
}, indent=2) + "\n", encoding="utf-8")
PY
}

run_raven() {
  local profile="$1" enabled="$2" methods="$3"
  local profile_root="$SUITE_ROOT/$profile"
  mkdir -p "$profile_root"
  GSM8K_SAMPLES="$GSM8K_SAMPLES" \
  HUMANEVAL_SAMPLES="$HUMANEVAL_SAMPLES" \
  RUN_TAG="$RUN_TAG/$profile" \
  METHODS_CSV="$methods" \
  INCLUDE_TARGET=0 \
  PROGRESS_EVERY="${PROGRESS_EVERY:-1}" \
  FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}" \
  RESUME_PARTIAL="${RESUME_PARTIAL:-1}" \
  CACTUS_DELTA="$RAVEN_BASE_DELTA" \
  MTP_TOKENS=3 \
  DYNAMIC_MAX_DEPTH=3 \
  DYNAMIC_MAX_NODES=9 \
  DYNAMIC_MAX_CHILDREN=3 \
  DYNAMIC_MIN_SIBLING_RATIO=0.01 \
  DYNAMIC_TAU_MIN=0.001 \
  DYNAMIC_KAPPA=1.5 \
  DYNAMIC_MU=0.5 \
  DYNAMIC_ETA=0.0 \
  DYNAMIC_ALLOCATION=residual_coverage \
  DYNAMIC_SUPPORT_MODE=residual_hit_anchor \
  DYNAMIC_CACTUS_DELTA="$RAVEN_BASE_DELTA" \
  DYNAMIC_RELAX_DEPTH_WEIGHTS=1.0,1.0,1.0 \
  DYNAMIC_ADAPTIVE_RELAXATION="$enabled" \
  DYNAMIC_ADAPTIVE_RELAX_MIX="$ADAPTIVE_MIX" \
  DYNAMIC_ADAPTIVE_RELAX_MAX_RATIO="$ADAPTIVE_MAX_RATIO" \
  DYNAMIC_ADAPTIVE_RELAX_TARGET_POWER="$ADAPTIVE_TARGET_POWER" \
  DYNAMIC_ADAPTIVE_RELAX_STEPS="$ADAPTIVE_STEPS" \
  DYNAMIC_FRONTIER_RESCUE=0 \
  DYNAMIC_PATH_SELECTION=balanced \
  DYNAMIC_PATH_TEMPERATURE=0 \
    "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"
}

cat <<EOF
RAVEN saturation-aware adaptive relaxation
  samples       : GSM8K=$GSM8K_SAMPLES HumanEval=$HUMANEVAL_SAMPLES
  base delta    : $RAVEN_BASE_DELTA
  adaptive      : mix=$ADAPTIVE_MIX max_ratio=$ADAPTIVE_MAX_RATIO target_power=$ADAPTIVE_TARGET_POWER steps=$ADAPTIVE_STEPS
  output        : $SUITE_ROOT/comparison.md
  live logs     : $PROJECT_DIR/logs/$RUN_TAG/{uniform,adaptive}/
  resumable     : yes (RESUME_PARTIAL=${RESUME_PARTIAL:-1})
EOF

uniform_root="$SUITE_ROOT/uniform"
mkdir -p "$uniform_root"
write_profile_manifest "$uniform_root" uniform 0 "RAVEN uniform delta"
if [[ -n "$CONTROL_ROOT" ]]; then
  echo "===== uniform reused from $CONTROL_ROOT ====="
  link_methods "$CONTROL_ROOT" "$uniform_root" \
    native cactus spec_cascade dynamic_tree
else
  echo "===== uniform control ====="
  run_raven uniform 0 native,cactus,spec_cascade,dynamic_tree
fi

adaptive_root="$SUITE_ROOT/adaptive"
mkdir -p "$adaptive_root"
write_profile_manifest \
  "$adaptive_root" adaptive 1 "RAVEN adaptive actual-TV"
link_methods "$uniform_root" "$adaptive_root" native cactus spec_cascade
echo "===== adaptive actual-TV ====="
run_raven adaptive 1 dynamic_tree

python -m remtp.raven_adaptive_relaxation_report --suite-root "$SUITE_ROOT"
echo
echo "===== combined result ====="
sed -n '1,300p' "$SUITE_ROOT/comparison.md"
