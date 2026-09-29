#!/usr/bin/env bash
# Compare uniform and native-MTP depth-calibrated RAVEN relaxation schedules.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

usage() {
  cat <<'EOF'
Run a small RAVEN depth-relaxation sweep on GSM8K and HumanEval.

Quick pilot (recommended first run):
  SAMPLES=20 PROGRESS_EVERY=1 \
    RUN_TAG=raven_depth_pilot \
    ./scripts/run_fastmtp_raven_depth_relaxation.sh

Larger confirmation:
  SAMPLES=100 PROGRESS_EVERY=1 \
    RUN_TAG=raven_depth_n100 \
    ./scripts/run_fastmtp_raven_depth_relaxation.sh

Default profiles:
  uniform    w=(1.000, 1.000, 1.000), historical control
  empirical  w=(1.143, 0.986, 0.871), mean-one observed reliability
  squared    w=(1.290, 0.960, 0.750), stronger mean-one redistribution

Select profiles or add one custom schedule:
  PROFILES_CSV=empirical,custom \
  CUSTOM_DEPTH_WEIGHTS=1.0,0.70,0.40 \
  SAMPLES=20 RUN_TAG=raven_depth_custom \
    ./scripts/run_fastmtp_raven_depth_relaxation.sh

Set BASELINE_ROOT to a protocol-identical prior comparison root containing
gsm8k/{native,cactus,spec_cascade} and humaneval/{...}. Baselines are then
linked read-only and are not rerun. Repeating the same RUN_TAG resumes partial
generation when RESUME_PARTIAL=1 (the default here).

For a prior protocol-identical uniform RAVEN run, set CONTROL_ROOT instead.
Its baselines and dynamic_tree control are all linked, so the sweep executes
only the new calibrated profiles.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if (( $# > 0 )); then usage >&2; exit 2; fi

SAMPLES="${SAMPLES:-20}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-$SAMPLES}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-$SAMPLES}"
RUN_TAG="${RUN_TAG:-raven_depth_relaxation_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="$PROJECT_DIR/results/$RUN_TAG"
PROFILES_CSV="${PROFILES_CSV:-uniform,empirical,squared}"
RAVEN_BASE_DELTA="${RAVEN_BASE_DELTA:-1.0}"
CUSTOM_DEPTH_WEIGHTS="${CUSTOM_DEPTH_WEIGHTS:-}"
BASELINE_ROOT="${BASELINE_ROOT:-}"
CONTROL_ROOT="${CONTROL_ROOT:-}"
if [[ -n "$CONTROL_ROOT" && -z "$BASELINE_ROOT" ]]; then
  BASELINE_ROOT="$CONTROL_ROOT"
fi
IFS=',' read -r -a PROFILES <<< "$PROFILES_CSV"

declare -A seen=()
for profile in "${PROFILES[@]}"; do
  case "$profile" in
    uniform|empirical|squared) ;;
    custom)
      [[ -n "$CUSTOM_DEPTH_WEIGHTS" ]] || {
        echo "CUSTOM_DEPTH_WEIGHTS is required for profile=custom" >&2
        exit 2
      }
      ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
  esac
  [[ -z "${seen[$profile]:-}" ]] || {
    echo "Duplicate profile: $profile" >&2
    exit 2
  }
  seen[$profile]=1
done
(( ${#PROFILES[@]} > 0 )) || { echo "PROFILES_CSV is empty" >&2; exit 2; }

mkdir -p "$SUITE_ROOT"
python - "$SUITE_ROOT/suite_manifest.json" "$RUN_TAG" "$PROFILES_CSV" \
  "$GSM8K_SAMPLES" "$HUMANEVAL_SAMPLES" "$RAVEN_BASE_DELTA" <<'PY'
import json, sys
from pathlib import Path
path, tag, profiles, gsm, human, delta = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_tag": tag,
    "profiles": profiles.split(","),
    "gsm8k_samples": int(gsm),
    "humaneval_samples": int(human),
    "base_delta": float(delta),
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
        echo "Missing baseline: $source/summary.json" >&2
        exit 2
      }
      destination="$destination_root/$dataset/$method"
      if [[ -e "$destination" || -L "$destination" ]]; then
        [[ -L "$destination" && "$(realpath "$destination")" == "$source" ]] || {
          echo "Refusing to replace existing baseline path: $destination" >&2
          exit 2
        }
      else
        ln -s "$source" "$destination"
      fi
    done
  done
}

link_baselines() {
  link_methods "$1" "$2" native cactus spec_cascade
}

profile_parameters() {
  local profile="$1"
  case "$profile" in
    uniform)
      PROFILE_LABEL="RAVEN uniform delta (control)"
      DEPTH_WEIGHTS="1.000,1.000,1.000"
      ;;
    empirical)
      PROFILE_LABEL="RAVEN depth-calibrated"
      DEPTH_WEIGHTS="1.143,0.986,0.871"
      ;;
    squared)
      PROFILE_LABEL="RAVEN depth-calibrated (squared)"
      DEPTH_WEIGHTS="1.290,0.960,0.750"
      ;;
    custom)
      PROFILE_LABEL="RAVEN depth-calibrated (custom)"
      DEPTH_WEIGHTS="$CUSTOM_DEPTH_WEIGHTS"
      ;;
  esac
  python - "$DEPTH_WEIGHTS" <<'PY'
import math, sys
weights = [float(part.strip()) for part in sys.argv[1].split(",")]
if len(weights) != 3:
    raise SystemExit("depth schedule must contain exactly three weights")
if any(not math.isfinite(value) or value < 0 for value in weights):
    raise SystemExit("depth weights must be finite and non-negative")
if any(b > a for a, b in zip(weights, weights[1:])):
    raise SystemExit("depth weights must be non-increasing")
PY
}

run_profile() {
  local profile="$1"
  local profile_tag="$RUN_TAG/$profile" profile_root="$SUITE_ROOT/$profile"
  local methods reuse_control=0
  profile_parameters "$profile"
  mkdir -p "$profile_root"
  if [[ "$profile" == "uniform" && -n "$CONTROL_ROOT" ]]; then
    link_methods "$CONTROL_ROOT" "$profile_root" \
      native cactus spec_cascade dynamic_tree
    reuse_control=1
    methods=""
  elif [[ -n "$BASELINE_ROOT" ]]; then
    link_baselines "$BASELINE_ROOT" "$profile_root"
    methods=dynamic_tree
  else
    methods=native,cactus,spec_cascade,dynamic_tree
  fi
  python - "$profile_root/profile_manifest.json" "$profile" "$PROFILE_LABEL" \
    "$RAVEN_BASE_DELTA" "$DEPTH_WEIGHTS" <<'PY'
import json, sys
from pathlib import Path
path, name, label, delta, weights = sys.argv[1:]
Path(path).write_text(json.dumps({
    "profile": name,
    "label": label,
    "base_delta": float(delta),
    "depth_weights": [float(part) for part in weights.split(",")],
    "effective_deltas": [float(delta) * float(part) for part in weights.split(",")],
    "support_mode": "residual_hit_anchor",
    "max_depth": 3,
    "max_nodes": 9,
}, indent=2) + "\n", encoding="utf-8")
PY

  if [[ "$reuse_control" == "1" ]]; then
    echo
    echo "===== profile=uniform reused from $CONTROL_ROOT ====="
    return
  fi

  echo
  echo "===== profile=$profile delta=$RAVEN_BASE_DELTA weights=$DEPTH_WEIGHTS ====="
  GSM8K_SAMPLES="$GSM8K_SAMPLES" \
  HUMANEVAL_SAMPLES="$HUMANEVAL_SAMPLES" \
  RUN_TAG="$profile_tag" \
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
  DYNAMIC_RELAX_DEPTH_WEIGHTS="$DEPTH_WEIGHTS" \
  DYNAMIC_FRONTIER_RESCUE=0 \
  DYNAMIC_PATH_SELECTION=balanced \
  DYNAMIC_PATH_TEMPERATURE=0 \
    "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"

  if [[ -z "$BASELINE_ROOT" ]]; then
    BASELINE_ROOT="$profile_root"
  fi
}

cat <<EOF
RAVEN native-MTP depth-relaxation sweep
  profiles      : ${PROFILES[*]}
  samples       : GSM8K=$GSM8K_SAMPLES HumanEval=$HUMANEVAL_SAMPLES
  base delta    : $RAVEN_BASE_DELTA
  shared base   : ${BASELINE_ROOT:-run once under ${PROFILES[0]}}
  output        : $SUITE_ROOT/comparison.md
  live logs     : $PROJECT_DIR/logs/$RUN_TAG/<profile>/
  resumable     : yes (RESUME_PARTIAL=${RESUME_PARTIAL:-1})
EOF

for profile in "${PROFILES[@]}"; do
  run_profile "$profile"
done

python -m remtp.raven_depth_relaxation_report --suite-root "$SUITE_ROOT"
echo
echo "===== combined result ====="
sed -n '1,280p' "$SUITE_ROOT/comparison.md"
