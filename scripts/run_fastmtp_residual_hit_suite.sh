#!/usr/bin/env bash
# Run controlled exact-residual-hit tree routes against shared FastMTP baselines.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

usage() {
  cat <<'EOF'
Run six next-step routes in one command:

  shadow          Cactus-equivalence control; tree cannot change output
  anchor          exact residual hit + one target anchor
  strict          exact residual hit + strict branch continuation (15 nodes max)
  cactus          exact residual hit + Cactus branch continuation (15 nodes max)
  strict_wide     strict with four candidates per primary frontier (20 nodes max)
  cactus_wide     Cactus with four candidates per primary frontier (20 nodes max)

Quick run (20 GSM8K + 20 HumanEval per route):
  ./scripts/run_fastmtp_residual_hit_suite.sh

Larger run:
  SAMPLES=100 PROGRESS_EVERY=1 \
    RUN_TAG=fastmtp_residual_hit_n100 \
    ./scripts/run_fastmtp_residual_hit_suite.sh

Resume the same interrupted run by repeating the identical command and
RUN_TAG. Completed route/dataset summaries are skipped. New interruptions are
resumed at request granularity from requests.checkpoint.jsonl. Runs created
before checkpoint support restart only their currently partial dataset.

Run only selected routes:
  ROUTES_CSV=shadow,strict,cactus_wide SAMPLES=20 \
    ./scripts/run_fastmtp_residual_hit_suite.sh

By default Native/Cactus/SpecCascade are run once under the first route, then
linked read-only into all later route directories. BASELINE_ROOT may point to
an existing protocol-identical comparison root to skip those baseline runs.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if (( $# > 0 )); then usage >&2; exit 2; fi

SAMPLES="${SAMPLES:-20}"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-$SAMPLES}"
HUMANEVAL_SAMPLES="${HUMANEVAL_SAMPLES:-$SAMPLES}"
RUN_TAG="${RUN_TAG:-fastmtp_residual_hit_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="$PROJECT_DIR/results/$RUN_TAG"
ROUTES_CSV="${ROUTES_CSV:-shadow,anchor,strict,cactus,strict_wide,cactus_wide}"
BASELINE_ROOT="${BASELINE_ROOT:-}"
IFS=',' read -r -a ROUTES <<< "$ROUTES_CSV"

declare -A seen=()
for route in "${ROUTES[@]}"; do
  case "$route" in
    shadow|anchor|strict|cactus|strict_wide|cactus_wide) ;;
    *) echo "Unknown route: $route" >&2; exit 2 ;;
  esac
  [[ -z "${seen[$route]:-}" ]] || {
    echo "Duplicate route: $route" >&2; exit 2;
  }
  seen[$route]=1
done
(( ${#ROUTES[@]} > 0 )) || { echo "ROUTES_CSV is empty" >&2; exit 2; }

mkdir -p "$SUITE_ROOT"
python - "$SUITE_ROOT/suite_manifest.json" "$RUN_TAG" "$ROUTES_CSV" \
  "$GSM8K_SAMPLES" "$HUMANEVAL_SAMPLES" <<'PY'
import json, sys
from pathlib import Path
path, tag, routes, gsm, human = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_tag": tag,
    "routes": routes.split(","),
    "gsm8k_samples": int(gsm),
    "humaneval_samples": int(human),
    "protocol": "temperature=0.6, seed=42, empty system, no visible thinking",
}, indent=2) + "\n", encoding="utf-8")
PY

link_baselines() {
  local source_root="$1" destination_root="$2" dataset method source destination
  for dataset in gsm8k humaneval; do
    mkdir -p "$destination_root/$dataset"
    for method in native cactus spec_cascade; do
      source="$(realpath "$source_root/$dataset/$method")"
      [[ -f "$source/summary.json" ]] || {
        echo "Missing baseline: $source/summary.json" >&2; exit 2;
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

route_parameters() {
  local route="$1"
  case "$route" in
    shadow)
      ROUTE_LABEL="Shadow: exact Cactus correction"
      SUPPORT_MODE=sampled_primary_shadow; MAX_NODES=9; MAX_CHILDREN=3
      SIBLING_RATIO=0.01; TAU_MIN=0.001; ALLOCATION=reach_first ;;
    anchor)
      ROUTE_LABEL="Residual hit + target anchor"
      SUPPORT_MODE=residual_hit_anchor; MAX_NODES=9; MAX_CHILDREN=3
      SIBLING_RATIO=0.01; TAU_MIN=0.001; ALLOCATION=residual_coverage ;;
    strict)
      ROUTE_LABEL="Residual hit + strict continuation"
      SUPPORT_MODE=residual_hit_strict; MAX_NODES=15; MAX_CHILDREN=3
      SIBLING_RATIO=0.01; TAU_MIN=0.001; ALLOCATION=residual_coverage ;;
    cactus)
      ROUTE_LABEL="Residual hit + Cactus continuation"
      SUPPORT_MODE=residual_hit_cactus; MAX_NODES=15; MAX_CHILDREN=3
      SIBLING_RATIO=0.01; TAU_MIN=0.001; ALLOCATION=residual_coverage ;;
    strict_wide)
      ROUTE_LABEL="Residual hit + strict continuation (wide)"
      SUPPORT_MODE=residual_hit_strict; MAX_NODES=20; MAX_CHILDREN=4
      SIBLING_RATIO=0.0; TAU_MIN=0.000001; ALLOCATION=residual_coverage ;;
    cactus_wide)
      ROUTE_LABEL="Residual hit + Cactus continuation (wide)"
      SUPPORT_MODE=residual_hit_cactus; MAX_NODES=20; MAX_CHILDREN=4
      SIBLING_RATIO=0.0; TAU_MIN=0.000001; ALLOCATION=residual_coverage ;;
  esac
}

run_route() {
  local route="$1"
  local route_tag="$RUN_TAG/$route" route_root="$SUITE_ROOT/$route"
  local methods
  route_parameters "$route"
  if [[ -n "$BASELINE_ROOT" ]]; then
    link_baselines "$BASELINE_ROOT" "$route_root"
    methods=dynamic_tree
  else
    methods=native,cactus,spec_cascade,dynamic_tree
  fi
  mkdir -p "$route_root"
  python - "$route_root/route_manifest.json" "$route" "$ROUTE_LABEL" \
    "$SUPPORT_MODE" "$MAX_NODES" "$MAX_CHILDREN" "$SIBLING_RATIO" "$TAU_MIN" "$ALLOCATION" <<'PY'
import json, sys
from pathlib import Path
path, name, label, mode, nodes, children, ratio, tau, allocation = sys.argv[1:]
Path(path).write_text(json.dumps({
    "route": name,
    "label": label,
    "support_mode": mode,
    "max_depth": 3,
    "max_nodes": int(nodes),
    "max_children": int(children),
    "min_sibling_ratio": float(ratio),
    "min_draft_probability": float(tau),
    "allocation": allocation,
    "cactus_delta": 1.0,
}, indent=2) + "\n", encoding="utf-8")
PY

  echo
  echo "===== route=$route mode=$SUPPORT_MODE D=3 N_max=$MAX_NODES children=$MAX_CHILDREN ====="
  GSM8K_SAMPLES="$GSM8K_SAMPLES" \
  HUMANEVAL_SAMPLES="$HUMANEVAL_SAMPLES" \
  RUN_TAG="$route_tag" \
  METHODS_CSV="$methods" \
  INCLUDE_TARGET=0 \
  PROGRESS_EVERY="${PROGRESS_EVERY:-1}" \
  FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}" \
  RESUME_PARTIAL="${RESUME_PARTIAL:-1}" \
  MTP_TOKENS=3 \
  DYNAMIC_MAX_DEPTH=3 \
  DYNAMIC_MAX_NODES="$MAX_NODES" \
  DYNAMIC_MAX_CHILDREN="$MAX_CHILDREN" \
  DYNAMIC_MIN_SIBLING_RATIO="$SIBLING_RATIO" \
  DYNAMIC_TAU_MIN="$TAU_MIN" \
  DYNAMIC_KAPPA=1.5 \
  DYNAMIC_MU=0.5 \
  DYNAMIC_ETA=0.0 \
  DYNAMIC_ALLOCATION="$ALLOCATION" \
  DYNAMIC_SUPPORT_MODE="$SUPPORT_MODE" \
  DYNAMIC_CACTUS_DELTA="${CACTUS_DELTA:-1.0}" \
  DYNAMIC_FRONTIER_RESCUE=0 \
  DYNAMIC_PATH_SELECTION=balanced \
  DYNAMIC_PATH_TEMPERATURE=0 \
    "$PROJECT_DIR/scripts/run_fastmtp_verified_comparison.sh"

  if [[ -z "$BASELINE_ROOT" ]]; then
    BASELINE_ROOT="$route_root"
  fi
}

cat <<EOF
Exact-residual-hit FastMTP suite
  routes        : ${ROUTES[*]}
  samples       : GSM8K=$GSM8K_SAMPLES HumanEval=$HUMANEVAL_SAMPLES
  shared base   : ${BASELINE_ROOT:-run once under ${ROUTES[0]}}
  output        : $SUITE_ROOT/comparison.md
  live logs     : $PROJECT_DIR/logs/$RUN_TAG/<route>/
  resumable     : yes (RESUME_PARTIAL=${RESUME_PARTIAL:-1})
EOF

for route in "${ROUTES[@]}"; do
  run_route "$route"
done

python -m remtp.fastmtp_residual_hit_report --suite-root "$SUITE_ROOT"
echo
echo "===== combined result ====="
sed -n '1,260p' "$SUITE_ROOT/comparison.md"
