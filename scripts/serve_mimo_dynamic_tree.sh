#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/cuda_env.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/MiMo-7B-Base-MTP3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-XiaomiMiMo/MiMo-7B-Base}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
PORT="${PORT:-8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TREE_RUN_DIR="${TREE_RUN_DIR:-$PROJECT_DIR/results/mimo_dynamic_tree_$RUN_TAG}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export REMTP_DYNAMIC_TREE_MTP_ROUTE="${REMTP_DYNAMIC_TREE_MTP_ROUTE:-physical_012}"
case "$REMTP_DYNAMIC_TREE_MTP_ROUTE" in
  physical_012)
    export REMTP_MIMO_MTP_LAYER_MODE=physical
    ;;
  repeat_000)
    export REMTP_MIMO_MTP_LAYER_MODE=layer0
    ;;
  *)
    echo "Unknown REMTP_DYNAMIC_TREE_MTP_ROUTE=$REMTP_DYNAMIC_TREE_MTP_ROUTE" >&2
    echo "Choose: physical_012 or repeat_000" >&2
    exit 2
    ;;
esac
export REMTP_ALLOW_EXPERIMENTAL_MIMO_MTP3=1
export REMTP_MIMO_PREFILL_ALL_LAYERS="${REMTP_MIMO_PREFILL_ALL_LAYERS:-1}"
export REMTP_TREE_VERIFY_MODE=target_dynamic_relaxed
export REMTP_TREE_TOKENIZER="$MODEL_PATH"
export REMTP_TREE_TRACE="${REMTP_TREE_TRACE:-1}"
export REMTP_TREE_TRACE_MAX_ROUNDS="${REMTP_TREE_TRACE_MAX_ROUNDS:-64}"
export REMTP_TREE_AUDIT_JSONL="${REMTP_TREE_AUDIT_JSONL:-$TREE_RUN_DIR/rounds.jsonl}"
export REMTP_TREE_TRACE_LOG="${REMTP_TREE_TRACE_LOG:-$TREE_RUN_DIR/tree_trace.log}"
REMTP_TREE_REPORT_MD="${REMTP_TREE_REPORT_MD:-$TREE_RUN_DIR/tree_summary.md}"

# Dynamic construction defaults. Every value remains externally configurable.
export REMTP_DYNAMIC_TREE_MAX_DEPTH="${REMTP_DYNAMIC_TREE_MAX_DEPTH:-3}"
export REMTP_DYNAMIC_TREE_MAX_NODES="${REMTP_DYNAMIC_TREE_MAX_NODES:-6}"
export REMTP_DYNAMIC_TREE_MAX_CHILDREN="${REMTP_DYNAMIC_TREE_MAX_CHILDREN:-2}"
export REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO="${REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO:-0.25}"
export REMTP_DYNAMIC_TREE_TAU_MIN="${REMTP_DYNAMIC_TREE_TAU_MIN:-0.02}"
export REMTP_DYNAMIC_TREE_KAPPA="${REMTP_DYNAMIC_TREE_KAPPA:-1.0}"
export REMTP_DYNAMIC_TREE_MU="${REMTP_DYNAMIC_TREE_MU:-0.5}"
export REMTP_DYNAMIC_TREE_ETA="${REMTP_DYNAMIC_TREE_ETA:-0.25}"
export REMTP_DYNAMIC_TREE_ALPHA="${REMTP_DYNAMIC_TREE_ALPHA:-0.5}"
export REMTP_DYNAMIC_TREE_TAU_RELAX="${REMTP_DYNAMIC_TREE_TAU_RELAX:-0.7}"
export REMTP_DYNAMIC_TREE_BETA="${REMTP_DYNAMIC_TREE_BETA:-0.5}"
export REMTP_DYNAMIC_TREE_PATH_TEMPERATURE="${REMTP_DYNAMIC_TREE_PATH_TEMPERATURE:-0.3}"
export REMTP_DYNAMIC_TREE_PATH_SELECTION="${REMTP_DYNAMIC_TREE_PATH_SELECTION:-longest}"
export REMTP_DYNAMIC_TREE_APPEND_ANCHOR="${REMTP_DYNAMIC_TREE_APPEND_ANCHOR:-1}"
if [[ -z "${REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS:-}" ]]; then
  REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS="$(python - "$MODEL_PATH/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get("eos_token_id", [])
if value is None:
    value = []
if isinstance(value, int):
    value = [value]
print(",".join(str(int(token_id)) for token_id in value))
PY
)"
fi
export REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS
export REMTP_DYNAMIC_TREE_EOS_THRESHOLD="${REMTP_DYNAMIC_TREE_EOS_THRESHOLD:-0.5}"

mkdir -p "$TREE_RUN_DIR"

write_tree_report() {
  if [[ -s "$REMTP_TREE_AUDIT_JSONL" ]]; then
    python -m remtp.mimo_tree_report \
      --audit "$REMTP_TREE_AUDIT_JSONL" \
      --output "$REMTP_TREE_REPORT_MD" \
      --json-output "$TREE_RUN_DIR/tree_summary.json" || true
    echo "[ReMTP][DynamicTree] summary=$REMTP_TREE_REPORT_MD"
  fi
}
trap write_tree_report EXIT

python -m remtp.mimo_checkpoint check "$MODEL_PATH"

node_budget="$REMTP_DYNAMIC_TREE_MAX_NODES"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}-Tree${node_budget}}"
python -m remtp.mimo_checkpoint logical-view \
  --source "$MODEL_PATH" \
  --output "$DRAFT_MODEL_PATH" \
  --width "$node_budget"

# vLLM validates a static maximum shape at startup. Runtime proposal returns
# only the real dynamic nodes and replaces this placeholder chain's bias before
# every target forward.
tree_choices="$(python - "$node_budget" <<'PY'
import sys
n = int(sys.argv[1])
print(str([(0,) * depth for depth in range(1, n + 1)]))
PY
)"
speculative_config="{\"method\":\"mtp\",\"model\":\"${DRAFT_MODEL_PATH}\",\"num_speculative_tokens\":${node_budget},\"speculative_token_tree\":\"$tree_choices\"}"

echo "[ReMTP][DynamicTree] D=$REMTP_DYNAMIC_TREE_MAX_DEPTH N_max=$node_budget children_max=$REMTP_DYNAMIC_TREE_MAX_CHILDREN sibling_ratio=$REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO"
echo "[ReMTP][DynamicTree] tau_min=$REMTP_DYNAMIC_TREE_TAU_MIN kappa=$REMTP_DYNAMIC_TREE_KAPPA mu=$REMTP_DYNAMIC_TREE_MU eta=$REMTP_DYNAMIC_TREE_ETA"
echo "[ReMTP][DynamicTree] alpha=$REMTP_DYNAMIC_TREE_ALPHA tau_relax=$REMTP_DYNAMIC_TREE_TAU_RELAX beta=$REMTP_DYNAMIC_TREE_BETA T_path=$REMTP_DYNAMIC_TREE_PATH_TEMPERATURE"
echo "[ReMTP][DynamicTree] mtp_route=$REMTP_DYNAMIC_TREE_MTP_ROUTE eos_ids=$REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS eos_threshold=$REMTP_DYNAMIC_TREE_EOS_THRESHOLD"
echo "[ReMTP][DynamicTree] audit=$REMTP_TREE_AUDIT_JSONL"

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --attention-backend TREE_ATTN \
  --worker-cls remtp.mimo_worker.MiMoDynamicTreeMTPWorker \
  --speculative-config "$speculative_config" \
  --no-async-scheduling \
  --enforce-eager \
  --port "$PORT"
