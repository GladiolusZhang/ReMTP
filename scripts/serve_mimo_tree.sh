#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/cuda_env.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/MiMo-7B-Base-MTP3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-XiaomiMiMo/MiMo-7B-Base}"
TREE_TOPOLOGY="${TREE_TOPOLOGY:-2x2x2}"
TREE_VERIFY_MODE="${TREE_VERIFY_MODE:-strict}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
PORT="${PORT:-8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TREE_RUN_DIR="${TREE_RUN_DIR:-$PROJECT_DIR/results/mimo_tree_$RUN_TAG}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export REMTP_MIMO_MTP_LAYER_MODE=physical
export REMTP_ALLOW_EXPERIMENTAL_MIMO_MTP3=1
export REMTP_TREE_TOPOLOGY="$TREE_TOPOLOGY"
export REMTP_TREE_VERIFY_MODE="$TREE_VERIFY_MODE"
export REMTP_TREE_TOKENIZER="$MODEL_PATH"
export REMTP_TREE_TRACE="${REMTP_TREE_TRACE:-1}"
export REMTP_TREE_TRACE_MAX_ROUNDS="${REMTP_TREE_TRACE_MAX_ROUNDS:-128}"
export REMTP_TREE_AUDIT_JSONL="${REMTP_TREE_AUDIT_JSONL:-$TREE_RUN_DIR/rounds.jsonl}"
export REMTP_TREE_TRACE_LOG="${REMTP_TREE_TRACE_LOG:-$TREE_RUN_DIR/tree_trace.log}"
REMTP_TREE_REPORT_MD="${REMTP_TREE_REPORT_MD:-$TREE_RUN_DIR/summary.md}"

mkdir -p "$TREE_RUN_DIR"

write_tree_report() {
  if [[ -s "$REMTP_TREE_AUDIT_JSONL" ]]; then
    python -m remtp.mimo_tree_report \
      --audit "$REMTP_TREE_AUDIT_JSONL" \
      --output "$REMTP_TREE_REPORT_MD" || true
    echo "[ReMTP][MiMo tree] summary=$REMTP_TREE_REPORT_MD"
  fi
}
trap write_tree_report EXIT

python -m remtp.mimo_checkpoint check "$MODEL_PATH"

tree_choices="$(python - "$TREE_TOPOLOGY" <<'PY'
import sys
from remtp.binary_mtp_tree import get_mimo_tree_topology
from remtp.fixed6_vllm import topology_config_string
print(topology_config_string(get_mimo_tree_topology(sys.argv[1])))
PY
)"
node_budget="$(python - "$TREE_TOPOLOGY" <<'PY'
import sys
from remtp.binary_mtp_tree import get_mimo_tree_topology
print(len(get_mimo_tree_topology(sys.argv[1]).nodes))
PY
)"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}-Tree${node_budget}}"
python -m remtp.mimo_checkpoint logical-view \
  --source "$MODEL_PATH" \
  --output "$DRAFT_MODEL_PATH" \
  --width "$node_budget"
speculative_config="{\"method\":\"mtp\",\"model\":\"${DRAFT_MODEL_PATH}\",\"num_speculative_tokens\":${node_budget},\"speculative_token_tree\":\"$tree_choices\"}"

echo "[ReMTP][MiMo tree] topology=$TREE_TOPOLOGY verify=$TREE_VERIFY_MODE logical_nodes=$node_budget physical_mtp_layers=3"
echo "[ReMTP][MiMo tree] target_model=$MODEL_PATH draft_config_view=$DRAFT_MODEL_PATH"
echo "[ReMTP][MiMo tree] audit=$REMTP_TREE_AUDIT_JSONL"
echo "[ReMTP][MiMo tree] readable_trace=$REMTP_TREE_TRACE_LOG"
echo "[ReMTP][MiMo tree] summary_after_stop=$REMTP_TREE_REPORT_MD"
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --attention-backend TREE_ATTN \
  --worker-cls remtp.mimo_worker.MiMoTreeMTPWorker \
  --speculative-config "$speculative_config" \
  --enforce-eager \
  --port "$PORT"
