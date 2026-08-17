#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/scripts/cuda_env.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
TREE_TOPOLOGY="${TREE_TOPOLOGY:-6-chain}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
TREE_AUDIT_JSONL="${TREE_AUDIT_JSONL:-$PROJECT_DIR/results/fixed6_microtree_${TREE_TOPOLOGY//+/_}_rounds.jsonl}"

case "$TREE_TOPOLOGY" in
  6-chain)
    TREE_CHOICES='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)]'
    ;;
  4+2)
    TREE_CHOICES='[(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (0, 0, 0, 0)]'
    ;;
  3+2+1)
    TREE_CHOICES='[(0,), (1,), (2,), (0, 0), (1, 0), (0, 0, 0)]'
    ;;
  *)
    echo "TREE_TOPOLOGY must be 6-chain, 4+2, or 3+2+1" >&2
    exit 2
    ;;
esac

if [[ "$TREE_TOPOLOGY" != "6-chain" && "${REMTP_ALLOW_UNVERIFIED_GDN_TREE:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to start an unverified Qwen3.5 GDN micro-tree.

The 6-chain control uses vLLM's native recurrent-state path. The complete
non-chain correctness/performance gate has not passed: 4+2 has a matching
reference probe but is too slow, while 3+2+1 still has a reference mismatch.

For state-validation/debugging only, explicitly set:
  REMTP_ALLOW_UNVERIFIED_GDN_TREE=1

Do not use that override for benchmark or quality claims.
EOF
  exit 3
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export REMTP_TREE_TOPOLOGY="$TREE_TOPOLOGY"
export REMTP_TREE_AUDIT_JSONL="$TREE_AUDIT_JSONL"

python -m remtp.checkpoint "$MODEL_PATH"
mkdir -p "$(dirname "$TREE_AUDIT_JSONL")"
rm -f "$TREE_AUDIT_JSONL"

SPEC_CONFIG=$(python - "$TREE_CHOICES" <<'PY'
import json, sys
print(json.dumps({
    "method": "mtp",
    "num_speculative_tokens": 6,
    "speculative_token_tree": sys.argv[1],
}))
PY
)

echo "[Fixed-6] topology=$TREE_TOPOLOGY nodes=6 target_calls_per_round=1"
exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --language-model-only \
  --enforce-eager \
  --no-async-scheduling \
  --attention-backend TREE_ATTN \
  --worker-cls remtp.worker.Fixed6MicroTreeWorker \
  --speculative-config "$SPEC_CONFIG"
