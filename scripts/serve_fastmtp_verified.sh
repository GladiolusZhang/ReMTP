#!/usr/bin/env bash
# Audited FastMTP service. METHOD=target|native|cactus|spec_cascade|dynamic_tree
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
source "$PROJECT_DIR/scripts/cuda_env.sh"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

METHOD="${METHOD:-native}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/FastMTP}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-TencentBAC/FastMTP}"
PORT="${PORT:-8000}"
MTP_TOKENS="${MTP_TOKENS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
NO_ASYNC_SCHEDULING="${NO_ASYNC_SCHEDULING:-1}"
FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-0}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "FastMTP checkpoint missing: $MODEL_PATH" >&2
  echo "Run ./scripts/download_fastmtp.sh first." >&2
  exit 2
fi
python -m remtp.fastmtp_checkpoint check "$MODEL_PATH"

common=(
  vllm serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --tensor-parallel-size 1
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs 1
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --port "$PORT"
)
[[ "$ENFORCE_EAGER" == "1" ]] && common+=(--enforce-eager)
[[ "$NO_ASYNC_SCHEDULING" == "1" ]] && common+=(--no-async-scheduling)
if [[ "$FAST_MTP_NO_THINK" == "1" ]]; then
  common+=(
    --chat-template "$PROJECT_DIR/configs/fastmtp_no_think_chat_template.jinja"
    --default-chat-template-kwargs '{"enable_thinking":false}'
  )
  echo "[ReMTP][FastMTPVerified] no_think=enabled template=MiMo-empty-think-prefill"
elif [[ "$FAST_MTP_NO_THINK" != "0" ]]; then
  echo "FAST_MTP_NO_THINK must be 0 or 1; received $FAST_MTP_NO_THINK" >&2
  exit 2
fi

export REMTP_MIMO_MTP_LAYER_MODE=layer0

case "$METHOD" in
  target)
    echo "[ReMTP][FastMTPVerified] method=target speculative_decoding=off"
    exec "${common[@]}"
    ;;
  native)
    worker=remtp.fastmtp_verified_worker.FastMTPVerifiedNativeWorker
    ;;
  cactus)
    worker=remtp.fastmtp_verified_worker.FastMTPVerifiedCactusWorker
    export REMTP_CACTUS_DELTA="${REMTP_CACTUS_DELTA:-1.0}"
    ;;
  spec_cascade)
    worker=remtp.fastmtp_verified_worker.FastMTPVerifiedSpecCascadeWorker
    export REMTP_CASCADE_RULE="${REMTP_CASCADE_RULE:-token_v3}"
    export REMTP_CASCADE_ALPHA="${REMTP_CASCADE_ALPHA:-0.5}"
    ;;
  dynamic_tree)
    worker=remtp.fastmtp_verified_worker.FastMTPVerifiedDynamicTreeWorker
    export REMTP_DYNAMIC_TREE_MTP_ROUTE=repeat_000
    export REMTP_DYNAMIC_TREE_SUPPORT_MODE="${REMTP_DYNAMIC_TREE_SUPPORT_MODE:-relative}"
    if [[ "$REMTP_DYNAMIC_TREE_SUPPORT_MODE" == "cactus_trunk_rescue" ]]; then
      export REMTP_TREE_VERIFY_MODE=cactus_trunk_rescue
      verifier_description="sampled-Q-Cactus-trunk+rejection-only-tree-rescue (approximate rescue)"
    elif [[ "$REMTP_DYNAMIC_TREE_SUPPORT_MODE" == "sampled_primary_reopen" ]]; then
      export REMTP_TREE_VERIFY_MODE=sampled_primary_reopen
      verifier_description="sampled-Q-Cactus-primary+rejection-tree-reopen (approximate recovery)"
    elif [[ "$REMTP_DYNAMIC_TREE_SUPPORT_MODE" == "sampled_primary_shadow" ]]; then
      export REMTP_TREE_VERIFY_MODE=sampled_primary_shadow
      verifier_description="sampled-Q-Cactus-primary+exact-residual-correction (tree shadow)"
    elif [[ "$REMTP_DYNAMIC_TREE_SUPPORT_MODE" == "residual_hit_anchor" ]]; then
      export REMTP_TREE_VERIFY_MODE=residual_hit_anchor
      verifier_description="sampled-Q-Cactus-primary+exact-residual-hit+target-anchor"
    elif [[ "$REMTP_DYNAMIC_TREE_SUPPORT_MODE" == "residual_hit_strict" ]]; then
      export REMTP_TREE_VERIFY_MODE=residual_hit_strict
      verifier_description="sampled-Q-Cactus-primary+exact-residual-hit+strict-continuation"
    elif [[ "$REMTP_DYNAMIC_TREE_SUPPORT_MODE" == "residual_hit_cactus" ]]; then
      export REMTP_TREE_VERIFY_MODE=residual_hit_cactus
      verifier_description="sampled-Q-Cactus-primary+exact-residual-hit+Cactus-continuation"
    elif [[ "$REMTP_DYNAMIC_TREE_SUPPORT_MODE" == "target_path_rescue" ]]; then
      export REMTP_TREE_VERIFY_MODE=target_path_rescue
      verifier_description="ordinary-relaxed-longest-path+one-token-rescue-extension (approximate)"
    elif [[ "$REMTP_DYNAMIC_TREE_SUPPORT_MODE" == "prefix_reopen_rescue" ]]; then
      export REMTP_TREE_VERIFY_MODE=prefix_reopen_rescue
      verifier_description="front-loaded-multi-path-rescue+ordinary-continuation (approximate)"
    else
      export REMTP_TREE_VERIFY_MODE=target_dynamic_relaxed
      verifier_description="target-dominant-relaxed (not distribution-preserving)"
    fi
    export REMTP_TREE_TOKENIZER="$MODEL_PATH"
    export REMTP_DYNAMIC_TREE_MAX_DEPTH="${REMTP_DYNAMIC_TREE_MAX_DEPTH:-3}"
    export REMTP_DYNAMIC_TREE_MAX_NODES="${REMTP_DYNAMIC_TREE_MAX_NODES:-6}"
    export REMTP_DYNAMIC_TREE_ADAPTIVE_BASE_NODES="${REMTP_DYNAMIC_TREE_ADAPTIVE_BASE_NODES:-0}"
    export REMTP_DYNAMIC_TREE_ADAPTIVE_ENTROPY_THRESHOLD="${REMTP_DYNAMIC_TREE_ADAPTIVE_ENTROPY_THRESHOLD:-1.0}"
    export REMTP_DYNAMIC_TREE_MAX_CHILDREN="${REMTP_DYNAMIC_TREE_MAX_CHILDREN:-2}"
    export REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO="${REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO:-0.25}"
    export REMTP_DYNAMIC_TREE_TAU_MIN="${REMTP_DYNAMIC_TREE_TAU_MIN:-0.02}"
    export REMTP_DYNAMIC_TREE_KAPPA="${REMTP_DYNAMIC_TREE_KAPPA:-1.0}"
    export REMTP_DYNAMIC_TREE_MU="${REMTP_DYNAMIC_TREE_MU:-0.5}"
    export REMTP_DYNAMIC_TREE_ETA="${REMTP_DYNAMIC_TREE_ETA:-0.25}"
    # The verified default uses coverage only as a parent veto.  The old
    # geometric mode mechanically rewarded wider sibling sets.
    export REMTP_DYNAMIC_TREE_COVERAGE_MODE="${REMTP_DYNAMIC_TREE_COVERAGE_MODE:-coverage_gate}"
    export REMTP_DYNAMIC_TREE_MIN_TARGET_COVERAGE="${REMTP_DYNAMIC_TREE_MIN_TARGET_COVERAGE:-0.05}"
    export REMTP_DYNAMIC_TREE_ALPHA="${REMTP_DYNAMIC_TREE_ALPHA:-0.5}"
    export REMTP_DYNAMIC_TREE_TAU_RELAX="${REMTP_DYNAMIC_TREE_TAU_RELAX:-0.50}"
    export REMTP_DYNAMIC_TREE_BETA="${REMTP_DYNAMIC_TREE_BETA:-0.25}"
    export REMTP_DYNAMIC_TREE_PATH_TEMPERATURE="${REMTP_DYNAMIC_TREE_PATH_TEMPERATURE:-0.10}"
    export REMTP_DYNAMIC_TREE_PATH_SELECTION="${REMTP_DYNAMIC_TREE_PATH_SELECTION:-longest}"
    export REMTP_DYNAMIC_TREE_FRONTIER_RESCUE="${REMTP_DYNAMIC_TREE_FRONTIER_RESCUE:-1}"
    export REMTP_DYNAMIC_TREE_RESCUE_DELTA="${REMTP_DYNAMIC_TREE_RESCUE_DELTA:-0.5}"
    export REMTP_DYNAMIC_TREE_RESCUE_SCORE_THRESHOLD="${REMTP_DYNAMIC_TREE_RESCUE_SCORE_THRESHOLD:-0.35}"
    export REMTP_DYNAMIC_TREE_RESCUE_CONTINUATION_DISCOUNT="${REMTP_DYNAMIC_TREE_RESCUE_CONTINUATION_DISCOUNT:-0.0}"
    export REMTP_DYNAMIC_TREE_RESCUE_CONTINUATION_MIN_DEPTH="${REMTP_DYNAMIC_TREE_RESCUE_CONTINUATION_MIN_DEPTH:-2}"
    export REMTP_DYNAMIC_TREE_RESCUE_MIN_RELATIVE="${REMTP_DYNAMIC_TREE_RESCUE_MIN_RELATIVE:-0.1}"
    export REMTP_DYNAMIC_TREE_RESCUE_MIN_TARGET_PROB="${REMTP_DYNAMIC_TREE_RESCUE_MIN_TARGET_PROB:-0.001}"
    export REMTP_DYNAMIC_TREE_APPEND_ANCHOR="${REMTP_DYNAMIC_TREE_APPEND_ANCHOR:-1}"
    export REMTP_DYNAMIC_TREE_EOS_THRESHOLD="${REMTP_DYNAMIC_TREE_EOS_THRESHOLD:-0.5}"
    if [[ -z "${REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS:-}" ]]; then
      REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS="$(python - "$MODEL_PATH/config.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8")).get("eos_token_id", [])
if value is None:
    value = []
if isinstance(value, int):
    value = [value]
print(",".join(str(int(item)) for item in value))
PY
)"
    fi
    export REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS
    export REMTP_TREE_AUDIT_JSONL="${REMTP_TREE_AUDIT_JSONL:-$PROJECT_DIR/results/fastmtp_dynamic_tree_rounds.jsonl}"
    export REMTP_TREE_TRACE="${REMTP_TREE_TRACE:-0}"
    export REMTP_TREE_AUDIT_DETAIL="${REMTP_TREE_AUDIT_DETAIL:-0}"
    mkdir -p "$(dirname "$REMTP_TREE_AUDIT_JSONL")"
    : > "$REMTP_TREE_AUDIT_JSONL"

    node_budget="$REMTP_DYNAMIC_TREE_MAX_NODES"
    DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_PATH}-Tree${node_budget}}"
    python -m remtp.fastmtp_checkpoint logical-view \
      --source "$MODEL_PATH" --output "$DRAFT_MODEL_PATH" --width "$node_budget"
    tree_choices="$(python - "$node_budget" <<'PY'
import sys
n = int(sys.argv[1])
print(str([(0,) * depth for depth in range(1, n + 1)]))
PY
)"
    speculative_config="{\"method\":\"mtp\",\"model\":\"${DRAFT_MODEL_PATH}\",\"num_speculative_tokens\":${node_budget},\"speculative_token_tree\":\"$tree_choices\"}"
    echo "[ReMTP][FastMTPVerified] method=dynamic_tree physical_heads=1 route=repeat_0 logical_nodes=$node_budget children_max=$REMTP_DYNAMIC_TREE_MAX_CHILDREN sibling_ratio=$REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO"
    echo "[ReMTP][FastMTPVerified] verifier=$verifier_description support_mode=$REMTP_DYNAMIC_TREE_SUPPORT_MODE coverage_mode=$REMTP_DYNAMIC_TREE_COVERAGE_MODE path_selection=$REMTP_DYNAMIC_TREE_PATH_SELECTION beta=$REMTP_DYNAMIC_TREE_BETA frontier_rescue=$REMTP_DYNAMIC_TREE_FRONTIER_RESCUE rescue_delta=$REMTP_DYNAMIC_TREE_RESCUE_DELTA"
    exec "${common[@]}" \
      --attention-backend TREE_ATTN \
      --worker-cls "$worker" \
      --speculative-config "$speculative_config"
    ;;
  *)
    echo "Unknown METHOD=$METHOD; choose target, native, cactus, spec_cascade, dynamic_tree" >&2
    exit 2
    ;;
esac

speculative_config="{\"method\":\"mtp\",\"model\":\"${MODEL_PATH}\",\"num_speculative_tokens\":${MTP_TOKENS}}"
echo "[ReMTP][FastMTPVerified] method=$METHOD worker=$worker physical_heads=1 route=repeat_0 draft_tokens=$MTP_TOKENS"
exec "${common[@]}" \
  --worker-cls "$worker" \
  --speculative-config "$speculative_config"
