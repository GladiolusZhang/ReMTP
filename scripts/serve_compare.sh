#!/usr/bin/env bash
# One server for one of the compared methods, with the same configuration as the
# 2026-11-14/15 Jetson experiments but no Jetson-specific settings (2026-11-15).
#
#   scripts/serve_compare.sh target|native|cactus|raven
#
# Defaults: BF16 FastMTP checkpoint in models/FastMTP, port 8000, no-think chat
# template, one request at a time.  Every method gets the same attention
# kernels (Triton) and target numerics switches.  ENGINE_OPT=1 (default) adds
# the token-identical engineering configuration of doc/RESULT-RAVEN-V4-TARGET-
# GATE-2026-11-13.md sections 10-11: full CUDA graphs for every method and spin
# waiting; RAVEN additionally capturable tree attention, async host reads and
# device-side tree construction.  ENGINE_OPT=0 runs everything eagerly.
#
# Override with environment variables: MODEL_PATH, QUANTIZATION, LOAD_FORMAT,
# PORT, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, ENGINE_OPT, SPIN_WAIT, GPU_BUILD.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${1:-${COMPARE_METHOD:-}}"
case "$name" in
  target|native|cactus|raven) ;;
  *) echo "usage: $0 target|native|cactus|raven" >&2; exit 2 ;;
esac

export MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/FastMTP}"
export QUANTIZATION="${QUANTIZATION:-}"
export LOAD_FORMAT="${LOAD_FORMAT:-}"
export PORT="${PORT:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export ENABLE_PREFIX_CACHING=0
# ASYNC_SCHEDULING=1 (2026-11-15): vLLM async scheduling, baselines only.  RAVEN's
# tree, verification and KV commit are decided on the host between steps, so it
# must run synchronously until verification moves to the GPU.
if [[ "${ASYNC_SCHEDULING:-0}" == "1" ]]; then
  if [[ "$name" == "raven" ]]; then
    echo "ASYNC_SCHEDULING=1 is not supported for raven yet (host-side tree/verification); use it for target/native/cactus only" >&2
    exit 2
  fi
  export NO_ASYNC_SCHEDULING=0
else
  export NO_ASYNC_SCHEDULING=1
fi
export FAST_MTP_NO_THINK="${FAST_MTP_NO_THINK:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export MTP_TOKENS=3
if [[ -n "$QUANTIZATION" && "$QUANTIZATION" != "bitsandbytes" ]]; then
  # A 4-bit target (e.g. gptq_marlin) keeps the trained MTP layer in BF16.
  export REMTP_JETSON_BNB_DRAFT=1
fi

# Same kernels and target numerics for every method (as in the Jetson runs).
export TARGET_ATTENTION_BACKEND=TRITON_ATTN
export REMTP_TARGET_TRITON_2D=1
export REMTP_TARGET_ATTENTION_FP32=1
export REMTP_TARGET_BF16_FULL_ACCUM=1
export REMTP_TARGET_RMSNORM_ROW_LIMIT=255
export REMTP_TARGET_LM_HEAD_ROW_BLOCK=16
export REMTP_CANONICAL_TREE_ATTENTION=0
export REMTP_HEAD_BATCH_ROWS=1

ENGINE_OPT="${ENGINE_OPT:-1}"
if [[ "$ENGINE_OPT" == "1" ]]; then
  export ENFORCE_EAGER=0
  export REMTP_CUDA_SPIN_WAIT="${SPIN_WAIT:-1}"
else
  export ENFORCE_EAGER=1
  unset REMTP_COMPILATION_CONFIG
fi
graph_config() {  # $1 = mode, $2 = JSON list of capture sizes
  [[ "$ENGINE_OPT" == "1" ]] && export REMTP_COMPILATION_CONFIG="{\"mode\":0,\"cudagraph_mode\":\"$1\",\"cudagraph_capture_sizes\":$2}"
  return 0
}

case "$name" in
  target)
    export METHOD=target
    graph_config FULL_DECODE_ONLY "[1]"
    ;;
  native)
    export METHOD=native
    graph_config FULL_DECODE_ONLY "[4]"
    ;;
  cactus)
    export METHOD=cactus
    export REMTP_CACTUS_DELTA=1.0
    graph_config FULL_DECODE_ONLY "[4]"
    ;;
  raven)
    export METHOD=dynamic_tree
    # RAVEN v4-safe: D3/N9 residual-hit tree, Cactus-sized allowance only where the
    # target supports the drafted token (p >= 0.01), never above Cactus per position.
    export REMTP_DYNAMIC_TREE_SUPPORT_MODE=residual_hit_anchor
    export REMTP_DYNAMIC_TREE_MAX_DEPTH=3
    export REMTP_DYNAMIC_TREE_MAX_NODES=9
    export REMTP_DYNAMIC_TREE_MAX_CHILDREN=3
    export REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO=0.01
    export REMTP_DYNAMIC_TREE_TAU_MIN=0.001
    export REMTP_DYNAMIC_TREE_KAPPA=1.5
    export REMTP_DYNAMIC_TREE_MU=0.5
    export REMTP_DYNAMIC_TREE_ETA=0.0
    export REMTP_DYNAMIC_TREE_ALLOCATION=residual_coverage
    export REMTP_DYNAMIC_TREE_CACTUS_DELTA=1.0
    export REMTP_DYNAMIC_TREE_RELAX_DEPTH_WEIGHTS=1.0,1.0,1.0
    export REMTP_DYNAMIC_TREE_ADAPTIVE_RELAXATION=1
    export REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MIX=0.5
    export REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MAX_RATIO=1.0
    export REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_TARGET_POWER=0.25
    export REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_STEPS=16
    export REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_PROB=0.01
    export REMTP_DYNAMIC_TREE_RELAX_MAX_PROBABILITY_RATIO=0
    export REMTP_DYNAMIC_TREE_RELAX_RISK_POLICY=none
    export REMTP_DYNAMIC_TREE_FRONTIER_RESCUE=0
    export REMTP_DYNAMIC_TREE_PATH_SELECTION=balanced
    export REMTP_DYNAMIC_TREE_PATH_TEMPERATURE=0
    export REMTP_DYNAMIC_TREE_PROTECT_EOS=0
    export REMTP_DYNAMIC_TREE_REPEAT_GUARD=0
    export REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS="${REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS:-151645,151643}"   # as in the Jetson runs
    export REMTP_DYNAMIC_TREE_PRUNE_ANCHOR_DESCENDANTS=0
    export REMTP_DYNAMIC_TREE_ANCHOR_ONLY_PROPOSAL=0
    export REMTP_DYNAMIC_TREE_EXACT_TOPK=0
    export REMTP_DYNAMIC_TREE_LAZY_RESIDUAL=0
    export REMTP_REQUEST_LOCAL_BRANCH_RNG=1
    export REMTP_TREE_AUDIT_JSONL=""
    # Overhead switches, all token-identical (see doc/PORTING-RAVEN-V4-2026-11-14.md).
    export REMTP_DYNAMIC_TREE_FAST_SELECT=1
    export REMTP_DYNAMIC_TREE_FAST_VERIFY=1
    export REMTP_DYNAMIC_TREE_FAST_COMMIT=2
    export REMTP_DYNAMIC_TREE_FAST_METADATA=1
    export REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV=1
    export REMTP_DYNAMIC_TREE_BATCH_SELECT=1
    if [[ "$ENGINE_OPT" == "1" ]]; then
      export REMTP_TREE_CUDAGRAPH=1
      export REMTP_DYNAMIC_TREE_ASYNC_HOST=1
      export REMTP_DYNAMIC_TREE_GPU_BUILD="${GPU_BUILD:-1}"
      graph_config FULL "[1,2,3,4,5,6,7,8,9,10]"
    fi
    ;;
esac

echo "[ReMTP][Compare] method=$name model=$MODEL_PATH quantization=${QUANTIZATION:-none} engine_opt=$ENGINE_OPT async_scheduling=$([[ $NO_ASYNC_SCHEDULING == 0 ]] && echo 1 || echo 0) port=$PORT"
exec "$PROJECT_DIR/scripts/serve_fastmtp_verified.sh"
