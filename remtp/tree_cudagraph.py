"""Host-side TREE_ATTN changes for sync-free and CUDA-graph target forwards.

Engineering only: no kernel, mask value, tensor value or sampling change.

``install_tree_host_metadata`` (REMTP_TREE_HOST_METADATA=1)
    vLLM 0.18 derives the decode/prefill sub-metadata lazily inside the first
    attention layer with two ``.item()`` calls (max query length and max KV
    length), i.e. two device synchronisations in the middle of every target and
    MTP forward.  For a batch that is all decodes or all prefills (always the
    case with ``max_num_seqs=1``) the same maxima are already on the host in
    ``CommonAttentionMetadata``; build the sub-metadata there.  The tensors
    passed to the kernel are the same views as before (decode) or equal values
    (prefill offsets, which start at 0).  The Triton 2D kernel used by TREE_ATTN
    reads neither maximum; they are only launch hints.
    ``REMTP_TREE_HOST_METADATA_CHECK=1`` recomputes both maxima on the device
    and raises on any difference (validation runs only; it re-adds the syncs).

``install_tree_cudagraph`` (REMTP_TREE_CUDAGRAPH=1, implies host metadata)
    * the target's tree mask lives in one persistent (N+1)x(N+1) buffer, the
      builder's original mask tensor.  Each round's mask (n<=N+1 rows) is
      copied into its top-left block.  The kernel addresses the mask through
      its row stride only; entries outside the n x n block belong to keys that
      the causal mask already excludes (they are added to -inf), so the
      attention logits are bitwise unchanged;
    * TREE_ATTN declares full CUDA-graph support.  Valid here because every
      batch holds exactly one request (``max_num_seqs=1``); the shapes, the
      mask buffer and all metadata tensors are persistent between capture and
      replay.  Prefill runs eagerly (it exceeds the largest capture size).
"""
from __future__ import annotations

import functools
import os
from typing import Any

import torch

_HOST_METADATA = False
_CUDAGRAPH = False


def host_metadata_enabled() -> bool:
    return _HOST_METADATA


def tree_cudagraph_enabled() -> bool:
    return _CUDAGRAPH


def _check_against_device(sub: Any) -> None:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    lengths = torch.diff(sub.query_start_loc)
    device_q = int(lengths.max().item())
    device_k = int(sub.seq_lens.max().item())
    if (device_q, device_k) != (sub.max_query_len, sub.max_seq_len):
        raise RuntimeError(
            "tree host metadata differs from device values: "
            f"host=({sub.max_query_len},{sub.max_seq_len}) device=({device_q},{device_k})")


def attach_host_submetadata(metadata: Any, common: Any, *, check: bool = False) -> None:
    """Fill the cached decode/prefill sub-metadata without device reads."""
    from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata

    if metadata.num_decodes and not metadata.num_prefills:
        n = metadata.num_decodes
        metadata._cached_decode_metadata = TreeAttentionMetadata(
            num_actual_tokens=metadata.num_decode_tokens,
            max_query_len=int(common.max_query_len),
            query_start_loc=metadata.query_start_loc[: n + 1],
            max_seq_len=int(common.max_seq_len),
            seq_lens=metadata.seq_lens[:n],
            block_table=metadata.block_table[:n],
            slot_mapping=metadata.slot_mapping[: metadata.num_decode_tokens],
            tree_attn_bias=metadata.tree_attn_bias,
        )
        if check:
            _check_against_device(metadata._cached_decode_metadata)
    elif metadata.num_prefills and not metadata.num_decodes:
        # query_start_loc[0] == 0, so ``q - q[0]`` of the lazy path equals q.
        metadata._cached_prefill_metadata = TreeAttentionMetadata(
            num_actual_tokens=metadata.num_prefill_tokens,
            max_query_len=int(common.max_query_len),
            query_start_loc=metadata.query_start_loc,
            max_seq_len=int(common.max_seq_len),
            seq_lens=metadata.seq_lens,
            block_table=metadata.block_table,
            slot_mapping=metadata.slot_mapping,
        )
        if check:
            sub = metadata._cached_prefill_metadata
            if int(sub.query_start_loc[0].item()) != 0:
                raise RuntimeError("prefill query_start_loc must start at 0")
            _check_against_device(sub)


def install_tree_host_metadata() -> None:
    global _HOST_METADATA
    if _HOST_METADATA:
        return
    from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadataBuilder

    check = os.getenv("REMTP_TREE_HOST_METADATA_CHECK", "0") == "1"
    original = TreeAttentionMetadataBuilder.build

    @functools.wraps(original)
    def build(self: Any, common_prefix_len: int, common_attn_metadata: Any,
              fast_build: bool = False) -> Any:
        metadata = original(self, common_prefix_len, common_attn_metadata, fast_build)
        attach_host_submetadata(metadata, common_attn_metadata, check=check)
        return metadata

    TreeAttentionMetadataBuilder.build = build
    _HOST_METADATA = True
    print("[ReMTP][DynamicTree] tree attention sub-metadata from host scalars "
          f"(no .item() syncs){' + device check' if check else ''}", flush=True)


def install_tree_cudagraph() -> None:
    global _CUDAGRAPH
    if _CUDAGRAPH:
        return
    install_tree_host_metadata()
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadataBuilder

    original_init = TreeAttentionMetadataBuilder.__init__

    @functools.wraps(original_init)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # The original (N+1)x(N+1) float32 mask becomes the persistent buffer.
        self._remtp_mask_buffer = self.tree_attn_bias.contiguous()
        self.tree_attn_bias = self._remtp_mask_buffer

    TreeAttentionMetadataBuilder.__init__ = init
    TreeAttentionMetadataBuilder._cudagraph_support = AttentionCGSupport.ALWAYS
    _CUDAGRAPH = True
    print("[ReMTP][DynamicTree] tree attention CUDA graphs: persistent mask buffer, "
          "full-graph support (max_num_seqs=1)", flush=True)


def write_persistent_mask(builder: Any, mask: torch.Tensor) -> None:
    """Copy ``mask`` into the builder's persistent buffer and point the builder at it."""
    buffer = getattr(builder, "_remtp_mask_buffer", None)
    if buffer is None:
        raise RuntimeError("tree CUDA graphs: builder has no persistent mask buffer")
    rows = mask.shape[0]
    if mask.shape != (rows, rows) or rows > buffer.shape[0] or mask.dtype != buffer.dtype:
        raise RuntimeError(
            f"tree mask {tuple(mask.shape)}/{mask.dtype} does not fit buffer "
            f"{tuple(buffer.shape)}/{buffer.dtype}")
    buffer[:rows, :rows].copy_(mask, non_blocking=True)
    builder.tree_attn_bias = buffer
    builder.reorder_batch_threshold = buffer.shape[0]
