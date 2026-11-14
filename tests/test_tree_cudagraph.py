"""TREE_ATTN host metadata and persistent mask must not change any value (2026-11-13)."""
from __future__ import annotations

import pytest
import torch

from remtp.tree_cudagraph import attach_host_submetadata, write_persistent_mask

tree_attn = pytest.importorskip("vllm.v1.attention.backends.tree_attn")


class _Common:
    def __init__(self, max_query_len: int, max_seq_len: int) -> None:
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len


def _metadata(query_lens, seq_lens, *, decodes: int, bias=None):
    starts = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
    tokens = int(starts[-1])
    decode_tokens = int(starts[decodes])
    return tree_attn.TreeAttentionMetadata(
        num_actual_tokens=tokens, max_query_len=max(query_lens), query_start_loc=starts,
        max_seq_len=max(seq_lens), seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.arange(len(query_lens) * 4, dtype=torch.int32).reshape(len(query_lens), 4),
        slot_mapping=torch.arange(tokens, dtype=torch.int64),
        num_prefill_tokens=tokens - decode_tokens, num_decode_tokens=decode_tokens,
        num_prefills=len(query_lens) - decodes, num_decodes=decodes, tree_attn_bias=bias)


def _same(a, b) -> None:
    assert a.num_actual_tokens == b.num_actual_tokens
    assert a.max_query_len == b.max_query_len and a.max_seq_len == b.max_seq_len
    for name in ("query_start_loc", "seq_lens", "block_table", "slot_mapping"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert (a.tree_attn_bias is None) == (b.tree_attn_bias is None)
    if a.tree_attn_bias is not None:
        assert a.tree_attn_bias is b.tree_attn_bias


@pytest.mark.parametrize("query_len", [1, 4, 7, 10])
def test_decode_submetadata_matches_lazy_property(query_len: int) -> None:
    bias = torch.zeros(10, 10)
    lazy = _metadata([query_len], [133 + query_len], decodes=1, bias=bias)
    host = _metadata([query_len], [133 + query_len], decodes=1, bias=bias)
    attach_host_submetadata(host, _Common(query_len, 133 + query_len), check=True)
    assert host._cached_decode_metadata is not None and host.prefill_metadata is None
    _same(host.decode_metadata, lazy.decode_metadata)


def test_prefill_submetadata_matches_lazy_property() -> None:
    lazy = _metadata([57], [57], decodes=0)
    host = _metadata([57], [57], decodes=0)
    attach_host_submetadata(host, _Common(57, 57), check=True)
    assert host.decode_metadata is None
    _same(host.prefill_metadata, lazy.prefill_metadata)


def test_host_check_rejects_wrong_scalars() -> None:
    host = _metadata([5], [40], decodes=1, bias=torch.zeros(10, 10))
    with pytest.raises(RuntimeError):
        attach_host_submetadata(host, _Common(5, 41), check=True)


def test_mixed_batches_stay_lazy() -> None:
    host = _metadata([3, 20], [50, 20], decodes=1, bias=torch.zeros(10, 10))
    attach_host_submetadata(host, _Common(20, 50), check=True)
    assert host._cached_decode_metadata is None and host._cached_prefill_metadata is None


class _Builder:
    def __init__(self, rows: int) -> None:
        self._remtp_mask_buffer = torch.full((rows, rows), 7.0)
        self.tree_attn_bias = None
        self.reorder_batch_threshold = 0


def test_persistent_mask_copies_top_left_block() -> None:
    builder = _Builder(10)
    pointer = builder._remtp_mask_buffer.data_ptr()
    mask = tree_attn._prepare_tree_attn_bias([(0,), (1,), (0, 0)], [2, 1], dtype=torch.float32, device="cpu")
    write_persistent_mask(builder, mask)
    assert builder.tree_attn_bias.data_ptr() == pointer
    assert torch.equal(builder.tree_attn_bias[:4, :4], mask)
    assert builder.reorder_batch_threshold == 10
    with pytest.raises(RuntimeError):
        write_persistent_mask(builder, torch.zeros(11, 11))
    with pytest.raises(RuntimeError):
        write_persistent_mask(builder, torch.zeros(3, 3, dtype=torch.float64))


_TREES = [
    [(0,)],
    [(0,), (1,)],
    [(0,), (0, 0), (0, 0, 0)],
    [(0,), (1,), (0, 0), (0, 1), (1, 0)],
    [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)],
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the Triton kernel on CUDA")
@pytest.mark.parametrize("fp32_kernel", [False, True])
def test_kernel_output_bitwise_equal_with_persistent_mask(fp32_kernel: bool, monkeypatch) -> None:
    from vllm.v1.attention.ops import triton_unified_attention as ops

    if fp32_kernel:
        from remtp.target_attention_fp32 import kernel_unified_attention_2d

        monkeypatch.setattr(ops, "kernel_unified_attention_2d", kernel_unified_attention_2d)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(3)
    heads, kv_heads, head_size, block = 32, 8, 128, 16
    blocks = 32
    k_cache = torch.randn(blocks, block, kv_heads, head_size, generator=generator, device=device).bfloat16()
    v_cache = torch.randn(blocks, block, kv_heads, head_size, generator=generator, device=device).bfloat16()
    buffer = torch.full((10, 10), float("-inf"), device=device)
    for context in (5, 37, 200):
        for choices in _TREES:
            n = len(choices) + 1
            depth = [sum(1 for c in choices if len(c) == d) for d in range(1, 4)]
            depth = [d for d in depth if d]
            mask = tree_attn._prepare_tree_attn_bias(choices, depth, dtype=torch.float32, device=device)
            buffer.fill_(float("-inf"))
            buffer[:n, :n].copy_(mask)
            q = torch.randn(n, heads, head_size, generator=generator, device=device).bfloat16()
            outputs = []
            for bias in (mask, buffer):
                out = torch.empty_like(q)
                ops.unified_attention(
                    q=q, k=k_cache, v=v_cache, out=out,
                    cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32, device=device), max_seqlen_q=n,
                    seqused_k=torch.tensor([context + n], dtype=torch.int32, device=device),
                    max_seqlen_k=context + n, softmax_scale=head_size ** -0.5, causal=True,
                    window_size=(-1, -1), block_table=torch.arange(blocks, dtype=torch.int32, device=device)[None],
                    softcap=0, q_descale=None, k_descale=torch.ones(1, kv_heads, device=device),
                    v_descale=torch.ones(1, kv_heads, device=device), alibi_slopes=None, qq_bias=bias)
                outputs.append(out)
            assert torch.equal(outputs[0], outputs[1]), (context, n)
