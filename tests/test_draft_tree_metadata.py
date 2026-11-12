"""Regression for old speculative slots leaking into MTP attention."""
from dataclasses import dataclass

import pytest
import torch

from remtp.fixed6_vllm import _draft_tree_metadata, _compact_draft_tree_bias


def test_singleton_tree_bias_has_dense_stride_despite_contiguous_view():
    original = torch.full((2, 2), float('inf'))
    original[1, 1] = 0
    sliced = original[1:, 1:].contiguous()
    assert sliced.is_contiguous() and sliced.stride(0) == 2
    safe = _compact_draft_tree_bias(sliced)
    assert safe.stride() == (1, 1) and safe.storage_offset() == 0
    assert safe.untyped_storage().nbytes() == safe.element_size()
    assert torch.equal(safe, sliced)
    assert torch.isinf(original[0, 0])


def test_already_dense_tree_bias_avoids_a_new_allocation():
    dense = torch.eye(6)
    assert _compact_draft_tree_bias(dense) is dense


@dataclass
class Metadata:
    seq_lens: torch.Tensor
    max_seq_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    num_actual_tokens: int
    max_query_len: int
    _seq_lens_cpu: torch.Tensor | None
    _num_computed_tokens_cpu: torch.Tensor | None


@pytest.mark.parametrize('count', [3, 6, 9, 12])
def test_draft_attention_excludes_stale_rejected_nodes(count):
    # Actual GPU reproduction: selected root=114, target seq_lens=123.
    base = Metadata(torch.tensor([123], dtype=torch.int32), 123,
                    torch.tensor([0, 10]), torch.tensor([0, 10]),
                    10, 10, torch.tensor([123]), torch.tensor([113]))
    result = _draft_tree_metadata(base, torch.tensor([0, count]), 114, count)
    prefix_length = int(result.seq_lens[0]) - result.max_query_len
    assert prefix_length == 115
    assert result.max_seq_len == 115 + count
    assert result.query_start_loc_cpu.tolist() == [0, count]
    assert result._seq_lens_cpu is None
    assert result._num_computed_tokens_cpu is None
    # Model prefix + tree positions contain no stale positions at their tail.
    visible_positions = torch.arange(int(result.seq_lens[0]))
    assert int(visible_positions[-1]) == 114 + count
    assert base.seq_lens.tolist() == [123]
    assert base.query_start_loc_cpu.tolist() == [0, 10]


def test_initial_prefill_length_remains_the_same():
    base = Metadata(torch.tensor([113]), 113, torch.tensor([0, 113]),
                    torch.tensor([0, 113]), 113, 113, None, None)
    result = _draft_tree_metadata(base, torch.tensor([0, 2]), 112, 2)
    assert result.seq_lens.tolist() == [115]
    assert result.max_seq_len == 115


def test_attention_is_invariant_to_rejected_cache_poisoning():
    # Two allocator histories: identical valid prefix and new tree nodes,
    # different bytes left behind by the previous rejected branches.
    root, count = 114, 3
    base = Metadata(torch.tensor([123]), 123, torch.tensor([0, 10]),
                    torch.tensor([0, 10]), 10, 10, None, None)
    meta = _draft_tree_metadata(base, torch.tensor([0, count]), root, count)
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(140, 8, generator=generator)
    values = torch.randn(140, 8, generator=generator)
    poisoned_keys, poisoned_values = keys.clone(), values.clone()
    poisoned_keys[root + 1 + count:] = 12
    poisoned_values[root + 1 + count:] = -25
    query = torch.randn(count, 8, generator=generator)

    def attend(length, k, v):
        # TreeAttention interprets seq_len-query_len as context length.
        prefix = length - count
        mask = torch.cat((torch.ones(count, prefix, dtype=torch.bool),
                          torch.eye(count, dtype=torch.bool)), dim=1)
        logits = query @ k[:length].T / 8**0.5
        return logits.masked_fill(~mask, -torch.inf).softmax(-1) @ v[:length]

    length = int(meta.seq_lens[0])
    assert torch.equal(attend(length, keys, values),
                       attend(length, poisoned_keys, poisoned_values))
    # The historical length computation fails this allocator-invariance test.
    historical_length = int(base.seq_lens[0]) + count
    assert not torch.allclose(attend(historical_length, keys, values),
                              attend(historical_length, poisoned_keys, poisoned_values))


# --- 4090 port (2026-11-12) -------------------------------------------------
@dataclass
class _MetadataWithCache(Metadata):
    _num_computed_tokens_cache: torch.Tensor | None = None


def test_device_computed_token_cache_is_cleared():
    base = _MetadataWithCache(torch.tensor([123]), 123, torch.tensor([0, 10]),
                              torch.tensor([0, 10]), 10, 10, torch.tensor([123]),
                              torch.tensor([113]), torch.tensor([113]))
    result = _draft_tree_metadata(base, torch.tensor([0, 3]), 114, 3)
    assert result._num_computed_tokens_cache is None
    assert base._num_computed_tokens_cache.tolist() == [113]


def test_legacy_switch_restores_historical_lengths(monkeypatch):
    from remtp.fixed6_vllm import _draft_common_metadata

    base = Metadata(torch.tensor([123]), 123, torch.tensor([0, 10]),
                    torch.tensor([0, 10]), 10, 10, None, None)
    monkeypatch.delenv("REMTP_DRAFT_TREE_LEGACY_METADATA", raising=False)
    assert _draft_common_metadata(base, torch.tensor([0, 3]), 114, 3).seq_lens.tolist() == [118]
    monkeypatch.setenv("REMTP_DRAFT_TREE_LEGACY_METADATA", "1")
    legacy = _draft_common_metadata(base, torch.tensor([0, 3]), 114, 3)
    assert legacy.seq_lens.tolist() == [126] and legacy.max_query_len == 3


def test_real_vllm_common_metadata_fields():
    backend = pytest.importorskip("vllm.v1.attention.backend")
    base = backend.CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 10], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 10], dtype=torch.int32),
        seq_lens=torch.tensor([123], dtype=torch.int32),
        num_reqs=1, num_actual_tokens=10, max_query_len=10, max_seq_len=123,
        block_table_tensor=torch.zeros((1, 8), dtype=torch.int32),
        slot_mapping=torch.arange(10, dtype=torch.int64),
        _seq_lens_cpu=torch.tensor([123], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([113], dtype=torch.int32),
    )
    result = _draft_tree_metadata(base, torch.tensor([0, 4], dtype=torch.int32), 114, 4)
    assert result.seq_lens.tolist() == [119] and result.seq_lens.dtype == torch.int32
    assert result.max_seq_len == 119 and result.max_query_len == 4
    assert result.num_actual_tokens == 4
    assert result.query_start_loc_cpu.tolist() == [0, 4]
    assert result.seq_lens_cpu.tolist() == [119]
    assert result.num_computed_tokens_cpu.tolist() == [115]
    assert result.compute_num_computed_tokens().tolist() == [115]
    assert base.seq_lens.tolist() == [123]
