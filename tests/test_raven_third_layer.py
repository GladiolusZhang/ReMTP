"""Third-layer RAVEN overhead optimisations must keep the reference data flow."""
from __future__ import annotations

import itertools

import pytest
import torch

from remtp import dynamic_tree_vllm as dynamic
from remtp import fixed6_vllm as fixed
from remtp.fixed6_vllm import _copy_cache_slots
from remtp.raven_fast_commit import copy_cache_slots_inplace, inplace_order_is_safe


class _Module:
    def __init__(self, cache: torch.Tensor):
        self.kv_cache = [cache]


def _caches(seed: int, layers: int = 4) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 4, 16, 2, 8, generator=generator).to(torch.bfloat16) for _ in range(layers)]


def _reference(caches: list[torch.Tensor], pairs: list[tuple[int, int]]) -> None:
    sources = torch.tensor([s for s, _ in pairs], dtype=torch.int64)
    destinations = torch.tensor([d for _, d in pairs], dtype=torch.int64)
    for cache in caches:
        _copy_cache_slots(_Module(cache), sources, destinations)


def test_inplace_commit_matches_reference_for_every_tree_path() -> None:
    """Every strictly increasing BFS selection of 1..3 of 9 nodes, any base slot."""
    checked = 0
    for base in (0, 13, 30):
        for length in (1, 2, 3):
            for selected in itertools.combinations(range(9), length):
                pairs = [(base + 1 + index, base + 1 + offset) for offset, index in enumerate(selected)]
                assert inplace_order_is_safe([p for p in pairs if p[0] != p[1]])
                expected = _caches(7)
                actual = [c.clone() for c in expected]
                _reference(expected, pairs)
                copies = copy_cache_slots_inplace([_Module(c) for c in actual], pairs)
                moved = sum(1 for s, d in pairs if s != d)
                assert copies == moved * len(actual) * 2
                for left, right in zip(expected, actual):
                    assert torch.equal(left, right)
                checked += 1
    assert checked == 3 * (9 + 36 + 84)


@pytest.mark.parametrize("pairs", [
    [(2, 1), (1, 2)],            # swap: sequential order would be wrong
    [(3, 2), (2, 5)],            # first write lands on a later source
    [(4, 4)],                    # identity only
    [],
])
def test_inplace_commit_falls_back_when_order_is_unsafe(pairs) -> None:
    expected = _caches(11)
    actual = [c.clone() for c in expected]
    _reference(expected, pairs)
    copy_cache_slots_inplace([_Module(c) for c in actual], pairs)
    for left, right in zip(expected, actual):
        assert torch.equal(left, right)
    assert inplace_order_is_safe([(5, 1), (7, 2)])
    assert not inplace_order_is_safe([(2, 1), (1, 2)])


def test_inplace_commit_skips_empty_caches_and_rejects_bad_shapes() -> None:
    good = _caches(5, layers=2)
    assert copy_cache_slots_inplace([_Module(torch.empty(0))] + [_Module(c) for c in good], [(3, 1)]) == 4
    with pytest.raises(RuntimeError):
        copy_cache_slots_inplace([_Module(torch.zeros(3, 4, 5))], [(1, 0)])


def _reference_bias(choices, depth_counts, dtype, device):
    """vLLM's ``_prepare_tree_attn_bias`` (0.18), restated for a vLLM-free test."""
    size = len(choices) + 1
    mask = torch.full((size, size), -torch.inf, device=device, dtype=dtype)
    for i in range(size):
        mask[i, i] = 0
    mask[:, 0] = 0
    start = 0
    for count in depth_counts:
        for j in range(count):
            choice = choices[start + j]
            if len(choice) > 1:
                ancestors = [choices.index(choice[: c + 1]) + 1 for c in range(len(choice) - 1)]
                mask[j + start + 1, ancestors] = 0
        start += count
    return mask


def _depth_counts(choices):
    counts: list[int] = []
    previous = 0
    for path in choices:
        if len(path) != previous:
            counts.append(0)
        counts[len(path) - 1] += 1
        previous = len(path)
    return counts


TOPOLOGIES = [
    ((0,),),
    ((0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0,), (1,), (0, 0), (1, 0), (0, 0, 0)),
    ((0,), (0, 0), (0, 0, 0)),
]


@pytest.mark.parametrize("choices", TOPOLOGIES)
def test_cached_tree_bias_equals_direct_construction(choices) -> None:
    dynamic._TREE_BIAS_CACHE.clear()
    device = torch.device("cpu")
    direct = _reference_bias(list(choices), _depth_counts(list(choices)), torch.float32, device)
    first = dynamic._cached_tree_attn_bias(choices, torch.float32, device, _reference_bias, _depth_counts)
    second = dynamic._cached_tree_attn_bias(choices, torch.float32, device, _reference_bias, _depth_counts)
    assert torch.equal(first, direct)
    assert second is first                      # one construction per topology
    assert len(dynamic._TREE_BIAS_CACHE) == 1


def test_cached_tree_bias_matches_installed_vllm() -> None:
    tree_attn = pytest.importorskip("vllm.v1.attention.backends.tree_attn")
    dynamic._TREE_BIAS_CACHE.clear()
    device = torch.device("cpu")
    for choices in TOPOLOGIES:
        ordered = list(choices)
        direct = tree_attn._prepare_tree_attn_bias(
            ordered, tree_attn._get_depth_counts(ordered), dtype=torch.float32, device=device)
        cached = dynamic._cached_tree_attn_bias(
            choices, torch.float32, device, tree_attn._prepare_tree_attn_bias, tree_attn._get_depth_counts)
        assert torch.equal(cached, direct)


def test_offset_tensor_is_cached_only_with_the_flag(monkeypatch) -> None:
    fixed._OFFSET_CACHE.clear()
    monkeypatch.delenv("REMTP_DYNAMIC_TREE_FAST_METADATA", raising=False)
    plain = fixed._offset_tensor((1, 1, 2, 3), torch.int64, torch.device("cpu"))
    assert plain.tolist() == [1, 1, 2, 3] and not fixed._OFFSET_CACHE
    monkeypatch.setenv("REMTP_DYNAMIC_TREE_FAST_METADATA", "1")
    first = fixed._offset_tensor((1, 1, 2, 3), torch.int64, torch.device("cpu"))
    second = fixed._offset_tensor((1, 1, 2, 3), torch.int64, torch.device("cpu"))
    assert first is second and torch.equal(first, plain) and first.dtype == plain.dtype
    assert fixed._offset_tensor((1, 2), torch.int64, torch.device("cpu")) is not first


def test_path_first_row_order() -> None:
    assert fixed.path_first_row_order((), 10) == list(range(10))
    assert fixed.path_first_row_order((0,), 4) == [0, 1, 2, 3]
    assert fixed.path_first_row_order((0, 3, 6), 10) == [0, 1, 4, 7, 2, 3, 5, 6, 8, 9]
    assert fixed.path_first_row_order((1,), 4) == [0, 2, 1, 3]
    with pytest.raises(RuntimeError):
        fixed.path_first_row_order((9,), 10)


@pytest.mark.parametrize("two_dimensional", [False, True])
def test_path_ordered_first_pass_inputs(two_dimensional) -> None:
    rows = 10
    ids = torch.arange(100, 100 + rows)
    hidden = torch.arange(rows, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    depth = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    positions = 50 + depth
    if two_dimensional:
        positions = positions.unsqueeze(0).repeat(3, 1)
    kwargs = dict(target_token_ids=ids, target_positions=positions, target_hidden_states=hidden,
                  token_indices_to_sample=torch.tensor([7], dtype=torch.int32), other="kept")
    selected = (0, 3, 6)                       # rows 1, 4, 7
    updated = fixed._path_ordered_first_pass(kwargs, selected)
    order = [0, 1, 4, 7, 2, 3, 5, 6, 8, 9]
    assert updated["target_token_ids"].tolist() == [100 + r for r in order]
    assert updated["target_hidden_states"][:, 0].tolist() == [float(r) for r in order]
    row_positions = updated["target_positions"] if not two_dimensional else updated["target_positions"][0]
    assert row_positions[:4].tolist() == [50, 51, 52, 53]          # anchor and path are consecutive
    assert updated["token_indices_to_sample"].tolist() == [3]
    assert updated["token_indices_to_sample"].dtype == torch.int32
    assert updated["other"] == "kept"
    assert kwargs["target_token_ids"].tolist() == list(range(100, 110))   # caller's tensors untouched
    assert kwargs["token_indices_to_sample"].tolist() == [7]
