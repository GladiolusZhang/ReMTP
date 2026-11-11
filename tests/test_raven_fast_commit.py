"""The fused KV path commit must move exactly what the per-layer reference moves."""
from __future__ import annotations

import pytest
import torch

from remtp.fixed6_vllm import _copy_cache_slots
from remtp.raven_fast_commit import copy_cache_slots_fused


class _Module:
    def __init__(self, cache: torch.Tensor):
        self.kv_cache = [cache]


def _caches(generator: torch.Generator, layers: int, blocks: int = 4, block_size: int = 16,
            heads: int = 2, dim: int = 8, dtype=torch.bfloat16) -> list[torch.Tensor]:
    return [torch.randn(2, blocks, block_size, heads, dim, generator=generator).to(dtype) for _ in range(layers)]


@pytest.mark.parametrize("pairs", [
    [(5, 1), (7, 2), (12, 3)],          # disjoint
    [(2, 1), (3, 2), (9, 3)],           # destination 2 is also a later source
    [(1, 1), (6, 2)],                   # identity pair
    [(40, 1)],                          # across blocks
    [],                                 # nothing selected
])
def test_fused_commit_matches_reference(pairs) -> None:
    generator = torch.Generator().manual_seed(3)
    reference = _caches(generator, layers=5)
    fused = [c.clone() for c in reference]
    sources = torch.tensor([s for s, _ in pairs], dtype=torch.int64)
    destinations = torch.tensor([d for _, d in pairs], dtype=torch.int64)
    for cache in reference:
        _copy_cache_slots(_Module(cache), sources, destinations)
    moved = copy_cache_slots_fused([_Module(c) for c in fused], pairs)
    assert moved == (len(pairs) * 2 * 5 if pairs else 0)
    for expected, actual in zip(reference, fused):
        assert torch.equal(expected, actual)


def test_fused_commit_skips_empty_caches_and_rejects_bad_shapes() -> None:
    generator = torch.Generator().manual_seed(4)
    good = _caches(generator, layers=2)
    empty = torch.empty(0)
    assert copy_cache_slots_fused([_Module(empty)] + [_Module(c) for c in good], [(3, 1)]) == 4
    with pytest.raises(RuntimeError):
        copy_cache_slots_fused([_Module(torch.zeros(3, 4, 5))], [(1, 0)])
