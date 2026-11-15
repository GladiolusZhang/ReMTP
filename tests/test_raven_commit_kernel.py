"""The one-launch KV path commit moves exactly the slots of copy_cache_slots_inplace."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _modules(layers=5, blocks=6, block_size=16, heads=2, dim=64, dtype=torch.bfloat16, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [SimpleNamespace(kv_cache=[torch.randn(2, blocks, block_size, heads, dim, device="cuda",
                                                  generator=generator).to(dtype)])
            for _ in range(layers)]


def _paths(rng, base, count=40, nodes=9):
    for _ in range(count):
        depth = rng.randint(1, 4)
        selected = tuple(sorted(rng.sample(range(nodes), depth)))
        slots = list(range(base, base + nodes + 1))
        if rng.random() < 0.3:  # non-contiguous slots (e.g. a block boundary)
            tail = slots[1:]
            rng.shuffle(tail)
            slots[1:] = tail
        yield selected, slots


def test_kernel_matches_inplace_reference():
    from remtp.raven_commit_kernel import LayerTable, attention_caches
    from remtp.raven_fast_commit import copy_cache_slots_inplace

    rng = random.Random(7)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        reference = _modules(dtype=dtype)
        candidate = _modules(dtype=dtype)
        table = LayerTable(attention_caches(candidate))
        for selected, slots in _paths(rng, base=rng.randint(0, 80)):
            pairs = [(slots[index + 1], slots[1 + offset]) for offset, index in enumerate(selected)]
            copy_cache_slots_inplace(reference, pairs)
            sources = torch.tensor([s for s, _ in pairs], dtype=torch.int64, device="cuda")
            destinations = torch.tensor([d for _, d in pairs], dtype=torch.int64, device="cuda")
            table.move(sources, destinations)
            for a, b in zip(reference, candidate):
                assert torch.equal(a.kv_cache[0], b.kv_cache[0])


def test_commit_selected_slots_kernel_uses_runtime_slots(monkeypatch):
    from vllm.model_executor.layers.attention.attention import Attention

    from remtp import fixed6_vllm as fixed
    from remtp import raven_commit_kernel as kernel
    from remtp.raven_fast_commit import copy_cache_slots_inplace

    reference = _modules(seed=3)
    candidate = _modules(seed=3)
    # Stand-in runner: the module scan only needs isinstance(Attention) objects.
    monkeypatch.setattr(kernel, "target_table", lambda runner: kernel.LayerTable(kernel.attention_caches(candidate)))
    slots = torch.arange(40, 50, dtype=torch.int64, device="cuda")
    monkeypatch.setattr(fixed._RUNTIME, "target_slots", slots)
    monkeypatch.setattr(fixed._RUNTIME, "draft_slots", None)
    selected = (0, 3, 7)
    kernel.commit_selected_slots_kernel(SimpleNamespace(), selected)
    pairs = [(int(slots[index + 1]), int(slots[1 + offset])) for offset, index in enumerate(selected)]
    copy_cache_slots_inplace(reference, pairs)
    for a, b in zip(reference, candidate):
        assert torch.equal(a.kv_cache[0], b.kv_cache[0])
    with pytest.raises(RuntimeError):
        kernel.commit_selected_slots_kernel(SimpleNamespace(), (3, 1))
    assert Attention is not None
