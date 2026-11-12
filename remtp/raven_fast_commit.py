"""Fused KV path commit: all layers and slots in two multi-tensor copies.

The reference ``fixed6_vllm._copy_cache_slots`` runs one gather, one clone
and one scatter per attention layer (about 110 small kernels per round on the
36-layer target plus the draft layer).  The data movement is identical here:
every source slot is first copied into a scratch buffer, then every
destination slot is written from that buffer, so overlapping source and
destination slots see the same values as the reference.  ``torch._foreach_copy_``
turns the per-layer loop into a handful of fused launches.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch


def _slot_views(modules: Iterable[Any], pairs: list[tuple[int, int]]):
    sources: list[torch.Tensor] = []
    destinations: list[torch.Tensor] = []
    for module in modules:
        cache = module.kv_cache[0]
        if cache.numel() == 0:
            continue
        if cache.ndim < 3 or cache.shape[0] != 2:
            raise RuntimeError(f"unexpected attention KV cache shape {tuple(cache.shape)}")
        flat = cache.flatten(1, 2)
        for source, destination in pairs:
            for kv in range(2):
                sources.append(flat[kv, source])
                destinations.append(flat[kv, destination])
    return sources, destinations


def copy_cache_slots_fused(modules: Iterable[Any], pairs: list[tuple[int, int]]) -> int:
    """Copy ``source -> destination`` slot pairs in every module's KV cache.

    Returns the number of tensor pairs moved.  Semantics match applying the
    reference ``_copy_cache_slots`` to each module with the same slot lists.
    """
    if not pairs:
        return 0
    sources, destinations = _slot_views(modules, pairs)
    if not sources:
        return 0
    # Group by (shape, dtype) so each group's scratch is one allocation.
    groups: dict[tuple, list[int]] = {}
    for index, tensor in enumerate(sources):
        groups.setdefault((tuple(tensor.shape), tensor.dtype), []).append(index)
    for key, members in groups.items():
        shape, dtype = key
        scratch = torch.empty((len(members),) + shape, dtype=dtype, device=sources[members[0]].device)
        scratch_views = [scratch[i] for i in range(len(members))]
        torch._foreach_copy_(scratch_views, [sources[i] for i in members])
        torch._foreach_copy_([destinations[i] for i in members], scratch_views)
    return len(sources)


def commit_selected_slots_fused(runner: Any, selected: tuple[int, ...]) -> None:
    """Fused replacement for the per-layer loops in ``_compact_selected_path``."""
    from vllm.model_executor.layers.attention.attention import Attention
    from remtp.fixed6_vllm import _RUNTIME

    target_slots = _RUNTIME.target_slots.tolist()
    pairs = [(int(target_slots[index + 1]), int(target_slots[1 + offset])) for offset, index in enumerate(selected)]
    modules = [m for m in runner.compilation_config.static_forward_context.values() if isinstance(m, Attention)]
    copy_cache_slots_fused(modules, pairs)

    if _RUNTIME.draft_slots is not None and hasattr(runner, "drafter"):
        draft_slots = _RUNTIME.draft_slots.tolist()
        draft_pairs = [(int(draft_slots[index]), int(draft_slots[offset])) for offset, index in enumerate(selected)]
        draft_modules = [m for m in runner.drafter.model.modules() if isinstance(m, Attention) and m.kv_cache]
        copy_cache_slots_fused(draft_modules, draft_pairs)


def inplace_order_is_safe(pairs: list[tuple[int, int]]) -> bool:
    """True when copying ``pairs`` one after another equals a buffered move.

    A sequential copy is wrong only if an earlier write lands on a slot that a
    later pair still has to read.
    """
    for position, (_, destination) in enumerate(pairs):
        if any(destination == source for source, _ in pairs[position + 1:]):
            return False
    return True


def copy_cache_slots_inplace(modules: Iterable[Any], pairs: list[tuple[int, int]]) -> int:
    """Move ``source -> destination`` slots without a scratch buffer.

    Identity pairs are dropped.  Each remaining pair is one multi-tensor copy
    over every module's key and value slot; pairs run one after another, which
    equals the buffered move whenever ``inplace_order_is_safe`` holds (always
    for a root-to-leaf path in BFS slots).  Otherwise the buffered two-phase
    copy is used.  Returns the number of tensor pairs moved.
    """
    moves = [(source, destination) for source, destination in pairs if source != destination]
    if not moves:
        return 0
    if not inplace_order_is_safe(moves):
        return copy_cache_slots_fused(modules, moves)
    flats = []
    for module in modules:
        cache = module.kv_cache[0]
        if cache.numel() == 0:
            continue
        if cache.ndim < 3 or cache.shape[0] != 2:
            raise RuntimeError(f"unexpected attention KV cache shape {tuple(cache.shape)}")
        flats.append(cache.flatten(1, 2))
    if not flats:
        return 0
    moved = 0
    for source, destination in moves:
        sources = [flat[kv, source] for flat in flats for kv in range(2)]
        destinations = [flat[kv, destination] for flat in flats for kv in range(2)]
        torch._foreach_copy_(destinations, sources)
        moved += len(sources)
    return moved


def commit_selected_slots_inplace(runner: Any, selected: tuple[int, ...]) -> None:
    """In-place ordered replacement for the per-layer loops in ``_compact_selected_path``."""
    from vllm.model_executor.layers.attention.attention import Attention
    from remtp.fixed6_vllm import _RUNTIME

    target_slots = _RUNTIME.target_slots.tolist()
    pairs = [(int(target_slots[index + 1]), int(target_slots[1 + offset])) for offset, index in enumerate(selected)]
    modules = [m for m in runner.compilation_config.static_forward_context.values() if isinstance(m, Attention)]
    copy_cache_slots_inplace(modules, pairs)

    if _RUNTIME.draft_slots is not None and hasattr(runner, "drafter"):
        draft_slots = _RUNTIME.draft_slots.tolist()
        draft_pairs = [(int(draft_slots[index]), int(draft_slots[offset])) for offset, index in enumerate(selected)]
        draft_modules = [m for m in runner.drafter.model.modules() if isinstance(m, Attention) and m.kv_cache]
        copy_cache_slots_inplace(draft_modules, draft_pairs)
