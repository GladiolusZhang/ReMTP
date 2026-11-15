"""One-launch KV path commit for the RAVEN tree (engineering only, 2026-11-15).

``REMTP_DYNAMIC_TREE_FAST_COMMIT=3`` moves the accepted path's KV slots in
every attention layer with a single Triton kernel whose slot indices stay on
the device.  Data movement equals ``raven_fast_commit.copy_cache_slots_inplace``
(same source -> destination pairs, applied in order): each program owns one
(layer, key/value) plane and copies the moves one after another, so a later
move reads what an earlier one wrote exactly as the sequential reference
does.  For a root-to-leaf path in BFS slots (selected indices strictly
increasing) no move reads a slot written by an earlier one, so the order only
matters for the reference semantics, not for the result.

Compared with mode 2 this removes, per round: the scan of
``static_forward_context`` for attention layers, the per-layer ``flatten`` and
per-slot view creation in Python, the host read of the target slot list, and
the per-move ``_foreach_copy_`` launches.  The layer table (base pointers and
strides) is built once per KV cache allocation.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _move_slots_kernel(base_ptr, kv_stride_ptr, slot_stride_ptr, src_ptr, dst_ptr, n_moves,
                       WORDS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    layer = pid // 2
    kv = pid % 2
    base = tl.load(base_ptr + layer).to(tl.pointer_type(tl.int32))
    kv_stride = tl.load(kv_stride_ptr + layer)
    slot_stride = tl.load(slot_stride_ptr + layer)
    plane = base + kv * kv_stride
    offs = tl.arange(0, BLOCK)
    mask = offs < WORDS
    for m in range(n_moves):
        src = tl.load(src_ptr + m).to(tl.int64)
        dst = tl.load(dst_ptr + m).to(tl.int64)
        values = tl.load(plane + src * slot_stride + offs, mask=mask)
        tl.store(plane + dst * slot_stride + offs, values, mask=mask)


@triton.jit
def _move_slots_count_kernel(base_ptr, kv_stride_ptr, slot_stride_ptr, src_ptr, dst_ptr, count_ptr,
                             MAX_MOVES: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    """``_move_slots_kernel`` with the number of moves read on the device (async mode)."""
    pid = tl.program_id(0)
    layer = pid // 2
    kv = pid % 2
    base = tl.load(base_ptr + layer).to(tl.pointer_type(tl.int32))
    kv_stride = tl.load(kv_stride_ptr + layer)
    slot_stride = tl.load(slot_stride_ptr + layer)
    plane = base + kv * kv_stride
    offs = tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    for m in tl.static_range(MAX_MOVES):
        live = (offs < WORDS) & (m < count)
        src = tl.load(src_ptr + m).to(tl.int64)
        dst = tl.load(dst_ptr + m).to(tl.int64)
        values = tl.load(plane + src * slot_stride + offs, mask=live)
        tl.store(plane + dst * slot_stride + offs, values, mask=live)


class LayerTable:
    """Base pointers and int32-word strides of each layer's [2, slots, ...] KV cache."""

    def __init__(self, caches: list[torch.Tensor]) -> None:
        if not caches:
            raise ValueError("no KV caches")
        device = caches[0].device
        bases, kv_strides, slot_strides = [], [], []
        words = None
        for cache in caches:
            if cache.ndim < 4 or cache.shape[0] != 2:
                raise RuntimeError(f"unexpected attention KV cache shape {tuple(cache.shape)}")
            flat = cache.flatten(1, 2)                       # [2, slots, heads, dim]
            chunk = flat[0, 0]
            if not chunk.is_contiguous():
                raise RuntimeError("KV slot chunk is not contiguous")
            size = cache.element_size()
            chunk_bytes = chunk.numel() * size
            kv_bytes, slot_bytes = flat.stride(0) * size, flat.stride(1) * size
            if chunk_bytes % 4 or kv_bytes % 4 or slot_bytes % 4 or cache.data_ptr() % 4:
                raise RuntimeError("KV cache layout is not 4-byte aligned")
            if words is None:
                words = chunk_bytes // 4
            elif words != chunk_bytes // 4:
                raise RuntimeError("KV slot size differs between layers")
            bases.append(cache.data_ptr())
            kv_strides.append(kv_bytes // 4)
            slot_strides.append(slot_bytes // 4)
        self.device = device
        self.layers = len(caches)
        self.words = int(words)
        self.block = triton.next_power_of_2(self.words)
        self.key = tuple(bases)
        self.bases = torch.tensor(bases, dtype=torch.int64, device=device)
        self.kv_strides = torch.tensor(kv_strides, dtype=torch.int64, device=device)
        self.slot_strides = torch.tensor(slot_strides, dtype=torch.int64, device=device)

    def move(self, sources: torch.Tensor, destinations: torch.Tensor) -> None:
        """Apply ``sources[i] -> destinations[i]`` in order in every layer (device index tensors)."""
        moves = int(sources.numel())
        if moves == 0:
            return
        _move_slots_kernel[(self.layers * 2,)](
            self.bases, self.kv_strides, self.slot_strides, sources, destinations, moves,
            WORDS=self.words, BLOCK=self.block, num_warps=4)

    def move_counted(self, sources: torch.Tensor, destinations: torch.Tensor, count: torch.Tensor) -> None:
        """The first ``count`` (device scalar) moves of ``sources``/``destinations``, in order."""
        _move_slots_count_kernel[(self.layers * 2,)](
            self.bases, self.kv_strides, self.slot_strides, sources, destinations, count,
            MAX_MOVES=int(sources.numel()), WORDS=self.words, BLOCK=self.block, num_warps=4)


def attention_caches(modules: Any) -> list[torch.Tensor]:
    caches = []
    for module in modules:
        cache = module.kv_cache[0]
        if cache.numel() == 0:
            continue
        caches.append(cache)
    return caches


_TABLES: dict[int, LayerTable] = {}


def target_table(runner: Any) -> LayerTable:
    """Cached table of the target model's attention layers (rebuilt if the caches move)."""
    from vllm.model_executor.layers.attention.attention import Attention

    table = _TABLES.get(id(runner))
    if table is not None:
        first = table.key[0]
        modules = getattr(runner, "_remtp_commit_modules", None)
        if modules and modules[0].kv_cache[0].data_ptr() == first:
            return table
    modules = [m for m in runner.compilation_config.static_forward_context.values() if isinstance(m, Attention)]
    runner._remtp_commit_modules = modules
    table = LayerTable(attention_caches(modules))
    _TABLES[id(runner)] = table
    return table


_PINNED: dict[str, torch.Tensor] = {}


def _device_indices(values: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Small int64 index tensor via a pinned staging buffer (no pageable copy, no sync)."""
    staging = torch.tensor(values, dtype=torch.int64).pin_memory()
    return staging.to(device, non_blocking=True)


def commit_selected_slots_kernel(runner: Any, selected: tuple[int, ...]) -> None:
    """Mode 3 of ``_compact_selected_path``: same moves as mode 2, one launch, no host reads."""
    from remtp.fixed6_vllm import _RUNTIME

    if _RUNTIME.draft_slots is not None:
        # The draft KV commit (SKIP_DEAD_DRAFT_KV=0) is not covered; use mode 2.
        from remtp.raven_fast_commit import commit_selected_slots_inplace

        commit_selected_slots_inplace(runner, selected)
        return
    if any(b <= a for a, b in zip(selected, selected[1:])):
        raise RuntimeError(f"selected path is not strictly increasing: {selected}")
    slots = _RUNTIME.target_slots
    index = _device_indices(tuple(selected), slots.device)
    sources = slots.index_select(0, index + 1).to(torch.int64)
    destinations = slots[1: 1 + len(selected)].to(torch.int64)
    target_table(runner).move(sources, destinations)


def commit_async(runner: Any) -> None:
    """Async mode: accepted slots and their number come from ``async_tree.STATE`` (no host values)."""
    from remtp import async_tree
    from remtp.fixed6_vllm import _RUNTIME

    state = async_tree.STATE
    slots = _RUNTIME.target_slots
    moves = state.accepted.shape[0]
    sources = slots.index_select(0, state.accepted + 1).to(torch.int64)
    destinations = slots[1: 1 + moves].to(torch.int64)
    target_table(runner).move_counted(sources, destinations, state.moves)
