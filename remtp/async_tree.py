"""RAVEN under vLLM asynchronous scheduling (engineering, 2026-11-15).

``REMTP_DYNAMIC_TREE_ASYNC=1`` (with ``NO_ASYNC_SCHEDULING=0``) keeps a whole
decode round on the device so the host can prepare the next step while the GPU
is still busy, as vLLM already does for chain MTP:

* the proposal (``gpu_tree.propose_tree``) ends without a host read: the tree
  lives in fixed-capacity device tensors (9 node slots, valid flags, depth,
  parent, path code, trunk nodes and trunk siblings); padded slots carry no
  real node, write KV only into slots beyond the committed length and are
  masked out of every real node's attention;
* the target tree mask is written straight into the persistent CUDA-graph
  mask buffer and the target positions are ``anchor + depth`` on the device;
* verification (``device_verify.verify_async``) writes vLLM's padded
  ``sampled_token_ids`` row, the accepted node list and its length;
* the KV path commit reads the accepted nodes and their number on the device;
* the next MTP first pass samples from the accepted leaf's row, read on the
  device;
* vLLM's async scheduler always schedules the full 9-slot capacity.

The tree itself stays adaptive: node count, branching, retained nodes and the
relaxation allowance are decided per round exactly as before.  Only the shape
the target sees is fixed: unused slots are masked rows.  The proposal and
verification rules are those of ``GPU_BUILD_FUSED`` and ``DEVICE_VERIFY``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import triton
import triton.language as tl

CAPACITY = 9
TRUNK = 3
NEG_INF = float("-inf")


def enabled() -> bool:
    return os.getenv("REMTP_DYNAMIC_TREE_ASYNC", "0") == "1"


@dataclass
class DeviceTree:
    """Persistent device state shared by proposal, target preparation, verification and commit."""
    tokens: torch.Tensor | None = None        # [9] int64 draft token per slot (0 when padded)
    probs: torch.Tensor | None = None         # [9, V] float32 proposal Q per slot
    valid: torch.Tensor | None = None         # [9] bool
    depth: torch.Tensor | None = None         # [9] int64 (1 for padded slots)
    parent: torch.Tensor | None = None        # [9] int64 parent slot, -1 at depth 1 / padded
    trunk: torch.Tensor | None = None         # [3] int64 slots of paths (0,), (0,0), (0,0,0)
    siblings: torch.Tensor | None = None      # [3, 2] int64 trunk siblings in topology order, -1 if none
    accepted: torch.Tensor | None = None      # [3] int64 accepted slots in path order (padded with 0)
    moves: torch.Tensor | None = None         # [] int64 number of accepted slots
    leaf_row: torch.Tensor | None = None      # [] int64 target row of the accepted leaf (0: anchor)
    mask: torch.Tensor | None = None          # [10, 10] float32 target tree mask of the current tree
    bank: torch.Tensor | None = None          # [9] int64 proposal row (root, depth-2 parents, depth-3 parents)
    ready: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def allocate(self, device: torch.device, vocabulary: int) -> None:
        if self.tokens is not None and self.probs is not None and self.probs.shape[1] == vocabulary:
            return
        i64 = torch.int64
        self.tokens = torch.zeros(CAPACITY, dtype=i64, device=device)
        self.probs = torch.zeros(CAPACITY, vocabulary, dtype=torch.float32, device=device)
        self.valid = torch.zeros(CAPACITY, dtype=torch.bool, device=device)
        self.depth = torch.ones(CAPACITY, dtype=i64, device=device)
        self.parent = torch.full((CAPACITY,), -1, dtype=i64, device=device)
        self.trunk = torch.zeros(TRUNK, dtype=i64, device=device)
        self.siblings = torch.full((TRUNK, 2), -1, dtype=i64, device=device)
        self.accepted = torch.zeros(TRUNK, dtype=i64, device=device)
        self.moves = torch.zeros((), dtype=i64, device=device)
        self.leaf_row = torch.zeros((), dtype=i64, device=device)
        self.mask = torch.zeros(CAPACITY + 1, CAPACITY + 1, dtype=torch.float32, device=device)
        self.bank = torch.zeros(CAPACITY, dtype=i64, device=device)


STATE = DeviceTree()
_CAPACITY_TOPOLOGY: Any = None


def capacity_topology() -> Any:
    """Host topology with 9 nodes; used only for counts and shapes in async mode."""
    global _CAPACITY_TOPOLOGY
    if _CAPACITY_TOPOLOGY is None:
        from remtp.dynamic_mtp_tree import DynamicTreeTopology

        _CAPACITY_TOPOLOGY = DynamicTreeTopology.from_paths(
            ((0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)))
    return _CAPACITY_TOPOLOGY


def tree_mask(valid: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
    """(N+1)x(N+1) float32 target mask: anchor column, self and ancestors (depth <= 3)."""
    device = valid.device
    nodes = valid.shape[0]
    index = torch.arange(nodes, device=device)
    p1 = parent
    p2 = torch.where(p1 >= 0, parent[p1.clamp_min(0)], torch.full_like(p1, -1))
    allowed = (index[None, :] == index[:, None]) | (
        valid[:, None] & ((index[None, :] == p1[:, None]) | (index[None, :] == p2[:, None])))
    mask = torch.full((nodes + 1, nodes + 1), NEG_INF, dtype=torch.float32, device=device)
    mask[1:, 1:] = torch.where(allowed, 0.0, NEG_INF)
    mask[:, 0].fill_(0.0)
    return mask


def publish_tree(*, n1: torch.Tensor, n2: torch.Tensor, n3: torch.Tensor, tokens_bfs: torch.Tensor,
                 codes_bfs: torch.Tensor, parent_all: torch.Tensor, target_for_pack: Any,
                 probs_bfs: torch.Tensor) -> torch.Tensor:
    """Store a device-built tree in ``STATE`` and return the padded [1, 9] draft ids.

    ``codes_bfs`` are the packed path codes (BFS order); ``parent_all`` holds,
    before packing, each candidate's parent slot (-1 at depth 1).
    """
    device = tokens_bfs.device
    STATE.allocate(device, probs_bfs.shape[1])
    index = torch.arange(CAPACITY, device=device)
    count = n1 + n2 + n3
    valid = index < count
    depth = 1 + (index >= n1).to(torch.int64) + (index >= n1 + n2).to(torch.int64)
    depth = torch.where(valid, depth, torch.ones_like(depth))
    parent = torch.where(valid, target_for_pack(parent_all), torch.full_like(index, -1))
    code_bfs = torch.where(valid, codes_bfs, torch.full_like(codes_bfs, -1))
    trunk = torch.stack((torch.zeros_like(n1), n1, n1 + n2))
    # Trunk siblings: the next two slots of the same depth whose parent code is the trunk's.
    starts = torch.stack((torch.zeros_like(n1), n1, n1 + n2))
    ends = torch.stack((n1, n1 + n2, count))
    offsets = torch.arange(1, 3, device=device)
    candidates = starts[:, None] + offsets[None, :]                         # [3, 2]
    in_level = candidates < ends[:, None]
    parent_code = torch.where(code_bfs[candidates.clamp(max=CAPACITY - 1)] >= 0,
                              code_bfs[candidates.clamp(max=CAPACITY - 1)] // 4, -1)
    same_parent = torch.stack((torch.ones(2, dtype=torch.bool, device=device), parent_code[1] == 0,
                               parent_code[2] == 0))
    siblings = torch.where(in_level & same_parent, candidates, torch.full_like(candidates, -1))
    STATE.tokens.copy_(torch.where(valid, tokens_bfs, torch.zeros_like(tokens_bfs)))
    STATE.probs.copy_(probs_bfs)
    STATE.valid.copy_(valid)
    STATE.depth.copy_(depth)
    STATE.parent.copy_(parent)
    STATE.trunk.copy_(trunk)
    STATE.siblings.copy_(siblings)
    STATE.mask.copy_(tree_mask(valid, parent))
    STATE.extra["mask_from_kernel"] = False
    STATE.ready = True
    return STATE.tokens.to(torch.int32).view(1, CAPACITY)


@triton.jit
def _publish_kernel(n1_ptr, n2_ptr, n3_ptr, tok1_ptr, tok2_ptr, code2_ptr, par2_ptr, tok3_ptr, code3_ptr, par3_ptr,
                    tokens_ptr, valid_ptr, depth_ptr, parent_ptr, trunk_ptr, siblings_ptr, bank_ptr, mask_ptr,
                    ids_ptr, CAP: tl.constexpr, B: tl.constexpr):
    n1 = tl.load(n1_ptr)
    n2 = tl.load(n2_ptr)
    n3 = tl.load(n3_ptr)
    count = n1 + n2 + n3
    i = tl.arange(0, B)
    im = i < CAP
    d1 = i < n1
    d2 = (i >= n1) & (i < n1 + n2)
    d3 = (i >= n1 + n2) & (i < count)
    j2 = tl.where(d2, i - n1, 0)
    j3 = tl.where(d3, i - n1 - n2, 0)
    tok = tl.where(d1, tl.load(tok1_ptr + tl.minimum(i, 2), mask=im, other=0),
                   tl.where(d2, tl.load(tok2_ptr + j2, mask=im, other=0),
                            tl.where(d3, tl.load(tok3_ptr + j3, mask=im, other=0), 0)))
    code = tl.where(d1, i.to(tl.int64), tl.where(d2, tl.load(code2_ptr + j2, mask=im, other=0),
                                               tl.where(d3, tl.load(code3_ptr + j3, mask=im, other=0), -1)))
    p2 = tl.load(par2_ptr + j2, mask=im, other=0)
    p3 = tl.load(par3_ptr + j3, mask=im, other=0)
    parent = tl.where(d2, p2, tl.where(d3, n1 + p3, -1))
    grand = tl.where(d3, tl.load(par2_ptr + p3, mask=im, other=0), -1)
    bank = tl.where(d2, 1 + p2, tl.where(d3, 4 + p3, 0))
    valid = d1 | d2 | d3
    depth = tl.where(d2, 2, tl.where(d3, 3, 1)).to(tl.int64)
    tl.store(tokens_ptr + i, tok, mask=im)
    tl.store(ids_ptr + i, tok.to(tl.int32), mask=im)
    tl.store(valid_ptr + i, valid, mask=im)
    tl.store(depth_ptr + i, depth, mask=im)
    tl.store(parent_ptr + i, parent, mask=im)
    tl.store(bank_ptr + i, bank, mask=im)
    # trunk slots and their siblings (next two slots of the level with the trunk's parent)
    k = tl.arange(0, 4)
    km = k < 3
    start = tl.where(k == 0, 0, tl.where(k == 1, n1, n1 + n2))
    end = tl.where(k == 0, n1, tl.where(k == 1, n1 + n2, count))
    tl.store(trunk_ptr + k, start, mask=km)
    for o in tl.static_range(1, 3):
        cand = start + o
        cc = tl.load(code2_ptr + tl.maximum(tl.minimum(cand - n1, 3), 0))
        c3 = tl.load(code3_ptr + tl.maximum(tl.minimum(cand - n1 - n2, 6), 0))
        cand_code = tl.where(k == 1, cc, c3)
        same = (k == 0) | ((cand_code // 4) == 0)
        ok = (cand < end) & same
        tl.store(siblings_ptr + k * 2 + (o - 1), tl.where(ok, cand, -1), mask=km)
    # target mask: anchor column, self, parent, grandparent (rows/cols are 1 + slot)
    r = tl.arange(0, 16)[:, None]
    c = tl.arange(0, 16)[None, :]
    node = tl.arange(0, 16) - 1
    nvalid = (node >= 0) & (node < count)
    nd2 = (node >= n1) & (node < n1 + n2)
    nd3 = (node >= n1 + n2) & (node < count)
    nj2 = tl.where(nd2, node - n1, 0)
    nj3 = tl.where(nd3, node - n1 - n2, 0)
    np2 = tl.load(par2_ptr + nj2, mask=nd2, other=0)
    np3 = tl.load(par3_ptr + nj3, mask=nd3, other=0)
    nparent = tl.where(nd2, np2, tl.where(nd3, n1 + np3, -1))
    ngrand = tl.where(nd3, tl.load(par2_ptr + np3, mask=nd3, other=0), -1)
    col = c - 1
    allowed = (c == 0) | ((r >= 1) & (c >= 1) & ((c == r) | (nvalid[:, None] & (
        (col == nparent[:, None]) | (col == ngrand[:, None])))))
    out = tl.where(allowed, 0.0, float("-inf"))
    tl.store(mask_ptr + r * (CAP + 1) + c, out, mask=(r < CAP + 1) & (c < CAP + 1))


def publish_levels(*, n1: torch.Tensor, n2: torch.Tensor, n3: torch.Tensor, tok1: torch.Tensor,
                   ret2: dict[str, torch.Tensor], ret3: dict[str, torch.Tensor],
                   q_bank: torch.Tensor) -> torch.Tensor:
    """``publish_tree`` from the per-level outputs of ``gpu_tree``: one Triton kernel plus the Q gather."""
    device = tok1.device
    STATE.allocate(device, q_bank.shape[1])
    ids = torch.empty(1, CAPACITY, dtype=torch.int32, device=device)
    i64 = torch.int64
    _publish_kernel[(1,)](
        n1.reshape(1), n2.reshape(1), n3.reshape(1), tok1.to(i64), ret2["token"].to(i64), ret2["code"].to(i64),
        ret2["parent"].to(i64), ret3["token"].to(i64), ret3["code"].to(i64), ret3["parent"].to(i64),
        STATE.tokens, STATE.valid, STATE.depth, STATE.parent, STATE.trunk, STATE.siblings, STATE.bank,
        STATE.mask, ids, CAP=CAPACITY, B=16, num_warps=1)
    torch.index_select(q_bank, 0, STATE.bank, out=STATE.probs)
    STATE.ready = True
    STATE.extra["mask_from_kernel"] = True
    return ids


def write_target_mask(runner: Any) -> None:
    """Copy the current tree's mask into every target tree-attention builder's persistent buffer."""
    from remtp.tree_cudagraph import write_persistent_mask

    mask = STATE.mask if STATE.extra.get("mask_from_kernel") else tree_mask(STATE.valid, STATE.parent)
    for groups in runner.attn_groups:
        for group in groups:
            write_persistent_mask(group.get_metadata_builder(), mask)


def set_target_positions(positions: torch.Tensor) -> None:
    """positions[1:10] = positions[0] + depth, on the device."""
    rows = CAPACITY + 1
    if positions.ndim == 1:
        positions[1:rows] = positions[0:1] + STATE.depth.to(positions.dtype)
    else:
        positions[:, 1:rows] = positions[:, 0:1] + STATE.depth.to(positions.dtype)[None, :]
