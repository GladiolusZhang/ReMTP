"""Caterpillar proposal for residual_hit_anchor (experimental, 2026-11-15).

``REMTP_DYNAMIC_TREE_CATERPILLAR=j`` (async mode only) builds exactly the part
of the RAVEN tree that ``residual_hit_anchor`` verification can use
(``tests/test_anchor_dead_nodes.py``): the sampled trunk ``(0,), (0,0),
(0,0,0)`` and, at each depth, up to ``j`` siblings -- the most draft-likely
tokens of the trunk parent's Q other than the trunk token, kept when their
probability is at least ``tau_min``.  Descendants of non-trunk nodes are never
consulted by this verifier, so they are not proposed.

* The trunk is sampled exactly as in ``gpu_tree.propose_tree`` (same request
  generator draws in the same order); the MTP passes shrink to the trunk
  (1 row, then 2 rows) and the LM head to one row per depth, because siblings
  need no pass of their own.
* With ``j = 2`` and ``tau_min`` below every retained sibling the proposed
  useful nodes equal those of the current tree under ``kappa -> inf``; larger
  ``j`` raises the node count to ``3 + 3j`` (a larger tree budget): more
  corrections hit a sibling and end with an exact target anchor token.
* Verification, commit and the async device state are unchanged apart from
  the capacity ``3 + 3j`` and the sibling width ``j``.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import triton
import triton.language as tl

SPLITS = 32


def width() -> int:
    return int(os.getenv("REMTP_DYNAMIC_TREE_CATERPILLAR", "0") or 0)


def enabled() -> bool:
    return width() > 0


def capacity(j: int | None = None) -> int:
    j = width() if j is None else j
    return 3 + 3 * j


@triton.jit
def _topn_partial(rows_ptr, row_stride, part_ptr, V, CHUNK, S: tl.constexpr, TOPN: tl.constexpr,
                  BLOCK: tl.constexpr):
    """Per (row, split): the split's TOPN largest values (value desc, index asc)."""
    row = tl.program_id(0)
    split = tl.program_id(1)
    base = rows_ptr + row.to(tl.int64) * row_stride
    lo = split * CHUNK
    out = part_ptr + (row * S + split) * TOPN * 2
    prev_top = tl.full([], float("inf"), dtype=tl.float32)
    prev_idx = tl.full([], -1, dtype=tl.int32)
    for t in tl.static_range(TOPN):
        best = tl.full([BLOCK], -1.0, dtype=tl.float32)
        best_idx = tl.full([BLOCK], 2147483647, dtype=tl.int32)
        for start in range(0, CHUNK, BLOCK):
            local = start + tl.arange(0, BLOCK)
            offs = lo + local
            m = (offs < V) & (local < CHUNK)
            x = tl.load(base + offs, mask=m, other=0.0)
            c = tl.where(x < 0.0, 0.0, x)
            # strictly after the previous pick in (value desc, index asc) order
            after = (c < prev_top) | ((c == prev_top) & (offs > prev_idx))
            c = tl.where(m & after, c, -1.0)
            better = c > best
            best_idx = tl.where(better, offs, best_idx)
            best = tl.where(better, c, best)
        top = tl.max(best, axis=0)
        idx = tl.min(tl.where(best == top, best_idx, 2147483647), axis=0)
        tl.store(out + 2 * t, top)
        tl.store(out + 2 * t + 1, idx.to(tl.float32))
        prev_top = top
        prev_idx = idx


@triton.jit
def _siblings_kernel(part_ptr, trunk_ptr, tau, sib_ptr, valid_ptr, S: tl.constexpr, TOPN: tl.constexpr,
                     CAND: tl.constexpr, J: tl.constexpr, JB: tl.constexpr):
    """Merge the split candidates of one row; the first J tokens other than the trunk, and q >= tau."""
    c = tl.arange(0, CAND)
    cm = c < S * TOPN
    vals = tl.load(part_ptr + c * 2, mask=cm, other=-1.0)
    ids = tl.load(part_ptr + c * 2 + 1, mask=cm, other=0.0).to(tl.int32)
    trunk = tl.load(trunk_ptr).to(tl.int32)
    vals = tl.where(ids == trunk, -2.0, vals)
    o = tl.arange(0, JB)
    sib = tl.full([JB], -1, dtype=tl.int64)
    ok = tl.zeros([JB], dtype=tl.int1)
    for t in tl.static_range(J):
        top = tl.max(vals, axis=0)
        idx = tl.min(tl.where(vals == top, ids, 2147483647), axis=0)
        sib = tl.where(o == t, idx.to(tl.int64), sib)
        ok = tl.where(o == t, top >= tau, ok)
        vals = tl.where(ids == idx, -2.0, vals)
    tl.store(sib_ptr + o, sib, mask=o < J)
    tl.store(valid_ptr + o, ok, mask=o < J)


def siblings(q_row: torch.Tensor, trunk: torch.Tensor, j: int, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-j non-trunk tokens of one Q row (value desc, index asc) and their ``q >= tau`` flags."""
    q_row = q_row.reshape(1, -1).contiguous()
    vocabulary = q_row.shape[1]
    topn = j + 1
    chunk = triton.cdiv(vocabulary, SPLITS)
    part = torch.empty(SPLITS * topn * 2, dtype=torch.float32, device=q_row.device)
    _topn_partial[(1, SPLITS)](q_row, q_row.stride(0), part, vocabulary, chunk, S=SPLITS, TOPN=topn, BLOCK=1024,
                               num_warps=4)
    sib = torch.empty(j, dtype=torch.int64, device=q_row.device)
    valid = torch.empty(j, dtype=torch.bool, device=q_row.device)
    _siblings_kernel[(1,)](part, trunk.reshape(1), float(tau), sib, valid, S=SPLITS, TOPN=topn,
                           CAND=triton.next_power_of_2(SPLITS * topn), J=j, JB=triton.next_power_of_2(j),
                           num_warps=4)
    return sib, valid


@triton.jit
def _publish_kernel(trunk_ptr, sib_ptr, valid_ptr, tokens_ptr, valid_out_ptr, depth_ptr, parent_ptr,
                    trunk_out_ptr, siblings_ptr, bank_ptr, mask_ptr, ids_ptr,
                    J: tl.constexpr, CAP: tl.constexpr, B: tl.constexpr, JB: tl.constexpr, MB: tl.constexpr):
    o = tl.arange(0, JB)
    om = o < J
    c1 = tl.sum((tl.load(valid_ptr + 0 * J + o, mask=om, other=0) != 0).to(tl.int64), axis=0)
    c2 = tl.sum((tl.load(valid_ptr + 1 * J + o, mask=om, other=0) != 0).to(tl.int64), axis=0)
    c3 = tl.sum((tl.load(valid_ptr + 2 * J + o, mask=om, other=0) != 0).to(tl.int64), axis=0)
    n1 = 1 + c1
    n2 = 1 + c2
    n3 = 1 + c3
    count = n1 + n2 + n3
    i = tl.arange(0, B)
    im = i < CAP
    d1 = i < n1
    d2 = (i >= n1) & (i < n1 + n2)
    d3 = (i >= n1 + n2) & (i < count)
    local = tl.where(d1, i, tl.where(d2, i - n1, i - n1 - n2))
    level = tl.where(d1, 0, tl.where(d2, 1, 2))
    is_trunk = local == 0
    sib_index = tl.maximum(local - 1, 0)
    trunk_tok = tl.load(trunk_ptr + level, mask=im, other=0)
    sib_tok = tl.load(sib_ptr + level * J + sib_index, mask=im & (local >= 1), other=0)
    valid = d1 | d2 | d3
    tok = tl.where(valid, tl.where(is_trunk, trunk_tok, sib_tok), 0)
    depth = tl.where(valid, level + 1, 1).to(tl.int64)
    parent = tl.where(d2, 0, tl.where(d3, n1, -1)).to(tl.int64)
    tl.store(tokens_ptr + i, tok, mask=im)
    tl.store(ids_ptr + i, tok.to(tl.int32), mask=im)
    tl.store(valid_out_ptr + i, valid, mask=im)
    tl.store(depth_ptr + i, depth, mask=im)
    tl.store(parent_ptr + i, parent, mask=im)
    tl.store(bank_ptr + i, tl.where(valid, level, 0).to(tl.int64), mask=im)
    k = tl.arange(0, 4)
    km = k < 3
    start = tl.where(k == 0, 0, tl.where(k == 1, n1, n1 + n2))
    cnt = tl.where(k == 0, c1, tl.where(k == 1, c2, c3))
    tl.store(trunk_out_ptr + k, start, mask=km)
    sib_slot = start[:, None] + 1 + o[None, :]
    sib_ok = (o[None, :] < cnt[:, None]) & km[:, None] & om[None, :]
    tl.store(siblings_ptr + k[:, None] * J + o[None, :], tl.where(sib_ok, sib_slot, -1),
             mask=km[:, None] & om[None, :])
    # target mask rows/cols = 1 + slot: anchor column, self, parent (trunk), grandparent (trunk 1)
    r = tl.arange(0, MB)[:, None]
    cc = tl.arange(0, MB)[None, :]
    node = tl.arange(0, MB) - 1
    nd2 = (node >= n1) & (node < n1 + n2)
    nd3 = (node >= n1 + n2) & (node < count)
    npar = tl.where(nd2, 0, tl.where(nd3, n1, -1))
    ngrand = tl.where(nd3, 0, -1)
    col = cc - 1
    allowed = (cc == 0) | ((r >= 1) & (cc >= 1) & ((cc == r) | ((col == npar[:, None]) | (col == ngrand[:, None]))))
    out = tl.where(allowed, 0.0, float("-inf"))
    tl.store(mask_ptr + r * (CAP + 1) + cc, out, mask=(r < CAP + 1) & (cc < CAP + 1))


def publish(state: Any, trunk_tokens: torch.Tensor, sib: torch.Tensor, sib_valid: torch.Tensor,
            q_bank: torch.Tensor, j: int) -> torch.Tensor:
    """Fill the async device state with a caterpillar tree and return the padded [1, 3 + 3j] draft ids."""
    cap = capacity(j)
    device = trunk_tokens.device
    state.allocate(device, q_bank.shape[1], capacity=cap, sibling_width=j)
    ids = torch.empty(1, cap, dtype=torch.int32, device=device)
    _publish_kernel[(1,)](trunk_tokens.to(torch.int64), sib.contiguous(), sib_valid.contiguous(), state.tokens,
                          state.valid, state.depth, state.parent, state.trunk, state.siblings, state.bank, state.mask,
                          ids, J=j, CAP=cap, B=triton.next_power_of_2(cap), JB=triton.next_power_of_2(j),
                          MB=triton.next_power_of_2(cap + 1), num_warps=1)
    torch.index_select(q_bank, 0, state.bank, out=state.probs)
    state.ready = True
    state.extra["mask_from_kernel"] = True
    return ids


def propose(proposer: Any, *, logits: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor,
            common_attn_metadata: Any, sampling_metadata: Any, config: Any) -> list[torch.Tensor]:
    from remtp import async_tree
    from remtp import fixed6_vllm as fixed
    from remtp import gpu_tree
    from remtp import raven_fast_path as fast
    from remtp.dynamic_mtp_tree import mtp_layer_for_tree_depth
    from remtp.mimo_mtp import force_mimo_mtp_layer

    j = width()
    tau = float(config.min_draft_prob)
    device = logits.device
    route = os.getenv("REMTP_DYNAMIC_TREE_MTP_ROUTE", "physical_012")
    root_q = fixed._temperature_probs(logits[:1], sampling_metadata)                   # [1, V]
    trunk1 = fast.sample_q_token_fast(root_q[0], sampling_metadata)                    # [1]
    sib1, ok1 = siblings(root_q[0], trunk1, j, tau)
    root_hidden = hidden_states.reshape(1, -1)
    one = torch.ones((), dtype=torch.int64, device=device)
    with force_mimo_mtp_layer(mtp_layer_for_tree_depth(2, route)):
        last2, hid2 = gpu_tree.run_pass(
            proposer, tokens=trunk1.reshape(1), hidden=root_hidden, depths=torch.ones(1, dtype=torch.int64, device=device),
            parent_row=torch.full((1,), -1, dtype=torch.int64, device=device), n_real=one,
            root_positions=positions, base_common_metadata=common_attn_metadata, draft_index=1)
    q2 = fixed._temperature_probs(proposer.model.compute_logits(last2), sampling_metadata)   # [1, V]
    trunk2 = fast.sample_q_token_fast(q2[0], sampling_metadata)
    sib2, ok2 = siblings(q2[0], trunk2, j, tau)
    with force_mimo_mtp_layer(mtp_layer_for_tree_depth(3, route)):
        last3, _ = gpu_tree.run_pass(
            proposer, tokens=torch.cat((trunk1.reshape(1), trunk2.reshape(1))),
            hidden=torch.cat((root_hidden, hid2[:1])),
            depths=torch.tensor([1, 2], dtype=torch.int64, device=device),
            parent_row=torch.tensor([-1, 0], dtype=torch.int64, device=device), n_real=2 * one,
            root_positions=positions, base_common_metadata=common_attn_metadata, draft_index=2)
    q3 = fixed._temperature_probs(proposer.model.compute_logits(last3[1:2]), sampling_metadata)
    trunk3 = fast.sample_q_token_fast(q3[0], sampling_metadata)
    sib3, ok3 = siblings(q3[0], trunk3, j, tau)
    trunk_tokens = torch.cat((trunk1.reshape(1), trunk2.reshape(1), trunk3.reshape(1)))
    ids = publish(async_tree.STATE, trunk_tokens, torch.stack((sib1, sib2, sib3)), torch.stack((ok1, ok2, ok3)),
                  torch.cat((root_q, q2, q3)), j)
    fixed._RUNTIME.topology = async_tree.capacity_topology(capacity(j))
    fixed._RUNTIME.draft_slots = None
    fixed._RUNTIME.draft_token_ids = async_tree.STATE.tokens
    fixed._RUNTIME.draft_probs = async_tree.STATE.probs
    fixed._RUNTIME.dynamic_construction = None
    return [ids]
