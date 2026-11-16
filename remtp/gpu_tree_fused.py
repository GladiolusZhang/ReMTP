"""Fused decision step for the device-side tree (2026-11-15, engineering only).

``gpu_tree._level`` issues ~22 kernels per parent row plus ~100 tiny kernels
for the threshold/sibling/backup logic and the retention ranking.  Inside a
CUDA graph the launches are cheap, but every kernel still costs a fixed
1.6 us (RTX 4090) to several us (Jetson) on the GPU.  This module keeps the
value-producing operations exactly as before and fuses only what follows:

* unchanged PyTorch ops: the three per-row 1-D sums (row sums must stay 1-D:
  a 2-D row reduction rounds differently), the normalising division, ``min``
  and ``topk``; elementwise ops run on all rows at once (same per-element
  arithmetic).  ``clamp_min(0)`` is dropped (identity on a valid softmax row;
  invalid rows abort as before) and the entropy terms are one Triton kernel
  (round-to-nearest division, libdevice ``logf``);
* one Triton kernel for the rest of the level: stable top-k ordering (value
  descending, index ascending), entropy scaling (as PyTorch's tensor / scalar:
  a multiply by the float32 reciprocal), validity, the float64
  threshold, float32 eligibility, the sibling rule, backups, and - for
  levels 2/3 - the ``residual_coverage`` retention ranking and the packed
  outputs.  Divisions use round-to-nearest; float64 exp/log are libdevice,
  as PyTorch's CUDA ops.

``fused_level`` returns the same keys and values as ``gpu_tree._level``;
``tests/test_gpu_tree.py`` checks every output for bit equality.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

K_STABLE = 32
TERM_BLOCK = 2048


@triton.jit
def _entropy_terms_kernel(q_ptr, second_ptr, out_ptr, V, BLOCK: tl.constexpr):
    """``q2 = q / q.sum(); q2 * q2.clamp_min(1e-30).log()`` for one block of one row.

    Round-to-nearest division and libdevice ``logf`` (PyTorch's CUDA ``log``)."""
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < V
    base = row.to(tl.int64) * V
    q = tl.load(q_ptr + base + offsets, mask=mask, other=1.0)
    q2 = tl.math.div_rn(q, tl.load(second_ptr + row))
    tl.store(out_ptr + base + offsets, q2 * libdevice.log(tl.maximum(q2, 1e-30)), mask=mask)


@triton.jit
def _decide_kernel(
    topv_ptr, topi_ptr, rows_ptr, row_stride,
    tot_ptr, ent_ptr, min_ptr, sampled_ptr,
    pvalid_ptr, pcode_ptr, plogq_ptr, ncur_ptr,
    c64_ptr, c32_ptr,
    sel_tok_ptr, sel_raw_ptr, sel_cnt_ptr, sel_ent_ptr, sel_inv_ptr, sel_tie_ptr, sel_thr_ptr, sel_qmax_ptr,
    ret_tok_ptr, ret_raw_ptr, ret_code_ptr, ret_logq_ptr, ret_par_ptr, ret_cnt_ptr,
    P, backup_budget, max_nodes, levels_after, levels_left,
    K: tl.constexpr, PB: tl.constexpr, NB: tl.constexpr, SLOTS: tl.constexpr, SB: tl.constexpr,
    RETAIN: tl.constexpr,
):
    depth_term = tl.load(c64_ptr + 0)
    neg_kappa = tl.load(c64_ptr + 1)
    min_draft = tl.load(c64_ptr + 2)
    sib_ratio = tl.load(c64_ptr + 3)
    value_factor = tl.load(c64_ptr + 4)
    inv_log_v32 = tl.load(c32_ptr + 0)
    min_draft32 = tl.load(c32_ptr + 1)

    r = tl.arange(0, PB)
    rmask = r < P
    k = tl.arange(0, K)
    v = tl.load(topv_ptr + r[:, None] * K + k[None, :], mask=rmask[:, None], other=-1.0)
    idx = tl.load(topi_ptr + r[:, None] * K + k[None, :], mask=rmask[:, None], other=0)
    # Stable order of the reference sort: rank = number of entries that precede.
    vi = v[:, :, None]
    vj = v[:, None, :]
    ii = idx[:, :, None]
    ij = idx[:, None, :]
    precede = (vj > vi) | ((vj == vi) & (ij < ii))
    rank = tl.sum(precede.to(tl.int32), axis=2)
    o0 = tl.sum(tl.where(rank == 0, v, 0.0), axis=1)
    o1 = tl.sum(tl.where(rank == 1, v, 0.0), axis=1)
    o2 = tl.sum(tl.where(rank == 2, v, 0.0), axis=1)
    o3 = tl.sum(tl.where(rank == 3, v, 0.0), axis=1)
    olast = tl.sum(tl.where(rank == K - 1, v, 0.0), axis=1)
    i0 = tl.sum(tl.where(rank == 0, idx, 0), axis=1)
    i1 = tl.sum(tl.where(rank == 1, idx, 0), axis=1)
    i2 = tl.sum(tl.where(rank == 2, idx, 0), axis=1)

    total = tl.load(tot_ptr + r, mask=rmask, other=1.0)
    ent_sum = tl.load(ent_ptr + r, mask=rmask, other=0.0)
    minimum = tl.load(min_ptr + r, mask=rmask, other=0.0)
    sampled = tl.load(sampled_ptr + r, mask=rmask, other=0)
    # PyTorch divides a CUDA tensor by a Python scalar as a * (1.0f / float(b)).
    entropy32 = -ent_sum * inv_log_v32
    finite = (total == total) & (tl.abs(total) < float("inf"))
    invalid = ~(finite & (total > 0.0) & (minimum >= 0.0))
    tie = (o3 == olast) & (o3 >= min_draft32)

    q_max = o0.to(tl.float64)
    uncertainty = entropy32.to(tl.float64) + depth_term
    threshold = tl.maximum(q_max * tl.exp(neg_kappa * uncertainty), min_draft)
    threshold32 = threshold.to(tl.float32)
    e0 = o0 >= threshold32
    e1 = o1 >= threshold32
    e2 = o2 >= threshold32
    o1d = o1.to(tl.float64)
    o2d = o2.to(tl.float64)
    s1 = (~e1) & (o1d >= min_draft) & (o1d >= q_max * sib_ratio)
    s2 = (~e2) & (o2d >= min_draft) & (o2d >= q_max * sib_ratio)
    b0 = e0 & (i0 != sampled)
    b1 = (e1 | s1) & e0 & (i1 != sampled)
    b2 = (e2 | s2) & e0 & (i2 != sampled)
    rb0 = b0.to(tl.int32)
    rb1 = rb0 + b1.to(tl.int32)
    rb2 = rb1 + b2.to(tl.int32)
    t0 = b0 & (rb0 <= backup_budget)
    t1 = b1 & (rb1 <= backup_budget)
    t2 = b2 & (rb2 <= backup_budget)
    has1 = (t0 & (rb0 == 1)) | (t1 & (rb1 == 1)) | (t2 & (rb2 == 1))
    tok1 = tl.where(t0 & (rb0 == 1), i0, tl.where(t1 & (rb1 == 1), i1, tl.where(t2 & (rb2 == 1), i2, 0)))
    has2 = (t1 & (rb1 == 2)) | (t2 & (rb2 == 2))
    tok2 = tl.where(t1 & (rb1 == 2), i1, tl.where(t2 & (rb2 == 2), i2, 0))
    count = 1 + t0.to(tl.int64) + t1.to(tl.int64) + t2.to(tl.int64)
    row_base = rows_ptr + r.to(tl.int64) * row_stride
    raw0 = tl.load(row_base + sampled, mask=rmask, other=0.0)
    raw1 = tl.where(has1, tl.load(row_base + tok1, mask=rmask & has1, other=0.0), 0.0)
    raw2 = tl.where(has2, tl.load(row_base + tok2, mask=rmask & has2, other=0.0), 0.0)

    tl.store(sel_tok_ptr + r * 3 + 0, sampled, mask=rmask)
    tl.store(sel_tok_ptr + r * 3 + 1, tok1, mask=rmask)
    tl.store(sel_tok_ptr + r * 3 + 2, tok2, mask=rmask)
    tl.store(sel_raw_ptr + r * 3 + 0, raw0, mask=rmask)
    tl.store(sel_raw_ptr + r * 3 + 1, raw1, mask=rmask)
    tl.store(sel_raw_ptr + r * 3 + 2, raw2, mask=rmask)
    tl.store(sel_cnt_ptr + r, count, mask=rmask)
    tl.store(sel_ent_ptr + r, entropy32, mask=rmask)
    tl.store(sel_inv_ptr + r, invalid.to(tl.int8), mask=rmask)
    tl.store(sel_tie_ptr + r, tie.to(tl.int8), mask=rmask)
    tl.store(sel_thr_ptr + r, threshold, mask=rmask)
    tl.store(sel_qmax_ptr + r, q_max, mask=rmask)

    if RETAIN:
        tl.debug_barrier()
        n = tl.arange(0, NB)
        rr = n // 3
        ss = n % 3
        cmask = rr < P
        tok = tl.load(sel_tok_ptr + rr * 3 + ss, mask=cmask, other=0)
        raw = tl.load(sel_raw_ptr + rr * 3 + ss, mask=cmask, other=0.0)
        cnt = tl.load(sel_cnt_ptr + rr, mask=cmask, other=0)
        pvalid = tl.load(pvalid_ptr + rr, mask=cmask, other=0)
        pcode = tl.load(pcode_ptr + rr, mask=cmask, other=0)
        plogq = tl.load(plogq_ptr + rr, mask=cmask, other=0.0)
        valid = cmask & (pvalid != 0) & (ss < cnt)
        code = pcode * 4 + ss
        logq = plogq + tl.log(tl.maximum(raw.to(tl.float64), 1e-30))
        value = tl.exp(logq) * value_factor
        trunk = pcode == 0
        klass = tl.where(trunk, 0, tl.where(ss == 0, 1, 2))
        orank = tl.where(trunk, ss, 0)
        ci = klass[:, None]
        cj = klass[None, :]
        ri = orank[:, None]
        rj = orank[None, :]
        vi2 = value[:, None]
        vj2 = value[None, :]
        si = code[:, None]
        sj = code[None, :]
        before = (cj < ci) | ((cj == ci) & ((rj < ri) | ((rj == ri) & ((vj2 > vi2) | ((vj2 == vi2) & (sj < si))))))
        position = tl.sum((before & valid[None, :]).to(tl.int32), axis=1)
        n_current = tl.load(ncur_ptr).to(tl.int32)
        remaining = max_nodes - n_current
        level = tl.maximum(remaining - levels_after, 0)
        spread = tl.maximum(remaining // levels_left, 1)
        budget = tl.where(remaining <= 0, 0, tl.minimum(level, spread))
        keep = valid & (position < budget)
        out_slot = tl.sum((keep[None, :] & (sj < si)).to(tl.int32), axis=1)
        t = tl.arange(0, SB)
        match = keep[None, :] & (out_slot[None, :] == t[:, None])
        tl.store(ret_tok_ptr + t, tl.sum(tl.where(match, tok[None, :], 0), axis=1), mask=t < SLOTS)
        tl.store(ret_raw_ptr + t, tl.sum(tl.where(match, raw[None, :], 0.0), axis=1), mask=t < SLOTS)
        tl.store(ret_code_ptr + t, tl.sum(tl.where(match, code[None, :], 0), axis=1), mask=t < SLOTS)
        tl.store(ret_logq_ptr + t, tl.sum(tl.where(match, logq[None, :], 0.0), axis=1), mask=t < SLOTS)
        tl.store(ret_par_ptr + t, tl.sum(tl.where(match, rr[None, :].to(tl.int64), 0), axis=1), mask=t < SLOTS)
        tl.store(ret_cnt_ptr, tl.sum(keep.to(tl.int64), axis=0))


_CONSTANTS: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}


def _constants(config: Any, depth: int, vocabulary: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (id(config), depth, vocabulary, str(device))
    if key not in _CONSTANTS:
        c64 = torch.tensor([config.depth_exploration * (depth / config.max_depth), -config.threshold_sensitivity,
                            config.min_draft_prob, config.min_sibling_ratio,
                            math.exp(config.depth_reward * depth / config.max_depth)],
                           dtype=torch.float64, device=device)
        import numpy as np
        inverse = np.float32(1.0) / np.float32(math.log(vocabulary))   # PyTorch's reciprocal for tensor / scalar
        c32 = torch.tensor([float(inverse), config.min_draft_prob], dtype=torch.float32, device=device)
        _CONSTANTS[key] = (c64, c32)
    return _CONSTANTS[key]


def fused_level(q: torch.Tensor, sampled: torch.Tensor, parent_valid: torch.Tensor, parent_code: torch.Tensor,
                parent_logq: torch.Tensor, n_current: torch.Tensor, *, depth: int, slots: int, backup_budget: int,
                config: Any) -> dict[str, torch.Tensor]:
    rows, vocabulary = q.shape
    if config.max_children_per_parent != 3 or vocabulary < K_STABLE or rows > 4:
        raise ValueError("fused level supports at most 4 parent rows, 3 children and V >= 32")
    device = q.device
    p32 = q if q.dtype == torch.float32 else q.to(torch.float32)
    p32 = p32.contiguous()
    # clamp_min(0) is the identity on a valid row (softmax output, min >= 0); a row
    # with a negative entry is flagged invalid below and aborts the proposal, as before.
    totals = torch.stack([p32[r].sum() for r in range(rows)])
    qn = p32 / totals[:, None]
    second = torch.stack([qn[r].sum() for r in range(rows)])
    terms = torch.empty_like(qn)
    _entropy_terms_kernel[(rows, triton.cdiv(vocabulary, TERM_BLOCK))](qn, second, terms, vocabulary,
                                                                       BLOCK=TERM_BLOCK)
    entropy_sums = torch.stack([terms[r].sum() for r in range(rows)])
    minimum = p32.amin(dim=-1)
    top_values, top_ids = torch.topk(qn, K_STABLE, dim=-1, sorted=True)
    c64, c32 = _constants(config, depth, vocabulary, device)
    i64, f32, f64 = torch.int64, torch.float32, torch.float64
    sel = dict(tokens=torch.empty(rows, 3, dtype=i64, device=device), raw=torch.empty(rows, 3, dtype=f32, device=device),
               count=torch.empty(rows, dtype=i64, device=device), entropy=torch.empty(rows, dtype=f32, device=device),
               invalid=torch.empty(rows, dtype=torch.int8, device=device),
               tie_risk=torch.empty(rows, dtype=torch.int8, device=device),
               threshold=torch.empty(rows, dtype=f64, device=device), q_max=torch.empty(rows, dtype=f64, device=device))
    width = max(slots, 1)
    ret = dict(token=torch.zeros(width, dtype=i64, device=device), raw=torch.zeros(width, dtype=f32, device=device),
               code=torch.zeros(width, dtype=i64, device=device), logq=torch.zeros(width, dtype=f64, device=device),
               parent=torch.zeros(width, dtype=i64, device=device), count=torch.zeros((), dtype=i64, device=device))
    _decide_kernel[(1,)](
        top_values, top_ids, p32, p32.stride(0), totals, entropy_sums, minimum, sampled.reshape(-1).to(i64),
        parent_valid.to(torch.int8), parent_code.to(i64), parent_logq.to(f64), n_current.reshape(()).to(i64),
        c64, c32,
        sel["tokens"], sel["raw"], sel["count"], sel["entropy"], sel["invalid"], sel["tie_risk"], sel["threshold"],
        sel["q_max"], ret["token"], ret["raw"], ret["code"], ret["logq"], ret["parent"], ret["count"],
        rows, backup_budget, config.max_nodes, config.max_depth - depth, config.max_depth - depth + 1,
        K=K_STABLE, PB=4, NB=16, SLOTS=width, SB=8, RETAIN=bool(slots),
    )
    out = {f"sel_{k}": v for k, v in sel.items()}
    out["sel_invalid"] = out["sel_invalid"].bool()
    out["sel_tie_risk"] = out["sel_tie_risk"].bool()
    if slots:
        out.update({f"ret_{k}": v for k, v in ret.items()})
    return out
