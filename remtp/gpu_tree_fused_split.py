"""Split-parallel decision step of the device-side RAVEN tree (engineering only, 2026-11-15).

(Renamed from ``gpu_tree_fused`` when merging the Jetson branch, whose
``remtp/gpu_tree_fused.py`` is the default ``REMTP_DYNAMIC_TREE_GPU_BUILD_FUSED=1``.)

``REMTP_DYNAMIC_TREE_GPU_BUILD_FUSED=exact|fast`` (with ``GPU_BUILD=1``) replaces
``gpu_tree._level`` -- per-parent candidate selection and ``residual_coverage``
retention, about 250 tiny kernels per level -- with:

* row statistics (clamped mass, minimum, normalized entropy):
  ``exact`` keeps the per-row torch op sequence of ``gpu_tree.select_children``
  (bitwise-equal scalars); ``fast`` computes them in one Triton kernel for all
  rows (deterministic, but a different summation order, so values can differ
  in the last bit);
* one batched ``torch.topk`` over all parent rows (on the probabilities; the
  normalization is a positive scale, so the order is the same and the
  normalized values are recomputed exactly as ``p / total``);
* one Triton kernel for everything after the top-k: the stable (value desc,
  index asc) order, entropy threshold, sibling rule, per-parent cap, then the
  retention order (class, trunk rank, path value desc, path code) under the
  level budget, writing the same outputs as ``gpu_tree._level``.

The arithmetic follows ``gpu_tree`` operation for operation: float32 IEEE
division (``div_rn``) for ``p / total``, float64 ``exp``/``log`` from libdevice
(the CUDA math library torch also calls), float64 constants computed in Python
as the torch code does, NaN-propagating clamps.  In ``exact`` mode the outputs
equal ``gpu_tree._level`` bit for bit on the same inputs, except when a
probability tie at the top-k cut extends past the 32 retained values (the
``tie_risk`` flag the existing code already reports).
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

WIDTH = 32          # gpu_tree.K_STABLE
PARENT_BLOCK = 4    # at most 4 parent rows per level (D3/N9/C3)
SLOT_BLOCK = 8      # at most 7 retained slots per level


# --------------------------------------------------------------------------
# Row statistics (fast mode).
# --------------------------------------------------------------------------
@triton.jit
def _row_stats_kernel(rows_ptr, row_stride, out_ptr, V, inv_log_v, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    base = rows_ptr + r.to(tl.int64) * row_stride
    offs0 = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    low = tl.full([BLOCK], float("inf"), dtype=tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + offs0
        m = offs < V
        x = tl.load(base + offs, mask=m, other=0.0)
        c = tl.where(x < 0.0, 0.0, x)
        acc += c
        low = tl.minimum(low, tl.where(m, x, float("inf")))
    total = tl.sum(acc, axis=0)
    minimum = tl.min(low, axis=0)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + offs0
        m = offs < V
        x = tl.load(base + offs, mask=m, other=0.0)
        c = tl.where(x < 0.0, 0.0, x)
        acc2 += tl.where(m, libdevice.div_rn(c, total), 0.0)
    s2 = tl.sum(acc2, axis=0)
    acc3 = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + offs0
        m = offs < V
        x = tl.load(base + offs, mask=m, other=0.0)
        c = tl.where(x < 0.0, 0.0, x)
        q2 = libdevice.div_rn(libdevice.div_rn(c, total), s2)
        acc3 += tl.where(m, q2 * libdevice.log(tl.where(q2 < 1e-30, 1e-30, q2)), 0.0)
    entropy = -tl.sum(acc3, axis=0) * inv_log_v
    tl.store(out_ptr + r * 3 + 0, total)
    tl.store(out_ptr + r * 3 + 1, minimum)
    tl.store(out_ptr + r * 3 + 2, entropy)


def row_stats_fast(q_rows: torch.Tensor) -> torch.Tensor:
    rows, vocabulary = q_rows.shape
    q_rows = q_rows.contiguous()
    out = torch.empty(rows, 3, dtype=torch.float32, device=q_rows.device)
    inv = float(np.float32(1.0) / np.float32(math.log(vocabulary)))
    _row_stats_kernel[(rows,)](q_rows, q_rows.stride(0), out, vocabulary, inv, BLOCK=4096, num_warps=8)
    return out


SPLITS = 32
PART = 11   # per (row, split): sum, sum c*log c, min, then 4 x (value, index)


@triton.jit
def _level_partial_kernel(rows_ptr, row_stride, part_ptr, V, CHUNK, S: tl.constexpr, BLOCK: tl.constexpr):
    """Per (row, split): clamped mass, sum of c*log(c), raw minimum and the split's top 4 (value desc, index asc)."""
    row = tl.program_id(0)
    split = tl.program_id(1)
    base = rows_ptr + row.to(tl.int64) * row_stride
    lo = split * CHUNK
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc_log = tl.zeros([BLOCK], dtype=tl.float32)
    low = tl.full([BLOCK], float("inf"), dtype=tl.float32)
    for start in range(0, CHUNK, BLOCK):
        local = start + tl.arange(0, BLOCK)
        offs = lo + local
        m = (offs < V) & (local < CHUNK)
        x = tl.load(base + offs, mask=m, other=0.0)
        c = tl.where(x < 0.0, 0.0, x)
        acc += c
        acc_log += tl.where(m, c * libdevice.log(tl.where(c < 1e-30, 1e-30, c)), 0.0)
        low = tl.minimum(low, tl.where(m, x, float("inf")))
    out = part_ptr + (row * S + split) * 11
    tl.store(out + 0, tl.sum(acc, axis=0))
    tl.store(out + 1, tl.sum(acc_log, axis=0))
    tl.store(out + 2, tl.min(low, axis=0))
    s0 = -1
    s1 = -1
    s2 = -1
    for t in tl.static_range(4):
        best = tl.full([BLOCK], -1.0, dtype=tl.float32)
        best_idx = tl.full([BLOCK], 2147483647, dtype=tl.int32)
        for start in range(0, CHUNK, BLOCK):
            local = start + tl.arange(0, BLOCK)
            offs = lo + local
            m = (offs < V) & (local < CHUNK) & (offs != s0) & (offs != s1) & (offs != s2)
            x = tl.load(base + offs, mask=m, other=0.0)
            c = tl.where(m, tl.where(x < 0.0, 0.0, x), -1.0)
            better = c > best
            best_idx = tl.where(better, offs, best_idx)
            best = tl.where(better, c, best)
        top = tl.max(best, axis=0)
        idx = tl.min(tl.where(best == top, best_idx, 2147483647), axis=0)
        tl.store(out + 3 + 2 * t, top)
        tl.store(out + 4 + 2 * t, idx.to(tl.float32))
        if t == 0:
            s0 = idx
        if t == 1:
            s1 = idx
        if t == 2:
            s2 = idx


@triton.jit
def _merge_kernel(part_ptr, rows_ptr, row_stride, stats_ptr, topv_ptr, topi_ptr, P, V, inv_log_v,
                  S: tl.constexpr, PB: tl.constexpr, W: tl.constexpr):
    """Fixed-order reduction of the partials into ``row_stats``/``topk`` layout (top 4 first, then padding)."""
    r = tl.arange(0, PB)
    rm = r < P
    sp = tl.arange(0, S)
    base = part_ptr + (r[:, None] * S + sp[None, :]) * 11
    total = tl.sum(tl.load(base + 0, mask=rm[:, None], other=0.0), axis=1)
    clogc = tl.sum(tl.load(base + 1, mask=rm[:, None], other=0.0), axis=1)
    minimum = tl.min(tl.load(base + 2, mask=rm[:, None], other=float("inf")), axis=1)
    safe = tl.where(total > 0.0, total, 1.0)
    entropy = (libdevice.log(safe) - clogc / safe) * inv_log_v
    tl.store(stats_ptr + r * 3 + 0, total, mask=rm)
    tl.store(stats_ptr + r * 3 + 1, minimum, mask=rm)
    tl.store(stats_ptr + r * 3 + 2, entropy, mask=rm)
    c = tl.arange(0, S * 4)
    vals = tl.load(part_ptr + (r[:, None] * S + c[None, :] // 4) * 11 + 3 + 2 * (c[None, :] % 4),
                   mask=rm[:, None], other=-1.0)
    ids = tl.load(part_ptr + (r[:, None] * S + c[None, :] // 4) * 11 + 4 + 2 * (c[None, :] % 4),
                  mask=rm[:, None], other=0.0).to(tl.int32)
    for t in tl.static_range(4):
        top = tl.max(vals, axis=1)
        idx = tl.min(tl.where(vals == top[:, None], ids, 2147483647), axis=1)
        tl.store(topv_ptr + r * W + t, top, mask=rm)
        tl.store(topi_ptr + r * W + t, idx.to(tl.int64), mask=rm)
        vals = tl.where(ids == idx[:, None], -2.0, vals)
    j = tl.arange(0, W)
    pad = (j >= 4)[None, :] & rm[:, None]
    tl.store(topv_ptr + r[:, None] * W + j[None, :], tl.full([PB, W], -1.0, dtype=tl.float32), mask=pad)
    tl.store(topi_ptr + r[:, None] * W + j[None, :], (V + j[None, :] + r[:, None] * 0).to(tl.int64), mask=pad)


def partial_stats_topk(p32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fast`` mode: row statistics and the exact top 4 in two kernels (split over the vocabulary)."""
    rows, vocabulary = p32.shape
    device = p32.device
    chunk = triton.cdiv(vocabulary, SPLITS)
    part = torch.empty(rows * SPLITS * PART, dtype=torch.float32, device=device)
    _level_partial_kernel[(rows, SPLITS)](p32, p32.stride(0), part, vocabulary, chunk, S=SPLITS, BLOCK=1024,
                                          num_warps=4)
    stats = torch.empty(rows, 3, dtype=torch.float32, device=device)
    top_values = torch.empty(rows, WIDTH, dtype=torch.float32, device=device)
    top_ids = torch.empty(rows, WIDTH, dtype=torch.int64, device=device)
    inv = float(np.float32(1.0) / np.float32(math.log(vocabulary)))
    _merge_kernel[(1,)](part, p32, p32.stride(0), stats, top_values, top_ids, rows, vocabulary, inv,
                        S=SPLITS, PB=PARENT_BLOCK, W=WIDTH, num_warps=4)
    return stats, top_values, top_ids


def row_stats_exact(q_rows: torch.Tensor) -> torch.Tensor:
    """The per-row op sequence of ``gpu_tree.select_children`` (bitwise-equal scalars)."""
    rows, vocabulary = q_rows.shape
    stats = []
    for row in range(rows):
        p32 = q_rows[row] if q_rows.dtype == torch.float32 else q_rows[row].to(torch.float32)
        clamped = p32.clamp_min(0.0)
        total = clamped.sum()
        q = clamped / total
        q2 = q / q.sum()
        entropy = -(q2 * q2.clamp_min(1e-30).log()).sum() / math.log(vocabulary)
        stats.append(torch.stack((total, p32.min(), entropy)))
    return torch.stack(stats)


# --------------------------------------------------------------------------
# Selection + retention.
# --------------------------------------------------------------------------
@triton.jit
def _pick(rank, s, q, ids, pv):
    sel = rank == s
    value = tl.sum(tl.where(sel, q, 0.0), axis=1)
    index = tl.sum(tl.where(sel, ids, 0), axis=1)
    raw = tl.sum(tl.where(sel, pv, 0.0), axis=1)
    return value, index, raw


@triton.jit
def _select_retain_kernel(
    topv_ptr, topi_ptr, stats_ptr, rows_ptr, row_stride, sampled_ptr,
    pvalid_ptr, pcode_ptr, plogq_ptr, ncur_ptr, consts_ptr,
    sel_tok_ptr, sel_raw_ptr, sel_logq_ptr, sel_count_ptr, invalid_ptr, tie_ptr,
    entropy_ptr, threshold_ptr, qmax_ptr,
    ret_tok_ptr, ret_raw_ptr, ret_code_ptr, ret_logq_ptr, ret_parent_ptr, ret_count_ptr,
    P, min_draft_f32, backup_budget, max_nodes, depth, max_depth,
    W: tl.constexpr, PB: tl.constexpr, NC: tl.constexpr, SLOTS: tl.constexpr, SB: tl.constexpr,
    ZERO_ENTROPY: tl.constexpr,
):
    r = tl.arange(0, PB)
    rmask = r < P
    w = tl.arange(0, W)
    pv = tl.load(topv_ptr + r[:, None] * W + w[None, :], mask=rmask[:, None], other=0.0)
    ids = tl.load(topi_ptr + r[:, None] * W + w[None, :], mask=rmask[:, None], other=0)
    total = tl.load(stats_ptr + r * 3 + 0, mask=rmask, other=1.0)
    minimum = tl.load(stats_ptr + r * 3 + 1, mask=rmask, other=0.0)
    entropy32 = tl.load(stats_ptr + r * 3 + 2, mask=rmask, other=0.0)

    # Normalized values q = clamp(p, 0) / total (IEEE division, as torch).
    q = libdevice.div_rn(tl.where(pv < 0.0, 0.0, pv), total[:, None])
    # Stable order of the reference: value descending, index ascending.
    qi = q[:, :, None]
    qj = q[:, None, :]
    ii = ids[:, :, None]
    ij = ids[:, None, :]
    before = (qj > qi) | ((qj == qi) & (ij < ii))
    rank = tl.sum(before.to(tl.int32), axis=2)
    v0, i0, r0 = _pick(rank, 0, q, ids, pv)
    v1, i1, r1 = _pick(rank, 1, q, ids, pv)
    v2, i2, r2 = _pick(rank, 2, q, ids, pv)
    v3, i3, r3 = _pick(rank, 3, q, ids, pv)
    last = tl.min(q, axis=1)

    finite = (total == total) & (tl.abs(total) != float("inf"))
    invalid = ~(finite & (total > 0.0) & (minimum >= 0.0))
    tie_risk = (v3 == last) & (v3 >= min_draft_f32)

    min_draft = tl.load(consts_ptr + 0)
    depth_term = tl.load(consts_ptr + 1)
    neg_sens = tl.load(consts_ptr + 2)
    sib_ratio = tl.load(consts_ptr + 3)
    value_mult = tl.load(consts_ptr + 4)
    tiny = tl.load(consts_ptr + 5)

    entropy = entropy32.to(tl.float64)
    if ZERO_ENTROPY:
        entropy = entropy * 0.0
    q_max = v0.to(tl.float64)
    uncertainty = entropy + depth_term
    threshold = q_max * libdevice.exp(neg_sens * uncertainty)
    threshold = tl.where(threshold < min_draft, min_draft, threshold)
    threshold32 = threshold.to(tl.float32)
    e0 = v0 >= threshold32
    e1 = v1 >= threshold32
    e2 = v2 >= threshold32
    sib_cut = q_max * sib_ratio
    v1d = v1.to(tl.float64)
    v2d = v2.to(tl.float64)
    s1 = (~e1) & (v1d >= min_draft) & (v1d >= sib_cut)
    s2 = (~e2) & (v2d >= min_draft) & (v2d >= sib_cut)
    c1 = (e1 | s1) & e0
    c2 = (e2 | s2) & e0
    trunk = tl.load(sampled_ptr + r, mask=rmask, other=0)
    b0 = e0 & (i0 != trunk)
    b1 = c1 & (i1 != trunk)
    b2 = c2 & (i2 != trunk)
    k0 = b0.to(tl.int64)
    k1 = k0 + b1.to(tl.int64)
    k2 = k1 + b2.to(tl.int64)
    t0 = b0 & (k0 <= backup_budget)
    t1 = b1 & (k1 <= backup_budget)
    t2 = b2 & (k2 <= backup_budget)
    count = 1 + t0.to(tl.int64) + t1.to(tl.int64) + t2.to(tl.int64)
    p10 = t0 & (k0 == 1)
    p11 = t1 & (k1 == 1)
    p12 = t2 & (k2 == 1)
    p20 = t0 & (k0 == 2)
    p21 = t1 & (k1 == 2)
    p22 = t2 & (k2 == 2)
    tok1 = tl.where(p10, i0, tl.where(p11, i1, tl.where(p12, i2, 0)))
    raw1 = tl.where(p10, r0, tl.where(p11, r1, tl.where(p12, r2, 0.0)))
    tok2 = tl.where(p20, i0, tl.where(p21, i1, tl.where(p22, i2, 0)))
    raw2 = tl.where(p20, r0, tl.where(p21, r1, tl.where(p22, r2, 0.0)))
    raw0 = tl.load(rows_ptr + r.to(tl.int64) * row_stride + trunk, mask=rmask, other=0.0)

    plogq = tl.load(plogq_ptr + r, mask=rmask, other=0.0)
    lq0 = plogq + libdevice.log(tl.where(raw0.to(tl.float64) < tiny, tiny, raw0.to(tl.float64)))
    lq1 = plogq + libdevice.log(tl.where(raw1.to(tl.float64) < tiny, tiny, raw1.to(tl.float64)))
    lq2 = plogq + libdevice.log(tl.where(raw2.to(tl.float64) < tiny, tiny, raw2.to(tl.float64)))

    tl.store(sel_tok_ptr + r * 3 + 0, trunk, mask=rmask)
    tl.store(sel_tok_ptr + r * 3 + 1, tok1, mask=rmask)
    tl.store(sel_tok_ptr + r * 3 + 2, tok2, mask=rmask)
    tl.store(sel_raw_ptr + r * 3 + 0, raw0, mask=rmask)
    tl.store(sel_raw_ptr + r * 3 + 1, raw1, mask=rmask)
    tl.store(sel_raw_ptr + r * 3 + 2, raw2, mask=rmask)
    tl.store(sel_logq_ptr + r * 3 + 0, lq0, mask=rmask)
    tl.store(sel_logq_ptr + r * 3 + 1, lq1, mask=rmask)
    tl.store(sel_logq_ptr + r * 3 + 2, lq2, mask=rmask)
    tl.store(sel_count_ptr + r, count, mask=rmask)
    tl.store(invalid_ptr + r, invalid, mask=rmask)
    tl.store(tie_ptr + r, tie_risk, mask=rmask)
    tl.store(entropy_ptr + r, entropy32, mask=rmask)
    tl.store(threshold_ptr + r, threshold, mask=rmask)
    tl.store(qmax_ptr + r, q_max, mask=rmask)

    if SLOTS > 0:
        col = tl.arange(0, 4)[None, :]
        pvalid = tl.load(pvalid_ptr + r, mask=rmask, other=0) != 0
        pcode = tl.load(pcode_ptr + r, mask=rmask, other=0)
        valid = (pvalid & rmask)[:, None] & (col < count[:, None]) & (col < 3)
        code = pcode[:, None] * 4 + col
        tok = tl.where(col == 0, trunk[:, None], tl.where(col == 1, tok1[:, None], tok2[:, None]))
        raw = tl.where(col == 0, raw0[:, None], tl.where(col == 1, raw1[:, None], raw2[:, None]))
        logq = tl.where(col == 0, lq0[:, None], tl.where(col == 1, lq1[:, None], lq2[:, None]))
        value = libdevice.exp(logq) * value_mult
        trunk_parent = (pcode == 0)[:, None]
        klass = tl.where(trunk_parent, 0, tl.where(col == 0, 1, 2))
        order_rank = tl.where(trunk_parent, col, 0)
        parent = (r[:, None] + col * 0).to(tl.int64)
        n: tl.constexpr = NC
        v_f = tl.reshape(valid, (n,))
        c_f = tl.reshape(klass, (n,))
        o_f = tl.reshape(order_rank, (n,))
        s_f = tl.reshape(code, (n,))
        val_f = tl.reshape(value, (n,))
        tok_f = tl.reshape(tok, (n,))
        raw_f = tl.reshape(raw, (n,))
        logq_f = tl.reshape(logq, (n,))
        par_f = tl.reshape(parent, (n,))
        ci = c_f[:, None]
        cj = c_f[None, :]
        ri = o_f[:, None]
        rj = o_f[None, :]
        vi = val_f[:, None]
        vj = val_f[None, :]
        si = s_f[:, None]
        sj = s_f[None, :]
        prec = (cj < ci) | ((cj == ci) & ((rj < ri) | ((rj == ri) & ((vj > vi) | ((vj == vi) & (sj < si))))))
        position = tl.sum((prec & v_f[None, :]).to(tl.int64), axis=1)
        ncur = tl.load(ncur_ptr)
        remaining = max_nodes - ncur
        level = tl.maximum(remaining - (max_depth - depth), 0)
        levels_left = max_depth - depth + 1
        # floor division; only used when remaining > 0
        spread = tl.maximum(tl.where(remaining > 0, remaining, 0) // levels_left, 1)
        budget = tl.where(remaining <= 0, 0, tl.minimum(level, spread))
        keep = v_f & (position < budget)
        out_slot = tl.sum((keep[None, :] & (sj < si)).to(tl.int64), axis=1)
        o = tl.arange(0, SB)
        hit = keep[None, :] & (out_slot[None, :] == o[:, None])
        tl.store(ret_tok_ptr + o, tl.sum(tl.where(hit, tok_f[None, :], 0), axis=1), mask=o < SLOTS)
        tl.store(ret_raw_ptr + o, tl.sum(tl.where(hit, raw_f[None, :], 0.0), axis=1), mask=o < SLOTS)
        tl.store(ret_code_ptr + o, tl.sum(tl.where(hit, s_f[None, :], 0), axis=1), mask=o < SLOTS)
        tl.store(ret_logq_ptr + o, tl.sum(tl.where(hit, logq_f[None, :], 0.0), axis=1), mask=o < SLOTS)
        tl.store(ret_parent_ptr + o, tl.sum(tl.where(hit, par_f[None, :], 0), axis=1), mask=o < SLOTS)
        tl.store(ret_count_ptr, tl.sum(keep.to(tl.int64), axis=0))


def _consts(depth: int, config: Any, device: torch.device) -> torch.Tensor:
    """float64 constants exactly as the torch code forms them from Python floats."""
    values = [
        float(config.min_draft_prob),
        config.depth_exploration * (depth / config.max_depth),
        -config.threshold_sensitivity,
        float(config.min_sibling_ratio),
        math.exp(config.depth_reward * depth / config.max_depth),
        1e-30,
    ]
    return torch.tensor(values, dtype=torch.float64).to(device, non_blocking=True)


_CONSTS: dict[tuple[Any, ...], torch.Tensor] = {}


def _cached_consts(depth: int, config: Any, device: torch.device) -> torch.Tensor:
    key = (depth, config.min_draft_prob, config.depth_exploration, config.max_depth,
           config.threshold_sensitivity, config.min_sibling_ratio, config.depth_reward, str(device))
    if key not in _CONSTS:
        _CONSTS[key] = _consts(depth, config, device)
    return _CONSTS[key]


def stats_mode() -> str:
    mode = os.getenv("REMTP_DYNAMIC_TREE_GPU_BUILD_FUSED", "1")
    return mode if mode in {"exact", "fast"} else ""


def split_level_exact(**kwargs: Any) -> dict[str, torch.Tensor]:
    return fused_level(**kwargs, mode="exact")


def split_level_fast(**kwargs: Any) -> dict[str, torch.Tensor]:
    return fused_level(**kwargs, mode="fast")


def fused_level(q: torch.Tensor, sampled: torch.Tensor, parent_valid: torch.Tensor, parent_code: torch.Tensor,
                parent_logq: torch.Tensor, n_current: torch.Tensor, *, depth: int, slots: int, backup_budget: int,
                config: Any, mode: str = "exact") -> dict[str, torch.Tensor]:
    """Same outputs as ``gpu_tree._level`` (keys ``sel_*`` and ``ret_*``)."""
    rows, vocabulary = q.shape
    if rows > PARENT_BLOCK or slots >= SLOT_BLOCK or config.max_children_per_parent != 3:
        raise ValueError("fused level supports at most 4 parents, 7 slots and 3 children")
    device = q.device
    p32 = q if q.dtype == torch.float32 else q.to(torch.float32)
    p32 = p32.contiguous()
    width = min(WIDTH, vocabulary)
    if width != WIDTH:
        raise ValueError("fused level needs a vocabulary of at least 32 tokens")
    if mode == "exact":
        stats = row_stats_exact(p32)
        top_values, top_ids = torch.topk(p32, width, dim=-1, sorted=True)
        top_values = top_values.contiguous()
        top_ids = top_ids.contiguous()
    else:
        stats, top_values, top_ids = partial_stats_topk(p32)
    i64, f32, f64 = torch.int64, torch.float32, torch.float64
    sel_tokens = torch.empty(rows, 3, dtype=i64, device=device)
    sel_raw = torch.empty(rows, 3, dtype=f32, device=device)
    sel_logq = torch.empty(rows, 3, dtype=f64, device=device)
    sel_count = torch.empty(rows, dtype=i64, device=device)
    invalid = torch.empty(rows, dtype=torch.bool, device=device)
    tie_risk = torch.empty(rows, dtype=torch.bool, device=device)
    entropy = torch.empty(rows, dtype=f32, device=device)
    threshold = torch.empty(rows, dtype=f64, device=device)
    q_max = torch.empty(rows, dtype=f64, device=device)
    out_slots = max(slots, 1)
    ret_token = torch.zeros(out_slots, dtype=i64, device=device)
    ret_raw = torch.zeros(out_slots, dtype=f32, device=device)
    ret_code = torch.zeros(out_slots, dtype=i64, device=device)
    ret_logq = torch.zeros(out_slots, dtype=f64, device=device)
    ret_parent = torch.zeros(out_slots, dtype=i64, device=device)
    ret_count = torch.zeros((), dtype=i64, device=device)
    min_draft_f32 = float(np.float32(config.min_draft_prob))
    _select_retain_kernel[(1,)](
        top_values, top_ids, stats, p32, p32.stride(0), sampled.reshape(-1).to(i64).contiguous(),
        parent_valid.contiguous(), parent_code.to(i64).contiguous(), parent_logq.to(f64).contiguous(),
        n_current.reshape(1).to(i64), _cached_consts(depth, config, device),
        sel_tokens, sel_raw, sel_logq, sel_count, invalid, tie_risk, entropy, threshold, q_max,
        ret_token, ret_raw, ret_code, ret_logq, ret_parent, ret_count,
        rows, min_draft_f32, max(backup_budget, 0), config.max_nodes, depth, config.max_depth,
        W=WIDTH, PB=PARENT_BLOCK, NC=PARENT_BLOCK * 4, SLOTS=slots, SB=SLOT_BLOCK, ZERO_ENTROPY=vocabulary <= 1, num_warps=4,
    )
    out = dict(sel_tokens=sel_tokens, sel_raw=sel_raw, sel_count=sel_count, sel_entropy=entropy,
               sel_invalid=invalid, sel_tie_risk=tie_risk, sel_threshold=threshold, sel_q_max=q_max,
               sel_logq=sel_logq)
    if slots:
        out.update(ret_token=ret_token[:slots], ret_raw=ret_raw[:slots], ret_code=ret_code[:slots],
                   ret_logq=ret_logq[:slots], ret_parent=ret_parent[:slots], ret_count=ret_count)
    return out
