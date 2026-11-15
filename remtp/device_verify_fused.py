"""Fused host-free residual-hit verification (engineering, 2026-11-15).

``REMTP_DYNAMIC_TREE_VERIFY_FUSED=1`` (async mode) replaces ``device_verify.
verify_async`` -- about 265 small kernels per round -- with four Triton kernels
plus the two random draws.  Every vocabulary pass is split over ``SPLITS``
blocks per row and reduced in a fixed order (deterministic):

1. ``_stats_partial``: clamped mass and maximum of the 10 target rows and the
   3 trunk draft rows;
2. ``_trunk_scalars``: totals, p(y), q(y), the v4-safe allowance (float32 op
   order of ``_plan_vectors``, IEEE ``sqrt_rn``/``div_rn``), the float64 gate,
   boosted value and scale per trunk position;
3. ``_draw_partial``: per trunk position the boosted row h, the residual
   ``(h - q)+``, its mass and the exponential-race maxima of ``r / E`` and of
   the ``residual_distribution_nosync`` fallback ``h / E``; per tree node the
   exponential-race maximum of its normalized target row;
4. ``_decide``: argmax reduction (largest key, then lowest index, as
   ``torch.argmax``), acceptance ``u <= min(1, h(y)/q(y))`` in float64, first
   rejection, sibling hit, vLLM's padded output row, accepted slots, their
   number and the leaf row.

Differences to ``verify_async`` are confined to the last bit: row sums are
reduced in another order, the boosted row is not divided by its own sum (it is
``1`` up to rounding by construction of the scale), and the exponential race
uses the unnormalized residual (a positive scale of the normalized one).  A
decision can only change when a value lies within such a difference of its
threshold.  The randomness budget is unchanged.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

SPLITS = 32
BLOCK = 1024


def enabled() -> bool:
    return os.getenv("REMTP_DYNAMIC_TREE_VERIFY_FUSED", "0") == "1"


@triton.jit
def _stats_partial(t_ptr, t_stride, d_ptr, d_stride, trunk_ptr, part_ptr, V, CHUNK, NT: tl.constexpr,
                   S: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    if row < NT:
        base = t_ptr + row.to(tl.int64) * t_stride
    else:
        base = d_ptr + tl.load(trunk_ptr + (row - NT)) * d_stride
    lo = split * CHUNK
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    high = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, CHUNK, BLOCK):
        offs = lo + start + tl.arange(0, BLOCK)
        m = (offs < V) & (start + tl.arange(0, BLOCK) < CHUNK)
        x = tl.load(base + offs, mask=m, other=0.0)
        c = tl.where(x < 0.0, 0.0, x)
        acc += c
        high = tl.maximum(high, c)
    tl.store(part_ptr + (row * S + split) * 2 + 0, tl.sum(acc, axis=0))
    tl.store(part_ptr + (row * S + split) * 2 + 1, tl.max(high, axis=0))


@triton.jit
def _trunk_scalars(t_ptr, t_stride, d_ptr, d_stride, tokens_ptr, trunk_ptr, part_ptr, consts_ptr, delta_ptr,
                   stats_ptr, scal_ptr, NT: tl.constexpr, NR: tl.constexpr, S: tl.constexpr, GATE: tl.constexpr):
    # totals and maxima of all rows (fixed order over splits)
    r = tl.arange(0, 16)
    rm = r < NR
    s = tl.arange(0, S)
    sums = tl.load(part_ptr + (r[:, None] * S + s[None, :]) * 2 + 0, mask=rm[:, None], other=0.0)
    highs = tl.load(part_ptr + (r[:, None] * S + s[None, :]) * 2 + 1, mask=rm[:, None], other=0.0)
    tl.store(stats_ptr + r * 2 + 0, tl.sum(sums, axis=1), mask=rm)
    tl.store(stats_ptr + r * 2 + 1, tl.max(highs, axis=1), mask=rm)
    tl.debug_barrier()
    k = tl.arange(0, 4)
    km = k < 3
    trunk = tl.load(trunk_ptr + k, mask=km, other=0)
    prev = tl.load(trunk_ptr + k - 1, mask=km & (k > 0), other=-1)
    prow = tl.where(k == 0, 0, prev + 1)
    tok = tl.load(tokens_ptr + trunk, mask=km, other=0)
    total_p = tl.load(stats_ptr + prow * 2, mask=km, other=1.0)
    max_p = tl.load(stats_ptr + prow * 2 + 1, mask=km, other=0.0)
    total_q = tl.load(stats_ptr + (NT + k) * 2, mask=km, other=1.0)
    p_raw = tl.load(t_ptr + prow.to(tl.int64) * t_stride + tok, mask=km, other=0.0)
    q_raw = tl.load(d_ptr + trunk * d_stride + tok, mask=km, other=0.0)
    p = libdevice.div_rn(tl.where(p_raw < 0.0, 0.0, p_raw), total_p)
    q = libdevice.div_rn(tl.where(q_raw < 0.0, 0.0, q_raw), total_q)
    p_max = libdevice.div_rn(max_p, total_p)
    delta = tl.load(delta_ptr + k, mask=km, other=0.0)
    inner = 2.0 * delta * p * (1.0 - p)
    baseline = tl.minimum(tl.sqrt_rn(tl.where(inner < 0.0, 0.0, inner)), 1.0 - p)
    sat = q - p
    sat = tl.where(sat < 0.0, 0.0, sat)
    floor = tl.minimum(baseline, sat)
    p64 = p.to(tl.float64)
    if GATE:
        min_p = tl.load(consts_ptr + 0)
        min_ratio = tl.load(consts_ptr + 1)
        denom = p_max.to(tl.float64)
        denom = tl.where(denom < 1e-30, 1e-30, denom)
        gated = (p64 < min_p) | ((p64 / denom) < min_ratio)
        floor = tl.where(gated, 0.0, floor)
    boosted = p64 + floor.to(tl.float64)
    boosted = tl.where(boosted < p64, p64, boosted)
    boosted = tl.where(boosted > 1.0, 1.0, boosted)
    boost = (boosted > p64) & (p64 < 1.0)
    one_minus = 1.0 - p64
    one_minus = tl.where(one_minus < 1e-30, 1e-30, one_minus)
    scale = ((1.0 - boosted) / one_minus).to(tl.float32)
    # layout per k: p, q, boosted(f32), scale, boost, tok, prow, total_p, total_q
    tl.store(scal_ptr + k * 9 + 0, p, mask=km)
    tl.store(scal_ptr + k * 9 + 1, q, mask=km)
    tl.store(scal_ptr + k * 9 + 2, boosted.to(tl.float32), mask=km)
    tl.store(scal_ptr + k * 9 + 3, scale, mask=km)
    tl.store(scal_ptr + k * 9 + 4, boost.to(tl.float32), mask=km)
    tl.store(scal_ptr + k * 9 + 5, tok.to(tl.float32), mask=km)
    tl.store(scal_ptr + k * 9 + 6, prow.to(tl.float32), mask=km)
    tl.store(scal_ptr + k * 9 + 7, total_p, mask=km)
    tl.store(scal_ptr + k * 9 + 8, total_q, mask=km)


@triton.jit
def _argmax_update(best, best_idx, key, offs, m):
    key = tl.where(m, key, -1.0)
    better = key > best
    return tl.where(better, key, best), tl.where(better, offs, best_idx)


@triton.jit
def _argmax_finish(best, best_idx):
    top = tl.max(best, axis=0)
    return top, tl.min(tl.where(best == top, best_idx, 2147483647), axis=0)


@triton.jit
def _draw_partial(t_ptr, t_stride, d_ptr, d_stride, noise_ptr, noise_stride, trunk_ptr, stats_ptr, scal_ptr,
                  part_ptr, V, CHUNK, S: tl.constexpr, BLOCK: tl.constexpr):
    """Per (row, split): [key, index] maxima; trunk rows add the fallback maxima, residual mass, h(token)."""
    row = tl.program_id(0)
    split = tl.program_id(1)
    lo = split * CHUNK
    out = part_ptr + (row * S + split) * 6
    if row < 3:
        prow = tl.load(scal_ptr + row * 9 + 6).to(tl.int64)
        trunk = tl.load(trunk_ptr + row)
        boosted = tl.load(scal_ptr + row * 9 + 2)
        scale = tl.load(scal_ptr + row * 9 + 3)
        boost = tl.load(scal_ptr + row * 9 + 4) != 0.0
        tok = tl.load(scal_ptr + row * 9 + 5).to(tl.int64)
        total_p = tl.load(scal_ptr + row * 9 + 7)
        total_q = tl.load(scal_ptr + row * 9 + 8)
        pbase = t_ptr + prow * t_stride
        qbase = d_ptr + trunk * d_stride
        nbase = noise_ptr + row.to(tl.int64) * noise_stride
        best = tl.full([BLOCK], -1.0, dtype=tl.float32)
        best_idx = tl.zeros([BLOCK], dtype=tl.int32)
        fb = tl.full([BLOCK], -1.0, dtype=tl.float32)
        fb_idx = tl.zeros([BLOCK], dtype=tl.int32)
        rsum = tl.zeros([BLOCK], dtype=tl.float32)
        hy = tl.zeros([BLOCK], dtype=tl.float32)
        for start in range(0, CHUNK, BLOCK):
            local = start + tl.arange(0, BLOCK)
            offs = lo + local
            m = (offs < V) & (local < CHUNK)
            x = tl.load(pbase + offs, mask=m, other=0.0)
            y = tl.load(qbase + offs, mask=m, other=0.0)
            e = tl.load(nbase + offs, mask=m, other=1.0)
            pn = libdevice.div_rn(tl.where(x < 0.0, 0.0, x), total_p)
            h = tl.where(boost, tl.where(offs == tok, boosted, pn * scale), pn)
            qn = libdevice.div_rn(tl.where(y < 0.0, 0.0, y), total_q)
            res = h - qn
            res = tl.where(res > 0.0, res, 0.0)
            rsum += tl.where(m, res, 0.0)
            hy += tl.where(m & (offs == tok), h, 0.0)
            best, best_idx = _argmax_update(best, best_idx, libdevice.div_rn(res, e), offs, m)
            fb, fb_idx = _argmax_update(fb, fb_idx, libdevice.div_rn(h, e), offs, m)
        top, idx = _argmax_finish(best, best_idx)
        ftop, fidx = _argmax_finish(fb, fb_idx)
        tl.store(out + 0, top)
        tl.store(out + 1, idx.to(tl.float32))
        tl.store(out + 2, ftop)
        tl.store(out + 3, fidx.to(tl.float32))
        tl.store(out + 4, tl.sum(rsum, axis=0))
        tl.store(out + 5, tl.sum(hy, axis=0))
    else:
        node_row = row - 2          # target row of node (row - 3)
        total = tl.load(stats_ptr + node_row * 2)
        base = t_ptr + node_row.to(tl.int64) * t_stride
        nbase = noise_ptr + row.to(tl.int64) * noise_stride
        best = tl.full([BLOCK], -1.0, dtype=tl.float32)
        best_idx = tl.zeros([BLOCK], dtype=tl.int32)
        for start in range(0, CHUNK, BLOCK):
            local = start + tl.arange(0, BLOCK)
            offs = lo + local
            m = (offs < V) & (local < CHUNK)
            x = tl.load(base + offs, mask=m, other=0.0)
            e = tl.load(nbase + offs, mask=m, other=1.0)
            pn = libdevice.div_rn(tl.where(x < 0.0, 0.0, x), total)
            best, best_idx = _argmax_update(best, best_idx, libdevice.div_rn(pn, e), offs, m)
        top, idx = _argmax_finish(best, best_idx)
        tl.store(out + 0, top)
        tl.store(out + 1, idx.to(tl.float32))


@triton.jit
def _reduce_argmax(part_ptr, row, which: tl.constexpr, S: tl.constexpr):
    s = tl.arange(0, S)
    keys = tl.load(part_ptr + (row * S + s) * 6 + which)
    idx = tl.load(part_ptr + (row * S + s) * 6 + which + 1).to(tl.int32)
    top = tl.max(keys, axis=0)
    return tl.min(tl.where(keys == top, idx, 2147483647), axis=0).to(tl.int64)


@triton.jit
def _decide(uniform_ptr, scal_ptr, part_ptr, tokens_ptr, trunk_ptr, siblings_ptr,
            out_ptr, accepted_ptr, moves_ptr, leaf_ptr, W: tl.constexpr, WB: tl.constexpr, S: tl.constexpr):
    k = tl.arange(0, 4)
    km = k < 3
    s = tl.arange(0, S)
    u = tl.load(uniform_ptr + k, mask=km, other=0.0).to(tl.float64)
    hy = tl.sum(tl.load(part_ptr + (k[:, None] * S + s[None, :]) * 6 + 5, mask=km[:, None], other=0.0), axis=1)
    q = tl.load(scal_ptr + k * 9 + 1, mask=km, other=1.0).to(tl.float64)
    q = tl.where(q < 1e-30, 1e-30, q)
    ratio = hy.to(tl.float64) / q
    ratio = tl.where(ratio > 1.0, 1.0, ratio)
    reject = km & ~(u <= ratio)
    first = tl.min(tl.where(reject, k, 3), axis=0)
    full = first >= 3
    at = tl.minimum(first, 2)
    rsum = tl.sum(tl.load(part_ptr + (at * S + s) * 6 + 4), axis=0)
    correction = tl.where(rsum > 1e-30, _reduce_argmax(part_ptr, at, 0, S), _reduce_argmax(part_ptr, at, 2, S))
    cand0 = tl.load(siblings_ptr + at * 2 + 0)
    cand1 = tl.load(siblings_ptr + at * 2 + 1)
    hit0 = (cand0 >= 0) & (tl.load(tokens_ptr + tl.maximum(cand0, 0)) == correction)
    hit1 = (cand1 >= 0) & (tl.load(tokens_ptr + tl.maximum(cand1, 0)) == correction)
    hit_node = tl.where(hit0, cand0, tl.where(hit1, cand1, -1))
    has_hit = (~full) & (hit_node >= 0)
    leaf_bonus = _reduce_argmax(part_ptr, 3 + tl.load(trunk_ptr + 2), 0, S)
    hit_anchor = _reduce_argmax(part_ptr, 3 + tl.maximum(hit_node, 0), 0, S)
    pos = tl.arange(0, WB)
    trunk_tok = tl.load(tokens_ptr + tl.load(trunk_ptr + tl.minimum(pos, 2)))
    out = tl.where(pos < first, trunk_tok, -1)
    out = tl.where(full & (pos == 3), leaf_bonus, out)
    out = tl.where((~full) & (pos == first), correction, out)
    out = tl.where(has_hit & (pos == first + 1), hit_anchor, out)
    tl.store(out_ptr + pos, out.to(tl.int32), mask=pos < W)
    trunk_k = tl.load(trunk_ptr + k, mask=km, other=0)
    accepted = tl.where(k < first, trunk_k, tl.where(has_hit & (k == first), hit_node, 0))
    tl.store(accepted_ptr + k, accepted, mask=km)
    moves = tl.minimum(first, 3) + has_hit.to(tl.int64)
    tl.store(moves_ptr, moves)
    leaf = tl.sum(tl.where(k == tl.maximum(moves - 1, 0), accepted, 0), axis=0)
    tl.store(leaf_ptr, tl.where(moves > 0, leaf + 1, 0))


_CONSTS: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
_BUFFERS: dict[Any, dict[str, torch.Tensor]] = {}


def _constants(config: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (config.relaxation_min_target_probability, config.relaxation_min_target_ratio,
           tuple(config.relaxation_delta(d) for d in (1, 2, 3)), str(device))
    if key not in _CONSTS:
        consts = torch.tensor([config.relaxation_min_target_probability, config.relaxation_min_target_ratio],
                              dtype=torch.float64).to(device)
        deltas = torch.tensor([config.relaxation_delta(d) for d in (1, 2, 3)] + [0.0], dtype=torch.float32).to(device)
        _CONSTS[key] = (consts, deltas)
    return _CONSTS[key]


def _buffers(device: torch.device, rows: int) -> dict[str, torch.Tensor]:
    key = (str(device), rows)
    if key not in _BUFFERS:
        f32 = torch.float32
        _BUFFERS[key] = dict(
            stats_part=torch.empty((rows + 3) * SPLITS * 2, dtype=f32, device=device),
            stats=torch.empty((rows + 3) * 2, dtype=f32, device=device),
            scalars=torch.empty(4 * 9, dtype=f32, device=device),
            draw_part=torch.empty((3 + rows - 1) * SPLITS * 6, dtype=f32, device=device),
        )
    return _BUFFERS[key]


def verify_async_fused(state: Any, target_probs_by_input: torch.Tensor, output_width: int,
                       generator: torch.Generator, *, config: Any) -> torch.Tensor:
    from remtp import device_verify

    if not device_verify.supported(config, frozenset()):
        raise RuntimeError("fused async verification needs the v4-safe residual_hit_anchor configuration")
    target = target_probs_by_input.contiguous()
    if target.dtype != torch.float32:
        target = target.float()
    draft = state.probs
    device = target.device
    rows_t, vocabulary = target.shape
    count = state.tokens.shape[0]
    if rows_t != count + 1 or state.trunk.shape[0] != 3:
        raise RuntimeError("fused async verification expects 10 target rows and a depth-3 trunk")
    uniforms = torch.rand(3, device=device, generator=generator)
    noise = torch.empty(3 + count, vocabulary, device=device, dtype=torch.float32)
    noise.exponential_(generator=generator)
    buffers = _buffers(device, rows_t)
    chunk = triton.cdiv(vocabulary, SPLITS)
    _stats_partial[(rows_t + 3, SPLITS)](target, target.stride(0), draft, draft.stride(0), state.trunk,
                                         buffers["stats_part"], vocabulary, chunk, NT=rows_t, S=SPLITS,
                                         BLOCK=BLOCK, num_warps=4)
    consts, deltas = _constants(config, device)
    _trunk_scalars[(1,)](target, target.stride(0), draft, draft.stride(0), state.tokens, state.trunk,
                         buffers["stats_part"], consts, deltas, buffers["stats"], buffers["scalars"],
                         NT=rows_t, NR=rows_t + 3, S=SPLITS, GATE=bool(config.target_support_gate), num_warps=4)
    _draw_partial[(3 + count, SPLITS)](target, target.stride(0), draft, draft.stride(0), noise, noise.stride(0),
                                       state.trunk, buffers["stats"], buffers["scalars"], buffers["draw_part"],
                                       vocabulary, chunk, S=SPLITS, BLOCK=BLOCK, num_warps=4)
    out = torch.empty(1, output_width, dtype=torch.int32, device=device)
    _decide[(1,)](uniforms, buffers["scalars"], buffers["draw_part"], state.tokens, state.trunk, state.siblings,
                  out, state.accepted, state.moves, state.leaf_row, W=output_width,
                  WB=triton.next_power_of_2(output_width), S=SPLITS, num_warps=1)
    device_verify.STATS["rounds"] += 1
    return out
