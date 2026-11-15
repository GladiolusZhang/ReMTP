"""Opt-in, sync-reduced RAVEN fast paths with reference-identical results.

Profiling on Jetson Orin NX showed that the per-round overhead of the dynamic
tree is dominated by host/device synchronisation (``float(tensor)``,
``.item()``) inside Python loops over full-vocabulary rows, not by GPU work.
This module re-implements the hot helpers so that each parent/node costs one
or two device transfers while computing exactly the same floating-point
values in the same order, consuming the request RNG identically, and emitting
identical diagnostics.  Whenever an input could make the fast rule differ from
the reference (ties at the top-k cutoff, unsupported options), the reference
implementation is used instead.

Flags (read by ``remtp.dynamic_tree_vllm``):

``REMTP_DYNAMIC_TREE_FAST_SELECT=1``
    candidate selection, sampled-trunk helpers and temperature lookup for the
    sampled-primary proposal modes.  Proposal validity is still checked, but
    after the exponential-race draw instead of before it; this only changes
    behaviour for requests that are rejected with an error anyway.
``REMTP_DYNAMIC_TREE_FAST_VERIFY=1``
    ``residual_hit_tree_verify`` with batched normalisation, one gather for all
    node probabilities, lazy residuals and one transfer per verified node.
``REMTP_DYNAMIC_TREE_ASYNC_HOST=1`` (2026-11-14)
    no blocking host/device round trip where the value is already on the
    host: host lists reach the device through pinned non-blocking copies,
    scalars are written with ``fill_``, a sampled token indexes through a
    one-element index tensor (a 0-dim CUDA index is read back by PyTorch), and
    the cached host temperature replaces ``float(temperature)``.  Device
    values, kernels, their order and RNG use are unchanged; only the host
    stops waiting, so later independent launches queue behind running GPU work.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import torch

ASYNC_HOST = os.getenv("REMTP_DYNAMIC_TREE_ASYNC_HOST", "0") == "1"
# Trunk verification values launched before the first host read (2026-11-15):
# boosted rows, verification probabilities and the request-RNG uniforms of every
# trunk node are queued behind the target forward; the host reads them with p/q
# and, after a rejection, resets the generator offset to where the per-node path
# would be, so every later draw is unchanged.
VERIFY_PRELAUNCH = os.getenv("REMTP_DYNAMIC_TREE_VERIFY_PRELAUNCH", "0") == "1"
PRELAUNCH_STATS = {"used": 0, "fallback": 0}


def to_device(values: Any, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """``torch.tensor(values, dtype=dtype, device=device)`` without a host wait.

    A pageable host-to-device copy synchronises the stream; a pinned buffer
    with ``non_blocking=True`` does not (PyTorch's caching host allocator keeps
    the buffer alive until the copy has run).
    """
    device = torch.device(device)
    if not ASYNC_HOST or device.type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def index_one(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``values[index].reshape(1)`` for a scalar index tensor, without reading it back."""
    if ASYNC_HOST:
        return values[index.reshape(1)]
    return values[index].reshape(1)

from remtp.dynamic_mtp_tree import (
    CandidateSelection,
    DynamicTreeConfig,
    DynamicTreeTopology,
    _candidate_boost_distribution,
    adaptive_trunk_relaxation_plan,
    adaptive_trunk_relaxation_plan_host,
    select_dynamic_candidates,
)
from remtp.cactus_mtp import cactus_target_distribution
from remtp.fixed6_microtree import TreeVerifyResult

SAMPLED_PRIMARY_MODES = frozenset({
    "sampled_primary_reopen",
    "sampled_primary_shadow",
    "residual_hit_anchor",
    "residual_hit_strict",
    "residual_hit_cactus",
})


# --------------------------------------------------------------------------
# Distribution helpers: same arithmetic as fixed6_microtree, no host syncs.
# --------------------------------------------------------------------------
def normalize_rows_nosync(values: torch.Tensor) -> torch.Tensor:
    """Row-wise ``normalize_distribution`` without ``float(total)``.

    Each row keeps its own full reduction (``row.sum()``), so the quotient is
    bit-identical to the reference; the degenerate branch is selected with
    ``torch.where`` instead of a host round-trip.
    """
    # Fewer eager launches (2026-11-13), same values: a 1-D row needs no
    # stack/unsqueeze, and the uniform fallback is a scalar operand of
    # ``where`` instead of a materialised full-vocabulary tensor.
    probs = values if values.dtype == torch.float32 else values.to(torch.float32)
    probs = probs.clamp_min(0.0)
    uniform = 1.0 / probs.shape[-1]
    if probs.ndim == 1:
        total = probs.sum()
        return torch.where(torch.isfinite(total) & (total > 1e-30), probs / total, uniform)
    totals = torch.stack([row.sum() for row in probs.unbind(0)])
    ok = (torch.isfinite(totals) & (totals > 1e-30)).unsqueeze(1)
    return torch.where(ok, probs / totals.unsqueeze(1), uniform)


def residual_distribution_nosync(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    residual = (target_probs.to(torch.float32) - draft_probs.to(torch.float32)).clamp_min(0.0)
    total = residual.sum()
    return torch.where(total > 1e-30, residual / total, normalize_rows_nosync(target_probs))


def sample_categorical_nosync(probs: torch.Tensor, generator: torch.Generator | None) -> int:
    return int(torch.multinomial(normalize_rows_nosync(probs), 1, generator=generator).item())


# --------------------------------------------------------------------------
# Candidate selection with two device transfers.
# --------------------------------------------------------------------------
@dataclass
class FastSelection:
    selection: CandidateSelection
    token_ids: list[int]
    token_views: list[torch.Tensor]
    raw_probs: list[float]
    trunk_id: int | None
    trunk_raw_prob: float | None
    valid: bool
    used_reference: bool


def _reference_selection(probs, *, depth, config, budget, trunk_token, valid) -> FastSelection:
    selection = select_dynamic_candidates(probs, depth=depth, config=config, budget=budget)
    ids = selection.token_ids
    pieces = [ids.to(torch.float64), probs[ids].to(torch.float64)]
    if trunk_token is not None:
        pieces += [trunk_token.reshape(1).to(torch.float64), probs[trunk_token.reshape(())].reshape(1).to(torch.float64)]
    flat = torch.cat(pieces).tolist()
    count = int(ids.numel())
    return FastSelection(
        selection=selection,
        token_ids=[int(x) for x in flat[:count]],
        token_views=[ids[i].reshape(()) for i in range(count)],
        raw_probs=flat[count:2 * count],
        trunk_id=int(flat[2 * count]) if trunk_token is not None else None,
        trunk_raw_prob=flat[2 * count + 1] if trunk_token is not None else None,
        valid=valid,
        used_reference=True,
    )


def _relevant_tie(values: torch.Tensor, config: DynamicTreeConfig) -> torch.Tensor:
    """Ties that could change the selection.

    A rank below ``min_draft_prob`` can never be chosen: eligibility needs
    ``q >= threshold >= min_draft_prob`` and a guarded sibling needs
    ``q >= min_draft_prob``.  Equal values below it (typically underflowed
    zeros of a peaked distribution) therefore cannot make the top-k prefix
    ambiguous for the decision, and need not defer to the full stable sort.
    The float32 comparison is conservative: float32(min_draft_prob) is at
    most min_draft_prob.
    """
    if values.numel() <= 1:
        return torch.zeros((), dtype=torch.bool, device=values.device)
    return ((values[:-1] == values[1:]) & (values[1:] >= config.min_draft_prob)).any()


def select_dynamic_candidates_fast(
    probs: torch.Tensor,
    *,
    depth: int,
    config: DynamicTreeConfig,
    budget: int | None = None,
    trunk_token: torch.Tensor | None = None,
) -> FastSelection:
    """Reference ``select_dynamic_candidates`` in two host/device transfers.

    The entropy, maximum and threshold are computed exactly as the reference
    (same device ops, same Python double arithmetic).  Eligibility of the top
    ranks is read from the same ``q >= threshold`` device comparison, and the
    guarded-sibling rule is evaluated on the same values.  A tie among the top
    ``max_children + 1`` values would make the top-k prefix ambiguous, so that
    case defers to the reference's stable full sort.
    """
    config.validate()
    if depth <= 0 or depth > config.max_depth:
        raise ValueError("candidate depth is outside the configured tree")
    vocabulary = probs.numel()
    q = normalize_rows_nosync(probs)
    q2 = normalize_rows_nosync(q)  # normalized_entropy renormalises its input
    entropy_t = -(q2 * q2.clamp_min(1e-30).log()).sum() / math.log(vocabulary)
    q_max_t = q.max()
    valid_t = torch.isfinite(probs).all() & (probs >= 0).all() & (probs.sum() > 0)
    entropy, q_max, valid_f = torch.stack((entropy_t, q_max_t, valid_t.to(torch.float32))).tolist()
    valid = bool(valid_f)
    if vocabulary <= 1:
        entropy = 0.0

    depth_ratio = depth / config.max_depth
    uncertainty = entropy + config.depth_exploration * depth_ratio
    threshold = max(
        config.min_draft_prob,
        q_max * math.exp(-config.threshold_sensitivity * uncertainty),
    )
    guarded_limit = min(config.max_children_per_parent, vocabulary)
    k = min(guarded_limit + 1, vocabulary)
    values, indices = torch.topk(q, k, sorted=True)
    tie = _relevant_tie(values, config)
    eligible_mask = q >= threshold
    pieces = [
        eligible_mask.sum().to(torch.float64).reshape(1),
        tie.to(torch.float64).reshape(1),
        values.to(torch.float64),
        indices.to(torch.float64),
        eligible_mask[indices].to(torch.float64),
        probs[indices].to(torch.float64),
    ]
    if trunk_token is not None:
        pieces += [trunk_token.reshape(1).to(torch.float64), probs[trunk_token.reshape(())].reshape(1).to(torch.float64)]
    flat = torch.cat(pieces).tolist()
    threshold_count = int(flat[0])
    if bool(flat[1]):
        return _reference_selection(probs, depth=depth, config=config, budget=budget, trunk_token=trunk_token, valid=valid)
    top_values = flat[2:2 + k]
    top_ids = [int(x) for x in flat[2 + k:2 + 2 * k]]
    top_eligible = [bool(x) for x in flat[2 + 2 * k:2 + 3 * k]]
    top_raw = flat[2 + 3 * k:2 + 4 * k]
    trunk_id = int(flat[2 + 4 * k]) if trunk_token is not None else None
    trunk_raw = flat[2 + 4 * k + 1] if trunk_token is not None else None

    if threshold_count == 0:
        selection = CandidateSelection(
            token_ids=torch.empty(0, dtype=torch.int64, device=probs.device),
            entropy=entropy, threshold=threshold, q_max=q_max,
            candidate_count_before_budget=0, threshold_candidate_count=0, sibling_added=False,
        )
        return FastSelection(selection, [], [], [], trunk_id, trunk_raw, valid, False)

    chosen_ranks = [j for j in range(guarded_limit) if top_eligible[j]]
    sibling_added = False
    for j in range(1, guarded_limit):
        guarded_prob = top_values[j]
        if (
            not top_eligible[j]
            and guarded_prob >= config.min_draft_prob
            and guarded_prob >= q_max * config.min_sibling_ratio
        ):
            chosen_ranks.append(j)
            sibling_added = True
    before_budget = threshold_count + sum(1 for j in chosen_ranks if not top_eligible[j])
    chosen_ranks = chosen_ranks[: config.max_children_per_parent]
    if budget is not None:
        chosen_ranks = chosen_ranks[: max(int(budget), 0)]
    views = [indices[j].reshape(()) for j in chosen_ranks]
    token_ids = (
        torch.stack(views) if views else torch.empty(0, dtype=torch.int64, device=probs.device)
    )
    selection = CandidateSelection(
        token_ids=token_ids, entropy=entropy, threshold=threshold, q_max=q_max,
        candidate_count_before_budget=before_budget, threshold_candidate_count=threshold_count,
        sibling_added=sibling_added,
    )
    return FastSelection(
        selection, [top_ids[j] for j in chosen_ranks], views, [top_raw[j] for j in chosen_ranks],
        trunk_id, trunk_raw, valid, False,
    )


def cactus_trunk_children_fast(
    probs: torch.Tensor,
    trunk_token: torch.Tensor,
    *,
    depth: int,
    backup_budget: int,
    config: DynamicTreeConfig,
) -> tuple[torch.Tensor, Any]:
    """Reference ``_cactus_trunk_children`` using the two-transfer selection."""
    fast = select_dynamic_candidates_fast(probs, depth=depth, config=config, trunk_token=trunk_token)
    if not fast.valid:
        raise ValueError("Invalid MTP proposal Q: nonfinite, negative, or zero mass; sampling aborted before RNG")
    backups = [view for tid, view in zip(fast.token_ids, fast.token_views) if tid != fast.trunk_id]
    backups = backups[: max(backup_budget, 0)]
    return torch.stack([trunk_token.reshape(())] + backups), fast.selection


# --------------------------------------------------------------------------
# Batched trunk children (2026-11-13): every parent of one tree level in one
# host transfer.  Decisions, RNG use and diagnostics equal the per-parent
# ``cactus_trunk_children_fast``; only the diagnostic threshold counts are read
# back later, in one transfer for the whole proposal (``resolve``).
# --------------------------------------------------------------------------
@dataclass
class BatchedChildren:
    candidate_ids: torch.Tensor           # sampled trunk first, then backups (device)
    candidate_probs: list[float]          # == probs[candidate_ids].tolist()
    entropy: float
    threshold: float
    q_max: float
    sibling_added: bool
    unchosen_backup_count: int            # chosen ranks that were not threshold-eligible
    count_tensor: torch.Tensor | None     # deferred device count of q >= threshold
    threshold_candidate_count: int | None = None
    candidate_count_before_budget: int | None = None


def resolve_batched_counts(items: list[BatchedChildren]) -> None:
    """Fill the diagnostic counts of every pending item with one transfer."""
    pending = [item for item in items if item.threshold_candidate_count is None]
    if not pending:
        return
    counts = torch.stack([item.count_tensor.to(torch.int64) for item in pending]).tolist()
    for item, count in zip(pending, counts):
        item.threshold_candidate_count = int(count)
        item.candidate_count_before_budget = int(count) + item.unchosen_backup_count if count else 0


def batched_trunk_children(
    rows: list[torch.Tensor],
    trunk_tokens: list[torch.Tensor],
    *,
    depth: int,
    backup_budget: int,
    config: DynamicTreeConfig,
) -> list[BatchedChildren]:
    config.validate()
    if depth <= 0 or depth > config.max_depth:
        raise ValueError("candidate depth is outside the configured tree")
    # Fewer device ops per parent (each eager op costs ~45 us of host time on
    # Jetson), with the same values as the per-parent path:
    #   * normalisation: for a valid row ``clamp / sum`` is exactly what
    #     ``normalize_rows_nosync`` returns; its uniform fallback only applies
    #     to invalid rows, which raise below before any decision is made;
    #   * q2 = q / q.sum(): q is already non-negative, so clamping is a no-op;
    #   * validity from the clamped total and the raw minimum (finite, positive
    #     total and no negative entry is the same predicate as the original);
    #   * q_max is the first top-k value; the decision-relevant tie test runs on
    #     the transferred float32 values with float32 semantics;
    #   * one float32 transfer (float32 -> Python float is exact; ids < 2**24).
    min_draft_f32 = float(torch.tensor(config.min_draft_prob, dtype=torch.float32))
    prepared, pieces = [], []
    for probs, trunk in zip(rows, trunk_tokens):
        vocabulary = probs.numel()
        p32 = probs if probs.dtype == torch.float32 else probs.to(torch.float32)
        clamped = p32.clamp_min(0.0)
        total = clamped.sum()
        q = clamped / total
        q2 = q / q.sum()
        entropy_t = -(q2 * q2.clamp_min(1e-30).log()).sum() / math.log(vocabulary)
        guarded_limit = min(config.max_children_per_parent, vocabulary)
        k = min(guarded_limit + 1, vocabulary)
        values, indices = torch.topk(q, k, sorted=True)
        trunk = trunk.reshape(())
        pieces.append(torch.cat((
            torch.stack((entropy_t, total, p32.min())), values, indices.to(torch.float32),
            p32[indices], trunk.to(torch.float32).reshape(1), index_one(p32, trunk),
        )))
        prepared.append((probs, q, values, indices, trunk, vocabulary, guarded_limit, k))
    flat = torch.cat(pieces).tolist() if pieces else []
    results, offset = [], 0
    for probs, q, values, indices, trunk, vocabulary, guarded_limit, k in prepared:
        width = 3 + 3 * k + 2
        row = flat[offset:offset + width]
        offset += width
        entropy, total_f, min_f = row[0], row[1], row[2]
        if not (math.isfinite(total_f) and total_f > 0.0 and min_f >= 0.0):
            raise ValueError("Invalid MTP proposal Q: nonfinite, negative, or zero mass; sampling aborted before RNG")
        top_values = row[3:3 + k]
        q_max = top_values[0]
        if any(top_values[j] == top_values[j + 1] and top_values[j + 1] >= min_draft_f32 for j in range(k - 1)):
            # Ambiguous top-k prefix: the per-parent path defers to the reference.
            ids, selection = cactus_trunk_children_fast(probs, trunk, depth=depth, backup_budget=backup_budget, config=config)
            item = BatchedChildren(ids, probs[ids].tolist(), selection.entropy, selection.threshold, selection.q_max,
                                   selection.sibling_added, 0, None, selection.threshold_candidate_count,
                                   selection.candidate_count_before_budget)
            results.append(item)
            continue
        if vocabulary <= 1:
            entropy = 0.0
        top_ids = [int(x) for x in row[3 + k:3 + 2 * k]]
        top_raw = row[3 + 2 * k:3 + 3 * k]
        trunk_id, trunk_raw = int(row[3 + 3 * k]), row[3 + 3 * k + 1]
        uncertainty = entropy + config.depth_exploration * (depth / config.max_depth)
        threshold = max(config.min_draft_prob, q_max * math.exp(-config.threshold_sensitivity * uncertainty))
        # Same comparison semantics as the device's ``q >= threshold`` (float32 tensor vs Python float).
        top_eligible = (torch.tensor(top_values, dtype=torch.float32) >= threshold).tolist()
        count_tensor = (q >= threshold).sum()
        if not top_eligible[0]:          # no token reaches the threshold
            results.append(BatchedChildren(trunk.reshape(1), [trunk_raw], entropy, threshold, q_max, False, 0,
                                           count_tensor, 0, 0))
            continue
        chosen = [j for j in range(guarded_limit) if top_eligible[j]]
        sibling_added = False
        for j in range(1, guarded_limit):
            if (not top_eligible[j] and top_values[j] >= config.min_draft_prob
                    and top_values[j] >= q_max * config.min_sibling_ratio):
                chosen.append(j)
                sibling_added = True
        unchosen = sum(1 for j in chosen if not top_eligible[j])
        chosen = chosen[: config.max_children_per_parent]
        backups = [j for j in chosen if top_ids[j] != trunk_id][: max(backup_budget, 0)]
        ids = torch.stack([trunk] + [indices[j].reshape(()) for j in backups])
        results.append(BatchedChildren(ids, [trunk_raw] + [top_raw[j] for j in backups], entropy, threshold, q_max,
                                       sibling_added, unchosen, count_tensor))
    return results


# --------------------------------------------------------------------------
# Sampling helpers with a cached temperature and deferred validity check.
# --------------------------------------------------------------------------
def cached_temperature(sampling_metadata: Any) -> float | None:
    temperatures = getattr(sampling_metadata, "temperature", None)
    if temperatures is None:
        return None
    cached = getattr(sampling_metadata, "_remtp_temperature", None)
    if cached is None:
        cached = float(temperatures[0])
        try:
            sampling_metadata._remtp_temperature = cached
        except (AttributeError, TypeError):
            pass
    return cached


def temperature_probs_fast(logits: torch.Tensor, sampling_metadata: Any) -> torch.Tensor:
    """``fixed6_vllm._temperature_probs`` with the host-side temperature cached."""
    temperatures = sampling_metadata.temperature
    if temperatures is None:
        ids = logits.argmax(dim=-1)
        return torch.nn.functional.one_hot(ids, num_classes=logits.shape[-1]).to(torch.float32)
    temperature = temperatures[0].to(device=logits.device, dtype=torch.float32)
    raw = logits.to(torch.float32)
    if cached_temperature(sampling_metadata) < 1e-5:
        ids = raw.argmax(dim=-1)
        return torch.nn.functional.one_hot(ids, num_classes=raw.shape[-1]).to(torch.float32)
    return torch.softmax(raw / temperature, dim=-1)


def sample_q_token_fast(probs: torch.Tensor, sampling_metadata: Any) -> torch.Tensor:
    temperature = cached_temperature(sampling_metadata)
    if temperature is None or temperature < 1e-5:
        return probs.argmax().reshape(1)
    noise = torch.empty_like(probs)
    generator = (getattr(sampling_metadata, "generators", None) or {}).get(0)
    if generator is not None:
        noise.exponential_(generator=generator)
    else:
        noise.exponential_()
    return (probs / noise).argmax().reshape(1)


def sample_branch_q_token_fast(
    probs: torch.Tensor,
    sampling_metadata: Any,
    *,
    parent_path: tuple[int, ...],
    proposal_round: int,
) -> torch.Tensor:
    temperature = cached_temperature(sampling_metadata)
    if temperature is None or temperature < 1e-5:
        return probs.argmax().reshape(1)
    request_generator = (getattr(sampling_metadata, "generators", None) or {}).get(0)
    initial_seed = int(request_generator.initial_seed()) if request_generator is not None else 0
    path_hash = 1469598103934665603
    for value in parent_path:
        path_hash ^= int(value) + 1
        path_hash = (path_hash * 1099511628211) & ((1 << 63) - 1)
    seed = (initial_seed ^ (int(proposal_round) * 0x9E3779B185EBCA87) ^ path_hash) & ((1 << 63) - 1)
    branch_generator = torch.Generator(device=probs.device)
    branch_generator.manual_seed(seed)
    noise = torch.empty_like(probs)
    noise.exponential_(generator=branch_generator)
    return (probs / noise).argmax().reshape(1)


# --------------------------------------------------------------------------
# Residual-hit verification with one transfer per verified node.
# --------------------------------------------------------------------------
def _boost_known(target_probs: torch.Tensor, token_id: int, boosted_probability: float, original: float) -> torch.Tensor:
    """``_candidate_boost_distribution`` given the already transferred ``original``."""
    boosted = min(max(float(boosted_probability), original), 1.0)
    if boosted <= original or original >= 1.0:
        return target_probs
    scale = (1.0 - boosted) / max(1.0 - original, 1e-30)
    result = target_probs * scale          # already a fresh tensor; no clone needed
    if ASYNC_HOST:
        result[token_id:token_id + 1].fill_(boosted)   # same float32 value, no host copy
    else:
        result[token_id] = boosted
    return result / result.sum().clamp_min(1e-30)


def _prelaunch_trunk(topology: DynamicTreeTopology, trunk: list[int], ids: torch.Tensor, target: torch.Tensor,
                     gathered: tuple[torch.Tensor, ...], generator: torch.Generator,
                     config: DynamicTreeConfig) -> dict[str, Any]:
    """Queue every trunk node's verification values before the first host read.

    Valid when the adaptive planner allocates exactly its floor (maximum ratio
    <= 1, no risk policy, no protected node): the per-node allowance is then
    ``min(baseline, saturation)``, zero at target-gated positions, computed
    with ``_plan_vectors``' op sequence.  The boosted rows are ``_boost_known``'s
    op sequence; the uniforms are drawn in the per-node path's order and the
    generator offsets after each are recorded for ``align_rng``.
    """
    device = target.device
    trunk_index = to_device(trunk, torch.int64, device)
    parent_index = to_device([0 if topology.nodes[n].parent is None else topology.nodes[n].parent + 1
                              for n in trunk], torch.int64, device)
    p = gathered[1].index_select(0, trunk_index).clamp_min(0.0)
    q = gathered[2].index_select(0, trunk_index).clamp_min(0.0)
    p_max = gathered[3].index_select(0, parent_index)
    base_delta = to_device([config.relaxation_delta(topology.nodes[n].depth) for n in trunk], torch.float32, device)
    baseline = torch.minimum(torch.sqrt((2.0 * base_delta * p * (1.0 - p)).clamp_min(0.0)), 1.0 - p)
    floor = torch.minimum(baseline, (q - p).clamp(min=0.0))
    p64 = p.double()
    if config.target_support_gate:
        relative = p64 / torch.clamp_min(p_max.double(), 1e-30)
        gated = (p64 < config.relaxation_min_target_probability) | (relative < config.relaxation_min_target_ratio)
        floor = torch.where(gated, torch.zeros_like(floor), floor)
    boosted = torch.clamp(torch.maximum(p64 + floor.double(), p64), max=1.0)
    boost = (boosted > p64) & (p64 < 1.0)
    scale = ((1.0 - boosted) / torch.clamp_min(1.0 - p64, 1e-30)).to(torch.float32)
    rows, verify_ys = [], []
    tokens = ids.index_select(0, trunk_index)
    for k in range(len(trunk)):
        p_row = target[parent_index[k]]
        result = p_row * scale[k]
        result.index_put_((tokens[k].reshape(1),), boosted[k].to(torch.float32).reshape(1))
        result = result / result.sum().clamp_min(1e-30)
        row = torch.where(boost[k], result, p_row)
        rows.append(row)
        verify_ys.append(row[tokens[k].reshape(1)])
    offsets = [generator.get_offset()]
    uniforms = []
    for _ in trunk:
        uniforms.append(torch.rand((), device=device, generator=generator).reshape(1))
        offsets.append(generator.get_offset())
    values = torch.cat((floor, torch.cat(verify_ys), torch.cat(uniforms)))
    return dict(rows=rows, values=values, offsets=offsets)


def residual_hit_tree_verify_fast(
    topology: DynamicTreeTopology,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    config: DynamicTreeConfig,
    protected_node_indices: frozenset[int] = frozenset(),
) -> TreeVerifyResult | None:
    """Reference ``residual_hit_tree_verify`` with the host syncs batched.

    Returns ``None`` for option combinations this path does not implement
    (EOS protection, risk policies, probability caps); callers then use the
    reference.  Decisions, RNG draws and diagnostics are otherwise identical.
    """
    config.validate()
    if config.support_mode not in {
        "sampled_primary_shadow", "residual_hit_anchor", "residual_hit_strict", "residual_hit_cactus",
    }:
        raise ValueError("residual_hit_tree_verify received an incompatible mode")
    if not config.append_target_anchor:
        raise ValueError("residual-hit verification requires a target anchor")
    if config.protect_eos_relaxation or config.relaxation_risk_policy != "none" or config.relaxation_max_probability_ratio:
        return None
    count = topology.node_count
    if draft_token_ids.shape != (count,):
        raise ValueError(f"draft_token_ids must have shape [{count}]")
    if draft_probs.ndim != 2 or draft_probs.shape[0] != count:
        raise ValueError(f"draft_probs must have shape [{count}, vocab]")
    if target_probs_by_input.shape != (count + 1, draft_probs.shape[1]):
        raise ValueError(f"target_probs_by_input must have shape [{count + 1}, vocab]")

    device = target_probs_by_input.device
    ids = draft_token_ids.to(device=device, dtype=torch.int64)
    draft = normalize_rows_nosync(draft_probs.to(device=device))
    target = normalize_rows_nosync(target_probs_by_input.to(device=device))
    path_to_index = {node.path: node.index for node in topology.nodes}
    trunk = []
    for depth in range(1, topology.max_depth + 1):
        path = (0,) * depth
        if path not in path_to_index:
            raise ValueError(f"residual-hit topology is missing sampled primary node {path}")
        trunk.append(path_to_index[path])

    protected = set(protected_node_indices)

    # One float32 transfer for every node (float32 -> Python float is exact,
    # ids < 2**24): its token, p(token | parent row), q(token), and every target
    # row's maximum.  The adaptive plan is then built from these values with
    # the planner's own op sequence on CPU tensors (identical plan, no extra
    # device launches or sync).
    parent_rows = to_device(
        [0 if node.parent is None else node.parent + 1 for node in topology.nodes],
        torch.int64, device,
    )
    node_rows = torch.arange(count, device=device)
    gathered = (ids.to(torch.float32), target[parent_rows, ids], draft[node_rows, ids], target.max(dim=-1).values)
    pre = None
    if (VERIFY_PRELAUNCH and device.type == "cuda" and generator is not None and config.adaptive_relaxation
            and not protected
            and config.adaptive_relaxation_max_ratio <= 1.0 and config.relaxation_risk_policy == "none"):
        pre = _prelaunch_trunk(topology, trunk, ids, target, gathered, generator, config)
        gathered = gathered + (pre["values"],)
    flat = torch.cat(gathered).tolist()
    ids_list = [int(x) for x in flat[:count]]
    p_list = flat[count:2 * count]
    q_list = flat[2 * count:3 * count]
    adaptive_plan = (
        adaptive_trunk_relaxation_plan_host(
            topology, p_list, q_list, flat[3 * count:4 * count + 1],
            config=config, protected_node_indices=frozenset(protected),
        )
        if config.adaptive_relaxation
        else {}
    )

    prelaunched: dict[int, tuple[torch.Tensor, float, float]] = {}
    if pre is not None:
        values = flat[4 * count + 1:]
        length = len(trunk)
        tvs, verify_ys, uniforms = values[:length], values[length:2 * length], values[2 * length:3 * length]
        if all(float(adaptive_plan[node]["adaptive_relaxation_tv"]) == tv for node, tv in zip(trunk, tvs)):
            prelaunched = {node: (pre["rows"][k], verify_ys[k], uniforms[k]) for k, node in enumerate(trunk)}
            PRELAUNCH_STATS["used"] += 1
        else:
            PRELAUNCH_STATS["fallback"] += 1
            # Never expected: fall back to the per-node path with the RNG rewound.
            generator.set_offset(pre["offsets"][0])
    trunk_draws = {node: k + 1 for k, node in enumerate(trunk)}

    statuses = {node.index: "NOT_VISITED" for node in topology.nodes}
    diagnostics: dict[int, dict[str, Any]] = {}
    accepted: list[int] = []

    def draw(probs):
        return sample_categorical_nosync(probs, generator)

    def align_rng(rejected_node: int | None) -> None:
        """Generator offset of the per-node path after the trunk uniforms it drew."""
        if prelaunched:
            drawn = trunk_draws[rejected_node] if rejected_node is not None else len(trunk)
            generator.set_offset(pre["offsets"][drawn])

    def verify_q_candidate(node_index: int, *, use_cactus: bool):
        node = topology.nodes[node_index]
        parent_row = 0 if node.parent is None else node.parent + 1
        p_row = target[parent_row]
        q_row = draft[node_index]
        token_id = ids_list[node_index]
        p_y = p_list[node_index]
        q_y = q_list[node_index]
        base_relaxation_weight = config.relaxation_weight(node.depth)
        base_effective_delta = config.relaxation_delta(node.depth)
        is_protected = node_index in protected
        node_plan = adaptive_plan.get(node_index) if use_cactus else None
        if node_plan is not None and node_plan.get("adaptive_risk_mode") == "strict":
            is_protected = True
        if node_plan is not None and node_index in prelaunched and use_cactus:
            relaxation_weight = float(node_plan["adaptive_depth_weight"])
            effective_delta = float(node_plan["adaptive_effective_delta"])
            verify_row = prelaunched[node_index][0]
        elif node_plan is not None:
            relaxation_weight = float(node_plan["adaptive_depth_weight"])
            effective_delta = float(node_plan["adaptive_effective_delta"])
            verify_row = _boost_known(p_row, token_id, p_y + float(node_plan["adaptive_relaxation_tv"]), p_y)
        else:
            relaxation_weight = base_relaxation_weight
            effective_delta = 0.0 if is_protected else base_effective_delta
            verify_row = (
                cactus_target_distribution(p_row.unsqueeze(0), ids[node_index:node_index + 1], effective_delta)[0]
                if use_cactus
                else p_row
            )
        if node_index in prelaunched and use_cactus:
            _, verify_y, random_value = prelaunched[node_index]
        else:
            verify_y, random_value = torch.stack((
                verify_row[token_id].to(torch.float32),
                torch.rand((), device=device, generator=generator),
            )).tolist()
        accept_probability = min(1.0, verify_y / max(q_y, 1e-30))
        keep = random_value <= accept_probability
        row = {
            "node": node_index,
            "proposal_role": (
                "sampled_primary" if node.path == (0,) * node.depth else "sampled_recovery_continuation"
            ),
            "trunk_position": node.depth,
            "target_probability": p_y,
            "draft_probability": q_y,
            "relaxation_depth_weight": relaxation_weight if use_cactus else 0.0,
            "base_relaxation_depth_weight": base_relaxation_weight if use_cactus else 0.0,
            "effective_relaxation_delta": effective_delta if use_cactus else 0.0,
            "base_effective_relaxation_delta": base_effective_delta if use_cactus else 0.0,
            "relaxation_tv": max(0.0, verify_y - p_y),
            "verification_probability": verify_y,
            "verification_rule": "cactus" if use_cactus else "strict",
            "accept_probability": accept_probability,
            "accept_random": random_value,
            "normal_survival": keep,
            "rescue_eligible": False,
            "rescued": False,
        }
        if node_plan is not None:
            row.update(node_plan)
            row["adaptive_relaxation_enabled"] = True
        else:
            row["adaptive_relaxation_enabled"] = False
        if protected_node_indices:
            row["relaxation_protected"] = is_protected
            row["target_eos_probability"] = 0.0
        # The residual is a pure computation; it is only needed on rejection.
        correction = None if keep else residual_distribution_nosync(verify_row, q_row)
        return keep, correction, row

    def result(output, terminal, surviving, score):
        return TreeVerifyResult(
            output_token_ids=tuple(ids_list[index] for index in accepted) + output,
            accepted_node_indices=tuple(accepted),
            selected_branch=topology.nodes[accepted[0]].branch if accepted else None,
            terminal=terminal,
            target_validation_nodes=count,
            evaluated_node_statuses=tuple(sorted(statuses.items())),
            node_diagnostics=tuple(diagnostics[index] for index in sorted(diagnostics)),
            surviving_paths=surviving,
            selected_path_score=score,
        )

    for position, trunk_node in enumerate(trunk, start=1):
        keep, correction_probs, row = verify_q_candidate(trunk_node, use_cactus=True)
        row["proposal_role"] = "sampled_cactus_primary"
        row["trunk_accepted"] = keep
        row["trunk_rejected"] = not keep
        diagnostics[trunk_node] = row
        statuses[trunk_node] = "SELECTED" if keep else "REJECT"
        if keep:
            accepted.append(trunk_node)
            for sibling in topology.children_of(topology.nodes[trunk_node].parent):
                if sibling != trunk_node and statuses[sibling] == "NOT_VISITED":
                    statuses[sibling] = "BACKUP_UNUSED"
            continue

        align_rng(trunk_node)
        correction = draw(correction_probs)
        siblings = tuple(
            sibling for sibling in topology.children_of(topology.nodes[trunk_node].parent)
            if sibling != trunk_node
        )
        hit_node = next((sibling for sibling in siblings if ids_list[sibling] == correction), None)
        residual_values = (
            correction_probs[to_device([ids_list[s] for s in siblings], torch.int64, device)].tolist()
            if siblings else []
        )
        for sibling, residual_value in zip(siblings, residual_values):
            is_hit = sibling == hit_node
            is_reused = is_hit and config.support_mode != "sampled_primary_shadow"
            statuses[sibling] = "CORRECTION_HIT" if is_hit else "PRUNE"
            diagnostics[sibling] = {
                "node": sibling,
                "proposal_role": "exact_residual_correction_candidate",
                "trunk_position": position,
                "target_probability": p_list[sibling],
                "draft_probability": q_list[sibling],
                "residual_target_probability": residual_value,
                "sampled_correction_token": correction,
                "correction_hit": is_hit,
                "rescue_eligible": is_reused,
                "rescued": is_reused,
            }

        if config.support_mode == "sampled_primary_shadow" or hit_node is None:
            return result(
                (correction,),
                "sampled-primary-shadow-correction" if config.support_mode == "sampled_primary_shadow"
                else "residual-correction-tree-miss",
                0, None,
            )

        accepted.append(hit_node)
        statuses[hit_node] = "SELECTED"
        hit_row = diagnostics[hit_node]
        hit_row["rescue_extension_tokens"] = 1
        hit_row["rescue_unlocked_tokens"] = 1
        hit_row["rescue_decision_mode"] = "exact-residual-token-hit"

        if config.support_mode == "residual_hit_anchor":
            bonus = draw(target[hit_node + 1])
            return result((bonus,), "residual-hit-target-anchor", 1, 1.0)

        current = hit_node
        continuation_is_cactus = config.support_mode == "residual_hit_cactus"
        while topology.nodes[current].depth < topology.max_depth:
            child = path_to_index.get(topology.nodes[current].path + (0,))
            if child is None:
                break
            keep_child, child_correction_probs, child_row = verify_q_candidate(child, use_cactus=continuation_is_cactus)
            diagnostics[child] = child_row
            statuses[child] = "SELECTED" if keep_child else "REJECT"
            if not keep_child:
                child_correction = draw(child_correction_probs)
                hit_row["rescue_unlocked_tokens"] = len(accepted) - position + 1
                return result(
                    (child_correction,),
                    "residual-hit-cactus-continuation-correction" if continuation_is_cactus
                    else "residual-hit-strict-continuation-correction",
                    1, 1.0,
                )
            accepted.append(child)
            current = child

        hit_row["rescue_unlocked_tokens"] = len(accepted) - position + 1
        bonus = draw(target[current + 1])
        return result(
            (bonus,),
            "residual-hit-cactus-continuation-bonus" if continuation_is_cactus
            else "residual-hit-strict-continuation-bonus",
            1, 1.0,
        )

    leaf = accepted[-1]
    align_rng(None)
    bonus = draw(target[leaf + 1])
    return result((bonus,), "sampled-primary-full-cactus-bonus", 1, 1.0)
