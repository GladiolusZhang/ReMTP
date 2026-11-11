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
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from remtp.dynamic_mtp_tree import (
    CandidateSelection,
    DynamicTreeConfig,
    DynamicTreeTopology,
    _candidate_boost_distribution,
    adaptive_trunk_relaxation_plan,
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
    probs = values.to(torch.float32).clamp_min(0.0)
    squeeze = probs.ndim == 1
    if squeeze:
        probs = probs.unsqueeze(0)
    totals = torch.stack([row.sum() for row in probs])
    ok = torch.isfinite(totals) & (totals > 1e-30)
    uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
    out = torch.where(ok.unsqueeze(1), probs / totals.unsqueeze(1), uniform)
    return out[0] if squeeze else out


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
    tie = (values[:-1] == values[1:]).any() if k > 1 else torch.zeros((), dtype=torch.bool, device=q.device)
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
    result = target_probs * scale
    result = result.clone()
    result[token_id] = boosted
    return result / result.sum().clamp_min(1e-30)


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
    adaptive_plan = (
        adaptive_trunk_relaxation_plan(
            topology, ids, draft, target, config=config, protected_node_indices=frozenset(protected),
        )
        if config.adaptive_relaxation
        else {}
    )

    # One transfer for every node: its token, p(token | parent row) and q(token).
    parent_rows = torch.tensor(
        [0 if node.parent is None else node.parent + 1 for node in topology.nodes],
        dtype=torch.int64, device=device,
    )
    node_rows = torch.arange(count, device=device)
    flat = torch.cat((
        ids.to(torch.float64),
        target[parent_rows, ids].to(torch.float64),
        draft[node_rows, ids].to(torch.float64),
    )).tolist()
    ids_list = [int(x) for x in flat[:count]]
    p_list = flat[count:2 * count]
    q_list = flat[2 * count:3 * count]

    statuses = {node.index: "NOT_VISITED" for node in topology.nodes}
    diagnostics: dict[int, dict[str, Any]] = {}
    accepted: list[int] = []

    def draw(probs):
        return sample_categorical_nosync(probs, generator)

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
        if node_plan is not None:
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

        correction = draw(correction_probs)
        siblings = tuple(
            sibling for sibling in topology.children_of(topology.nodes[trunk_node].parent)
            if sibling != trunk_node
        )
        hit_node = next((sibling for sibling in siblings if ids_list[sibling] == correction), None)
        residual_values = (
            correction_probs[torch.tensor([ids_list[s] for s in siblings], device=device)].tolist()
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
    bonus = draw(target[leaf + 1])
    return result((bonus,), "sampled-primary-full-cactus-bonus", 1, 1.0)
