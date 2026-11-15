"""Residual-hit tree verification on the device (engineering, 2026-11-15).

``REMTP_DYNAMIC_TREE_DEVICE_VERIFY=1`` replaces ``raven_fast_path.
residual_hit_tree_verify_fast`` (one host read per trunk node, then a host
decision) with device computation and a single host read per round.  The
verification rule is unchanged:

* the adaptive allowance is the v4-safe closed form the planner reduces to
  when ``adaptive_relaxation_max_ratio <= 1`` (no reallocation possible): the
  floor ``min(min(sqrt(2 delta p (1-p)), 1-p), (q-p)+)``, zero at
  target-gated positions -- the same float32/float64 op sequence as
  ``_plan_vectors``/``_plan_from_values`` and ``_prelaunch_trunk``;
* every trunk row is boosted with ``_boost_known``'s op sequence and the
  acceptance test is ``u <= min(1, h(y)/q(y))`` in float64;
* the first rejection draws the correction from the exact residual
  ``(h-q)+``; a sibling whose token equals it is reused and followed by one
  token from the unrelaxed target row (``residual_hit_anchor``); otherwise
  the round ends with the correction; a fully accepted trunk ends with a
  target bonus token.

What changes is how the randomness is consumed (accepted by the user on
2026-11-15): each round draws a fixed set -- one uniform per trunk position
and one exponential-race noise row per possible categorical draw (a
correction per trunk position, an anchor/bonus per tree node) -- from the
request generator in two calls, independent of the outcome.  Exponential-race
argmax is an exact categorical sample, so the output distribution is the same,
but the sampled tokens differ from the per-node path's.  A fixed consumption
is what lets a later step keep the whole round on the device.

Supported: ``residual_hit_anchor`` with adaptive relaxation, maximum ratio
<= 1, no risk policy, no probability cap, no EOS protection and no protected
nodes; anything else returns ``None`` (callers fall back).
"""
from __future__ import annotations

import os
from typing import Any

import torch

from remtp.fixed6_microtree import TreeVerifyResult
from remtp.raven_fast_path import normalize_rows_nosync, residual_distribution_nosync, to_device

STATS = {"rounds": 0}


def enabled() -> bool:
    return os.getenv("REMTP_DYNAMIC_TREE_DEVICE_VERIFY", "0") == "1"


def supported(config: Any, protected: frozenset[int]) -> bool:
    return (
        config.support_mode == "residual_hit_anchor"
        and config.append_target_anchor
        and config.adaptive_relaxation
        and config.adaptive_relaxation_max_ratio <= 1.0
        and config.relaxation_risk_policy == "none"
        and not config.relaxation_max_probability_ratio
        and not config.protect_eos_relaxation
        and not protected
    )


def _trunk_floor(p: torch.Tensor, q: torch.Tensor, p_max: torch.Tensor, base_delta: torch.Tensor,
                 config: Any) -> torch.Tensor:
    """Allocated allowance per trunk position (float32), as the v4-safe planner."""
    baseline = torch.minimum(torch.sqrt((2.0 * base_delta * p * (1.0 - p)).clamp_min(0.0)), 1.0 - p)
    floor = torch.minimum(baseline, (q - p).clamp(min=0.0))
    if config.target_support_gate:
        p64 = p.double()
        relative = p64 / torch.clamp_min(p_max.double(), 1e-30)
        gated = (p64 < config.relaxation_min_target_probability) | (relative < config.relaxation_min_target_ratio)
        floor = torch.where(gated, torch.zeros_like(floor), floor)
    return floor


def residual_hit_tree_verify_device(
    topology: Any,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    config: Any,
    protected_node_indices: frozenset[int] = frozenset(),
) -> TreeVerifyResult | None:
    if not supported(config, protected_node_indices) or generator is None:
        return None
    config.validate()
    count = topology.node_count
    device = target_probs_by_input.device
    ids = draft_token_ids.to(device=device, dtype=torch.int64)
    draft = normalize_rows_nosync(draft_probs.to(device=device))
    target = normalize_rows_nosync(target_probs_by_input.to(device=device))
    vocabulary = target.shape[1]

    # Host-known structure (from the topology the host already holds).
    path_to_index = {node.path: node.index for node in topology.nodes}
    trunk = []
    for depth in range(1, topology.max_depth + 1):
        path = (0,) * depth
        if path not in path_to_index:
            raise ValueError(f"residual-hit topology is missing sampled primary node {path}")
        trunk.append(path_to_index[path])
    length = len(trunk)
    parent_rows = [0 if topology.nodes[n].parent is None else topology.nodes[n].parent + 1 for n in trunk]
    siblings = [[s for s in topology.children_of(topology.nodes[n].parent) if s != n] for n in trunk]
    width = max(1, max(len(s) for s in siblings))
    sibling_matrix = [s + [-1] * (width - len(s)) for s in siblings]
    deltas = [config.relaxation_delta(topology.nodes[n].depth) for n in trunk]
    structure = to_device(trunk + parent_rows + [x for row in sibling_matrix for x in row], torch.int64, device)
    trunk_index = structure[:length]
    parent_index = structure[length:2 * length]
    sibling_index = structure[2 * length:].view(length, width)
    base_delta = to_device(deltas, torch.float32, device)

    tokens = ids.index_select(0, trunk_index)
    p_rows = target.index_select(0, parent_index)
    q_rows = draft.index_select(0, trunk_index)
    p = p_rows.gather(1, tokens[:, None])[:, 0]
    q = q_rows.gather(1, tokens[:, None])[:, 0]
    p_max = target.max(dim=-1).values.index_select(0, parent_index)
    floor = _trunk_floor(p.clamp_min(0.0), q.clamp_min(0.0), p_max, base_delta, config)

    # _boost_known: boosted = min(max(p + tv, p), 1); rows scaled, token set, renormalized.
    p64 = p.double()
    boosted = torch.clamp(torch.maximum(p64 + floor.double(), p64), max=1.0)
    boost = (boosted > p64) & (p64 < 1.0)
    scale = ((1.0 - boosted) / torch.clamp_min(1.0 - p64, 1e-30)).to(torch.float32)
    verify_rows, residual_rows = [], []
    for k in range(length):
        row = p_rows[k] * scale[k]
        row.index_put_((tokens[k].reshape(1),), boosted[k].to(torch.float32).reshape(1))
        row = row / row.sum().clamp_min(1e-30)
        row = torch.where(boost[k], row, p_rows[k])
        verify_rows.append(row)
        residual_rows.append(residual_distribution_nosync(row, q_rows[k]))
    verify = torch.stack(verify_rows)
    residual = torch.stack(residual_rows)
    verify_y = verify.gather(1, tokens[:, None])[:, 0]

    # Fixed randomness per round: uniforms, then exponential-race noise rows.
    uniforms = torch.rand(length, device=device, generator=generator)
    noise = torch.empty(length + count, vocabulary, device=device, dtype=torch.float32)
    noise.exponential_(generator=generator)
    corrections = (residual / noise[:length]).argmax(dim=-1)                       # [L]
    anchors = (target[1:count + 1] / noise[length:]).argmax(dim=-1)                 # [count]

    ratio = torch.clamp(verify_y.double() / torch.clamp_min(q.double(), 1e-30), max=1.0)
    accept = uniforms.double() <= ratio
    reject = ~accept
    first = torch.where(reject.any(), reject.to(torch.int64).argmax(), torch.full((), length, device=device,
                                                                                  dtype=torch.int64))
    at = torch.clamp(first, max=length - 1)
    correction = corrections[at]
    candidates = sibling_index[at]
    candidate_tokens = ids[candidates.clamp_min(0)]
    hits = (candidates >= 0) & (candidate_tokens == correction)
    hit_node = torch.where(hits.any(), candidates[hits.to(torch.int64).argmax()], torch.full_like(correction, -1))
    leaf_bonus = anchors[trunk_index[-1]]
    hit_anchor = anchors[hit_node.clamp_min(0)]
    packed = torch.cat((
        torch.stack((first, hit_node, correction, leaf_bonus, hit_anchor)).to(torch.float64),
        uniforms.double(), verify_y.double(), p.double(), q.double(), floor.double(), tokens.double(),
    ))
    host = packed.tolist()                                                           # the one host read
    STATS["rounds"] += 1
    first_h, hit_h, correction_h, bonus_h, anchor_h = (int(x) for x in host[:5])
    offset = 5
    u_h = host[offset:offset + length]
    vy_h = host[offset + length:offset + 2 * length]
    p_h = host[offset + 2 * length:offset + 3 * length]
    q_h = host[offset + 3 * length:offset + 4 * length]
    tv_h = host[offset + 4 * length:offset + 5 * length]
    tok_h = [int(x) for x in host[offset + 5 * length:offset + 6 * length]]

    statuses = {node.index: "NOT_VISITED" for node in topology.nodes}
    diagnostics: dict[int, dict[str, Any]] = {}
    accepted: list[int] = []
    for k in range(min(first_h + 1, length)):
        node = trunk[k]
        keep = k < first_h
        diagnostics[node] = {
            "node": node, "proposal_role": "sampled_cactus_primary", "trunk_position": k + 1,
            "target_probability": p_h[k], "draft_probability": q_h[k],
            "relaxation_tv": max(0.0, vy_h[k] - p_h[k]), "adaptive_relaxation_tv": tv_h[k],
            "verification_probability": vy_h[k], "verification_rule": "cactus",
            "accept_probability": min(1.0, vy_h[k] / max(q_h[k], 1e-30)), "accept_random": u_h[k],
            "normal_survival": keep, "trunk_accepted": keep, "trunk_rejected": not keep,
            "rescue_eligible": False, "rescued": False, "adaptive_relaxation_enabled": True,
            "device_verify": True,
        }
        statuses[node] = "SELECTED" if keep else "REJECT"
        if keep:
            accepted.append(node)
            for sibling in siblings[k]:
                if statuses[sibling] == "NOT_VISITED":
                    statuses[sibling] = "BACKUP_UNUSED"

    def result(output: tuple[int, ...], terminal: str, surviving: int, score: float | None) -> TreeVerifyResult:
        return TreeVerifyResult(
            output_token_ids=tuple(tok_h[:first_h]) + output,      # accepted trunk tokens, then the tail
            accepted_node_indices=tuple(accepted),
            selected_branch=topology.nodes[accepted[0]].branch if accepted else None,
            terminal=terminal,
            target_validation_nodes=count,
            evaluated_node_statuses=tuple(sorted(statuses.items())),
            node_diagnostics=tuple(diagnostics[index] for index in sorted(diagnostics)),
            surviving_paths=surviving,
            selected_path_score=score,
        )

    if first_h >= length:
        return result((bonus_h,), "sampled-primary-full-cactus-bonus", 1, 1.0)
    for sibling in siblings[first_h]:
        is_hit = sibling == hit_h
        statuses[sibling] = "CORRECTION_HIT" if is_hit else "PRUNE"
        diagnostics[sibling] = {
            "node": sibling, "proposal_role": "exact_residual_correction_candidate",
            "trunk_position": first_h + 1, "sampled_correction_token": correction_h,
            "correction_hit": is_hit, "rescue_eligible": is_hit, "rescued": is_hit,
        }
    if hit_h < 0:
        return result((correction_h,), "residual-correction-tree-miss", 0, None)
    accepted.append(hit_h)
    statuses[hit_h] = "SELECTED"
    diagnostics[hit_h].update(rescue_extension_tokens=1, rescue_unlocked_tokens=1,
                              rescue_decision_mode="exact-residual-token-hit")
    return result((correction_h, anchor_h), "residual-hit-target-anchor", 1, 1.0)
