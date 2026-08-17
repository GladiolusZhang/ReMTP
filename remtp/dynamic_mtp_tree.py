"""Dynamic MTP candidate trees and target-dominant relaxed verification.

The proposer uses only MTP probabilities, entropy, depth and a global node
budget.  Target probabilities are first consulted after the complete tree has
been drafted and evaluated by one TreeAttention forward.

The relaxed verifier in this module is intentionally *not* lossless
speculative sampling: it selects among target-supported draft paths rather
than reconstructing the exact target distribution.  Every round nevertheless
ends with an unmodified target correction/bonus token so that the next round
starts from a target-produced anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

from remtp.fixed6_microtree import (
    TreeNode,
    TreeVerifyResult,
    normalize_distribution,
    residual_distribution,
    sample_categorical,
)
from remtp.cactus_mtp import cactus_target_distribution


@dataclass(frozen=True)
class DynamicTreeConfig:
    max_depth: int = 3
    max_nodes: int = 6
    adaptive_base_nodes: int = 0
    adaptive_entropy_threshold: float = 1.0
    max_children_per_parent: int = 2
    min_sibling_ratio: float = 0.25
    min_draft_prob: float = 0.02
    threshold_sensitivity: float = 1.0
    depth_exploration: float = 0.5
    depth_reward: float = 0.25
    allocation_mode: str = "geometric"
    rank_penalty: float = 2.0
    coverage_weight: float = 0.5
    coverage_mode: str = "geometric"
    min_target_coverage: float = 0.0
    relax_threshold: float = 0.7
    support_mode: str = "relative"
    proposal_support_ratio: float = 1.0
    confirmation_min_relative: float = 0.05
    cactus_delta: float = 1.0
    cactus_target_weight: float = 0.25
    max_guided_rescues_per_path: int = 1
    rescue_score_threshold: float = 0.35
    rescue_depth_penalty: float = 0.0
    rescue_continuation_discount: float = 0.0
    rescue_continuation_min_depth: int = 2
    rescue_margin_reference: float = 0.0
    rescue_margin_penalty: float = 0.0
    length_reward: float = 0.5
    path_temperature: float = 0.3
    path_selection_mode: str = "longest"
    frontier_rescue: bool = False
    rescue_delta: float = 0.5
    rescue_min_relative: float = 0.1
    rescue_min_target_prob: float = 0.001
    append_target_anchor: bool = True
    eos_token_ids: tuple[int, ...] = ()
    eos_protection_threshold: float = 0.5

    def validate(self) -> None:
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if self.adaptive_base_nodes < 0:
            raise ValueError("adaptive_base_nodes must be non-negative")
        if self.adaptive_base_nodes > self.max_nodes:
            raise ValueError("adaptive_base_nodes cannot exceed max_nodes")
        if not 0.0 <= self.adaptive_entropy_threshold <= 1.0:
            raise ValueError(
                "adaptive_entropy_threshold must be in [0,1]"
            )
        if self.max_children_per_parent <= 0:
            raise ValueError("max_children_per_parent must be positive")
        if not 0.0 <= self.min_sibling_ratio <= 1.0:
            raise ValueError("min_sibling_ratio must be in [0,1]")
        if not 0.0 <= self.min_draft_prob < 1.0:
            raise ValueError("min_draft_prob must be in [0,1)")
        if self.threshold_sensitivity < 0.0:
            raise ValueError("threshold_sensitivity must be non-negative")
        if self.depth_exploration < 0.0:
            raise ValueError("depth_exploration must be non-negative")
        if self.allocation_mode not in {
            "geometric",
            "reach_first",
            "soft_reach",
            "residual_coverage",
        }:
            raise ValueError(
                "allocation_mode must be geometric, reach_first, soft_reach "
                "or residual_coverage"
            )
        if self.rank_penalty < 0.0:
            raise ValueError("rank_penalty must be non-negative")
        if not 0.0 <= self.coverage_weight <= 1.0:
            raise ValueError("coverage_weight must be in [0,1]")
        if self.coverage_mode not in {"geometric", "coverage_gate"}:
            raise ValueError(
                "coverage_mode must be geometric or coverage_gate"
            )
        if not 0.0 <= self.min_target_coverage <= 1.0:
            raise ValueError("min_target_coverage must be in [0,1]")
        if not 0.0 <= self.relax_threshold <= 1.0:
            raise ValueError("relax_threshold must be in [0,1]")
        if self.support_mode not in {
            "relative",
            "dual",
            "confirmed",
            "cactus",
            "cactus_guided",
            "cactus_trunk_rescue",
            "sampled_primary_reopen",
            "sampled_primary_shadow",
            "residual_hit_anchor",
            "residual_hit_strict",
            "residual_hit_cactus",
            "target_path_rescue",
            "prefix_reopen_rescue",
        }:
            raise ValueError(
                "support_mode must be relative, dual, confirmed, cactus or "
                "cactus_guided, cactus_trunk_rescue, sampled_primary_reopen, "
                "sampled_primary_shadow, residual_hit_anchor, "
                "residual_hit_strict, residual_hit_cactus, "
                "target_path_rescue, or prefix_reopen_rescue"
            )
        if self.proposal_support_ratio < 0.0:
            raise ValueError("proposal_support_ratio must be non-negative")
        if not 0.0 <= self.confirmation_min_relative <= 1.0:
            raise ValueError("confirmation_min_relative must be in [0,1]")
        if self.cactus_delta < 0.0:
            raise ValueError("cactus_delta must be non-negative")
        if not 0.0 <= self.cactus_target_weight <= 1.0:
            raise ValueError("cactus_target_weight must be in [0,1]")
        if self.max_guided_rescues_per_path < 0:
            raise ValueError(
                "max_guided_rescues_per_path must be non-negative"
            )
        if not 0.0 <= self.rescue_score_threshold <= 1.0:
            raise ValueError("rescue_score_threshold must be in [0,1]")
        if self.rescue_depth_penalty < 0.0:
            raise ValueError("rescue_depth_penalty must be non-negative")
        if not 0.0 <= self.rescue_continuation_discount < 1.0:
            raise ValueError(
                "rescue_continuation_discount must be in [0,1)"
            )
        if self.rescue_continuation_min_depth <= 0:
            raise ValueError(
                "rescue_continuation_min_depth must be positive"
            )
        if self.rescue_margin_reference < 0.0:
            raise ValueError("rescue_margin_reference must be non-negative")
        if self.rescue_margin_penalty < 0.0:
            raise ValueError("rescue_margin_penalty must be non-negative")
        if (
            self.rescue_margin_penalty > 0.0
            and self.rescue_margin_reference == 0.0
        ):
            raise ValueError(
                "rescue_margin_reference must be positive when margin penalty is enabled"
            )
        if self.length_reward < 0.0:
            raise ValueError("length_reward must be non-negative")
        if self.path_temperature < 0.0:
            raise ValueError("path_temperature must be non-negative")
        if self.path_selection_mode not in {"longest", "balanced"}:
            raise ValueError(
                "path_selection_mode must be longest or balanced"
            )
        if self.rescue_delta < 0.0:
            raise ValueError("rescue_delta must be non-negative")
        if not 0.0 <= self.rescue_min_relative <= 1.0:
            raise ValueError("rescue_min_relative must be in [0,1]")
        if not 0.0 <= self.rescue_min_target_prob <= 1.0:
            raise ValueError("rescue_min_target_prob must be in [0,1]")
        if any(token_id < 0 for token_id in self.eos_token_ids):
            raise ValueError("eos_token_ids must be non-negative")
        if len(set(self.eos_token_ids)) != len(self.eos_token_ids):
            raise ValueError("eos_token_ids must be unique")
        if not 0.0 <= self.eos_protection_threshold <= 1.0:
            raise ValueError("eos_protection_threshold must be in [0,1]")


@dataclass(frozen=True)
class CandidateSelection:
    token_ids: torch.Tensor
    entropy: float
    threshold: float
    q_max: float
    candidate_count_before_budget: int
    threshold_candidate_count: int
    sibling_added: bool


@dataclass(frozen=True)
class DynamicTreeTopology:
    name: str
    nodes: tuple[TreeNode, ...]

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[tuple[int, ...]],
        *,
        name: str = "dynamic-mtp",
    ) -> "DynamicTreeTopology":
        ordered = tuple(sorted((tuple(path) for path in paths), key=lambda p: (len(p), p)))
        if not ordered:
            raise ValueError("dynamic tree must contain at least one node")
        if len(set(ordered)) != len(ordered):
            raise ValueError("dynamic tree paths must be unique")
        path_to_index = {path: index for index, path in enumerate(ordered)}
        nodes: list[TreeNode] = []
        for index, path in enumerate(ordered):
            if not path:
                raise ValueError("the common anchor is not a draft node")
            parent = path_to_index.get(path[:-1]) if len(path) > 1 else None
            if len(path) > 1 and parent is None:
                raise ValueError(f"missing parent for path {path}")
            nodes.append(
                TreeNode(
                    index=index,
                    branch=path[0],
                    depth=len(path),
                    path=path,
                    parent=parent,
                )
            )
        result = cls(name=name, nodes=tuple(nodes))
        result.validate()
        return result

    def validate(self) -> None:
        if tuple(node.index for node in self.nodes) != tuple(range(len(self.nodes))):
            raise ValueError("dynamic node indices must be contiguous BFS indices")
        paths = tuple(node.path for node in self.nodes)
        if paths != tuple(sorted(paths, key=lambda p: (len(p), p))):
            raise ValueError("dynamic nodes must be in breadth-first order")
        for node in self.nodes:
            if node.depth == 1 and node.parent is not None:
                raise ValueError("root candidate cannot have a parent")
            if node.depth > 1:
                if node.parent is None:
                    raise ValueError("non-root candidate is missing its parent")
                if self.nodes[node.parent].path != node.path[:-1]:
                    raise ValueError("dynamic parent/path relation is invalid")

    @property
    def choices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(node.path for node in self.nodes)

    @property
    def roots(self) -> tuple[int, ...]:
        return self.children_of(None)

    @property
    def leaves(self) -> tuple[int, ...]:
        parents = {node.parent for node in self.nodes if node.parent is not None}
        return tuple(node.index for node in self.nodes if node.index not in parents)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def max_depth(self) -> int:
        return max(node.depth for node in self.nodes)

    @property
    def node_position_offsets(self) -> tuple[int, ...]:
        return tuple(node.depth for node in self.nodes)

    @property
    def parent_logit_rows(self) -> tuple[int, ...]:
        return tuple(0 if node.parent is None else node.parent + 1 for node in self.nodes)

    def level_nodes(self, depth: int) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes if node.depth == depth)

    def children_of(self, parent: int | None) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes if node.parent == parent)

    def path_to(self, node_index: int) -> tuple[int, ...]:
        indices: list[int] = []
        current: int | None = int(node_index)
        while current is not None:
            indices.append(current)
            current = self.nodes[current].parent
        return tuple(reversed(indices))

    def causal_tree_mask(self, *, include_anchor: bool = True) -> torch.Tensor:
        offset = 1 if include_anchor else 0
        mask = torch.zeros((self.node_count + offset, self.node_count + offset), dtype=torch.bool)
        if include_anchor:
            mask[:, 0] = True
        for node in self.nodes:
            row = node.index + offset
            mask[row, row] = True
            parent = node.parent
            while parent is not None:
                mask[row, parent + offset] = True
                parent = self.nodes[parent].parent
        return mask


def normalized_entropy(probs: torch.Tensor) -> float:
    q = normalize_distribution(probs)
    if q.numel() <= 1:
        return 0.0
    entropy = -(q * q.clamp_min(1e-30).log()).sum()
    return float(entropy / math.log(q.numel()))


def select_dynamic_candidates(
    probs: torch.Tensor,
    *,
    depth: int,
    config: DynamicTreeConfig,
    budget: int | None = None,
) -> CandidateSelection:
    """Select an uncertainty-aware set with guarded top-k backups.

    The entropy threshold remains the primary rule.  To avoid collapsing an
    otherwise useful tree into a chain, candidates up to the configured child
    cap may additionally enter when each has both meaningful absolute mass
    and at least ``min_sibling_ratio`` of top-1. A hard per-parent cap and the
    global node budget prevent this guarded widening from becoming an
    unrestricted tree. The budget is never filled with candidates that fail
    both probability guards.
    """
    config.validate()
    if depth <= 0 or depth > config.max_depth:
        raise ValueError("candidate depth is outside the configured tree")
    q = normalize_distribution(probs)
    entropy = normalized_entropy(q)
    depth_ratio = depth / config.max_depth
    uncertainty = entropy + config.depth_exploration * depth_ratio
    q_max = float(q.max())
    threshold = max(
        config.min_draft_prob,
        q_max * math.exp(-config.threshold_sensitivity * uncertainty),
    )
    eligible = torch.nonzero(q >= threshold, as_tuple=False).flatten()
    threshold_count = int(eligible.numel())
    if threshold_count == 0:
        return CandidateSelection(
            token_ids=eligible,
            entropy=entropy,
            threshold=threshold,
            q_max=q_max,
            candidate_count_before_budget=0,
            threshold_candidate_count=0,
            sibling_added=False,
        )

    global_order = torch.argsort(q, descending=True, stable=True)
    sibling_added = False
    guarded_limit = min(config.max_children_per_parent, q.numel())
    for guarded in global_order[1:guarded_limit]:
        guarded_prob = float(q[guarded])
        guarded_is_eligible = bool((eligible == guarded).any())
        if (
            not guarded_is_eligible
            and guarded_prob >= config.min_draft_prob
            and guarded_prob >= q_max * config.min_sibling_ratio
        ):
            eligible = torch.cat((eligible, guarded.reshape(1)))
            sibling_added = True

    order = torch.argsort(q[eligible], descending=True, stable=True)
    eligible = eligible[order]
    before_budget = int(eligible.numel())
    eligible = eligible[: config.max_children_per_parent]
    if budget is not None:
        eligible = eligible[: max(int(budget), 0)]
    return CandidateSelection(
        token_ids=eligible,
        entropy=entropy,
        threshold=threshold,
        q_max=q_max,
        candidate_count_before_budget=before_budget,
        threshold_candidate_count=threshold_count,
        sibling_added=sibling_added,
    )


def path_expansion_value(
    path_log_q_sum: float,
    depth: int,
    config: DynamicTreeConfig,
) -> float:
    if config.allocation_mode in {
        "reach_first",
        "soft_reach",
        "residual_coverage",
    }:
        # MAL is earned only when every earlier token on a path survives.
        # The cumulative proposal mass is therefore a better reach proxy than
        # the geometric mean, which made an early low-Q backup look comparable
        # to the primary path again after several confident descendants.
        proposal_reach = math.exp(path_log_q_sum)
    else:
        proposal_reach = math.exp(path_log_q_sum / max(depth, 1))
    return proposal_reach * math.exp(
        config.depth_reward * depth / config.max_depth
    )


def level_expansion_order(
    proposals: Sequence[tuple[float, tuple[int, ...]]],
    *,
    config: DynamicTreeConfig,
) -> tuple[int, ...]:
    """Order one level's candidates under the global node budget.

    ``reach_first`` implements progressive widening. Every parent's top-1
    continuation competes before rank-1 backups, which in turn compete before
    rank-2 backups. Within a rank tier, cumulative path probability decides.
    This prevents a confident descendant of a low-reach backup from consuming
    the only continuation slot of a much more reachable parent prefix.
    """
    config.validate()
    indices = range(len(proposals))
    if config.allocation_mode == "reach_first":
        return tuple(
            sorted(
                indices,
                key=lambda index: (
                    proposals[index][1][-1],
                    -proposals[index][0],
                    proposals[index][1],
                ),
            )
        )
    if config.allocation_mode == "residual_coverage":
        # The exact residual can be reused only if the chosen correction is a
        # sibling of the rejected primary token. Spend the first slots on all
        # meaningful siblings of the sampled primary parent, then preserve a
        # sampled-Q continuation for already retained recovery branches.
        return tuple(
            sorted(
                indices,
                key=lambda index: (
                    0
                    if proposals[index][1][:-1]
                    == (0,) * (len(proposals[index][1]) - 1)
                    else (1 if proposals[index][1][-1] == 0 else 2),
                    proposals[index][1][-1]
                    if proposals[index][1][:-1]
                    == (0,) * (len(proposals[index][1]) - 1)
                    else 0,
                    -proposals[index][0],
                    proposals[index][1],
                ),
            )
        )
    if config.allocation_mode == "soft_reach":
        # Rank is empirical reliability information, not a hard quota. A
        # sufficiently high-reach rank-1/2 candidate may still outrank a weak
        # rank-0 continuation. No layer or parent is guaranteed a node.
        return tuple(
            sorted(
                indices,
                key=lambda index: (
                    -proposals[index][0]
                    / (
                        1.0
                        + config.rank_penalty
                        * proposals[index][1][-1]
                    ),
                    proposals[index][1],
                ),
            )
        )
    return tuple(
        sorted(
            indices,
            key=lambda index: (-proposals[index][0], proposals[index][1]),
        )
    )


def level_expansion_budget(
    *,
    current_nodes: int,
    next_depth: int,
    config: DynamicTreeConfig,
) -> int:
    """Return this level's budget while reserving one future chain slot.

    Breadth must never consume every node before the primary path reaches the
    configured maximum depth. For every level after ``next_depth`` we reserve
    one node, which is enough to continue the highest-value surviving draft
    path without permitting uncontrolled width.
    """
    if current_nodes < 0:
        raise ValueError("current_nodes must be non-negative")
    if next_depth <= 0 or next_depth > config.max_depth:
        raise ValueError("next_depth is outside the configured tree")
    remaining = config.max_nodes - current_nodes
    future_chain_reserve = config.max_depth - next_depth
    return max(remaining - future_chain_reserve, 0)


def adaptive_node_limit(
    entropies: Sequence[float], config: DynamicTreeConfig
) -> int:
    """Choose the level node cap using only current MTP uncertainty."""
    config.validate()
    if config.adaptive_base_nodes == 0:
        return config.max_nodes
    if any(
        float(entropy) >= config.adaptive_entropy_threshold
        for entropy in entropies
    ):
        return config.max_nodes
    return config.adaptive_base_nodes


def mtp_layer_for_tree_depth(depth: int, route: str) -> int:
    """Map the current parent depth to the module that expands it.

    The root proposal has already been produced by module 0 before this
    helper is called. Therefore expanding depth-1 roots uses module 1 in the
    physical 0-1-2 route, while repeat-0 always returns module 0.
    """
    if depth <= 0:
        raise ValueError("tree depth must be positive")
    normalized = route.strip().lower()
    if normalized == "physical_012":
        return min(depth, 2)
    if normalized == "repeat_000":
        return 0
    raise ValueError(
        "unknown MTP tree route; choose physical_012 or repeat_000"
    )


def _sample_path_index(
    log_scores: torch.Tensor,
    temperature: float,
    generator: torch.Generator | None,
) -> int:
    if temperature <= 1e-8:
        return int(log_scores.argmax().item())
    probs = torch.softmax(log_scores / temperature, dim=0)
    return int(torch.multinomial(probs, 1, generator=generator).item())


def cactus_trunk_rescue_verify(
    topology: DynamicTreeTopology,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    config: DynamicTreeConfig,
) -> TreeVerifyResult:
    """Verify an exact sampled-Q Cactus trunk, then rescue one rejection.

    Paths ``(0,)``, ``(0, 0)``, ... form the sampled trunk.  Its verification
    is exactly the chain Cactus rule: candidate-specific ``h``, acceptance
    ``min(1, h(y) / q(y))``, and correction distribution ``(h-q)+``.

    Only the first rejected trunk position consults its already-verified tree
    siblings.  A sibling is scored against the Cactus correction residual. In
    ``cactus_trunk_rescue`` it replaces the correction by one draft token. In
    ``sampled_primary_reopen`` its already-verified descendants may continue
    under the ordinary target-relative rule.  Both tree recoveries are
    deliberately approximate: sibling tokens were selected deterministically
    from Q rather than sampled from a joint multi-proposal law.  If recovery
    does not fire, the round falls back to the same Cactus residual
    distribution as the chain baseline.
    """
    config.validate()
    if not config.append_target_anchor:
        raise ValueError("cactus_trunk_rescue requires a target anchor")
    count = topology.node_count
    if draft_token_ids.shape != (count,):
        raise ValueError(f"draft_token_ids must have shape [{count}]")
    if draft_probs.ndim != 2 or draft_probs.shape[0] != count:
        raise ValueError(f"draft_probs must have shape [{count}, vocab]")
    if target_probs_by_input.shape != (count + 1, draft_probs.shape[1]):
        raise ValueError(
            f"target_probs_by_input must have shape [{count + 1}, vocab]"
        )

    device = target_probs_by_input.device
    ids = draft_token_ids.to(device=device, dtype=torch.int64)
    draft = torch.stack(
        [normalize_distribution(row) for row in draft_probs.to(device=device)]
    )
    target = torch.stack(
        [
            normalize_distribution(row)
            for row in target_probs_by_input.to(device=device)
        ]
    )
    path_to_index = {node.path: node.index for node in topology.nodes}
    trunk: list[int] = []
    for depth in range(1, topology.max_depth + 1):
        path = (0,) * depth
        if path not in path_to_index:
            raise ValueError(
                "cactus_trunk_rescue topology is missing sampled trunk "
                f"node {path}"
            )
        trunk.append(path_to_index[path])

    eos_ids = tuple(int(token_id) for token_id in config.eos_token_ids)
    if any(token_id >= target.shape[-1] for token_id in eos_ids):
        raise ValueError("an eos_token_id is outside the target vocabulary")
    eos_index = (
        torch.tensor(eos_ids, dtype=torch.int64, device=device)
        if eos_ids
        else None
    )
    statuses: dict[int, str] = {
        node.index: "NOT_VISITED" for node in topology.nodes
    }
    diagnostics: dict[int, dict[str, Any]] = {}
    accepted: list[int] = []

    for position, trunk_node in enumerate(trunk, start=1):
        node = topology.nodes[trunk_node]
        parent_row = 0 if node.parent is None else node.parent + 1
        p_row = target[parent_row]
        q_row = draft[trunk_node]
        token_id = int(ids[trunk_node])
        h_row = cactus_target_distribution(
            p_row.unsqueeze(0),
            ids[trunk_node : trunk_node + 1],
            config.cactus_delta,
        )[0]
        p_y = float(p_row[token_id])
        q_y = float(q_row[token_id])
        h_y = float(h_row[token_id])
        accept_probability = min(1.0, h_y / max(q_y, 1e-30))
        random_value = float(
            torch.rand((), device=device, generator=generator).item()
        )
        accepted_here = random_value <= accept_probability
        statuses[trunk_node] = "SELECTED" if accepted_here else "REJECT"
        diagnostics[trunk_node] = {
            "node": trunk_node,
            "proposal_role": "sampled_cactus_trunk",
            "trunk_position": position,
            "target_probability": p_y,
            "draft_probability": q_y,
            "cactus_boosted_probability": h_y,
            "cactus_acceptance_probability": accept_probability,
            "cactus_random": random_value,
            "trunk_accepted": accepted_here,
            "trunk_rejected": not accepted_here,
            "rescue_eligible": False,
            "rescued": False,
        }
        if accepted_here:
            accepted.append(trunk_node)
            # Backups at an accepted position are intentionally unused.  The
            # tree is a rejection fallback, not a path selector competing
            # with the sampled Cactus proposal.
            for sibling in topology.children_of(node.parent):
                if sibling != trunk_node and statuses[sibling] == "NOT_VISITED":
                    statuses[sibling] = "BACKUP_UNUSED"
            continue

        correction_probs = residual_distribution(h_row, q_row)
        correction_max = float(correction_probs.max())
        siblings = tuple(
            sibling
            for sibling in topology.children_of(node.parent)
            if sibling != trunk_node
        )
        rescue_rows: list[tuple[float, int]] = []
        for sibling in siblings:
            side_token = int(ids[sibling])
            side_q = draft[sibling]
            residual_p = float(correction_probs[side_token])
            residual_relative = residual_p / max(correction_max, 1e-30)
            original_p = float(p_row[side_token])
            side_q_y = float(side_q[side_token])
            eos_veto = False
            if eos_index is not None:
                target_top1 = int(p_row.argmax())
                target_top1_is_eos = bool((eos_index == target_top1).any())
                side_is_eos = bool((eos_index == side_token).any())
                eos_mass = float(p_row.index_select(0, eos_index).sum())
                eos_veto = (
                    target_top1_is_eos
                    and eos_mass >= config.eos_protection_threshold
                    and not side_is_eos
                )
            eligible = (
                residual_p >= config.rescue_min_target_prob
                and residual_relative >= config.rescue_min_relative
                and not eos_veto
            )
            rescue_h = cactus_target_distribution(
                correction_probs.unsqueeze(0),
                ids[sibling : sibling + 1],
                config.rescue_delta,
            )[0]
            rescue_h_y = float(rescue_h[side_token])
            rescue_accept = min(
                1.0, rescue_h_y / max(side_q_y, 1e-30)
            )
            score = (
                max(residual_relative, 1e-30)
                ** config.cactus_target_weight
                * max(rescue_accept, 1e-30)
                ** (1.0 - config.cactus_target_weight)
            )
            diagnostics[sibling] = {
                "node": sibling,
                "proposal_role": "rejection_rescue_side",
                "trunk_position": position,
                "target_probability": original_p,
                "draft_probability": side_q_y,
                "residual_target_probability": residual_p,
                "residual_relative_support": residual_relative,
                "rescue_cactus_probability": rescue_h_y,
                "rescue_accept_probability": rescue_accept,
                "rescue_score": score,
                "eos_protected": eos_veto,
                "rescue_eligible": eligible,
                "rescue_random": None,
                "rescued": False,
            }
            statuses[sibling] = "RESCUE_CANDIDATE" if eligible else "PRUNE"
            if eligible:
                rescue_rows.append((score, sibling))

        if rescue_rows:
            _, rescue_node = max(
                rescue_rows,
                key=lambda item: (
                    item[0],
                    diagnostics[item[1]]["residual_target_probability"],
                    -item[1],
                ),
            )
            rescue_random = float(
                torch.rand((), device=device, generator=generator).item()
            )
            rescue_row = diagnostics[rescue_node]
            rescue_row["rescue_random"] = rescue_random
            if rescue_random <= float(rescue_row["rescue_accept_probability"]):
                selected_path = topology.path_to(rescue_node)
                expected_prefix = tuple(accepted) + (rescue_node,)
                if selected_path != expected_prefix:
                    raise RuntimeError(
                        "rescue path does not extend the accepted Cactus trunk"
                    )
                rescue_row["rescued"] = True
                statuses[rescue_node] = "SELECTED"
                for sibling in siblings:
                    if sibling != rescue_node and statuses[sibling] != "PRUNE":
                        statuses[sibling] = "PRUNE"

                # The historical caterpillar stops after one rescued token.
                # The sampled-primary mode instead gives the recovered branch
                # access to its already verified subtree.  No second Cactus
                # rescue is allowed: descendants must pass the inexpensive,
                # target-anchored relative-support rule on their own.
                path_score = float(rescue_row["rescue_score"])
                if config.support_mode == "sampled_primary_reopen":
                    reachable = {rescue_node}
                    confidence_by_node = {
                        rescue_node: max(path_score, 1e-30)
                    }
                    for next_depth in range(position + 1, topology.max_depth + 1):
                        next_reachable: set[int] = set()
                        for parent in sorted(reachable):
                            for child in topology.children_of(parent):
                                child_token = int(ids[child])
                                child_p = target[parent + 1]
                                child_p_y = float(child_p[child_token])
                                child_p_max = max(float(child_p.max()), 1e-30)
                                child_relative = child_p_y / child_p_max
                                eos_veto = False
                                if eos_index is not None:
                                    target_top1 = int(child_p.argmax())
                                    target_top1_is_eos = bool(
                                        (eos_index == target_top1).any()
                                    )
                                    child_is_eos = bool(
                                        (eos_index == child_token).any()
                                    )
                                    eos_mass = float(
                                        child_p.index_select(0, eos_index).sum()
                                    )
                                    eos_veto = (
                                        target_top1_is_eos
                                        and eos_mass
                                        >= config.eos_protection_threshold
                                        and not child_is_eos
                                    )
                                keep = (
                                    child_relative >= config.relax_threshold
                                    and not eos_veto
                                )
                                statuses[child] = (
                                    "SURVIVE" if keep else "PRUNE"
                                )
                                diagnostics[child] = {
                                    "node": child,
                                    "proposal_role": "post_rescue_ordinary_continuation",
                                    "trunk_position": next_depth,
                                    "target_probability": child_p_y,
                                    "relative_target_support": child_relative,
                                    "ordinary_threshold": config.relax_threshold,
                                    "normal_survival": keep,
                                    "eos_protected": eos_veto,
                                    "rescue_eligible": False,
                                    "rescued": False,
                                }
                                if keep:
                                    next_reachable.add(child)
                                    confidence_by_node[child] = (
                                        confidence_by_node[parent]
                                        * max(child_relative, 1e-30)
                                    )
                        if not next_reachable:
                            break
                        reachable = next_reachable

                    candidate_leaves = tuple(sorted(reachable))
                    best_leaf = max(
                        candidate_leaves,
                        key=lambda leaf: (
                            topology.nodes[leaf].depth,
                            confidence_by_node[leaf]
                            ** (1.0 / max(topology.nodes[leaf].depth - position + 1, 1)),
                            -leaf,
                        ),
                    )
                    selected_path = topology.path_to(best_leaf)
                    path_score = confidence_by_node[best_leaf] ** (
                        1.0 / max(len(selected_path) - len(accepted), 1)
                    )
                    for node_index in selected_path[len(accepted) + 1 :]:
                        statuses[node_index] = "SELECTED"
                    rescue_row["rescue_extension_tokens"] = 1
                    rescue_row["rescue_unlocked_tokens"] = (
                        len(selected_path) - len(accepted)
                    )

                leaf = selected_path[-1]
                bonus = sample_categorical(target[leaf + 1], generator)
                return TreeVerifyResult(
                    output_token_ids=tuple(int(ids[index]) for index in selected_path)
                    + (bonus,),
                    accepted_node_indices=selected_path,
                    selected_branch=topology.nodes[selected_path[0]].branch,
                    terminal=(
                        "sampled-primary-reopen-bonus"
                        if config.support_mode == "sampled_primary_reopen"
                        else "cactus-trunk-rescue-bonus"
                    ),
                    target_validation_nodes=count,
                    evaluated_node_statuses=tuple(sorted(statuses.items())),
                    node_diagnostics=tuple(
                        diagnostics[index] for index in sorted(diagnostics)
                    ),
                    surviving_paths=1,
                    selected_path_score=path_score,
                )

        correction = sample_categorical(correction_probs, generator)
        return TreeVerifyResult(
            output_token_ids=tuple(int(ids[index]) for index in accepted)
            + (correction,),
            accepted_node_indices=tuple(accepted),
            selected_branch=(
                topology.nodes[accepted[0]].branch if accepted else None
            ),
            terminal="cactus-trunk-correction",
            target_validation_nodes=count,
            evaluated_node_statuses=tuple(sorted(statuses.items())),
            node_diagnostics=tuple(
                diagnostics[index] for index in sorted(diagnostics)
            ),
            surviving_paths=0,
            selected_path_score=None,
        )

    leaf = accepted[-1]
    bonus = sample_categorical(target[leaf + 1], generator)
    return TreeVerifyResult(
        output_token_ids=tuple(int(ids[index]) for index in accepted) + (bonus,),
        accepted_node_indices=tuple(accepted),
        selected_branch=topology.nodes[accepted[0]].branch,
        terminal="cactus-trunk-bonus",
        target_validation_nodes=count,
        evaluated_node_statuses=tuple(sorted(statuses.items())),
        node_diagnostics=tuple(
            diagnostics[index] for index in sorted(diagnostics)
        ),
        surviving_paths=1,
        selected_path_score=1.0,
    )


def residual_hit_tree_verify(
    topology: DynamicTreeTopology,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    config: DynamicTreeConfig,
) -> TreeVerifyResult:
    """Keep a sampled Cactus trunk and let its exact correction hit the tree.

    The first rejected primary token always samples the ordinary Cactus
    correction from ``(h-q)+`` *before* consulting any backup.  ``shadow``
    stops there.  The other modes reuse an already verified sibling only when
    its token ID exactly equals that sampled correction:

    * ``residual_hit_anchor`` appends one unmodified target token;
    * ``residual_hit_strict`` verifies sampled-Q descendants with ``p/q``;
    * ``residual_hit_cactus`` verifies them with the same Cactus rule.

    Anchor and strict modes preserve the Cactus-primary output distribution
    provided that every local-rank-0 descendant was independently sampled
    from its recorded Q.  The Cactus continuation deliberately follows the
    relaxed Cactus distribution at subsequent positions.
    """
    config.validate()
    modes = {
        "sampled_primary_shadow",
        "residual_hit_anchor",
        "residual_hit_strict",
        "residual_hit_cactus",
    }
    if config.support_mode not in modes:
        raise ValueError("residual_hit_tree_verify received an incompatible mode")
    if not config.append_target_anchor:
        raise ValueError("residual-hit verification requires a target anchor")
    count = topology.node_count
    if draft_token_ids.shape != (count,):
        raise ValueError(f"draft_token_ids must have shape [{count}]")
    if draft_probs.ndim != 2 or draft_probs.shape[0] != count:
        raise ValueError(f"draft_probs must have shape [{count}, vocab]")
    if target_probs_by_input.shape != (count + 1, draft_probs.shape[1]):
        raise ValueError(
            f"target_probs_by_input must have shape [{count + 1}, vocab]"
        )

    device = target_probs_by_input.device
    ids = draft_token_ids.to(device=device, dtype=torch.int64)
    draft = torch.stack(
        [normalize_distribution(row) for row in draft_probs.to(device=device)]
    )
    target = torch.stack(
        [normalize_distribution(row) for row in target_probs_by_input.to(device=device)]
    )
    path_to_index = {node.path: node.index for node in topology.nodes}
    trunk = []
    for depth in range(1, topology.max_depth + 1):
        path = (0,) * depth
        if path not in path_to_index:
            raise ValueError(
                "residual-hit topology is missing sampled primary node "
                f"{path}"
            )
        trunk.append(path_to_index[path])

    statuses = {node.index: "NOT_VISITED" for node in topology.nodes}
    diagnostics: dict[int, dict[str, Any]] = {}
    accepted: list[int] = []

    def verify_q_candidate(
        node_index: int,
        *,
        use_cactus: bool,
    ) -> tuple[bool, torch.Tensor, dict[str, Any]]:
        node = topology.nodes[node_index]
        parent_row = 0 if node.parent is None else node.parent + 1
        p_row = target[parent_row]
        q_row = draft[node_index]
        token_id = int(ids[node_index])
        verify_row = (
            cactus_target_distribution(
                p_row.unsqueeze(0),
                ids[node_index : node_index + 1],
                config.cactus_delta,
            )[0]
            if use_cactus
            else p_row
        )
        p_y = float(p_row[token_id])
        q_y = float(q_row[token_id])
        verify_y = float(verify_row[token_id])
        accept_probability = min(1.0, verify_y / max(q_y, 1e-30))
        random_value = float(
            torch.rand((), device=device, generator=generator).item()
        )
        keep = random_value <= accept_probability
        row = {
            "node": node_index,
            "proposal_role": (
                "sampled_primary" if node.path == (0,) * node.depth
                else "sampled_recovery_continuation"
            ),
            "trunk_position": node.depth,
            "target_probability": p_y,
            "draft_probability": q_y,
            "verification_probability": verify_y,
            "verification_rule": "cactus" if use_cactus else "strict",
            "accept_probability": accept_probability,
            "accept_random": random_value,
            "normal_survival": keep,
            "rescue_eligible": False,
            "rescued": False,
        }
        return keep, residual_distribution(verify_row, q_row), row

    for position, trunk_node in enumerate(trunk, start=1):
        keep, correction_probs, row = verify_q_candidate(
            trunk_node,
            use_cactus=True,
        )
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

        # This draw is exactly the correction Chain Cactus would make.  Tree
        # coverage is consulted only after the token has already been chosen.
        correction = sample_categorical(correction_probs, generator)
        siblings = tuple(
            sibling
            for sibling in topology.children_of(topology.nodes[trunk_node].parent)
            if sibling != trunk_node
        )
        hit_node = next(
            (sibling for sibling in siblings if int(ids[sibling]) == correction),
            None,
        )
        for sibling in siblings:
            token_id = int(ids[sibling])
            is_hit = sibling == hit_node
            is_reused = (
                is_hit and config.support_mode != "sampled_primary_shadow"
            )
            statuses[sibling] = "CORRECTION_HIT" if is_hit else "PRUNE"
            diagnostics[sibling] = {
                "node": sibling,
                "proposal_role": "exact_residual_correction_candidate",
                "trunk_position": position,
                "target_probability": float(
                    target[0 if topology.nodes[sibling].parent is None else topology.nodes[sibling].parent + 1][token_id]
                ),
                "draft_probability": float(draft[sibling][token_id]),
                "residual_target_probability": float(correction_probs[token_id]),
                "sampled_correction_token": correction,
                "correction_hit": is_hit,
                "rescue_eligible": is_reused,
                "rescued": is_reused,
            }

        if config.support_mode == "sampled_primary_shadow" or hit_node is None:
            return TreeVerifyResult(
                output_token_ids=tuple(int(ids[index]) for index in accepted)
                + (correction,),
                accepted_node_indices=tuple(accepted),
                selected_branch=(
                    topology.nodes[accepted[0]].branch if accepted else None
                ),
                terminal=(
                    "sampled-primary-shadow-correction"
                    if config.support_mode == "sampled_primary_shadow"
                    else "residual-correction-tree-miss"
                ),
                target_validation_nodes=count,
                evaluated_node_statuses=tuple(sorted(statuses.items())),
                node_diagnostics=tuple(
                    diagnostics[index] for index in sorted(diagnostics)
                ),
                surviving_paths=0,
                selected_path_score=None,
            )

        accepted.append(hit_node)
        statuses[hit_node] = "SELECTED"
        hit_row = diagnostics[hit_node]
        hit_row["rescue_extension_tokens"] = 1
        hit_row["rescue_unlocked_tokens"] = 1
        hit_row["rescue_decision_mode"] = "exact-residual-token-hit"

        if config.support_mode == "residual_hit_anchor":
            bonus = sample_categorical(target[hit_node + 1], generator)
            return TreeVerifyResult(
                output_token_ids=tuple(int(ids[index]) for index in accepted)
                + (bonus,),
                accepted_node_indices=tuple(accepted),
                selected_branch=topology.nodes[accepted[0]].branch,
                terminal="residual-hit-target-anchor",
                target_validation_nodes=count,
                evaluated_node_statuses=tuple(sorted(statuses.items())),
                node_diagnostics=tuple(
                    diagnostics[index] for index in sorted(diagnostics)
                ),
                surviving_paths=1,
                selected_path_score=1.0,
            )

        current = hit_node
        continuation_is_cactus = config.support_mode == "residual_hit_cactus"
        while topology.nodes[current].depth < topology.max_depth:
            sampled_path = topology.nodes[current].path + (0,)
            child = path_to_index.get(sampled_path)
            if child is None:
                break
            keep_child, child_correction_probs, child_row = verify_q_candidate(
                child,
                use_cactus=continuation_is_cactus,
            )
            diagnostics[child] = child_row
            statuses[child] = "SELECTED" if keep_child else "REJECT"
            if not keep_child:
                child_correction = sample_categorical(
                    child_correction_probs,
                    generator,
                )
                hit_row["rescue_unlocked_tokens"] = len(accepted) - position + 1
                return TreeVerifyResult(
                    output_token_ids=tuple(int(ids[index]) for index in accepted)
                    + (child_correction,),
                    accepted_node_indices=tuple(accepted),
                    selected_branch=topology.nodes[accepted[0]].branch,
                    terminal=(
                        "residual-hit-cactus-continuation-correction"
                        if continuation_is_cactus
                        else "residual-hit-strict-continuation-correction"
                    ),
                    target_validation_nodes=count,
                    evaluated_node_statuses=tuple(sorted(statuses.items())),
                    node_diagnostics=tuple(
                        diagnostics[index] for index in sorted(diagnostics)
                    ),
                    surviving_paths=1,
                    selected_path_score=1.0,
                )
            accepted.append(child)
            current = child

        hit_row["rescue_unlocked_tokens"] = len(accepted) - position + 1
        bonus = sample_categorical(target[current + 1], generator)
        return TreeVerifyResult(
            output_token_ids=tuple(int(ids[index]) for index in accepted) + (bonus,),
            accepted_node_indices=tuple(accepted),
            selected_branch=topology.nodes[accepted[0]].branch,
            terminal=(
                "residual-hit-cactus-continuation-bonus"
                if continuation_is_cactus
                else "residual-hit-strict-continuation-bonus"
            ),
            target_validation_nodes=count,
            evaluated_node_statuses=tuple(sorted(statuses.items())),
            node_diagnostics=tuple(
                diagnostics[index] for index in sorted(diagnostics)
            ),
            surviving_paths=1,
            selected_path_score=1.0,
        )

    leaf = accepted[-1]
    bonus = sample_categorical(target[leaf + 1], generator)
    return TreeVerifyResult(
        output_token_ids=tuple(int(ids[index]) for index in accepted) + (bonus,),
        accepted_node_indices=tuple(accepted),
        selected_branch=topology.nodes[accepted[0]].branch,
        terminal="sampled-primary-full-cactus-bonus",
        target_validation_nodes=count,
        evaluated_node_statuses=tuple(sorted(statuses.items())),
        node_diagnostics=tuple(
            diagnostics[index] for index in sorted(diagnostics)
        ),
        surviving_paths=1,
        selected_path_score=1.0,
    )


def target_dominant_relaxed_verify(
    topology: DynamicTreeTopology,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    config: DynamicTreeConfig,
) -> TreeVerifyResult:
    """Keep target-supported paths and optionally rescue one failed frontier.

    Normal survival remains target-dominant. ``relative`` uses only local
    target support, ``dual`` also admits proposal-underconfident candidates,
    and ``confirmed`` requires either that signal or an exact target-top1
    continuation before entering the wider support band. ``cactus`` is the
    direct chain-Cactus-strength ablation. ``cactus_guided`` keeps the normal
    target-relative rule and consults Cactus only at a dead frontier, where it
    may rescue one tree-supported candidate. ``target_path_rescue`` instead
    performs two explicit stages: ordinary relaxed verification first selects
    the longest surviving path, then a threshold-qualified failed child on
    that selected frontier may extend it by exactly one token. Rescue never
    re-enters whole-tree path competition and cannot unlock another token.
    ``prefix_reopen_rescue`` instead lets every currently reachable path spend
    at most one front-loaded rescue. Every threshold-qualified failed child
    survives, and its already verified descendants may continue under the
    ordinary target-relative rule; no descendant may spend a second rescue.
    The optional legacy frontier rescue follows the dead-frontier restriction.

    This verifier is approximate even without rescue because it selects a path
    from a proposal tree instead of reconstructing the exact target sequence
    distribution. The rescue rule controls risk; it is not lossless sampling.
    """
    config.validate()
    count = topology.node_count
    if draft_token_ids.shape != (count,):
        raise ValueError(f"draft_token_ids must have shape [{count}]")
    if draft_probs.ndim != 2 or draft_probs.shape[0] != count:
        raise ValueError(f"draft_probs must have shape [{count}, vocab]")
    if target_probs_by_input.shape != (count + 1, draft_probs.shape[1]):
        raise ValueError(f"target_probs_by_input must have shape [{count + 1}, vocab]")

    ids = draft_token_ids.to(device=target_probs_by_input.device, dtype=torch.int64)
    draft = draft_probs.to(
        device=target_probs_by_input.device, dtype=torch.float32
    ).clamp_min(0.0)
    draft_totals = draft.sum(dim=-1, keepdim=True)
    valid_draft_rows = torch.isfinite(draft_totals) & (draft_totals > 1e-30)
    draft = torch.where(
        valid_draft_rows,
        draft / draft_totals.clamp_min(1e-30),
        torch.full_like(draft, 1.0 / draft.shape[-1]),
    )
    target = target_probs_by_input.to(torch.float32).clamp_min(0.0)
    row_totals = target.sum(dim=-1, keepdim=True)
    valid_rows = torch.isfinite(row_totals) & (row_totals > 1e-30)
    target = torch.where(
        valid_rows,
        target / row_totals.clamp_min(1e-30),
        torch.full_like(target, 1.0 / target.shape[-1]),
    )
    eos_ids = tuple(int(token_id) for token_id in config.eos_token_ids)
    if any(token_id >= target.shape[-1] for token_id in eos_ids):
        raise ValueError("an eos_token_id is outside the target vocabulary")
    survived: set[int] = set()
    statuses: dict[int, str] = {node.index: "NOT_VISITED" for node in topology.nodes}
    diagnostic_rows: list[dict[str, Any]] = []
    s_by_node: dict[int, float] = {}
    rescue_count_by_node: dict[int, int] = {}
    rescued_nodes: set[int] = set()
    target_path_mode = config.support_mode == "target_path_rescue"
    prefix_reopen_mode = config.support_mode == "prefix_reopen_rescue"

    # Compute every local target signal in one GPU batch and cross the
    # device/host boundary once.  The subsequent prefix scan is deliberately
    # kept on CPU because parent survival is a tiny, irregular tree operation.
    # This avoids one synchronizing ``float(cuda_tensor)`` per tree node.
    parent_rows = torch.tensor(
        topology.parent_logit_rows,
        dtype=torch.int64,
        device=target.device,
    )
    child_target_prob = target[parent_rows, ids]
    child_draft_prob = draft[torch.arange(count, device=draft.device), ids]
    coverage_by_row = torch.zeros(
        count + 1,
        dtype=torch.float32,
        device=target.device,
    )
    coverage_by_row.scatter_add_(0, parent_rows, child_target_prob)
    coverage = coverage_by_row[parent_rows].clamp(max=1.0)
    row_max = target.max(dim=-1).values.clamp_min(1e-30)
    row_top2 = target.topk(k=2, dim=-1).values.clamp_min(1e-30)
    row_log_margin = (row_top2[:, 0].log() - row_top2[:, 1].log()).clamp_min(0.0)
    relative = (child_target_prob / row_max[parent_rows]).clamp(max=1.0)
    proposal_ratio = child_target_prob / child_draft_prob.clamp_min(1e-30)
    target_top1 = target.argmax(dim=-1)
    future_top1_match = torch.zeros(
        count,
        dtype=torch.bool,
        device=target.device,
    )
    child_nodes = [
        node.index for node in topology.nodes if node.parent is not None
    ]
    if child_nodes:
        child_index = torch.tensor(
            child_nodes,
            dtype=torch.int64,
            device=target.device,
        )
        child_parent = torch.tensor(
            [int(topology.nodes[index].parent) for index in child_nodes],
            dtype=torch.int64,
            device=target.device,
        )
        child_matches = (
            ids[child_index] == target_top1[child_parent + 1]
        ).to(torch.int64)
        match_counts = torch.zeros(
            count,
            dtype=torch.int64,
            device=target.device,
        )
        match_counts.scatter_add_(0, child_parent, child_matches)
        future_top1_match = match_counts > 0
    if config.coverage_mode == "coverage_gate":
        # Candidate-set coverage is a parent-level reliability veto.  It must
        # not positively boost every sibling merely because the proposer made
        # the tree wider; the actual token score remains target-anchored.
        confidence = relative
    elif config.coverage_weight == 0.0:
        confidence = relative
    elif config.coverage_weight == 1.0:
        confidence = coverage
    else:
        confidence = (
            coverage.pow(config.coverage_weight)
            * relative.pow(1.0 - config.coverage_weight)
        )
    if eos_ids:
        eos_index = torch.tensor(eos_ids, dtype=torch.int64, device=target.device)
        eos_mass_by_row = target.index_select(1, eos_index).sum(dim=-1)
        target_top1_is_eos = (target_top1[:, None] == eos_index[None, :]).any(dim=-1)
        child_is_eos = (ids[:, None] == eos_index[None, :]).any(dim=-1)
        eos_mass = eos_mass_by_row[parent_rows]
        eos_protected = (
            target_top1_is_eos[parent_rows]
            & (eos_mass >= config.eos_protection_threshold)
            & ~child_is_eos
        )
    else:
        eos_mass = torch.zeros_like(confidence)
        eos_protected = torch.zeros_like(confidence, dtype=torch.bool)

    # Match the local relaxation strength used by the chain Cactus baseline.
    # Direct ``cactus`` uses this as its complete node-survival rule. The
    # proposed ``cactus_guided`` mode uses it only as a risk-calibration prior
    # after the target-relative tree verifier reaches a dead frontier.
    cactus_boosted = (
        child_target_prob
        + torch.sqrt(
            (
                2.0
                * config.cactus_delta
                * child_target_prob
                * (1.0 - child_target_prob)
            ).clamp_min(0.0)
        )
    ).clamp(max=1.0)
    cactus_acceptance = torch.minimum(
        torch.ones_like(cactus_boosted),
        cactus_boosted / child_draft_prob.clamp_min(1e-30),
    )
    cactus_path_confidence = (
        relative.clamp_min(1e-30).pow(config.cactus_target_weight)
        * cactus_acceptance.clamp_min(1e-30).pow(
            1.0 - config.cactus_target_weight
        )
    )
    if config.support_mode == "cactus":
        cactus_random = torch.rand(
            cactus_acceptance.shape,
            device=target.device,
            generator=generator,
        )
    else:
        # Do not consume request RNG state for the existing verifier modes.
        cactus_random = torch.full_like(cactus_acceptance, -1.0)
    local_signals = torch.stack(
        (
            coverage,
            relative,
            confidence,
            eos_mass,
            eos_protected.to(torch.float32),
            child_target_prob,
            child_draft_prob,
            proposal_ratio,
            future_top1_match.to(torch.float32),
            cactus_boosted,
            cactus_acceptance,
            cactus_random,
            cactus_path_confidence,
            row_log_margin[parent_rows],
        ),
        dim=-1,
    )
    local_signal_rows = local_signals.detach().cpu().tolist()

    for depth in range(1, topology.max_depth + 1):
        if depth == 1:
            parents: tuple[int | None, ...] = (None,)
        else:
            parents = tuple(
                node.index
                for node in topology.nodes
                if node.depth == depth - 1 and node.index in survived
            )
        for parent in parents:
            children = topology.children_of(parent)
            if not children:
                continue
            parent_rescue_count = (
                0 if parent is None else rescue_count_by_node[parent]
            )
            group: list[dict[str, Any]] = []
            for child in children:
                (
                    node_coverage,
                    node_relative,
                    node_confidence,
                    node_eos_probability,
                    node_eos_protected,
                    node_target_probability,
                    node_draft_probability,
                    node_proposal_ratio,
                    node_future_top1_match,
                    node_cactus_boosted,
                    node_cactus_acceptance,
                    node_cactus_random,
                    node_cactus_path_confidence,
                    node_target_log_margin,
                ) = local_signal_rows[child]
                eos_veto = bool(node_eos_protected)
                # target_path_rescue deliberately judges the concrete token
                # against P at its own parent prefix. Candidate-set coverage
                # remains diagnostic only: otherwise merely making the tree
                # wider can make every sibling easier to accept.
                coverage_pass = (
                    config.support_mode
                    in {"target_path_rescue", "prefix_reopen_rescue"}
                    or node_coverage >= config.min_target_coverage
                )
                proposal_supported = (
                    config.support_mode == "dual"
                    and node_proposal_ratio
                    >= config.proposal_support_ratio
                )
                target_confirmed = (
                    config.support_mode == "confirmed"
                    and node_relative
                    >= config.confirmation_min_relative
                    and (
                        node_proposal_ratio
                        >= config.proposal_support_ratio
                        or bool(node_future_top1_match)
                    )
                )
                cactus_survival = (
                    config.support_mode == "cactus"
                    and node_cactus_random <= node_cactus_acceptance
                )
                support_survival = (
                    cactus_survival
                    if config.support_mode == "cactus"
                    else (
                        node_confidence >= config.relax_threshold
                        or proposal_supported
                        or target_confirmed
                    )
                )
                keep = (
                    support_survival
                    and coverage_pass
                    and not eos_veto
                )
                group.append(
                    {
                        "node": child,
                        "parent": parent,
                        "coverage": node_coverage,
                        "relative_target_support": node_relative,
                        "relaxed_confidence": node_confidence,
                        "coverage_pass": coverage_pass,
                        "target_probability": node_target_probability,
                        "target_log_margin": node_target_log_margin,
                        "draft_probability": node_draft_probability,
                        "proposal_support_ratio": node_proposal_ratio,
                        "proposal_supported": proposal_supported,
                        "future_top1_match": bool(node_future_top1_match),
                        "target_confirmed": target_confirmed,
                        "cactus_boosted_probability": node_cactus_boosted,
                        "cactus_acceptance_probability": node_cactus_acceptance,
                        "cactus_random": (
                            node_cactus_random
                            if config.support_mode == "cactus"
                            else None
                        ),
                        "cactus_survival": cactus_survival,
                        "cactus_guided_eligible": False,
                        "cactus_guided_score": None,
                        "cactus_guided_rescued": False,
                        "target_path_rescue_eligible": False,
                        "target_path_rescue_score": None,
                        "target_path_rescue_threshold": (
                            config.rescue_score_threshold
                        ),
                        "target_path_rescued": False,
                        "prefix_reopen_eligible": False,
                        "prefix_reopen_rescued": False,
                        "rescue_depth_threshold": None,
                        "rescue_continuation_certified": False,
                        "rescue_continuation_discount": 0.0,
                        "rescue_margin_multiplier": 1.0,
                        "guided_rescue_count_before": parent_rescue_count,
                        "path_confidence": (
                            node_cactus_path_confidence
                            if config.support_mode == "cactus"
                            else node_confidence
                        ),
                        "target_eos_probability": node_eos_probability,
                        "eos_protected": eos_veto,
                        "normal_survival": keep,
                        "rescued": False,
                        "rescue_eligible": False,
                        "rescue_accept_probability": None,
                        "rescue_random": None,
                    }
                )

            normal_rows = [row for row in group if row["normal_survival"]]
            for row in normal_rows:
                child = int(row["node"])
                survived.add(child)
                rescue_count_by_node[child] = parent_rescue_count
                s_by_node[child] = (
                    float(row["relative_target_support"])
                    if config.support_mode
                    in {"target_path_rescue", "prefix_reopen_rescue"}
                    else float(row["path_confidence"])
                )

            # target_path_rescue deliberately keeps the ordinary tree scan
            # free of rescue state.  A single extension is considered only
            # after the longest, most target-supported ordinary path has been
            # selected.  This makes every successful rescue contribute
            # exactly one additional draft token instead of merely entering
            # another all-tree path competition.
            # Log analysis showed that the largest MAL loss is at the root.
            # In prefix-reopen mode, every reachable path may therefore spend
            # one deterministic rescue, with a threshold that becomes more
            # conservative at later depths. Unlike the historical v1 rule,
            # all eligible failed siblings survive instead of retaining only
            # one per parent. A rescued child carries its one-rescue debt into
            # descendants, which may continue only through ordinary survival.
            if (
                prefix_reopen_mode
                and parent_rescue_count
                < config.max_guided_rescues_per_path
            ):
                target_log_margin = float(group[0]["target_log_margin"])
                margin_ratio = (
                    min(
                        target_log_margin / config.rescue_margin_reference,
                        1.0,
                    )
                    if config.rescue_margin_reference > 0.0
                    else 0.0
                )
                margin_multiplier = (
                    1.0 + config.rescue_margin_penalty * margin_ratio
                )
                depth_threshold = min(
                    1.0,
                    config.rescue_score_threshold
                    * (1.0 + config.rescue_depth_penalty * (depth - 1))
                    * margin_multiplier,
                )
                for row in group:
                    if bool(row["normal_survival"]):
                        continue
                    relative_support = float(row["relative_target_support"])
                    target_probability = float(row["target_probability"])
                    cactus_probability = float(
                        row["cactus_acceptance_probability"]
                    )
                    # A lower rescue threshold is useful only when accepting
                    # this node immediately unlocks an ordinary target-
                    # supported continuation.  The target tree forward has
                    # already evaluated every direct child at the correct
                    # parent prefix, so this certificate adds no model call
                    # and does not use future requests/history.
                    child = int(row["node"])
                    continuation_certified = any(
                        float(local_signal_rows[future_child][2])
                        >= config.relax_threshold
                        and not bool(local_signal_rows[future_child][4])
                        for future_child in topology.children_of(child)
                    )
                    row_threshold = depth_threshold
                    if (
                        continuation_certified
                        and depth >= config.rescue_continuation_min_depth
                    ):
                        row_threshold *= (
                            1.0 - config.rescue_continuation_discount
                        )
                    rescue_score = (
                        max(relative_support, 1e-30)
                        ** config.cactus_target_weight
                        * max(cactus_probability, 1e-30)
                        ** (1.0 - config.cactus_target_weight)
                    )
                    eligible = (
                        not bool(row["eos_protected"])
                        and relative_support >= config.rescue_min_relative
                        and target_probability >= config.rescue_min_target_prob
                        and rescue_score >= row_threshold
                    )
                    row["target_path_rescue_score"] = rescue_score
                    row["target_path_rescue_threshold"] = row_threshold
                    row["prefix_reopen_eligible"] = eligible
                    row["rescue_depth_threshold"] = row_threshold
                    row["rescue_continuation_certified"] = (
                        continuation_certified
                    )
                    row["rescue_continuation_discount"] = (
                        config.rescue_continuation_discount
                        if continuation_certified
                        and depth >= config.rescue_continuation_min_depth
                        else 0.0
                    )
                    row["rescue_margin_multiplier"] = margin_multiplier
                    if not eligible:
                        continue
                    child = int(row["node"])
                    row["rescue_eligible"] = True
                    row["rescue_accept_probability"] = float(
                        row["cactus_acceptance_probability"]
                    )
                    row["rescue_random"] = None
                    row["rescue_decision_mode"] = (
                        "deterministic-prefix-reopen"
                    )
                    row["rescued"] = True
                    row["prefix_reopen_rescued"] = True
                    survived.add(child)
                    rescued_nodes.add(child)
                    rescue_count_by_node[child] = parent_rescue_count + 1
                    s_by_node[child] = max(relative_support, 1e-30)

            # Rescue only a dead frontier. In ``cactus_guided`` mode this is
            # the only place where Cactus can change survival: the ordinary
            # tree rule above remains target-relative, and a path can cross at
            # most one calibrated frontier. A candidate additionally needs
            # tree-specific evidence (proposal overlap, retained target-top1
            # continuation, or medium target-relative support), so this mode
            # cannot collapse into independent Cactus verification per node.
            # The configured per-path limit defaults to one for backward
            # compatibility; wider experimental profiles can spend a second
            # rescue only after another later frontier has completely died.
            guided_mode = config.support_mode == "cactus_guided"
            rescue_limit = (
                config.max_guided_rescues_per_path if guided_mode else 1
            )
            if (
                not target_path_mode
                and not prefix_reopen_mode
                and
                (config.frontier_rescue or guided_mode)
                and not normal_rows
                and parent_rescue_count < rescue_limit
            ):
                guided_direct_floor = max(
                    config.confirmation_min_relative,
                    0.5 * config.relax_threshold,
                )
                rescue_rows = [
                    row
                    for row in group
                    if not row["eos_protected"]
                    and bool(row["coverage_pass"])
                    and float(row["relative_target_support"])
                    >= (
                        config.confirmation_min_relative
                        if guided_mode
                        else config.rescue_min_relative
                    )
                    and float(row["target_probability"])
                    >= config.rescue_min_target_prob
                    and (
                        not guided_mode
                        or bool(row["future_top1_match"])
                        or float(row["proposal_support_ratio"])
                        >= config.proposal_support_ratio
                        or float(row["relative_target_support"])
                        >= guided_direct_floor
                    )
                ]
                if rescue_rows:
                    if guided_mode:
                        for row in rescue_rows:
                            guided_score = (
                                max(
                                    float(row["relative_target_support"]),
                                    1e-30,
                                )
                                ** config.cactus_target_weight
                                * max(
                                    float(
                                        row["cactus_acceptance_probability"]
                                    ),
                                    1e-30,
                                )
                                ** (1.0 - config.cactus_target_weight)
                            )
                            if bool(row["future_top1_match"]):
                                guided_score *= 1.1
                            row["cactus_guided_eligible"] = True
                            row["cactus_guided_score"] = guided_score
                    rescue_row = max(
                        rescue_rows,
                        key=lambda row: (
                            float(
                                row["cactus_guided_score"]
                                if guided_mode
                                else row["relative_target_support"]
                            ),
                            bool(row["future_top1_match"]),
                            float(row["relative_target_support"]),
                            float(row["target_probability"]),
                            -int(row["node"]),
                        ),
                    )
                    rescue_row["rescue_eligible"] = True
                    p_y = float(rescue_row["target_probability"])
                    q_y = float(rescue_row["draft_probability"])
                    if guided_mode:
                        accept_probability = float(
                            rescue_row["cactus_acceptance_probability"]
                        )
                    else:
                        boosted = min(
                            1.0,
                            p_y
                            + math.sqrt(
                                max(
                                    2.0
                                    * config.rescue_delta
                                    * p_y
                                    * (1.0 - p_y),
                                    0.0,
                                )
                            ),
                        )
                        accept_probability = min(
                            1.0, boosted / max(q_y, 1e-30)
                        )
                    random_value = float(
                        torch.rand(
                            (), device=target.device, generator=generator
                        ).item()
                    )
                    rescue_row["rescue_accept_probability"] = accept_probability
                    rescue_row["rescue_random"] = random_value
                    if random_value <= accept_probability:
                        child = int(rescue_row["node"])
                        rescue_row["rescued"] = True
                        rescue_row["cactus_guided_rescued"] = guided_mode
                        survived.add(child)
                        rescued_nodes.add(child)
                        rescue_count_by_node[child] = parent_rescue_count + 1
                        if guided_mode:
                            s_by_node[child] = max(
                                float(rescue_row["cactus_guided_score"]),
                                1e-30,
                            )
                        else:
                            s_by_node[child] = max(
                                float(rescue_row["relaxed_confidence"])
                                * accept_probability,
                                1e-30,
                            )

            for row in group:
                child = int(row["node"])
                keep = child in survived
                statuses[child] = "SURVIVE" if keep else "PRUNE"
                row["survived"] = keep
                diagnostic_rows.append(row)

    # A surviving node is a complete path endpoint only when no surviving
    # child extends it.  Counting every prefix as a path overstated the actual
    # tree width and allowed a short prefix to compete against its own valid
    # continuation.
    endpoint_nodes = sorted(
        node
        for node in survived
        if not any(child in survived for child in topology.children_of(node))
    )
    if not endpoint_nodes and not target_path_mode:
        correction = sample_categorical(target[0], generator)
        return TreeVerifyResult(
            output_token_ids=(correction,),
            accepted_node_indices=(),
            selected_branch=None,
            terminal="target-fallback",
            target_validation_nodes=count,
            evaluated_node_statuses=tuple(sorted(statuses.items())),
            node_diagnostics=tuple(diagnostic_rows),
            surviving_paths=0,
            selected_path_score=None,
        )

    all_surviving_paths = tuple(
        topology.path_to(endpoint) for endpoint in endpoint_nodes
    )
    selected_score: float | None = None
    if all_surviving_paths:
        longest_length = max(len(path) for path in all_surviving_paths)
        if target_path_mode or config.path_selection_mode == "longest":
            # Accepted length is the primary objective. Target confidence is
            # used only between paths with the same maximum surviving depth.
            candidate_paths = [
                path
                for path in all_surviving_paths
                if len(path) == longest_length
            ]
        else:
            # Every surviving prefix is a valid stopping point. This restores
            # the original confidence--length objective instead of forcing a
            # weak tail merely because one exists.
            candidate_paths = [
                topology.path_to(node) for node in sorted(survived)
            ]
        log_scores: list[float] = []
        for path in candidate_paths:
            confidence_log = sum(
                math.log(max(s_by_node[node], 1e-30)) for node in path
            )
            length = len(path)
            # target_path_rescue has already filtered to equal maximum
            # length. Its tie break is target confidence only.
            length_term = (
                0.0
                if target_path_mode
                else config.length_reward * math.log(length)
            )
            log_scores.append(confidence_log / length + length_term)
        score_tensor = torch.tensor(
            log_scores,
            dtype=torch.float32,
            device=target.device,
        )
        selected_index = _sample_path_index(
            score_tensor, config.path_temperature, generator
        )
        selected_path = candidate_paths[selected_index]
        selected_score = float(score_tensor[selected_index].exp())
    else:
        selected_path = ()

    # A target-path rescue is a literal one-token extension of the already
    # chosen ordinary path. It does not re-enter whole-tree path competition
    # and it never unlocks a second speculative token. Thus every successful
    # rescue has an auditable +1 contribution to accepted draft length.
    target_path_rescue_selected = False
    if target_path_mode and len(selected_path) < topology.max_depth:
        frontier_parent = selected_path[-1] if selected_path else None
        diagnostics_by_node = {
            int(row["node"]): row for row in diagnostic_rows
        }
        extension_rows: list[dict[str, Any]] = []
        for child in topology.children_of(frontier_parent):
            row = diagnostics_by_node.get(child)
            if row is None or bool(row["normal_survival"]):
                continue
            relative_support = float(row["relative_target_support"])
            target_probability = float(row["target_probability"])
            cactus_probability = float(
                row["cactus_acceptance_probability"]
            )
            score = (
                max(relative_support, 1e-30)
                ** config.cactus_target_weight
                * max(cactus_probability, 1e-30)
                ** (1.0 - config.cactus_target_weight)
            )
            eligible = (
                not bool(row["eos_protected"])
                and relative_support >= config.rescue_min_relative
                and target_probability >= config.rescue_min_target_prob
                and score >= config.rescue_score_threshold
            )
            row["target_path_rescue_score"] = score
            row["target_path_rescue_eligible"] = eligible
            if eligible:
                extension_rows.append(row)
        if extension_rows:
            rescue_row = max(
                extension_rows,
                key=lambda row: (
                    float(row["target_path_rescue_score"]),
                    float(row["relative_target_support"]),
                    float(row["target_probability"]),
                    -int(row["node"]),
                ),
            )
            rescue_node = int(rescue_row["node"])
            rescue_row["rescue_eligible"] = True
            rescue_row["rescue_accept_probability"] = float(
                rescue_row["cactus_acceptance_probability"]
            )
            rescue_row["rescue_random"] = None
            rescue_row["rescue_decision_mode"] = "score-threshold"
            rescue_row["rescued"] = True
            rescue_row["target_path_rescued"] = True
            rescue_row["rescue_extension_tokens"] = 1
            rescued_nodes.add(rescue_node)
            s_by_node[rescue_node] = max(
                float(rescue_row["relative_target_support"]),
                1e-30,
            )
            selected_path = tuple(selected_path) + (rescue_node,)
            target_path_rescue_selected = True
            confidence_log = sum(
                math.log(max(s_by_node[node], 1e-30))
                for node in selected_path
            )
            selected_score = math.exp(
                confidence_log / len(selected_path)
            )

    if not selected_path:
        correction = sample_categorical(target[0], generator)
        for row in diagnostic_rows:
            row["status"] = statuses[int(row["node"])]
        return TreeVerifyResult(
            output_token_ids=(correction,),
            accepted_node_indices=(),
            selected_branch=None,
            terminal="target-fallback",
            target_validation_nodes=count,
            evaluated_node_statuses=tuple(sorted(statuses.items())),
            node_diagnostics=tuple(diagnostic_rows),
            surviving_paths=0,
            selected_path_score=None,
        )

    selected_rescues = sum(node in rescued_nodes for node in selected_path)
    if prefix_reopen_mode and selected_rescues:
        diagnostics_by_node = {
            int(row["node"]): row for row in diagnostic_rows
        }
        for position, node in enumerate(selected_path):
            if node not in rescued_nodes:
                continue
            diagnostics_by_node[node]["rescue_unlocked_tokens"] = (
                len(selected_path) - position
            )
    for node in survived:
        statuses[node] = "SELECTED" if node in selected_path else "SURVIVE"
    if target_path_rescue_selected:
        statuses[selected_path[-1]] = "SELECTED"
    for row in diagnostic_rows:
        row["status"] = statuses[int(row["node"])]

    output = [int(ids[node]) for node in selected_path]
    guided_selected = any(
        bool(row.get("cactus_guided_rescued"))
        and int(row["node"]) in selected_path
        for row in diagnostic_rows
    )
    if target_path_mode:
        terminal = (
            "target-path-extension"
            if target_path_rescue_selected
            else "target-path"
        )
    elif prefix_reopen_mode:
        terminal = (
            "prefix-reopen-rescue-path"
            if selected_rescues
            else "prefix-ordinary-path"
        )
    else:
        terminal = (
            "cactus-guided-path"
            if guided_selected
            else (
                "frontier-rescue-path" if selected_rescues else "relaxed-path"
            )
        )
    if config.append_target_anchor:
        output.append(sample_categorical(target[selected_path[-1] + 1], generator))
        if target_path_mode:
            terminal = (
                "target-path-extension-bonus"
                if target_path_rescue_selected
                else "target-path-bonus"
            )
        elif prefix_reopen_mode:
            terminal = (
                "prefix-reopen-rescue-bonus"
                if selected_rescues
                else "prefix-ordinary-bonus"
            )
        else:
            terminal = (
                "cactus-guided-bonus"
                if guided_selected
                else (
                    "frontier-rescue-bonus"
                    if selected_rescues
                    else "relaxed-bonus"
                )
            )
    return TreeVerifyResult(
        output_token_ids=tuple(output),
        accepted_node_indices=selected_path,
        selected_branch=topology.nodes[selected_path[0]].branch,
        terminal=terminal,
        target_validation_nodes=count,
        evaluated_node_statuses=tuple(sorted(statuses.items())),
        node_diagnostics=tuple(diagnostic_rows),
        surviving_paths=len(all_surviving_paths),
        selected_path_score=selected_score,
    )
