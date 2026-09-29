from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from remtp.dynamic_mtp_tree import (
    adaptive_trunk_relaxation_plan,
    adaptive_node_limit,
    cactus_trunk_rescue_verify,
    DynamicTreeConfig,
    DynamicTreeTopology,
    level_expansion_order,
    level_expansion_budget,
    mtp_layer_for_tree_depth,
    normalized_entropy,
    path_expansion_value,
    residual_hit_tree_verify,
    select_dynamic_candidates,
    target_dominant_relaxed_verify,
)


def test_adaptive_node_limit_uses_entropy_without_target_signal() -> None:
    config = DynamicTreeConfig(
        max_nodes=12,
        adaptive_base_nodes=9,
        adaptive_entropy_threshold=0.05,
    )
    assert adaptive_node_limit((0.01, 0.04), config) == 9
    assert adaptive_node_limit((0.01, 0.05), config) == 12
    assert adaptive_node_limit(
        (0.20,), replace(config, adaptive_base_nodes=0)
    ) == 12


def _cactus_caterpillar_topology() -> DynamicTreeTopology:
    return DynamicTreeTopology.from_paths(
        (
            (0,),
            (1,),
            (0, 0),
            (0, 1),
            (0, 0, 0),
            (0, 0, 1),
        ),
        name="cactus-trunk-rescue",
    )


def _cactus_rescue_config(**overrides: object) -> DynamicTreeConfig:
    values: dict[str, object] = {
        "max_depth": 3,
        "max_nodes": 6,
        "max_children_per_parent": 2,
        "support_mode": "cactus_trunk_rescue",
        "cactus_delta": 0.0,
        "rescue_delta": 0.0,
        "rescue_min_relative": 0.0,
        "rescue_min_target_prob": 0.0,
        "append_target_anchor": True,
    }
    values.update(overrides)
    return DynamicTreeConfig(**values)


def _residual_hit_config(
    support_mode: str,
    **overrides: object,
) -> DynamicTreeConfig:
    values: dict[str, object] = {
        "max_depth": 3,
        "max_nodes": 6,
        "max_children_per_parent": 2,
        "support_mode": support_mode,
        "cactus_delta": 0.0,
        "append_target_anchor": True,
    }
    values.update(overrides)
    return DynamicTreeConfig(**values)


def _target_path_config(**overrides: object) -> DynamicTreeConfig:
    values: dict[str, object] = {
        "max_depth": 3,
        "max_nodes": 8,
        "max_children_per_parent": 3,
        "coverage_mode": "coverage_gate",
        "min_target_coverage": 0.0,
        "relax_threshold": 0.50,
        "support_mode": "target_path_rescue",
        "cactus_delta": 1.0,
        "cactus_target_weight": 0.5,
        "max_guided_rescues_per_path": 1,
        "rescue_score_threshold": 0.70,
        "rescue_min_relative": 0.01,
        "rescue_min_target_prob": 0.0,
        "path_selection_mode": "longest",
        "path_temperature": 0.0,
        "append_target_anchor": False,
    }
    values.update(overrides)
    return DynamicTreeConfig(**values)


def _prefix_reopen_config(**overrides: object) -> DynamicTreeConfig:
    values: dict[str, object] = {
        "max_depth": 3,
        "max_nodes": 8,
        "max_children_per_parent": 3,
        "coverage_mode": "coverage_gate",
        "min_target_coverage": 0.0,
        "relax_threshold": 0.50,
        "support_mode": "prefix_reopen_rescue",
        "cactus_delta": 1.0,
        "cactus_target_weight": 0.5,
        "max_guided_rescues_per_path": 1,
        "rescue_score_threshold": 0.50,
        "rescue_depth_penalty": 0.0,
        "rescue_min_relative": 0.01,
        "rescue_min_target_prob": 0.0,
        "path_selection_mode": "longest",
        "path_temperature": 0.0,
        "append_target_anchor": False,
    }
    values.update(overrides)
    return DynamicTreeConfig(**values)


def test_prefix_reopen_rescue_unlocks_ordinary_descendants() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (1,), (1, 0), (1, 0, 0))
    )
    draft_ids = torch.tensor([0, 1, 2, 3])
    draft = torch.tensor(
        [
            [0.55, 0.35, 0.04, 0.03, 0.03],
            [0.55, 0.35, 0.04, 0.03, 0.03],
            [0.05, 0.05, 0.80, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.80, 0.05],
        ]
    )
    target = torch.tensor(
        [
            [0.60, 0.24, 0.05, 0.05, 0.06],
            [0.70, 0.05, 0.10, 0.10, 0.05],
            [0.05, 0.05, 0.75, 0.10, 0.05],
            [0.05, 0.05, 0.05, 0.80, 0.05],
            [0.20, 0.20, 0.20, 0.20, 0.20],
        ]
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        target,
        config=_prefix_reopen_config(),
    )
    assert result.accepted_node_indices == (1, 2, 3)
    rows = {int(row["node"]): row for row in result.node_diagnostics}
    assert rows[1]["prefix_reopen_rescued"] is True
    assert rows[2]["normal_survival"] is True
    assert rows[3]["normal_survival"] is True
    assert rows[1]["rescue_unlocked_tokens"] == 3
    assert result.terminal == "prefix-reopen-rescue-path"


def test_prefix_reopen_rescue_keeps_all_eligible_failed_siblings() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (1,), (2,)))
    draft_ids = torch.tensor([0, 1, 2])
    draft = torch.full((3, 4), 0.25)
    target = torch.tensor(
        [
            [0.25, 0.20, 0.15, 0.40],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        target,
        config=_prefix_reopen_config(
            max_nodes=3,
            relax_threshold=0.80,
            cactus_target_weight=1.0,
            rescue_score_threshold=0.30,
        ),
    )
    rows = tuple(result.node_diagnostics)
    assert sum(bool(row["prefix_reopen_rescued"]) for row in rows) == 3
    assert result.surviving_paths == 3
    assert result.accepted_drafts == 1


def test_prefix_reopen_rescue_cannot_be_spent_twice_on_one_path() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (0, 0)))
    draft_ids = torch.tensor([0, 1])
    draft = torch.tensor([[0.40, 0.10, 0.50], [0.10, 0.40, 0.50]])
    target = torch.tensor(
        [
            [0.20, 0.30, 0.50],
            [0.30, 0.20, 0.50],
            [0.30, 0.30, 0.40],
        ]
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        target,
        config=_prefix_reopen_config(
            max_nodes=2,
            max_children_per_parent=1,
            cactus_target_weight=1.0,
            rescue_score_threshold=0.30,
        ),
    )
    rows = {int(row["node"]): row for row in result.node_diagnostics}
    assert rows[0]["prefix_reopen_rescued"] is True
    assert rows[1]["prefix_reopen_rescued"] is False
    assert rows[1]["normal_survival"] is False
    assert result.accepted_node_indices == (0,)


def test_prefix_reopen_discount_requires_ordinary_continuation() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (0, 0), (0, 0, 0), (1,), (1, 0), (1, 0, 0))
    )
    draft_ids = torch.tensor([0, 1, 2, 2, 2, 2])
    draft = torch.full((6, 4), 0.25)
    target = torch.tensor(
        [
            [0.40, 0.40, 0.10, 0.10],  # both roots survive
            [0.05, 0.55, 0.35, 0.05],  # node 2: relative 0.636
            [0.05, 0.55, 0.35, 0.05],  # node 3: same support
            [0.05, 0.05, 0.85, 0.05],  # node 4 certifies node 2
            [0.05, 0.05, 0.05, 0.85],  # node 5 does not certify node 3
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        target,
        config=_prefix_reopen_config(
            max_nodes=6,
            relax_threshold=0.70,
            cactus_target_weight=1.0,
            rescue_score_threshold=0.50,
            rescue_depth_penalty=0.50,
            rescue_continuation_discount=0.20,
            rescue_continuation_min_depth=2,
        ),
    )
    rows = {int(row["node"]): row for row in result.node_diagnostics}
    assert rows[2]["normal_survival"] is False
    assert rows[2]["rescue_continuation_certified"] is True
    assert rows[2]["rescue_depth_threshold"] == pytest.approx(0.60)
    assert rows[2]["prefix_reopen_rescued"] is True
    assert rows[3]["rescue_continuation_certified"] is False
    assert rows[3]["rescue_depth_threshold"] == pytest.approx(0.75)
    assert rows[3]["prefix_reopen_rescued"] is False


def test_prefix_reopen_margin_adapts_rescue_to_target_decisiveness() -> None:
    topology = DynamicTreeTopology.from_paths(((0,),))
    draft_ids = torch.tensor([0])
    draft = torch.tensor([[0.90, 0.05, 0.03, 0.02]])
    uncertain_target = torch.tensor(
        [
            [0.045, 0.46, 0.45, 0.045],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    decisive_target = torch.tensor(
        [
            [0.03, 0.85, 0.10, 0.02],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    config = _prefix_reopen_config(
        max_depth=1,
        max_nodes=1,
        max_children_per_parent=1,
        rescue_score_threshold=0.06,
        rescue_margin_reference=1.0,
        rescue_margin_penalty=1.0,
    )

    uncertain = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        uncertain_target,
        config=config,
    )
    decisive = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        decisive_target,
        generator=torch.Generator().manual_seed(7),
        config=config,
    )

    uncertain_row = uncertain.node_diagnostics[0]
    decisive_row = decisive.node_diagnostics[0]
    assert uncertain_row["prefix_reopen_rescued"] is True
    assert uncertain_row["rescue_depth_threshold"] < 0.065
    assert decisive_row["prefix_reopen_rescued"] is False
    assert decisive_row["rescue_depth_threshold"] == pytest.approx(0.12)
    assert decisive_row["rescue_margin_multiplier"] == pytest.approx(2.0)


def test_target_path_rescue_prefers_longest_surviving_path() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (1,), (1, 0)))
    draft_ids = torch.tensor([0, 1, 0])
    draft = torch.tensor(
        [
            [0.60, 0.40, 0.00],
            [0.40, 0.60, 0.00],
            [0.60, 0.20, 0.20],
        ]
    )
    target = torch.tensor(
        [
            [0.60, 0.36, 0.04],  # both roots survive
            [0.90, 0.05, 0.05],
            [0.55, 0.40, 0.05],  # child of root 1 survives
            [0.50, 0.25, 0.25],
        ]
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        target,
        config=_target_path_config(
            max_nodes=3,
            relax_threshold=0.50,
            max_guided_rescues_per_path=0,
        ),
    )
    assert result.accepted_node_indices == (1, 2)
    assert result.accepted_drafts == 2
    assert result.terminal == "target-path"


def test_target_path_rescue_uses_target_confidence_to_break_length_tie() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (1,), (0, 0), (1, 0))
    )
    draft_ids = torch.tensor([0, 1, 0, 1])
    draft = torch.full((4, 3), 1.0 / 3.0)
    target = torch.tensor(
        [
            [0.60, 0.40, 0.00],
            [0.90, 0.05, 0.05],
            [0.30, 0.60, 0.10],
            [0.50, 0.25, 0.25],
            [0.50, 0.25, 0.25],
        ]
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        target,
        config=_target_path_config(
            max_nodes=4,
            relax_threshold=0.50,
            max_guided_rescues_per_path=0,
        ),
    )
    assert result.accepted_node_indices == (0, 2)
    assert result.terminal == "target-path"


def test_target_path_rescue_extends_the_selected_path_by_exactly_one() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (1,), (0, 0), (1, 0), (0, 0, 0))
    )
    draft_ids = torch.tensor([0, 1, 0, 0, 0])
    draft = torch.tensor(
        [
            [0.60, 0.20, 0.20],
            [0.60, 0.20, 0.20],
            [0.20, 0.70, 0.10],
            [0.70, 0.20, 0.10],
            [0.70, 0.20, 0.10],
        ]
    )
    target = torch.tensor(
        [
            [0.55, 0.35, 0.10],  # root 1 fails tau=.8, but is rescuable
            [0.80, 0.10, 0.10],
            [0.80, 0.10, 0.10],
            [0.35, 0.55, 0.10],  # selected frontier child is rescuable
            [0.90, 0.05, 0.05],
            [0.50, 0.25, 0.25],
        ]
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        target,
        config=_target_path_config(
            max_nodes=5,
            relax_threshold=0.80,
            rescue_score_threshold=0.70,
        ),
    )
    rows = {int(row["node"]): row for row in result.node_diagnostics}
    assert rows[4]["normal_survival"] is False
    assert rows[4]["target_path_rescue_eligible"] is True
    assert rows[4]["target_path_rescued"] is True
    assert rows[4]["rescue_extension_tokens"] == 1
    assert result.accepted_node_indices == (0, 2, 4)
    assert result.terminal == "target-path-extension"


def test_target_path_rescue_can_fill_one_remaining_depth_slot() -> None:
    topology = DynamicTreeTopology.from_paths(((0,),))
    draft_ids = torch.tensor([0])
    draft = torch.tensor([[0.10, 0.90]])
    target = torch.tensor([[0.40, 0.60], [0.50, 0.50]])
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(1),
        config=_target_path_config(
            max_depth=1,
            max_nodes=1,
            relax_threshold=0.80,
            rescue_score_threshold=0.0,
        ),
    )
    assert result.accepted_drafts == 1
    assert result.terminal == "target-path-extension"
    assert result.node_diagnostics[0]["target_path_rescued"] is True
    assert result.node_diagnostics[0]["rescue_extension_tokens"] == 1


def test_cactus_trunk_rescue_keeps_exact_sampled_trunk_and_bonus() -> None:
    topology = _cactus_caterpillar_topology()
    draft_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    draft = torch.tensor([[0.20, 0.50, 0.20, 0.10]]).repeat(6, 1)
    target = torch.full((7, 4), 0.25)
    # Score the three trunk nodes with p(y) >= q(y), then make the leaf bonus
    # deterministic. Rows are anchor, N0, N1, N2, N3, N4, N5.
    target[0] = torch.tensor([0.70, 0.10, 0.10, 0.10])
    target[1] = torch.tensor([0.70, 0.10, 0.10, 0.10])
    target[3] = torch.tensor([0.70, 0.10, 0.10, 0.10])
    target[5] = torch.tensor([0.00, 0.00, 0.00, 1.00])

    result = cactus_trunk_rescue_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(7),
        config=_cactus_rescue_config(),
    )
    assert result.accepted_node_indices == (0, 2, 4)
    assert result.output_token_ids == (0, 0, 0, 3)
    assert result.terminal == "cactus-trunk-bonus"
    assert dict(result.evaluated_node_statuses)[1] == "BACKUP_UNUSED"


def test_cactus_trunk_rejection_can_commit_one_rescue_and_target_bonus() -> None:
    topology = _cactus_caterpillar_topology()
    draft_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    draft = torch.tensor([[0.90, 0.05, 0.05, 0.00]]).repeat(6, 1)
    target = torch.full((7, 4), 0.25)
    # Root trunk token 0 is impossible under P and must reject. Its correction
    # residual strongly supports sibling token 1, which is accepted with
    # probability one; row N1+1 then supplies the target bonus.
    target[0] = torch.tensor([0.00, 0.80, 0.20, 0.00])
    target[2] = torch.tensor([0.00, 0.00, 0.00, 1.00])

    result = cactus_trunk_rescue_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(11),
        config=_cactus_rescue_config(),
    )
    assert result.accepted_node_indices == (1,)
    assert result.output_token_ids == (1, 3)
    assert result.terminal == "cactus-trunk-rescue-bonus"
    by_node = {row["node"]: row for row in result.node_diagnostics}
    assert by_node[0]["trunk_rejected"] is True
    assert by_node[1]["rescued"] is True


def test_sampled_primary_reopen_continues_down_rescued_subtree() -> None:
    topology = DynamicTreeTopology.from_paths(
        (
            (0,),
            (1,),
            (0, 0),
            (1, 0),
            (0, 0, 0),
            (1, 0, 0),
        ),
        name="sampled-primary-reopen",
    )
    draft_ids = torch.tensor([0, 1, 0, 2, 0, 3])
    draft = torch.tensor([[0.90, 0.05, 0.03, 0.02, 0.00]]).repeat(6, 1)
    target = torch.full((7, 5), 0.20)
    # Reject sampled root token 0, accept sibling token 1 from the Cactus
    # residual, then let its two descendants pass target-relative support.
    target[0] = torch.tensor([0.00, 0.80, 0.10, 0.05, 0.05])
    target[2] = torch.tensor([0.05, 0.05, 0.80, 0.05, 0.05])
    target[4] = torch.tensor([0.05, 0.05, 0.05, 0.80, 0.05])
    target[6] = torch.tensor([0.00, 0.00, 0.00, 0.00, 1.00])

    result = cactus_trunk_rescue_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(11),
        config=_cactus_rescue_config(
            max_nodes=6,
            support_mode="sampled_primary_reopen",
            relax_threshold=0.50,
        ),
    )

    assert result.accepted_node_indices == (1, 3, 5)
    assert result.output_token_ids == (1, 2, 3, 4)
    assert result.terminal == "sampled-primary-reopen-bonus"
    by_node = {row["node"]: row for row in result.node_diagnostics}
    assert by_node[1]["rescued"] is True
    assert by_node[1]["rescue_unlocked_tokens"] == 3
    assert by_node[3]["normal_survival"] is True
    assert by_node[5]["normal_survival"] is True


def _exact_residual_hit_fixture() -> tuple[
    DynamicTreeTopology,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0)),
        name="exact-residual-hit",
    )
    draft_ids = torch.tensor([0, 1, 0, 2, 0, 3])
    draft = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    return topology, draft_ids, draft, target


def test_sampled_primary_shadow_matches_exact_correction_without_reuse() -> None:
    topology, draft_ids, draft, target = _exact_residual_hit_fixture()
    result = residual_hit_tree_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(7),
        config=_residual_hit_config("sampled_primary_shadow"),
    )
    assert result.output_token_ids == (1,)
    assert result.accepted_node_indices == ()
    assert result.terminal == "sampled-primary-shadow-correction"
    by_node = {row["node"]: row for row in result.node_diagnostics}
    assert by_node[1]["correction_hit"] is True
    assert by_node[1]["rescued"] is False


def test_exact_residual_hit_can_append_unmodified_target_anchor() -> None:
    topology, draft_ids, draft, target = _exact_residual_hit_fixture()
    result = residual_hit_tree_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(7),
        config=_residual_hit_config("residual_hit_anchor"),
    )
    assert result.output_token_ids == (1, 2)
    assert result.accepted_node_indices == (1,)
    assert result.terminal == "residual-hit-target-anchor"


@pytest.mark.parametrize(
    ("mode", "terminal"),
    (
        ("residual_hit_strict", "residual-hit-strict-continuation-bonus"),
        ("residual_hit_cactus", "residual-hit-cactus-continuation-bonus"),
    ),
)
def test_exact_residual_hit_continuation_uses_branch_local_q(
    mode: str,
    terminal: str,
) -> None:
    topology, draft_ids, draft, target = _exact_residual_hit_fixture()
    result = residual_hit_tree_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(7),
        config=_residual_hit_config(mode),
    )
    assert result.output_token_ids == (1, 2, 3, 4)
    assert result.accepted_node_indices == (1, 3, 5)
    assert result.terminal == terminal
    by_node = {row["node"]: row for row in result.node_diagnostics}
    assert by_node[3]["proposal_role"] == "sampled_recovery_continuation"
    assert by_node[3]["verification_rule"] == (
        "cactus" if mode == "residual_hit_cactus" else "strict"
    )


def test_depth_calibrated_relaxation_changes_each_native_mtp_position() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (0, 0), (0, 0, 0)),
        name="depth-calibrated-chain",
    )
    draft_ids = torch.tensor([0, 0, 0])
    draft = torch.tensor([[0.25, 0.75]]).repeat(3, 1)
    target = torch.tensor([[0.25, 0.75]]).repeat(4, 1)
    weights = (1.0, 0.5, 0.25)
    config = _residual_hit_config(
        "residual_hit_anchor",
        cactus_delta=1.0,
        relaxation_depth_weights=weights,
    )

    result = residual_hit_tree_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(3),
        config=config,
    )

    rows = {int(row["trunk_position"]): row for row in result.node_diagnostics}
    assert result.accepted_drafts == 3
    for depth, weight in enumerate(weights, start=1):
        row = rows[depth]
        expected = min(
            1.0,
            0.25 + math.sqrt(2.0 * weight * 0.25 * 0.75),
        )
        assert row["relaxation_depth_weight"] == pytest.approx(weight)
        assert row["effective_relaxation_delta"] == pytest.approx(weight)
        assert row["verification_probability"] == pytest.approx(expected)
        assert row["relaxation_tv"] == pytest.approx(expected - 0.25)


def test_relaxation_depth_schedule_is_non_increasing() -> None:
    with pytest.raises(ValueError, match="non-increasing"):
        DynamicTreeConfig(
            relaxation_depth_weights=(1.0, 0.8, 0.9)
        ).validate()

    config = DynamicTreeConfig(
        relaxation_depth_weights=(1.0, 0.75, 0.5)
    )
    config.validate()
    assert config.relaxation_delta(1) == pytest.approx(1.0)
    assert config.relaxation_delta(2) == pytest.approx(0.75)
    assert config.relaxation_delta(4) == pytest.approx(0.5)


def test_adaptive_relaxation_recycles_only_saturation_waste() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (0, 0), (0, 0, 0)),
        name="adaptive-relaxation-chain",
    )
    draft_ids = torch.tensor([0, 0, 0])
    draft = torch.tensor(
        [
            [0.35, 0.65],
            [0.90, 0.10],
            [0.90, 0.10],
        ]
    )
    target = torch.tensor(
        [
            [0.30, 0.70],
            [0.05, 0.95],
            [0.10, 0.90],
            [0.50, 0.50],
        ]
    )
    config = _residual_hit_config(
        "residual_hit_anchor",
        cactus_delta=1.0,
        adaptive_relaxation=True,
        adaptive_relaxation_mix=0.5,
        adaptive_relaxation_max_ratio=1.5,
        adaptive_relaxation_target_power=0.25,
        adaptive_relaxation_steps=16,
    )

    plan = adaptive_trunk_relaxation_plan(
        topology,
        draft_ids,
        draft,
        target,
        config=config,
    )

    rows = [plan[index] for index in range(3)]
    assert rows[0]["adaptive_relaxation_tv"] == pytest.approx(0.05)
    assert rows[0]["baseline_relaxation_tv"] > 0.05
    assert sum(row["adaptive_relaxation_tv"] for row in rows) <= (
        rows[0]["adaptive_block_baseline_tv"] + 1e-7
    )
    assert any(
        row["adaptive_relaxation_tv"] > row["baseline_relaxation_tv"]
        for row in rows[1:]
    )
    for row in rows:
        assert row["adaptive_relaxation_tv"] <= (
            row["saturation_relaxation_tv"] + 1e-7
        )


def test_adaptive_relaxation_is_audited_on_residual_hit_trunk() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (0, 0), (0, 0, 0)),
        name="adaptive-relaxation-audit",
    )
    draft_ids = torch.tensor([0, 0, 0])
    draft = torch.tensor([[0.30, 0.70]]).repeat(3, 1)
    target = torch.tensor([[0.25, 0.75]]).repeat(4, 1)
    result = residual_hit_tree_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(4),
        config=_residual_hit_config(
            "residual_hit_anchor",
            cactus_delta=1.0,
            adaptive_relaxation=True,
        ),
    )

    rows = tuple(result.node_diagnostics)
    assert rows
    assert all(bool(row["adaptive_relaxation_enabled"]) for row in rows)
    assert all("adaptive_block_baseline_tv" in row for row in rows)
    assert all(
        float(row["adaptive_block_tv"])
        <= float(row["adaptive_block_baseline_tv"]) + 1e-7
        for row in rows
    )


def test_adaptive_relaxation_never_reduces_uniform_acceptance() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (0, 0), (0, 0, 0)),
        name="adaptive-relaxation-property",
    )
    generator = torch.Generator().manual_seed(20260819)
    config = _residual_hit_config(
        "residual_hit_anchor",
        cactus_delta=1.0,
        adaptive_relaxation=True,
    )
    for _ in range(32):
        draft = torch.rand((3, 7), generator=generator)
        draft /= draft.sum(dim=-1, keepdim=True)
        target = torch.rand((4, 7), generator=generator)
        target /= target.sum(dim=-1, keepdim=True)
        draft_ids = draft.argmax(dim=-1)
        plan = adaptive_trunk_relaxation_plan(
            topology,
            draft_ids,
            draft,
            target,
            config=config,
        )
        adaptive_total = 0.0
        baseline_total = 0.0
        for node in topology.nodes:
            row = plan[node.index]
            parent_row = 0 if node.parent is None else node.parent + 1
            token_id = int(draft_ids[node.index])
            p_y = float(target[parent_row, token_id])
            q_y = float(draft[node.index, token_id])
            baseline_acceptance = min(
                1.0,
                (p_y + row["baseline_relaxation_tv"])
                / max(q_y, 1e-30),
            )
            adaptive_acceptance = min(
                1.0,
                (p_y + row["adaptive_relaxation_tv"])
                / max(q_y, 1e-30),
            )
            assert adaptive_acceptance + 1e-7 >= baseline_acceptance
            assert row["adaptive_relaxation_tv"] <= (
                row["saturation_relaxation_tv"] + 1e-7
            )
            adaptive_total += row["adaptive_relaxation_tv"]
            baseline_total += row["baseline_relaxation_tv"]
        assert adaptive_total <= baseline_total + 1e-7


def test_cactus_trunk_rescue_falls_back_to_cactus_residual() -> None:
    topology = _cactus_caterpillar_topology()
    draft_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    draft = torch.tensor([[1.00, 0.00, 0.00, 0.00]]).repeat(6, 1)
    target = torch.full((7, 4), 0.25)
    target[0] = torch.tensor([0.00, 0.00, 1.00, 0.00])

    result = cactus_trunk_rescue_verify(
        topology,
        draft_ids,
        draft,
        target,
        torch.Generator().manual_seed(17),
        config=_cactus_rescue_config(rescue_min_target_prob=0.01),
    )
    assert result.accepted_node_indices == ()
    assert result.output_token_ids == (2,)
    assert result.terminal == "cactus-trunk-correction"
    assert dict(result.evaluated_node_statuses)[1] == "PRUNE"


def test_mtp_physical_routes_are_explicit() -> None:
    physical = [
        mtp_layer_for_tree_depth(depth, "physical_012")
        for depth in (1, 2, 3)
    ]
    repeated = [
        mtp_layer_for_tree_depth(depth, "repeat_000")
        for depth in (1, 2, 3)
    ]
    assert physical == [1, 2, 2]
    assert repeated == [0, 0, 0]


def test_dynamic_topology_keeps_bfs_parents_and_mask() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (1,), (0, 0), (0, 1), (1, 0), (0, 0, 0))
    )
    assert topology.choices == (
        (0,), (1,), (0, 0), (0, 1), (1, 0), (0, 0, 0)
    )
    assert topology.children_of(None) == (0, 1)
    assert topology.children_of(0) == (2, 3)
    assert topology.path_to(5) == (0, 2, 5)
    mask = topology.causal_tree_mask()
    assert mask[6].nonzero().flatten().tolist() == [0, 1, 3, 6]
    assert not bool(mask[6, 2])


def test_entropy_depth_threshold_widens_later() -> None:
    probs = torch.tensor([0.55, 0.25, 0.12, 0.08])
    config = DynamicTreeConfig(
        max_depth=3,
        max_nodes=16,
        min_draft_prob=0.01,
        threshold_sensitivity=1.0,
        depth_exploration=1.0,
    )
    early = select_dynamic_candidates(probs, depth=1, config=config)
    late = select_dynamic_candidates(probs, depth=3, config=config)
    assert late.threshold < early.threshold
    assert late.token_ids.numel() >= early.token_ids.numel()
    assert 0.0 < normalized_entropy(probs) < 1.0


def test_guarded_top2_adds_only_one_meaningful_sibling() -> None:
    config = DynamicTreeConfig(
        max_depth=3,
        max_nodes=8,
        max_children_per_parent=2,
        min_sibling_ratio=0.10,
        min_draft_prob=0.02,
        threshold_sensitivity=0.0,
    )
    selected = select_dynamic_candidates(
        torch.tensor([0.70, 0.20, 0.06, 0.04]), depth=1, config=config
    )
    assert selected.token_ids.tolist() == [0, 1]
    assert selected.threshold_candidate_count == 1
    assert selected.sibling_added is True


def test_guarded_topk_can_add_two_backups_without_forcing_budget_fill() -> None:
    config = DynamicTreeConfig(
        max_depth=3,
        max_nodes=10,
        max_children_per_parent=3,
        min_sibling_ratio=0.10,
        min_draft_prob=0.02,
        threshold_sensitivity=0.0,
    )
    selected = select_dynamic_candidates(
        torch.tensor([0.55, 0.25, 0.12, 0.08]), depth=1, config=config
    )
    assert selected.token_ids.tolist() == [0, 1, 2]
    assert selected.threshold_candidate_count == 1
    assert selected.candidate_count_before_budget == 3
    assert selected.sibling_added is True

    # The fourth candidate is above the probability guards but the explicit
    # per-parent cap prevents the proposal stage from filling arbitrary width.
    assert 3 not in selected.token_ids.tolist()


def test_guarded_top2_respects_absolute_floor_and_child_cap() -> None:
    config = DynamicTreeConfig(
        max_depth=3,
        max_nodes=8,
        max_children_per_parent=2,
        min_sibling_ratio=0.01,
        min_draft_prob=0.02,
        threshold_sensitivity=10.0,
    )
    selected = select_dynamic_candidates(
        torch.tensor([0.40, 0.30, 0.20, 0.10]), depth=3, config=config
    )
    assert selected.token_ids.tolist() == [0, 1]
    assert selected.candidate_count_before_budget == 4

    sharp = select_dynamic_candidates(
        torch.tensor([0.99, 0.009, 0.001]), depth=1, config=config
    )
    assert sharp.token_ids.tolist() == [0]
    assert sharp.sibling_added is False


def test_level_budget_preserves_one_slot_for_future_chain_depth() -> None:
    config = DynamicTreeConfig(max_depth=3, max_nodes=6)
    # Two root siblings leave four physical slots. At depth 2 only three may
    # be used, so at least one node remains for depth 3.
    assert level_expansion_budget(
        current_nodes=2, next_depth=2, config=config
    ) == 3
    assert level_expansion_budget(
        current_nodes=5, next_depth=3, config=config
    ) == 1


def test_reach_first_uses_prefix_mass_and_progressive_rank_tiers() -> None:
    legacy = DynamicTreeConfig(
        max_depth=3, max_nodes=6, allocation_mode="geometric"
    )
    reach_first = DynamicTreeConfig(
        max_depth=3, max_nodes=6, allocation_mode="reach_first"
    )
    low_reach_top1 = math.log(0.10) + math.log(0.80)
    high_reach_backup = math.log(0.80) + math.log(0.30)
    proposals = (
        (
            path_expansion_value(low_reach_top1, 2, reach_first),
            (1, 0),
        ),
        (
            path_expansion_value(high_reach_backup, 2, reach_first),
            (0, 1),
        ),
    )

    # Progressive widening protects the top-1 continuation even though a
    # rank-1 backup on another parent has greater raw prefix probability.
    assert level_expansion_order(proposals, config=reach_first) == (0, 1)
    assert level_expansion_order(proposals, config=legacy) == (1, 0)


def test_reach_first_penalizes_early_low_probability_paths_cumulatively() -> None:
    legacy = DynamicTreeConfig(
        max_depth=3, max_nodes=6, allocation_mode="geometric"
    )
    reach_first = DynamicTreeConfig(
        max_depth=3, max_nodes=6, allocation_mode="reach_first"
    )
    path_log = math.log(0.20) + math.log(0.20)
    legacy_value = path_expansion_value(path_log, 2, legacy)
    reach_value = path_expansion_value(path_log, 2, reach_first)
    assert reach_value < legacy_value


def test_soft_reach_has_no_hard_rank_or_parent_quota() -> None:
    config = DynamicTreeConfig(
        max_depth=3,
        max_nodes=6,
        allocation_mode="soft_reach",
        rank_penalty=2.0,
    )
    proposals = (
        (0.05, (1, 0)),
        (0.30, (0, 1)),
        (0.20, (2, 2)),
    )
    # After the soft penalty the rank-1 candidate has utility 0.10, which is
    # still greater than the weak rank-0 candidate's 0.05. Rank therefore
    # informs allocation but never establishes a mandatory tier or quota.
    assert level_expansion_order(proposals, config=config) == (1, 0, 2)


def test_residual_coverage_prioritizes_primary_siblings_then_continuations() -> None:
    config = DynamicTreeConfig(
        max_depth=3,
        max_nodes=15,
        allocation_mode="residual_coverage",
    )
    proposals = (
        (0.90, (0, 0)),
        (0.20, (0, 1)),
        (0.10, (0, 2)),
        (0.50, (1, 0)),
        (0.40, (2, 0)),
        (0.80, (1, 1)),
    )
    # Exact-correction coverage under the sampled primary prefix comes first.
    # Sampled continuations of existing backups follow, while a second backup
    # under a non-primary prefix is last even if its raw utility is high.
    assert level_expansion_order(proposals, config=config) == (
        0,
        1,
        2,
        3,
        4,
        5,
    )


def test_absolute_probability_floor_can_truncate_all_candidates() -> None:
    config = DynamicTreeConfig(
        max_depth=2,
        max_nodes=8,
        min_draft_prob=0.8,
        threshold_sensitivity=2.0,
    )
    selected = select_dynamic_candidates(
        torch.tensor([0.5, 0.3, 0.2]), depth=1, config=config
    )
    assert selected.token_ids.numel() == 0


def test_target_relaxation_keeps_multiple_paths_then_selects_long_path() -> None:
    topology = DynamicTreeTopology.from_paths(
        ((0,), (1,), (0, 0), (1, 0))
    )
    draft_ids = torch.tensor([0, 1, 2, 3])
    draft_probs = torch.full((4, 5), 0.2)
    target = torch.tensor(
        [
            [0.40, 0.30, 0.10, 0.10, 0.10],
            [0.05, 0.05, 0.80, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.80, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.80],
            [0.05, 0.05, 0.05, 0.05, 0.80],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=2,
        max_nodes=4,
        coverage_weight=0.5,
        relax_threshold=0.70,
        length_reward=1.0,
        path_temperature=0.0,
        append_target_anchor=True,
    )
    generator = torch.Generator().manual_seed(0)
    result = target_dominant_relaxed_verify(
        topology, draft_ids, draft_probs, target, generator, config=config
    )
    assert result.accepted_drafts == 2
    assert result.accepted_node_indices in {(0, 2), (1, 3)}
    assert len(result.output_token_ids) == 3
    assert result.terminal == "relaxed-bonus"
    assert result.surviving_paths == 2
    statuses = dict(result.evaluated_node_statuses)
    assert sum(value == "SELECTED" for value in statuses.values()) == 2
    assert sum(value == "SURVIVE" for value in statuses.values()) == 2


def test_target_relaxation_prefers_depth_before_confidence() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (1,), (1, 0)))
    draft_ids = torch.tensor([0, 1, 2])
    draft_probs = torch.full((3, 4), 0.25)
    target = torch.tensor(
        [
            [0.50, 0.45, 0.03, 0.02],
            [0.25, 0.25, 0.25, 0.25],
            [0.05, 0.05, 0.35, 0.55],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=2,
        max_nodes=3,
        coverage_mode="coverage_gate",
        min_target_coverage=0.0,
        relax_threshold=0.6,
        length_reward=0.0,
        path_temperature=0.0,
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft_probs,
        target,
        torch.Generator().manual_seed(0),
        config=config,
    )
    # Root token 0 has stronger local target support, but root token 1 has a
    # valid continuation.  Longest-prefix-first must commit the depth-2 path.
    assert result.accepted_node_indices == (1, 2)
    assert result.output_token_ids == (1, 2)
    assert result.surviving_paths == 2


def test_balanced_path_selection_can_stop_before_a_weak_tail() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (0, 0)))
    draft_ids = torch.tensor([0, 1])
    draft_probs = torch.full((2, 3), 1 / 3)
    target = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.64, 0.16, 0.20],
            [0.34, 0.33, 0.33],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=2,
        max_nodes=2,
        coverage_mode="coverage_gate",
        min_target_coverage=0.0,
        relax_threshold=0.1,
        length_reward=0.75,
        path_temperature=0.0,
        path_selection_mode="balanced",
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology, draft_ids, draft_probs, target, config=config
    )
    # Both tokens survive, but the second token has only 0.25 relative target
    # support.  C(path) * L**beta therefore prefers the safe one-token prefix.
    assert result.accepted_node_indices == (0,)
    assert result.output_token_ids == (0,)
    assert result.surviving_paths == 1


def test_dual_support_restores_target_over_proposal_candidate() -> None:
    topology = DynamicTreeTopology.from_paths(((0,),))
    draft_ids = torch.tensor([0])
    draft_probs = torch.tensor([[0.05, 0.50, 0.45]])
    target = torch.tensor(
        [
            [0.10, 0.50, 0.40],
            [0.20, 0.40, 0.40],
        ]
    )
    relative_only = DynamicTreeConfig(
        max_depth=1,
        max_nodes=1,
        coverage_mode="coverage_gate",
        relax_threshold=0.35,
        support_mode="relative",
        append_target_anchor=False,
    )
    dual = DynamicTreeConfig(
        max_depth=1,
        max_nodes=1,
        coverage_mode="coverage_gate",
        relax_threshold=0.35,
        support_mode="dual",
        proposal_support_ratio=1.0,
        append_target_anchor=False,
    )
    rejected = target_dominant_relaxed_verify(
        topology, draft_ids, draft_probs, target, config=relative_only
    )
    accepted = target_dominant_relaxed_verify(
        topology, draft_ids, draft_probs, target, config=dual
    )
    assert rejected.accepted_drafts == 0
    assert accepted.accepted_drafts == 1
    assert accepted.node_diagnostics[0]["proposal_supported"] is True
    assert accepted.node_diagnostics[0]["proposal_support_ratio"] == 2.0


def test_confirmed_support_uses_future_target_top1_without_forcing_length() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (0, 0)))
    draft_ids = torch.tensor([0, 2])
    draft_probs = torch.tensor(
        [
            [0.60, 0.30, 0.10],
            [0.10, 0.10, 0.80],
        ]
    )
    target = torch.tensor(
        [
            [0.10, 0.70, 0.20],
            [0.10, 0.10, 0.80],
            [0.30, 0.30, 0.40],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=2,
        max_nodes=2,
        coverage_mode="coverage_gate",
        relax_threshold=0.35,
        support_mode="confirmed",
        confirmation_min_relative=0.05,
        path_selection_mode="balanced",
        length_reward=0.0,
        path_temperature=0.0,
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology, draft_ids, draft_probs, target, config=config
    )
    root = result.node_diagnostics[0]
    assert root["relative_target_support"] < config.relax_threshold
    assert root["proposal_supported"] is False
    assert root["future_top1_match"] is True
    assert root["target_confirmed"] is True
    assert result.accepted_drafts == 2


def test_cactus_support_matches_chain_relaxation_strength() -> None:
    topology = DynamicTreeTopology.from_paths(((0,),))
    draft_ids = torch.tensor([0])
    draft_probs = torch.tensor([[0.10, 0.90]])
    target = torch.tensor(
        [
            [0.01, 0.99],
            [0.50, 0.50],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=1,
        max_nodes=1,
        coverage_mode="coverage_gate",
        min_target_coverage=0.0,
        relax_threshold=0.99,
        support_mode="cactus",
        cactus_delta=1.0,
        cactus_target_weight=0.25,
        path_selection_mode="balanced",
        path_temperature=0.0,
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology, draft_ids, draft_probs, target, config=config
    )
    row = result.node_diagnostics[0]
    expected_h = 0.01 + math.sqrt(2.0 * 0.01 * 0.99)
    assert row["relative_target_support"] < config.relax_threshold
    assert math.isclose(
        row["cactus_boosted_probability"], expected_h, rel_tol=1e-6
    )
    assert math.isclose(
        row["cactus_acceptance_probability"], 1.0, rel_tol=1e-6
    )
    assert row["cactus_survival"] is True
    assert result.accepted_drafts == 1


def test_cactus_guided_rescues_target_supported_dead_frontier() -> None:
    topology = DynamicTreeTopology.from_paths(((0,),))
    draft_ids = torch.tensor([0])
    draft_probs = torch.tensor([[0.10, 0.90]])
    target = torch.tensor(
        [
            [0.20, 0.80],
            [0.50, 0.50],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=1,
        max_nodes=1,
        coverage_mode="coverage_gate",
        min_target_coverage=0.05,
        relax_threshold=0.60,
        support_mode="cactus_guided",
        proposal_support_ratio=1.0,
        confirmation_min_relative=0.05,
        cactus_delta=1.0,
        cactus_target_weight=0.65,
        path_selection_mode="balanced",
        path_temperature=0.0,
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft_probs,
        target,
        torch.Generator().manual_seed(0),
        config=config,
    )
    row = result.node_diagnostics[0]
    assert row["relative_target_support"] < config.relax_threshold
    assert row["proposal_support_ratio"] > config.proposal_support_ratio
    assert row["cactus_guided_eligible"] is True
    assert row["cactus_guided_rescued"] is True
    assert result.accepted_drafts == 1
    assert result.terminal == "cactus-guided-path"


def test_cactus_guided_does_not_rescue_without_tree_evidence() -> None:
    topology = DynamicTreeTopology.from_paths(((0,),))
    draft_ids = torch.tensor([0])
    draft_probs = torch.tensor([[0.90, 0.10]])
    target = torch.tensor(
        [
            [0.20, 0.80],
            [0.50, 0.50],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=1,
        max_nodes=1,
        coverage_mode="coverage_gate",
        min_target_coverage=0.05,
        relax_threshold=0.60,
        support_mode="cactus_guided",
        proposal_support_ratio=1.0,
        confirmation_min_relative=0.05,
        cactus_delta=1.0,
        path_selection_mode="balanced",
        path_temperature=0.0,
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft_probs,
        target,
        torch.Generator().manual_seed(0),
        config=config,
    )
    row = result.node_diagnostics[0]
    assert row["cactus_acceptance_probability"] > 0.80
    assert row["cactus_guided_eligible"] is False
    assert result.accepted_drafts == 0


def test_cactus_guided_is_used_only_when_normal_frontier_is_dead() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (1,)))
    draft_ids = torch.tensor([0, 1])
    draft_probs = torch.tensor(
        [
            [0.50, 0.50],
            [0.10, 0.90],
        ]
    )
    target = torch.tensor(
        [
            [0.80, 0.20],
            [0.50, 0.50],
            [0.50, 0.50],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=1,
        max_nodes=2,
        coverage_mode="coverage_gate",
        min_target_coverage=0.05,
        relax_threshold=0.60,
        support_mode="cactus_guided",
        proposal_support_ratio=1.0,
        confirmation_min_relative=0.05,
        cactus_delta=1.0,
        path_selection_mode="balanced",
        path_temperature=0.0,
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft_probs,
        target,
        torch.Generator().manual_seed(0),
        config=config,
    )
    rows = {int(row["node"]): row for row in result.node_diagnostics}
    assert rows[0]["normal_survival"] is True
    assert rows[1]["cactus_guided_eligible"] is False
    assert result.accepted_node_indices == (0,)


def test_cactus_guided_allows_at_most_one_rescue_per_path() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (0, 0)))
    draft_ids = torch.tensor([0, 1])
    draft_probs = torch.tensor(
        [
            [0.10, 0.90],
            [0.90, 0.10],
        ]
    )
    target = torch.tensor(
        [
            [0.20, 0.80],
            [0.80, 0.20],
            [0.50, 0.50],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=2,
        max_nodes=2,
        coverage_mode="coverage_gate",
        min_target_coverage=0.05,
        relax_threshold=0.60,
        support_mode="cactus_guided",
        proposal_support_ratio=1.0,
        confirmation_min_relative=0.05,
        cactus_delta=1.0,
        path_selection_mode="balanced",
        path_temperature=0.0,
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft_probs,
        target,
        torch.Generator().manual_seed(0),
        config=config,
    )
    rows = {int(row["node"]): row for row in result.node_diagnostics}
    assert rows[0]["cactus_guided_rescued"] is True
    assert rows[1]["cactus_guided_rescued"] is False
    assert result.accepted_node_indices == (0,)


def test_cactus_guided_can_spend_configured_second_rescue() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (0, 0)))
    draft_ids = torch.tensor([0, 1])
    draft_probs = torch.tensor(
        [
            [0.10, 0.90],
            [0.90, 0.10],
        ]
    )
    target = torch.tensor(
        [
            [0.20, 0.80],
            [0.80, 0.20],
            [0.50, 0.50],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=2,
        max_nodes=2,
        coverage_mode="coverage_gate",
        min_target_coverage=0.01,
        relax_threshold=0.60,
        support_mode="cactus_guided",
        proposal_support_ratio=0.25,
        confirmation_min_relative=0.01,
        cactus_delta=1.0,
        cactus_target_weight=0.50,
        max_guided_rescues_per_path=2,
        path_selection_mode="balanced",
        path_temperature=0.0,
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft_probs,
        target,
        torch.Generator().manual_seed(0),
        config=config,
    )
    rows = {int(row["node"]): row for row in result.node_diagnostics}
    assert rows[0]["guided_rescue_count_before"] == 0
    assert rows[1]["guided_rescue_count_before"] == 1
    assert rows[0]["cactus_guided_rescued"] is True
    assert rows[1]["cactus_guided_rescued"] is True
    assert result.accepted_node_indices == (0, 1)


def test_target_relaxation_falls_back_when_every_candidate_is_pruned() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (1,)))
    draft_ids = torch.tensor([0, 1])
    draft_probs = torch.full((2, 3), 1 / 3)
    target = torch.tensor(
        [
            [0.01, 0.01, 0.98],
            [0.10, 0.10, 0.80],
            [0.10, 0.10, 0.80],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=1,
        max_nodes=2,
        coverage_weight=0.5,
        relax_threshold=0.5,
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft_probs,
        target,
        torch.Generator().manual_seed(0),
        config=config,
    )
    assert result.accepted_drafts == 0
    assert result.output_token_ids == (2,)
    assert result.terminal == "target-fallback"


def test_coverage_gate_does_not_reward_wider_sibling_set() -> None:
    config = DynamicTreeConfig(
        max_depth=1,
        max_nodes=3,
        coverage_mode="coverage_gate",
        min_target_coverage=0.1,
        relax_threshold=0.6,
        append_target_anchor=False,
    )
    target = torch.tensor(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    narrow = DynamicTreeTopology.from_paths(((0,),))
    wide = DynamicTreeTopology.from_paths(((0,), (1,), (2,)))
    narrow_result = target_dominant_relaxed_verify(
        narrow,
        torch.tensor([2]),
        torch.full((1, 4), 0.25),
        target[:2],
        config=config,
    )
    wide_result = target_dominant_relaxed_verify(
        wide,
        torch.tensor([2, 1, 3]),
        torch.full((3, 4), 0.25),
        target,
        config=config,
    )
    narrow_score = narrow_result.node_diagnostics[0]["relaxed_confidence"]
    wide_score = wide_result.node_diagnostics[0]["relaxed_confidence"]
    assert abs(narrow_score - wide_score) < 1e-7
    assert narrow_result.accepted_drafts == 0
    assert 0 not in wide_result.accepted_node_indices


def test_eos_protection_prunes_non_eos_when_target_wants_to_stop() -> None:
    topology = DynamicTreeTopology.from_paths(((0,),))
    draft_ids = torch.tensor([0])
    draft_probs = torch.tensor([[0.5, 0.1, 0.4]])
    target = torch.tensor(
        [
            [0.30, 0.10, 0.60],
            [0.10, 0.10, 0.80],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=1,
        max_nodes=1,
        coverage_weight=0.0,
        relax_threshold=0.4,
        eos_token_ids=(2,),
        eos_protection_threshold=0.5,
    )
    generator = torch.Generator().manual_seed(0)
    result = target_dominant_relaxed_verify(
        topology, draft_ids, draft_probs, target, generator, config=config
    )
    assert result.accepted_drafts == 0
    assert result.output_token_ids == (2,)
    assert result.node_diagnostics[0]["eos_protected"] is True
    eos_probability = result.node_diagnostics[0]["target_eos_probability"]
    assert abs(eos_probability - 0.6) < 1e-6


def test_eos_candidate_is_never_vetoed_by_eos_protection() -> None:
    topology = DynamicTreeTopology.from_paths(((0,),))
    draft_ids = torch.tensor([2])
    draft_probs = torch.tensor([[0.1, 0.1, 0.8]])
    target = torch.tensor(
        [
            [0.20, 0.10, 0.70],
            [0.10, 0.10, 0.80],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=1,
        max_nodes=1,
        coverage_weight=0.0,
        relax_threshold=0.5,
        eos_token_ids=(2,),
        eos_protection_threshold=0.5,
        append_target_anchor=False,
    )
    result = target_dominant_relaxed_verify(
        topology, draft_ids, draft_probs, target, config=config
    )
    assert result.accepted_drafts == 1
    assert result.output_token_ids == (2,)
    assert result.node_diagnostics[0]["eos_protected"] is False


def test_frontier_rescue_extends_a_dead_root_once() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (0, 0), (0, 0, 0)))
    draft_ids = torch.tensor([0, 1, 2])
    draft_probs = torch.tensor(
        [
            [0.50, 0.30, 0.20],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ]
    )
    target = torch.tensor(
        [
            [0.20, 0.50, 0.30],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.20, 0.30, 0.50],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=3,
        max_nodes=3,
        coverage_mode="coverage_gate",
        relax_threshold=0.6,
        frontier_rescue=True,
        rescue_delta=0.5,
        rescue_min_relative=0.1,
        rescue_min_target_prob=0.001,
        append_target_anchor=False,
        path_temperature=0.0,
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft_probs,
        target,
        torch.Generator().manual_seed(0),
        config=config,
    )
    assert result.accepted_node_indices == (0, 1, 2)
    assert result.terminal == "frontier-rescue-path"
    assert result.node_diagnostics[0]["rescued"] is True
    assert result.node_diagnostics[1]["rescued"] is False
    assert result.node_diagnostics[2]["rescued"] is False


def test_frontier_rescue_cannot_be_used_twice_on_one_path() -> None:
    topology = DynamicTreeTopology.from_paths(((0,), (0, 0)))
    draft_ids = torch.tensor([0, 1])
    draft_probs = torch.tensor(
        [
            [0.50, 0.30, 0.20],
            [0.30, 0.50, 0.20],
        ]
    )
    target = torch.tensor(
        [
            [0.20, 0.50, 0.30],
            [0.50, 0.20, 0.30],
            [0.20, 0.30, 0.50],
        ]
    )
    config = DynamicTreeConfig(
        max_depth=2,
        max_nodes=2,
        coverage_mode="coverage_gate",
        relax_threshold=0.6,
        frontier_rescue=True,
        rescue_delta=0.5,
        append_target_anchor=False,
        path_temperature=0.0,
    )
    result = target_dominant_relaxed_verify(
        topology,
        draft_ids,
        draft_probs,
        target,
        torch.Generator().manual_seed(0),
        config=config,
    )
    assert result.accepted_node_indices == (0,)
    assert result.terminal == "frontier-rescue-path"
    assert result.node_diagnostics[0]["rescued"] is True
    assert result.node_diagnostics[1]["rescue_eligible"] is False
