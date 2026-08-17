from __future__ import annotations

import torch

from remtp.risk_entropy_mtp import (
    RiskEntropyConfig,
    _STATE,
    _begin_round,
    risk_entropy_distribution,
)


def _case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_logits = torch.tensor(
        [
            [2.0, 1.8, 0.2, -1.0],
            [2.2, 1.7, 0.4, -1.0],
            [2.0, 1.9, 0.1, -1.0],
            [2.5, 0.5, 0.2, -1.0],
        ],
        dtype=torch.float32,
    )
    target = target_logits.softmax(dim=-1)
    draft = torch.tensor(
        [
            [0.35, 0.55, 0.08, 0.02],
            [0.42, 0.48, 0.08, 0.02],
            [0.36, 0.54, 0.08, 0.02],
            [0.50, 0.30, 0.15, 0.05],
        ],
        dtype=torch.float32,
    )
    ids = torch.tensor([1, 1, 1, 0], dtype=torch.int64)
    return target, draft, target_logits, ids


def test_async_request_id_preserves_debt_within_request_only() -> None:
    class EmptySamplingMetadata:
        output_token_ids: list[list[int]] = []

    _STATE.reset()
    try:
        _STATE.active_request_id = "request-a"
        first = _begin_round(EmptySamplingMetadata(), torch.device("cpu"))
        assert first == 0
        _STATE.debt = torch.tensor(1.25)
        same = _begin_round(EmptySamplingMetadata(), torch.device("cpu"))
        assert same == 1.25

        _STATE.active_request_id = "request-b"
        next_request = _begin_round(
            EmptySamplingMetadata(), torch.device("cpu")
        )
        assert next_request == 0
    finally:
        _STATE.reset()


def test_scheme1_constructs_normalized_relaxed_distribution() -> None:
    target, draft, logits, ids = _case()
    config = RiskEntropyConfig(
        variant="scheme1",
        expected_draft_tokens=4,
        max_target_rank=2,
        top_m=2,
        base_log_gap=1.0,
        risk_budget=4.0,
        per_token_tv_cap=0.20,
        block_tv_cap=0.50,
    )
    result = risk_entropy_distribution(target, draft, logits, ids, config)

    assert torch.allclose(result.probs.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.all(result.boosted_candidate_probs >= result.target_candidate_probs)
    assert result.allocated_tv.sum() <= config.block_tv_cap + 1e-6
    assert torch.all(result.relaxed_acceptance >= result.strict_acceptance)


def test_saturated_target_probability_is_never_reduced() -> None:
    target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    draft = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    logits = torch.tensor([[100.0, -100.0]], dtype=torch.float32)
    result = risk_entropy_distribution(
        target,
        draft,
        logits,
        torch.tensor([0]),
        RiskEntropyConfig(
            variant="scheme2_relaxed",
            expected_draft_tokens=1,
            max_target_rank=2,
            scheme2_rank_limit=2,
            adaptive_draft_depth=False,
            medium_draft_depth=1,
            short_draft_depth=1,
        ),
    )

    assert result.boosted_candidate_probs[0] == 1.0
    assert result.relaxed_acceptance[0] == result.strict_acceptance[0]
    assert torch.allclose(result.probs, target)


def test_cumulative_risk_stops_relaxation_prefix() -> None:
    target, draft, logits, ids = _case()
    config = RiskEntropyConfig(
        variant="scheme1",
        expected_draft_tokens=4,
        max_target_rank=2,
        top_m=2,
        base_log_gap=2.0,
        risk_budget=0.25,
        per_token_tv_cap=0.20,
        block_tv_cap=0.80,
    )
    result = risk_entropy_distribution(target, draft, logits, ids, config)

    failures = (~result.relaxed_prefix).nonzero().flatten()
    assert failures.numel() > 0
    first = int(failures[0])
    assert torch.all(result.allocated_tv[first:] == 0)


def test_scheme2_future_disagreement_vetoes_current_relaxation() -> None:
    target, draft, logits, ids = _case()
    # Force Q at the second position to disagree with target top-2 and make
    # the actual next candidate low confidence. Only one sentinel check passes.
    draft[1] = torch.tensor([0.01, 0.01, 0.01, 0.97])
    config = RiskEntropyConfig(
        variant="scheme2",
        expected_draft_tokens=4,
        max_target_rank=2,
        top_m=2,
        base_log_gap=2.0,
        risk_budget=8.0,
        sentinel_log_gap=2.0,
        sentinel_mtp_floor=0.05,
        sentinel_min_checks=2,
    )
    result = risk_entropy_distribution(target, draft, logits, ids, config)

    assert not bool(result.sentinel_pass[0])
    assert result.allocated_tv[0] == 0


def test_cross_round_debt_tightens_scheme12() -> None:
    target, draft, logits, ids = _case()
    config = RiskEntropyConfig(
        variant="scheme12",
        expected_draft_tokens=4,
        max_target_rank=2,
        top_m=2,
        base_log_gap=1.5,
        risk_budget=4.0,
        block_tv_cap=0.8,
        sentinel_log_gap=3.0,
        sentinel_min_checks=1,
        debt_scale=1.0,
    )
    no_debt = risk_entropy_distribution(
        target, draft, logits, ids, config, previous_debt=0.0
    )
    high_debt = risk_entropy_distribution(
        target, draft, logits, ids, config, previous_debt=3.0
    )

    assert torch.all(high_debt.threshold <= no_debt.threshold)
    assert high_debt.allocated_tv.sum() <= no_debt.allocated_tv.sum() + 1e-6


def test_relaxed_scheme2_is_not_limited_by_scheme1_cumulative_risk() -> None:
    target, draft, logits, ids = _case()
    shared = dict(
        expected_draft_tokens=4,
        max_target_rank=4,
        scheme2_rank_limit=4,
        top_m=2,
        base_log_gap=2.0,
        risk_budget=0.01,
        per_token_tv_cap=0.20,
        block_tv_cap=0.80,
        sentinel_log_gap=3.0,
        sentinel_min_checks=1,
        last_token_gap_scale=1.0,
        adaptive_draft_depth=False,
    )
    combined = risk_entropy_distribution(
        target,
        draft,
        logits,
        ids,
        RiskEntropyConfig(variant="scheme12", **shared),
    )
    relaxed_scheme2 = risk_entropy_distribution(
        target,
        draft,
        logits,
        ids,
        RiskEntropyConfig(
            variant="scheme2_relaxed",
            sentinel_soft_floor=1.0,
            **shared,
        ),
    )

    assert combined.allocated_tv.sum() == 0
    assert relaxed_scheme2.allocated_tv.sum() > 0


def test_relaxed_scheme2_soft_sentinel_scales_instead_of_binary_veto() -> None:
    target, draft, logits, ids = _case()
    # At the next position only candidate confidence passes: target-head mass
    # and next target gap fail. One check is enough to continue, but it should
    # receive less TV than an otherwise identical full-strength sentinel.
    draft[1] = torch.tensor([0.001, 0.02, 0.001, 0.978])
    common = dict(
        variant="scheme2_relaxed",
        expected_draft_tokens=4,
        max_target_rank=4,
        scheme2_rank_limit=4,
        top_m=2,
        base_log_gap=2.0,
        risk_budget=8.0,
        per_token_tv_cap=0.20,
        block_tv_cap=0.80,
        sentinel_log_gap=0.01,
        sentinel_mtp_floor=0.01,
        sentinel_head_mass_floor=0.90,
        sentinel_min_checks=1,
        last_token_gap_scale=1.0,
        adaptive_draft_depth=False,
    )
    soft = risk_entropy_distribution(
        target,
        draft,
        logits,
        ids,
        RiskEntropyConfig(sentinel_soft_floor=0.25, **common),
    )
    full = risk_entropy_distribution(
        target,
        draft,
        logits,
        ids,
        RiskEntropyConfig(sentinel_soft_floor=1.0, **common),
    )

    assert int(soft.sentinel_checks[0]) == 1
    assert bool(soft.sentinel_pass[0])
    assert 0 < soft.allocated_tv[0] < full.allocated_tv[0]


def test_remtp_target_margin_tightens_decisive_positions() -> None:
    logits = torch.tensor(
        [
            [2.0, 1.9, 0.0, -1.0],
            [3.0, 1.5, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    target = logits.softmax(dim=-1)
    draft = torch.tensor(
        [
            [0.35, 0.55, 0.08, 0.02],
            [0.25, 0.65, 0.08, 0.02],
        ],
        dtype=torch.float32,
    )
    config = RiskEntropyConfig(
        variant="remtp",
        expected_draft_tokens=2,
        max_target_rank=4,
        scheme2_rank_limit=4,
        base_log_gap=2.0,
        remtp_gap_floor=0.75,
        remtp_margin_temperature=0.50,
        remtp_risky_rank=4,
        remtp_risky_min_checks=1,
        sentinel_log_gap=3.0,
        sentinel_min_checks=1,
        last_token_gap_scale=1.0,
        adaptive_draft_depth=False,
        medium_draft_depth=2,
        short_draft_depth=1,
    )
    result = risk_entropy_distribution(
        target,
        draft,
        logits,
        torch.tensor([1, 1]),
        config,
    )

    assert result.target_margin[0] < result.target_margin[1]
    assert result.threshold[0] > result.threshold[1]
    assert bool(result.local_eligible[0])
    assert not bool(result.local_eligible[1])


def test_remtp_reuses_block_budget_after_an_ineligible_head() -> None:
    target, draft, logits, ids = _case()
    ids = ids.clone()
    ids[0] = 2
    draft[0] = torch.tensor([0.01, 0.01, 0.95, 0.03])
    config = RiskEntropyConfig(
        variant="remtp",
        expected_draft_tokens=4,
        max_target_rank=4,
        scheme2_rank_limit=2,
        top_m=2,
        base_log_gap=2.5,
        remtp_gap_floor=2.5,
        remtp_risky_rank=2,
        remtp_risky_min_checks=1,
        per_token_tv_cap=0.30,
        block_tv_cap=0.80,
        risk_budget=4.0,
        sentinel_log_gap=3.0,
        sentinel_min_checks=1,
        sentinel_soft_floor=0.50,
        last_token_gap_scale=1.0,
        adaptive_draft_depth=False,
    )
    result = risk_entropy_distribution(target, draft, logits, ids, config)

    assert not bool(result.local_eligible[0])
    assert result.allocated_tv[0] == 0
    assert bool(result.local_eligible[1])
    assert result.allocated_tv[1] > 0
    assert result.allocated_tv.sum() <= config.block_tv_cap + 1e-6
    assert torch.allclose(result.probs.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_remtp_block_rescues_only_target_supported_prefixes() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    target = logits.softmax(dim=-1)
    draft = torch.tensor(
        [
            [0.10, 0.85, 0.05],
            [0.70, 0.20, 0.10],
            [0.70, 0.20, 0.10],
        ],
        dtype=torch.float32,
    )
    ids = torch.tensor([1, 0, 0], dtype=torch.int64)
    common = dict(
        expected_draft_tokens=3,
        max_target_rank=3,
        scheme2_rank_limit=3,
        base_log_gap=1.0,
        remtp_gap_floor=0.5,
        remtp_risky_rank=2,
        remtp_risky_min_checks=1,
        remtp_continuation_gap=2.5,
        remtp_continuation_support=0.65,
        remtp_continuation_regret=0.5,
        remtp_continuation_horizon=2,
        sentinel_log_gap=3.0,
        sentinel_min_checks=1,
        last_token_gap_scale=1.0,
        risk_budget=10.0,
        block_tv_cap=1.0,
        per_token_tv_cap=0.5,
        adaptive_draft_depth=False,
        medium_draft_depth=2,
        short_draft_depth=1,
    )
    token_local = risk_entropy_distribution(
        target,
        draft,
        logits,
        ids,
        RiskEntropyConfig(variant="remtp", **common),
    )
    block = risk_entropy_distribution(
        target,
        draft,
        logits,
        ids,
        RiskEntropyConfig(variant="remtp_block", **common),
    )

    assert not bool(token_local.local_eligible[0])
    assert token_local.allocated_tv[0] == 0
    assert bool(block.positive_certificate[0])
    assert block.continuation_support[0] == 1
    assert block.continuation_regret[0] == 0
    assert block.allocated_tv[0] > 0
    assert not bool(block.positive_certificate[-1])


def test_joint_scheme12_respects_tv_and_risk_mass_budgets() -> None:
    target, draft, logits, ids = _case()
    config = RiskEntropyConfig(
        variant="scheme12_joint",
        expected_draft_tokens=4,
        max_target_rank=4,
        top_m=2,
        base_log_gap=2.0,
        risk_budget=8.0,
        per_token_tv_cap=0.30,
        block_tv_cap=0.55,
        joint_risk_mass_cap=0.08,
        sentinel_log_gap=3.0,
        sentinel_min_checks=1,
        sentinel_soft_floor=0.40,
        last_token_gap_scale=1.0,
        adaptive_draft_depth=False,
    )
    result = risk_entropy_distribution(target, draft, logits, ids, config)

    assert torch.allclose(result.probs.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert result.allocated_tv.sum() <= config.block_tv_cap + 1e-6
    assert result.allocated_risk_mass.sum() <= config.joint_risk_mass_cap + 1e-6
    assert torch.any(result.allocation_utility > 0)
    assert torch.all(result.relaxed_acceptance >= result.strict_acceptance)


def test_joint_scheme12_keeps_later_relaxation_after_unsafe_position() -> None:
    target, draft, logits, ids = _case()
    ids = ids.clone()
    ids[0] = 2
    draft[0] = torch.tensor([0.01, 0.01, 0.95, 0.03])
    config = RiskEntropyConfig(
        variant="scheme12_joint",
        expected_draft_tokens=4,
        max_target_rank=4,
        top_m=2,
        base_log_gap=0.60,
        ambiguity_scale=0.60,
        threshold_position_scale=0.10,
        rank_position_scale=0.0,
        per_token_tv_cap=0.30,
        block_tv_cap=0.80,
        joint_risk_mass_cap=1.0,
        sentinel_log_gap=3.0,
        sentinel_min_checks=1,
        sentinel_soft_floor=0.40,
        last_token_gap_scale=1.0,
        adaptive_draft_depth=False,
    )
    result = risk_entropy_distribution(target, draft, logits, ids, config)

    assert not bool(result.local_eligible[0])
    assert result.allocated_tv[0] == 0
    assert bool(result.local_eligible[1])
    assert result.allocated_tv[1] > 0


def test_anchored_joint_reserves_most_budget_for_target_core() -> None:
    target, draft, logits, ids = _case()
    ids = ids.clone()
    ids[0] = 2
    draft[0] = torch.tensor([0.01, 0.01, 0.95, 0.03])
    config = RiskEntropyConfig(
        variant="scheme12_anchored",
        expected_draft_tokens=4,
        max_target_rank=4,
        top_m=2,
        base_log_gap=2.5,
        ambiguity_scale=0.0,
        threshold_position_scale=0.0,
        rank_position_scale=0.0,
        per_token_tv_cap=0.50,
        block_tv_cap=1.0,
        joint_risk_mass_cap=1.0,
        joint_core_rank=2,
        joint_core_log_gap=0.75,
        joint_core_fraction=0.80,
        joint_explore_min_checks=1,
        joint_support_power=1.0,
        joint_explore_risk_multiplier=2.0,
        sentinel_log_gap=3.0,
        sentinel_min_checks=1,
        sentinel_soft_floor=0.50,
        last_token_gap_scale=1.0,
        adaptive_draft_depth=False,
    )
    result = risk_entropy_distribution(target, draft, logits, ids, config)

    assert result.core_allocated_tv[0] == 0
    assert result.allocated_tv[0] <= (
        config.block_tv_cap * (1.0 - config.joint_core_fraction) + 1e-6
    )
    assert result.core_allocated_tv.sum() <= (
        config.block_tv_cap * config.joint_core_fraction + 1e-6
    )
    assert result.allocated_tv.sum() <= config.block_tv_cap + 1e-6
    assert result.allocated_risk_mass.sum() <= config.joint_risk_mass_cap + 1e-6
