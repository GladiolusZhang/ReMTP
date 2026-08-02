"""Target-anchored block relaxation for native probabilistic MTP.

The method removes vocabulary-wide P/Q agreement signals.  It spends the
exact candidate-probability mass (and therefore exact total-variation mass)
that per-position Cactus would request, but stops boosting a candidate once
``h(y) == q(y)`` because its speculative acceptance is already saturated.

Three ablations are implemented:

* ``cactus_cap``: independent Cactus boosts capped at ``q(y)``;
* ``tv_head``: redistribute Cactus' block TV using target support and a
  calibrated reliability prior for each MTP head;
* ``tv_hidden_veto``: additionally use aligned current MTP/target hidden
  cosine and allow later target support only to veto, never reward, an
  earlier candidate.
* ``tv_debt_control``: keep saturation-aware block-TV recycling, but treat
  the increase in speculative acceptance probability as verification debt.
  Per-position risk caps and a cumulative block limit act before a risky
  relaxed token can be committed.
* ``tv_top1_surplus``: strict ablation that spends saturated Cactus surplus
  only where the MTP candidate is the target top-1.
* ``tv_target_surplus``: practical variant that spends the same surplus in
  prefix order on target-supported candidates within one relative log gap.
* ``tv_risk_swap``: preserve Cactus on target-supported positions, prune its
  risky tail continuously, and spend both the pruned mass and saturation
  surplus where it has higher target-anchored prefix utility.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


_V1_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False
_HIDDEN_FALLBACK_EMITTED = False
_AUDIT_ROUND = 0
_AUDIT_TV = torch.zeros(3, dtype=torch.float64)
_AUDIT_DEBT = torch.zeros(6, dtype=torch.float64)
_AUDIT_SURPLUS = torch.zeros(4, dtype=torch.float64)
_AUDIT_ELIGIBILITY = torch.zeros(10, dtype=torch.float64)
_DEFAULT_HEAD_RELIABILITY = (1.0, 0.85, 0.70, 0.55, 0.40, 0.30)
_PRE_VERIFICATION_HOOK: Any | None = None
_POST_VERIFICATION_HOOK: Any | None = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _parse_head_reliability(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(","))
    if not result or any(not 0.0 <= item <= 1.0 for item in result):
        raise ValueError(
            "REMTP_TA_HEAD_RELIABILITY must be comma-separated values in [0,1]"
        )
    return result


@dataclass(frozen=True)
class TargetAnchoredConfig:
    """Configuration for target-anchored exact-TV redistribution."""

    variant: str = "tv_hidden_veto"
    cactus_delta: float = 1.0
    expected_draft_tokens: int = 6
    head_reliability: tuple[float, ...] = _DEFAULT_HEAD_RELIABILITY
    target_log_gap_scale: float = 2.0
    max_target_log_gap: float = 8.0
    future_veto_floor: float = 0.20
    hidden_reliability_floor: float = 0.25
    use_prefix_value: bool = True
    debt_position_limit: float = 0.35
    debt_block_limit: float = 1.20
    debt_soft_log_gap: float = 2.0
    debt_hard_log_gap: float = 6.0
    debt_max_position_tv: float = 0.15
    debt_max_cactus_ratio: float = 2.0
    debt_fallback: str = "strict"
    surplus_max_log_gap: float = 2.0
    risk_swap_soft_log_gap: float = 4.0
    risk_swap_hard_log_gap: float = 10.0
    risk_swap_destination_log_gap: float = 2.0
    recovery_mode: str = "residual"

    def validate(self) -> None:
        if self.variant not in {
            "cactus_cap",
            "tv_head",
            "tv_hidden_veto",
            "tv_debt_control",
            "tv_top1_surplus",
            "tv_target_surplus",
            "tv_risk_swap",
        }:
            raise ValueError(f"unsupported target-anchored variant: {self.variant}")
        if self.cactus_delta < 0:
            raise ValueError("cactus_delta must be non-negative")
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if len(self.head_reliability) < self.expected_draft_tokens:
            raise ValueError(
                "head_reliability must cover every expected MTP position"
            )
        if any(not 0.0 <= item <= 1.0 for item in self.head_reliability):
            raise ValueError("head reliability values must be in [0,1]")
        if self.target_log_gap_scale <= 0:
            raise ValueError("target_log_gap_scale must be positive")
        if self.max_target_log_gap <= 0:
            raise ValueError("max_target_log_gap must be positive")
        if not 0.0 <= self.future_veto_floor <= 1.0:
            raise ValueError("future_veto_floor must be in [0,1]")
        if not 0.0 <= self.hidden_reliability_floor <= 1.0:
            raise ValueError("hidden_reliability_floor must be in [0,1]")
        if not 0.0 <= self.debt_position_limit <= 1.0:
            raise ValueError("debt_position_limit must be in [0,1]")
        if self.debt_block_limit < 0.0:
            raise ValueError("debt_block_limit must be non-negative")
        if self.debt_soft_log_gap < 0.0:
            raise ValueError("debt_soft_log_gap must be non-negative")
        if self.debt_hard_log_gap <= self.debt_soft_log_gap:
            raise ValueError(
                "debt_hard_log_gap must exceed debt_soft_log_gap"
            )
        if self.debt_max_position_tv < 0.0:
            raise ValueError("debt_max_position_tv must be non-negative")
        if self.debt_max_cactus_ratio < 1.0:
            raise ValueError("debt_max_cactus_ratio must be at least one")
        if self.debt_fallback not in {"strict", "cactus_cap"}:
            raise ValueError(
                "debt_fallback must be strict or cactus_cap"
            )
        if self.surplus_max_log_gap < 0.0:
            raise ValueError("surplus_max_log_gap must be non-negative")
        if self.risk_swap_soft_log_gap < 0.0:
            raise ValueError("risk_swap_soft_log_gap must be non-negative")
        if self.risk_swap_hard_log_gap <= self.risk_swap_soft_log_gap:
            raise ValueError(
                "risk_swap_hard_log_gap must exceed the soft gap"
            )
        if self.risk_swap_destination_log_gap < 0.0:
            raise ValueError(
                "risk_swap_destination_log_gap must be non-negative"
            )
        if self.recovery_mode not in {"residual", "target"}:
            raise ValueError("recovery_mode must be residual or target")

    @classmethod
    def from_env(cls) -> TargetAnchoredConfig:
        expected = int(os.getenv("REMTP_TA_EXPECTED_DRAFT_TOKENS", "6"))
        default_heads = ",".join(
            str(value)
            for value in _DEFAULT_HEAD_RELIABILITY[:expected]
        )
        config = cls(
            variant=os.getenv(
                "REMTP_TA_VARIANT",
                "tv_hidden_veto",
            ),
            cactus_delta=float(os.getenv("REMTP_CACTUS_DELTA", "1.0")),
            expected_draft_tokens=expected,
            head_reliability=_parse_head_reliability(
                os.getenv("REMTP_TA_HEAD_RELIABILITY", default_heads)
            ),
            target_log_gap_scale=float(
                os.getenv("REMTP_TA_TARGET_LOG_GAP_SCALE", "2.0")
            ),
            max_target_log_gap=float(
                os.getenv("REMTP_TA_MAX_TARGET_LOG_GAP", "8.0")
            ),
            future_veto_floor=float(
                os.getenv("REMTP_TA_FUTURE_VETO_FLOOR", "0.20")
            ),
            hidden_reliability_floor=float(
                os.getenv("REMTP_TA_HIDDEN_RELIABILITY_FLOOR", "0.25")
            ),
            use_prefix_value=_env_flag(
                "REMTP_TA_USE_PREFIX_VALUE",
                True,
            ),
            debt_position_limit=float(
                os.getenv("REMTP_TA_DEBT_POSITION_LIMIT", "0.35")
            ),
            debt_block_limit=float(
                os.getenv("REMTP_TA_DEBT_BLOCK_LIMIT", "1.20")
            ),
            debt_soft_log_gap=float(
                os.getenv("REMTP_TA_DEBT_SOFT_LOG_GAP", "2.0")
            ),
            debt_hard_log_gap=float(
                os.getenv("REMTP_TA_DEBT_HARD_LOG_GAP", "6.0")
            ),
            debt_max_position_tv=float(
                os.getenv("REMTP_TA_DEBT_MAX_POSITION_TV", "0.15")
            ),
            debt_max_cactus_ratio=float(
                os.getenv("REMTP_TA_DEBT_MAX_CACTUS_RATIO", "2.0")
            ),
            debt_fallback=os.getenv(
                "REMTP_TA_DEBT_FALLBACK",
                "strict",
            ),
            surplus_max_log_gap=float(
                os.getenv("REMTP_TA_SURPLUS_MAX_LOG_GAP", "2.0")
            ),
            risk_swap_soft_log_gap=float(
                os.getenv("REMTP_TA_RISK_SWAP_SOFT_LOG_GAP", "4.0")
            ),
            risk_swap_hard_log_gap=float(
                os.getenv("REMTP_TA_RISK_SWAP_HARD_LOG_GAP", "10.0")
            ),
            risk_swap_destination_log_gap=float(
                os.getenv(
                    "REMTP_TA_RISK_SWAP_DESTINATION_LOG_GAP",
                    "2.0",
                )
            ),
            recovery_mode=os.getenv(
                "REMTP_TA_RECOVERY_MODE",
                "residual",
            ),
        )
        config.validate()
        return config


@dataclass
class TargetAnchoredResult:
    """Scalar decisions and optional materialized verifier distributions."""

    probs: torch.Tensor | None
    target_candidate_probs: torch.Tensor
    draft_candidate_probs: torch.Tensor
    boosted_candidate_probs: torch.Tensor
    target_log_gaps: torch.Tensor
    target_support: torch.Tensor
    head_reliability: torch.Tensor
    hidden_similarity: torch.Tensor
    future_veto: torch.Tensor
    strict_acceptance: torch.Tensor
    prefix_value: torch.Tensor
    priority: torch.Tensor
    cactus_tv: torch.Tensor
    useful_tv_capacity: torch.Tensor
    raw_allocated_tv: torch.Tensor
    risk_capacity: torch.Tensor
    risk_multiplier: torch.Tensor
    allocated_tv: torch.Tensor
    relaxed_acceptance: torch.Tensor
    acceptance_residual: torch.Tensor
    cumulative_debt: torch.Tensor
    fallback_mask: torch.Tensor
    stopped_mask: torch.Tensor


def cactus_tv_increment(
    candidate_probs: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Return Cactus' exact TV shift for each sampled candidate."""
    bonus = torch.sqrt(
        (
            2.0
            * delta
            * candidate_probs
            * (1.0 - candidate_probs)
        ).clamp_min(0.0)
    )
    return torch.minimum(
        bonus,
        (1.0 - candidate_probs).clamp_min(0.0),
    )


def aligned_hidden_cosine(
    draft_hidden: torch.Tensor | None,
    target_hidden: torch.Tensor | None,
    rows: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    """Return non-negative current-position cosine or a neutral fallback."""
    if (
        draft_hidden is None
        or target_hidden is None
        or draft_hidden.ndim != 2
        or target_hidden.ndim != 2
        or draft_hidden.shape != target_hidden.shape
        or draft_hidden.shape[0] != rows
    ):
        return torch.ones(rows, device=device, dtype=torch.float32), False
    similarity = F.cosine_similarity(
        draft_hidden.to(torch.float32),
        target_hidden.to(torch.float32),
        dim=-1,
        eps=1e-8,
    )
    return similarity.clamp(0.0, 1.0), True


def _prefix_marginal_values(
    strict_acceptance: torch.Tensor,
) -> torch.Tensor:
    rows = strict_acceptance.shape[0]
    reach = torch.ones_like(strict_acceptance)
    for depth in range(1, rows):
        reach[depth] = reach[depth - 1] * strict_acceptance[depth - 1]
    suffix = torch.ones_like(strict_acceptance)
    for depth in range(rows - 2, -1, -1):
        suffix[depth] = 1.0 + strict_acceptance[depth + 1] * suffix[depth + 1]
    return reach * suffix


def _future_target_veto(
    local_support: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    """Penalize an earlier token if any later target support is poor."""
    rows = local_support.shape[0]
    result = torch.ones_like(local_support)
    for depth in range(rows - 1):
        later_support = local_support[depth + 1 :].amin()
        result[depth] = floor + (1.0 - floor) * later_support
    return result


def _allocate_capped_tv(
    capacities: torch.Tensor,
    priorities: torch.Tensor,
    total_budget: torch.Tensor,
) -> torch.Tensor:
    """Exact four-pass capped water filling over one MTP block."""
    allocated = torch.zeros_like(capacities)
    remaining = total_budget.clamp_min(0.0)
    for _ in range(capacities.shape[0]):
        residual = (capacities - allocated).clamp_min(0.0)
        active = torch.where(
            residual > 1e-12,
            priorities,
            torch.zeros_like(priorities),
        )
        proposed = (
            remaining
            * active
            / active.sum().clamp_min(1e-30)
        )
        step = torch.minimum(proposed, residual)
        allocated = allocated + step
        remaining = (remaining - step.sum()).clamp_min(0.0)
    return allocated


def _acceptance_from_candidate_probability(
    candidate_probability: torch.Tensor,
    draft_candidate_probability: torch.Tensor,
) -> torch.Tensor:
    """Return speculative acceptance for one or more proposed tokens."""
    return torch.minimum(
        torch.ones_like(candidate_probability),
        candidate_probability
        / draft_candidate_probability.clamp_min(1e-30),
    )


def _target_surplus_tv(
    *,
    p_y: torch.Tensor,
    q_y: torch.Tensor,
    top_p: torch.Tensor,
    cactus_tv: torch.Tensor,
    max_log_gap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve Cactus acceptance and recycle only saturated surplus.

    ``cactus_floor`` has exactly the same sampled-candidate acceptance as
    Cactus because candidate probability above ``q_y`` is already saturated.
    The otherwise acceptance-inert mass is reassigned, in prefix order, only
    to positions whose proposed token stays within ``max_log_gap`` of the
    verifier's top-1 probability. A zero gap gives the strict top-1 ablation.
    """
    useful_capacity = (q_y - p_y).clamp_min(0.0)
    cactus_floor = torch.minimum(cactus_tv, useful_capacity)
    surplus = (cactus_tv - cactus_floor).clamp_min(0.0).sum()
    log_gap = (
        torch.log(top_p.clamp_min(1e-30))
        - torch.log(p_y.clamp_min(1e-30))
    ).clamp_min(0.0)
    target_supported = log_gap <= max_log_gap
    destination_capacity = torch.where(
        target_supported,
        (useful_capacity - cactus_floor).clamp_min(0.0),
        torch.zeros_like(useful_capacity),
    )

    received = torch.zeros_like(useful_capacity)
    remaining = surplus
    for depth in range(useful_capacity.shape[0]):
        step = torch.minimum(destination_capacity[depth], remaining)
        received[depth] = step
        remaining = (remaining - step).clamp_min(0.0)
    return (
        cactus_floor + received,
        cactus_floor,
        destination_capacity,
        target_supported,
    )


def top1_surplus_candidate_probs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    delta: float,
    max_log_gap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast scalar path for Cactus-dominant top-1 surplus recycling."""
    ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    rows = torch.arange(ids.shape[0], device=target_probs.device)
    p_y = target_probs[rows, ids]
    q_y = draft_probs[rows, ids]
    top_p = target_probs.amax(dim=-1)
    cactus_tv = cactus_tv_increment(p_y, delta)
    allocated, _, _, _ = _target_surplus_tv(
        p_y=p_y,
        q_y=q_y,
        top_p=top_p,
        cactus_tv=cactus_tv,
        max_log_gap=max_log_gap,
    )
    return p_y, p_y + allocated


def _risk_swap_tv(
    *,
    p_y: torch.Tensor,
    q_y: torch.Tensor,
    log_gap: torch.Tensor,
    cactus_tv: torch.Tensor,
    soft_log_gap: float,
    hard_log_gap: float,
    destination_log_gap: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Swap risky Cactus TV into safer, higher-value prefix positions.

    Cactus effective TV is kept unchanged up to ``soft_log_gap``, attenuated
    linearly until ``hard_log_gap``, and removed at or beyond the hard gap.
    The removed TV and candidate-saturation surplus share one exact block
    budget and are reallocated only to target-supported destinations. Safe
    capacity is filled in prefix order because no later speculative token can
    be committed after an earlier rejection.
    """
    useful_capacity = (q_y - p_y).clamp_min(0.0)
    cactus_floor = torch.minimum(cactus_tv, useful_capacity)
    gap_width = hard_log_gap - soft_log_gap
    keep_multiplier = (
        (hard_log_gap - log_gap) / gap_width
    ).clamp(0.0, 1.0)
    kept = cactus_floor * keep_multiplier
    pool = (cactus_tv - kept).clamp_min(0.0).sum()

    destination_mask = log_gap <= destination_log_gap
    destination_capacity = torch.where(
        destination_mask,
        (useful_capacity - kept).clamp_min(0.0),
        torch.zeros_like(useful_capacity),
    )
    received = torch.zeros_like(useful_capacity)
    remaining = pool
    for depth in range(useful_capacity.shape[0]):
        step = torch.minimum(destination_capacity[depth], remaining)
        received[depth] = step
        remaining = (remaining - step).clamp_min(0.0)
    return (
        kept + received,
        cactus_floor,
        kept,
        destination_capacity,
        keep_multiplier,
    )


def _debt_controlled_tv(
    *,
    p_y: torch.Tensor,
    q_y: torch.Tensor,
    log_gap: torch.Tensor,
    strict_acceptance: torch.Tensor,
    cactus_tv: torch.Tensor,
    useful_capacity: torch.Tensor,
    priority: torch.Tensor,
    config: TargetAnchoredConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Allocate block TV under position and cumulative verification debt.

    Debt is the exact increase in speculative acceptance probability caused
    by relaxation, ``A_relaxed - A_strict``. Risk caps are constructed before
    water filling so TV reclaimed from an unsafe position can still move to a
    safer, high-value position. A prefix-order pass enforces the block debt
    limit and stops later relaxation after the limit or a hard target veto.
    """
    raw_allocated = _allocate_capped_tv(
        useful_capacity,
        priority,
        cactus_tv.sum(),
    )
    capped_cactus = torch.minimum(cactus_tv, useful_capacity)

    gap_width = config.debt_hard_log_gap - config.debt_soft_log_gap
    gap_multiplier = (
        (config.debt_hard_log_gap - log_gap) / gap_width
    ).clamp(0.0, 1.0)

    # A_relaxed - A_strict <= position_limit. Before saturation this maps
    # exactly to h-p <= q*position_limit.
    max_acceptance = (
        strict_acceptance + config.debt_position_limit
    ).clamp(max=1.0)
    debt_tv_cap = (
        q_y * max_acceptance - p_y
    ).clamp_min(0.0)
    cactus_ratio_cap = capped_cactus * config.debt_max_cactus_ratio
    absolute_tv_cap = torch.full_like(
        useful_capacity,
        config.debt_max_position_tv,
    )
    risk_capacity = torch.minimum(
        useful_capacity,
        torch.minimum(
            debt_tv_cap,
            torch.minimum(cactus_ratio_cap, absolute_tv_cap),
        ),
    ) * gap_multiplier

    proposed = _allocate_capped_tv(
        risk_capacity,
        priority,
        cactus_tv.sum(),
    )

    kept = torch.zeros_like(proposed)
    relaxed_acceptance = strict_acceptance.clone()
    acceptance_residual = torch.zeros_like(proposed)
    cumulative_debt = torch.zeros_like(proposed)
    fallback_mask = torch.zeros_like(proposed, dtype=torch.bool)
    stopped_mask = torch.zeros_like(proposed, dtype=torch.bool)
    running_debt = torch.zeros_like(proposed[0])
    active = torch.ones_like(proposed[0], dtype=torch.bool)
    use_cactus_fallback = config.debt_fallback == "cactus_cap"

    for depth in range(proposed.shape[0]):
        hard_veto = log_gap[depth] >= config.debt_hard_log_gap
        remaining_debt = (
            config.debt_block_limit - running_debt
        ).clamp_min(0.0)
        allowed_acceptance = torch.minimum(
            _acceptance_from_candidate_probability(
                p_y[depth] + proposed[depth],
                q_y[depth],
            ),
            strict_acceptance[depth] + remaining_debt,
        ).clamp(max=1.0)
        block_tv_cap = (
            q_y[depth] * allowed_acceptance - p_y[depth]
        ).clamp_min(0.0)
        controlled = torch.minimum(proposed[depth], block_tv_cap)

        risk_stop = hard_veto | (~active) | (remaining_debt <= 1e-12)
        fallback = (
            capped_cactus[depth]
            if use_cactus_fallback
            else torch.zeros_like(controlled)
        )
        chosen = torch.where(risk_stop, fallback, controlled)
        kept[depth] = chosen
        current_acceptance = _acceptance_from_candidate_probability(
            p_y[depth] + chosen,
            q_y[depth],
        )
        current_debt = (
            current_acceptance - strict_acceptance[depth]
        ).clamp_min(0.0)
        relaxed_acceptance[depth] = current_acceptance
        acceptance_residual[depth] = current_debt
        running_debt = running_debt + current_debt
        cumulative_debt[depth] = running_debt
        fallback_mask[depth] = risk_stop
        stopped_mask[depth] = ~active

        hit_block_limit = running_debt >= (
            config.debt_block_limit - 1e-12
        )
        active = active & (~hard_veto) & (~hit_block_limit)

    return (
        kept,
        raw_allocated,
        risk_capacity,
        gap_multiplier,
        relaxed_acceptance,
        acceptance_residual,
        cumulative_debt,
        fallback_mask,
        stopped_mask,
    )


def target_anchored_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: TargetAnchoredConfig,
    *,
    hidden_similarity: torch.Tensor | None = None,
    assume_normalized: bool = False,
    construct_probs: bool = True,
) -> TargetAnchoredResult:
    """Build the selected target-anchored verifier distribution."""
    config.validate()
    if target_probs.ndim != 2 or target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must share [rows,vocab]")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must have shape [rows]")
    row_count = target_probs.shape[0]
    if row_count != draft_token_ids.shape[0]:
        raise ValueError("probability and token rows must match")
    if not 1 <= row_count <= config.expected_draft_tokens:
        raise ValueError("unexpected MTP block length")

    target = target_probs.to(torch.float32)
    draft = draft_probs.to(device=target.device, dtype=torch.float32)
    if not assume_normalized:
        target = target.clamp_min(0.0)
        draft = draft.clamp_min(0.0)
        target /= target.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        draft /= draft.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    ids = draft_token_ids.to(device=target.device, dtype=torch.int64)
    rows = torch.arange(row_count, device=target.device)
    p_y = target[rows, ids]
    q_y = draft[rows, ids]
    top_p = target.amax(dim=-1)
    log_gap = (
        torch.log(top_p.clamp_min(1e-30))
        - torch.log(p_y.clamp_min(1e-30))
    ).clamp_min(0.0)
    local_support = torch.exp(
        -log_gap / config.target_log_gap_scale
    )
    local_support = torch.where(
        log_gap <= config.max_target_log_gap,
        local_support,
        torch.zeros_like(local_support),
    )

    head_prior = torch.as_tensor(
        config.head_reliability[:row_count],
        device=target.device,
        dtype=torch.float32,
    )
    if hidden_similarity is None:
        hidden = torch.ones_like(p_y)
    else:
        if hidden_similarity.shape != p_y.shape:
            raise ValueError("hidden_similarity must have one value per row")
        hidden = hidden_similarity.to(
            device=target.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

    strict_acceptance = torch.minimum(
        torch.ones_like(p_y),
        p_y / q_y.clamp_min(1e-30),
    )
    prefix_value = (
        _prefix_marginal_values(strict_acceptance)
        if config.use_prefix_value
        else torch.ones_like(p_y)
    )
    future_veto = _future_target_veto(
        local_support,
        config.future_veto_floor,
    )
    hidden_reliability = (
        config.hidden_reliability_floor
        + (1.0 - config.hidden_reliability_floor) * hidden
    )
    rescue = strict_acceptance.sqrt() * (1.0 - strict_acceptance)
    priority = rescue * prefix_value * local_support * head_prior
    if config.variant == "tv_hidden_veto":
        priority = priority * hidden_reliability * future_veto

    cactus_tv = cactus_tv_increment(p_y, config.cactus_delta)
    useful_capacity = (q_y - p_y).clamp_min(0.0)
    raw_allocated = torch.zeros_like(useful_capacity)
    risk_capacity = useful_capacity.clone()
    risk_multiplier = torch.ones_like(useful_capacity)
    relaxed_acceptance = strict_acceptance.clone()
    acceptance_residual = torch.zeros_like(useful_capacity)
    cumulative_debt = torch.zeros_like(useful_capacity)
    fallback_mask = torch.zeros_like(useful_capacity, dtype=torch.bool)
    stopped_mask = torch.zeros_like(useful_capacity, dtype=torch.bool)
    if config.variant == "cactus_cap":
        allocated = torch.minimum(cactus_tv, useful_capacity)
        raw_allocated = allocated.clone()
    elif config.variant in {"tv_top1_surplus", "tv_target_surplus"}:
        (
            allocated,
            cactus_floor,
            destination_capacity,
            target_supported,
        ) = _target_surplus_tv(
            p_y=p_y,
            q_y=q_y,
            top_p=top_p,
            cactus_tv=cactus_tv,
            max_log_gap=(
                0.0
                if config.variant == "tv_top1_surplus"
                else config.surplus_max_log_gap
            ),
        )
        raw_allocated = allocated.clone()
        risk_capacity = destination_capacity
        risk_multiplier = target_supported.to(torch.float32)
    elif config.variant == "tv_risk_swap":
        (
            allocated,
            cactus_floor,
            kept_cactus,
            destination_capacity,
            keep_multiplier,
        ) = _risk_swap_tv(
            p_y=p_y,
            q_y=q_y,
            log_gap=log_gap,
            cactus_tv=cactus_tv,
            soft_log_gap=config.risk_swap_soft_log_gap,
            hard_log_gap=config.risk_swap_hard_log_gap,
            destination_log_gap=config.risk_swap_destination_log_gap,
        )
        raw_allocated = cactus_floor
        risk_capacity = kept_cactus + destination_capacity
        risk_multiplier = keep_multiplier
    elif config.variant == "tv_debt_control":
        (
            allocated,
            raw_allocated,
            risk_capacity,
            risk_multiplier,
            relaxed_acceptance,
            acceptance_residual,
            cumulative_debt,
            fallback_mask,
            stopped_mask,
        ) = _debt_controlled_tv(
            p_y=p_y,
            q_y=q_y,
            log_gap=log_gap,
            strict_acceptance=strict_acceptance,
            cactus_tv=cactus_tv,
            useful_capacity=useful_capacity,
            priority=priority,
            config=config,
        )
    else:
        allocated = _allocate_capped_tv(
            useful_capacity,
            priority,
            cactus_tv.sum(),
        )
        raw_allocated = allocated.clone()
    if config.variant != "tv_debt_control":
        relaxed_acceptance = _acceptance_from_candidate_probability(
            p_y + allocated,
            q_y,
        )
        acceptance_residual = (
            relaxed_acceptance - strict_acceptance
        ).clamp_min(0.0)
        cumulative_debt = acceptance_residual.cumsum(dim=0)
    boosted = p_y + allocated

    relaxed = None
    if construct_probs:
        scale = (
            (1.0 - boosted) / (1.0 - p_y)
        ).nan_to_num(nan=1.0, posinf=1.0, neginf=0.0)
        relaxed = target * scale.unsqueeze(-1)
        relaxed.scatter_(1, ids.unsqueeze(1), boosted.unsqueeze(1))
        relaxed = relaxed.clamp_min(0.0)
        relaxed /= relaxed.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    return TargetAnchoredResult(
        probs=relaxed,
        target_candidate_probs=p_y,
        draft_candidate_probs=q_y,
        boosted_candidate_probs=boosted,
        target_log_gaps=log_gap,
        target_support=local_support,
        head_reliability=head_prior,
        hidden_similarity=hidden,
        future_veto=future_veto,
        strict_acceptance=strict_acceptance,
        prefix_value=prefix_value,
        priority=priority,
        cactus_tv=cactus_tv,
        useful_tv_capacity=useful_capacity,
        raw_allocated_tv=raw_allocated,
        risk_capacity=risk_capacity,
        risk_multiplier=risk_multiplier,
        allocated_tv=allocated,
        relaxed_acceptance=relaxed_acceptance,
        acceptance_residual=acceptance_residual,
        cumulative_debt=cumulative_debt,
        fallback_mask=fallback_mask,
        stopped_mask=stopped_mask,
    )


def target_anchored_candidate_probs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    hidden_similarity: torch.Tensor,
    config: TargetAnchoredConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.variant in {"tv_top1_surplus", "tv_target_surplus"}:
        return top1_surplus_candidate_probs(
            target_probs,
            draft_probs,
            draft_token_ids,
            config.cactus_delta,
            (
                0.0
                if config.variant == "tv_top1_surplus"
                else config.surplus_max_log_gap
            ),
        )
    result = target_anchored_distribution(
        target_probs,
        draft_probs,
        draft_token_ids,
        config,
        hidden_similarity=hidden_similarity,
        assume_normalized=True,
        construct_probs=False,
    )
    return result.target_candidate_probs, result.boosted_candidate_probs


def boost_candidate_logits(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    original_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
) -> torch.Tensor:
    """Apply candidate TV shifts without materializing full h."""
    logits = target_logits.to(torch.float32).clone()
    ids = draft_token_ids.to(device=logits.device, dtype=torch.int64)
    rows = torch.arange(logits.shape[0], device=logits.device)
    original = original_candidate_probs.clamp(1e-12, 1.0 - 1e-6)
    boosted = boosted_candidate_probs.clamp(1e-12, 1.0 - 1e-6)
    logits[rows, ids] += torch.logit(boosted) - torch.logit(original)
    return logits


def _sample_with_target_recovery(
    *,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    verification_logits: torch.Tensor,
    original_target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Keep relaxed acceptance but recover a rejection from target ``p``.

    The ordinary sampler draws RECOVER from ``(h-q)+``.  This ablation keeps
    the same acceptance uniforms and relaxed distribution ``h`` but samples
    the first correction token from the original target distribution.  The
    accepted prefix and target bonus rules are unchanged.
    """
    module = importlib.import_module(_V1_REJECTION_MODULE)
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = verification_logits.shape[-1]
    device = verification_logits.device
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        module.PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = (
            sampling_metadata.temperature == module.GREEDY_TEMPERATURE
        )
    if not sampling_metadata.all_random:
        target_argmax = verification_logits.argmax(dim=-1)
        module.rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    verification_probs = verification_logits.softmax(
        dim=-1,
        dtype=torch.float32,
    ).contiguous()
    uniform_probs = module.generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )
    recovered_token_ids = module.sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        None,
        original_target_probs.contiguous(),
        sampling_metadata,
        device,
    )
    module.rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        verification_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=False,
    )
    return output_token_ids


def _target_anchored_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    global _AUDIT_ROUND, _AUDIT_TV, _AUDIT_DEBT, _AUDIT_SURPLUS
    global _AUDIT_ELIGIBILITY
    global _DIAGNOSTIC_EMITTED
    global _HIDDEN_FALLBACK_EMITTED

    original = getattr(_target_anchored_rejection_sample, "_remtp_original")
    if target_logits.shape[0] == 0 or draft_probs is None:
        return original(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )
    if len(num_draft_tokens) != 1:
        raise RuntimeError(
            "target-anchored MTP requires --max-num-seqs 1"
        )

    config = getattr(_target_anchored_rejection_sample, "_remtp_config")
    if not 1 <= max_spec_len <= config.expected_draft_tokens:
        raise RuntimeError(
            f"configured MTP={config.expected_draft_tokens}, "
            f"but vLLM max_spec_len={max_spec_len}"
        )

    logits = target_logits.to(torch.float32)
    target_probs = torch.softmax(logits, dim=-1)
    hidden_similarity = torch.ones(
        draft_token_ids.shape[0],
        device=logits.device,
        dtype=torch.float32,
    )
    hidden_available = False
    if config.variant == "tv_hidden_veto":
        from remtp.probabilistic_mtp import get_last_aligned_hidden_states

        draft_hidden, target_hidden = get_last_aligned_hidden_states(
            draft_token_ids.shape[0]
        )
        hidden_similarity, hidden_available = aligned_hidden_cosine(
            draft_hidden,
            target_hidden,
            draft_token_ids.shape[0],
            device=logits.device,
        )
        if not hidden_available and not _HIDDEN_FALLBACK_EMITTED:
            print(
                "[ReMTP][TargetAnchored] aligned hidden unavailable; "
                "using neutral cosine and static head reliability",
                flush=True,
            )
            _HIDDEN_FALLBACK_EMITTED = True

    post_verification_hook = _POST_VERIFICATION_HOOK
    detailed = (
        getattr(_target_anchored_rejection_sample, "_remtp_diagnostics")
        and not _DIAGNOSTIC_EMITTED
    ) or (
        getattr(_target_anchored_rejection_sample, "_remtp_audit_interval") > 0
    )
    result: TargetAnchoredResult | None = None
    if detailed:
        result = target_anchored_distribution(
            target_probs,
            draft_probs,
            draft_token_ids,
            config,
            hidden_similarity=hidden_similarity,
            assume_normalized=True,
            construct_probs=False,
        )
        p_y = result.target_candidate_probs
        h_y = result.boosted_candidate_probs
    else:
        fast = getattr(_target_anchored_rejection_sample, "_remtp_fast")
        p_y, h_y = fast(
            target_probs,
            draft_probs,
            draft_token_ids,
            hidden_similarity,
        )

    pre_verification_hook = _PRE_VERIFICATION_HOOK
    if pre_verification_hook is not None:
        h_y = pre_verification_hook(
            target_probs=target_probs,
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            target_candidate_probs=p_y,
            boosted_candidate_probs=h_y,
            hidden_similarity=hidden_similarity,
            sampling_metadata=sampling_metadata,
        )
        h_y = torch.maximum(h_y, p_y)
        rows = torch.arange(
            draft_token_ids.shape[0],
            device=target_probs.device,
        )
        q_y = draft_probs[
            rows,
            draft_token_ids.to(torch.int64),
        ]
        h_y = torch.minimum(h_y, q_y)

    diagnostics = getattr(
        _target_anchored_rejection_sample,
        "_remtp_diagnostics",
    )
    if diagnostics and not _DIAGNOSTIC_EMITTED and result is not None:
        print(
            "[ReMTP][TargetAnchored][diagnostic] "
            f"variant={config.variant} hidden={int(hidden_available)} "
            f"p={result.target_candidate_probs.tolist()} "
            f"q={result.draft_candidate_probs.tolist()} "
            f"h={result.boosted_candidate_probs.tolist()} "
            f"log_gap={result.target_log_gaps.tolist()} "
            f"support={result.target_support.tolist()} "
            f"head={result.head_reliability.tolist()} "
            f"hidden_cos={result.hidden_similarity.tolist()} "
            f"future_veto={result.future_veto.tolist()} "
            f"strict_A={result.strict_acceptance.tolist()} "
            f"priority={result.priority.tolist()} "
            f"cactus_TV={result.cactus_tv.tolist()} "
            f"capacity={result.useful_tv_capacity.tolist()} "
            f"raw_allocated_TV={result.raw_allocated_tv.tolist()} "
            f"risk_capacity={result.risk_capacity.tolist()} "
            f"risk_multiplier={result.risk_multiplier.tolist()} "
            f"allocated_TV={result.allocated_tv.tolist()} "
            f"relaxed_A={result.relaxed_acceptance.tolist()} "
            f"acceptance_debt={result.acceptance_residual.tolist()} "
            f"cumulative_debt={result.cumulative_debt.tolist()} "
            f"fallback={result.fallback_mask.tolist()} "
            f"stopped={result.stopped_mask.tolist()} "
            f"target_supported={result.risk_multiplier.tolist()} "
            f"block_cactus_TV={result.cactus_tv.sum().item():.6f} "
            f"block_allocated_TV={result.allocated_tv.sum().item():.6f}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True

    audit_interval = getattr(
        _target_anchored_rejection_sample,
        "_remtp_audit_interval",
    )
    if audit_interval > 0 and result is not None:
        _AUDIT_ROUND += 1
        _AUDIT_TV += torch.tensor(
            (
                result.cactus_tv.sum().item(),
                result.allocated_tv.sum().item(),
                result.useful_tv_capacity.sum().item(),
            ),
            dtype=torch.float64,
        )
        _AUDIT_DEBT += torch.tensor(
            (
                result.raw_allocated_tv.sum().item(),
                result.allocated_tv.sum().item(),
                result.acceptance_residual.sum().item(),
                result.fallback_mask.sum().item(),
                result.stopped_mask.sum().item(),
                (
                    result.raw_allocated_tv - result.allocated_tv
                ).clamp_min(0.0).sum().item(),
            ),
            dtype=torch.float64,
        )
        cactus_floor = torch.minimum(
            result.cactus_tv,
            result.useful_tv_capacity,
        )
        surplus_generated = (
            result.cactus_tv - cactus_floor
        ).clamp_min(0.0).sum().item()
        surplus_received = (
            result.allocated_tv - cactus_floor
        ).clamp_min(0.0).sum().item()
        _AUDIT_SURPLUS += torch.tensor(
            (
                surplus_generated,
                surplus_received,
                (
                    (result.allocated_tv - cactus_floor) > 1e-12
                ).sum().item(),
                max(surplus_generated - surplus_received, 0.0),
            ),
            dtype=torch.float64,
        )
        available_after_floor = (
            result.useful_tv_capacity - cactus_floor
        ).clamp_min(0.0)
        eligibility_values: list[float] = []
        for threshold in (0.5, 1.0, 1.5, 2.0, 3.0):
            eligible = result.target_log_gaps <= threshold
            eligibility_values.extend(
                (
                    eligible.logical_and(
                        available_after_floor > 1e-12
                    ).sum().item(),
                    torch.where(
                        eligible,
                        available_after_floor,
                        torch.zeros_like(available_after_floor),
                    ).sum().item(),
                )
            )
        _AUDIT_ELIGIBILITY += torch.tensor(
            eligibility_values,
            dtype=torch.float64,
        )
        if _AUDIT_ROUND % audit_interval == 0:
            values = (_AUDIT_TV / audit_interval).tolist()
            debt_values = (_AUDIT_DEBT / audit_interval).tolist()
            surplus_values = (_AUDIT_SURPLUS / audit_interval).tolist()
            eligibility = (
                _AUDIT_ELIGIBILITY / audit_interval
            ).tolist()
            surplus_suffix = ""
            if config.variant in {"tv_top1_surplus", "tv_target_surplus"}:
                surplus_suffix = (
                    f" mean_surplus_generated={surplus_values[0]:.6f}"
                    f" mean_surplus_received={surplus_values[1]:.6f}"
                    f" mean_surplus_destinations={surplus_values[2]:.6f}"
                    f" mean_surplus_remaining={surplus_values[3]:.6f}"
                    f" eligible_gap0.5={eligibility[0]:.6f}:"
                    f"{eligibility[1]:.6f}"
                    f" eligible_gap1.0={eligibility[2]:.6f}:"
                    f"{eligibility[3]:.6f}"
                    f" eligible_gap1.5={eligibility[4]:.6f}:"
                    f"{eligibility[5]:.6f}"
                    f" eligible_gap2.0={eligibility[6]:.6f}:"
                    f"{eligibility[7]:.6f}"
                    f" eligible_gap3.0={eligibility[8]:.6f}:"
                    f"{eligibility[9]:.6f}"
                )
            print(
                "[ReMTP][TargetAnchored][audit] "
                f"rounds={_AUDIT_ROUND-audit_interval+1}-{_AUDIT_ROUND} "
                f"mean_cactus_TV={values[0]:.6f} "
                f"mean_allocated_TV={values[1]:.6f} "
                f"mean_useful_capacity={values[2]:.6f} "
                f"budget_utilization={values[1]/max(values[0],1e-30):.6f} "
                f"mean_raw_exact_TV={debt_values[0]:.6f} "
                f"mean_controlled_TV={debt_values[1]:.6f} "
                f"mean_acceptance_debt={debt_values[2]:.6f} "
                f"mean_fallback_positions={debt_values[3]:.6f} "
                f"mean_stopped_positions={debt_values[4]:.6f} "
                f"mean_risk_pruned_TV={debt_values[5]:.6f}"
                f"{surplus_suffix}",
                flush=True,
            )
            _AUDIT_TV.zero_()
            _AUDIT_DEBT.zero_()
            _AUDIT_SURPLUS.zero_()
            _AUDIT_ELIGIBILITY.zero_()

    original_top_ids = (
        logits.argmax(dim=-1) if sampling_metadata.all_greedy else None
    )
    relaxed_logits = boost_candidate_logits(
        logits,
        draft_token_ids,
        p_y,
        h_y,
    )
    if sampling_metadata.all_greedy:
        assert original_top_ids is not None
        relaxed_top_ids = relaxed_logits.argmax(dim=-1)
        desired_ids = torch.where(
            draft_token_ids == relaxed_top_ids,
            draft_token_ids,
            original_top_ids,
        )
        verification_logits = torch.full_like(logits, float("-inf"))
        verification_logits.scatter_(1, desired_ids.unsqueeze(1), 0.0)
    else:
        verification_logits = relaxed_logits

    if config.recovery_mode == "target" and not sampling_metadata.all_greedy:
        output_token_ids = _sample_with_target_recovery(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            max_spec_len=max_spec_len,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_probs=draft_probs,
            verification_logits=verification_logits,
            original_target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
        )
    else:
        output_token_ids = original(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            verification_logits,
            bonus_token_ids,
            sampling_metadata,
        )
    if post_verification_hook is not None:
        ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
        rows = torch.arange(ids.shape[0], device=target_probs.device)
        q_y = draft_probs[rows, ids]
        post_verification_hook(
            target_probs=target_probs,
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            output_token_ids=output_token_ids,
            target_candidate_probs=p_y,
            boosted_candidate_probs=h_y,
            allocated_tv=(h_y - p_y).clamp_min(0.0),
            strict_acceptance=torch.minimum(
                torch.ones_like(p_y),
                p_y / q_y.clamp_min(1e-30),
            ),
            hidden_similarity=hidden_similarity,
            sampling_metadata=sampling_metadata,
        )
    return output_token_ids


def set_post_verification_hook(hook: Any | None) -> None:
    """Register an optional observer after target-anchored verification.

    The hook receives the exact constrained target distribution, full MTP
    proposal distribution, compiled verifier scalars, and committed output.
    It cannot change the verifier result or force the verifier off its fast
    path.
    """
    global _POST_VERIFICATION_HOOK

    _POST_VERIFICATION_HOOK = hook


def set_pre_verification_hook(hook: Any | None) -> None:
    """Register an optional constrained-budget controller before sampling.

    The hook may only change the candidate probabilities used by the verifier.
    The wrapper enforces ``p(y) <= h(y) <= q(y)`` after the hook returns.
    """
    global _PRE_VERIFICATION_HOOK

    _PRE_VERIFICATION_HOOK = hook


def install_target_anchored_mtp() -> None:
    """Install target-anchored exact-TV verification."""
    config = TargetAnchoredConfig.from_env()
    diagnostics = _env_flag("REMTP_TA_DIAGNOSTICS", False)
    audit_interval = int(os.getenv("REMTP_TA_AUDIT_INTERVAL", "0"))
    if audit_interval < 0:
        raise ValueError("REMTP_TA_AUDIT_INTERVAL must be non-negative")
    compile_fast_path = _env_flag("REMTP_TA_COMPILE", True)

    def fast(
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        hidden_similarity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return target_anchored_candidate_probs(
            target_probs,
            draft_probs,
            draft_token_ids,
            hidden_similarity,
            config,
        )

    if compile_fast_path:
        fast = torch.compile(fast, fullgraph=True, dynamic=False)

    module = importlib.import_module(_V1_REJECTION_MODULE)
    current = module.rejection_sample
    if not getattr(current, "_remtp_target_anchored_mtp", False):
        wrapper = _target_anchored_rejection_sample
        wrapper._remtp_target_anchored_mtp = True
        wrapper._remtp_original = current
        wrapper._remtp_config = config
        wrapper._remtp_diagnostics = diagnostics
        wrapper._remtp_audit_interval = audit_interval
        wrapper._remtp_fast = fast
        module.rejection_sample = wrapper

    print(
        "[ReMTP][TargetAnchored] "
        f"variant={config.variant} cactus_delta={config.cactus_delta:g} "
        f"draft_tokens={config.expected_draft_tokens} "
        f"head_reliability={list(config.head_reliability)} "
        f"gap_scale={config.target_log_gap_scale:g} "
        f"max_gap={config.max_target_log_gap:g} "
        f"future_veto_floor={config.future_veto_floor:g} "
        f"hidden_floor={config.hidden_reliability_floor:g} "
        f"debt_position_limit={config.debt_position_limit:g} "
        f"debt_block_limit={config.debt_block_limit:g} "
        f"debt_gap={config.debt_soft_log_gap:g}:"
        f"{config.debt_hard_log_gap:g} "
        f"debt_max_position_TV={config.debt_max_position_tv:g} "
        f"debt_max_cactus_ratio={config.debt_max_cactus_ratio:g} "
        f"debt_fallback={config.debt_fallback} "
        f"surplus_max_gap={config.surplus_max_log_gap:g} "
        f"risk_swap_gap={config.risk_swap_soft_log_gap:g}:"
        f"{config.risk_swap_hard_log_gap:g} "
        f"risk_swap_destination_gap="
        f"{config.risk_swap_destination_log_gap:g} "
        f"recovery={config.recovery_mode} "
        f"compiled={int(compile_fast_path)} "
        "budget=exact-Cactus-TV cap=h(y)<=q(y) "
        f"surplus={int(config.variant in {'tv_top1_surplus', 'tv_target_surplus'})}",
        flush=True,
    )
