"""Block-aware, feature-consistent relaxation for probabilistic MTP.

The verifier sees the complete MTP block in one target forward pass. This
module uses that information to allocate one closed-form Cactus delta budget
across the whole block instead of independently relaxing each token:

* target top-k rank and target top-1 gap for the sampled draft token;
* target-led top-k+candidate+tail JS similarity for q_d/p_d;
* agreement at later positions in the sampled MTP block;
* the number of later draft/bonus tokens unlocked by an accepted prefix.

The fast path avoids iterative KL inversion and full-vocabulary entropy/rank
reductions. It keeps vLLM's standard stochastic acceptance and residual
sampling after replacing each target distribution p_d by the block-conditioned
temporary distribution h_d.  Bonus tokens always come from the original target.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch


_V1_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False
_AUDIT_ROUND = 0
_AUDIT_WINDOW: torch.Tensor | None = None
_AUDIT_WINDOW_COUNT = 0


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


@dataclass(frozen=True)
class BlockFeatureConfig:
    """Configuration for block-aware feature-consistent verification."""

    block_delta_budget: float = 4.0
    expected_draft_tokens: int = 4
    use_prefix_value: bool = True
    use_current_distribution: bool = True
    use_future_distribution: bool = True
    future_decay: float = 0.7
    distribution_top_k: int = 8
    token_rank_scale: float = 4.0
    token_log_gap_scale: float = 2.0
    token_signal_weight: float = 1.0
    current_distribution_weight: float = 1.0
    future_distribution_weight: float = 1.0
    reliability_power: float = 2.0
    min_reliability: float = 0.2
    min_target_probability: float = 1e-3
    max_target_log_gap: float = 8.0
    min_prefix_reach_probability: float = 0.01
    trust_token_reference: float = 0.25
    trust_reliability_reference: float = 0.6
    trust_acceptance_reference: float = 0.01

    def validate(self) -> None:
        if self.block_delta_budget < 0:
            raise ValueError("block_delta_budget must be non-negative")
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if not 0 <= self.future_decay <= 1:
            raise ValueError("future_decay must be in [0, 1]")
        if self.distribution_top_k < 1:
            raise ValueError("distribution_top_k must be positive")
        if self.token_rank_scale <= 0 or self.token_log_gap_scale <= 0:
            raise ValueError("token support scales must be positive")
        signal_weights = (
            self.token_signal_weight,
            self.current_distribution_weight,
            self.future_distribution_weight,
        )
        if any(weight < 0 for weight in signal_weights):
            raise ValueError("signal weights must be non-negative")
        if self.token_signal_weight <= 0:
            raise ValueError("token_signal_weight must be positive")
        if self.reliability_power <= 0:
            raise ValueError("reliability_power must be positive")
        if not 0 <= self.min_reliability <= 1:
            raise ValueError("min_reliability must be in [0, 1]")
        if not 0 <= self.min_target_probability < 1:
            raise ValueError("min_target_probability must be in [0, 1)")
        if self.max_target_log_gap <= 0:
            raise ValueError("max_target_log_gap must be positive")
        if not 0 <= self.min_prefix_reach_probability <= 1:
            raise ValueError(
                "min_prefix_reach_probability must be in [0, 1]"
            )
        if self.trust_token_reference <= 0:
            raise ValueError("trust_token_reference must be positive")
        if self.trust_reliability_reference <= 0:
            raise ValueError("trust_reliability_reference must be positive")
        if not 0 < self.trust_acceptance_reference <= 1:
            raise ValueError("trust_acceptance_reference must be in (0, 1]")
    @classmethod
    def from_env(cls) -> BlockFeatureConfig:
        delta_budget = os.getenv(
            "REMTP_BLOCK_DELTA_BUDGET",
            os.getenv("REMTP_BLOCK_KL_BUDGET", "4.0"),
        )
        config = cls(
            block_delta_budget=float(delta_budget),
            expected_draft_tokens=int(
                os.getenv("REMTP_BLOCK_EXPECTED_DRAFT_TOKENS", "4")
            ),
            use_prefix_value=_env_flag(
                "REMTP_BLOCK_USE_PREFIX_VALUE", True
            ),
            use_current_distribution=_env_flag(
                "REMTP_BLOCK_USE_CURRENT_DISTRIBUTION", True
            ),
            use_future_distribution=_env_flag(
                "REMTP_BLOCK_USE_FUTURE_DISTRIBUTION", True
            ),
            future_decay=float(
                os.getenv("REMTP_BLOCK_FUTURE_DECAY", "0.7")
            ),
            distribution_top_k=int(
                os.getenv("REMTP_BLOCK_DISTRIBUTION_TOP_K", "8")
            ),
            token_rank_scale=float(
                os.getenv("REMTP_BLOCK_TOKEN_RANK_SCALE", "4.0")
            ),
            token_log_gap_scale=float(
                os.getenv("REMTP_BLOCK_TOKEN_LOG_GAP_SCALE", "2.0")
            ),
            token_signal_weight=float(
                os.getenv("REMTP_BLOCK_TOKEN_SIGNAL_WEIGHT", "1.0")
            ),
            current_distribution_weight=float(
                os.getenv(
                    "REMTP_BLOCK_CURRENT_DISTRIBUTION_WEIGHT",
                    "1.0",
                )
            ),
            future_distribution_weight=float(
                os.getenv(
                    "REMTP_BLOCK_FUTURE_DISTRIBUTION_WEIGHT",
                    "1.0",
                )
            ),
            reliability_power=float(
                os.getenv("REMTP_BLOCK_RELIABILITY_POWER", "2.0")
            ),
            min_reliability=float(
                os.getenv("REMTP_BLOCK_MIN_RELIABILITY", "0.2")
            ),
            min_target_probability=float(
                os.getenv("REMTP_BLOCK_MIN_TARGET_PROBABILITY", "0.001")
            ),
            max_target_log_gap=float(
                os.getenv("REMTP_BLOCK_MAX_TARGET_LOG_GAP", "8.0")
            ),
            min_prefix_reach_probability=float(
                os.getenv("REMTP_BLOCK_MIN_PREFIX_REACH", "0.01")
            ),
            trust_token_reference=float(
                os.getenv("REMTP_BLOCK_TRUST_TOKEN_REFERENCE", "0.25")
            ),
            trust_reliability_reference=float(
                os.getenv(
                    "REMTP_BLOCK_TRUST_RELIABILITY_REFERENCE",
                    "0.6",
                )
            ),
            trust_acceptance_reference=float(
                os.getenv(
                    "REMTP_BLOCK_TRUST_ACCEPTANCE_REFERENCE",
                    "0.01",
                )
            ),
        )
        config.validate()
        return config


@dataclass
class BlockFeatureResult:
    probs: torch.Tensor | None
    safe: torch.Tensor
    target_candidate_probs: torch.Tensor
    draft_candidate_probs: torch.Tensor
    boosted_candidate_probs: torch.Tensor
    target_candidate_ranks: torch.Tensor
    target_candidate_log_gaps: torch.Tensor
    token_support: torch.Tensor
    current_js_similarity: torch.Tensor
    current_distribution_consistency: torch.Tensor
    future_distribution_consistency: torch.Tensor
    reliability: torch.Tensor
    strict_acceptance: torch.Tensor
    strict_rejection_probability: torch.Tensor
    prefix_reach_probability: torch.Tensor
    prefix_value: torch.Tensor
    priority: torch.Tensor
    delta_capacity: torch.Tensor
    trusted_delta_capacity: torch.Tensor
    allocated_delta: torch.Tensor
    realized_kl: torch.Tensor | None
    realized_tv: torch.Tensor | None


def bernoulli_kl(
    boosted: torch.Tensor,
    original: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(Bernoulli(boosted) || Bernoulli(original))."""
    dtype = boosted.dtype
    work_dtype = torch.float32 if boosted.is_cuda else torch.float64
    upper = 1.0 - torch.finfo(work_dtype).eps
    boosted_work = boosted.to(work_dtype).clamp(1e-12, upper)
    original_work = original.to(work_dtype).clamp(1e-12, upper)
    kl = boosted_work * torch.log(boosted_work / original_work)
    kl += (1.0 - boosted_work) * torch.log(
        (1.0 - boosted_work) / (1.0 - original_work)
    )
    return kl.to(dtype)


@dataclass
class TargetLedJSAgreement:
    target_top_ids: torch.Tensor
    target_top_probs: torch.Tensor
    js_similarity: torch.Tensor


def target_led_js_agreement(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    top_k: int,
) -> TargetLedJSAgreement:
    """Compare P/Q on target top-k, the candidate, and one tail bucket.

    This target-led support needs only one top-k operation. If Q concentrates
    on tokens outside target top-k, that mass appears in Q's tail bucket and
    lowers JS similarity without separately computing Q top-k.
    """
    if target_probs.shape != draft_probs.shape or target_probs.ndim != 2:
        raise ValueError("target_probs and draft_probs must share [rows, vocab]")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must have shape [rows]")
    if draft_token_ids.shape[0] != target_probs.shape[0]:
        raise ValueError("draft token rows must match probability rows")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    k = min(top_k, target_probs.shape[-1])
    target_top_probs, target_top_ids = torch.topk(
        target_probs,
        k=k,
        dim=-1,
        sorted=True,
    )
    ids = draft_token_ids.to(
        device=target_top_ids.device,
        dtype=torch.int64,
    )
    candidate_is_new = ~(
        target_top_ids == ids.unsqueeze(1)
    ).any(dim=-1, keepdim=True)
    support_ids = torch.cat((target_top_ids, ids.unsqueeze(1)), dim=-1)
    support_weights = torch.cat(
        (
            torch.ones_like(target_top_probs),
            candidate_is_new.to(target_top_probs.dtype),
        ),
        dim=-1,
    )
    target_vector = target_probs.gather(1, support_ids) * support_weights
    draft_vector = draft_probs.gather(1, support_ids) * support_weights

    # One additional bucket preserves all mass outside the selected support.
    target_tail = (
        1.0 - target_vector.sum(dim=-1, keepdim=True)
    ).clamp_min(0.0)
    draft_tail = (
        1.0 - draft_vector.sum(dim=-1, keepdim=True)
    ).clamp_min(0.0)
    target_js = torch.cat((target_vector, target_tail), dim=-1)
    draft_js = torch.cat((draft_vector, draft_tail), dim=-1)
    midpoint = 0.5 * (target_js + draft_js)
    target_term = torch.where(
        target_js > 0,
        target_js
        * (
            torch.log(target_js.clamp_min(1e-30))
            - torch.log(midpoint.clamp_min(1e-30))
        ),
        torch.zeros_like(target_js),
    )
    draft_term = torch.where(
        draft_js > 0,
        draft_js
        * (
            torch.log(draft_js.clamp_min(1e-30))
            - torch.log(midpoint.clamp_min(1e-30))
        ),
        torch.zeros_like(draft_js),
    )
    js_divergence = 0.5 * (
        target_term.sum(dim=-1) + draft_term.sum(dim=-1)
    )
    js_similarity = (
        1.0 - js_divergence / 0.6931471805599453
    ).clamp(0.0, 1.0)
    return TargetLedJSAgreement(
        target_top_ids=target_top_ids,
        target_top_probs=target_top_probs,
        js_similarity=js_similarity,
    )


def _future_distribution_consistency(
    current_consistency: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    """Weighted P/Q agreement over positions strictly after each token."""
    rows = current_consistency.shape[0]
    result = torch.zeros_like(current_consistency)
    for depth in range(rows - 1):
        offsets = torch.arange(
            rows - depth - 1,
            device=current_consistency.device,
            dtype=current_consistency.dtype,
        )
        weights = torch.pow(
            torch.full_like(offsets, decay),
            offsets,
        )
        result[depth] = (
            current_consistency[depth + 1 :] * weights
        ).sum() / weights.sum().clamp_min(1e-12)
    return result


def _prefix_marginal_values(
    strict_acceptance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Probability of reaching d and its marginal accepted-length value.

    For first-rejection verification,

        E[length] = 1 + sum_k prod_{i<=k} A_i.

    The derivative with respect to A_d is the probability of reaching d times
    the expected number of downstream draft/bonus positions it unlocks.
    """
    rows = strict_acceptance.shape[0]
    reach = torch.ones_like(strict_acceptance)
    for depth in range(1, rows):
        reach[depth] = reach[depth - 1] * strict_acceptance[depth - 1]

    suffix_value = torch.ones_like(strict_acceptance)
    for depth in range(rows - 2, -1, -1):
        suffix_value[depth] = (
            1.0
            + strict_acceptance[depth + 1]
            * suffix_value[depth + 1]
        )
    return reach, reach * suffix_value


def _allocate_capped_delta_fast(
    capacities: torch.Tensor,
    priorities: torch.Tensor,
    total_budget: float,
) -> torch.Tensor:
    """Two-pass capped allocation without iterative water filling."""
    budget = torch.as_tensor(
        total_budget,
        device=capacities.device,
        dtype=capacities.dtype,
    )
    active_priority = torch.where(
        capacities > 1e-12,
        priorities,
        torch.zeros_like(priorities),
    )
    first = torch.minimum(
        budget
        * active_priority
        / active_priority.sum().clamp_min(1e-30),
        capacities,
    )

    # One redistribution recovers most budget stranded by positions that need
    # very little delta to reach q(y). Avoiding a variable-length loop keeps
    # the verifier launch count fixed.
    residual_capacity = (capacities - first).clamp_min(0.0)
    residual_priority = torch.where(
        residual_capacity > 1e-12,
        priorities,
        torch.zeros_like(priorities),
    )
    remaining = (budget - first.sum()).clamp_min(0.0)
    second = torch.minimum(
        remaining
        * residual_priority
        / residual_priority.sum().clamp_min(1e-30),
        residual_capacity,
    )
    return first + second


def _closed_form_boost(
    original: torch.Tensor,
    upper: torch.Tensor,
    allocated_delta: torch.Tensor,
) -> torch.Tensor:
    """Cactus/Pinsker closed-form candidate boost capped at q(y)."""
    bonus = torch.sqrt(
        (
            2.0
            * allocated_delta
            * original
            * (1.0 - original)
        ).clamp_min(0.0)
    )
    boosted = torch.minimum(original + bonus, upper)
    return torch.where(allocated_delta > 0, boosted, original)


def block_feature_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: BlockFeatureConfig,
    *,
    assume_normalized: bool = False,
    construct_probs: bool = True,
    compute_shift_metrics: bool = True,
) -> BlockFeatureResult:
    """Construct block-conditioned temporary verifier distributions."""
    config.validate()
    if target_probs.ndim != 2:
        raise ValueError("target_probs must have shape [tokens, vocab]")
    if target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must have equal shape")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must have shape [tokens]")
    if target_probs.shape[0] != draft_token_ids.shape[0]:
        raise ValueError("probability rows and draft token IDs must match")
    if not 1 <= target_probs.shape[0] <= config.expected_draft_tokens:
        raise ValueError(
            "draft block length must be in [1, expected_draft_tokens]"
        )

    target = target_probs.to(torch.float32)
    draft = draft_probs.to(
        device=target.device,
        dtype=torch.float32,
    )
    if not assume_normalized:
        target = target.clamp_min(0.0)
        draft = draft.clamp_min(0.0)
        target = target / target.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-30)
        draft = draft / draft.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-30)
    ids = draft_token_ids.to(device=target.device, dtype=torch.int64)
    rows = torch.arange(target.shape[0], device=target.device)

    target_candidate = target[rows, ids]
    draft_candidate = draft[rows, ids]
    needs_distribution = (
        config.use_current_distribution
        or config.use_future_distribution
    )
    if needs_distribution:
        agreement = target_led_js_agreement(
            target,
            draft,
            ids,
            config.distribution_top_k,
        )
        target_top_ids = agreement.target_top_ids
        target_top_probs = agreement.target_top_probs
        current_js_similarity = agreement.js_similarity
        current_distribution = current_js_similarity
    else:
        topk_size = min(config.distribution_top_k, target.shape[-1])
        target_top_probs, target_top_ids = torch.topk(
            target,
            k=topk_size,
            dim=-1,
            sorted=True,
        )
        current_js_similarity = torch.zeros_like(target_candidate)
        current_distribution = torch.zeros_like(target_candidate)

    candidate_matches = target_top_ids == ids.unsqueeze(1)
    candidate_in_topk = candidate_matches.any(dim=-1)
    candidate_topk_rank = (
        candidate_matches.to(torch.int64).argmax(dim=-1) + 1
    )
    candidate_ranks = torch.where(
        candidate_in_topk,
        candidate_topk_rank,
        torch.full_like(
            candidate_topk_rank,
            target_top_ids.shape[-1] + 1,
        ),
    )
    target_top_probs = target_top_probs[:, 0]
    candidate_log_gaps = (
        torch.log(target_top_probs.clamp_min(1e-30))
        - torch.log(target_candidate.clamp_min(1e-30))
    ).clamp_min(0.0)

    rank_support = torch.exp(
        -(candidate_ranks.to(torch.float32) - 1.0)
        / config.token_rank_scale
    )
    gap_support = torch.exp(
        -candidate_log_gaps / config.token_log_gap_scale
    )
    token_support = torch.sqrt(rank_support * gap_support).clamp(0.0, 1.0)

    future_distribution = _future_distribution_consistency(
        current_distribution,
        config.future_decay,
    )

    reliability_numerator = config.token_signal_weight * token_support
    reliability_denominator = torch.full_like(
        token_support,
        config.token_signal_weight,
    )
    if config.use_current_distribution:
        reliability_numerator = (
            reliability_numerator
            + config.current_distribution_weight * current_distribution
        )
        reliability_denominator = (
            reliability_denominator
            + config.current_distribution_weight
        )
    if config.use_future_distribution and target.shape[0] > 1:
        has_future = torch.arange(
            target.shape[0],
            device=target.device,
        ) < target.shape[0] - 1
        future_weight = torch.where(
            has_future,
            torch.full_like(
                token_support,
                config.future_distribution_weight,
            ),
            torch.zeros_like(token_support),
        )
        reliability_numerator = (
            reliability_numerator
            + future_weight * future_distribution
        )
        reliability_denominator = (
            reliability_denominator + future_weight
        )
    reliability = (
        reliability_numerator / reliability_denominator.clamp_min(1e-30)
    ).clamp(0.0, 1.0)

    # h_d(y_d) is only increased above p_d(y_d). With the same verification
    # uniform u, every event accepted by strict p/q verification remains
    # accepted; relaxation only adds a rescue band above p/q.
    strict_acceptance = torch.minimum(
        torch.ones_like(target_candidate),
        target_candidate / draft_candidate.clamp_min(1e-30),
    )
    strict_rejection_probability = 1.0 - strict_acceptance
    reach_probability, marginal_value = _prefix_marginal_values(
        strict_acceptance
    )
    if config.use_prefix_value:
        prefix_value = marginal_value
    else:
        prefix_value = torch.ones_like(target_candidate)

    safe = (
        (target_candidate >= config.min_target_probability)
        & (candidate_log_gaps <= config.max_target_log_gap)
        & (reliability >= config.min_reliability)
    )
    if config.use_prefix_value:
        safe &= reach_probability >= config.min_prefix_reach_probability

    rescue_opportunity = (
        strict_acceptance.sqrt() * strict_rejection_probability
    )
    priority = torch.where(
        safe,
        rescue_opportunity
        * prefix_value
        * token_support
        * reliability.pow(config.reliability_power),
        torch.zeros_like(strict_rejection_probability),
    )
    candidate_cap = torch.where(
        draft_candidate > target_candidate,
        draft_candidate.clamp_max(1.0 - 1e-6),
        target_candidate,
    )
    delta_capacity = (
        (candidate_cap - target_candidate).square()
        / (
            2.0
            * target_candidate
            * (1.0 - target_candidate)
        ).clamp_min(1e-30)
    )
    delta_capacity = torch.where(
        safe & (draft_candidate > target_candidate),
        delta_capacity,
        torch.zeros_like(delta_capacity),
    )
    # A low-A candidate is more likely to be a genuine MTP error than a useful
    # alternate expression. Restrict its maximum spend even when it is the only
    # active position, so relative priority normalization cannot give it the
    # whole block budget.
    token_gate = (
        token_support / config.trust_token_reference
    ).clamp(0.0, 1.0)
    reliability_gate = (
        reliability / config.trust_reliability_reference
    ).clamp(0.0, 1.0).pow(config.reliability_power)
    acceptance_gate = (
        strict_acceptance / config.trust_acceptance_reference
    ).clamp(0.0, 1.0)
    trust_gate = token_gate * reliability_gate * acceptance_gate
    trusted_delta_capacity = delta_capacity * trust_gate
    allocated_delta = _allocate_capped_delta_fast(
        trusted_delta_capacity,
        priority,
        config.block_delta_budget,
    )
    boosted = _closed_form_boost(
        target_candidate,
        candidate_cap,
        allocated_delta,
    )

    relaxed = None
    realized_boosted = boosted
    if construct_probs:
        scale = (
            (1.0 - boosted) / (1.0 - target_candidate)
        ).nan_to_num(
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        relaxed = target * scale.unsqueeze(-1)
        relaxed.scatter_(1, ids.unsqueeze(1), boosted.unsqueeze(1))
        relaxed = relaxed.clamp_min(0.0)
        relaxed /= relaxed.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        realized_boosted = relaxed[rows, ids]
    if compute_shift_metrics:
        realized_kl = bernoulli_kl(realized_boosted, target_candidate)
        realized_tv = (realized_boosted - target_candidate).abs()
    else:
        realized_kl = None
        realized_tv = None
    return BlockFeatureResult(
        probs=relaxed,
        safe=safe,
        target_candidate_probs=target_candidate,
        draft_candidate_probs=draft_candidate,
        boosted_candidate_probs=realized_boosted,
        target_candidate_ranks=candidate_ranks,
        target_candidate_log_gaps=candidate_log_gaps,
        token_support=token_support,
        current_js_similarity=current_js_similarity,
        current_distribution_consistency=current_distribution,
        future_distribution_consistency=future_distribution,
        reliability=reliability,
        strict_acceptance=strict_acceptance,
        strict_rejection_probability=strict_rejection_probability,
        prefix_reach_probability=reach_probability,
        prefix_value=prefix_value,
        priority=priority,
        delta_capacity=delta_capacity,
        trusted_delta_capacity=trusted_delta_capacity,
        allocated_delta=allocated_delta,
        realized_kl=realized_kl,
        realized_tv=realized_tv,
    )


def block_feature_candidate_probs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: BlockFeatureConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return only p(y) and h(y) for the compiled serving fast path."""
    result = block_feature_distribution(
        target_probs,
        draft_probs,
        draft_token_ids,
        config,
        assume_normalized=True,
        construct_probs=False,
        compute_shift_metrics=False,
    )
    return (
        result.target_candidate_probs,
        result.boosted_candidate_probs,
    )


def greedy_verification_ids(
    target_probs: torch.Tensor,
    relaxed_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return candidate if relaxed-top1, otherwise original target top-1."""
    token_ids = draft_token_ids.to(
        device=target_probs.device,
        dtype=torch.int64,
    )
    relaxed_top = relaxed_probs.argmax(dim=-1)
    target_top = target_probs.argmax(dim=-1)
    return torch.where(token_ids == relaxed_top, token_ids, target_top)


def boost_candidate_logits(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    original_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
    *,
    copy: bool = True,
) -> torch.Tensor:
    """Apply one-token boosts while preserving other-token logit gaps."""
    logits = target_logits.to(torch.float32)
    if copy:
        logits = logits.clone()
    ids = draft_token_ids.to(device=logits.device, dtype=torch.int64)
    rows = torch.arange(logits.shape[0], device=logits.device)
    original = original_candidate_probs.clamp(1e-12, 1.0 - 1e-6)
    boosted = boosted_candidate_probs.clamp(1e-12, 1.0 - 1e-6)
    logits[rows, ids] += torch.logit(boosted) - torch.logit(original)
    return logits


def _block_feature_rejection_sample_v1(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Replace p by a full-block-informed h before standard verification."""
    global _AUDIT_ROUND, _AUDIT_WINDOW, _AUDIT_WINDOW_COUNT
    global _DIAGNOSTIC_EMITTED

    original = getattr(
        _block_feature_rejection_sample_v1,
        "_remtp_original",
    )
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
            "BlockFeature MTP currently requires --max-num-seqs 1"
        )

    config = getattr(
        _block_feature_rejection_sample_v1,
        "_remtp_config",
    )
    audit_interval = getattr(
        _block_feature_rejection_sample_v1,
        "_remtp_audit_interval",
    )
    diagnostics_enabled = getattr(
        _block_feature_rejection_sample_v1,
        "_remtp_diagnostics_enabled",
    )
    if max_spec_len != config.expected_draft_tokens:
        raise RuntimeError(
            "BlockFeature MTP was configured for "
            f"{config.expected_draft_tokens} draft tokens, "
            f"but vLLM max_spec_len={max_spec_len}"
        )

    logits = target_logits.to(torch.float32)
    target_probs = torch.softmax(logits, dim=-1)
    need_details = (
        audit_interval > 0
        or (diagnostics_enabled and not _DIAGNOSTIC_EMITTED)
    )
    result: BlockFeatureResult | None = None
    if need_details:
        result = block_feature_distribution(
            target_probs,
            draft_probs,
            draft_token_ids,
            config,
            assume_normalized=True,
            construct_probs=False,
            compute_shift_metrics=True,
        )
        target_candidate_probs = result.target_candidate_probs
        boosted_candidate_probs = result.boosted_candidate_probs
    else:
        fast_candidate_probs = getattr(
            _block_feature_rejection_sample_v1,
            "_remtp_fast_candidate_probs",
        )
        target_candidate_probs, boosted_candidate_probs = (
            fast_candidate_probs(
                target_probs,
                draft_probs,
                draft_token_ids,
            )
        )

    if (
        result is not None
        and not _DIAGNOSTIC_EMITTED
        and diagnostics_enabled
        and result.allocated_delta.sum().item() > 0
    ):
        assert result.realized_kl is not None
        assert result.realized_tv is not None
        print(
            "[ReMTP][BlockFeature][diagnostic] "
            f"draft_ids={draft_token_ids.tolist()} "
            f"safe={result.safe.tolist()} "
            f"p(D)={result.target_candidate_probs.tolist()} "
            f"q(D)={result.draft_candidate_probs.tolist()} "
            f"h(D)={result.boosted_candidate_probs.tolist()} "
            f"target_topk_rank={result.target_candidate_ranks.tolist()} "
            f"log_gap={result.target_candidate_log_gaps.tolist()} "
            f"token_support={result.token_support.tolist()} "
            f"js_similarity={result.current_js_similarity.tolist()} "
            f"current_dist={result.current_distribution_consistency.tolist()} "
            f"future_dist={result.future_distribution_consistency.tolist()} "
            f"reliability={result.reliability.tolist()} "
            f"strict_A={result.strict_acceptance.tolist()} "
            f"prefix_reach={result.prefix_reach_probability.tolist()} "
            f"prefix_value={result.prefix_value.tolist()} "
            f"priority={result.priority.tolist()} "
            f"delta_cap={result.delta_capacity.tolist()} "
            f"trusted_delta_cap={result.trusted_delta_capacity.tolist()} "
            f"delta_alloc={result.allocated_delta.tolist()} "
            f"KL_realized={result.realized_kl.tolist()} "
            f"TV_realized={result.realized_tv.tolist()} "
            f"KL_total={result.realized_kl.sum().item():.6f} "
            f"TV_total={result.realized_tv.sum().item():.6f}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True

    if audit_interval > 0:
        assert result is not None
        assert result.realized_kl is not None
        assert result.realized_tv is not None
        _AUDIT_ROUND += 1
        _AUDIT_WINDOW_COUNT += 1
        budgeted = result.allocated_delta > 0
        low_target = result.target_candidate_probs < 1e-3
        low_target_delta = torch.where(
            budgeted & low_target,
            result.allocated_delta,
            torch.zeros_like(result.allocated_delta),
        ).sum()
        total_delta = result.allocated_delta.sum()
        total_kl = result.realized_kl.sum()
        total_tv = result.realized_tv.sum()
        min_budgeted_p = torch.where(
            budgeted,
            result.target_candidate_probs,
            torch.ones_like(result.target_candidate_probs),
        ).min()
        max_budgeted_gap = torch.where(
            budgeted,
            result.target_candidate_log_gaps,
            torch.zeros_like(result.target_candidate_log_gaps),
        ).max()
        round_audit = torch.stack(
            (
                total_delta,
                total_kl,
                total_tv,
                low_target_delta,
                min_budgeted_p,
                max_budgeted_gap,
                total_kl,
                total_tv,
            )
        )
        if _AUDIT_WINDOW is None:
            _AUDIT_WINDOW = round_audit
        else:
            _AUDIT_WINDOW[:4] += round_audit[:4]
            _AUDIT_WINDOW[4] = torch.minimum(
                _AUDIT_WINDOW[4],
                round_audit[4],
            )
            _AUDIT_WINDOW[5] = torch.maximum(
                _AUDIT_WINDOW[5],
                round_audit[5],
            )
            _AUDIT_WINDOW[6:] = torch.maximum(
                _AUDIT_WINDOW[6:],
                round_audit[6:],
            )
        if _AUDIT_WINDOW_COUNT == audit_interval:
            audit = _AUDIT_WINDOW.tolist()
            count = _AUDIT_WINDOW_COUNT
            print(
                "[ReMTP][BlockFeature][audit_window] "
                f"rounds={_AUDIT_ROUND - count + 1}-{_AUDIT_ROUND} "
                f"mean_delta={audit[0] / count:.6f} "
                "mean_sum_position_KL="
                f"{audit[1] / count:.6f} "
                f"max_sum_position_KL={audit[6]:.6f} "
                "mean_sum_position_TV="
                f"{audit[2] / count:.6f} "
                f"max_sum_position_TV={audit[7]:.6f} "
                "low_p_delta_fraction="
                f"{audit[3] / max(audit[0], 1e-30):.6f} "
                f"min_budgeted_p={audit[4]:.6e} "
                f"max_budgeted_log_gap={audit[5]:.6f}",
                flush=True,
            )
            _AUDIT_WINDOW = None
            _AUDIT_WINDOW_COUNT = 0

    original_top_ids = (
        logits.argmax(dim=-1) if sampling_metadata.all_greedy else None
    )
    relaxed_logits = boost_candidate_logits(
        logits,
        draft_token_ids,
        target_candidate_probs,
        boosted_candidate_probs,
        copy=False,
    )
    if sampling_metadata.all_greedy:
        assert original_top_ids is not None
        relaxed_top_ids = relaxed_logits.argmax(dim=-1)
        desired_ids = torch.where(
            draft_token_ids == relaxed_top_ids,
            draft_token_ids,
            original_top_ids,
        )
        proxy_logits = torch.full_like(target_logits, float("-inf"))
        proxy_logits.scatter_(1, desired_ids.unsqueeze(1), 0.0)
        verification_logits = proxy_logits
    else:
        verification_logits = relaxed_logits

    return original(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        verification_logits,
        bonus_token_ids,
        sampling_metadata,
    )


def install_block_feature_mtp() -> None:
    """Install block-aware feature-consistent verification."""
    config = BlockFeatureConfig.from_env()
    audit_interval = int(os.getenv("REMTP_BLOCK_AUDIT_INTERVAL", "0"))
    if audit_interval < 0:
        raise ValueError("REMTP_BLOCK_AUDIT_INTERVAL must be non-negative")
    diagnostics_enabled = _env_flag("REMTP_BLOCK_DIAGNOSTICS", False)
    compile_fast_path = _env_flag("REMTP_BLOCK_COMPILE", True)

    def fast_candidate_probs(
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return block_feature_candidate_probs(
            target_probs,
            draft_probs,
            draft_token_ids,
            config,
        )

    if compile_fast_path:
        fast_candidate_probs = torch.compile(
            fast_candidate_probs,
            fullgraph=True,
            dynamic=False,
        )

    module = importlib.import_module(_V1_REJECTION_MODULE)
    current = module.rejection_sample
    if not getattr(current, "_remtp_block_feature_mtp", False):
        wrapper = _block_feature_rejection_sample_v1
        wrapper._remtp_block_feature_mtp = True
        wrapper._remtp_original = current
        wrapper._remtp_config = config
        wrapper._remtp_audit_interval = audit_interval
        wrapper._remtp_diagnostics_enabled = diagnostics_enabled
        wrapper._remtp_fast_candidate_probs = fast_candidate_probs
        module.rejection_sample = wrapper

    print(
        "[ReMTP][BlockFeature] "
        "fast_closed_form=1 "
        f"block_delta_budget={config.block_delta_budget:g} "
        f"draft_tokens={config.expected_draft_tokens} "
        f"prefix_value={int(config.use_prefix_value)} "
        f"current_distribution={int(config.use_current_distribution)} "
        f"future_distribution={int(config.use_future_distribution)} "
        f"future_decay={config.future_decay:g} "
        f"top_k={config.distribution_top_k} "
        f"min_reliability={config.min_reliability:g} "
        f"min_target_p={config.min_target_probability:g} "
        f"max_target_log_gap={config.max_target_log_gap:g} "
        f"min_prefix_reach={config.min_prefix_reach_probability:g} "
        "trust_refs="
        f"{config.trust_token_reference:g}/"
        f"{config.trust_reliability_reference:g}/"
        f"{config.trust_acceptance_reference:g} "
        f"audit_interval={audit_interval} "
        f"diagnostics={int(diagnostics_enabled)} "
        f"compiled={int(compile_fast_path)} "
        "draft=probabilistic-MTP bonus=original-target",
        flush=True,
    )
