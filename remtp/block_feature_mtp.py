"""Block-aware, feature-consistent relaxation for probabilistic MTP.

The verifier sees the complete MTP block in one target forward pass.  This
module uses that information to allocate one KL budget across the whole block
instead of independently relaxing each token:

* target rank and target top-1 gap for the sampled draft token;
* full-distribution top-k agreement between MTP q_d and target p_d;
* agreement at later positions in the sampled MTP block;
* the number of later draft/bonus tokens unlocked by an accepted prefix.

The implementation targets this repository's single-request, four-token MTP
experiments.  It keeps vLLM's standard stochastic acceptance and residual
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

    block_kl_budget: float = 1.2
    expected_draft_tokens: int = 4
    use_prefix_value: bool = True
    use_current_distribution: bool = True
    use_future_distribution: bool = True
    future_decay: float = 0.7
    distribution_top_k: int = 8
    token_rank_scale: float = 4.0
    token_log_gap_scale: float = 2.0
    overlap_weight: float = 0.4
    probability_cosine_weight: float = 0.4
    entropy_consistency_weight: float = 0.2
    token_signal_weight: float = 1.0
    current_distribution_weight: float = 1.0
    future_distribution_weight: float = 1.0
    reliability_power: float = 2.0
    min_reliability: float = 0.2
    max_normalized_surprisal: float = 12.0
    min_prefix_reach_probability: float = 0.01
    bisection_steps: int = 20

    def validate(self) -> None:
        if self.block_kl_budget < 0:
            raise ValueError("block_kl_budget must be non-negative")
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if not 0 <= self.future_decay <= 1:
            raise ValueError("future_decay must be in [0, 1]")
        if self.distribution_top_k < 1:
            raise ValueError("distribution_top_k must be positive")
        if self.token_rank_scale <= 0 or self.token_log_gap_scale <= 0:
            raise ValueError("token support scales must be positive")
        distribution_weights = (
            self.overlap_weight,
            self.probability_cosine_weight,
            self.entropy_consistency_weight,
        )
        signal_weights = (
            self.token_signal_weight,
            self.current_distribution_weight,
            self.future_distribution_weight,
        )
        weights = distribution_weights + signal_weights
        if any(weight < 0 for weight in weights):
            raise ValueError("signal weights must be non-negative")
        if sum(distribution_weights) <= 0:
            raise ValueError(
                "at least one distribution-agreement weight must be positive"
            )
        if self.token_signal_weight <= 0:
            raise ValueError("token_signal_weight must be positive")
        if self.reliability_power <= 0:
            raise ValueError("reliability_power must be positive")
        if not 0 <= self.min_reliability <= 1:
            raise ValueError("min_reliability must be in [0, 1]")
        if self.max_normalized_surprisal <= 0:
            raise ValueError("max_normalized_surprisal must be positive")
        if not 0 <= self.min_prefix_reach_probability <= 1:
            raise ValueError(
                "min_prefix_reach_probability must be in [0, 1]"
            )
        if self.bisection_steps < 1:
            raise ValueError("bisection_steps must be positive")

    @classmethod
    def from_env(cls) -> BlockFeatureConfig:
        config = cls(
            block_kl_budget=float(
                os.getenv("REMTP_BLOCK_KL_BUDGET", "1.2")
            ),
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
            overlap_weight=float(
                os.getenv("REMTP_BLOCK_OVERLAP_WEIGHT", "0.4")
            ),
            probability_cosine_weight=float(
                os.getenv("REMTP_BLOCK_PROBABILITY_COSINE_WEIGHT", "0.4")
            ),
            entropy_consistency_weight=float(
                os.getenv("REMTP_BLOCK_ENTROPY_WEIGHT", "0.2")
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
            max_normalized_surprisal=float(
                os.getenv(
                    "REMTP_BLOCK_MAX_NORMALIZED_SURPRISAL",
                    "12.0",
                )
            ),
            min_prefix_reach_probability=float(
                os.getenv("REMTP_BLOCK_MIN_PREFIX_REACH", "0.01")
            ),
            bisection_steps=int(
                os.getenv("REMTP_BLOCK_BISECTION_STEPS", "20")
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
    target_entropy: torch.Tensor
    draft_entropy: torch.Tensor
    normalized_surprisal: torch.Tensor
    token_support: torch.Tensor
    topk_overlap: torch.Tensor
    topk_probability_cosine: torch.Tensor
    entropy_consistency: torch.Tensor
    current_distribution_consistency: torch.Tensor
    future_distribution_consistency: torch.Tensor
    reliability: torch.Tensor
    strict_acceptance: torch.Tensor
    strict_rejection_probability: torch.Tensor
    prefix_reach_probability: torch.Tensor
    prefix_value: torch.Tensor
    priority: torch.Tensor
    kl_capacity: torch.Tensor
    allocated_kl: torch.Tensor
    realized_kl: torch.Tensor


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
class DistributionAgreement:
    target_top_ids: torch.Tensor
    target_top_probs: torch.Tensor
    draft_top_ids: torch.Tensor
    draft_top_probs: torch.Tensor
    target_entropy: torch.Tensor
    draft_entropy: torch.Tensor
    topk_overlap: torch.Tensor
    probability_cosine: torch.Tensor
    entropy_consistency: torch.Tensor
    combined: torch.Tensor


def topk_distribution_agreement(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    top_k: int,
    overlap_weight: float,
    probability_cosine_weight: float,
    entropy_consistency_weight: float,
) -> DistributionAgreement:
    """Compare complete MTP and target distributions on aligned top-k support."""
    if target_probs.shape != draft_probs.shape or target_probs.ndim != 2:
        raise ValueError("target_probs and draft_probs must share [rows, vocab]")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    weights = (
        overlap_weight,
        probability_cosine_weight,
        entropy_consistency_weight,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("distribution-agreement weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError(
            "at least one distribution-agreement weight must be positive"
        )

    k = min(top_k, target_probs.shape[-1])
    target_top_probs, target_top_ids = torch.topk(
        target_probs,
        k=k,
        dim=-1,
        sorted=True,
    )
    draft_top_probs, draft_top_ids = torch.topk(
        draft_probs,
        k=k,
        dim=-1,
        sorted=True,
    )

    matches = (
        target_top_ids.unsqueeze(2) == draft_top_ids.unsqueeze(1)
    )
    overlap = matches.any(dim=2).to(torch.float32).mean(dim=-1)

    draft_is_new = ~matches.any(dim=1)
    support_ids = torch.cat((target_top_ids, draft_top_ids), dim=-1)
    support_weights = torch.cat(
        (
            torch.ones_like(target_top_probs),
            draft_is_new.to(target_top_probs.dtype),
        ),
        dim=-1,
    )
    target_vector = target_probs.gather(1, support_ids) * support_weights
    draft_vector = draft_probs.gather(1, support_ids) * support_weights
    numerator = (target_vector * draft_vector).sum(dim=-1)
    denominator = (
        target_vector.square().sum(dim=-1).sqrt()
        * draft_vector.square().sum(dim=-1).sqrt()
    )
    probability_cosine = (
        numerator / denominator.clamp_min(1e-30)
    ).clamp(0.0, 1.0)

    target_entropy = -(
        target_probs * torch.log(target_probs.clamp_min(1e-30))
    ).sum(dim=-1)
    draft_entropy = -(
        draft_probs * torch.log(draft_probs.clamp_min(1e-30))
    ).sum(dim=-1)
    entropy_consistency = (
        1.0
        - (target_entropy - draft_entropy).abs()
        / (target_entropy + draft_entropy).clamp_min(1e-6)
    ).clamp(0.0, 1.0)

    weight_sum = (
        overlap_weight
        + probability_cosine_weight
        + entropy_consistency_weight
    )
    combined = (
        overlap_weight * overlap
        + probability_cosine_weight * probability_cosine
        + entropy_consistency_weight * entropy_consistency
    ) / weight_sum
    return DistributionAgreement(
        target_top_ids=target_top_ids,
        target_top_probs=target_top_probs,
        draft_top_ids=draft_top_ids,
        draft_top_probs=draft_top_probs,
        target_entropy=target_entropy,
        draft_entropy=draft_entropy,
        topk_overlap=overlap,
        probability_cosine=probability_cosine,
        entropy_consistency=entropy_consistency,
        combined=combined,
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


def _allocate_capped_block_budget(
    capacities: torch.Tensor,
    priorities: torch.Tensor,
    total_budget: float,
) -> torch.Tensor:
    """Weighted water filling with per-position KL capacity caps."""
    allocation = torch.zeros_like(capacities)
    budget = torch.as_tensor(
        total_budget,
        device=capacities.device,
        dtype=capacities.dtype,
    )
    for _ in range(capacities.shape[0] + 1):
        remaining_capacity = (capacities - allocation).clamp_min(0.0)
        active = (remaining_capacity > 1e-12) & (priorities > 0)
        active_priority = torch.where(
            active,
            priorities,
            torch.zeros_like(priorities),
        )
        remaining_budget = (budget - allocation.sum()).clamp_min(0.0)
        share = (
            remaining_budget
            * active_priority
            / active_priority.sum().clamp_min(1e-30)
        )
        grant = torch.minimum(share, remaining_capacity)
        allocation = allocation + torch.where(
            active,
            grant,
            torch.zeros_like(grant),
        )
    return allocation


def _solve_boosted_probability(
    original: torch.Tensor,
    upper: torch.Tensor,
    epsilon: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    upper = torch.maximum(upper, original)
    low = original.clone()
    high = upper
    active = (epsilon > 0) & (upper > original)
    for _ in range(steps):
        midpoint = (low + high) * 0.5
        within_budget = bernoulli_kl(midpoint, original) <= epsilon
        low = torch.where(active & within_budget, midpoint, low)
        high = torch.where(active & ~within_budget, midpoint, high)
    return torch.where(active, low, original)


def block_feature_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: BlockFeatureConfig,
    *,
    assume_normalized: bool = False,
    construct_probs: bool = True,
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
    agreement = topk_distribution_agreement(
        target,
        draft,
        config.distribution_top_k,
        config.overlap_weight,
        config.probability_cosine_weight,
        config.entropy_consistency_weight,
    )
    candidate_ranks = 1 + (
        target > target_candidate.unsqueeze(-1)
    ).sum(dim=-1)
    target_top_probs = agreement.target_top_probs[:, 0]
    candidate_log_gaps = (
        torch.log(target_top_probs.clamp_min(1e-30))
        - torch.log(target_candidate.clamp_min(1e-30))
    ).clamp_min(0.0)

    surprisal = -torch.log(target_candidate.clamp_min(1e-30))
    normalized_surprisal = (
        surprisal / agreement.target_entropy.clamp_min(1e-6)
    )
    rank_support = torch.exp(
        -(candidate_ranks.to(torch.float32) - 1.0)
        / config.token_rank_scale
    )
    gap_support = torch.exp(
        -candidate_log_gaps / config.token_log_gap_scale
    )
    token_support = torch.sqrt(rank_support * gap_support).clamp(0.0, 1.0)

    current_distribution = agreement.combined
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
        (normalized_surprisal <= config.max_normalized_surprisal)
        & (reliability >= config.min_reliability)
    )
    if config.use_prefix_value:
        safe &= reach_probability >= config.min_prefix_reach_probability

    priority = torch.where(
        safe,
        strict_rejection_probability
        * prefix_value
        * reliability.pow(config.reliability_power),
        torch.zeros_like(strict_rejection_probability),
    )
    candidate_cap = torch.where(
        draft_candidate > target_candidate,
        draft_candidate.clamp_max(1.0 - 1e-6),
        target_candidate,
    )
    kl_capacity = bernoulli_kl(candidate_cap, target_candidate)
    kl_capacity = torch.where(
        safe & (draft_candidate > target_candidate),
        kl_capacity,
        torch.zeros_like(kl_capacity),
    )
    allocated_kl = _allocate_capped_block_budget(
        kl_capacity,
        priority,
        config.block_kl_budget,
    )
    boosted = _solve_boosted_probability(
        target_candidate,
        candidate_cap,
        allocated_kl,
        config.bisection_steps,
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
    realized_kl = bernoulli_kl(realized_boosted, target_candidate)
    return BlockFeatureResult(
        probs=relaxed,
        safe=safe,
        target_candidate_probs=target_candidate,
        draft_candidate_probs=draft_candidate,
        boosted_candidate_probs=realized_boosted,
        target_candidate_ranks=candidate_ranks,
        target_candidate_log_gaps=candidate_log_gaps,
        target_entropy=agreement.target_entropy,
        draft_entropy=agreement.draft_entropy,
        normalized_surprisal=normalized_surprisal,
        token_support=token_support,
        topk_overlap=agreement.topk_overlap,
        topk_probability_cosine=agreement.probability_cosine,
        entropy_consistency=agreement.entropy_consistency,
        current_distribution_consistency=current_distribution,
        future_distribution_consistency=future_distribution,
        reliability=reliability,
        strict_acceptance=strict_acceptance,
        strict_rejection_probability=strict_rejection_probability,
        prefix_reach_probability=reach_probability,
        prefix_value=prefix_value,
        priority=priority,
        kl_capacity=kl_capacity,
        allocated_kl=allocated_kl,
        realized_kl=realized_kl,
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
) -> torch.Tensor:
    """Apply one-token boosts while preserving other-token logit gaps."""
    logits = target_logits.to(torch.float32).clone()
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
    if max_spec_len != config.expected_draft_tokens:
        raise RuntimeError(
            "BlockFeature MTP was configured for "
            f"{config.expected_draft_tokens} draft tokens, "
            f"but vLLM max_spec_len={max_spec_len}"
        )

    logits = target_logits.to(torch.float32)
    target_probs = torch.softmax(logits, dim=-1)
    result = block_feature_distribution(
        target_probs,
        draft_probs,
        draft_token_ids,
        config,
        assume_normalized=True,
        construct_probs=False,
    )

    if (
        not _DIAGNOSTIC_EMITTED
        and os.getenv("REMTP_BLOCK_DIAGNOSTICS", "1") == "1"
        and result.target_candidate_probs.sum().item() > 0
    ):
        print(
            "[ReMTP][BlockFeature][diagnostic] "
            f"draft_ids={draft_token_ids.tolist()} "
            f"safe={result.safe.tolist()} "
            f"p(D)={result.target_candidate_probs.tolist()} "
            f"q(D)={result.draft_candidate_probs.tolist()} "
            f"h(D)={result.boosted_candidate_probs.tolist()} "
            f"target_rank={result.target_candidate_ranks.tolist()} "
            f"log_gap={result.target_candidate_log_gaps.tolist()} "
            f"norm_surprisal={result.normalized_surprisal.tolist()} "
            f"token_support={result.token_support.tolist()} "
            f"topk_overlap={result.topk_overlap.tolist()} "
            f"prob_cos={result.topk_probability_cosine.tolist()} "
            f"entropy_match={result.entropy_consistency.tolist()} "
            f"current_dist={result.current_distribution_consistency.tolist()} "
            f"future_dist={result.future_distribution_consistency.tolist()} "
            f"reliability={result.reliability.tolist()} "
            f"strict_A={result.strict_acceptance.tolist()} "
            f"prefix_reach={result.prefix_reach_probability.tolist()} "
            f"prefix_value={result.prefix_value.tolist()} "
            f"priority={result.priority.tolist()} "
            f"KL_alloc={result.allocated_kl.tolist()} "
            f"KL_realized={result.realized_kl.tolist()} "
            f"KL_total={result.realized_kl.sum().item():.6f}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True

    relaxed_logits = boost_candidate_logits(
        logits,
        draft_token_ids,
        result.target_candidate_probs,
        result.boosted_candidate_probs,
    )
    if sampling_metadata.all_greedy:
        original_top_ids = logits.argmax(dim=-1)
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
    module = importlib.import_module(_V1_REJECTION_MODULE)
    current = module.rejection_sample
    if not getattr(current, "_remtp_block_feature_mtp", False):
        wrapper = _block_feature_rejection_sample_v1
        wrapper._remtp_block_feature_mtp = True
        wrapper._remtp_original = current
        wrapper._remtp_config = config
        module.rejection_sample = wrapper

    print(
        "[ReMTP][BlockFeature] "
        f"block_kl_budget={config.block_kl_budget:g} "
        f"draft_tokens={config.expected_draft_tokens} "
        f"prefix_value={int(config.use_prefix_value)} "
        f"current_distribution={int(config.use_current_distribution)} "
        f"future_distribution={int(config.use_future_distribution)} "
        f"future_decay={config.future_decay:g} "
        f"top_k={config.distribution_top_k} "
        f"min_reliability={config.min_reliability:g} "
        f"max_norm_surprisal={config.max_normalized_surprisal:g} "
        f"min_prefix_reach={config.min_prefix_reach_probability:g} "
        "draft=probabilistic-MTP bonus=original-target",
        flush=True,
    )
