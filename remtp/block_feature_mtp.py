"""Block-aware, feature-consistent relaxation for probabilistic MTP.

The verifier sees the complete MTP block in one target forward pass.  This
module uses that information to allocate one KL budget across the whole block
instead of independently relaxing each token:

* entropy-normalized local target support;
* target support at later positions in the sampled MTP block;
* MTP/target consistency after projection through their shared LM head;
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
    use_future_support: bool = True
    use_feature_consistency: bool = True
    future_decay: float = 0.7
    local_weight: float = 1.0
    future_weight: float = 1.0
    consistency_weight: float = 1.0
    rescue_weight: float = 2.0
    max_normalized_surprisal: float = 12.0
    min_feature_consistency: float = 0.02
    min_prefix_reach_probability: float = 0.01
    bisection_steps: int = 20

    def validate(self) -> None:
        if self.block_kl_budget < 0:
            raise ValueError("block_kl_budget must be non-negative")
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if not 0 <= self.future_decay <= 1:
            raise ValueError("future_decay must be in [0, 1]")
        weights = (
            self.local_weight,
            self.future_weight,
            self.consistency_weight,
            self.rescue_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("signal weights must be non-negative")
        if sum(weights[:3]) <= 0:
            raise ValueError("at least one base signal weight must be positive")
        if self.max_normalized_surprisal <= 0:
            raise ValueError("max_normalized_surprisal must be positive")
        if not 0 <= self.min_feature_consistency <= 1:
            raise ValueError("min_feature_consistency must be in [0, 1]")
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
            use_future_support=_env_flag(
                "REMTP_BLOCK_USE_FUTURE_SUPPORT", True
            ),
            use_feature_consistency=_env_flag(
                "REMTP_BLOCK_USE_FEATURE_CONSISTENCY", True
            ),
            future_decay=float(
                os.getenv("REMTP_BLOCK_FUTURE_DECAY", "0.7")
            ),
            local_weight=float(
                os.getenv("REMTP_BLOCK_LOCAL_WEIGHT", "1.0")
            ),
            future_weight=float(
                os.getenv("REMTP_BLOCK_FUTURE_WEIGHT", "1.0")
            ),
            consistency_weight=float(
                os.getenv("REMTP_BLOCK_CONSISTENCY_WEIGHT", "1.0")
            ),
            rescue_weight=float(
                os.getenv("REMTP_BLOCK_RESCUE_WEIGHT", "2.0")
            ),
            max_normalized_surprisal=float(
                os.getenv(
                    "REMTP_BLOCK_MAX_NORMALIZED_SURPRISAL",
                    "12.0",
                )
            ),
            min_feature_consistency=float(
                os.getenv("REMTP_BLOCK_MIN_FEATURE_CONSISTENCY", "0.02")
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
    entropy: torch.Tensor
    normalized_surprisal: torch.Tensor
    local_support: torch.Tensor
    future_support: torch.Tensor
    row_feature_consistency: torch.Tensor
    state_feature_consistency: torch.Tensor
    standard_acceptance: torch.Tensor
    prefix_reach_probability: torch.Tensor
    prefix_value: torch.Tensor
    relaxation_need: torch.Tensor
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


def projected_logit_consistency(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity of p and q on a block-specific LM-head projection.

    The projection contains the token IDs in the four-token MTP block.  Qwen's
    native MTP and target share the LM head, so centered log-probability
    directions along the proposed trajectory provide a cheap, training-free
    proxy for hidden-state direction agreement.
    """
    if target_probs.shape != draft_probs.shape or target_probs.ndim != 2:
        raise ValueError("target_probs and draft_probs must share [rows, vocab]")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must have shape [tokens]")
    if draft_token_ids.shape[0] != target_probs.shape[0]:
        raise ValueError("draft token IDs must match probability rows")
    block_ids = draft_token_ids.to(
        device=target_probs.device,
        dtype=torch.int64,
    ).unsqueeze(0).expand(target_probs.shape[0], -1)
    target_log = torch.log(
        target_probs.gather(1, block_ids).clamp_min(1e-30)
    )
    draft_log = torch.log(
        draft_probs.gather(1, block_ids).clamp_min(1e-30)
    )
    target_direction = target_log - target_log.mean(dim=-1, keepdim=True)
    draft_direction = draft_log - draft_log.mean(dim=-1, keepdim=True)
    numerator = (target_direction * draft_direction).sum(dim=-1)
    denominator = (
        target_direction.square().sum(dim=-1).sqrt()
        * draft_direction.square().sum(dim=-1).sqrt()
    )
    cosine = numerator / denominator.clamp_min(1e-12)
    cosine = torch.where(
        denominator > 1e-12,
        cosine,
        torch.zeros_like(cosine),
    )
    return ((cosine.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def _future_support(
    local_support: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    """Weighted support from later positions of the verified draft path."""
    rows = local_support.shape[0]
    result = torch.zeros_like(local_support)
    for depth in range(rows - 1):
        offsets = torch.arange(
            rows - depth - 1,
            device=local_support.device,
            dtype=local_support.dtype,
        )
        weights = torch.pow(
            torch.full_like(offsets, decay),
            offsets,
        )
        result[depth] = (
            local_support[depth + 1 :] * weights
        ).sum() / weights.sum().clamp_min(1e-12)
    return result


def _prefix_marginal_values(
    standard_acceptance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Probability of reaching d and its marginal accepted-length value.

    For first-rejection verification,

        E[length] = 1 + sum_k prod_{i<=k} A_i.

    The derivative with respect to A_d is the probability of reaching d times
    the expected number of downstream draft/bonus positions it unlocks.
    """
    rows = standard_acceptance.shape[0]
    reach = torch.ones_like(standard_acceptance)
    for depth in range(1, rows):
        reach[depth] = reach[depth - 1] * standard_acceptance[depth - 1]

    suffix_value = torch.ones_like(standard_acceptance)
    for depth in range(rows - 2, -1, -1):
        suffix_value[depth] = (
            1.0
            + standard_acceptance[depth + 1]
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
    entropy = -(
        target * torch.log(target.clamp_min(1e-30))
    ).sum(dim=-1)
    surprisal = -torch.log(target_candidate.clamp_min(1e-30))
    normalized_surprisal = surprisal / entropy.clamp_min(1e-6)
    local_support = torch.exp(-0.5 * normalized_surprisal).clamp(0.0, 1.0)
    future_support = _future_support(local_support, config.future_decay)

    row_consistency = projected_logit_consistency(
        target,
        draft,
        ids,
    )
    if row_consistency.shape[0] > 1:
        state_consistency = torch.cat(
            (row_consistency[1:], row_consistency[-1:]),
            dim=0,
        )
    else:
        state_consistency = row_consistency

    standard_acceptance = torch.minimum(
        torch.ones_like(target_candidate),
        target_candidate / draft_candidate.clamp_min(1e-30),
    )
    reach_probability, marginal_value = _prefix_marginal_values(
        standard_acceptance
    )
    if config.use_prefix_value:
        prefix_value = marginal_value
    else:
        prefix_value = torch.ones_like(target_candidate)

    need = (
        (draft_candidate - target_candidate).clamp_min(0.0)
        / draft_candidate.clamp_min(1e-30)
    )
    safe = normalized_surprisal <= config.max_normalized_surprisal
    if config.use_feature_consistency:
        safe &= state_consistency >= config.min_feature_consistency
    if config.use_prefix_value:
        safe &= reach_probability >= config.min_prefix_reach_probability

    signal = config.local_weight * local_support
    if config.use_future_support:
        signal = signal + config.future_weight * future_support
    if config.use_feature_consistency:
        signal = signal + config.consistency_weight * state_consistency
    if config.use_future_support and config.use_feature_consistency:
        rescue = (
            (1.0 - local_support)
            * future_support
            * state_consistency
        )
        signal = signal + config.rescue_weight * rescue

    priority = torch.where(
        safe,
        need * prefix_value * signal.clamp_min(1e-8),
        torch.zeros_like(need),
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
        entropy=entropy,
        normalized_surprisal=normalized_surprisal,
        local_support=local_support,
        future_support=future_support,
        row_feature_consistency=row_consistency,
        state_feature_consistency=state_consistency,
        standard_acceptance=standard_acceptance,
        prefix_reach_probability=reach_probability,
        prefix_value=prefix_value,
        relaxation_need=need,
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
            f"norm_surprisal={result.normalized_surprisal.tolist()} "
            f"local={result.local_support.tolist()} "
            f"future={result.future_support.tolist()} "
            f"feature={result.state_feature_consistency.tolist()} "
            f"standard_A={result.standard_acceptance.tolist()} "
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
        f"future_support={int(config.use_future_support)} "
        f"feature_consistency={int(config.use_feature_consistency)} "
        f"future_decay={config.future_decay:g} "
        f"max_norm_surprisal={config.max_normalized_surprisal:g} "
        f"min_feature_consistency={config.min_feature_consistency:g} "
        f"min_prefix_reach={config.min_prefix_reach_probability:g} "
        "draft=probabilistic-MTP bonus=original-target",
        flush=True,
    )
