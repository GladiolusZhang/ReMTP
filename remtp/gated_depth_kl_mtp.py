"""Safety-gated, depth-decayed KL relaxation for probabilistic MTP."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch


_V1_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False


@dataclass(frozen=True)
class GatedDepthKLConfig:
    epsilon_0: float = 0.02
    depth_decay: float = 0.7
    max_rank: int = 8
    min_target_prob: float = 0.005
    max_log_gap: float = 1.5
    max_prob_ratio: float = 4.0
    bisection_steps: int = 20

    def validate(self) -> None:
        if self.epsilon_0 < 0:
            raise ValueError("epsilon_0 must be non-negative")
        if not 0 <= self.depth_decay <= 1:
            raise ValueError("depth_decay must be in [0, 1]")
        if self.max_rank < 1:
            raise ValueError("max_rank must be at least 1")
        if not 0 <= self.min_target_prob <= 1:
            raise ValueError("min_target_prob must be in [0, 1]")
        if self.max_log_gap <= 0:
            raise ValueError("max_log_gap must be positive")
        if self.max_prob_ratio < 1:
            raise ValueError("max_prob_ratio must be at least 1")
        if self.bisection_steps < 1:
            raise ValueError("bisection_steps must be positive")

    @classmethod
    def from_env(cls) -> GatedDepthKLConfig:
        config = cls(
            epsilon_0=float(
                os.getenv("REMTP_GATED_KL_EPSILON_0", "0.02")
            ),
            depth_decay=float(
                os.getenv("REMTP_GATED_KL_DEPTH_DECAY", "0.7")
            ),
            max_rank=int(os.getenv("REMTP_GATED_KL_MAX_RANK", "8")),
            min_target_prob=float(
                os.getenv("REMTP_GATED_KL_MIN_TARGET_PROB", "0.005")
            ),
            max_log_gap=float(
                os.getenv("REMTP_GATED_KL_MAX_LOG_GAP", "1.5")
            ),
            max_prob_ratio=float(
                os.getenv("REMTP_GATED_KL_MAX_PROB_RATIO", "4.0")
            ),
            bisection_steps=int(
                os.getenv("REMTP_GATED_KL_BISECTION_STEPS", "20")
            ),
        )
        config.validate()
        return config


@dataclass
class GatedDepthKLResult:
    probs: torch.Tensor
    safe: torch.Tensor
    ranks: torch.Tensor
    log_gaps: torch.Tensor
    epsilon: torch.Tensor
    original_candidate_probs: torch.Tensor
    boosted_candidate_probs: torch.Tensor
    realized_kl: torch.Tensor


@dataclass
class CandidateRelaxation:
    safe: torch.Tensor
    epsilon: torch.Tensor
    boosted_probs: torch.Tensor
    realized_kl: torch.Tensor


def bernoulli_kl(
    boosted: torch.Tensor,
    original: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(Bernoulli(boosted) || Bernoulli(original))."""
    dtype = boosted.dtype
    work_dtype = torch.float32 if boosted.is_cuda else torch.float64
    boosted_work = boosted.to(work_dtype).clamp(1e-12, 1.0 - 1e-12)
    original_work = original.to(work_dtype).clamp(
        1e-12, 1.0 - 1e-12
    )
    kl = boosted_work * torch.log(boosted_work / original_work)
    kl += (1.0 - boosted_work) * torch.log(
        (1.0 - boosted_work) / (1.0 - original_work)
    )
    return kl.to(dtype)


def _solve_boosted_probability(
    original: torch.Tensor,
    epsilon: torch.Tensor,
    max_prob_ratio: float,
    steps: int,
) -> torch.Tensor:
    ratio_cap = original * max_prob_ratio
    probability_cap = torch.full_like(original, 1.0 - 1e-6)
    upper = torch.minimum(ratio_cap, probability_cap)
    upper = torch.maximum(upper, original)

    low = original.clone()
    high = upper
    active = epsilon > 0
    for _ in range(steps):
        midpoint = (low + high) * 0.5
        within_budget = bernoulli_kl(midpoint, original) <= epsilon
        low = torch.where(active & within_budget, midpoint, low)
        high = torch.where(active & ~within_budget, midpoint, high)
    return torch.where(active, low, original)


def _candidate_relaxation(
    candidate_probs: torch.Tensor,
    ranks: torch.Tensor,
    log_gaps: torch.Tensor,
    draft_depths: torch.Tensor,
    config: GatedDepthKLConfig,
) -> CandidateRelaxation:
    safe = (
        (ranks <= config.max_rank)
        & (candidate_probs >= config.min_target_prob)
        & (log_gaps <= config.max_log_gap)
    )
    support = (1.0 - log_gaps / config.max_log_gap).clamp(0.0, 1.0)
    depth_budget = config.epsilon_0 * torch.pow(
        torch.full_like(draft_depths, config.depth_decay),
        draft_depths - 1.0,
    )
    epsilon = torch.where(
        safe,
        depth_budget * support,
        torch.zeros_like(depth_budget),
    )
    boosted = _solve_boosted_probability(
        candidate_probs,
        epsilon,
        config.max_prob_ratio,
        config.bisection_steps,
    )
    boosted = torch.where(safe, boosted, candidate_probs)
    return CandidateRelaxation(
        safe=safe,
        epsilon=epsilon,
        boosted_probs=boosted,
        realized_kl=bernoulli_kl(boosted, candidate_probs),
    )


def gated_depth_kl_distribution(
    target_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_depths: torch.Tensor,
    config: GatedDepthKLConfig,
) -> GatedDepthKLResult:
    """Boost only safe MTP candidates under a depth-decayed KL budget."""
    config.validate()
    if target_probs.ndim != 2:
        raise ValueError("target_probs must have shape [tokens, vocab]")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must have shape [tokens]")
    if draft_depths.ndim != 1:
        raise ValueError("draft_depths must have shape [tokens]")
    if (
        target_probs.shape[0] != draft_token_ids.shape[0]
        or draft_token_ids.shape[0] != draft_depths.shape[0]
    ):
        raise ValueError("target rows, candidate IDs, and depths must match")
    if draft_depths.device.type == "cpu" and bool((draft_depths < 1).any()):
        raise ValueError("draft depths must start from 1")

    probs = target_probs.to(torch.float32)
    token_ids = draft_token_ids.to(
        device=probs.device,
        dtype=torch.int64,
    )
    depths = draft_depths.to(
        device=probs.device,
        dtype=torch.float32,
    )
    rows = torch.arange(probs.shape[0], device=probs.device)

    candidate_probs = probs[rows, token_ids]
    top_probs = probs.amax(dim=-1)
    ranks = 1 + (probs > candidate_probs.unsqueeze(-1)).sum(dim=-1)
    log_gaps = torch.log(top_probs.clamp_min(1e-30)) - torch.log(
        candidate_probs.clamp_min(1e-30)
    )

    relaxation = _candidate_relaxation(
        candidate_probs,
        ranks,
        log_gaps,
        depths,
        config,
    )
    boosted = relaxation.boosted_probs

    scale = ((1.0 - boosted) / (1.0 - candidate_probs)).nan_to_num(
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    relaxed = probs * scale.unsqueeze(-1)
    relaxed.scatter_(1, token_ids.unsqueeze(1), boosted.unsqueeze(1))
    relaxed = relaxed.clamp_min(0.0)
    relaxed /= relaxed.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    realized_boosted = relaxed[rows, token_ids]
    realized_kl = bernoulli_kl(realized_boosted, candidate_probs)
    return GatedDepthKLResult(
        probs=relaxed,
        safe=relaxation.safe,
        ranks=ranks,
        log_gaps=log_gaps,
        epsilon=relaxation.epsilon,
        original_candidate_probs=candidate_probs,
        boosted_candidate_probs=realized_boosted,
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
    """Apply the one-token boost while preserving all other logit gaps."""
    logits = target_logits.to(torch.float32).clone()
    ids = draft_token_ids.to(device=logits.device, dtype=torch.int64)
    rows = torch.arange(logits.shape[0], device=logits.device)
    original = original_candidate_probs.clamp(1e-12, 1.0 - 1e-6)
    boosted = boosted_candidate_probs.clamp(1e-12, 1.0 - 1e-6)
    logits[rows, ids] += torch.logit(boosted) - torch.logit(original)
    return logits


def _gated_depth_kl_rejection_sample_v1(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Replace target p by the gated depth-KL distribution before verify."""
    global _DIAGNOSTIC_EMITTED

    original = getattr(
        _gated_depth_kl_rejection_sample_v1,
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
            "GatedDepthKL MTP currently requires --max-num-seqs 1"
        )

    config = getattr(
        _gated_depth_kl_rejection_sample_v1,
        "_remtp_config",
    )
    logits = target_logits.to(torch.float32)
    rows = torch.arange(logits.shape[0], device=logits.device)
    ids = draft_token_ids.to(device=logits.device, dtype=torch.int64)
    candidate_logits = logits[rows, ids]
    log_normalizers = torch.logsumexp(logits, dim=-1)
    candidate_probs = torch.exp(candidate_logits - log_normalizers)

    top_count = min(config.max_rank, logits.shape[-1])
    top_logits, top_ids = torch.topk(
        logits,
        k=top_count,
        dim=-1,
        sorted=True,
    )
    top_matches = top_ids == ids.unsqueeze(-1)
    in_top_k = top_matches.any(dim=-1)
    rank_in_top_k = top_matches.to(torch.int64).argmax(dim=-1) + 1
    ranks = torch.where(
        in_top_k,
        rank_in_top_k,
        torch.full_like(rank_in_top_k, top_count + 1),
    )
    log_gaps = top_logits[:, 0] - candidate_logits
    depths = torch.arange(
        1,
        draft_token_ids.shape[0] + 1,
        device=target_logits.device,
        dtype=torch.float32,
    )
    relaxation = _candidate_relaxation(
        candidate_probs,
        ranks,
        log_gaps,
        depths,
        config,
    )

    if (
        not _DIAGNOSTIC_EMITTED
        and os.getenv("REMTP_GATED_KL_DIAGNOSTICS", "1") == "1"
        and candidate_probs.sum().item() > 0
    ):
        q_at_draft = draft_probs[rows, ids]
        print(
            "[ReMTP][GatedDepthKL][diagnostic] "
            f"depth={depths.tolist()} "
            f"draft_ids={ids.tolist()} "
            f"rank={ranks.tolist()} "
            f"safe={relaxation.safe.tolist()} "
            f"log_gap={log_gaps.tolist()} "
            f"epsilon={relaxation.epsilon.tolist()} "
            f"q(D)={q_at_draft.tolist()} "
            f"p(D)={candidate_probs.tolist()} "
            f"relaxed_p(D)={relaxation.boosted_probs.tolist()} "
            f"KL={relaxation.realized_kl.tolist()}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True

    original_top_ids = logits.argmax(dim=-1)
    relaxed_logits = boost_candidate_logits(
        logits,
        ids,
        candidate_probs,
        relaxation.boosted_probs,
    )

    if sampling_metadata.all_greedy:
        relaxed_top_ids = relaxed_logits.argmax(dim=-1)
        desired_ids = torch.where(
            ids == relaxed_top_ids,
            ids,
            original_top_ids,
        )
        proxy_logits = torch.full_like(
            target_logits,
            float("-inf"),
        )
        proxy_logits.scatter_(1, desired_ids.unsqueeze(1), 0.0)
        verification_logits = proxy_logits
    else:
        # Raising one candidate logit preserves all non-candidate probability
        # ratios and gives the same relaxed distribution without an extra
        # full-vocabulary softmax/log round trip.
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


def install_gated_depth_kl_mtp() -> None:
    """Install the gated depth-KL verifier over probabilistic MTP."""
    config = GatedDepthKLConfig.from_env()
    module = importlib.import_module(_V1_REJECTION_MODULE)
    current = module.rejection_sample
    if not getattr(current, "_remtp_gated_depth_kl_mtp", False):
        wrapper = _gated_depth_kl_rejection_sample_v1
        wrapper._remtp_gated_depth_kl_mtp = True
        wrapper._remtp_original = current
        wrapper._remtp_config = config
        module.rejection_sample = wrapper

    print(
        "[ReMTP][GatedDepthKL] "
        f"epsilon_0={config.epsilon_0:g} "
        f"depth_decay={config.depth_decay:g} "
        f"max_rank={config.max_rank} "
        f"min_target_prob={config.min_target_prob:g} "
        f"max_log_gap={config.max_log_gap:g} "
        f"max_prob_ratio={config.max_prob_ratio:g} "
        f"bisection_steps={config.bisection_steps} "
        "draft=probabilistic-MTP bonus=original-target",
        flush=True,
    )
