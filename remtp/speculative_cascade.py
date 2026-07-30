"""Speculative-cascade target distributions for vLLM's MTP proposer.

The target distributions follow Narasimhan et al., "Faster Cascades via
Speculative Decoding" (ICLR 2025).  The adapter is intentionally pinned to the
probabilistic MTP rejection-sampling path in vLLM 0.18.0.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Literal

import torch


CascadeRule = Literal["chow", "diff", "opt", "token_v3"]
SUPPORTED_RULES = ("chow", "diff", "opt", "token_v3")
_UPSTREAM_MODULE = "vllm.v1.worker.gpu.spec_decode.rejection_sampler"
_V1_UPSTREAM_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False


def target_distribution(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs_unscaled: torch.Tensor,
    target_probs_unscaled: torch.Tensor,
    rule: CascadeRule,
    alpha: float,
) -> torch.Tensor:
    """Construct the paper's cascade target distribution ``pi``.

    Scaled distributions are used for sampling and TV distance.  Unscaled
    distributions are used for confidence-based deferral, as prescribed in
    Appendix C.1 of the paper.
    """
    if rule not in SUPPORTED_RULES:
        raise ValueError(
            f"unsupported cascade rule {rule!r}; choose from {SUPPORTED_RULES}"
        )
    if alpha < 0:
        raise ValueError("cascade alpha must be non-negative")
    if rule == "token_v3" and alpha > 1:
        raise ValueError("token_v3 alpha must be in [0, 1]")

    max_draft = draft_probs_unscaled.amax(dim=-1, keepdim=True)
    max_target = target_probs_unscaled.amax(dim=-1, keepdim=True)

    if rule == "chow":
        pick_draft = max_draft >= 1.0 - alpha
        probs = torch.where(pick_draft, draft_probs, target_probs)
    elif rule == "diff":
        pick_draft = max_draft >= max_target - alpha
        probs = torch.where(pick_draft, draft_probs, target_probs)
    elif rule == "opt":
        total_variation = torch.relu(draft_probs - target_probs).sum(
            dim=-1,
            keepdim=True,
        )
        pick_draft = (
            max_draft >= max_target - alpha * total_variation
        )
        probs = torch.where(pick_draft, draft_probs, target_probs)
    else:
        # Equation (15) and Appendix C.1:
        # keep q(v) on target-confident tokens and redistribute the rejected
        # q mass according to p.
        keep = target_probs_unscaled >= max_target * (1.0 - alpha)
        kept_draft = draft_probs * keep
        rejected_mass = 1.0 - kept_draft.sum(dim=-1, keepdim=True)
        probs = kept_draft + rejected_mass * target_probs

    # Guard against low-precision accumulation without altering the intended
    # distribution in normal operation.
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def recover_unscaled_probs(
    processed_logits: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Undo temperature scaling and return the original distribution.

    This is exact for this repository's benchmark, which uses temperature
    sampling without top-k, top-p, or penalties.
    """
    logits = processed_logits.to(torch.float32)
    temperature = temperature.to(
        device=logits.device,
        dtype=torch.float32,
    )
    while temperature.ndim < logits.ndim:
        temperature = temperature.unsqueeze(-1)
    multiplier = torch.where(
        temperature > 0,
        temperature,
        torch.ones_like(temperature),
    )
    return torch.softmax(logits * multiplier, dim=-1)


def _cascade_probabilistic_rejection_sample(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM-compatible probabilistic rejection sampler using cascade ``pi``."""
    global _DIAGNOSTIC_EMITTED

    upstream = importlib.import_module(_UPSTREAM_MODULE)
    gumbel = importlib.import_module(
        "vllm.v1.worker.gpu.sample.gumbel"
    )

    num_reqs = cu_num_logits.shape[0] - 1
    if num_reqs != 1:
        raise RuntimeError(
            "SpecCascade MTP currently requires --max-num-seqs 1"
        )

    num_drafts = target_logits.shape[0] - 1
    if num_drafts <= 0 or num_drafts > draft_logits.shape[1]:
        raise RuntimeError(
            "unexpected speculative batch layout: "
            f"target rows={target_logits.shape[0]}, "
            f"draft steps={draft_logits.shape[1]}"
        )

    rule = os.getenv("REMTP_CASCADE_RULE", "token_v3").lower()
    alpha = float(os.getenv("REMTP_CASCADE_ALPHA", "0.5"))

    target_probs_all = torch.softmax(
        target_logits.to(torch.float32),
        dim=-1,
    )
    target_probs = target_probs_all[:num_drafts]
    draft_probs = torch.softmax(
        draft_logits[0, :num_drafts].to(torch.float32),
        dim=-1,
    )

    request_temperature = temperature[
        idx_mapping[:1].to(torch.int64)
    ].reshape(1)
    target_unscaled = recover_unscaled_probs(
        target_logits[:num_drafts],
        request_temperature,
    )
    draft_unscaled = recover_unscaled_probs(
        draft_logits[0, :num_drafts],
        request_temperature,
    )
    cascade_probs = target_distribution(
        draft_probs,
        target_probs,
        draft_unscaled,
        target_unscaled,
        rule=rule,  # type: ignore[arg-type]
        alpha=alpha,
    )

    if (
        not _DIAGNOSTIC_EMITTED
        and os.getenv("REMTP_CASCADE_DIAGNOSTICS", "1") == "1"
    ):
        proposed_ids = draft_sampled[:num_drafts].to(
            device=draft_probs.device,
            dtype=torch.int64,
        )
        row_ids = torch.arange(num_drafts, device=draft_probs.device)
        q_at_draft = draft_probs[row_ids, proposed_ids]
        p_at_draft = target_probs[row_ids, proposed_ids]
        pi_at_draft = cascade_probs[row_ids, proposed_ids]
        keep = target_unscaled >= (
            target_unscaled.amax(dim=-1, keepdim=True) * (1.0 - alpha)
        )
        kept_drafts = keep[row_ids, proposed_ids]
        print(
            "[ReMTP][SpecCascade][diagnostic] "
            f"invoked=1 rule={rule} alpha={alpha:g} "
            f"max|pi-p|={(cascade_probs - target_probs).abs().max().item():.6g} "
            f"max|pi-q|={(cascade_probs - draft_probs).abs().max().item():.6g} "
            f"draft_ids={proposed_ids.tolist()} "
            f"draft_in_T={kept_drafts.tolist()} "
            f"q(D)={q_at_draft.tolist()} "
            f"p(D)={p_at_draft.tolist()} "
            f"pi(D)={pi_at_draft.tolist()}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True

    # The draft positions follow the paper's pi.  vLLM has no q distribution
    # for the extra bonus position, so its native verifier p bonus is retained.
    verification_probs = target_probs_all.clone()
    verification_probs[:num_drafts] = cascade_probs
    verification_probs = verification_probs.contiguous()
    draft_probs = draft_probs.unsqueeze(0).contiguous()

    sampled = draft_sampled.new_empty(
        num_reqs,
        num_speculative_steps + 1,
        dtype=torch.int64,
    )
    rejected_steps = sampled.new_empty(num_reqs)
    upstream._probabilistic_rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        rejected_steps,
        draft_sampled,
        verification_probs,
        verification_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        cu_num_logits,
        pos,
        idx_mapping,
        seed,
        num_warps=1,
    )

    vocab_size = target_logits.shape[-1]
    residual_logits = target_logits.new_empty(
        num_reqs,
        vocab_size,
    )
    residual_pos = pos.new_empty(num_reqs)
    block_size = 1024
    num_blocks = upstream.triton.cdiv(vocab_size, block_size)
    upstream._compute_residual_logits_kernel[(num_reqs, num_blocks)](
        residual_logits,
        residual_logits.stride(0),
        residual_pos,
        target_logits,
        target_logits.stride(0),
        verification_probs,
        verification_probs.stride(0),
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        rejected_steps,
        cu_num_logits,
        pos,
        vocab_size,
        BLOCK_SIZE=block_size,
    )

    resampled = gumbel.gumbel_sample(
        residual_logits,
        idx_mapping,
        temperature,
        seed,
        residual_pos,
        apply_temperature=False,
    )
    sampled.scatter_(
        1,
        rejected_steps.unsqueeze(1),
        resampled.unsqueeze(1),
    )
    return sampled, rejected_steps + 1


def _cascade_rejection_sample_v1(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Adapt the sampler used by Qwen3.5's hybrid-model runner.

    ``SpecCascadeMTPWorker`` installs the probabilistic-MTP adapter first, so
    every real stochastic request reaches this function with the complete
    proposal distribution. The ``None`` branch below only preserves vLLM's
    startup profiling and compatibility behavior.
    """
    global _DIAGNOSTIC_EMITTED

    upstream = importlib.import_module(_V1_UPSTREAM_MODULE)
    original = getattr(_cascade_rejection_sample_v1, "_remtp_original")

    # The cascade benchmark is stochastic. Preserve upstream behavior for
    # all-greedy requests and for empty speculative batches.
    if sampling_metadata.all_greedy or target_logits.shape[0] == 0:
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

    num_tokens, vocab_size = target_logits.shape
    target_probs = torch.softmax(
        target_logits.to(torch.float32),
        dim=-1,
    )

    if draft_probs is None:
        proposal_probs = torch.zeros_like(target_probs)
        proposal_probs.scatter_(
            1,
            draft_token_ids.to(torch.int64).unsqueeze(1),
            1.0,
        )
        proposal_unscaled = proposal_probs
        proposal_kind = "deterministic-mtp-argmax"
    else:
        proposal_probs = draft_probs.to(torch.float32)
        proposal_kind = "probabilistic"

    token_temperatures = upstream.expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=upstream.GREEDY_TEMPERATURE,
        replace_to=1,
    ).to(torch.float32)
    target_unscaled = recover_unscaled_probs(
        target_logits,
        token_temperatures,
    )
    if draft_probs is not None:
        proposal_unscaled = torch.softmax(
            torch.log(proposal_probs.clamp_min(1e-30))
            * token_temperatures.unsqueeze(-1),
            dim=-1,
        )

    rule = os.getenv("REMTP_CASCADE_RULE", "token_v3").lower()
    alpha = float(os.getenv("REMTP_CASCADE_ALPHA", "0.5"))
    cascade_probs = target_distribution(
        proposal_probs,
        target_probs,
        proposal_unscaled,
        target_unscaled,
        rule=rule,  # type: ignore[arg-type]
        alpha=alpha,
    )

    if (
        not _DIAGNOSTIC_EMITTED
        and os.getenv("REMTP_CASCADE_DIAGNOSTICS", "1") == "1"
    ):
        rows = torch.arange(num_tokens, device=target_logits.device)
        ids = draft_token_ids.to(torch.int64)
        q_at_draft = proposal_probs[rows, ids]
        p_at_draft = target_probs[rows, ids]
        pi_at_draft = cascade_probs[rows, ids]
        keep = target_unscaled >= (
            target_unscaled.amax(dim=-1, keepdim=True) * (1.0 - alpha)
        )
        kept_drafts = keep[rows, ids]
        # The engine profiles with a dummy token whose target probability is
        # exactly zero. Wait for the first real decode round.
        if p_at_draft.sum().item() > 0:
            print(
                "[ReMTP][SpecCascade][diagnostic] "
                f"invoked=1 proposal={proposal_kind} "
                f"rule={rule} alpha={alpha:g} "
                f"max|pi-p|="
                f"{(cascade_probs - target_probs).abs().max().item():.6g} "
                f"max|pi-q|="
                f"{(cascade_probs - proposal_probs).abs().max().item():.6g} "
                f"draft_ids={ids.tolist()} "
                f"draft_in_T={kept_drafts.tolist()} "
                f"q(D)={q_at_draft.tolist()} "
                f"p(D)={p_at_draft.tolist()} "
                f"pi(D)={pi_at_draft.tolist()}",
                flush=True,
            )
            _DIAGNOSTIC_EMITTED = True

    # The upstream sampler applies softmax internally, so log(pi) supplies the
    # desired target distribution. For real benchmark requests draft_probs is
    # the complete probabilistic-MTP q; the None case is startup/compatibility.
    cascade_logits = torch.log(cascade_probs.clamp_min(1e-30))
    return original(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        cascade_logits,
        bonus_token_ids,
        sampling_metadata,
    )


def install_speculative_cascade() -> None:
    """Install adapters for both vLLM 0.18 speculative sampler paths."""
    module = importlib.import_module(_UPSTREAM_MODULE)
    current = module.probabilistic_rejection_sample
    if not getattr(current, "_remtp_speculative_cascade", False):
        _cascade_probabilistic_rejection_sample._remtp_speculative_cascade = (
            True
        )
        _cascade_probabilistic_rejection_sample._remtp_original = current
        module.probabilistic_rejection_sample = (
            _cascade_probabilistic_rejection_sample
        )

    v1_module = importlib.import_module(_V1_UPSTREAM_MODULE)
    v1_current = v1_module.rejection_sample
    if not getattr(v1_current, "_remtp_speculative_cascade", False):
        _cascade_rejection_sample_v1._remtp_speculative_cascade = True
        _cascade_rejection_sample_v1._remtp_original = v1_current
        v1_module.rejection_sample = _cascade_rejection_sample_v1

    rule = os.getenv("REMTP_CASCADE_RULE", "token_v3").lower()
    alpha = float(os.getenv("REMTP_CASCADE_ALPHA", "0.5"))
    if rule not in SUPPORTED_RULES:
        raise ValueError(
            f"unsupported REMTP_CASCADE_RULE={rule!r}; "
            f"choose from {SUPPORTED_RULES}"
        )
    if alpha < 0 or (rule == "token_v3" and alpha > 1):
        raise ValueError(
            f"invalid REMTP_CASCADE_ALPHA={alpha} for rule {rule}"
        )
    print(
        "[ReMTP][SpecCascade] "
        f"rule={rule} alpha={alpha:g} "
        "draft=MTP verifier=target bonus=target",
        flush=True,
    )
