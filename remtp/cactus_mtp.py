"""Cactus constrained-acceptance sampling for probabilistic MTP.

This module adapts Hao and Mou, "Cactus: Accelerating Auto-Regressive
Decoding with Constrained Acceptance Speculative Sampling" (ICLR 2026), to
the vLLM 0.18 Qwen3.5 hybrid-model rejection sampler used by this repository.

Repository notation:

* ``draft_probs`` is the complete MTP proposal distribution q;
* ``target_probs`` is the verifier distribution p;
* ``cactus_probs`` is the sampled-token-specific Cactus distribution h.

The paper uses p for the draft and q for the verifier.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import torch


_V1_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False
_POST_VERIFICATION_HOOK: Any | None = None


def cactus_target_distribution(
    target_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Construct the paper's approximate KL-constrained distribution h.

    For a drafted token D with verifier probability p(D), Cactus sets

        gamma = min(p(D) + sqrt(2 delta p(D) (1 - p(D))), 1)

    and scales every non-D probability proportionally so the result remains a
    normalized distribution.
    """
    if delta < 0:
        raise ValueError("Cactus delta must be non-negative")
    if target_probs.ndim != 2:
        raise ValueError("target_probs must have shape [tokens, vocab]")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must have shape [tokens]")
    if target_probs.shape[0] != draft_token_ids.shape[0]:
        raise ValueError(
            "target probability rows must match the number of draft tokens"
        )

    probs = target_probs.to(torch.float32)
    token_ids = draft_token_ids.to(
        device=probs.device,
        dtype=torch.int64,
    )
    rows = torch.arange(probs.shape[0], device=probs.device)
    selected = probs[rows, token_ids]

    bonus = torch.sqrt(
        (2.0 * delta * selected * (1.0 - selected)).clamp_min(0.0)
    )
    gamma = (selected + bonus).clamp(max=1.0)

    scale = ((1.0 - gamma) / (1.0 - selected)).nan_to_num(
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    cactus_probs = probs * scale.unsqueeze(-1)
    cactus_probs.scatter_(1, token_ids.unsqueeze(1), gamma.unsqueeze(1))
    cactus_probs = cactus_probs.clamp_min(0.0)
    return cactus_probs / cactus_probs.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-30)


def _cactus_rejection_sample_v1(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Replace verifier p by Cactus h before standard rejection sampling."""
    global _DIAGNOSTIC_EMITTED

    original = getattr(_cactus_rejection_sample_v1, "_remtp_original")

    # Greedy decoding has no probabilistic acceptance rule. A missing q occurs
    # during vLLM startup profiling; every real stochastic request is guarded
    # by remtp.probabilistic_mtp and arrives with the complete MTP q.
    if (
        sampling_metadata.all_greedy
        or target_logits.shape[0] == 0
        or draft_probs is None
    ):
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

    delta = float(os.getenv("REMTP_CACTUS_DELTA", "1.0"))
    if delta < 0:
        raise ValueError("REMTP_CACTUS_DELTA must be non-negative")

    target_probs = torch.softmax(
        target_logits.to(torch.float32),
        dim=-1,
    )
    cactus_probs = cactus_target_distribution(
        target_probs,
        draft_token_ids,
        delta,
    )

    if (
        not _DIAGNOSTIC_EMITTED
        and os.getenv("REMTP_CACTUS_DIAGNOSTICS", "1") == "1"
    ):
        rows = torch.arange(
            draft_token_ids.shape[0],
            device=target_probs.device,
        )
        ids = draft_token_ids.to(torch.int64)
        mtp_at_draft = draft_probs[rows, ids]
        target_at_draft = target_probs[rows, ids]
        cactus_at_draft = cactus_probs[rows, ids]
        kl = (
            cactus_probs
            * (
                torch.log(cactus_probs.clamp_min(1e-30))
                - torch.log(target_probs.clamp_min(1e-30))
            )
        ).sum(dim=-1)
        # Ignore the zero-probability dummy profile round.
        if target_at_draft.sum().item() > 0:
            print(
                "[ReMTP][Cactus][diagnostic] "
                f"invoked=1 delta={delta:g} full_q=1 "
                f"draft_ids={ids.tolist()} "
                f"q_mtp(D)={mtp_at_draft.tolist()} "
                f"p_target(D)={target_at_draft.tolist()} "
                f"h_cactus(D)={cactus_at_draft.tolist()} "
                f"KL(h||p)={kl.tolist()}",
                flush=True,
            )
            _DIAGNOSTIC_EMITTED = True

    # vLLM applies softmax internally. Passing log(h) makes the unmodified
    # rejection sampler use min(1, h(D)/q_mtp(D)) and residual (h-q_mtp)+.
    cactus_logits = torch.log(cactus_probs.clamp_min(1e-30))
    output_token_ids = original(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        cactus_logits,
        bonus_token_ids,
        sampling_metadata,
    )
    if _POST_VERIFICATION_HOOK is not None:
        _POST_VERIFICATION_HOOK(
            target_probs=target_probs,
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            cactus_probs=cactus_probs,
            output_token_ids=output_token_ids,
            sampling_metadata=sampling_metadata,
        )
    return output_token_ids


def set_post_verification_hook(hook: Any | None) -> None:
    """Observe a completed Cactus round without changing its distribution."""
    global _POST_VERIFICATION_HOOK

    _POST_VERIFICATION_HOOK = hook


def install_cactus_mtp() -> None:
    """Install the Cactus target-distribution adapter for vLLM 0.18."""
    delta = float(os.getenv("REMTP_CACTUS_DELTA", "1.0"))
    if delta < 0:
        raise ValueError("REMTP_CACTUS_DELTA must be non-negative")

    module = importlib.import_module(_V1_REJECTION_MODULE)
    current = module.rejection_sample
    if not getattr(current, "_remtp_cactus_mtp", False):
        _cactus_rejection_sample_v1._remtp_cactus_mtp = True
        _cactus_rejection_sample_v1._remtp_original = current
        module.rejection_sample = _cactus_rejection_sample_v1

    print(
        "[ReMTP][Cactus] "
        f"delta={delta:g} draft=probabilistic-MTP "
        "verifier=Cactus-h bonus=target",
        flush=True,
    )
