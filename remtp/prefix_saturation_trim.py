"""Block-aware removal of Cactus probability that cannot aid a prefix.

Joint Block Verification carries a running prefix-survival probability.  A
Cactus candidate probability above ``q(y)`` can therefore be useful when it
repairs an earlier deficit.  Only the mass above the exact repair amount is
redundant.  This adapter removes that mass while preserving every running
Cactus candidate-prefix probability, then delegates to the exact block
verifier.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import torch

from remtp.block_verification import prefix_joint_probability


_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False


def cactus_candidate_probabilities(
    target_candidate_probs: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Return Cactus h(y) without constructing the full vocabulary rows."""
    if delta < 0.0:
        raise ValueError("Cactus delta must be non-negative")
    p_y = target_candidate_probs.to(torch.float32)
    increment = torch.sqrt(
        (2.0 * delta * p_y * (1.0 - p_y)).clamp_min(0.0)
    )
    return (p_y + increment).clamp(max=1.0)


def trim_prefix_saturation(
    target_candidate_probs: torch.Tensor,
    draft_candidate_probs: torch.Tensor,
    cactus_candidate_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trim only Cactus mass redundant to candidate-prefix survival.

    Returns trimmed candidate probabilities, the Cactus prefix survival, and
    the trimmed prefix survival.  The latter two are equal up to floating
    point error by construction.
    """
    vectors = (
        target_candidate_probs,
        draft_candidate_probs,
        cactus_candidate_probs,
    )
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("candidate probabilities must have shape [rows]")
    if len({vector.shape[0] for vector in vectors}) != 1:
        raise ValueError("candidate probability vectors must align")
    p_y = target_candidate_probs.to(torch.float32)
    q_y = draft_candidate_probs.to(device=p_y.device, dtype=torch.float32)
    cactus_h = cactus_candidate_probs.to(
        device=p_y.device,
        dtype=torch.float32,
    )
    cactus_prefix = prefix_joint_probability(cactus_h, q_y)

    trimmed_values: list[torch.Tensor] = []
    running = torch.ones((), device=p_y.device, dtype=torch.float32)
    for depth in range(p_y.shape[0]):
        required = (
            cactus_prefix[depth]
            * q_y[depth]
            / running.clamp_min(1e-30)
        )
        # Never move farther from the original target distribution.  The
        # min also protects against tiny recurrence round-off.
        candidate = torch.maximum(
            p_y[depth],
            torch.minimum(cactus_h[depth], required),
        )
        trimmed_values.append(candidate)
        running = torch.minimum(
            running * candidate / q_y[depth].clamp_min(1e-30),
            torch.ones_like(running),
        )
    trimmed = torch.stack(trimmed_values)
    trimmed_prefix = prefix_joint_probability(trimmed, q_y)
    return trimmed, cactus_prefix, trimmed_prefix


def candidate_shift_distribution(
    target_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    candidate_probs: torch.Tensor,
) -> torch.Tensor:
    """Set h(y), proportionally retaining target ratios for other tokens."""
    ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    rows = torch.arange(ids.shape[0], device=target_probs.device)
    original = target_probs[rows, ids]
    candidate = torch.maximum(candidate_probs, original).clamp(max=1.0)
    scale = ((1.0 - candidate) / (1.0 - original)).nan_to_num(
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    shifted = target_probs.to(torch.float32) * scale.unsqueeze(-1)
    shifted.scatter_(1, ids.unsqueeze(1), candidate.unsqueeze(1))
    return shifted / shifted.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def _prefix_trim_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    global _DIAGNOSTIC_EMITTED

    original = getattr(_prefix_trim_rejection_sample, "_remtp_original")
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
    if len(num_draft_tokens) != 1:
        raise RuntimeError("prefix trim requires --max-num-seqs 1")

    delta = float(os.getenv("REMTP_CACTUS_DELTA", "1.0"))
    target_probs = target_logits.to(torch.float32).softmax(dim=-1)
    ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    rows = torch.arange(ids.shape[0], device=target_probs.device)
    p_y = target_probs[rows, ids]
    q_y = draft_probs[rows, ids]
    cactus_h = cactus_candidate_probabilities(p_y, delta)
    fast_trim = getattr(_prefix_trim_rejection_sample, "_remtp_fast_trim")
    trimmed_h, cactus_prefix, trimmed_prefix = fast_trim(
        p_y,
        q_y,
        cactus_h,
    )
    verification_probs = candidate_shift_distribution(
        target_probs,
        ids,
        trimmed_h,
    )

    if (
        not _DIAGNOSTIC_EMITTED
        and os.getenv("REMTP_PREFIX_TRIM_DIAGNOSTICS", "0") == "1"
    ):
        reclaimed = (cactus_h - trimmed_h).clamp_min(0.0)
        print(
            "[ReMTP][PrefixTrim][diagnostic] "
            f"reclaimed={reclaimed.tolist()} "
            f"total={reclaimed.sum().item():.6f} "
            f"prefix_error="
            f"{(cactus_prefix-trimmed_prefix).abs().amax().item():.3e}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True

    verification_logits = torch.log(verification_probs.clamp_min(1e-30))
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


def install_prefix_saturation_trim() -> None:
    """Install prefix-neutral Cactus saturation trimming."""
    delta = float(os.getenv("REMTP_CACTUS_DELTA", "1.0"))
    if delta < 0.0:
        raise ValueError("REMTP_CACTUS_DELTA must be non-negative")
    compiled = os.getenv("REMTP_PREFIX_TRIM_COMPILE", "1") == "1"
    module = importlib.import_module(_REJECTION_MODULE)
    current = module.rejection_sample
    if not getattr(current, "_remtp_prefix_saturation_trim", False):
        fast_trim = trim_prefix_saturation
        if compiled:
            fast_trim = torch.compile(
                fast_trim,
                fullgraph=True,
                dynamic=False,
            )
        wrapper = _prefix_trim_rejection_sample
        wrapper._remtp_prefix_saturation_trim = True
        wrapper._remtp_original = current
        wrapper._remtp_fast_trim = fast_trim
        module.rejection_sample = wrapper
    print(
        "[ReMTP][PrefixTrim] "
        f"delta={delta:g} preserve=Cactus-prefix remove=redundant-mass "
        f"compiled={int(compiled)}",
        flush=True,
    )
