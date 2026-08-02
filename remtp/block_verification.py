"""Joint block verification for probabilistic native MTP.

This is the stochastic block-verification algorithm of Sun et al.
(``arXiv:2403.10444``), adapted to vLLM 0.18's legacy V1 rejection-sampler
interface.  It is intentionally restricted to the single-request benchmark
configuration used by this repository.

The module only changes how an already-computed draft block is verified.  It
does not change MTP drafting or the target distribution passed by an outer
adapter.  Consequently it can be installed below Cactus or the repository's
target-anchored risk swap: the outer method constructs ``h`` and this module
jointly verifies the full block against that same ``h``.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch


_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False


@dataclass(frozen=True)
class BlockVerificationState:
    """Quantities needed by exact stochastic block verification."""

    prefix_joint_probability: torch.Tensor
    subblock_acceptance_probability: torch.Tensor
    correction_scale: torch.Tensor
    residual_mass: torch.Tensor


def block_verification_state(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    *,
    assume_normalized: bool = False,
) -> BlockVerificationState:
    """Compute exact joint-prefix acceptance and residual scalars.

    ``target_probs[i]`` and ``draft_probs[i]`` are the conditional
    distributions before draft token ``i``.  If ``a_i`` is the probability
    that prefix ``y_1..y_i`` survives, the paper's recurrence is

    ``a_i = min(a_(i-1) * p_i(y_i) / q_i(y_i), 1)``.

    For every non-final prefix, the independent subblock test uses the mass
    of ``max(a_i * p_(i+1) - q_(i+1), 0)``.  The longest passing prefix is
    committed.  ``correction_scale[i]`` is ``a_i`` for the correction after
    a committed prefix of length ``i``; its first value is ``a_0 = 1``.
    """
    if target_probs.ndim != 2 or target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must share [rows,vocab]")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must have shape [rows]")
    rows = target_probs.shape[0]
    if rows < 1 or draft_token_ids.shape[0] != rows:
        raise ValueError("one draft token is required for every probability row")

    target = target_probs.to(torch.float32)
    draft = draft_probs.to(device=target.device, dtype=torch.float32)
    if not assume_normalized:
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        draft = draft / draft.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    ids = draft_token_ids.to(device=target.device, dtype=torch.int64)
    row_ids = torch.arange(rows, device=target.device)
    p_y = target[row_ids, ids]
    q_y = draft[row_ids, ids]

    prefix_values: list[torch.Tensor] = []
    running = torch.ones((), device=target.device, dtype=torch.float32)
    for depth in range(rows):
        ratio = p_y[depth] / q_y[depth].clamp_min(1e-30)
        running = torch.minimum(running * ratio, torch.ones_like(running))
        prefix_values.append(running)
    prefix = torch.stack(prefix_values)

    # A correction after tau accepted drafts uses a_tau * p_tau - q_tau.
    correction_scale = torch.cat((torch.ones_like(prefix[:1]), prefix[:-1]))
    residual = (
        correction_scale.unsqueeze(-1) * target - draft
    ).clamp_min(0.0)
    residual_mass = residual.sum(dim=-1)

    acceptance = torch.empty_like(prefix)
    if rows > 1:
        # Prefix i (1-based) is corrected from probability row i, hence the
        # one-row shift relative to prefix[i-1].
        future_mass = residual_mass[1:]
        denominator = future_mass + 1.0 - prefix[:-1]
        acceptance[:-1] = torch.where(
            denominator > 0.0,
            future_mass / denominator,
            torch.ones_like(denominator),
        )
    acceptance[-1] = prefix[-1]
    return BlockVerificationState(
        prefix_joint_probability=prefix,
        subblock_acceptance_probability=acceptance.clamp(0.0, 1.0),
        correction_scale=correction_scale,
        residual_mass=residual_mass,
    )


def _block_verification_tensors(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tuple-returning fast path suitable for ``torch.compile``."""
    state = block_verification_state(
        target_probs,
        draft_probs,
        draft_token_ids,
        assume_normalized=True,
    )
    return (
        state.prefix_joint_probability,
        state.subblock_acceptance_probability,
        state.correction_scale,
        state.residual_mass,
    )


def longest_accepted_prefix(
    acceptance_probabilities: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Return the longest independently accepted subblock length."""
    if acceptance_probabilities.shape != uniforms.shape:
        raise ValueError("acceptance probabilities and uniforms must align")
    lengths = torch.arange(
        1,
        acceptance_probabilities.shape[0] + 1,
        device=acceptance_probabilities.device,
        dtype=torch.int64,
    )
    return torch.where(
        uniforms <= acceptance_probabilities,
        lengths,
        torch.zeros_like(lengths),
    ).amax()


def _block_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Drop-in replacement for vLLM's token-wise rejection sampler."""
    global _DIAGNOSTIC_EMITTED

    original = getattr(_block_rejection_sample, "_remtp_original")
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
        raise RuntimeError("block verification requires --max-num-seqs 1")
    num_drafts = num_draft_tokens[0]
    if num_drafts < 1 or draft_token_ids.shape[0] != num_drafts:
        raise RuntimeError("unexpected flattened draft-token layout")

    module = importlib.import_module(_REJECTION_MODULE)
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    fast_state = getattr(_block_rejection_sample, "_remtp_fast_state")
    (
        prefix_joint_probability,
        subblock_acceptance_probability,
        correction_scale,
        residual_mass,
    ) = fast_state(
        target_probs,
        draft_probs,
        draft_token_ids,
    )
    state = BlockVerificationState(
        prefix_joint_probability=prefix_joint_probability,
        subblock_acceptance_probability=subblock_acceptance_probability,
        correction_scale=correction_scale,
        residual_mass=residual_mass,
    )
    uniforms = module.generate_uniform_probs(
        num_drafts,
        num_draft_tokens,
        sampling_metadata.generators,
        target_logits.device,
    )
    accepted_length = longest_accepted_prefix(
        state.subblock_acceptance_probability,
        uniforms,
    )

    # vLLM's exponential-race kernel does not require normalized inputs.
    # Scaling each target row by a_tau therefore samples exactly from
    # max(a_tau * p_tau - q_tau, 0) after normalization.
    scaled_target = (
        state.correction_scale.unsqueeze(-1) * target_probs
    ).contiguous()
    recovered_token_ids = module.sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        scaled_target,
        sampling_metadata,
        target_logits.device,
    )

    output = torch.full(
        (1, max_spec_len + 1),
        module.PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=target_logits.device,
    )
    output[0, :num_drafts] = draft_token_ids.to(torch.int32)
    positions = torch.arange(
        max_spec_len + 1,
        device=target_logits.device,
        dtype=torch.int64,
    )
    output[0] = torch.where(
        positions < accepted_length,
        output[0],
        torch.full_like(output[0], module.PLACEHOLDER_TOKEN_ID),
    )
    safe_recovery_index = accepted_length.clamp(max=num_drafts - 1)
    correction = torch.where(
        accepted_length == num_drafts,
        bonus_token_ids.reshape(-1)[0].to(torch.int32),
        recovered_token_ids[safe_recovery_index].to(torch.int32),
    )
    output[0].scatter_(
        0,
        accepted_length.reshape(1),
        correction.reshape(1),
    )

    if (
        not _DIAGNOSTIC_EMITTED
        and os.getenv("REMTP_BLOCK_VERIFY_DIAGNOSTICS", "0") == "1"
    ):
        print(
            "[ReMTP][BlockVerify][diagnostic] "
            f"prefix_joint={state.prefix_joint_probability.tolist()} "
            f"subblock_A={state.subblock_acceptance_probability.tolist()} "
            f"residual_mass={state.residual_mass.tolist()} "
            f"u={uniforms.tolist()} accepted={accepted_length.item()}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True
    return output


def install_block_verification() -> None:
    """Install joint stochastic verification below any target adapter."""
    compile_fast_path = os.getenv("REMTP_BLOCK_VERIFY_COMPILE", "1") == "1"
    fast_state = _block_verification_tensors
    if compile_fast_path:
        fast_state = torch.compile(fast_state, fullgraph=True, dynamic=False)
    module = importlib.import_module(_REJECTION_MODULE)
    current = module.rejection_sample
    if not getattr(current, "_remtp_block_verification", False):
        wrapper = _block_rejection_sample
        wrapper._remtp_block_verification = True
        wrapper._remtp_original = current
        wrapper._remtp_fast_state = fast_state
        module.rejection_sample = wrapper
    print(
        "[ReMTP][BlockVerify] exact joint stochastic block verification enabled "
        "(single request; greedy delegates to vLLM) "
        f"compiled={int(compile_fast_path)}",
        flush=True,
    )
