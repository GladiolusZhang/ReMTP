"""Expose Qwen3.5 MTP proposal distributions to vLLM verification.

vLLM 0.18's hybrid-model runner samples MTP drafts with argmax and passes
``draft_probs=None`` to the rejection sampler.  The helpers in this module
replace that narrow path with temperature sampling from the full MTP
distribution and carry the matching q tensor into rejection sampling.

The adapter intentionally targets this repository's single-request benchmark
configuration (``--max-num-seqs 1``).
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import torch


_EAGLE_MODULE = "vllm.v1.spec_decode.eagle"
_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_GPU_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
_LAST_DRAFT_PROBS: torch.Tensor | None = None
_LAST_DRAFT_TOKEN_IDS: torch.Tensor | None = None
_LAST_DRAFT_HIDDEN_STATES: torch.Tensor | None = None
_LAST_TARGET_HIDDEN_STATES: torch.Tensor | None = None
_LAST_GENERATOR_ROWS: tuple[int, ...] = ()
_PROPOSAL_COUNT = 0
_DIAGNOSTIC_EMITTED = False
_CAPTURE_ALIGNED_HIDDEN_STATES = False
_BONUS_LOGITS_HOOK: Any | None = None


def sample_mtp_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temperature-sample MTP logits and return both token IDs and q.

    Exponential-race sampling matches the categorical sampling primitive used
    by vLLM. Greedy rows (temperature zero) still return argmax tokens.
    """
    if logits.ndim != 2:
        raise ValueError("MTP logits must have shape [batch, vocab]")
    if temperatures.ndim != 1 or temperatures.shape[0] != logits.shape[0]:
        raise ValueError(
            "temperatures must have one value per MTP logits row"
        )

    raw_logits = logits.to(torch.float32)
    temperatures = temperatures.to(
        device=raw_logits.device,
        dtype=torch.float32,
    )
    greedy = temperatures < 1e-5
    safe_temperatures = torch.where(
        greedy,
        torch.ones_like(temperatures),
        temperatures,
    )
    probs = torch.softmax(
        raw_logits / safe_temperatures.unsqueeze(-1),
        dim=-1,
    )

    exponential_noise = torch.empty_like(probs)
    exponential_noise.exponential_()
    for row, generator in generators.items():
        if 0 <= row < probs.shape[0]:
            exponential_noise[row].exponential_(generator=generator)

    sampled = (probs / exponential_noise).argmax(dim=-1)
    if greedy.any():
        sampled = torch.where(
            greedy,
            raw_logits.argmax(dim=-1),
            sampled,
        )
    return sampled, probs


def _sample_from_full_mtp_distribution(
    self: Any,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    global _LAST_GENERATOR_ROWS

    original = getattr(
        _sample_from_full_mtp_distribution,
        "_remtp_original",
    )
    sampling_metadata = getattr(
        self,
        "_remtp_sampling_metadata",
        None,
    )
    if sampling_metadata is None:
        return original(self, hidden_states)
    if self.use_local_argmax_reduction:
        raise RuntimeError(
            "probabilistic MTP requires full logits; "
            "disable use_local_argmax_reduction"
        )

    batch_size = hidden_states.shape[0]
    if batch_size != 1:
        raise RuntimeError(
            "probabilistic MTP currently requires --max-num-seqs 1"
        )
    temperatures = sampling_metadata.temperature
    if temperatures is None:
        raise RuntimeError("MTP sampling metadata has no temperature tensor")
    temperatures = temperatures[:batch_size]

    logits = self.model.compute_logits(hidden_states)
    sampled, probs = sample_mtp_logits(
        logits,
        temperatures,
        sampling_metadata.generators,
    )
    _LAST_GENERATOR_ROWS = tuple(sorted(sampling_metadata.generators))
    self._remtp_current_draft_probs.append(probs.contiguous())
    if _CAPTURE_ALIGNED_HIDDEN_STATES:
        self._remtp_current_draft_hidden_states.append(
            hidden_states.contiguous()
        )
    return sampled


def _propose_with_full_distribution(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    global _LAST_DRAFT_HIDDEN_STATES, _LAST_DRAFT_PROBS
    global _LAST_DRAFT_TOKEN_IDS, _PROPOSAL_COUNT

    original = getattr(
        _propose_with_full_distribution,
        "_remtp_original",
    )
    sampling_metadata = kwargs.get("sampling_metadata")
    if sampling_metadata is None:
        raise RuntimeError("EagleProposer.propose has no sampling_metadata")

    self._remtp_sampling_metadata = sampling_metadata
    self._remtp_current_draft_probs = []
    if _CAPTURE_ALIGNED_HIDDEN_STATES:
        self._remtp_current_draft_hidden_states = []
    try:
        draft_token_ids = original(self, *args, **kwargs)
    finally:
        del self._remtp_sampling_metadata

    collected = self._remtp_current_draft_probs
    del self._remtp_current_draft_probs
    if not collected:
        raise RuntimeError("MTP proposer did not expose any probability rows")

    # The repository fixes max_num_seqs=1, so sequential MTP calls are already
    # ordered D0, D1, ... when concatenated.
    draft_probs = torch.cat(collected, dim=0)
    flattened_ids = draft_token_ids.reshape(-1)
    if draft_probs.shape[0] != flattened_ids.shape[0]:
        raise RuntimeError(
            "MTP probability/token layout mismatch: "
            f"q rows={draft_probs.shape[0]}, "
            f"tokens={flattened_ids.shape[0]}"
        )

    _LAST_DRAFT_PROBS = draft_probs.contiguous()
    _LAST_DRAFT_TOKEN_IDS = flattened_ids
    if _CAPTURE_ALIGNED_HIDDEN_STATES:
        collected_hidden_states = self._remtp_current_draft_hidden_states
        del self._remtp_current_draft_hidden_states
        if len(collected_hidden_states) != len(collected):
            raise RuntimeError("MTP probability/hidden collection mismatch")
        _LAST_DRAFT_HIDDEN_STATES = torch.cat(
            collected_hidden_states,
            dim=0,
        ).contiguous()
    _PROPOSAL_COUNT += 1
    return draft_token_ids


def _sample_tokens_with_target_hidden(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Cache target hidden rows aligned with speculative verification logits."""
    global _LAST_TARGET_HIDDEN_STATES

    state = self.execute_model_state
    _LAST_TARGET_HIDDEN_STATES = None
    if state is not None and state.spec_decode_metadata is not None:
        indices = state.spec_decode_metadata.target_logits_indices
        _LAST_TARGET_HIDDEN_STATES = (
            state.sample_hidden_states[indices].contiguous()
        )

    original = getattr(
        _sample_tokens_with_target_hidden,
        "_remtp_original",
    )
    return original(self, *args, **kwargs)


def get_last_aligned_hidden_states(
    expected_rows: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return MTP/target hidden rows when their block layouts are aligned."""
    draft = _LAST_DRAFT_HIDDEN_STATES
    target = _LAST_TARGET_HIDDEN_STATES
    if draft is None or target is None:
        return None, None
    if draft.shape[0] < expected_rows or target.shape[0] < expected_rows:
        return None, None
    draft = draft[:expected_rows]
    target = target[:expected_rows]
    if draft.ndim != 2 or target.ndim != 2:
        return None, None
    if draft.shape != target.shape:
        return None, None
    return draft, target


def install_aligned_hidden_capture() -> None:
    """Enable hidden capture only for methods that actually consume it."""
    global _CAPTURE_ALIGNED_HIDDEN_STATES

    _CAPTURE_ALIGNED_HIDDEN_STATES = True
    runner_module = importlib.import_module(_GPU_RUNNER_MODULE)
    runner_cls = runner_module.GPUModelRunner
    current_sample_tokens = runner_cls.sample_tokens
    if not getattr(
        current_sample_tokens,
        "_remtp_target_hidden_cache",
        False,
    ):
        _sample_tokens_with_target_hidden._remtp_target_hidden_cache = True
        _sample_tokens_with_target_hidden._remtp_original = (
            current_sample_tokens
        )
        runner_cls.sample_tokens = _sample_tokens_with_target_hidden


def _forward_with_mtp_probs(
    self: Any,
    metadata: Any,
    draft_probs: torch.Tensor | None,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> Any:
    global _DIAGNOSTIC_EMITTED

    original = getattr(_forward_with_mtp_probs, "_remtp_original")
    if _BONUS_LOGITS_HOOK is not None:
        _BONUS_LOGITS_HOOK(
            logits[metadata.bonus_logits_indices].to(torch.float32),
            sampling_metadata,
        )
    if draft_probs is None and _LAST_DRAFT_PROBS is not None:
        num_drafts = sum(metadata.num_draft_tokens)
        if num_drafts > _LAST_DRAFT_PROBS.shape[0]:
            raise RuntimeError(
                "not enough cached MTP probability rows: "
                f"need={num_drafts}, have={_LAST_DRAFT_PROBS.shape[0]}"
            )
        draft_probs = _LAST_DRAFT_PROBS[:num_drafts]

        if (
            not _DIAGNOSTIC_EMITTED
            and os.getenv("REMTP_PROB_MTP_DIAGNOSTICS", "1") == "1"
        ):
            ids = metadata.draft_token_ids.to(torch.int64)
            rows = torch.arange(num_drafts, device=draft_probs.device)
            q_at_draft = draft_probs[rows, ids]
            q_max = draft_probs.amax(dim=-1)
            entropy = -(
                draft_probs
                * torch.log(draft_probs.clamp_min(1e-30))
            ).sum(dim=-1)
            cached_ids = (
                _LAST_DRAFT_TOKEN_IDS[:num_drafts]
                if _LAST_DRAFT_TOKEN_IDS is not None
                else None
            )
            ids_match = (
                bool(torch.equal(ids, cached_ids))
                if cached_ids is not None
                else False
            )
            print(
                "[ReMTP][ProbMTP][diagnostic] "
                f"full_q=1 ids_match={ids_match} "
                f"seeded_rows={list(_LAST_GENERATOR_ROWS)} "
                f"draft_ids={ids.tolist()} "
                f"q(D)={q_at_draft.tolist()} "
                f"max(q)={q_max.tolist()} "
                f"H(q)={entropy.tolist()}",
                flush=True,
            )
            _DIAGNOSTIC_EMITTED = True

    if (
        draft_probs is None
        and not sampling_metadata.all_greedy
        and _PROPOSAL_COUNT > 0
    ):
        raise RuntimeError(
            "a real probabilistic MTP request is missing the full q tensor"
        )

    return original(
        self,
        metadata,
        draft_probs,
        logits,
        sampling_metadata,
    )


def set_bonus_logits_hook(hook: Any | None) -> None:
    """Expose the unmodified target bonus row to an optional observer."""
    global _BONUS_LOGITS_HOOK

    _BONUS_LOGITS_HOOK = hook


def install_probabilistic_mtp() -> None:
    """Patch vLLM 0.18's hybrid MTP proposer and verifier data path."""
    eagle_module = importlib.import_module(_EAGLE_MODULE)
    proposer_cls = eagle_module.EagleProposer

    current_sample = proposer_cls._greedy_sample
    if not getattr(current_sample, "_remtp_probabilistic_mtp", False):
        _sample_from_full_mtp_distribution._remtp_probabilistic_mtp = True
        _sample_from_full_mtp_distribution._remtp_original = current_sample
        proposer_cls._greedy_sample = _sample_from_full_mtp_distribution

    current_propose = proposer_cls.propose
    if not getattr(current_propose, "_remtp_probabilistic_mtp", False):
        _propose_with_full_distribution._remtp_probabilistic_mtp = True
        _propose_with_full_distribution._remtp_original = current_propose
        proposer_cls.propose = _propose_with_full_distribution

    rejection_module = importlib.import_module(_REJECTION_MODULE)
    sampler_cls = rejection_module.RejectionSampler
    current_forward = sampler_cls.forward
    if not getattr(current_forward, "_remtp_probabilistic_mtp", False):
        _forward_with_mtp_probs._remtp_probabilistic_mtp = True
        _forward_with_mtp_probs._remtp_original = current_forward
        sampler_cls.forward = _forward_with_mtp_probs

    print(
        "[ReMTP][ProbMTP] full proposal distributions enabled "
        "(temperature sampling, max_num_seqs=1)",
        flush=True,
    )
