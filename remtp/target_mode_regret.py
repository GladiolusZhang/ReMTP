"""High-confidence target-mode rescue with within-block regret feedback.

Broad Cactus/Exact-TV relaxation improves acceptance but can commit a draft
token that the target model does not actually prefer.  This verifier makes a
much narrower intervention:

* standard probabilistic MTP rejection sampling remains the baseline;
* extra probability is assigned only when the drafted token already has at
  least 0.5 probability under the processed target distribution, which makes
  it a target mode without an extra top-k/argmax operation;
* the boost saturates at q(y), because mass beyond q(y) cannot further improve
  acceptance;
* allocated TV mass becomes within-block regret debt.  Later MTP positions
  require progressively stronger target confidence and share one block cap.

The implementation reuses the target softmax already required by rejection
sampling, plus candidate gathers and one proportional rescale. It does not
compute vocabulary top-k, JS divergence, hidden similarity, or cross-block
state. A fused verifier path avoids the duplicate target softmax used by the
earlier Cactus/Exact-TV adapters.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch


_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False
_CONFIG: TargetModeRegretConfig | None = None
_VLLM_REJECTION: Any | None = None


@dataclass(frozen=True)
class TargetModeRegretConfig:
    """Controls conservative mode rescue and its negative feedback."""

    min_target_prob: float = 0.50
    per_token_tv_cap: float = 0.49
    block_tv_cap: float = 0.60
    debt_confidence_slope: float = 0.50
    depth_confidence_slope: float = 0.00
    diagnostics: bool = False

    @classmethod
    def from_env(cls) -> "TargetModeRegretConfig":
        config = cls(
            min_target_prob=float(
                os.getenv("REMTP_MODE_REGRET_MIN_TARGET_PROB", "0.50")
            ),
            per_token_tv_cap=float(
                os.getenv("REMTP_MODE_REGRET_PER_TOKEN_TV", "0.49")
            ),
            block_tv_cap=float(
                os.getenv("REMTP_MODE_REGRET_BLOCK_TV", "0.60")
            ),
            debt_confidence_slope=float(
                os.getenv("REMTP_MODE_REGRET_DEBT_SLOPE", "0.50")
            ),
            depth_confidence_slope=float(
                os.getenv("REMTP_MODE_REGRET_DEPTH_SLOPE", "0.00")
            ),
            diagnostics=os.getenv("REMTP_MODE_REGRET_DIAGNOSTICS", "0")
            == "1",
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.5 <= self.min_target_prob < 1.0:
            raise ValueError(
                "minimum target probability must be in [0.5, 1)"
            )
        if not 0.0 <= self.per_token_tv_cap < 1.0:
            raise ValueError("per-token TV cap must be in [0, 1)")
        if not 0.0 <= self.block_tv_cap < 1.0:
            raise ValueError("block TV cap must be in [0, 1)")
        if self.debt_confidence_slope < 0.0:
            raise ValueError("debt confidence slope must be non-negative")
        if self.depth_confidence_slope < 0.0:
            raise ValueError("depth confidence slope must be non-negative")


@dataclass(frozen=True)
class TargetModeRegretResult:
    probs: torch.Tensor
    target_candidate_probs: torch.Tensor
    draft_candidate_probs: torch.Tensor
    boosted_candidate_probs: torch.Tensor
    allocated_tv: torch.Tensor
    confidence_thresholds: torch.Tensor
    expected_regret: torch.Tensor


def target_mode_regret_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: TargetModeRegretConfig,
) -> TargetModeRegretResult:
    """Construct the temporary verification distribution in one pass."""

    config.validate()
    if target_probs.ndim != 2 or draft_probs.ndim != 2:
        raise ValueError("target and draft probabilities must be 2-D")
    if target_probs.shape != draft_probs.shape:
        raise ValueError("target and draft vocabulary layouts must match")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft token IDs must be one-dimensional")
    rows_count = target_probs.shape[0]
    if draft_token_ids.shape[0] != rows_count:
        raise ValueError("one draft token ID is required per probability row")
    if rows_count == 0:
        empty = target_probs.new_empty((0,), dtype=torch.float32)
        return TargetModeRegretResult(
            probs=target_probs,
            target_candidate_probs=empty,
            draft_candidate_probs=empty,
            boosted_candidate_probs=empty,
            allocated_tv=empty,
            confidence_thresholds=empty,
            expected_regret=empty,
        )

    probs = target_probs.to(torch.float32)
    ids = draft_token_ids.to(device=probs.device, dtype=torch.int64)
    rows = torch.arange(rows_count, device=probs.device)
    target_at_draft = probs[rows, ids].clamp(0.0, 1.0)
    draft_at_draft = draft_probs.to(torch.float32)[rows, ids].clamp(0.0, 1.0)

    required_tv = (draft_at_draft - target_at_draft).clamp_min(0.0)
    raw_tv = required_tv.clamp(max=config.per_token_tv_cap)

    depth = rows.to(torch.float32)
    base_thresholds = (
        config.min_target_prob + config.depth_confidence_slope * depth
    )
    base_eligible = target_at_draft >= base_thresholds
    raw_tv = torch.where(base_eligible, raw_tv, torch.zeros_like(raw_tv))

    # The expected distribution shift already allocated to earlier positions
    # is the causal regret debt visible to this position.
    prior_debt = raw_tv.cumsum(dim=0) - raw_tv
    confidence_thresholds = (
        base_thresholds + config.debt_confidence_slope * prior_debt
    ).clamp(max=1.0)
    debt_eligible = target_at_draft >= confidence_thresholds
    allocated_tv = torch.where(
        debt_eligible,
        raw_tv,
        torch.zeros_like(raw_tv),
    )

    prior_allocated = allocated_tv.cumsum(dim=0) - allocated_tv
    remaining_block_tv = (config.block_tv_cap - prior_allocated).clamp_min(0.0)
    allocated_tv = torch.minimum(allocated_tv, remaining_block_tv)
    boosted = (target_at_draft + allocated_tv).clamp(max=1.0 - 1e-6)

    scale = ((1.0 - boosted) / (1.0 - target_at_draft)).nan_to_num(
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    verification_probs = probs * scale.unsqueeze(-1)
    verification_probs.scatter_(1, ids.unsqueeze(1), boosted.unsqueeze(1))
    verification_probs = verification_probs.contiguous()

    strict_acceptance = torch.minimum(
        torch.ones_like(target_at_draft),
        target_at_draft / draft_at_draft.clamp_min(1e-30),
    )
    relaxed_acceptance = torch.minimum(
        torch.ones_like(boosted),
        boosted / draft_at_draft.clamp_min(1e-30),
    )
    expected_regret = allocated_tv * (
        relaxed_acceptance - strict_acceptance
    ).clamp_min(0.0)

    return TargetModeRegretResult(
        probs=verification_probs,
        target_candidate_probs=target_at_draft,
        draft_candidate_probs=draft_at_draft,
        boosted_candidate_probs=boosted,
        allocated_tv=allocated_tv,
        confidence_thresholds=confidence_thresholds,
        expected_regret=expected_regret,
    )


def _target_mode_regret_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Use the temporary mode-rescue distribution in standard verification."""

    global _DIAGNOSTIC_EMITTED

    original = getattr(
        _target_mode_regret_rejection_sample,
        "_remtp_original",
    )
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
        raise RuntimeError(
            "target-mode regret currently requires --max-num-seqs 1"
        )

    if _CONFIG is None or _VLLM_REJECTION is None:
        raise RuntimeError("target-mode regret has not been installed")
    config = _CONFIG
    module = _VLLM_REJECTION
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
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
        target_argmax = target_logits.argmax(dim=-1)
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

    # Unlike the earlier Cactus/Exact-TV adapters, compute softmax only once
    # and pass the already-adjusted probabilities directly to vLLM's recovery
    # sampler and rejection kernel.
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    result = target_mode_regret_distribution(
        target_probs,
        draft_probs,
        draft_token_ids,
        config,
    )
    if config.diagnostics and not _DIAGNOSTIC_EMITTED:
        print(
            "[ReMTP][TargetModeRegret][diagnostic] "
            f"p={result.target_candidate_probs.tolist()} "
            f"q={result.draft_candidate_probs.tolist()} "
            f"h={result.boosted_candidate_probs.tolist()} "
            f"tv={result.allocated_tv.tolist()} "
            f"threshold={result.confidence_thresholds.tolist()} "
            f"expected_regret={result.expected_regret.tolist()}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True

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
        draft_probs,
        result.probs,
        sampling_metadata,
        device,
    )
    module.rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        result.probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=False,
    )
    return output_token_ids


def install_target_mode_regret() -> None:
    """Install target-mode rescue after probabilistic MTP proposal capture."""

    global _CONFIG, _VLLM_REJECTION

    config = TargetModeRegretConfig.from_env()
    module = importlib.import_module(_REJECTION_MODULE)
    _CONFIG = config
    _VLLM_REJECTION = module
    current = module.rejection_sample
    if not getattr(current, "_remtp_target_mode_regret", False):
        _target_mode_regret_rejection_sample._remtp_target_mode_regret = True
        _target_mode_regret_rejection_sample._remtp_original = current
        module.rejection_sample = _target_mode_regret_rejection_sample

    print(
        "[ReMTP][TargetModeRegret] "
        f"min_p={config.min_target_prob:g} "
        f"per_token_tv={config.per_token_tv_cap:g} "
        f"block_tv={config.block_tv_cap:g} "
        f"debt_slope={config.debt_confidence_slope:g} "
        f"depth_slope={config.depth_confidence_slope:g}",
        flush=True,
    )
