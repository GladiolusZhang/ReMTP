"""High-confidence target-mode rescue with within-block regret feedback.

Broad Cactus/Exact-TV relaxation improves acceptance but can commit a draft
token that the target model does not actually prefer.  This verifier makes a
much narrower intervention:

* standard probabilistic MTP rejection sampling remains the baseline;
* the zero-overhead safety baseline assigns extra probability only when the
  drafted token already has at least 0.5 probability under the processed
  target distribution, which makes it a target mode without an argmax;
* the extended path admits lower-probability candidates only when they are the
  exact target top-1 and the target top-1/top-2 log-probability margin clears a
  configurable threshold;
* the target-band path admits a small target-head set when the drafted token's
  log-probability gap from target top-1 is bounded; within-block regret debt
  narrows that band for later heads;
* the boost saturates at q(y), because mass beyond q(y) cannot further improve
  acceptance;
* allocated TV mass becomes within-block regret debt.  Later MTP positions
  require progressively stronger target confidence and share one block cap.

The implementation reuses the target softmax already required by rejection
sampling, plus candidate gathers and one proportional rescale. The safety
baseline performs no vocabulary reduction. The extended path performs one
batched ``topk(k=2)`` over all six verification rows on the GPU; it does not
run a Python loop over heads or synchronize the result to the CPU. A fused
verifier path avoids the duplicate target softmax used by the earlier
Cactus/Exact-TV adapters.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch


_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False
_CONFIG: TargetModeRegretConfig | None = None
_VLLM_REJECTION: Any | None = None
_COMPILED_DISTRIBUTION: Callable[..., torch.Tensor] | None = None


@dataclass(frozen=True)
class TargetModeRegretConfig:
    """Controls target-anchored rescue and its within-block feedback."""

    min_target_prob: float = 0.50
    per_token_tv_cap: float = 0.49
    block_tv_cap: float = 0.60
    debt_confidence_slope: float = 0.50
    depth_confidence_slope: float = 0.00
    eligibility_mode: str = "probability"
    min_top1_margin: float = 0.0
    debt_margin_slope: float = 0.50
    depth_margin_slope: float = 0.0
    max_target_rank: int = 4
    max_candidate_log_gap: float = 0.50
    band_debt_slope: float = 0.75
    band_depth_slope: float = 0.0
    compile_distribution: bool = True
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
            eligibility_mode=os.getenv(
                "REMTP_MODE_REGRET_ELIGIBILITY", "probability"
            ),
            min_top1_margin=float(
                os.getenv("REMTP_MODE_REGRET_MIN_TOP1_MARGIN", "0.00")
            ),
            debt_margin_slope=float(
                os.getenv("REMTP_MODE_REGRET_MARGIN_DEBT_SLOPE", "0.50")
            ),
            depth_margin_slope=float(
                os.getenv("REMTP_MODE_REGRET_MARGIN_DEPTH_SLOPE", "0.00")
            ),
            max_target_rank=int(
                os.getenv("REMTP_MODE_REGRET_MAX_TARGET_RANK", "4")
            ),
            max_candidate_log_gap=float(
                os.getenv("REMTP_MODE_REGRET_MAX_CANDIDATE_GAP", "0.50")
            ),
            band_debt_slope=float(
                os.getenv("REMTP_MODE_REGRET_BAND_DEBT_SLOPE", "0.75")
            ),
            band_depth_slope=float(
                os.getenv("REMTP_MODE_REGRET_BAND_DEPTH_SLOPE", "0.00")
            ),
            compile_distribution=os.getenv(
                "REMTP_MODE_REGRET_COMPILE", "1"
            )
            == "1",
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
        if self.eligibility_mode not in {
            "probability",
            "top1_margin",
            "target_band",
        }:
            raise ValueError(
                "eligibility mode must be probability, top1_margin, "
                "or target_band"
            )
        if self.min_top1_margin < 0.0:
            raise ValueError("minimum top-1 margin must be non-negative")
        if self.debt_margin_slope < 0.0:
            raise ValueError("margin debt slope must be non-negative")
        if self.depth_margin_slope < 0.0:
            raise ValueError("margin depth slope must be non-negative")
        if self.max_target_rank < 1:
            raise ValueError("maximum target rank must be positive")
        if self.max_candidate_log_gap < 0.0:
            raise ValueError("maximum candidate log gap must be non-negative")
        if self.band_debt_slope < 0.0:
            raise ValueError("band debt slope must be non-negative")
        if self.band_depth_slope < 0.0:
            raise ValueError("band depth slope must be non-negative")


@dataclass(frozen=True)
class TargetModeRegretResult:
    probs: torch.Tensor
    target_candidate_probs: torch.Tensor
    draft_candidate_probs: torch.Tensor
    boosted_candidate_probs: torch.Tensor
    allocated_tv: torch.Tensor
    confidence_thresholds: torch.Tensor
    expected_regret: torch.Tensor
    eligibility: torch.Tensor
    target_top1_margins: torch.Tensor
    target_candidate_ranks: torch.Tensor
    target_candidate_log_gaps: torch.Tensor


def target_top1_margin_stats(
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact target top-1 IDs and top-1/top-2 log-prob gaps.

    ``target_logits`` already contains vLLM's temperature and sampling-logit
    processing.  Logit differences therefore equal log-probability
    differences under the processed target distribution.  One batched GPU
    reduction covers the entire MTP block.
    """

    if target_logits.ndim != 2:
        raise ValueError("target logits must be 2-D")
    if target_logits.shape[-1] < 2:
        raise ValueError("top-1 margin requires a vocabulary of at least two")
    top_values, top_ids = torch.topk(
        target_logits,
        k=2,
        dim=-1,
        largest=True,
        sorted=True,
    )
    margins = (top_values[:, 0] - top_values[:, 1]).to(torch.float32)
    return top_ids[:, 0].to(torch.int64), margins.clamp_min(0.0)


def target_band_stats(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    max_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return candidate rank, top-1 gap, and top-1/top-2 margin.

    The vocabulary reduction is performed once for all MTP heads. Candidate
    gaps use a direct gather from the same processed target logits, so no
    second softmax or vocabulary-sized temporary is introduced.
    """

    if target_logits.ndim != 2:
        raise ValueError("target logits must be 2-D")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft token IDs must be one-dimensional")
    if target_logits.shape[0] != draft_token_ids.shape[0]:
        raise ValueError("one draft token ID is required per logit row")
    if max_rank < 1:
        raise ValueError("maximum target rank must be positive")
    top_k = max(2, max_rank)
    if target_logits.shape[-1] < top_k:
        raise ValueError("target vocabulary is smaller than requested top-k")

    top_values, top_ids = torch.topk(
        target_logits,
        k=top_k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    ids = draft_token_ids.to(device=target_logits.device, dtype=torch.int64)
    matches = top_ids[:, :max_rank] == ids.unsqueeze(1)
    positions = torch.arange(
        1,
        max_rank + 1,
        device=target_logits.device,
        dtype=torch.int64,
    ).unsqueeze(0)
    candidate_ranks = torch.where(
        matches,
        positions,
        torch.full_like(positions, max_rank + 1),
    ).amin(dim=-1)
    candidate_logits = target_logits.gather(1, ids.unsqueeze(1)).squeeze(1)
    candidate_gaps = (top_values[:, 0] - candidate_logits).to(torch.float32)
    top1_margins = (top_values[:, 0] - top_values[:, 1]).to(torch.float32)
    return (
        candidate_ranks,
        candidate_gaps.clamp_min(0.0),
        top1_margins.clamp_min(0.0),
    )


def target_mode_regret_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: TargetModeRegretConfig,
    *,
    target_top1_ids: torch.Tensor | None = None,
    target_top1_margins: torch.Tensor | None = None,
    target_candidate_ranks: torch.Tensor | None = None,
    target_candidate_log_gaps: torch.Tensor | None = None,
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
            eligibility=torch.empty(
                (0,), device=target_probs.device, dtype=torch.bool
            ),
            target_top1_margins=empty,
            target_candidate_ranks=torch.empty(
                (0,), device=target_probs.device, dtype=torch.int64
            ),
            target_candidate_log_gaps=empty,
        )

    probs = target_probs.to(torch.float32)
    ids = draft_token_ids.to(device=probs.device, dtype=torch.int64)
    rows = torch.arange(rows_count, device=probs.device)
    target_at_draft = probs[rows, ids].clamp(0.0, 1.0)
    draft_at_draft = draft_probs.to(torch.float32)[rows, ids].clamp(0.0, 1.0)

    required_tv = (draft_at_draft - target_at_draft).clamp_min(0.0)
    raw_tv = required_tv.clamp(max=config.per_token_tv_cap)

    depth = rows.to(torch.float32)
    reported_ranks = torch.ones_like(ids)
    reported_candidate_gaps = torch.zeros_like(target_at_draft)
    greater_is_better = True
    if config.eligibility_mode == "probability":
        base_thresholds = (
            config.min_target_prob + config.depth_confidence_slope * depth
        )
        eligibility_values = target_at_draft
        target_is_top1 = torch.ones_like(target_at_draft, dtype=torch.bool)
        reported_margins = torch.zeros_like(target_at_draft)
        debt_slope = config.debt_confidence_slope
    elif config.eligibility_mode == "top1_margin":
        if target_top1_ids is None or target_top1_margins is None:
            raise ValueError(
                "top1_margin eligibility requires target top-1 IDs and margins"
            )
        if target_top1_ids.shape != draft_token_ids.shape:
            raise ValueError("one target top-1 ID is required per row")
        if target_top1_margins.shape != draft_token_ids.shape:
            raise ValueError("one target top-1 margin is required per row")
        top1_ids = target_top1_ids.to(device=probs.device, dtype=torch.int64)
        reported_margins = target_top1_margins.to(
            device=probs.device,
            dtype=torch.float32,
        ).clamp_min(0.0)
        base_thresholds = (
            config.min_top1_margin + config.depth_margin_slope * depth
        )
        eligibility_values = reported_margins
        target_is_top1 = ids == top1_ids
        debt_slope = config.debt_margin_slope
    else:
        if target_candidate_ranks is None or target_candidate_log_gaps is None:
            raise ValueError(
                "target_band eligibility requires candidate ranks and gaps"
            )
        if target_candidate_ranks.shape != draft_token_ids.shape:
            raise ValueError("one target candidate rank is required per row")
        if target_candidate_log_gaps.shape != draft_token_ids.shape:
            raise ValueError("one target candidate gap is required per row")
        reported_ranks = target_candidate_ranks.to(
            device=probs.device,
            dtype=torch.int64,
        )
        reported_candidate_gaps = target_candidate_log_gaps.to(
            device=probs.device,
            dtype=torch.float32,
        ).clamp_min(0.0)
        if target_top1_margins is None:
            reported_margins = torch.zeros_like(target_at_draft)
        else:
            reported_margins = target_top1_margins.to(
                device=probs.device,
                dtype=torch.float32,
            ).clamp_min(0.0)
        base_thresholds = (
            config.max_candidate_log_gap - config.band_depth_slope * depth
        ).clamp_min(0.0)
        eligibility_values = reported_candidate_gaps
        target_is_top1 = reported_ranks <= config.max_target_rank
        debt_slope = config.band_debt_slope
        greater_is_better = False

    if greater_is_better:
        base_eligible = target_is_top1 & (
            eligibility_values >= base_thresholds
        )
    else:
        base_eligible = target_is_top1 & (
            eligibility_values <= base_thresholds
        )
    raw_tv = torch.where(base_eligible, raw_tv, torch.zeros_like(raw_tv))

    if config.eligibility_mode == "target_band":
        # Use *actual* preceding allocation as debt. The six-step loop is
        # statically unrolled by torch.compile in the serving path, so the
        # exact recurrence does not introduce CPU synchronization or a Python
        # loop at inference time.
        running_debt = target_at_draft.new_zeros(())
        threshold_parts: list[torch.Tensor] = []
        eligibility_parts: list[torch.Tensor] = []
        allocation_parts: list[torch.Tensor] = []
        for index in range(rows_count):
            threshold = (
                base_thresholds[index] - debt_slope * running_debt
            ).clamp_min(0.0)
            eligible = target_is_top1[index] & (
                eligibility_values[index] <= threshold
            )
            remaining = (config.block_tv_cap - running_debt).clamp_min(0.0)
            allocation = torch.where(
                eligible,
                torch.minimum(raw_tv[index], remaining),
                torch.zeros_like(raw_tv[index]),
            )
            threshold_parts.append(threshold)
            eligibility_parts.append(eligible)
            allocation_parts.append(allocation)
            running_debt = running_debt + allocation
        confidence_thresholds = torch.stack(threshold_parts)
        debt_eligible = torch.stack(eligibility_parts)
        allocated_tv = torch.stack(allocation_parts)
    else:
        # The original vector path is retained bit-for-bit for the completed
        # p>=0.5 and exact-top-1 ablations.
        prior_debt = raw_tv.cumsum(dim=0) - raw_tv
        confidence_thresholds = base_thresholds + debt_slope * prior_debt
        if config.eligibility_mode == "probability":
            confidence_thresholds = confidence_thresholds.clamp(max=1.0)
        debt_eligible = target_is_top1 & (
            eligibility_values >= confidence_thresholds
        )
        allocated_tv = torch.where(
            debt_eligible,
            raw_tv,
            torch.zeros_like(raw_tv),
        )
        prior_allocated = allocated_tv.cumsum(dim=0) - allocated_tv
        remaining_block_tv = (
            config.block_tv_cap - prior_allocated
        ).clamp_min(0.0)
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
        eligibility=debt_eligible,
        target_top1_margins=reported_margins,
        target_candidate_ranks=reported_ranks,
        target_candidate_log_gaps=reported_candidate_gaps,
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

    global _COMPILED_DISTRIBUTION, _DIAGNOSTIC_EMITTED

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
    target_top1_ids: torch.Tensor | None = None
    target_top1_margins: torch.Tensor | None = None
    target_candidate_ranks: torch.Tensor | None = None
    target_candidate_log_gaps: torch.Tensor | None = None
    if config.eligibility_mode == "top1_margin":
        target_top1_ids, target_top1_margins = target_top1_margin_stats(
            target_logits
        )
    elif config.eligibility_mode == "target_band":
        (
            target_candidate_ranks,
            target_candidate_log_gaps,
            target_top1_margins,
        ) = target_band_stats(
            target_logits,
            draft_token_ids,
            config.max_target_rank,
        )
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    result: TargetModeRegretResult | None = None
    verification_probs: torch.Tensor
    if (
        _COMPILED_DISTRIBUTION is not None
        and target_top1_margins is not None
        and (
            target_top1_ids is not None
            or (
                target_candidate_ranks is not None
                and target_candidate_log_gaps is not None
            )
        )
        and not config.diagnostics
    ):
        try:
            if config.eligibility_mode == "top1_margin":
                assert target_top1_ids is not None
                stat_a = target_top1_ids
                stat_b = target_top1_margins
            else:
                assert target_candidate_ranks is not None
                assert target_candidate_log_gaps is not None
                stat_a = target_candidate_ranks
                stat_b = target_candidate_log_gaps
            verification_probs = _COMPILED_DISTRIBUTION(
                target_probs,
                draft_probs,
                draft_token_ids,
                stat_a,
                stat_b,
            )
        except Exception as exc:  # pragma: no cover - backend dependent
            # Compilation is a performance optimization, never a semantic
            # requirement. Fall back to eager tensor operations if the local
            # torch/Triton stack cannot compile this small verifier graph.
            print(
                "[ReMTP][TargetModeRegret] verifier compilation failed; "
                f"falling back to eager: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _COMPILED_DISTRIBUTION = None
            result = target_mode_regret_distribution(
                target_probs,
                draft_probs,
                draft_token_ids,
                config,
                target_top1_ids=target_top1_ids,
                target_top1_margins=target_top1_margins,
                target_candidate_ranks=target_candidate_ranks,
                target_candidate_log_gaps=target_candidate_log_gaps,
            )
            verification_probs = result.probs
    else:
        result = target_mode_regret_distribution(
            target_probs,
            draft_probs,
            draft_token_ids,
            config,
            target_top1_ids=target_top1_ids,
            target_top1_margins=target_top1_margins,
            target_candidate_ranks=target_candidate_ranks,
            target_candidate_log_gaps=target_candidate_log_gaps,
        )
        verification_probs = result.probs
    if config.diagnostics and not _DIAGNOSTIC_EMITTED:
        assert result is not None
        print(
            "[ReMTP][TargetModeRegret][diagnostic] "
            f"p={result.target_candidate_probs.tolist()} "
            f"q={result.draft_candidate_probs.tolist()} "
            f"h={result.boosted_candidate_probs.tolist()} "
            f"tv={result.allocated_tv.tolist()} "
            f"threshold={result.confidence_thresholds.tolist()} "
            f"margin={result.target_top1_margins.tolist()} "
            f"rank={result.target_candidate_ranks.tolist()} "
            f"gap={result.target_candidate_log_gaps.tolist()} "
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
        verification_probs,
        sampling_metadata,
        device,
    )
    module.rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        verification_probs,
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

    global _COMPILED_DISTRIBUTION, _CONFIG, _VLLM_REJECTION

    config = TargetModeRegretConfig.from_env()
    module = importlib.import_module(_REJECTION_MODULE)
    _CONFIG = config
    _VLLM_REJECTION = module
    _COMPILED_DISTRIBUTION = None
    if (
        config.compile_distribution
        and config.eligibility_mode in {"top1_margin", "target_band"}
        and not config.diagnostics
        and config.block_tv_cap > 0.0
    ):
        def distribution_probs(
            target_probs: torch.Tensor,
            draft_probs: torch.Tensor,
            draft_token_ids: torch.Tensor,
            stat_a: torch.Tensor,
            stat_b: torch.Tensor,
        ) -> torch.Tensor:
            if config.eligibility_mode == "top1_margin":
                kwargs = {
                    "target_top1_ids": stat_a,
                    "target_top1_margins": stat_b,
                }
            else:
                kwargs = {
                    "target_candidate_ranks": stat_a,
                    "target_candidate_log_gaps": stat_b,
                }
            return target_mode_regret_distribution(
                target_probs,
                draft_probs,
                draft_token_ids,
                config,
                **kwargs,
            ).probs

        _COMPILED_DISTRIBUTION = torch.compile(
            distribution_probs,
            fullgraph=True,
            dynamic=False,
        )
    current = module.rejection_sample
    if not getattr(current, "_remtp_target_mode_regret", False):
        _target_mode_regret_rejection_sample._remtp_target_mode_regret = True
        _target_mode_regret_rejection_sample._remtp_original = current
        module.rejection_sample = _target_mode_regret_rejection_sample

    print(
        "[ReMTP][TargetModeRegret] "
        f"eligibility={config.eligibility_mode} "
        f"min_p={config.min_target_prob:g} "
        f"min_margin={config.min_top1_margin:g} "
        f"per_token_tv={config.per_token_tv_cap:g} "
        f"block_tv={config.block_tv_cap:g} "
        f"debt_slope={config.debt_confidence_slope:g} "
        f"depth_slope={config.depth_confidence_slope:g} "
        f"margin_debt_slope={config.debt_margin_slope:g} "
        f"margin_depth_slope={config.depth_margin_slope:g} "
        f"max_rank={config.max_target_rank} "
        f"max_gap={config.max_candidate_log_gap:g} "
        f"band_debt_slope={config.band_debt_slope:g} "
        f"band_depth_slope={config.band_depth_slope:g} "
        f"compiled={int(_COMPILED_DISTRIBUTION is not None)}",
        flush=True,
    )
