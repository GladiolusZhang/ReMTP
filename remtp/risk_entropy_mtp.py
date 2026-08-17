"""Risk/entropy relaxation and same-round sentinel feedback for native MTP.

The module implements three verification profiles over the same probabilistic
MTP proposal ``q`` and processed target distribution ``p``:

``scheme1``
    Position-aware marginal/entropy relaxation with a cumulative risk budget.
``scheme2``
    A fixed local target band plus a next-position sentinel and cross-round
    risk debt.
``scheme12``
    Scheme 1 eligibility followed by Scheme 2's sentinel and debt controller.
``scheme2_relaxed``
    A stronger Scheme 2 profile: fixed target support, a soft same-round
    sentinel, and cross-round debt without Scheme 1's duplicate cumulative
    risk constraint.
``scheme12_joint``
    A joint Scheme 1+2 allocator. Scheme 1 supplies a continuous target-risk
    price and Scheme 2 supplies future-support credit; one exact block-TV
    budget is distributed by marginal expected prefix value.
``scheme12_anchored``
    A dual-pool joint allocator. Most TV is reserved for target-supported
    candidates and only a small exploration pool may rescue broader tokens.
``remtp``
    The frozen main method.  It keeps Scheme 2's same-round sentinel and
    request-local debt, calibrates the candidate band by the target top-1
    margin, and water-fills reclaimed TV toward high-value native-MTP prefix
    positions.
``remtp_block``
    ReMTP plus a target-anchored prefix certificate.  A locally borderline
    token may use the block budget only when the already-computed target
    verification strongly supports the following draft prefix.

All profiles construct a complete temporary distribution ``h`` and delegate
acceptance and residual recovery to vLLM's standard speculative sampler.  The
target bonus/recovery token is therefore preserved as the target anchor.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from typing import Any

import torch


_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_GPU_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
_CONFIG: "RiskEntropyConfig | None" = None
_VLLM_REJECTION: Any | None = None
_DIAGNOSTIC_EMITTED = False
_SENTINEL_VARIANTS = {
    "scheme2",
    "scheme12",
    "scheme2_relaxed",
    "scheme12_joint",
    "scheme12_anchored",
    "remtp",
    "remtp_block",
}
_DEBT_VARIANTS = _SENTINEL_VARIANTS


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True)
class RiskEntropyConfig:
    """Inference-only controls shared by Schemes 1, 2 and 1+2."""

    variant: str = "scheme1"
    expected_draft_tokens: int = 6
    max_target_rank: int = 8
    scheme2_rank_limit: int = 4
    top_m: int = 4
    base_log_gap: float = 0.90
    ambiguity_scale: float = 0.60
    threshold_position_scale: float = 0.35
    rank_position_scale: float = 0.50
    risk_budget: float = 2.40
    risk_position_scale: float = 0.50
    risk_confidence_scale: float = 0.50
    mtp_confidence_floor: float = 0.05
    cactus_delta: float = 1.0
    per_token_tv_cap: float = 0.25
    block_tv_cap: float = 0.90
    joint_risk_mass_cap: float = 0.90
    joint_risk_temperature: float = 1.50
    joint_prefix_weight: float = 1.00
    joint_core_rank: int = 2
    joint_core_log_gap: float = 0.75
    joint_core_fraction: float = 0.85
    joint_explore_min_checks: int = 2
    joint_support_power: float = 1.00
    joint_explore_risk_multiplier: float = 2.00
    sentinel_log_gap: float = 1.00
    sentinel_mtp_floor: float = 0.02
    sentinel_head_mass_floor: float = 0.10
    sentinel_min_checks: int = 2
    sentinel_soft_floor: float = 1.00
    last_token_gap_scale: float = 0.70
    remtp_gap_floor: float = 1.15
    remtp_margin_temperature: float = 0.75
    remtp_risky_rank: int = 4
    remtp_risky_log_gap: float = 1.20
    remtp_risky_min_checks: int = 2
    remtp_prefix_weight: float = 0.75
    remtp_support_temperature: float = 1.00
    remtp_continuation_gap: float = 2.50
    remtp_continuation_support: float = 0.65
    remtp_continuation_regret: float = 0.50
    remtp_continuation_horizon: int = 3
    debt_decay: float = 0.80
    debt_scale: float = 0.75
    adaptive_draft_depth: bool = True
    medium_draft_depth: int = 4
    short_draft_depth: int = 2
    medium_debt_threshold: float = 0.50
    high_debt_threshold: float = 1.50
    diagnostics: bool = False
    audit_path: str = ""
    audit_dataset: str = "unknown"

    @classmethod
    def from_env(cls) -> "RiskEntropyConfig":
        config = cls(
            variant=os.getenv("REMTP_RISK_VARIANT", "scheme1"),
            expected_draft_tokens=int(
                os.getenv("REMTP_RISK_EXPECTED_DRAFT_TOKENS", "6")
            ),
            max_target_rank=int(os.getenv("REMTP_RISK_MAX_RANK", "8")),
            scheme2_rank_limit=int(
                os.getenv("REMTP_RISK_SCHEME2_RANK", "4")
            ),
            top_m=int(os.getenv("REMTP_RISK_TOP_M", "4")),
            base_log_gap=float(os.getenv("REMTP_RISK_BASE_GAP", "0.90")),
            ambiguity_scale=float(
                os.getenv("REMTP_RISK_AMBIGUITY_SCALE", "0.60")
            ),
            threshold_position_scale=float(
                os.getenv("REMTP_RISK_THRESHOLD_POSITION_SCALE", "0.35")
            ),
            rank_position_scale=float(
                os.getenv("REMTP_RISK_RANK_POSITION_SCALE", "0.50")
            ),
            risk_budget=float(os.getenv("REMTP_RISK_BUDGET", "2.40")),
            risk_position_scale=float(
                os.getenv("REMTP_RISK_POSITION_SCALE", "0.50")
            ),
            risk_confidence_scale=float(
                os.getenv("REMTP_RISK_CONFIDENCE_SCALE", "0.50")
            ),
            mtp_confidence_floor=float(
                os.getenv("REMTP_RISK_MTP_CONFIDENCE_FLOOR", "0.05")
            ),
            cactus_delta=float(os.getenv("REMTP_CACTUS_DELTA", "1.0")),
            per_token_tv_cap=float(
                os.getenv("REMTP_RISK_PER_TOKEN_TV", "0.25")
            ),
            block_tv_cap=float(
                os.getenv("REMTP_RISK_BLOCK_TV", "0.90")
            ),
            joint_risk_mass_cap=float(
                os.getenv("REMTP_JOINT_RISK_MASS", "0.90")
            ),
            joint_risk_temperature=float(
                os.getenv("REMTP_JOINT_RISK_TEMPERATURE", "1.50")
            ),
            joint_prefix_weight=float(
                os.getenv("REMTP_JOINT_PREFIX_WEIGHT", "1.00")
            ),
            joint_core_rank=int(
                os.getenv("REMTP_JOINT_CORE_RANK", "2")
            ),
            joint_core_log_gap=float(
                os.getenv("REMTP_JOINT_CORE_GAP", "0.75")
            ),
            joint_core_fraction=float(
                os.getenv("REMTP_JOINT_CORE_FRACTION", "0.85")
            ),
            joint_explore_min_checks=int(
                os.getenv("REMTP_JOINT_EXPLORE_MIN_CHECKS", "2")
            ),
            joint_support_power=float(
                os.getenv("REMTP_JOINT_SUPPORT_POWER", "1.00")
            ),
            joint_explore_risk_multiplier=float(
                os.getenv("REMTP_JOINT_EXPLORE_RISK_MULTIPLIER", "2.00")
            ),
            sentinel_log_gap=float(
                os.getenv("REMTP_SENTINEL_MAX_GAP", "1.00")
            ),
            sentinel_mtp_floor=float(
                os.getenv("REMTP_SENTINEL_MTP_FLOOR", "0.02")
            ),
            sentinel_head_mass_floor=float(
                os.getenv("REMTP_SENTINEL_HEAD_MASS_FLOOR", "0.10")
            ),
            sentinel_min_checks=int(
                os.getenv("REMTP_SENTINEL_MIN_CHECKS", "2")
            ),
            sentinel_soft_floor=float(
                os.getenv("REMTP_SENTINEL_SOFT_FLOOR", "1.00")
            ),
            last_token_gap_scale=float(
                os.getenv("REMTP_SENTINEL_LAST_GAP_SCALE", "0.70")
            ),
            remtp_gap_floor=float(
                os.getenv("REMTP_FINAL_GAP_FLOOR", "1.15")
            ),
            remtp_margin_temperature=float(
                os.getenv("REMTP_FINAL_MARGIN_TEMPERATURE", "0.75")
            ),
            remtp_risky_rank=int(
                os.getenv("REMTP_FINAL_RISKY_RANK", "4")
            ),
            remtp_risky_log_gap=float(
                os.getenv("REMTP_FINAL_RISKY_GAP", "1.20")
            ),
            remtp_risky_min_checks=int(
                os.getenv("REMTP_FINAL_RISKY_MIN_CHECKS", "2")
            ),
            remtp_prefix_weight=float(
                os.getenv("REMTP_FINAL_PREFIX_WEIGHT", "0.75")
            ),
            remtp_support_temperature=float(
                os.getenv("REMTP_FINAL_SUPPORT_TEMPERATURE", "1.00")
            ),
            remtp_continuation_gap=float(
                os.getenv("REMTP_BLOCK_CONTINUATION_GAP", "2.50")
            ),
            remtp_continuation_support=float(
                os.getenv("REMTP_BLOCK_CONTINUATION_SUPPORT", "0.65")
            ),
            remtp_continuation_regret=float(
                os.getenv("REMTP_BLOCK_CONTINUATION_REGRET", "0.50")
            ),
            remtp_continuation_horizon=int(
                os.getenv("REMTP_BLOCK_CONTINUATION_HORIZON", "3")
            ),
            debt_decay=float(os.getenv("REMTP_RISK_DEBT_DECAY", "0.80")),
            debt_scale=float(os.getenv("REMTP_RISK_DEBT_SCALE", "0.75")),
            adaptive_draft_depth=_env_flag(
                "REMTP_RISK_ADAPTIVE_DRAFT", True
            ),
            medium_draft_depth=int(
                os.getenv("REMTP_RISK_MEDIUM_DRAFT_DEPTH", "4")
            ),
            short_draft_depth=int(
                os.getenv("REMTP_RISK_SHORT_DRAFT_DEPTH", "2")
            ),
            medium_debt_threshold=float(
                os.getenv("REMTP_RISK_MEDIUM_DEBT", "0.50")
            ),
            high_debt_threshold=float(
                os.getenv("REMTP_RISK_HIGH_DEBT", "1.50")
            ),
            diagnostics=_env_flag("REMTP_RISK_DIAGNOSTICS", False),
            audit_path=os.getenv("REMTP_RISK_AUDIT_PATH", "").strip(),
            audit_dataset=os.getenv(
                "REMTP_RISK_AUDIT_DATASET", "unknown"
            ).strip(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.variant not in {
            "scheme1",
            "scheme2",
            "scheme12",
            "scheme2_relaxed",
            "scheme12_joint",
            "scheme12_anchored",
            "remtp",
            "remtp_block",
        }:
            raise ValueError(
                "variant must be scheme1, scheme2, scheme12, or "
                "scheme2_relaxed, scheme12_joint, scheme12_anchored, or "
                "remtp or remtp_block"
            )
        if self.expected_draft_tokens < 1:
            raise ValueError("expected draft tokens must be positive")
        if (
            self.max_target_rank < 1
            or self.scheme2_rank_limit < 1
            or self.top_m < 1
        ):
            raise ValueError("rank and top-m must be positive")
        if min(
            self.base_log_gap,
            self.rank_position_scale,
            self.risk_budget,
            self.cactus_delta,
            self.per_token_tv_cap,
            self.block_tv_cap,
            self.joint_risk_mass_cap,
            self.joint_risk_temperature,
            self.joint_prefix_weight,
            self.joint_core_log_gap,
            self.joint_support_power,
            self.joint_explore_risk_multiplier,
            self.sentinel_log_gap,
            self.sentinel_mtp_floor,
            self.sentinel_head_mass_floor,
            self.remtp_gap_floor,
            self.remtp_risky_log_gap,
            self.remtp_prefix_weight,
            self.remtp_continuation_gap,
            self.remtp_continuation_regret,
            self.debt_scale,
        ) < 0.0:
            raise ValueError("risk, gap, TV and debt values must be non-negative")
        if not 0.0 <= self.debt_decay <= 1.0:
            raise ValueError("debt decay must be in [0,1]")
        if not 0.0 < self.last_token_gap_scale <= 1.0:
            raise ValueError("last-token gap scale must be in (0,1]")
        if not 1 <= self.sentinel_min_checks <= 3:
            raise ValueError("sentinel minimum checks must be between 1 and 3")
        if not 1 <= self.remtp_risky_min_checks <= 3:
            raise ValueError("ReMTP risky minimum checks must be between 1 and 3")
        if self.remtp_margin_temperature <= 0.0:
            raise ValueError("ReMTP margin temperature must be positive")
        if self.remtp_support_temperature <= 0.0:
            raise ValueError("ReMTP support temperature must be positive")
        if self.variant in {"remtp", "remtp_block"}:
            if not 1 <= self.remtp_risky_rank <= self.max_target_rank:
                raise ValueError(
                    "ReMTP risky rank must be within the target top-k"
                )
            if self.remtp_gap_floor > self.base_log_gap:
                raise ValueError(
                    "ReMTP gap floor cannot exceed the gap ceiling"
                )
        if not 0.0 <= self.remtp_continuation_support <= 1.0:
            raise ValueError("continuation support must be in [0,1]")
        if self.remtp_continuation_horizon < 1:
            raise ValueError("continuation horizon must be positive")
        if not 0.0 <= self.sentinel_soft_floor <= 1.0:
            raise ValueError("sentinel soft floor must be in [0,1]")
        if self.joint_risk_temperature <= 0.0:
            raise ValueError("joint risk temperature must be positive")
        if self.joint_core_rank < 1:
            raise ValueError("joint core rank must be positive")
        if not 0.0 <= self.joint_core_fraction <= 1.0:
            raise ValueError("joint core fraction must be in [0,1]")
        if not 1 <= self.joint_explore_min_checks <= 3:
            raise ValueError("joint exploration checks must be between 1 and 3")
        if not (
            1
            <= self.short_draft_depth
            <= self.medium_draft_depth
            <= self.expected_draft_tokens
        ):
            raise ValueError("debt-controlled draft depths are inconsistent")
        if not (
            0.0 <= self.medium_debt_threshold <= self.high_debt_threshold
        ):
            raise ValueError("debt depth thresholds are inconsistent")


@dataclass
class _RiskState:
    debt: torch.Tensor | None = None
    last_output_length: int | None = None
    next_draft_depth: int = 6
    request_id: str | None = None
    active_request_id: str | None = None
    active_output_length: int = 0

    def reset(self) -> None:
        self.debt = None
        self.last_output_length = None
        self.next_draft_depth = 6
        self.request_id = None
        self.active_request_id = None
        self.active_output_length = 0


@dataclass(frozen=True)
class RiskEntropyResult:
    probs: torch.Tensor
    target_candidate_probs: torch.Tensor
    draft_candidate_probs: torch.Tensor
    boosted_candidate_probs: torch.Tensor
    strict_acceptance: torch.Tensor
    relaxed_acceptance: torch.Tensor
    allocated_tv: torch.Tensor
    local_risk: torch.Tensor
    cumulative_risk: torch.Tensor
    candidate_rank: torch.Tensor
    candidate_log_gap: torch.Tensor
    target_margin: torch.Tensor
    compact_entropy: torch.Tensor
    threshold: torch.Tensor
    local_eligible: torch.Tensor
    sentinel_pass: torch.Tensor
    sentinel_checks: torch.Tensor
    sentinel_strength: torch.Tensor
    allocation_utility: torch.Tensor
    allocated_risk_mass: torch.Tensor
    core_allocated_tv: torch.Tensor
    relaxed_prefix: torch.Tensor
    desired_tv: torch.Tensor
    target_head_mass: torch.Tensor
    continuation_support: torch.Tensor
    continuation_regret: torch.Tensor
    positive_certificate: torch.Tensor


_STATE = _RiskState()


def _request_output_length(sampling_metadata: Any) -> int:
    rows = getattr(sampling_metadata, "output_token_ids", None)
    if not rows:
        return 0
    if len(rows) != 1:
        raise RuntimeError("risk-entropy verification requires --max-num-seqs 1")
    return len(rows[0])


def _begin_round(sampling_metadata: Any, device: torch.device) -> torch.Tensor:
    # ``SamplingMetadata.output_token_ids`` is deliberately empty in vLLM's
    # async fast path unless a penalty/logits processor needs it.  Request
    # identity is therefore captured from GPUModelRunner._sample instead of
    # inferring boundaries from that optional list.
    if _STATE.active_request_id is not None:
        if _STATE.active_request_id != _STATE.request_id:
            _STATE.request_id = _STATE.active_request_id
            _STATE.debt = torch.zeros((), device=device, dtype=torch.float32)
            _STATE.next_draft_depth = (
                _CONFIG.expected_draft_tokens if _CONFIG is not None else 6
            )
        _STATE.last_output_length = _STATE.active_output_length
        if _STATE.debt is None:
            _STATE.debt = torch.zeros((), device=device, dtype=torch.float32)
        return _STATE.debt.to(device=device, dtype=torch.float32)

    current_length = _request_output_length(sampling_metadata)
    # Fallback for direct unit tests and older synchronous vLLM releases.
    # A perpetually empty list is not evidence of a new request.
    if _STATE.last_output_length is None or (
        current_length > 0 and current_length <= _STATE.last_output_length
    ):
        _STATE.debt = torch.zeros((), device=device, dtype=torch.float32)
    _STATE.last_output_length = current_length
    if _STATE.debt is None:
        _STATE.debt = torch.zeros((), device=device, dtype=torch.float32)
    return _STATE.debt.to(device=device, dtype=torch.float32)


def _risk_runner_sample(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Expose the active vLLM request to the verification callback.

    The wrapper performs CPU-only list/dict reads already present in the
    runner.  It adds no device synchronization and keeps the hot rejection
    path free of request-ID plumbing changes to vLLM itself.
    """

    original = getattr(_risk_runner_sample, "_remtp_original")
    request_ids = list(getattr(self.input_batch, "req_ids", ()))
    if len(request_ids) > 1:
        raise RuntimeError("risk-entropy verification requires --max-num-seqs 1")
    if request_ids:
        request_id = str(request_ids[0])
        _STATE.active_request_id = request_id
        request_state = getattr(self, "requests", {}).get(request_ids[0])
        output_ids = getattr(request_state, "output_token_ids", ())
        _STATE.active_output_length = len(output_ids)
    else:
        _STATE.active_request_id = None
        _STATE.active_output_length = 0
    return original(self, *args, **kwargs)


def _top_statistics(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: RiskEntropyConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    top_count = max(2, config.max_target_rank, config.top_m)
    top_count = min(top_count, target_logits.shape[-1])
    top_values, top_ids = torch.topk(
        target_logits, k=top_count, dim=-1, largest=True, sorted=True
    )
    ids = draft_token_ids.to(device=target_logits.device, dtype=torch.int64)
    candidate_logits = target_logits.gather(1, ids.unsqueeze(1)).squeeze(1)
    gap = (top_values[:, 0] - candidate_logits).to(torch.float32).clamp(0.0, 30.0)
    margin = (top_values[:, 0] - top_values[:, 1]).to(torch.float32).clamp_min(0.0)

    rank_ids = top_ids[:, : min(config.max_target_rank, top_count)]
    matches = rank_ids == ids.unsqueeze(1)
    rank_positions = torch.arange(
        1, rank_ids.shape[1] + 1, device=target_logits.device, dtype=torch.int64
    ).unsqueeze(0)
    rank = torch.where(
        matches,
        rank_positions,
        torch.full_like(rank_positions, config.max_target_rank + 1),
    ).amin(dim=-1)

    entropy_count = min(config.top_m, top_count)
    top_probs = target_probs.gather(1, top_ids[:, :entropy_count])
    tail = (1.0 - top_probs.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    buckets = torch.cat((top_probs, tail), dim=-1)
    compact_entropy = -(
        buckets * torch.log(buckets.clamp_min(1e-30))
    ).sum(dim=-1) / math.log(entropy_count + 1.0)

    if config.variant in {
        "scheme2_relaxed",
        "scheme12_joint",
        "scheme12_anchored",
        "remtp",
        "remtp_block",
    }:
        # A target-head mass gather is cheaper than a second full-vocabulary
        # top-k over Q.  It asks whether the MTP distribution assigns material
        # mass to the target model's current head, while retaining full-Q
        # information rather than looking only at q(y).
        p_top_ids = top_ids[:, : min(config.top_m, top_count)]
        target_head_mass = draft_probs.gather(1, p_top_ids).sum(dim=-1)
        overlap = target_head_mass >= config.sentinel_head_mass_floor
    elif config.variant in {"scheme2", "scheme12"}:
        q_top_ids = torch.topk(
            draft_probs,
            k=min(config.top_m, draft_probs.shape[-1]),
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        p_top_ids = top_ids[:, : min(config.top_m, top_count)]
        overlap = (
            p_top_ids.unsqueeze(2) == q_top_ids.unsqueeze(1)
        ).any(dim=2).any(dim=1)
        target_head_mass = draft_probs.gather(1, p_top_ids).sum(dim=-1)
    else:
        # Scheme 1 has no distribution-overlap sentinel, so it avoids the
        # additional Q top-k vocabulary reduction entirely.
        overlap = torch.ones(
            target_logits.shape[0],
            device=target_logits.device,
            dtype=torch.bool,
        )
        target_head_mass = torch.ones(
            target_logits.shape[0],
            device=target_logits.device,
            dtype=torch.float32,
        )
    return (
        rank,
        gap,
        margin,
        compact_entropy.clamp(0.0, 1.0),
        overlap,
        target_head_mass.to(torch.float32),
    )


def _joint_tv_allocation(
    *,
    desired_tv: torch.Tensor,
    eligible: torch.Tensor,
    strict_acceptance: torch.Tensor,
    draft_candidate_probs: torch.Tensor,
    local_risk: torch.Tensor,
    sentinel_strength: torch.Tensor,
    support_credit: torch.Tensor,
    tv_budget: torch.Tensor,
    risk_mass_cap: torch.Tensor | float,
    config: RiskEntropyConfig,
    prefix_weight: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate one block budget by expected prefix gain per unit TV.

    For stochastic verification, one unit of candidate probability increases
    acceptance by ``1/q(y)`` until saturation.  That marginal gain is weighted
    by the probability of reaching the position, the number of suffix tokens
    unlocked, Scheme 1 target risk, and Scheme 2 future support.  A short
    capped water-filling loop then distributes the exact TV budget without a
    host synchronization or a vocabulary-sized reduction.
    """

    rows_count = desired_tv.shape[0]
    device = desired_tv.device
    dtype = desired_tv.dtype
    if rows_count == 1:
        reach = torch.ones_like(strict_acceptance)
    else:
        reach = torch.cat(
            (
                torch.ones(1, device=device, dtype=dtype),
                torch.cumprod(
                    strict_acceptance[:-1].clamp(0.0, 1.0), dim=0
                ),
            )
        )
    depth = torch.arange(rows_count, device=device, dtype=dtype)
    remaining_fraction = (rows_count - depth) / float(rows_count)
    resolved_prefix_weight = (
        config.joint_prefix_weight
        if prefix_weight is None
        else prefix_weight
    )
    prefix_value = 1.0 + resolved_prefix_weight * remaining_fraction
    future_credit = config.sentinel_soft_floor + (
        1.0 - config.sentinel_soft_floor
    ) * sentinel_strength
    risk_credit = torch.exp(
        -local_risk / max(config.joint_risk_temperature, 1e-12)
    )
    utility = (
        reach
        * prefix_value
        * future_credit
        * risk_credit
        * support_credit
        / draft_candidate_probs.clamp_min(1e-6)
    )
    utility = torch.where(eligible, utility, torch.zeros_like(utility))

    allocation = torch.zeros_like(desired_tv)
    remaining = tv_budget.clamp_min(0.0)
    # K is six in the intended configuration. Fixed-size tensor iterations
    # avoid CPU decisions while redistributing unused saturated shares.
    for _ in range(rows_count):
        capacity = (desired_tv - allocation).clamp_min(0.0)
        active_weight = torch.where(
            capacity > 1e-12, utility, torch.zeros_like(utility)
        )
        weight_sum = active_weight.sum()
        share = remaining * active_weight / weight_sum.clamp_min(1e-30)
        addition = torch.minimum(capacity, share)
        addition = torch.where(
            weight_sum > 0.0, addition, torch.zeros_like(addition)
        )
        allocation = allocation + addition
        remaining = (remaining - addition.sum()).clamp_min(0.0)

    risk_mass = (allocation * local_risk).sum()
    risk_scale = torch.minimum(
        torch.ones((), device=device, dtype=dtype),
        torch.as_tensor(risk_mass_cap, device=device, dtype=dtype)
        / risk_mass.clamp_min(1e-30),
    )
    allocation = allocation * risk_scale
    allocated_risk_mass = allocation * local_risk
    return allocation, utility, allocated_risk_mass


def risk_entropy_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: RiskEntropyConfig,
    *,
    previous_debt: torch.Tensor | float = 0.0,
    stochastic: bool = True,
) -> RiskEntropyResult:
    """Construct ``h`` using only batched GPU reductions and a six-step scan."""

    config.validate()
    if target_probs.ndim != 2 or draft_probs.ndim != 2:
        raise ValueError("target and draft probabilities must be two-dimensional")
    if target_probs.shape != draft_probs.shape:
        raise ValueError("target and draft probability layouts must match")
    if target_logits.shape != target_probs.shape:
        raise ValueError("target logits and probabilities must have equal shape")
    if draft_token_ids.ndim != 1 or draft_token_ids.shape[0] != target_probs.shape[0]:
        raise ValueError("one draft token is required per probability row")

    rows_count = target_probs.shape[0]
    if rows_count == 0:
        raise ValueError("at least one draft row is required")
    device = target_probs.device
    ids = draft_token_ids.to(device=device, dtype=torch.int64)
    rows = torch.arange(rows_count, device=device)
    p_y = target_probs.to(torch.float32)[rows, ids].clamp(0.0, 1.0)
    q_y = draft_probs.to(torch.float32)[rows, ids].clamp(0.0, 1.0)
    strict = torch.minimum(torch.ones_like(p_y), p_y / q_y.clamp_min(1e-30))
    debt = torch.as_tensor(previous_debt, device=device, dtype=torch.float32).reshape(())

    rank, gap, margin, entropy, overlap, target_head_mass = _top_statistics(
        target_probs.to(torch.float32),
        draft_probs.to(torch.float32),
        target_logits,
        ids,
        config,
    )
    depth = rows.to(torch.float32)
    depth_denominator = float(max(rows_count - 1, 1))
    normalized_depth = depth / depth_denominator
    ambiguity = 0.5 * (entropy + torch.exp(-margin).clamp(0.0, 1.0))
    if config.variant == "scheme2_relaxed":
        # Scheme 2 is intentionally independent from Scheme 1 here: current
        # target support is a fixed band and only cross-round debt tightens it.
        threshold = torch.full_like(gap, config.base_log_gap) / (
            1.0 + config.debt_scale * debt
        )
    elif config.variant in {"remtp", "remtp_block"}:
        # The target top-1/top-2 margin calibrates how much of the configured
        # gap ceiling may be used. A decisive target approaches the protected
        # floor; an ambiguous target approaches the relaxed ceiling.
        ambiguity_weight = torch.exp(
            -margin / config.remtp_margin_temperature
        )
        threshold = (
            config.remtp_gap_floor
            + (config.base_log_gap - config.remtp_gap_floor)
            * ambiguity_weight
        ) / (1.0 + config.debt_scale * debt)
    else:
        threshold = (
            config.base_log_gap
            * (1.0 + config.ambiguity_scale * ambiguity)
            / (1.0 + config.threshold_position_scale * normalized_depth)
            / (1.0 + config.debt_scale * debt)
        )

    confidence_shortfall = (
        (config.mtp_confidence_floor - q_y).clamp_min(0.0)
        / max(config.mtp_confidence_floor, 1e-12)
    )
    local_risk = (
        gap
        * (1.0 + config.risk_position_scale * (depth / rows_count))
        * (1.0 + config.risk_confidence_scale * confidence_shortfall)
    )
    needs_relaxation = strict < (1.0 - 1e-7)

    if config.variant in {
        "scheme2",
        "scheme2_relaxed",
        "remtp",
        "remtp_block",
    }:
        rank_limit = torch.full_like(
            rank,
            min(config.scheme2_rank_limit, config.max_target_rank),
        )
        local_eligible = (
            (rank <= rank_limit)
            & (gap <= threshold)
        )
    else:
        rank_limit = torch.floor(
            config.max_target_rank
            / (1.0 + config.rank_position_scale * normalized_depth)
        ).to(torch.int64).clamp_min(1)
        local_eligible = (rank <= rank_limit) & (gap <= threshold)

    sentinel_pass = torch.ones(rows_count, device=device, dtype=torch.bool)
    sentinel_checks = torch.full(
        (rows_count,), 3, device=device, dtype=torch.int32
    )
    sentinel_strength = torch.ones(
        rows_count, device=device, dtype=torch.float32
    )
    if config.variant in _SENTINEL_VARIANTS:
        if rows_count > 1:
            next_gap_ok = gap[1:] <= (
                config.sentinel_log_gap / (1.0 + config.debt_scale * debt)
            )
            next_overlap_ok = overlap[1:]
            next_q_ok = q_y[1:] >= config.sentinel_mtp_floor
            checks = (
                next_gap_ok.to(torch.int32)
                + next_overlap_ok.to(torch.int32)
                + next_q_ok.to(torch.int32)
            )
            sentinel_checks[:-1] = checks
            sentinel_strength[:-1] = checks.to(torch.float32) / 3.0
            if config.variant in {"remtp", "remtp_block"}:
                risky = (
                    (rank[:-1] > config.remtp_risky_rank)
                    | (gap[:-1] > config.remtp_risky_log_gap)
                )
                required_checks = torch.where(
                    risky,
                    torch.full_like(checks, config.remtp_risky_min_checks),
                    torch.full_like(checks, config.sentinel_min_checks),
                )
                sentinel_pass[:-1] = checks >= required_checks
            else:
                sentinel_pass[:-1] = checks >= config.sentinel_min_checks
        sentinel_pass[-1] = gap[-1] <= (
            threshold[-1] * config.last_token_gap_scale
        )
        sentinel_checks[-1] = torch.where(
            sentinel_pass[-1],
            torch.full_like(sentinel_checks[-1], 3),
            torch.zeros_like(sentinel_checks[-1]),
        )
        sentinel_strength[-1] = sentinel_pass[-1].to(torch.float32)

    effective_risk_budget = config.risk_budget
    effective_tv_budget = config.block_tv_cap
    if config.variant in _DEBT_VARIANTS:
        debt_factor = 1.0 + config.debt_scale * debt
        effective_risk_budget = config.risk_budget / debt_factor
        effective_tv_budget = config.block_tv_cap / debt_factor

    cactus_tv = torch.sqrt(
        (
            2.0
            * config.cactus_delta
            * p_y
            * (1.0 - p_y)
        ).clamp_min(0.0)
    )
    if stochastic:
        useful_capacity = (q_y - p_y).clamp_min(0.0)
    else:
        useful_capacity = (1.0 - p_y).clamp_min(0.0)
    desired_tv = torch.minimum(
        torch.minimum(cactus_tv, useful_capacity),
        torch.full_like(cactus_tv, config.per_token_tv_cap),
    )

    # Target-anchored prefix evidence.  For token i, the target forward has
    # already evaluated the draft-conditioned positions i+1..i+horizon.  A
    # high product of strict acceptance probabilities says that the proposed
    # continuation is stable, while the discounted target gaps prevent an
    # MTP-only agreement from certifying a target-opposed trajectory.
    continuation_support = torch.ones_like(strict)
    continuation_regret = torch.zeros_like(gap)
    has_future = torch.zeros(rows_count, device=device, dtype=torch.bool)
    # Three offset-wise vector operations are cheaper than a per-position
    # vocabulary statistic or a host-side scan, and CUDA can capture them.
    for offset in range(
        1, min(config.remtp_continuation_horizon, rows_count - 1) + 1
    ):
        continuation_support[:-offset] *= strict[offset:]
        continuation_regret[:-offset] += (0.5 ** (offset - 1)) * gap[offset:]
        has_future[:-offset] = True
    continuation_support = torch.where(
        has_future, continuation_support, torch.zeros_like(strict)
    )
    continuation_regret = torch.where(
        has_future,
        continuation_regret,
        torch.full_like(gap, 1e6),
    )
    base_safe = local_eligible & sentinel_pass
    positive_certificate = torch.zeros_like(base_safe)
    if config.variant == "remtp_block":
        positive_certificate = (
            ~base_safe
            & needs_relaxation
            & (rank <= config.max_target_rank)
            & (gap <= config.remtp_continuation_gap)
            & (continuation_support >= config.remtp_continuation_support)
            & (continuation_regret <= config.remtp_continuation_regret)
        )

    if config.variant in {"scheme12_joint", "scheme12_anchored"}:
        joint_eligible = local_eligible & sentinel_pass & needs_relaxation
        support_credit = torch.exp(
            -config.joint_support_power * gap
        )
        if config.variant == "scheme12_anchored":
            core_eligible = (
                joint_eligible
                & (rank <= config.joint_core_rank)
                & (gap <= config.joint_core_log_gap)
            )
            core_budget = torch.as_tensor(
                effective_tv_budget, device=device, dtype=torch.float32
            ) * config.joint_core_fraction
            (
                core_allocated_tv,
                core_utility,
                core_risk_mass,
            ) = _joint_tv_allocation(
                desired_tv=desired_tv,
                eligible=core_eligible,
                strict_acceptance=strict,
                draft_candidate_probs=q_y,
                local_risk=local_risk,
                sentinel_strength=sentinel_strength,
                support_credit=support_credit,
                tv_budget=core_budget,
                risk_mass_cap=config.joint_risk_mass_cap,
                config=config,
            )
            explore_eligible = (
                joint_eligible
                & ~core_eligible
                & (
                    sentinel_checks
                    >= config.joint_explore_min_checks
                )
            )
            explore_budget = torch.as_tensor(
                effective_tv_budget, device=device, dtype=torch.float32
            ) * (1.0 - config.joint_core_fraction)
            remaining_risk_cap = (
                torch.as_tensor(
                    config.joint_risk_mass_cap,
                    device=device,
                    dtype=torch.float32,
                )
                - core_risk_mass.sum()
            ).clamp_min(0.0)
            (
                explore_tv,
                explore_utility,
                explore_risk_mass,
            ) = _joint_tv_allocation(
                desired_tv=desired_tv,
                eligible=explore_eligible,
                strict_acceptance=strict,
                draft_candidate_probs=q_y,
                local_risk=(
                    local_risk
                    * config.joint_explore_risk_multiplier
                ),
                sentinel_strength=sentinel_strength,
                support_credit=support_credit,
                tv_budget=explore_budget,
                risk_mass_cap=remaining_risk_cap,
                config=config,
            )
            allocated_tv = core_allocated_tv + explore_tv
            allocation_utility = core_utility + explore_utility
            allocated_risk_mass = core_risk_mass + explore_risk_mass
        else:
            allocated_tv, allocation_utility, allocated_risk_mass = (
                _joint_tv_allocation(
                    desired_tv=desired_tv,
                    eligible=joint_eligible,
                    strict_acceptance=strict,
                    draft_candidate_probs=q_y,
                    local_risk=local_risk,
                    sentinel_strength=sentinel_strength,
                    support_credit=torch.ones_like(gap),
                    tv_budget=torch.as_tensor(
                        effective_tv_budget,
                        device=device,
                        dtype=torch.float32,
                    ),
                    risk_mass_cap=config.joint_risk_mass_cap,
                    config=config,
                )
            )
            core_allocated_tv = torch.zeros_like(allocated_tv)
        cumulative_risk = torch.cumsum(allocated_risk_mass, dim=0)
        # Unlike the old prefix scan, an unsafe position does not erase useful
        # relaxation at later positions: if strict sampling accepts it, those
        # later distributions are reachable and their budget remains useful.
        relaxed_prefix = local_eligible & sentinel_pass
    elif config.variant in {"remtp", "remtp_block"}:
        # Treat the TV budget as a reusable block resource. An unsafe earlier
        # token receives no relaxation, but later safe heads can still use TV
        # if the standard strict sampler accepts that earlier token. Capped
        # water-filling also reclaims probability mass that saturated heads
        # cannot use.
        remtp_eligible = (
            (base_safe | positive_certificate) & needs_relaxation
        )
        support_credit = torch.exp(
            -gap / config.remtp_support_temperature
        )
        allocated_tv, allocation_utility, allocated_risk_mass = (
            _joint_tv_allocation(
                desired_tv=desired_tv,
                eligible=remtp_eligible,
                strict_acceptance=strict,
                draft_candidate_probs=q_y,
                local_risk=local_risk,
                sentinel_strength=sentinel_strength,
                support_credit=support_credit,
                tv_budget=torch.as_tensor(
                    effective_tv_budget,
                    device=device,
                    dtype=torch.float32,
                ),
                risk_mass_cap=effective_risk_budget,
                config=config,
                prefix_weight=config.remtp_prefix_weight,
            )
        )
        cumulative_risk = torch.cumsum(allocated_risk_mass, dim=0)
        core_allocated_tv = torch.zeros_like(allocated_tv)
        relaxed_prefix = base_safe | positive_certificate
    else:
        # Legacy profiles retain their sequential scan for reproducibility.
        running_risk = torch.zeros((), device=device, dtype=torch.float32)
        running_tv = torch.zeros((), device=device, dtype=torch.float32)
        active = torch.ones((), device=device, dtype=torch.bool)
        prefix_parts: list[torch.Tensor] = []
        risk_parts: list[torch.Tensor] = []
        tv_parts: list[torch.Tensor] = []
        for index in range(rows_count):
            strict_safe = ~needs_relaxation[index]
            if config.variant == "scheme2_relaxed":
                # Cross-round debt and the exact block-TV cap already control
                # total deviation; do not duplicate Scheme 1's risk limit.
                risk_fits = torch.ones((), device=device, dtype=torch.bool)
            else:
                risk_fits = (
                    running_risk + local_risk[index]
                    <= effective_risk_budget
                )
            eligible = (
                local_eligible[index]
                & sentinel_pass[index]
                & risk_fits
            )
            allow = active & (strict_safe | eligible)
            prefix_parts.append(allow)

            add_risk = torch.where(
                allow & needs_relaxation[index],
                local_risk[index],
                torch.zeros_like(local_risk[index]),
            )
            running_risk = running_risk + add_risk
            risk_parts.append(running_risk)

            remaining_tv = (effective_tv_budget - running_tv).clamp_min(0.0)
            allocation = torch.minimum(
                desired_tv[index],
                remaining_tv,
            )
            if config.variant == "scheme2_relaxed":
                soft_weight = config.sentinel_soft_floor + (
                    1.0 - config.sentinel_soft_floor
                ) * sentinel_strength[index]
                allocation = allocation * soft_weight
            allocation = torch.where(
                allow & needs_relaxation[index],
                allocation,
                torch.zeros_like(allocation),
            )
            tv_parts.append(allocation)
            running_tv = running_tv + allocation
            active = allow

        relaxed_prefix = torch.stack(prefix_parts)
        cumulative_risk = torch.stack(risk_parts)
        allocated_tv = torch.stack(tv_parts)
        allocated_risk_mass = allocated_tv * local_risk
        allocation_utility = torch.zeros_like(allocated_tv)
        core_allocated_tv = torch.zeros_like(allocated_tv)
    # Never lower an already saturated target probability.  Clamping to
    # 1-1e-6 made h(y)<p(y) when FP32 softmax rounded p(y) to exactly one.
    boosted = (p_y + allocated_tv).clamp(max=1.0)

    scale = ((1.0 - boosted) / (1.0 - p_y)).nan_to_num(
        nan=1.0, posinf=1.0, neginf=0.0
    )
    verification_probs = target_probs.to(torch.float32) * scale.unsqueeze(1)
    verification_probs.scatter_(1, ids.unsqueeze(1), boosted.unsqueeze(1))
    verification_probs = verification_probs.clamp_min(0.0)
    verification_probs = verification_probs / verification_probs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-30)
    relaxed_acceptance = torch.minimum(
        torch.ones_like(boosted), boosted / q_y.clamp_min(1e-30)
    )
    return RiskEntropyResult(
        probs=verification_probs.contiguous(),
        target_candidate_probs=p_y,
        draft_candidate_probs=q_y,
        boosted_candidate_probs=boosted,
        strict_acceptance=strict,
        relaxed_acceptance=relaxed_acceptance,
        allocated_tv=allocated_tv,
        local_risk=local_risk,
        cumulative_risk=cumulative_risk,
        candidate_rank=rank,
        candidate_log_gap=gap,
        target_margin=margin,
        compact_entropy=entropy,
        threshold=threshold,
        local_eligible=local_eligible,
        sentinel_pass=sentinel_pass,
        sentinel_checks=sentinel_checks,
        sentinel_strength=sentinel_strength,
        allocation_utility=allocation_utility,
        allocated_risk_mass=allocated_risk_mass,
        core_allocated_tv=core_allocated_tv,
        relaxed_prefix=relaxed_prefix,
        desired_tv=desired_tv,
        target_head_mass=target_head_mass,
        continuation_support=continuation_support,
        continuation_regret=continuation_regret,
        positive_certificate=positive_certificate,
    )


def _update_debt(
    result: RiskEntropyResult,
    output_token_ids: torch.Tensor,
    config: RiskEntropyConfig,
) -> None:
    if config.variant not in _DEBT_VARIANTS:
        return
    module = _VLLM_REJECTION
    if module is None:
        return
    valid_count = (output_token_ids[0] != module.PLACEHOLDER_TOKEN_ID).sum()
    accepted_drafts = (valid_count - 1).clamp(min=0, max=result.local_risk.shape[0])
    depths = torch.arange(result.local_risk.shape[0], device=output_token_ids.device)
    committed = depths < accepted_drafts
    responsibility = (
        result.relaxed_acceptance - result.strict_acceptance
    ).clamp_min(0.0)
    new_debt = (
        result.local_risk
        * responsibility
        * committed.to(result.local_risk.dtype)
    ).sum()
    old = _STATE.debt
    if old is None:
        old = torch.zeros_like(new_debt)
    _STATE.debt = (
        old.to(device=new_debt.device, dtype=torch.float32) * config.debt_decay
        + new_debt
    ).detach()
    if config.adaptive_draft_depth:
        debt_value = float(_STATE.debt.item())
        if debt_value >= config.high_debt_threshold:
            _STATE.next_draft_depth = config.short_draft_depth
        elif debt_value >= config.medium_debt_threshold:
            _STATE.next_draft_depth = config.medium_draft_depth
        else:
            _STATE.next_draft_depth = config.expected_draft_tokens


def _risk_get_draft_token_ids_cpu(
    self: Any,
) -> tuple[list[list[int]], list[str]]:
    """Expose a debt-controlled draft prefix to the non-async scheduler."""

    original = getattr(_risk_get_draft_token_ids_cpu, "_remtp_original")
    draft_rows, request_ids = original(self)
    config = _CONFIG
    if (
        config is None
        or config.variant not in _DEBT_VARIANTS
        or not config.adaptive_draft_depth
        or not draft_rows
        or not request_ids
    ):
        return draft_rows, request_ids
    if len(draft_rows) != 1 or len(request_ids) != 1:
        raise RuntimeError("debt-controlled draft depth requires --max-num-seqs 1")
    request_id = request_ids[0]
    if request_id != _STATE.request_id:
        _STATE.request_id = request_id
        _STATE.next_draft_depth = config.expected_draft_tokens
    depth = max(
        1,
        min(_STATE.next_draft_depth, config.expected_draft_tokens),
    )
    return [list(draft_rows[0][:depth])], request_ids


def _risk_entropy_rejection_sample(
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

    original = getattr(_risk_entropy_rejection_sample, "_remtp_original")
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
        raise RuntimeError("risk-entropy verification requires --max-num-seqs 1")
    if _CONFIG is None or _VLLM_REJECTION is None:
        raise RuntimeError("risk-entropy verification is not installed")

    config = _CONFIG
    module = _VLLM_REJECTION
    previous_debt = _begin_round(sampling_metadata, target_logits.device)
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    result = risk_entropy_distribution(
        target_probs,
        draft_probs,
        target_logits,
        draft_token_ids,
        config,
        previous_debt=previous_debt,
        stochastic=not sampling_metadata.all_greedy,
    )

    # Use vLLM's unchanged GPU kernels directly with h. This avoids the extra
    # log(h)->softmax(h) vocabulary pass incurred by a generic logits adapter.
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    output = torch.full(
        (batch_size, max_spec_len + 1),
        module.PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == module.GREEDY_TEMPERATURE
    uniform_probs = None
    if not sampling_metadata.all_random:
        target_argmax = result.probs.argmax(dim=-1)
        module.rejection_greedy_sample_kernel[(batch_size,)](
            output,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
    if not sampling_metadata.all_greedy:
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
            output,
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
    _update_debt(result, output, config)

    if config.audit_path:
        from remtp.block_oracle import append_audit_round

        append_audit_round(
            path=config.audit_path,
            dataset=config.audit_dataset,
            result=result,
            target_probs=target_probs,
            draft_probs=draft_probs,
            target_logits=target_logits,
            draft_token_ids=draft_token_ids,
            uniform_probs=uniform_probs,
            output_token_ids=output,
            sampling_metadata=sampling_metadata,
            request_id=_STATE.request_id,
            output_length_before_round=_STATE.active_output_length,
            previous_debt=previous_debt,
            config=config,
            placeholder_token_id=module.PLACEHOLDER_TOKEN_ID,
        )

    if config.diagnostics and not _DIAGNOSTIC_EMITTED:
        debt_value = _STATE.debt
        print(
            "[ReMTP][RiskEntropy][diagnostic] "
            f"variant={config.variant} "
            f"p={result.target_candidate_probs.tolist()} "
            f"q={result.draft_candidate_probs.tolist()} "
            f"h={result.boosted_candidate_probs.tolist()} "
            f"rank={result.candidate_rank.tolist()} "
            f"gap={result.candidate_log_gap.tolist()} "
            f"margin={result.target_margin.tolist()} "
            f"entropy={result.compact_entropy.tolist()} "
            f"risk={result.local_risk.tolist()} "
            f"cum_risk={result.cumulative_risk.tolist()} "
            f"sentinel={result.sentinel_pass.tolist()} "
            f"sentinel_checks={result.sentinel_checks.tolist()} "
            f"sentinel_strength={result.sentinel_strength.tolist()} "
            f"eligible={result.local_eligible.tolist()} "
            f"relaxed_prefix={result.relaxed_prefix.tolist()} "
            f"strict_A={result.strict_acceptance.tolist()} "
            f"relaxed_A={result.relaxed_acceptance.tolist()} "
            f"tv={result.allocated_tv.tolist()} "
            f"core_tv={result.core_allocated_tv.tolist()} "
            f"utility={result.allocation_utility.tolist()} "
            f"risk_mass={result.allocated_risk_mass.tolist()} "
            f"next_debt={None if debt_value is None else debt_value.item():}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True
    return output


def install_risk_entropy_mtp() -> None:
    """Install one of the three risk/sentinel verification variants."""

    global _CONFIG, _VLLM_REJECTION

    config = RiskEntropyConfig.from_env()
    module = importlib.import_module(_REJECTION_MODULE)
    _CONFIG = config
    _VLLM_REJECTION = module
    _STATE.reset()
    current = module.rejection_sample
    if not getattr(current, "_remtp_risk_entropy", False):
        _risk_entropy_rejection_sample._remtp_risk_entropy = True
        _risk_entropy_rejection_sample._remtp_original = current
        module.rejection_sample = _risk_entropy_rejection_sample

    # Always install request tracking for stateful variants and for audit
    # collection.  This fixes request boundaries under async scheduling while
    # adding only two Python metadata reads per speculative round.
    if config.variant in _DEBT_VARIANTS or config.audit_path:
        runner_module = importlib.import_module(_GPU_RUNNER_MODULE)
        runner_cls = runner_module.GPUModelRunner
        current_sample = runner_cls._sample
        if not getattr(current_sample, "_remtp_risk_request_tracking", False):
            _risk_runner_sample._remtp_risk_request_tracking = True
            _risk_runner_sample._remtp_original = current_sample
            runner_cls._sample = _risk_runner_sample

    if config.variant in _DEBT_VARIANTS and config.adaptive_draft_depth:
        runner_module = importlib.import_module(_GPU_RUNNER_MODULE)
        runner_cls = runner_module.GPUModelRunner
        current_get = runner_cls._get_draft_token_ids_cpu
        if not getattr(current_get, "_remtp_risk_entropy_depth", False):
            _risk_get_draft_token_ids_cpu._remtp_risk_entropy_depth = True
            _risk_get_draft_token_ids_cpu._remtp_original = current_get
            runner_cls._get_draft_token_ids_cpu = _risk_get_draft_token_ids_cpu

    print(
        "[ReMTP][RiskEntropy] "
        f"variant={config.variant} max_rank={config.max_target_rank} "
        f"base_gap={config.base_log_gap:g} risk_budget={config.risk_budget:g} "
        f"per_token_TV={config.per_token_tv_cap:g} "
        f"block_TV={config.block_tv_cap:g} "
        f"sentinel_checks={config.sentinel_min_checks}/3 "
        f"debt_decay={config.debt_decay:g} "
        f"adaptive_depth={int(config.adaptive_draft_depth)}",
        flush=True,
    )
