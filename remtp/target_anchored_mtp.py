"""Target-anchored block relaxation for native probabilistic MTP.

The method removes vocabulary-wide P/Q agreement signals.  It spends the
exact candidate-probability mass (and therefore exact total-variation mass)
that per-position Cactus would request, but stops boosting a candidate once
``h(y) == q(y)`` because its speculative acceptance is already saturated.

Three ablations are implemented:

* ``cactus_cap``: independent Cactus boosts capped at ``q(y)``;
* ``tv_head``: redistribute Cactus' block TV using target support and a
  calibrated reliability prior for each MTP head;
* ``tv_hidden_veto``: additionally use aligned current MTP/target hidden
  cosine and allow later target support only to veto, never reward, an
  earlier candidate.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


_V1_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DIAGNOSTIC_EMITTED = False
_HIDDEN_FALLBACK_EMITTED = False
_AUDIT_ROUND = 0
_AUDIT_TV = torch.zeros(3, dtype=torch.float64)
_DEFAULT_HEAD_RELIABILITY = (1.0, 0.85, 0.70, 0.55, 0.40, 0.30)
_POST_VERIFICATION_HOOK: Any | None = None


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


def _parse_head_reliability(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(","))
    if not result or any(not 0.0 <= item <= 1.0 for item in result):
        raise ValueError(
            "REMTP_TA_HEAD_RELIABILITY must be comma-separated values in [0,1]"
        )
    return result


@dataclass(frozen=True)
class TargetAnchoredConfig:
    """Configuration for target-anchored exact-TV redistribution."""

    variant: str = "tv_hidden_veto"
    cactus_delta: float = 1.0
    expected_draft_tokens: int = 6
    head_reliability: tuple[float, ...] = _DEFAULT_HEAD_RELIABILITY
    target_log_gap_scale: float = 2.0
    max_target_log_gap: float = 8.0
    future_veto_floor: float = 0.20
    hidden_reliability_floor: float = 0.25
    use_prefix_value: bool = True

    def validate(self) -> None:
        if self.variant not in {
            "cactus_cap",
            "tv_head",
            "tv_hidden_veto",
        }:
            raise ValueError(f"unsupported target-anchored variant: {self.variant}")
        if self.cactus_delta < 0:
            raise ValueError("cactus_delta must be non-negative")
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if len(self.head_reliability) < self.expected_draft_tokens:
            raise ValueError(
                "head_reliability must cover every expected MTP position"
            )
        if any(not 0.0 <= item <= 1.0 for item in self.head_reliability):
            raise ValueError("head reliability values must be in [0,1]")
        if self.target_log_gap_scale <= 0:
            raise ValueError("target_log_gap_scale must be positive")
        if self.max_target_log_gap <= 0:
            raise ValueError("max_target_log_gap must be positive")
        if not 0.0 <= self.future_veto_floor <= 1.0:
            raise ValueError("future_veto_floor must be in [0,1]")
        if not 0.0 <= self.hidden_reliability_floor <= 1.0:
            raise ValueError("hidden_reliability_floor must be in [0,1]")

    @classmethod
    def from_env(cls) -> TargetAnchoredConfig:
        expected = int(os.getenv("REMTP_TA_EXPECTED_DRAFT_TOKENS", "6"))
        default_heads = ",".join(
            str(value)
            for value in _DEFAULT_HEAD_RELIABILITY[:expected]
        )
        config = cls(
            variant=os.getenv(
                "REMTP_TA_VARIANT",
                "tv_hidden_veto",
            ),
            cactus_delta=float(os.getenv("REMTP_CACTUS_DELTA", "1.0")),
            expected_draft_tokens=expected,
            head_reliability=_parse_head_reliability(
                os.getenv("REMTP_TA_HEAD_RELIABILITY", default_heads)
            ),
            target_log_gap_scale=float(
                os.getenv("REMTP_TA_TARGET_LOG_GAP_SCALE", "2.0")
            ),
            max_target_log_gap=float(
                os.getenv("REMTP_TA_MAX_TARGET_LOG_GAP", "8.0")
            ),
            future_veto_floor=float(
                os.getenv("REMTP_TA_FUTURE_VETO_FLOOR", "0.20")
            ),
            hidden_reliability_floor=float(
                os.getenv("REMTP_TA_HIDDEN_RELIABILITY_FLOOR", "0.25")
            ),
            use_prefix_value=_env_flag(
                "REMTP_TA_USE_PREFIX_VALUE",
                True,
            ),
        )
        config.validate()
        return config


@dataclass
class TargetAnchoredResult:
    """Scalar decisions and optional materialized verifier distributions."""

    probs: torch.Tensor | None
    target_candidate_probs: torch.Tensor
    draft_candidate_probs: torch.Tensor
    boosted_candidate_probs: torch.Tensor
    target_log_gaps: torch.Tensor
    target_support: torch.Tensor
    head_reliability: torch.Tensor
    hidden_similarity: torch.Tensor
    future_veto: torch.Tensor
    strict_acceptance: torch.Tensor
    prefix_value: torch.Tensor
    priority: torch.Tensor
    cactus_tv: torch.Tensor
    useful_tv_capacity: torch.Tensor
    allocated_tv: torch.Tensor


def cactus_tv_increment(
    candidate_probs: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Return Cactus' exact TV shift for each sampled candidate."""
    bonus = torch.sqrt(
        (
            2.0
            * delta
            * candidate_probs
            * (1.0 - candidate_probs)
        ).clamp_min(0.0)
    )
    return torch.minimum(
        bonus,
        (1.0 - candidate_probs).clamp_min(0.0),
    )


def aligned_hidden_cosine(
    draft_hidden: torch.Tensor | None,
    target_hidden: torch.Tensor | None,
    rows: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    """Return non-negative current-position cosine or a neutral fallback."""
    if (
        draft_hidden is None
        or target_hidden is None
        or draft_hidden.ndim != 2
        or target_hidden.ndim != 2
        or draft_hidden.shape != target_hidden.shape
        or draft_hidden.shape[0] != rows
    ):
        return torch.ones(rows, device=device, dtype=torch.float32), False
    similarity = F.cosine_similarity(
        draft_hidden.to(torch.float32),
        target_hidden.to(torch.float32),
        dim=-1,
        eps=1e-8,
    )
    return similarity.clamp(0.0, 1.0), True


def _prefix_marginal_values(
    strict_acceptance: torch.Tensor,
) -> torch.Tensor:
    rows = strict_acceptance.shape[0]
    reach = torch.ones_like(strict_acceptance)
    for depth in range(1, rows):
        reach[depth] = reach[depth - 1] * strict_acceptance[depth - 1]
    suffix = torch.ones_like(strict_acceptance)
    for depth in range(rows - 2, -1, -1):
        suffix[depth] = 1.0 + strict_acceptance[depth + 1] * suffix[depth + 1]
    return reach * suffix


def _future_target_veto(
    local_support: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    """Penalize an earlier token if any later target support is poor."""
    rows = local_support.shape[0]
    result = torch.ones_like(local_support)
    for depth in range(rows - 1):
        later_support = local_support[depth + 1 :].amin()
        result[depth] = floor + (1.0 - floor) * later_support
    return result


def _allocate_capped_tv(
    capacities: torch.Tensor,
    priorities: torch.Tensor,
    total_budget: torch.Tensor,
) -> torch.Tensor:
    """Exact four-pass capped water filling over one MTP block."""
    allocated = torch.zeros_like(capacities)
    remaining = total_budget.clamp_min(0.0)
    for _ in range(capacities.shape[0]):
        residual = (capacities - allocated).clamp_min(0.0)
        active = torch.where(
            residual > 1e-12,
            priorities,
            torch.zeros_like(priorities),
        )
        proposed = (
            remaining
            * active
            / active.sum().clamp_min(1e-30)
        )
        step = torch.minimum(proposed, residual)
        allocated = allocated + step
        remaining = (remaining - step.sum()).clamp_min(0.0)
    return allocated


def target_anchored_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    config: TargetAnchoredConfig,
    *,
    hidden_similarity: torch.Tensor | None = None,
    assume_normalized: bool = False,
    construct_probs: bool = True,
) -> TargetAnchoredResult:
    """Build the selected target-anchored verifier distribution."""
    config.validate()
    if target_probs.ndim != 2 or target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must share [rows,vocab]")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must have shape [rows]")
    row_count = target_probs.shape[0]
    if row_count != draft_token_ids.shape[0]:
        raise ValueError("probability and token rows must match")
    if not 1 <= row_count <= config.expected_draft_tokens:
        raise ValueError("unexpected MTP block length")

    target = target_probs.to(torch.float32)
    draft = draft_probs.to(device=target.device, dtype=torch.float32)
    if not assume_normalized:
        target = target.clamp_min(0.0)
        draft = draft.clamp_min(0.0)
        target /= target.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        draft /= draft.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    ids = draft_token_ids.to(device=target.device, dtype=torch.int64)
    rows = torch.arange(row_count, device=target.device)
    p_y = target[rows, ids]
    q_y = draft[rows, ids]
    top_p = target.amax(dim=-1)
    log_gap = (
        torch.log(top_p.clamp_min(1e-30))
        - torch.log(p_y.clamp_min(1e-30))
    ).clamp_min(0.0)
    local_support = torch.exp(
        -log_gap / config.target_log_gap_scale
    )
    local_support = torch.where(
        log_gap <= config.max_target_log_gap,
        local_support,
        torch.zeros_like(local_support),
    )

    head_prior = torch.as_tensor(
        config.head_reliability[:row_count],
        device=target.device,
        dtype=torch.float32,
    )
    if hidden_similarity is None:
        hidden = torch.ones_like(p_y)
    else:
        if hidden_similarity.shape != p_y.shape:
            raise ValueError("hidden_similarity must have one value per row")
        hidden = hidden_similarity.to(
            device=target.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

    strict_acceptance = torch.minimum(
        torch.ones_like(p_y),
        p_y / q_y.clamp_min(1e-30),
    )
    prefix_value = (
        _prefix_marginal_values(strict_acceptance)
        if config.use_prefix_value
        else torch.ones_like(p_y)
    )
    future_veto = _future_target_veto(
        local_support,
        config.future_veto_floor,
    )
    hidden_reliability = (
        config.hidden_reliability_floor
        + (1.0 - config.hidden_reliability_floor) * hidden
    )
    rescue = strict_acceptance.sqrt() * (1.0 - strict_acceptance)
    priority = rescue * prefix_value * local_support * head_prior
    if config.variant == "tv_hidden_veto":
        priority = priority * hidden_reliability * future_veto

    cactus_tv = cactus_tv_increment(p_y, config.cactus_delta)
    useful_capacity = (q_y - p_y).clamp_min(0.0)
    if config.variant == "cactus_cap":
        allocated = torch.minimum(cactus_tv, useful_capacity)
    else:
        allocated = _allocate_capped_tv(
            useful_capacity,
            priority,
            cactus_tv.sum(),
        )
    boosted = p_y + allocated

    relaxed = None
    if construct_probs:
        scale = (
            (1.0 - boosted) / (1.0 - p_y)
        ).nan_to_num(nan=1.0, posinf=1.0, neginf=0.0)
        relaxed = target * scale.unsqueeze(-1)
        relaxed.scatter_(1, ids.unsqueeze(1), boosted.unsqueeze(1))
        relaxed = relaxed.clamp_min(0.0)
        relaxed /= relaxed.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    return TargetAnchoredResult(
        probs=relaxed,
        target_candidate_probs=p_y,
        draft_candidate_probs=q_y,
        boosted_candidate_probs=boosted,
        target_log_gaps=log_gap,
        target_support=local_support,
        head_reliability=head_prior,
        hidden_similarity=hidden,
        future_veto=future_veto,
        strict_acceptance=strict_acceptance,
        prefix_value=prefix_value,
        priority=priority,
        cactus_tv=cactus_tv,
        useful_tv_capacity=useful_capacity,
        allocated_tv=allocated,
    )


def target_anchored_candidate_probs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    hidden_similarity: torch.Tensor,
    config: TargetAnchoredConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = target_anchored_distribution(
        target_probs,
        draft_probs,
        draft_token_ids,
        config,
        hidden_similarity=hidden_similarity,
        assume_normalized=True,
        construct_probs=False,
    )
    return result.target_candidate_probs, result.boosted_candidate_probs


def boost_candidate_logits(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    original_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
) -> torch.Tensor:
    """Apply candidate TV shifts without materializing full h."""
    logits = target_logits.to(torch.float32).clone()
    ids = draft_token_ids.to(device=logits.device, dtype=torch.int64)
    rows = torch.arange(logits.shape[0], device=logits.device)
    original = original_candidate_probs.clamp(1e-12, 1.0 - 1e-6)
    boosted = boosted_candidate_probs.clamp(1e-12, 1.0 - 1e-6)
    logits[rows, ids] += torch.logit(boosted) - torch.logit(original)
    return logits


def _target_anchored_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    global _AUDIT_ROUND, _AUDIT_TV, _DIAGNOSTIC_EMITTED
    global _HIDDEN_FALLBACK_EMITTED

    original = getattr(_target_anchored_rejection_sample, "_remtp_original")
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
            "target-anchored MTP requires --max-num-seqs 1"
        )

    config = getattr(_target_anchored_rejection_sample, "_remtp_config")
    if max_spec_len != config.expected_draft_tokens:
        raise RuntimeError(
            f"configured MTP={config.expected_draft_tokens}, "
            f"but vLLM max_spec_len={max_spec_len}"
        )

    logits = target_logits.to(torch.float32)
    target_probs = torch.softmax(logits, dim=-1)
    hidden_similarity = torch.ones(
        draft_token_ids.shape[0],
        device=logits.device,
        dtype=torch.float32,
    )
    hidden_available = False
    if config.variant == "tv_hidden_veto":
        from remtp.probabilistic_mtp import get_last_aligned_hidden_states

        draft_hidden, target_hidden = get_last_aligned_hidden_states(
            draft_token_ids.shape[0]
        )
        hidden_similarity, hidden_available = aligned_hidden_cosine(
            draft_hidden,
            target_hidden,
            draft_token_ids.shape[0],
            device=logits.device,
        )
        if not hidden_available and not _HIDDEN_FALLBACK_EMITTED:
            print(
                "[ReMTP][TargetAnchored] aligned hidden unavailable; "
                "using neutral cosine and static head reliability",
                flush=True,
            )
            _HIDDEN_FALLBACK_EMITTED = True

    post_verification_hook = _POST_VERIFICATION_HOOK
    detailed = (
        getattr(_target_anchored_rejection_sample, "_remtp_diagnostics")
        and not _DIAGNOSTIC_EMITTED
    ) or (
        getattr(_target_anchored_rejection_sample, "_remtp_audit_interval") > 0
    )
    result: TargetAnchoredResult | None = None
    if detailed:
        result = target_anchored_distribution(
            target_probs,
            draft_probs,
            draft_token_ids,
            config,
            hidden_similarity=hidden_similarity,
            assume_normalized=True,
            construct_probs=False,
        )
        p_y = result.target_candidate_probs
        h_y = result.boosted_candidate_probs
    else:
        fast = getattr(_target_anchored_rejection_sample, "_remtp_fast")
        p_y, h_y = fast(
            target_probs,
            draft_probs,
            draft_token_ids,
            hidden_similarity,
        )

    diagnostics = getattr(
        _target_anchored_rejection_sample,
        "_remtp_diagnostics",
    )
    if diagnostics and not _DIAGNOSTIC_EMITTED and result is not None:
        print(
            "[ReMTP][TargetAnchored][diagnostic] "
            f"variant={config.variant} hidden={int(hidden_available)} "
            f"p={result.target_candidate_probs.tolist()} "
            f"q={result.draft_candidate_probs.tolist()} "
            f"h={result.boosted_candidate_probs.tolist()} "
            f"log_gap={result.target_log_gaps.tolist()} "
            f"support={result.target_support.tolist()} "
            f"head={result.head_reliability.tolist()} "
            f"hidden_cos={result.hidden_similarity.tolist()} "
            f"future_veto={result.future_veto.tolist()} "
            f"strict_A={result.strict_acceptance.tolist()} "
            f"priority={result.priority.tolist()} "
            f"cactus_TV={result.cactus_tv.tolist()} "
            f"capacity={result.useful_tv_capacity.tolist()} "
            f"allocated_TV={result.allocated_tv.tolist()} "
            f"block_cactus_TV={result.cactus_tv.sum().item():.6f} "
            f"block_allocated_TV={result.allocated_tv.sum().item():.6f}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True

    audit_interval = getattr(
        _target_anchored_rejection_sample,
        "_remtp_audit_interval",
    )
    if audit_interval > 0 and result is not None:
        _AUDIT_ROUND += 1
        _AUDIT_TV += torch.tensor(
            (
                result.cactus_tv.sum().item(),
                result.allocated_tv.sum().item(),
                result.useful_tv_capacity.sum().item(),
            ),
            dtype=torch.float64,
        )
        if _AUDIT_ROUND % audit_interval == 0:
            values = (_AUDIT_TV / audit_interval).tolist()
            print(
                "[ReMTP][TargetAnchored][audit] "
                f"rounds={_AUDIT_ROUND-audit_interval+1}-{_AUDIT_ROUND} "
                f"mean_cactus_TV={values[0]:.6f} "
                f"mean_allocated_TV={values[1]:.6f} "
                f"mean_useful_capacity={values[2]:.6f} "
                f"budget_utilization={values[1]/max(values[0],1e-30):.6f}",
                flush=True,
            )
            _AUDIT_TV.zero_()

    original_top_ids = (
        logits.argmax(dim=-1) if sampling_metadata.all_greedy else None
    )
    relaxed_logits = boost_candidate_logits(
        logits,
        draft_token_ids,
        p_y,
        h_y,
    )
    if sampling_metadata.all_greedy:
        assert original_top_ids is not None
        relaxed_top_ids = relaxed_logits.argmax(dim=-1)
        desired_ids = torch.where(
            draft_token_ids == relaxed_top_ids,
            draft_token_ids,
            original_top_ids,
        )
        verification_logits = torch.full_like(logits, float("-inf"))
        verification_logits.scatter_(1, desired_ids.unsqueeze(1), 0.0)
    else:
        verification_logits = relaxed_logits

    output_token_ids = original(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        verification_logits,
        bonus_token_ids,
        sampling_metadata,
    )
    if post_verification_hook is not None:
        ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
        rows = torch.arange(ids.shape[0], device=target_probs.device)
        q_y = draft_probs[rows, ids]
        post_verification_hook(
            target_probs=target_probs,
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            output_token_ids=output_token_ids,
            target_candidate_probs=p_y,
            boosted_candidate_probs=h_y,
            allocated_tv=(h_y - p_y).clamp_min(0.0),
            strict_acceptance=torch.minimum(
                torch.ones_like(p_y),
                p_y / q_y.clamp_min(1e-30),
            ),
            hidden_similarity=hidden_similarity,
            sampling_metadata=sampling_metadata,
        )
    return output_token_ids


def set_post_verification_hook(hook: Any | None) -> None:
    """Register an optional observer after target-anchored verification.

    The hook receives the exact constrained target distribution, full MTP
    proposal distribution, compiled verifier scalars, and committed output.
    It cannot change the verifier result or force the verifier off its fast
    path.
    """
    global _POST_VERIFICATION_HOOK

    _POST_VERIFICATION_HOOK = hook


def install_target_anchored_mtp() -> None:
    """Install target-anchored exact-TV verification."""
    config = TargetAnchoredConfig.from_env()
    diagnostics = _env_flag("REMTP_TA_DIAGNOSTICS", False)
    audit_interval = int(os.getenv("REMTP_TA_AUDIT_INTERVAL", "0"))
    if audit_interval < 0:
        raise ValueError("REMTP_TA_AUDIT_INTERVAL must be non-negative")
    compile_fast_path = _env_flag("REMTP_TA_COMPILE", True)

    def fast(
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        hidden_similarity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return target_anchored_candidate_probs(
            target_probs,
            draft_probs,
            draft_token_ids,
            hidden_similarity,
            config,
        )

    if compile_fast_path:
        fast = torch.compile(fast, fullgraph=True, dynamic=False)

    module = importlib.import_module(_V1_REJECTION_MODULE)
    current = module.rejection_sample
    if not getattr(current, "_remtp_target_anchored_mtp", False):
        wrapper = _target_anchored_rejection_sample
        wrapper._remtp_target_anchored_mtp = True
        wrapper._remtp_original = current
        wrapper._remtp_config = config
        wrapper._remtp_diagnostics = diagnostics
        wrapper._remtp_audit_interval = audit_interval
        wrapper._remtp_fast = fast
        module.rejection_sample = wrapper

    print(
        "[ReMTP][TargetAnchored] "
        f"variant={config.variant} cactus_delta={config.cactus_delta:g} "
        f"draft_tokens={config.expected_draft_tokens} "
        f"head_reliability={list(config.head_reliability)} "
        f"gap_scale={config.target_log_gap_scale:g} "
        f"max_gap={config.max_target_log_gap:g} "
        f"future_veto_floor={config.future_veto_floor:g} "
        f"hidden_floor={config.hidden_reliability_floor:g} "
        f"compiled={int(compile_fast_path)} "
        "budget=exact-Cactus-TV cap=h(y)<=q(y)",
        flush=True,
    )
