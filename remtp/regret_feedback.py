"""Cross-block regret feedback for target-anchored probabilistic MTP.

The verifier remains the target-anchored exact-TV method.  This module only
observes causal relaxed acceptances (the same uniform random number fails the
strict test but passes the relaxed test), remembers the target-side probability
mass sacrificed by those events, and weakly steers the *next* MTP root hidden
state.  Target-model hidden states, logits, caches, and verifier distributions
are never modified.

The runtime adapter intentionally targets the repository benchmark contract:
one request at a time, probabilistic MTP, tensor parallel size one.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


_EAGLE_MODULE = "vllm.v1.spec_decode.eagle"
_GPU_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_DEFAULT_HEAD_RELIABILITY = (1.0, 0.85, 0.70, 0.55, 0.40, 0.30)


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
            "REMTP_REGRET_HEAD_RELIABILITY must contain values in [0,1]"
        )
    return result


@dataclass(frozen=True)
class RegretFeedbackConfig:
    """Configuration for Budget-Induced Regret Feedback."""

    expected_draft_tokens: int = 6
    regret_top_k: int = 16
    compatibility_top_k: int = 32
    alpha: float = 0.03
    token_decay: float = 0.90
    rejection_reset: float = 0.25
    max_idle_blocks: int = 2
    incompatible_decay: float = 0.50
    head_reliability: tuple[float, ...] = _DEFAULT_HEAD_RELIABILITY
    audit_interval: int = 0
    diagnostics: bool = False

    def validate(self) -> None:
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if self.regret_top_k < 1 or self.compatibility_top_k < 2:
            raise ValueError("regret/compatibility top-k values are invalid")
        if self.alpha < 0.0 or self.alpha > 0.25:
            raise ValueError("alpha must be in [0, 0.25]")
        if not 0.0 < self.token_decay <= 1.0:
            raise ValueError("token_decay must be in (0,1]")
        if not 0.0 <= self.rejection_reset <= 1.0:
            raise ValueError("rejection_reset must be in [0,1]")
        if self.max_idle_blocks < 1:
            raise ValueError("max_idle_blocks must be positive")
        if not 0.0 <= self.incompatible_decay <= 1.0:
            raise ValueError("incompatible_decay must be in [0,1]")
        if len(self.head_reliability) < self.expected_draft_tokens:
            raise ValueError("head reliability must cover every MTP position")
        if self.audit_interval < 0:
            raise ValueError("audit_interval must be non-negative")

    @classmethod
    def from_env(cls) -> RegretFeedbackConfig:
        expected = int(os.getenv("REMTP_REGRET_EXPECTED_DRAFT_TOKENS", "6"))
        default_heads = ",".join(
            str(value)
            for value in _DEFAULT_HEAD_RELIABILITY[:expected]
        )
        config = cls(
            expected_draft_tokens=expected,
            regret_top_k=int(os.getenv("REMTP_REGRET_TOP_K", "16")),
            compatibility_top_k=int(
                os.getenv("REMTP_REGRET_COMPATIBILITY_TOP_K", "32")
            ),
            alpha=float(os.getenv("REMTP_REGRET_ALPHA", "0.03")),
            token_decay=float(os.getenv("REMTP_REGRET_TOKEN_DECAY", "0.90")),
            rejection_reset=float(
                os.getenv("REMTP_REGRET_REJECTION_RESET", "0.25")
            ),
            max_idle_blocks=int(
                os.getenv("REMTP_REGRET_MAX_IDLE_BLOCKS", "2")
            ),
            incompatible_decay=float(
                os.getenv("REMTP_REGRET_INCOMPATIBLE_DECAY", "0.50")
            ),
            head_reliability=_parse_head_reliability(
                os.getenv("REMTP_REGRET_HEAD_RELIABILITY", default_heads)
            ),
            audit_interval=int(
                os.getenv("REMTP_REGRET_AUDIT_INTERVAL", "0")
            ),
            diagnostics=_env_flag("REMTP_REGRET_DIAGNOSTICS", False),
        )
        config.validate()
        return config


@dataclass
class RegretRuntimeState:
    """Request-local memory and the latest target compatibility anchor."""

    memory: torch.Tensor | None = None
    idle_blocks: torch.Tensor | None = None
    output_weight: torch.Tensor | None = None
    compatibility_ids: torch.Tensor | None = None
    compatibility_probs: torch.Tensor | None = None
    bonus_ids: torch.Tensor | None = None
    bonus_probs: torch.Tensor | None = None

    def reset_request(self) -> None:
        self.memory = None
        self.idle_blocks = None
        self.compatibility_ids = None
        self.compatibility_probs = None
        self.bonus_ids = None
        self.bonus_probs = None


@dataclass
class RegretAudit:
    """GPU-side windowed mechanism counters."""

    rounds: int = 0
    accepted: torch.Tensor | None = None
    strict_accepted: torch.Tensor | None = None
    causal_relaxed: torch.Tensor | None = None
    accepted_tv: torch.Tensor | None = None
    injected: torch.Tensor | None = None
    gate_sum: torch.Tensor | None = None
    memory_strength: torch.Tensor | None = None
    head_reached: torch.Tensor | None = None
    head_strict: torch.Tensor | None = None
    head_causal: torch.Tensor | None = None
    head_target_prob: torch.Tensor | None = None
    head_hidden_cos: torch.Tensor | None = None

    def ensure(self, rows: int, device: torch.device) -> None:
        if self.accepted is not None:
            return
        scalar = torch.zeros((), device=device, dtype=torch.float64)
        vector = torch.zeros(rows, device=device, dtype=torch.float64)
        self.accepted = scalar.clone()
        self.strict_accepted = scalar.clone()
        self.causal_relaxed = scalar.clone()
        self.accepted_tv = scalar.clone()
        self.injected = scalar.clone()
        self.gate_sum = scalar.clone()
        self.memory_strength = scalar.clone()
        self.head_reached = vector.clone()
        self.head_strict = vector.clone()
        self.head_causal = vector.clone()
        self.head_target_prob = vector.clone()
        self.head_hidden_cos = vector.clone()

    def zero(self) -> None:
        for name in (
            "accepted",
            "strict_accepted",
            "causal_relaxed",
            "accepted_tv",
            "injected",
            "gate_sum",
            "memory_strength",
            "head_reached",
            "head_strict",
            "head_causal",
            "head_target_prob",
            "head_hidden_cos",
        ):
            value = getattr(self, name)
            if value is not None:
                value.zero_()


_CONFIG: RegretFeedbackConfig | None = None
_STATE = RegretRuntimeState()
_AUDIT = RegretAudit()
_LAST_UNIFORM_PROBS: torch.Tensor | None = None
_DIAGNOSTIC_EMITTED = False


def accepted_prefix_mask(
    output_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the leading draft tokens committed by the verifier."""
    if output_token_ids.ndim != 2 or output_token_ids.shape[0] != 1:
        raise ValueError("regret feedback requires one output row")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must be one-dimensional")
    equal = (
        output_token_ids[0, : draft_token_ids.shape[0]].to(torch.int64)
        == draft_token_ids.to(torch.int64)
    )
    return torch.cumprod(equal.to(torch.int32), dim=0).to(torch.bool)


def strict_and_causal_masks(
    accepted: torch.Tensor,
    strict_acceptance: torch.Tensor,
    *,
    uniform_probs: torch.Tensor | None,
    greedy_strict: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify reached positions under the exact verifier coupling."""
    reached = torch.cat(
        (
            torch.ones(1, device=accepted.device, dtype=torch.bool),
            accepted[:-1],
        )
    )
    if uniform_probs is not None:
        strict_pass = (
            uniform_probs[: accepted.shape[0]].to(strict_acceptance.device)
            <= strict_acceptance
        )
    elif greedy_strict is not None:
        strict_pass = greedy_strict.to(device=accepted.device, dtype=torch.bool)
    else:
        raise RuntimeError("the verifier uniform random values were not captured")
    strict_accepted = accepted & strict_pass
    causal_relaxed = accepted & ~strict_pass
    return reached, strict_accepted, causal_relaxed


def _compact_non_candidate_topk(
    top_probs: torch.Tensor,
    top_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    keep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(
        top_ids.shape[1],
        device=top_ids.device,
    ).expand_as(top_ids)
    positions = torch.where(
        top_ids == candidate_ids.unsqueeze(1),
        torch.full_like(positions, top_ids.shape[1]),
        positions,
    )
    compact_order = positions.argsort(dim=1)
    compact_probs = top_probs.gather(1, compact_order)[:, :keep]
    compact_ids = top_ids.gather(1, compact_order)[:, :keep]
    return compact_probs, compact_ids


def regret_block_vector(
    target_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    allocated_tv: torch.Tensor,
    causal_relaxed: torch.Tensor,
    accepted_count: torch.Tensor,
    output_weight: torch.Tensor,
    config: RegretFeedbackConfig,
    *,
    top_probs: torch.Tensor | None = None,
    top_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the weighted target-preference residual for one block."""
    rows, vocab_size = target_probs.shape
    if top_probs is None or top_ids is None:
        search_k = min(config.regret_top_k + 1, vocab_size)
        top_probs, top_ids = target_probs.topk(search_k, dim=-1)
    alt_probs, alt_ids = _compact_non_candidate_topk(
        top_probs,
        top_ids,
        candidate_ids,
        min(config.regret_top_k, top_probs.shape[1] - 1),
    )

    row_index = torch.arange(rows, device=target_probs.device)
    candidate_probs = target_probs[row_index, candidate_ids]
    alternative_mass = (1.0 - candidate_probs).clamp_min(1e-30)
    captured_mass = (
        alt_probs.sum(dim=-1) / alternative_mass
    ).clamp(0.0, 1.0)

    # Concentration is computed over the explicitly retained alternatives plus
    # one tail bucket.  Diffuse target preferences therefore receive little
    # confidence even if the top-k entries themselves look sharp.
    tail_mass = (
        alternative_mass - alt_probs.sum(dim=-1)
    ).clamp_min(0.0)
    entropy_components = torch.cat(
        (alt_probs, tail_mass.unsqueeze(1)),
        dim=1,
    ) / alternative_mass.unsqueeze(1)
    normalized_entropy = -(
        entropy_components
        * torch.log(entropy_components.clamp_min(1e-30))
    ).sum(dim=-1) / math.log(entropy_components.shape[1])
    concentration = (
        (1.0 - normalized_entropy).clamp(0.0, 1.0)
        * captured_mass
    )

    preference_weights = (
        alt_probs / alt_probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    )
    preferred_rows = F.normalize(
        output_weight[alt_ids].to(torch.float32),
        dim=-1,
        eps=1e-8,
    )
    candidate_rows = F.normalize(
        output_weight[candidate_ids].to(torch.float32),
        dim=-1,
        eps=1e-8,
    )
    preferred_center = (
        preference_weights.unsqueeze(-1) * preferred_rows
    ).sum(dim=1)
    directions = preferred_center - candidate_rows
    direction_rms = torch.sqrt(
        directions.square().mean(dim=-1, keepdim=True)
    ).clamp_min(1e-8)
    directions = directions / direction_rms

    reliability = torch.as_tensor(
        config.head_reliability[:rows],
        device=target_probs.device,
        dtype=torch.float32,
    )
    reliability_range = (
        reliability.amax() - reliability.amin()
    ).clamp_min(1e-8)
    head_risk = (
        0.5
        + (reliability.amax() - reliability) / reliability_range
    ).clamp(0.5, 1.5)
    depth = torch.arange(rows, device=target_probs.device)
    distance = (
        accepted_count.to(depth.dtype) - 1 - depth
    ).clamp_min(0)
    distance_weight = torch.pow(
        torch.full_like(allocated_tv, config.token_decay),
        distance.to(allocated_tv.dtype),
    )
    event_weight = (
        allocated_tv
        * concentration
        * head_risk
        * distance_weight
        * causal_relaxed.to(allocated_tv.dtype)
    )
    raw_vector = (event_weight.unsqueeze(1) * directions).sum(dim=0)
    total_strength = event_weight.sum()
    vector_rms = torch.sqrt(raw_vector.square().mean()).clamp_min(1e-8)
    block_vector = raw_vector / vector_rms * total_strength
    return block_vector, total_strength, concentration


def update_regret_memory(
    old_memory: torch.Tensor,
    idle_blocks: torch.Tensor,
    block_vector: torch.Tensor,
    block_strength: torch.Tensor,
    committed_tokens: torch.Tensor,
    rejected: torch.Tensor,
    config: RegretFeedbackConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply token decay, rejection reset, reinforcement, and finite lifetime."""
    decay = torch.pow(
        torch.as_tensor(
            config.token_decay,
            device=old_memory.device,
            dtype=torch.float32,
        ),
        committed_tokens.to(torch.float32),
    )
    retained = old_memory * decay
    retained = torch.where(
        rejected,
        retained * config.rejection_reset,
        retained,
    )
    has_new = block_strength > 0.0
    next_idle = torch.where(
        has_new,
        torch.zeros_like(idle_blocks),
        idle_blocks + 1,
    )
    memory = retained + block_vector
    expired = (~has_new) & (next_idle >= config.max_idle_blocks)
    memory = torch.where(expired, torch.zeros_like(memory), memory)
    return memory, next_idle


def compatibility_gate(
    memory: torch.Tensor,
    output_weight: torch.Tensor,
    top_ids: torch.Tensor,
    top_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure whether regret preferentially lifts current high-P tokens."""
    rows = F.normalize(
        output_weight[top_ids].to(torch.float32),
        dim=-1,
        eps=1e-8,
    )
    memory_direction = memory.to(torch.float32)
    memory_direction = memory_direction / torch.sqrt(
        memory_direction.square().mean()
    ).clamp_min(1e-8)
    pushes = rows @ memory_direction
    weights = top_probs.to(torch.float32)
    weights = weights / weights.sum().clamp_min(1e-30)
    correlation = (weights * pushes).sum() - pushes.mean()
    normalized = correlation / pushes.std(unbiased=False).clamp_min(1e-8)
    return normalized.clamp(0.0, 1.0), correlation


def inject_regret(
    hidden: torch.Tensor,
    memory: torch.Tensor,
    gate: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Orthogonal, RMS-preserving activation steering."""
    source = hidden.to(torch.float32)
    direction = memory.to(device=source.device, dtype=torch.float32)
    while direction.ndim < source.ndim:
        direction = direction.unsqueeze(0)
    projection = (
        (source * direction).sum(dim=-1, keepdim=True)
        / source.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    orthogonal = direction - projection * source
    source_rms = torch.sqrt(source.square().mean(dim=-1, keepdim=True))
    direction_rms = torch.sqrt(
        orthogonal.square().mean(dim=-1, keepdim=True)
    ).clamp_min(1e-8)
    steered = source + alpha * gate * orthogonal / direction_rms * source_rms
    steered_rms = torch.sqrt(
        steered.square().mean(dim=-1, keepdim=True)
    ).clamp_min(1e-8)
    steered = steered * source_rms / steered_rms
    return steered.to(hidden.dtype)


def _capture_uniform_probs(*args: Any, **kwargs: Any) -> torch.Tensor:
    global _LAST_UNIFORM_PROBS

    original = getattr(_capture_uniform_probs, "_remtp_original")
    result = original(*args, **kwargs)
    _LAST_UNIFORM_PROBS = result
    return result


def _capture_bonus_logits(
    bonus_logits: torch.Tensor,
    sampling_metadata: Any,
) -> None:
    config = _require_config()
    if bonus_logits.shape[0] != 1:
        raise RuntimeError("regret feedback requires --max-num-seqs 1")
    k = min(config.compatibility_top_k, bonus_logits.shape[-1])
    values, ids = bonus_logits[0].topk(k)
    temperature = sampling_metadata.temperature[0].to(torch.float32)
    safe_temperature = torch.where(
        temperature < 1e-5,
        torch.ones_like(temperature),
        temperature,
    )
    _STATE.bonus_ids = ids
    _STATE.bonus_probs = torch.softmax(
        values.to(torch.float32) / safe_temperature,
        dim=-1,
    )


def _post_verification(
    *,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    output_token_ids: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
    allocated_tv: torch.Tensor,
    strict_acceptance: torch.Tensor,
    hidden_similarity: torch.Tensor,
    sampling_metadata: Any,
) -> None:
    global _DIAGNOSTIC_EMITTED

    config = _require_config()
    rows = draft_token_ids.shape[0]
    if rows != config.expected_draft_tokens:
        raise RuntimeError(
            f"regret configured for MTP={config.expected_draft_tokens}, got {rows}"
        )
    output_weight = _STATE.output_weight
    if output_weight is None:
        raise RuntimeError("MTP shared output head was not registered")

    accepted = accepted_prefix_mask(output_token_ids, draft_token_ids)
    accepted_count = accepted.to(torch.int64).sum()
    rejected = accepted_count < rows
    if sampling_metadata.all_greedy:
        target_top = target_probs.argmax(dim=-1)
        greedy_strict = draft_token_ids == target_top
        uniform_probs = None
    else:
        greedy_strict = None
        uniform_probs = _LAST_UNIFORM_PROBS
    reached, strict_accepted, causal_relaxed = strict_and_causal_masks(
        accepted,
        strict_acceptance,
        uniform_probs=uniform_probs,
        greedy_strict=greedy_strict,
    )

    search_k = min(
        max(config.regret_top_k + 1, config.compatibility_top_k),
        target_probs.shape[-1],
    )
    top_probs, top_ids = target_probs.topk(search_k, dim=-1)
    block_vector, block_strength, concentration = regret_block_vector(
        target_probs,
        draft_token_ids.to(torch.int64),
        allocated_tv,
        causal_relaxed,
        accepted_count,
        output_weight,
        config,
        top_probs=top_probs,
        top_ids=top_ids,
    )

    if _STATE.memory is None:
        _STATE.memory = torch.zeros_like(block_vector)
        _STATE.idle_blocks = torch.zeros(
            (),
            device=block_vector.device,
            dtype=torch.int64,
        )
    assert _STATE.idle_blocks is not None
    committed_tokens = accepted_count + 1
    _STATE.memory, _STATE.idle_blocks = update_regret_memory(
        _STATE.memory,
        _STATE.idle_blocks,
        block_vector,
        block_strength,
        committed_tokens,
        rejected,
        config,
    )

    reference_index = accepted_count.clamp(max=rows - 1)
    target_reference_ids = top_ids[reference_index, : config.compatibility_top_k]
    target_reference_probs = top_probs[
        reference_index, : config.compatibility_top_k
    ]
    target_reference_probs = (
        target_reference_probs
        / target_reference_probs.sum().clamp_min(1e-30)
    )
    if _STATE.bonus_ids is not None and _STATE.bonus_probs is not None:
        all_accepted = ~rejected
        _STATE.compatibility_ids = torch.where(
            all_accepted,
            _STATE.bonus_ids,
            target_reference_ids,
        )
        _STATE.compatibility_probs = torch.where(
            all_accepted,
            _STATE.bonus_probs,
            target_reference_probs,
        )
    else:
        _STATE.compatibility_ids = target_reference_ids
        _STATE.compatibility_probs = target_reference_probs
    _STATE.bonus_ids = None
    _STATE.bonus_probs = None

    _AUDIT.ensure(rows, target_probs.device)
    assert _AUDIT.accepted is not None
    assert _AUDIT.strict_accepted is not None
    assert _AUDIT.causal_relaxed is not None
    assert _AUDIT.accepted_tv is not None
    assert _AUDIT.memory_strength is not None
    assert _AUDIT.head_reached is not None
    assert _AUDIT.head_strict is not None
    assert _AUDIT.head_causal is not None
    assert _AUDIT.head_target_prob is not None
    assert _AUDIT.head_hidden_cos is not None
    _AUDIT.rounds += 1
    _AUDIT.accepted += accepted.sum().to(torch.float64)
    _AUDIT.strict_accepted += strict_accepted.sum().to(torch.float64)
    _AUDIT.causal_relaxed += causal_relaxed.sum().to(torch.float64)
    _AUDIT.accepted_tv += (
        allocated_tv * accepted.to(allocated_tv.dtype)
    ).sum().to(torch.float64)
    _AUDIT.memory_strength += block_strength.to(torch.float64)
    _AUDIT.head_reached += reached.to(torch.float64)
    _AUDIT.head_strict += (reached & strict_accepted).to(torch.float64)
    _AUDIT.head_causal += causal_relaxed.to(torch.float64)
    _AUDIT.head_target_prob += (
        target_candidate_probs * reached
    ).to(torch.float64)
    _AUDIT.head_hidden_cos += (
        hidden_similarity * reached
    ).to(torch.float64)

    if config.diagnostics and not _DIAGNOSTIC_EMITTED:
        print(
            "[ReMTP][Regret][diagnostic] "
            f"accepted={accepted.tolist()} "
            f"strict={strict_accepted.tolist()} "
            f"causal={causal_relaxed.tolist()} "
            f"p={target_candidate_probs.tolist()} "
            f"h={boosted_candidate_probs.tolist()} "
            f"allocated_TV={allocated_tv.tolist()} "
            f"concentration={concentration.tolist()} "
            f"block_strength={block_strength.item():.6f}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True
    if (
        config.audit_interval > 0
        and _AUDIT.rounds % config.audit_interval == 0
    ):
        _print_audit(config.audit_interval)


def _print_audit(window_rounds: int) -> None:
    assert _AUDIT.accepted is not None
    assert _AUDIT.strict_accepted is not None
    assert _AUDIT.causal_relaxed is not None
    assert _AUDIT.accepted_tv is not None
    assert _AUDIT.injected is not None
    assert _AUDIT.gate_sum is not None
    assert _AUDIT.memory_strength is not None
    assert _AUDIT.head_reached is not None
    assert _AUDIT.head_strict is not None
    assert _AUDIT.head_causal is not None
    assert _AUDIT.head_target_prob is not None
    assert _AUDIT.head_hidden_cos is not None
    accepted = _AUDIT.accepted.item()
    strict_ratio = _AUDIT.strict_accepted.item() / max(accepted, 1.0)
    tv_per_accepted = _AUDIT.accepted_tv.item() / max(accepted, 1.0)
    injection_count = _AUDIT.injected.item()
    print(
        "[ReMTP][Regret][audit] "
        f"rounds={_AUDIT.rounds-window_rounds+1}-{_AUDIT.rounds} "
        f"accepted={accepted:.0f} "
        f"strict_accepted={_AUDIT.strict_accepted.item():.0f} "
        f"causal_relaxed={_AUDIT.causal_relaxed.item():.0f} "
        f"strict_ratio={strict_ratio:.6f} "
        f"tv_per_accepted={tv_per_accepted:.6f} "
        f"injections={injection_count:.0f} "
        f"mean_gate={_AUDIT.gate_sum.item()/max(injection_count,1.0):.6f} "
        f"mean_regret_strength="
        f"{_AUDIT.memory_strength.item()/window_rounds:.6f} "
        f"head_reached={_AUDIT.head_reached.tolist()} "
        f"head_strict_count={_AUDIT.head_strict.tolist()} "
        f"head_causal_count={_AUDIT.head_causal.tolist()} "
        f"head_target_p_sum={_AUDIT.head_target_prob.tolist()} "
        f"head_hidden_cos_sum={_AUDIT.head_hidden_cos.tolist()}",
        flush=True,
    )
    _AUDIT.zero()


def _propose_with_regret(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    config = _require_config()
    original = getattr(_propose_with_regret, "_remtp_original")

    lm_head = getattr(getattr(self, "model", None), "lm_head", None)
    output_weight = getattr(lm_head, "weight", None)
    if isinstance(output_weight, torch.Tensor):
        _STATE.output_weight = output_weight.detach()

    if "target_hidden_states" in kwargs:
        target_hidden = kwargs["target_hidden_states"]
        hidden_location: tuple[str, int] = ("kwargs", -1)
    elif len(args) >= 3:
        target_hidden = args[2]
        hidden_location = ("args", 2)
    else:
        raise RuntimeError("EagleProposer.propose target hidden argument changed")

    if (
        _STATE.memory is not None
        and _STATE.output_weight is not None
        and _STATE.compatibility_ids is not None
        and _STATE.compatibility_probs is not None
    ):
        token_indices = kwargs.get("token_indices_to_sample")
        if token_indices is None and len(args) >= 5:
            token_indices = args[4]
        common_metadata = kwargs.get("common_attn_metadata")
        if common_metadata is None and len(args) >= 6:
            common_metadata = args[5]
        if token_indices is None:
            token_indices = common_metadata.query_start_loc[1:] - 1
        token_indices = token_indices.to(
            device=target_hidden.device,
            dtype=torch.int64,
        )
        selected = target_hidden.index_select(0, token_indices)
        gate, correlation = compatibility_gate(
            _STATE.memory,
            _STATE.output_weight,
            _STATE.compatibility_ids,
            _STATE.compatibility_probs,
        )
        positive = correlation > 0.0
        _STATE.memory = torch.where(
            positive,
            _STATE.memory,
            _STATE.memory * config.incompatible_decay,
        )
        modified = inject_regret(
            selected,
            _STATE.memory,
            gate,
            config.alpha,
        )
        target_hidden = target_hidden.clone()
        target_hidden.index_copy_(0, token_indices, modified)
        if hidden_location[0] == "kwargs":
            kwargs["target_hidden_states"] = target_hidden
        else:
            mutable_args = list(args)
            mutable_args[hidden_location[1]] = target_hidden
            args = tuple(mutable_args)

        _AUDIT.ensure(config.expected_draft_tokens, target_hidden.device)
        assert _AUDIT.injected is not None
        assert _AUDIT.gate_sum is not None
        _AUDIT.injected += (gate > 0.0).to(torch.float64)
        _AUDIT.gate_sum += gate.to(torch.float64)

    return original(self, *args, **kwargs)


def _sample_tokens_with_request_reset(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    state = getattr(self, "execute_model_state", None)
    if state is None or state.spec_decode_metadata is None:
        _STATE.reset_request()
    original = getattr(_sample_tokens_with_request_reset, "_remtp_original")
    return original(self, *args, **kwargs)


def _require_config() -> RegretFeedbackConfig:
    if _CONFIG is None:
        raise RuntimeError("regret feedback has not been installed")
    return _CONFIG


def install_regret_feedback() -> None:
    """Install cross-block regret memory without changing verification."""
    global _CONFIG

    config = RegretFeedbackConfig.from_env()
    _CONFIG = config

    rejection_module = importlib.import_module(_REJECTION_MODULE)
    current_uniform = rejection_module.generate_uniform_probs
    if not getattr(current_uniform, "_remtp_regret_uniform_capture", False):
        _capture_uniform_probs._remtp_regret_uniform_capture = True
        _capture_uniform_probs._remtp_original = current_uniform
        rejection_module.generate_uniform_probs = _capture_uniform_probs

    from remtp.probabilistic_mtp import set_bonus_logits_hook
    from remtp.target_anchored_mtp import set_post_verification_hook

    set_bonus_logits_hook(_capture_bonus_logits)
    set_post_verification_hook(_post_verification)

    eagle_module = importlib.import_module(_EAGLE_MODULE)
    proposer_cls = eagle_module.EagleProposer
    current_propose = proposer_cls.propose
    if not getattr(current_propose, "_remtp_regret_feedback", False):
        _propose_with_regret._remtp_regret_feedback = True
        _propose_with_regret._remtp_original = current_propose
        proposer_cls.propose = _propose_with_regret

    runner_module = importlib.import_module(_GPU_RUNNER_MODULE)
    runner_cls = runner_module.GPUModelRunner
    current_sample_tokens = runner_cls.sample_tokens
    if not getattr(current_sample_tokens, "_remtp_regret_feedback", False):
        _sample_tokens_with_request_reset._remtp_regret_feedback = True
        _sample_tokens_with_request_reset._remtp_original = current_sample_tokens
        runner_cls.sample_tokens = _sample_tokens_with_request_reset

    print(
        "[ReMTP][Regret] Budget-Induced Regret Feedback enabled "
        f"draft_tokens={config.expected_draft_tokens} "
        f"top_k={config.regret_top_k} "
        f"compat_top_k={config.compatibility_top_k} "
        f"alpha={config.alpha:g} "
        f"token_decay={config.token_decay:g} "
        f"rejection_reset={config.rejection_reset:g} "
        f"max_idle_blocks={config.max_idle_blocks} "
        "causal_only=1 verifier_unchanged=1 root_injection=1",
        flush=True,
    )
