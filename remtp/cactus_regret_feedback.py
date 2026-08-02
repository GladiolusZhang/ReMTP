"""One-block residual regret feedback on top of an unchanged Cactus verifier.

Only future probabilistic-MTP proposals are modified.  Cactus target
distributions, rejection sampling, recovery, target logits, and target caches
remain untouched.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch


_EAGLE_MODULE = "vllm.v1.spec_decode.eagle"
_GPU_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"


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


@dataclass(frozen=True)
class CactusRegretConfig:
    """Small, task-agnostic feedback configuration."""

    expected_draft_tokens: int = 6
    top_k: int = 16
    alpha: float = 0.03
    strength_reference: float = 0.10
    depth_decay: float = 0.90
    responsibility: str = "posterior"
    injection_site: str = "head1"
    residual_space: str = "output"
    audit_interval: int = 0
    diagnostics: bool = False

    def validate(self) -> None:
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.alpha <= 0.10:
            raise ValueError("alpha must be in [0,0.10]")
        if self.strength_reference <= 0.0:
            raise ValueError("strength_reference must be positive")
        if not 0.0 < self.depth_decay <= 1.0:
            raise ValueError("depth_decay must be in (0,1]")
        if self.responsibility not in {"posterior", "realized"}:
            raise ValueError(
                "responsibility must be 'posterior' or 'realized'"
            )
        if self.injection_site not in {"root", "head1"}:
            raise ValueError("injection_site must be 'root' or 'head1'")
        if self.residual_space not in {"boundary_hidden", "hidden", "output"}:
            raise ValueError(
                "residual_space must be 'boundary_hidden', 'hidden', or 'output'"
            )
        if self.audit_interval < 0:
            raise ValueError("audit_interval must be non-negative")

    @classmethod
    def from_env(cls) -> "CactusRegretConfig":
        config = cls(
            expected_draft_tokens=int(
                os.getenv("REMTP_CACTUS_REGRET_EXPECTED_DRAFT_TOKENS", "6")
            ),
            top_k=int(os.getenv("REMTP_CACTUS_REGRET_TOP_K", "16")),
            alpha=float(os.getenv("REMTP_CACTUS_REGRET_ALPHA", "0.03")),
            strength_reference=float(
                os.getenv("REMTP_CACTUS_REGRET_STRENGTH_REFERENCE", "0.10")
            ),
            depth_decay=float(
                os.getenv("REMTP_CACTUS_REGRET_DEPTH_DECAY", "0.90")
            ),
            responsibility=os.getenv(
                "REMTP_CACTUS_REGRET_RESPONSIBILITY", "posterior"
            ).strip().lower(),
            injection_site=os.getenv(
                "REMTP_CACTUS_REGRET_INJECTION_SITE", "head1"
            ).strip().lower(),
            residual_space=os.getenv(
                "REMTP_CACTUS_REGRET_RESIDUAL_SPACE", "output"
            ).strip().lower(),
            audit_interval=int(
                os.getenv("REMTP_CACTUS_REGRET_AUDIT_INTERVAL", "0")
            ),
            diagnostics=_env_flag("REMTP_CACTUS_REGRET_DIAGNOSTICS", False),
        )
        config.validate()
        return config


@dataclass
class CactusRegretState:
    memory: torch.Tensor | None = None
    strength: torch.Tensor | None = None
    output_weight: torch.Tensor | None = None
    applied_gate: torch.Tensor | None = None

    def reset_request(self) -> None:
        self.memory = None
        self.strength = None
        self.output_weight = None
        self.applied_gate = None


@dataclass
class CactusRegretAudit:
    rounds: int = 0
    accepted: torch.Tensor | None = None
    strict_accepted: torch.Tensor | None = None
    causal_accepted: torch.Tensor | None = None
    posterior_responsibility: torch.Tensor | None = None
    transferred_tv: torch.Tensor | None = None
    injections: torch.Tensor | None = None
    gate_sum: torch.Tensor | None = None
    injected_target_prob: torch.Tensor | None = None
    injected_reached: torch.Tensor | None = None
    plain_target_prob: torch.Tensor | None = None
    plain_reached: torch.Tensor | None = None

    def ensure(self, device: torch.device) -> None:
        if self.accepted is not None:
            return
        scalar = torch.zeros((), device=device, dtype=torch.float64)
        self.accepted = scalar.clone()
        self.strict_accepted = scalar.clone()
        self.causal_accepted = scalar.clone()
        self.posterior_responsibility = scalar.clone()
        self.transferred_tv = scalar.clone()
        self.injections = scalar.clone()
        self.gate_sum = scalar.clone()
        self.injected_target_prob = scalar.clone()
        self.injected_reached = scalar.clone()
        self.plain_target_prob = scalar.clone()
        self.plain_reached = scalar.clone()

    def zero(self) -> None:
        for name in (
            "accepted",
            "strict_accepted",
            "causal_accepted",
            "posterior_responsibility",
            "transferred_tv",
            "injections",
            "gate_sum",
            "injected_target_prob",
            "injected_reached",
            "plain_target_prob",
            "plain_reached",
        ):
            value = getattr(self, name)
            if value is not None:
                value.zero_()


_CONFIG: CactusRegretConfig | None = None
_STATE = CactusRegretState()
_AUDIT = CactusRegretAudit()
_LAST_UNIFORM_PROBS: torch.Tensor | None = None
_DIAGNOSTIC_EMITTED = False


def accepted_prefix_mask(
    output_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Identify the leading draft prefix actually committed by Cactus."""
    if output_token_ids.ndim != 2 or output_token_ids.shape[0] != 1:
        raise ValueError("Cactus regret feedback requires one output row")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must be one-dimensional")
    equal = (
        output_token_ids[0, : draft_token_ids.shape[0]].to(torch.int64)
        == draft_token_ids.to(torch.int64)
    )
    return torch.cumprod(equal.to(torch.int32), dim=0).to(torch.bool)


def classify_cactus_causal_acceptance(
    accepted: torch.Tensor,
    target_probs: torch.Tensor,
    cactus_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify strict versus Cactus-causal accepts under shared uniforms."""
    rows = draft_token_ids.shape[0]
    if accepted.shape != (rows,):
        raise ValueError("accepted mask must have one value per draft")
    row_ids = torch.arange(rows, device=target_probs.device)
    token_ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    q_y = draft_probs[row_ids, token_ids].to(torch.float32).clamp_min(1e-30)
    p_y = target_probs[row_ids, token_ids].to(torch.float32)
    h_y = cactus_probs[row_ids, token_ids].to(torch.float32)
    strict_acceptance = torch.minimum(torch.ones_like(q_y), p_y / q_y)
    cactus_acceptance = torch.minimum(torch.ones_like(q_y), h_y / q_y)
    uniforms = uniform_probs[:rows].to(
        device=target_probs.device,
        dtype=torch.float32,
    )
    strict_pass = uniforms <= strict_acceptance
    cactus_pass = uniforms <= cactus_acceptance
    strict_accepted = accepted & strict_pass
    causal = accepted & ~strict_pass & cactus_pass
    return strict_accepted, causal, strict_acceptance, cactus_acceptance


def posterior_causal_responsibility(
    accepted: torch.Tensor,
    strict_acceptance: torch.Tensor,
    cactus_acceptance: torch.Tensor,
) -> torch.Tensor:
    """Estimate how much each committed token owes its acceptance to Cactus.

    Under the shared-uniform coupling, ``A_cactus - A_strict`` is the
    probability mass on which Cactus changes rejection into acceptance.
    Conditional on observing a Cactus acceptance, its posterior causal
    responsibility is therefore ``(A_cactus - A_strict) / A_cactus``.
    This has the same expectation as the realized binary causal event while
    avoiding a high-variance steering decision from one sampled uniform.
    """
    if not (
        accepted.shape == strict_acceptance.shape == cactus_acceptance.shape
    ):
        raise ValueError("acceptance tensors must have identical shapes")
    extra_acceptance = (cactus_acceptance - strict_acceptance).clamp_min(0.0)
    responsibility = extra_acceptance / cactus_acceptance.clamp_min(1e-30)
    return accepted.to(responsibility.dtype) * responsibility.clamp(0.0, 1.0)


def cactus_residual_direction(
    target_probs: torch.Tensor,
    cactus_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    responsibility: torch.Tensor,
    accepted_count: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    top_k: int,
    depth_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate ``W^T(P-H)`` with a fixed-size target top-k gather."""
    if target_probs.shape != cactus_probs.shape or target_probs.ndim != 2:
        raise ValueError("target and Cactus distributions must share [rows,vocab]")
    rows, vocab = target_probs.shape
    if draft_token_ids.shape != (rows,) or responsibility.shape != (rows,):
        raise ValueError(
            "candidate and responsibility tensors must match probability rows"
        )
    if output_weight.ndim != 2 or output_weight.shape[0] != vocab:
        raise ValueError("output_weight must have shape [vocab,hidden]")

    token_ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    row_ids = torch.arange(rows, device=target_probs.device)
    p_y = target_probs[row_ids, token_ids].to(torch.float32)
    h_y = cactus_probs[row_ids, token_ids].to(torch.float32)
    transferred = (h_y - p_y).clamp_min(0.0)

    search_k = min(top_k + 1, vocab)
    top_probs, top_ids = target_probs.to(torch.float32).topk(search_k, dim=-1)
    positions = torch.arange(search_k, device=top_ids.device).expand_as(top_ids)
    positions = torch.where(
        top_ids == token_ids.unsqueeze(1),
        torch.full_like(positions, search_k),
        positions,
    )
    order = positions.argsort(dim=1)
    keep = min(top_k, search_k)
    alternative_ids = top_ids.gather(1, order)[:, :keep]
    alternative_probs = top_probs.gather(1, order)[:, :keep]
    alternative_probs = alternative_probs / alternative_probs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-30)

    preferred_rows = output_weight[alternative_ids].to(torch.float32)
    preferred_center = (
        alternative_probs.unsqueeze(-1) * preferred_rows
    ).sum(dim=1)
    candidate_rows = output_weight[token_ids].to(torch.float32)
    per_row_direction = preferred_center - candidate_rows

    depth = torch.arange(rows, device=target_probs.device)
    distance = (accepted_count.to(depth.dtype) - 1 - depth).clamp_min(0)
    recency = torch.pow(
        torch.full_like(transferred, depth_decay),
        distance.to(torch.float32),
    )
    event_weight = transferred * responsibility.to(torch.float32) * recency
    raw = (event_weight.unsqueeze(1) * per_row_direction).sum(dim=0)
    strength = event_weight.sum()
    rms = torch.sqrt(raw.square().mean()).clamp_min(1e-8)
    direction = raw / rms
    direction = torch.where(strength > 0.0, direction, torch.zeros_like(direction))
    return direction, strength, transferred


def cactus_hidden_residual_direction(
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    target_probs: torch.Tensor,
    cactus_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    responsibility: torch.Tensor,
    accepted_count: torch.Tensor,
    *,
    depth_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate aligned target-minus-MTP feature prediction errors."""
    if draft_hidden.shape != target_hidden.shape or draft_hidden.ndim != 2:
        raise ValueError("draft and target hidden states must share [rows,hidden]")
    if target_probs.shape != cactus_probs.shape or target_probs.ndim != 2:
        raise ValueError("target and Cactus distributions must share [rows,vocab]")
    rows = target_probs.shape[0]
    if draft_hidden.shape[0] != rows:
        raise ValueError("hidden and probability rows must align")
    if draft_token_ids.shape != (rows,) or responsibility.shape != (rows,):
        raise ValueError("candidate and responsibility rows must align")

    token_ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    row_ids = torch.arange(rows, device=target_probs.device)
    p_y = target_probs[row_ids, token_ids].to(torch.float32)
    h_y = cactus_probs[row_ids, token_ids].to(torch.float32)
    transferred = (h_y - p_y).clamp_min(0.0)

    draft = draft_hidden.to(torch.float32)
    target = target_hidden.to(torch.float32)
    draft = draft / torch.sqrt(draft.square().mean(dim=-1, keepdim=True)).clamp_min(
        1e-8
    )
    target = target / torch.sqrt(
        target.square().mean(dim=-1, keepdim=True)
    ).clamp_min(1e-8)
    per_row_direction = target - draft

    depth = torch.arange(rows, device=target_probs.device)
    distance = (accepted_count.to(depth.dtype) - 1 - depth).clamp_min(0)
    recency = torch.pow(
        torch.full_like(transferred, depth_decay),
        distance.to(torch.float32),
    )
    event_weight = transferred * responsibility.to(torch.float32) * recency
    raw = (event_weight.unsqueeze(1) * per_row_direction).sum(dim=0)
    strength = event_weight.sum()
    direction = raw / torch.sqrt(raw.square().mean()).clamp_min(1e-8)
    direction = torch.where(strength > 0.0, direction, torch.zeros_like(direction))
    return direction, strength, transferred


def cactus_boundary_hidden_residual_direction(
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    target_probs: torch.Tensor,
    cactus_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    responsibility: torch.Tensor,
    accepted_count: torch.Tensor,
    *,
    depth_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the full-block boundary error; target recovery clears feedback.

    If Cactus rejects any draft token, its correction token changes the real
    trajectory and the earlier feature errors are no longer adjacent to the
    next MTP block. Only a fully committed block transports the last aligned
    target-minus-MTP hidden residual across the bonus token.
    """
    if draft_hidden.shape != target_hidden.shape or draft_hidden.ndim != 2:
        raise ValueError("draft and target hidden states must share [rows,hidden]")
    if target_probs.shape != cactus_probs.shape or target_probs.ndim != 2:
        raise ValueError("target and Cactus distributions must share [rows,vocab]")
    rows = target_probs.shape[0]
    if draft_hidden.shape[0] != rows:
        raise ValueError("hidden and probability rows must align")
    if draft_token_ids.shape != (rows,) or responsibility.shape != (rows,):
        raise ValueError("candidate and responsibility rows must align")

    token_ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    row_ids = torch.arange(rows, device=target_probs.device)
    p_y = target_probs[row_ids, token_ids].to(torch.float32)
    h_y = cactus_probs[row_ids, token_ids].to(torch.float32)
    transferred = (h_y - p_y).clamp_min(0.0)

    depth = torch.arange(rows, device=target_probs.device)
    distance = (accepted_count.to(depth.dtype) - 1 - depth).clamp_min(0)
    recency = torch.pow(
        torch.full_like(transferred, depth_decay),
        distance.to(torch.float32),
    )
    regret_mass = (
        transferred * responsibility.to(torch.float32) * recency
    ).sum()

    draft_boundary = draft_hidden[-1].to(torch.float32)
    target_boundary = target_hidden[-1].to(torch.float32)
    cosine = torch.nn.functional.cosine_similarity(
        draft_boundary.unsqueeze(0),
        target_boundary.unsqueeze(0),
    )[0]
    feature_trust = (0.25 + 0.75 * cosine).clamp(0.25, 1.0)
    full_block = (accepted_count == rows).to(torch.float32)
    strength = regret_mass * feature_trust * full_block
    raw = target_boundary - draft_boundary
    direction = raw / torch.sqrt(raw.square().mean()).clamp_min(1e-8)
    direction = torch.where(strength > 0.0, direction, torch.zeros_like(direction))
    return direction, strength, transferred


def inject_cactus_regret(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    gate: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply an orthogonal RMS-preserving correction in LM-head hidden space."""
    if alpha == 0.0:
        return hidden
    source = hidden.to(torch.float32)
    vector = direction.to(device=source.device, dtype=torch.float32)
    while vector.ndim < source.ndim:
        vector = vector.unsqueeze(0)
    projection = (
        (source * vector).sum(dim=-1, keepdim=True)
        / source.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    orthogonal = vector - projection * source
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


def _correct_first_draft_hidden(
    hidden_states: torch.Tensor,
    proposal_depth: int,
) -> torch.Tensor:
    config = _require_config()
    if (
        config.injection_site != "head1"
        or proposal_depth != 0
        or _STATE.memory is None
        or _STATE.strength is None
    ):
        return hidden_states
    gate = _feedback_gate(config)
    _STATE.applied_gate = gate.detach()
    return inject_cactus_regret(
        hidden_states,
        _STATE.memory,
        gate,
        config.alpha,
    )


def inject_cactus_regret_at_root(
    target_hidden_states: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    direction: torch.Tensor,
    gate: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Steer only request-end target rows copied into MTP module 1."""
    indices = token_indices_to_sample.to(
        device=target_hidden_states.device,
        dtype=torch.int64,
    )
    selected = target_hidden_states.index_select(0, indices)
    steered = inject_cactus_regret(selected, direction, gate, alpha)
    corrected = target_hidden_states.clone()
    corrected.index_copy_(0, indices, steered)
    return corrected


def _feedback_gate(config: CactusRegretConfig) -> torch.Tensor:
    assert _STATE.strength is not None
    gate = (_STATE.strength.to(torch.float32) / config.strength_reference).clamp(
        0.0, 1.0
    )
    if config.alpha == 0.0:
        gate = torch.zeros_like(gate)
    return gate


def _root_token_indices(kwargs: dict[str, Any]) -> torch.Tensor:
    indices = kwargs.get("token_indices_to_sample")
    if isinstance(indices, torch.Tensor):
        return indices
    metadata = kwargs.get("common_attn_metadata")
    if metadata is None:
        raise RuntimeError("MTP proposer has no common attention metadata")
    return metadata.query_start_loc[1:] - 1


def _propose_with_output_head(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    original = getattr(_propose_with_output_head, "_remtp_original")
    lm_head = getattr(getattr(self, "model", None), "lm_head", None)
    output_weight = getattr(lm_head, "weight", None)
    if isinstance(output_weight, torch.Tensor):
        _STATE.output_weight = output_weight.detach()
        _STATE.applied_gate = torch.zeros(
            (), device=output_weight.device, dtype=torch.float32
        )
    config = _require_config()
    if (
        config.injection_site == "root"
        and _STATE.memory is not None
        and _STATE.strength is not None
    ):
        target_hidden_states = kwargs.get("target_hidden_states")
        if not isinstance(target_hidden_states, torch.Tensor):
            raise RuntimeError("MTP proposer target hidden states were not exposed")
        gate = _feedback_gate(config)
        kwargs = dict(kwargs)
        kwargs["target_hidden_states"] = inject_cactus_regret_at_root(
            target_hidden_states,
            _root_token_indices(kwargs),
            _STATE.memory,
            gate,
            config.alpha,
        )
        _STATE.applied_gate = gate.detach()
    return original(self, *args, **kwargs)


def _post_cactus_verification(
    *,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    cactus_probs: torch.Tensor,
    output_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> None:
    global _DIAGNOSTIC_EMITTED

    config = _require_config()
    rows = draft_token_ids.shape[0]
    if not 1 <= rows <= config.expected_draft_tokens:
        raise RuntimeError(
            f"Cactus regret configured for MTP={config.expected_draft_tokens}, "
            f"got {rows}"
        )
    if sampling_metadata.all_greedy:
        return
    if _LAST_UNIFORM_PROBS is None:
        raise RuntimeError("Cactus verifier uniforms were not captured")
    if _STATE.output_weight is None:
        raise RuntimeError("MTP shared output head was not captured")

    accepted = accepted_prefix_mask(output_token_ids, draft_token_ids)
    accepted_count = accepted.to(torch.int64).sum()
    strict, causal, strict_a, cactus_a = classify_cactus_causal_acceptance(
        accepted,
        target_probs,
        cactus_probs,
        draft_probs,
        draft_token_ids,
        _LAST_UNIFORM_PROBS,
    )
    posterior = posterior_causal_responsibility(
        accepted,
        strict_a,
        cactus_a,
    )
    responsibility = (
        posterior
        if config.responsibility == "posterior"
        else causal.to(torch.float32)
    )
    if config.residual_space in {"boundary_hidden", "hidden"}:
        from remtp.probabilistic_mtp import get_last_aligned_hidden_states

        draft_hidden, target_hidden = get_last_aligned_hidden_states(rows)
        if draft_hidden is None or target_hidden is None:
            raise RuntimeError("aligned MTP/target hidden states were not captured")
        if config.residual_space == "boundary_hidden":
            direction, strength, transferred = (
                cactus_boundary_hidden_residual_direction(
                    draft_hidden,
                    target_hidden,
                    target_probs,
                    cactus_probs,
                    draft_token_ids,
                    responsibility,
                    accepted_count,
                    depth_decay=config.depth_decay,
                )
            )
        else:
            direction, strength, transferred = cactus_hidden_residual_direction(
                draft_hidden,
                target_hidden,
                target_probs,
                cactus_probs,
                draft_token_ids,
                responsibility,
                accepted_count,
                depth_decay=config.depth_decay,
            )
    else:
        direction, strength, transferred = cactus_residual_direction(
            target_probs,
            cactus_probs,
            draft_token_ids,
            responsibility,
            accepted_count,
            _STATE.output_weight,
            top_k=config.top_k,
            depth_decay=config.depth_decay,
        )

    # One-block memory: the current observation replaces the already-applied
    # previous observation. A no-event block therefore clears feedback.
    _STATE.memory = direction.detach()
    _STATE.strength = strength.detach()

    reached = torch.cat(
        (
            torch.ones(1, device=accepted.device, dtype=torch.bool),
            accepted[:-1],
        )
    )
    token_ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    row_ids = torch.arange(rows, device=target_probs.device)
    target_at_draft = target_probs[row_ids, token_ids].to(torch.float32)
    applied_gate = _STATE.applied_gate
    if applied_gate is None:
        applied_gate = torch.zeros((), device=target_probs.device)
    was_injected = applied_gate > 0.0

    _AUDIT.ensure(target_probs.device)
    assert _AUDIT.accepted is not None
    assert _AUDIT.strict_accepted is not None
    assert _AUDIT.causal_accepted is not None
    assert _AUDIT.posterior_responsibility is not None
    assert _AUDIT.transferred_tv is not None
    assert _AUDIT.injections is not None
    assert _AUDIT.gate_sum is not None
    assert _AUDIT.injected_target_prob is not None
    assert _AUDIT.injected_reached is not None
    assert _AUDIT.plain_target_prob is not None
    assert _AUDIT.plain_reached is not None
    _AUDIT.rounds += 1
    _AUDIT.accepted += accepted.sum().to(torch.float64)
    _AUDIT.strict_accepted += strict.sum().to(torch.float64)
    _AUDIT.causal_accepted += causal.sum().to(torch.float64)
    _AUDIT.posterior_responsibility += posterior.sum().to(torch.float64)
    _AUDIT.transferred_tv += (
        transferred * responsibility.to(transferred.dtype)
    ).sum().to(torch.float64)
    _AUDIT.injections += was_injected.to(torch.float64)
    _AUDIT.gate_sum += applied_gate.to(torch.float64)
    reached_f = reached.to(torch.float32)
    _AUDIT.injected_target_prob += torch.where(
        was_injected,
        (target_at_draft * reached_f).sum(),
        torch.zeros((), device=target_probs.device),
    ).to(torch.float64)
    _AUDIT.injected_reached += torch.where(
        was_injected,
        reached_f.sum(),
        torch.zeros((), device=target_probs.device),
    ).to(torch.float64)
    _AUDIT.plain_target_prob += torch.where(
        was_injected,
        torch.zeros((), device=target_probs.device),
        (target_at_draft * reached_f).sum(),
    ).to(torch.float64)
    _AUDIT.plain_reached += torch.where(
        was_injected,
        torch.zeros((), device=target_probs.device),
        reached_f.sum(),
    ).to(torch.float64)

    if config.diagnostics and not _DIAGNOSTIC_EMITTED:
        print(
            "[ReMTP][CactusRegret][diagnostic] "
            "verifier_unchanged=1 "
            f"accepted={accepted.tolist()} strict_A={strict_a.tolist()} "
            f"cactus_A={cactus_a.tolist()} causal={causal.tolist()} "
            f"responsibility={responsibility.tolist()} "
            f"transfer={transferred.tolist()} "
            f"next_strength={strength.item():.6f}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True
    if config.audit_interval > 0 and _AUDIT.rounds % config.audit_interval == 0:
        _print_audit()


def _print_audit() -> None:
    assert _AUDIT.accepted is not None
    assert _AUDIT.strict_accepted is not None
    assert _AUDIT.causal_accepted is not None
    assert _AUDIT.posterior_responsibility is not None
    assert _AUDIT.transferred_tv is not None
    assert _AUDIT.injections is not None
    assert _AUDIT.gate_sum is not None
    assert _AUDIT.injected_target_prob is not None
    assert _AUDIT.injected_reached is not None
    assert _AUDIT.plain_target_prob is not None
    assert _AUDIT.plain_reached is not None
    accepted = _AUDIT.accepted.item()
    injections = _AUDIT.injections.item()
    print(
        "[ReMTP][CactusRegret][audit] "
        f"rounds={_AUDIT.rounds} "
        f"accepted={accepted:.0f} "
        f"strict_accepted={_AUDIT.strict_accepted.item():.0f} "
        f"causal_cactus={_AUDIT.causal_accepted.item():.0f} "
        f"posterior_responsibility={_AUDIT.posterior_responsibility.item():.3f} "
        f"strict_ratio={_AUDIT.strict_accepted.item()/max(accepted,1.0):.6f} "
        f"transferred_tv={_AUDIT.transferred_tv.item():.6f} "
        f"injections={injections:.0f} "
        f"mean_gate={_AUDIT.gate_sum.item()/max(injections,1.0):.6f} "
        f"injected_target_p={_AUDIT.injected_target_prob.item()/max(_AUDIT.injected_reached.item(),1.0):.6f} "
        f"plain_target_p={_AUDIT.plain_target_prob.item()/max(_AUDIT.plain_reached.item(),1.0):.6f}",
        flush=True,
    )
    _AUDIT.zero()


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


def _require_config() -> CactusRegretConfig:
    if _CONFIG is None:
        raise RuntimeError("Cactus regret feedback has not been installed")
    return _CONFIG


def install_cactus_regret_feedback() -> None:
    """Install MTP-only one-block feedback around an unchanged Cactus."""
    global _CONFIG

    config = CactusRegretConfig.from_env()
    _CONFIG = config

    rejection_module = importlib.import_module(_REJECTION_MODULE)
    current_uniform = rejection_module.generate_uniform_probs
    if not getattr(current_uniform, "_remtp_cactus_regret_uniform", False):
        _capture_uniform_probs._remtp_cactus_regret_uniform = True
        _capture_uniform_probs._remtp_original = current_uniform
        rejection_module.generate_uniform_probs = _capture_uniform_probs

    from remtp.cactus_mtp import set_post_verification_hook
    from remtp.probabilistic_mtp import (
        install_aligned_hidden_capture,
        set_draft_hidden_hook,
    )

    if config.residual_space in {"boundary_hidden", "hidden"}:
        install_aligned_hidden_capture()

    set_post_verification_hook(_post_cactus_verification)
    set_draft_hidden_hook(_correct_first_draft_hidden)

    eagle_module = importlib.import_module(_EAGLE_MODULE)
    proposer_cls = eagle_module.EagleProposer
    current_propose = proposer_cls.propose
    if not getattr(current_propose, "_remtp_cactus_regret_output_head", False):
        _propose_with_output_head._remtp_cactus_regret_output_head = True
        _propose_with_output_head._remtp_original = current_propose
        proposer_cls.propose = _propose_with_output_head

    runner_module = importlib.import_module(_GPU_RUNNER_MODULE)
    runner_cls = runner_module.GPUModelRunner
    current_sample_tokens = runner_cls.sample_tokens
    if not getattr(current_sample_tokens, "_remtp_cactus_regret_reset", False):
        _sample_tokens_with_request_reset._remtp_cactus_regret_reset = True
        _sample_tokens_with_request_reset._remtp_original = current_sample_tokens
        runner_cls.sample_tokens = _sample_tokens_with_request_reset

    print(
        "[ReMTP][CactusRegret] enabled verifier=Cactus-unchanged "
        f"feedback=one-block-MTP-{config.injection_site} "
        f"draft_tokens={config.expected_draft_tokens} "
        f"top_k={config.top_k} alpha={config.alpha:g} "
        f"strength_ref={config.strength_reference:g} "
        f"depth_decay={config.depth_decay:g} "
        f"responsibility={config.responsibility} "
        f"injection_site={config.injection_site} "
        f"residual_space={config.residual_space}",
        flush=True,
    )
