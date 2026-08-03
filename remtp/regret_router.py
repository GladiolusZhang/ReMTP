"""Expected-regret routing on top of saturation-aware Exact-TV.

The verifier remains bounded by the Exact-TV allocation.  Cross-block state
is split into a scalar debt and a one-block target-preference direction.  A
tiny optional router converts only causally available context into bounded
per-head proposal and verification controls.
"""

from __future__ import annotations

import atexit
import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from remtp.regret_router_model import (
    RegretRouter,
    RegretRouterArchitecture,
    RegretRouterControls,
    load_regret_router_checkpoint,
)
from remtp.target_anchored_mtp import cactus_tv_increment


_EAGLE_MODULE = "vllm.v1.spec_decode.eagle"
_GPU_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
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
        raise ValueError("head reliability must contain values in [0,1]")
    return result


@dataclass(frozen=True)
class RegretRouterConfig:
    mode: str = "identity"
    checkpoint: str = ""
    expected_draft_tokens: int = 6
    head_reliability: tuple[float, ...] = _DEFAULT_HEAD_RELIABILITY
    top_k: int = 32
    cactus_delta: float = 1.0
    debt_decay: float = 0.90
    debt_reference: float = 0.05
    rejection_reset: float = 0.0
    max_direction_strength: float = 0.03
    max_logit_scale: float = 0.08
    max_budget_reduction: float = 0.50
    collector_dir: str = ""
    collection_id: str = ""
    collector_shard_size: int = 128
    audit_interval: int = 0
    diagnostics: bool = False

    def validate(self) -> None:
        if self.mode not in {"identity", "fixed", "collect", "learned"}:
            raise ValueError("router mode must be identity, fixed, collect, or learned")
        if self.mode == "learned" and not self.checkpoint:
            raise ValueError("learned router mode requires a checkpoint")
        if self.mode == "collect" and not self.collector_dir:
            raise ValueError("collect mode requires collector_dir")
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if len(self.head_reliability) < self.expected_draft_tokens:
            raise ValueError("head reliability must cover all MTP heads")
        if self.top_k < 2:
            raise ValueError("top_k must be at least two")
        if self.cactus_delta < 0.0:
            raise ValueError("cactus_delta must be non-negative")
        if not 0.0 < self.debt_decay <= 1.0:
            raise ValueError("debt_decay must be in (0,1]")
        if self.debt_reference <= 0.0:
            raise ValueError("debt_reference must be positive")
        if not 0.0 <= self.rejection_reset <= 1.0:
            raise ValueError("rejection_reset must be in [0,1]")
        if self.max_direction_strength < 0.0 or self.max_logit_scale < 0.0:
            raise ValueError("router control caps must be non-negative")
        if not 0.0 <= self.max_budget_reduction <= 1.0:
            raise ValueError("max_budget_reduction must be in [0,1]")
        if self.collector_shard_size < 1:
            raise ValueError("collector_shard_size must be positive")
        if self.audit_interval < 0:
            raise ValueError("audit_interval must be non-negative")

    @classmethod
    def from_env(cls) -> "RegretRouterConfig":
        expected = int(os.getenv("REMTP_ROUTER_EXPECTED_DRAFT_TOKENS", "6"))
        defaults = ",".join(
            str(value) for value in _DEFAULT_HEAD_RELIABILITY[:expected]
        )
        config = cls(
            mode=os.getenv("REMTP_ROUTER_MODE", "identity").strip().lower(),
            checkpoint=os.getenv("REMTP_ROUTER_CHECKPOINT", ""),
            expected_draft_tokens=expected,
            head_reliability=_parse_head_reliability(
                os.getenv("REMTP_ROUTER_HEAD_RELIABILITY", defaults)
            ),
            top_k=int(os.getenv("REMTP_ROUTER_TOP_K", "32")),
            cactus_delta=float(os.getenv("REMTP_CACTUS_DELTA", "1.0")),
            debt_decay=float(os.getenv("REMTP_ROUTER_DEBT_DECAY", "0.90")),
            debt_reference=float(
                os.getenv("REMTP_ROUTER_DEBT_REFERENCE", "0.05")
            ),
            rejection_reset=float(
                os.getenv("REMTP_ROUTER_REJECTION_RESET", "0.0")
            ),
            max_direction_strength=float(
                os.getenv("REMTP_ROUTER_MAX_DIRECTION_STRENGTH", "0.03")
            ),
            max_logit_scale=float(
                os.getenv("REMTP_ROUTER_MAX_LOGIT_SCALE", "0.08")
            ),
            max_budget_reduction=float(
                os.getenv("REMTP_ROUTER_MAX_BUDGET_REDUCTION", "0.50")
            ),
            collector_dir=os.getenv("REMTP_ROUTER_COLLECT_DIR", ""),
            collection_id=os.getenv("REMTP_ROUTER_COLLECTION_ID", "").strip(),
            collector_shard_size=int(
                os.getenv("REMTP_ROUTER_COLLECT_SHARD_SIZE", "128")
            ),
            audit_interval=int(os.getenv("REMTP_ROUTER_AUDIT_INTERVAL", "0")),
            diagnostics=_env_flag("REMTP_ROUTER_DIAGNOSTICS", False),
        )
        config.validate()
        return config


@dataclass
class RegretRouterState:
    request_id: str = ""
    debt: torch.Tensor | None = None
    direction: torch.Tensor | None = None
    source_entropy: torch.Tensor | None = None
    source_margin: torch.Tensor | None = None
    output_weight: torch.Tensor | None = None
    router: RegretRouter | None = None
    context_root: torch.Tensor | None = None
    context_direction: torch.Tensor | None = None
    context_debt: torch.Tensor | None = None
    context_entropy: torch.Tensor | None = None
    context_margin: torch.Tensor | None = None
    proposal_controls: RegretRouterControls | None = None
    verification_controls: RegretRouterControls | None = None
    verification_entropy: torch.Tensor | None = None
    verification_margin: torch.Tensor | None = None
    verification_top_probs: torch.Tensor | None = None
    verification_top_ids: torch.Tensor | None = None

    def reset_request(self, request_id: str) -> None:
        router = self.router
        self.__dict__.update(
            RegretRouterState(request_id=request_id, router=router).__dict__
        )


@dataclass
class RegretRouterAudit:
    rounds: int = 0
    accepted: float = 0.0
    strict_expected: float = 0.0
    relaxed_expected: float = 0.0
    debt_added: float = 0.0
    allocated_tv: float = 0.0
    budget_scale: float = 0.0
    direction_strength: float = 0.0
    logit_scale: float = 0.0

    def reset_values(self) -> None:
        self.accepted = 0.0
        self.strict_expected = 0.0
        self.relaxed_expected = 0.0
        self.debt_added = 0.0
        self.allocated_tv = 0.0
        self.budget_scale = 0.0
        self.direction_strength = 0.0
        self.logit_scale = 0.0


@dataclass
class RouterShardWriter:
    directory: Path
    shard_size: int
    records: list[dict[str, Any]] = field(default_factory=list)
    shard_index: int = 0

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.directory.glob("router_shard_*.pt"))
        numeric = [
            int(path.stem.rsplit("_", 1)[-1])
            for path in existing
            if path.stem.rsplit("_", 1)[-1].isdigit()
        ]
        self.shard_index = max(numeric, default=-1) + 1
        pending = self.directory / "router_pending.pt"
        if pending.exists():
            recovered = self.directory / f"router_shard_{self.shard_index:06d}.pt"
            pending.replace(recovered)
            self.shard_index += 1

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if len(self.records) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"router_shard_{self.shard_index:06d}.pt"
        torch.save(
            {"format_version": 1, "records": self.records},
            path,
        )
        print(
            f"[ReMTP][RegretRouter] wrote {len(self.records)} records to {path}",
            flush=True,
        )
        self.records = []
        self.shard_index += 1
        (self.directory / "router_pending.pt").unlink(missing_ok=True)

    def checkpoint(self) -> None:
        """Persist an incomplete shard without synchronizing every round."""
        if not self.records:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"format_version": 1, "records": self.records},
            self.directory / "router_pending.pt",
        )


_CONFIG: RegretRouterConfig | None = None
_STATE = RegretRouterState()
_AUDIT = RegretRouterAudit()
_COLLECTOR: RouterShardWriter | None = None
_DIAGNOSTIC_EMITTED = False
_REQUEST_COUNTER = -1


def accepted_prefix_mask(
    output_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> torch.Tensor:
    if output_token_ids.ndim != 2 or output_token_ids.shape[0] != 1:
        raise ValueError("regret router requires one output row")
    equal = (
        output_token_ids[0, : draft_token_ids.shape[0]].to(torch.int64)
        == draft_token_ids.to(torch.int64)
    )
    return torch.cumprod(equal.to(torch.int32), dim=0).to(torch.bool)


def target_head_statistics(
    probs: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return top-k entropy, top-1 margin, probabilities, and token ids."""
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape [rows,vocab]")
    k = min(top_k, probs.shape[-1])
    top_probs, top_ids = probs.to(torch.float32).topk(k, dim=-1)
    tail = (1.0 - top_probs.sum(dim=-1)).clamp_min(0.0)
    entropy = -(
        top_probs * torch.log(top_probs.clamp_min(1e-30))
    ).sum(dim=-1) - tail * torch.log(tail.clamp_min(1e-30))
    entropy = entropy / torch.log(
        torch.tensor(float(k + 1), device=probs.device)
    )
    margin = (
        torch.log(top_probs[:, 0].clamp_min(1e-30))
        - torch.log(top_probs[:, 1].clamp_min(1e-30))
    ).clamp(0.0, 8.0) / 8.0
    return entropy.clamp(0.0, 1.0), margin, top_probs, top_ids


def expected_regret_debt(
    accepted: torch.Tensor,
    allocated_tv: torch.Tensor,
    strict_acceptance: torch.Tensor,
    relaxed_acceptance: torch.Tensor,
    concentration: torch.Tensor,
    reallocation_fraction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected debt, independent of the sampled verifier uniform."""
    tensors = (
        allocated_tv,
        strict_acceptance,
        relaxed_acceptance,
        concentration,
        reallocation_fraction,
    )
    if any(value.shape != accepted.shape for value in tensors):
        raise ValueError("regret inputs must have one value per draft position")
    responsibility = (relaxed_acceptance - strict_acceptance).clamp_min(0.0)
    weight = (
        accepted.to(torch.float32)
        * allocated_tv.to(torch.float32)
        * responsibility
        * (0.25 + 0.75 * concentration.clamp(0.0, 1.0))
        * (1.0 + reallocation_fraction.clamp(0.0, 1.0))
    )
    return weight, responsibility


def regret_direction_from_target_mass(
    target_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    event_weight: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Project target alternatives minus accepted candidates into hidden space."""
    rows, vocab = target_probs.shape
    if draft_token_ids.shape != (rows,) or event_weight.shape != (rows,):
        raise ValueError("candidate and event weights must match target rows")
    if output_weight.ndim != 2 or output_weight.shape[0] != vocab:
        raise ValueError("output weight must have shape [vocab,hidden]")
    ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    search_k = min(top_k + 1, vocab)
    values, alternatives = target_probs.to(torch.float32).topk(search_k, dim=-1)
    positions = torch.arange(search_k, device=alternatives.device).expand_as(
        alternatives
    )
    positions = torch.where(
        alternatives == ids.unsqueeze(1),
        torch.full_like(positions, search_k),
        positions,
    )
    order = positions.argsort(dim=1)
    keep = min(top_k, search_k)
    alternatives = alternatives.gather(1, order)[:, :keep]
    values = values.gather(1, order)[:, :keep]
    values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    preferred = (
        values.unsqueeze(-1)
        * output_weight[alternatives].to(torch.float32)
    ).sum(dim=1)
    candidate = output_weight[ids].to(torch.float32)
    raw = (event_weight.unsqueeze(1) * (preferred - candidate)).sum(dim=0)
    rms = torch.sqrt(raw.square().mean()).clamp_min(1e-8)
    direction = raw / rms
    return torch.where(
        event_weight.sum() > 0.0,
        direction,
        torch.zeros_like(direction),
    )


def regret_direction_from_top_mass(
    top_probs: torch.Tensor,
    top_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    event_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    """Reuse an existing target top-k reduction to form the regret direction."""
    if top_probs.shape != top_ids.shape or top_probs.ndim != 2:
        raise ValueError("target top-k probabilities and ids must share [rows,k]")
    rows = top_probs.shape[0]
    if draft_token_ids.shape != (rows,) or event_weight.shape != (rows,):
        raise ValueError("candidate and event weights must match target rows")
    ids = draft_token_ids.to(device=top_ids.device, dtype=torch.int64)
    weights = torch.where(
        top_ids.to(torch.int64) == ids.unsqueeze(1),
        torch.zeros_like(top_probs),
        top_probs.to(torch.float32),
    )
    weights /= weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    preferred = (
        weights.unsqueeze(-1)
        * output_weight[top_ids.to(torch.int64)].to(torch.float32)
    ).sum(dim=1)
    candidate = output_weight[ids].to(torch.float32)
    raw = (event_weight.unsqueeze(1) * (preferred - candidate)).sum(dim=0)
    rms = torch.sqrt(raw.square().mean()).clamp_min(1e-8)
    direction = raw / rms
    return torch.where(
        event_weight.sum() > 0.0,
        direction,
        torch.zeros_like(direction),
    )


def apply_direction_to_hidden(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    strength: torch.Tensor,
) -> torch.Tensor:
    """Orthogonal, RMS-preserving MTP-only feature correction."""
    if hidden.ndim != 2 or direction.ndim != 1:
        raise ValueError("hidden must be [B,H] and direction [H]")
    source = hidden.to(torch.float32)
    vector = direction.to(device=source.device, dtype=torch.float32).unsqueeze(0)
    projection = (
        (source * vector).sum(dim=-1, keepdim=True)
        / source.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    orthogonal = vector - projection * source
    source_rms = torch.sqrt(source.square().mean(dim=-1, keepdim=True))
    direction_rms = torch.sqrt(
        orthogonal.square().mean(dim=-1, keepdim=True)
    ).clamp_min(1e-8)
    steered = source + strength * orthogonal / direction_rms * source_rms
    steered = steered * source_rms / torch.sqrt(
        steered.square().mean(dim=-1, keepdim=True)
    ).clamp_min(1e-8)
    return steered.to(hidden.dtype)


def apply_budget_scale(
    target_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
    budget_scale: torch.Tensor,
) -> torch.Tensor:
    """Reduce an Exact-TV allocation without ever creating extra TV."""
    if not (
        target_candidate_probs.shape
        == boosted_candidate_probs.shape
        == budget_scale.shape
    ):
        raise ValueError("budget scalars must share one value per draft head")
    scale = budget_scale.to(
        device=target_candidate_probs.device,
        dtype=torch.float32,
    ).clamp(0.0, 1.0)
    target = target_candidate_probs.to(torch.float32)
    boosted = boosted_candidate_probs.to(torch.float32)
    return target + scale * (boosted - target).clamp_min(0.0)


def _identity_controls(device: torch.device, heads: int) -> RegretRouterControls:
    zeros = torch.zeros(heads, device=device, dtype=torch.float32)
    return RegretRouterControls(zeros, zeros.clone(), torch.ones_like(zeros))


def _fixed_controls(
    config: RegretRouterConfig,
    debt: torch.Tensor,
    device: torch.device,
) -> RegretRouterControls:
    reliability = torch.as_tensor(
        config.head_reliability[: config.expected_draft_tokens],
        device=device,
        dtype=torch.float32,
    )
    gate = (debt.to(torch.float32) / config.debt_reference).clamp(0.0, 1.0)
    tail_risk = 1.0 - reliability
    direction = config.max_direction_strength * gate * (0.5 + 0.5 * tail_risk)
    logit = -config.max_logit_scale * gate * tail_risk
    budget = 1.0 - config.max_budget_reduction * gate * tail_risk
    return RegretRouterControls(direction, logit, budget.clamp(0.0, 1.0))


def _route(
    config: RegretRouterConfig,
    *,
    entropy: torch.Tensor,
    margin: torch.Tensor,
) -> RegretRouterControls:
    root = _STATE.context_root
    direction = _STATE.context_direction
    debt = _STATE.context_debt
    if root is None or direction is None or debt is None:
        device = entropy.device
        return _identity_controls(device, config.expected_draft_tokens)
    if config.mode in {"identity", "collect"}:
        return _identity_controls(root.device, config.expected_draft_tokens)
    if config.mode == "fixed":
        return _fixed_controls(config, debt, root.device)
    if _STATE.router is None:
        raise RuntimeError("learned regret router was not loaded")
    reliability = torch.as_tensor(
        config.head_reliability[: config.expected_draft_tokens],
        device=root.device,
        dtype=torch.float32,
    )
    controls = _STATE.router(
        root.to(torch.float32),
        direction.to(torch.float32),
        debt.to(torch.float32) / config.debt_reference,
        entropy.to(device=root.device, dtype=torch.float32),
        margin.to(device=root.device, dtype=torch.float32),
        reliability,
    )
    return RegretRouterControls(
        controls.direction_strength[0],
        controls.logit_scale[0],
        controls.budget_scale[0],
    )


def _pad_head_feature(
    value: torch.Tensor,
    expected_heads: int,
) -> torch.Tensor:
    """Pad a partial speculative block for the fixed-head router input."""
    if value.ndim != 1:
        raise ValueError("head feature must be one-dimensional")
    if value.shape[0] > expected_heads:
        raise ValueError("head feature is longer than the configured MTP block")
    if value.shape[0] == expected_heads:
        return value
    fill = value[-1] if value.numel() else torch.zeros((), device=value.device)
    padding = fill.expand(expected_heads - value.shape[0])
    return torch.cat((value, padding), dim=0)


def _root_indices(kwargs: dict[str, Any]) -> torch.Tensor:
    indices = kwargs.get("token_indices_to_sample")
    if isinstance(indices, torch.Tensor):
        return indices.to(torch.int64)
    metadata = kwargs.get("common_attn_metadata")
    if metadata is None:
        raise RuntimeError("MTP proposal did not expose attention metadata")
    return (metadata.query_start_loc[1:] - 1).to(torch.int64)


def _propose_with_router_context(self: Any, *args: Any, **kwargs: Any) -> Any:
    original = getattr(_propose_with_router_context, "_remtp_original")
    config = _require_config()
    lm_head = getattr(getattr(self, "model", None), "lm_head", None)
    output_weight = getattr(lm_head, "weight", None)
    if not isinstance(output_weight, torch.Tensor):
        raise RuntimeError("MTP shared output head was not exposed")
    _STATE.output_weight = output_weight.detach()

    target_hidden = kwargs.get("target_hidden_states")
    if not isinstance(target_hidden, torch.Tensor):
        raise RuntimeError("MTP target hidden states were not exposed")
    root = target_hidden.index_select(0, _root_indices(kwargs).to(target_hidden.device))
    if root.shape[0] != 1:
        raise RuntimeError("regret router requires --max-num-seqs 1")
    hidden_size = root.shape[-1]
    if config.mode == "learned":
        assert _STATE.router is not None
        if _STATE.router.architecture.hidden_size != hidden_size:
            raise RuntimeError("router checkpoint hidden size does not match MTP")
        if next(_STATE.router.parameters()).device != root.device:
            _STATE.router.to(root.device)
        _STATE.router.eval()

    zero = torch.zeros(1, device=root.device, dtype=torch.float32)
    direction = _STATE.direction
    if direction is None:
        direction = torch.zeros(hidden_size, device=root.device, dtype=torch.float32)
    debt = _STATE.debt if _STATE.debt is not None else zero
    entropy = _STATE.source_entropy if _STATE.source_entropy is not None else zero
    margin = _STATE.source_margin if _STATE.source_margin is not None else zero
    _STATE.context_root = root.detach()
    _STATE.context_direction = direction.to(root.device).unsqueeze(0).detach()
    _STATE.context_debt = debt.to(root.device).reshape(1).detach()
    _STATE.context_entropy = entropy.to(root.device).reshape(1).detach()
    _STATE.context_margin = margin.to(root.device).reshape(1).detach()
    _STATE.proposal_controls = _route(
        config,
        entropy=_STATE.context_entropy,
        margin=_STATE.context_margin,
    )
    return original(self, *args, **kwargs)


def _router_hidden_hook(hidden: torch.Tensor, proposal_depth: int) -> torch.Tensor:
    controls = _STATE.proposal_controls
    direction = _STATE.context_direction
    if controls is None or direction is None or proposal_depth >= controls.direction_strength.numel():
        return hidden
    return apply_direction_to_hidden(
        hidden,
        direction[0],
        controls.direction_strength[proposal_depth],
    )


def _router_logits_hook(
    logits: torch.Tensor,
    proposal_depth: int,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    del temperatures
    controls = _STATE.proposal_controls
    if controls is None or proposal_depth >= controls.logit_scale.numel():
        return logits
    return logits * torch.exp(controls.logit_scale[proposal_depth]).to(logits.dtype)


def _pre_verification_budget_hook(
    *,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
    hidden_similarity: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    del draft_probs, draft_token_ids, hidden_similarity, sampling_metadata
    config = _require_config()
    if config.mode in {"identity", "collect"}:
        _STATE.verification_entropy = None
        _STATE.verification_margin = None
        _STATE.verification_top_probs = None
        _STATE.verification_top_ids = None
        _STATE.verification_controls = _identity_controls(
            target_probs.device,
            config.expected_draft_tokens,
        )
        return boosted_candidate_probs
    entropy, margin, top_probs, top_ids = target_head_statistics(
        target_probs,
        top_k=config.top_k,
    )
    _STATE.verification_entropy = entropy
    _STATE.verification_margin = margin
    _STATE.verification_top_probs = top_probs
    _STATE.verification_top_ids = top_ids
    controls = _route(
        config,
        entropy=_pad_head_feature(
            entropy,
            config.expected_draft_tokens,
        ).unsqueeze(0),
        margin=_pad_head_feature(
            margin,
            config.expected_draft_tokens,
        ).unsqueeze(0),
    )
    _STATE.verification_controls = controls
    rows = target_candidate_probs.shape[0]
    scale = controls.budget_scale[:rows].to(target_candidate_probs.device)
    return apply_budget_scale(
        target_candidate_probs,
        boosted_candidate_probs,
        scale,
    )


def _collector_record(
    *,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_entropy: torch.Tensor,
    target_margin: torch.Tensor,
    allocated_tv: torch.Tensor,
    strict_acceptance: torch.Tensor,
    relaxed_acceptance: torch.Tensor,
    accepted: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    draft_candidate_probs: torch.Tensor,
    reallocation_fraction: torch.Tensor,
    expected_regret: torch.Tensor,
    target_top_probs: torch.Tensor,
    target_top_ids: torch.Tensor,
) -> None:
    if _COLLECTOR is None:
        return
    config = _require_config()
    root = _STATE.context_root
    direction = _STATE.context_direction
    debt = _STATE.context_debt
    source_entropy = _STATE.context_entropy
    source_margin = _STATE.context_margin
    if any(
        item is None
        for item in (root, direction, debt, source_entropy, source_margin)
    ):
        return
    if target_probs.shape[0] != config.expected_draft_tokens:
        # Keep the on-disk tensor schema dense. Short final blocks are rare and
        # do not justify padding ambiguous verifier outcomes into training.
        return
    if not _STATE.request_id:
        raise RuntimeError("collector request_id was not initialized")
    p_values = target_top_probs
    p_ids = target_top_ids
    q_values, q_ids = draft_probs.to(torch.float32).topk(
        min(config.top_k, draft_probs.shape[-1]), dim=-1
    )
    p_on_q = target_probs.to(torch.float32).gather(1, q_ids)
    q_on_p = draft_probs.to(torch.float32).gather(1, p_ids)
    output_weight = _STATE.output_weight
    assert output_weight is not None
    vector = direction[0].to(output_weight.device, dtype=torch.float32)
    p_delta = (
        output_weight[p_ids].to(torch.float32) * vector.view(1, 1, -1)
    ).sum(dim=-1)
    q_delta = (
        output_weight[q_ids].to(torch.float32) * vector.view(1, 1, -1)
    ).sum(dim=-1)
    record = {
        "request_id": _STATE.request_id,
        "root_hidden": root[0].detach().to("cpu", dtype=torch.float16),
        "regret_direction": direction[0].detach().to("cpu", dtype=torch.float16),
        "regret_debt": debt.detach().to("cpu", dtype=torch.float32),
        "source_entropy": source_entropy.detach().to("cpu", dtype=torch.float32),
        "source_margin": source_margin.detach().to("cpu", dtype=torch.float32),
        "target_entropy": target_entropy.detach().to("cpu", dtype=torch.float32),
        "target_margin": target_margin.detach().to("cpu", dtype=torch.float32),
        "p_top_ids": p_ids.detach().to("cpu", dtype=torch.int32),
        "p_top_probs": p_values.detach().to("cpu", dtype=torch.float32),
        "q_top_ids": q_ids.detach().to("cpu", dtype=torch.int32),
        "q_top_probs": q_values.detach().to("cpu", dtype=torch.float32),
        "p_on_q_top": p_on_q.detach().to("cpu", dtype=torch.float32),
        "q_on_p_top": q_on_p.detach().to("cpu", dtype=torch.float32),
        "p_direction_delta": p_delta.detach().to("cpu", dtype=torch.float32),
        "q_direction_delta": q_delta.detach().to("cpu", dtype=torch.float32),
        "draft_token_ids": draft_token_ids.detach().to("cpu", dtype=torch.int32),
        "target_candidate_probs": target_candidate_probs.detach().to(
            "cpu", dtype=torch.float32
        ),
        "draft_candidate_probs": draft_candidate_probs.detach().to(
            "cpu", dtype=torch.float32
        ),
        "allocated_tv": allocated_tv.detach().to("cpu", dtype=torch.float32),
        "strict_acceptance": strict_acceptance.detach().to("cpu", dtype=torch.float32),
        "relaxed_acceptance": relaxed_acceptance.detach().to("cpu", dtype=torch.float32),
        "accepted": accepted.detach().to("cpu"),
        "reallocation_fraction": reallocation_fraction.detach().to(
            "cpu", dtype=torch.float32
        ),
        "expected_regret": expected_regret.detach().to(
            "cpu", dtype=torch.float32
        ),
        "head_reliability": torch.as_tensor(
            config.head_reliability[: config.expected_draft_tokens],
            dtype=torch.float32,
        ),
    }
    _COLLECTOR.add(record)


def _post_verification_regret_hook(
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
    del hidden_similarity, sampling_metadata
    global _DIAGNOSTIC_EMITTED

    config = _require_config()
    rows = draft_token_ids.shape[0]
    if not 1 <= rows <= config.expected_draft_tokens:
        raise RuntimeError("router observed an unexpected draft block length")
    accepted = accepted_prefix_mask(output_token_ids, draft_token_ids)
    ids = draft_token_ids.to(target_probs.device, dtype=torch.int64)
    row_ids = torch.arange(rows, device=target_probs.device)
    q_y = draft_probs[row_ids, ids].to(torch.float32).clamp_min(1e-30)
    relaxed_acceptance = torch.minimum(
        torch.ones_like(q_y),
        boosted_candidate_probs.to(torch.float32) / q_y,
    )
    entropy = _STATE.verification_entropy
    margin = _STATE.verification_margin
    top_probs = _STATE.verification_top_probs
    top_ids = _STATE.verification_top_ids
    if any(value is None for value in (entropy, margin, top_probs, top_ids)):
        entropy, margin, top_probs, top_ids = target_head_statistics(
            target_probs,
            top_k=config.top_k,
        )
    assert entropy is not None and margin is not None
    assert top_probs is not None and top_ids is not None
    concentration = 1.0 - entropy
    direct_capacity = torch.minimum(
        cactus_tv_increment(
            target_candidate_probs,
            delta=config.cactus_delta,
        ),
        (q_y - target_candidate_probs).clamp_min(0.0),
    )
    reallocated = (allocated_tv - direct_capacity).clamp_min(0.0)
    reallocation_fraction = reallocated / allocated_tv.clamp_min(1e-30)
    event_weight, responsibility = expected_regret_debt(
        accepted,
        allocated_tv,
        strict_acceptance,
        relaxed_acceptance,
        concentration,
        reallocation_fraction,
    )

    _collector_record(
        target_probs=target_probs,
        draft_probs=draft_probs,
        draft_token_ids=draft_token_ids,
        target_entropy=entropy,
        target_margin=margin,
        allocated_tv=allocated_tv,
        strict_acceptance=strict_acceptance,
        relaxed_acceptance=relaxed_acceptance,
        accepted=accepted,
        target_candidate_probs=target_candidate_probs,
        draft_candidate_probs=q_y,
        reallocation_fraction=reallocation_fraction,
        expected_regret=event_weight,
        target_top_probs=top_probs,
        target_top_ids=top_ids,
    )

    output_weight = _STATE.output_weight
    if output_weight is None:
        raise RuntimeError("router did not capture the shared output head")
    direction = regret_direction_from_top_mass(
        top_probs,
        top_ids,
        draft_token_ids,
        event_weight,
        output_weight,
    )
    accepted_count = accepted.to(torch.int64).sum()
    rejected = accepted_count < rows
    reset = torch.where(
        rejected,
        torch.tensor(config.rejection_reset, device=target_probs.device),
        torch.ones((), device=target_probs.device),
    )
    old_debt = _STATE.debt
    if old_debt is None:
        old_debt = torch.zeros(1, device=target_probs.device)
    decay = torch.pow(
        torch.tensor(config.debt_decay, device=target_probs.device),
        accepted_count.to(torch.float32),
    )
    new_debt = event_weight.sum().reshape(1)
    _STATE.debt = (old_debt.to(target_probs.device) * decay + new_debt) * reset
    _STATE.direction = direction.detach() * reset
    weights = accepted.to(torch.float32)
    normalizer = weights.sum().clamp_min(1.0)
    _STATE.source_entropy = ((entropy * weights).sum() / normalizer).reshape(1)
    _STATE.source_margin = ((margin * weights).sum() / normalizer).reshape(1)

    _AUDIT.rounds += 1
    _AUDIT.accepted += accepted.sum().item()
    _AUDIT.strict_expected += (strict_acceptance * accepted).sum().item()
    _AUDIT.relaxed_expected += (relaxed_acceptance * accepted).sum().item()
    _AUDIT.debt_added += new_debt.item()
    _AUDIT.allocated_tv += allocated_tv.sum().item()
    controls = _STATE.verification_controls or _STATE.proposal_controls
    if controls is not None:
        _AUDIT.budget_scale += controls.budget_scale[:rows].mean().item()
        _AUDIT.direction_strength += controls.direction_strength[:rows].mean().item()
        _AUDIT.logit_scale += controls.logit_scale[:rows].abs().mean().item()

    if config.diagnostics and not _DIAGNOSTIC_EMITTED:
        print(
            "[ReMTP][RegretRouter][diagnostic] "
            f"mode={config.mode} accepted={accepted.tolist()} "
            f"responsibility={responsibility.tolist()} "
            f"event_debt={event_weight.tolist()} "
            f"next_debt={_STATE.debt.item():.6f}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True
    if config.audit_interval > 0 and _AUDIT.rounds % config.audit_interval == 0:
        denominator = float(config.audit_interval)
        print(
            "[ReMTP][RegretRouter][audit] "
            f"rounds={_AUDIT.rounds-config.audit_interval+1}-{_AUDIT.rounds} "
            f"accepted={_AUDIT.accepted/denominator:.4f} "
            f"strict_expected={_AUDIT.strict_expected/denominator:.4f} "
            f"relaxed_expected={_AUDIT.relaxed_expected/denominator:.4f} "
            f"debt_added={_AUDIT.debt_added/denominator:.6f} "
            f"allocated_TV={_AUDIT.allocated_tv/denominator:.6f} "
            f"budget_scale={_AUDIT.budget_scale/denominator:.6f} "
            f"direction_strength={_AUDIT.direction_strength/denominator:.6f} "
            f"abs_logit_scale={_AUDIT.logit_scale/denominator:.6f}",
            flush=True,
        )
        _AUDIT.reset_values()


def _sample_tokens_with_request_reset(self: Any, *args: Any, **kwargs: Any) -> Any:
    global _REQUEST_COUNTER

    state = getattr(self, "execute_model_state", None)
    if state is None or state.spec_decode_metadata is None:
        if _COLLECTOR is not None:
            _COLLECTOR.checkpoint()
        _REQUEST_COUNTER += 1
        config = _require_config()
        collection_id = config.collection_id or f"engine-{os.getpid()}"
        _STATE.reset_request(f"{collection_id}:{_REQUEST_COUNTER}")
    original = getattr(_sample_tokens_with_request_reset, "_remtp_original")
    return original(self, *args, **kwargs)


def _require_config() -> RegretRouterConfig:
    if _CONFIG is None:
        raise RuntimeError("regret router has not been installed")
    return _CONFIG


def install_regret_router() -> None:
    """Install expected regret state and bounded proposal/budget controls."""
    global _CONFIG, _COLLECTOR

    config = RegretRouterConfig.from_env()
    _CONFIG = config
    if config.mode == "learned":
        router, metadata = load_regret_router_checkpoint(config.checkpoint)
        if router.architecture.num_heads != config.expected_draft_tokens:
            raise ValueError("router checkpoint head count mismatch")
        _STATE.router = router.eval()
        print(
            f"[ReMTP][RegretRouter] checkpoint={config.checkpoint} "
            f"metadata={metadata}",
            flush=True,
        )
    if config.mode == "collect":
        _COLLECTOR = RouterShardWriter(
            Path(config.collector_dir),
            config.collector_shard_size,
        )
        atexit.register(_COLLECTOR.flush)

    from remtp.probabilistic_mtp import (
        set_draft_hidden_hook,
        set_draft_logits_hook,
    )
    from remtp.target_anchored_mtp import (
        set_post_verification_hook,
        set_pre_verification_hook,
    )

    set_draft_hidden_hook(_router_hidden_hook)
    set_draft_logits_hook(_router_logits_hook)
    set_pre_verification_hook(_pre_verification_budget_hook)
    set_post_verification_hook(_post_verification_regret_hook)

    eagle_module = importlib.import_module(_EAGLE_MODULE)
    proposer_cls = eagle_module.EagleProposer
    current_propose = proposer_cls.propose
    if not getattr(current_propose, "_remtp_regret_router", False):
        _propose_with_router_context._remtp_regret_router = True
        _propose_with_router_context._remtp_original = current_propose
        proposer_cls.propose = _propose_with_router_context

    runner_module = importlib.import_module(_GPU_RUNNER_MODULE)
    runner_cls = runner_module.GPUModelRunner
    current_sample = runner_cls.sample_tokens
    if not getattr(current_sample, "_remtp_regret_router_reset", False):
        _sample_tokens_with_request_reset._remtp_regret_router_reset = True
        _sample_tokens_with_request_reset._remtp_original = current_sample
        runner_cls.sample_tokens = _sample_tokens_with_request_reset

    print(
        "[ReMTP][RegretRouter] enabled "
        f"mode={config.mode} draft_tokens={config.expected_draft_tokens} "
        f"top_k={config.top_k} cactus_delta={config.cactus_delta:g} "
        f"debt_decay={config.debt_decay:g} "
        f"rejection_reset={config.rejection_reset:g} "
        f"direction_cap={config.max_direction_strength:g} "
        f"logit_cap={config.max_logit_scale:g} "
        f"budget_reduction_cap={config.max_budget_reduction:g}",
        flush=True,
    )
