"""Training-free calibration of native-MTP proposal distributions.

The target distribution is deliberately absent from every online transform.
The calibrated proposal ``q_tilde`` is sampled by the MTP proposer and the
same tensor is passed to vLLM's strict rejection and residual sampler.  This
preserves the target distribution while changing proposal efficiency only.

Trace mode is intentionally synchronization-heavy.  It evaluates a finite
calibration grid on GPU and writes only compact scalar overlaps, never a full
vocabulary distribution.  It must not be enabled for throughput reporting.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_GPU_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
_TRACE_ROUND = 0
_TRACE_REQUEST = -1
_TRACE_LAST_LENGTH: int | None = None
_TRACE_ACTIVE_REQUEST_ID: str | None = None
_TRACE_SEEN_REQUEST_ID: str | None = None
_TRACE_ACTIVE_OUTPUT_LENGTH = 0
_DIAGNOSTIC_EMITTED = False


@dataclass(frozen=True)
class HeadTransform:
    """One proposal-only transform for one recursive MTP head."""

    kind: str = "identity"
    temperature: float = 1.0
    top_k: int = 3
    strength: float = 0.0

    def validate(self) -> None:
        if self.kind not in {
            "identity",
            "temperature",
            "topk_redistribute",
            "topk_concentrate",
            "topk_uniform_mix",
        }:
            raise ValueError(f"unknown proposal transform: {self.kind}")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if self.top_k not in {2, 3}:
            raise ValueError("top_k must be 2 or 3")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")

    @property
    def name(self) -> str:
        if self.kind == "identity":
            return "identity"
        if self.kind == "temperature":
            return f"temperature:{self.temperature:g}"
        return f"{self.kind}:k{self.top_k}:s{self.strength:g}"


@dataclass(frozen=True)
class ProposalCalibrationConfig:
    heads: tuple[HeadTransform, ...]

    def validate(self) -> None:
        if not self.heads:
            raise ValueError("at least one MTP head transform is required")
        for transform in self.heads:
            transform.validate()

    def head(self, depth: int) -> HeadTransform:
        return self.heads[min(max(depth, 0), len(self.heads) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "head_transforms": [asdict(item) for item in self.heads],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProposalCalibrationConfig":
        rows = payload.get("head_transforms")
        if not isinstance(rows, list) or not rows:
            raise ValueError("config requires a non-empty head_transforms list")
        config = cls(tuple(HeadTransform(**dict(row)) for row in rows))
        config.validate()
        return config


def identity_config(heads: int = 6) -> ProposalCalibrationConfig:
    return ProposalCalibrationConfig(tuple(HeadTransform() for _ in range(heads)))


def load_config(path: str | Path | None, heads: int = 6) -> ProposalCalibrationConfig:
    if path is None or str(path).strip() == "":
        return identity_config(heads)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "selected_config" in payload:
        payload = payload["selected_config"]
    return ProposalCalibrationConfig.from_dict(payload)


def normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.to(torch.float32).clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def apply_probability_transform(
    probs: torch.Tensor,
    transform: HeadTransform,
) -> torch.Tensor:
    """Apply a target-free transform to one or more proposal rows."""
    transform.validate()
    q = normalize_probs(probs)
    if transform.kind == "identity" or (
        transform.strength == 0.0 and transform.kind.startswith("topk")
    ):
        return q
    if transform.kind == "temperature":
        if abs(transform.temperature - 1.0) <= 1e-12:
            return q
        powered = q.clamp_min(1e-30).pow(1.0 / transform.temperature)
        return normalize_probs(powered)

    values, ids = torch.topk(q, k=transform.top_k, dim=-1, sorted=True)
    output = q.clone()
    if transform.kind == "topk_redistribute":
        top_mass = values.sum(dim=-1, keepdim=True)
        replacement = (
            (1.0 - transform.strength) * values
            + transform.strength * top_mass / float(transform.top_k)
        )
        output.scatter_(1, ids, replacement)
    elif transform.kind == "topk_concentrate":
        replacement = values.clone()
        moved = transform.strength * values[:, 1:].sum(dim=-1)
        replacement[:, 0] += moved
        replacement[:, 1:] *= 1.0 - transform.strength
        output.scatter_(1, ids, replacement)
    elif transform.kind == "topk_uniform_mix":
        output.mul_(1.0 - transform.strength)
        addition = torch.full_like(values, transform.strength / transform.top_k)
        output.scatter_add_(1, ids, addition)
    else:  # guarded by validate; helps static type checkers.
        raise AssertionError(transform.kind)
    return normalize_probs(output)


def overlap_mass(target: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
    """Return per-row strict speculative acceptance, sum_x min(P,Q)."""
    p = normalize_probs(target)
    q = normalize_probs(proposal)
    if p.shape != q.shape:
        raise ValueError("target and proposal shapes must match")
    return torch.minimum(p, q).sum(dim=-1)


def strict_output_distribution(target: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
    """Analytic one-token output law of strict rejection sampling.

    This helper is used by tests.  It explicitly combines accepted proposal
    mass with the positive residual and should equal ``target`` up to floating
    point error for every normalized proposal, including calibrated ones.
    """
    p = normalize_probs(target)
    q = normalize_probs(proposal)
    accepted = torch.minimum(p, q)
    rejected_mass = 1.0 - accepted.sum(dim=-1, keepdim=True)
    residual = torch.relu(p - q)
    residual_total = residual.sum(dim=-1, keepdim=True)
    fallback = torch.where(
        residual_total > 1e-12,
        residual / residual_total.clamp_min(1e-30),
        p,
    )
    return accepted + rejected_mass.clamp_min(0.0) * fallback


def default_search_transforms() -> tuple[HeadTransform, ...]:
    rows: list[HeadTransform] = [HeadTransform()]
    rows.extend(
        HeadTransform(kind="temperature", temperature=value)
        for value in (
            0.30,
            0.40,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.90,
            1.10,
            1.25,
            1.50,
            2.00,
        )
    )
    for kind in (
        "topk_redistribute",
        "topk_concentrate",
        "topk_uniform_mix",
    ):
        for top_k in (2, 3):
            for strength in (0.02, 0.05, 0.10, 0.20, 0.30):
                rows.append(
                    HeadTransform(kind=kind, top_k=top_k, strength=strength)
                )
    return tuple(rows)


def _request_index(sampling_metadata: Any) -> tuple[int, int]:
    global _TRACE_LAST_LENGTH, _TRACE_REQUEST, _TRACE_SEEN_REQUEST_ID
    if _TRACE_ACTIVE_REQUEST_ID is not None:
        if _TRACE_ACTIVE_REQUEST_ID != _TRACE_SEEN_REQUEST_ID:
            _TRACE_REQUEST += 1
            _TRACE_SEEN_REQUEST_ID = _TRACE_ACTIVE_REQUEST_ID
        _TRACE_LAST_LENGTH = _TRACE_ACTIVE_OUTPUT_LENGTH
        return _TRACE_REQUEST, _TRACE_ACTIVE_OUTPUT_LENGTH

    output_rows = getattr(sampling_metadata, "output_token_ids", None)
    output_length = len(output_rows[0]) if output_rows else 0
    if _TRACE_LAST_LENGTH is None or (
        output_length > 0 and output_length <= _TRACE_LAST_LENGTH
    ):
        _TRACE_REQUEST += 1
    _TRACE_LAST_LENGTH = output_length
    return _TRACE_REQUEST, output_length


def _trace_runner_sample(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Expose stable request identity to the optional decision trace."""
    global _TRACE_ACTIVE_OUTPUT_LENGTH, _TRACE_ACTIVE_REQUEST_ID

    original = getattr(_trace_runner_sample, "_remtp_original")
    request_ids = list(getattr(self.input_batch, "req_ids", ()))
    if len(request_ids) > 1:
        raise RuntimeError("proposal trace requires --max-num-seqs 1")
    if request_ids:
        request_id = request_ids[0]
        _TRACE_ACTIVE_REQUEST_ID = str(request_id)
        request_state = getattr(self, "requests", {}).get(request_id)
        output_ids = getattr(request_state, "output_token_ids", ())
        _TRACE_ACTIVE_OUTPUT_LENGTH = len(output_ids)
    else:
        _TRACE_ACTIVE_REQUEST_ID = None
        _TRACE_ACTIVE_OUTPUT_LENGTH = 0
    return original(self, *args, **kwargs)


def _trace_round(
    *,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    output_token_ids: torch.Tensor,
    sampling_metadata: Any,
    path: str,
) -> None:
    global _TRACE_ROUND
    _TRACE_ROUND += 1
    p = torch.softmax(target_logits.to(torch.float32), dim=-1)
    q = normalize_probs(draft_probs)
    rows = q.shape[0]
    row_ids = torch.arange(rows, device=q.device)
    candidate_ids = draft_token_ids.to(torch.int64)
    q_y = q[row_ids, candidate_ids]
    p_y = p[row_ids, candidate_ids]
    strict = torch.minimum(torch.ones_like(q_y), p_y / q_y.clamp_min(1e-30))

    trace_mode = os.getenv("REMTP_PC_TRACE_MODE", "search").strip().lower()
    top_count = min(
        max(2, int(os.getenv("REMTP_PC_TRACE_TOPK", "5"))),
        q.shape[-1],
    )
    q_values, q_ids = torch.topk(q, k=top_count, dim=-1)
    p_top_values, p_top_ids = torch.topk(p, k=top_count, dim=-1)
    p_at_q_top = p.gather(1, q_ids)
    entropy = -(q * torch.log(q.clamp_min(1e-30))).sum(dim=-1)
    q_margin = torch.log(q_values[:, 0].clamp_min(1e-30)) - torch.log(
        q_values[:, 1].clamp_min(1e-30)
    )

    strategy_overlap: dict[str, list[float]] = {}
    if trace_mode == "search":
        for transform in default_search_transforms():
            calibrated = apply_probability_transform(q, transform)
            strategy_overlap[transform.name] = (
                overlap_mass(p, calibrated).detach().cpu().tolist()
            )

    try:
        from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
    except ImportError:
        PLACEHOLDER_TOKEN_ID = -1
    valid = int((output_token_ids[0] != PLACEHOLDER_TOKEN_ID).sum().item())
    accepted_drafts = max(0, min(rows, valid - 1))
    committed_ids = output_token_ids[0, :valid].to(torch.int64)
    rejected_head = accepted_drafts if accepted_drafts < rows else None
    decisions = [
        (
            "accepted"
            if index < accepted_drafts
            else "rejected"
            if index == rejected_head
            else "skipped"
        )
        for index in range(rows)
    ]
    request_index, output_length = _request_index(sampling_metadata)

    record = {
        "version": 2,
        "trace_mode": trace_mode,
        "dataset": os.getenv("REMTP_PC_TRACE_DATASET", "unknown"),
        "round_index": _TRACE_ROUND,
        "request_index": request_index,
        "output_length_before_round": output_length,
        "num_heads": rows,
        "target_forward_calls": 1,
        "target_verified_nodes": rows,
        "draft_tokens": candidate_ids.detach().cpu().tolist(),
        "accepted_drafts": accepted_drafts,
        "decisions": decisions,
        "rejected_head": None if rejected_head is None else rejected_head + 1,
        "committed_tokens": committed_ids.detach().cpu().tolist(),
        "anchor_kind": "bonus" if rejected_head is None else "recovery",
        "p_y": p_y.detach().cpu().tolist(),
        "q_y": q_y.detach().cpu().tolist(),
        "sampled_strict_acceptance": strict.detach().cpu().tolist(),
        "native_overlap": overlap_mass(p, q).detach().cpu().tolist(),
        "q_entropy": entropy.detach().cpu().tolist(),
        "q_log_margin": q_margin.detach().cpu().tolist(),
        "q_top_ids": q_ids.detach().cpu().tolist(),
        "q_top_probs": q_values.detach().cpu().tolist(),
        "p_top1_ids": p_top_ids[:, 0].detach().cpu().tolist(),
        "p_top_ids": p_top_ids.detach().cpu().tolist(),
        "p_top_probs": p_top_values.detach().cpu().tolist(),
        "p_at_q_top": p_at_q_top.detach().cpu().tolist(),
        "strategy_overlap": strategy_overlap,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _traced_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    original = getattr(_traced_rejection_sample, "_remtp_original")
    output = original(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
    )
    trace_path = os.getenv("REMTP_PC_TRACE_PATH", "").strip()
    if trace_path and draft_probs is not None and draft_token_ids.numel() > 0:
        _trace_round(
            draft_token_ids=draft_token_ids,
            draft_probs=draft_probs,
            target_logits=target_logits,
            output_token_ids=output,
            sampling_metadata=sampling_metadata,
            path=trace_path,
        )
    return output


def _make_temperature_hook(config: ProposalCalibrationConfig):
    def hook(temperatures: torch.Tensor, depth: int) -> torch.Tensor:
        transform = config.head(depth)
        if transform.kind != "temperature":
            return temperatures
        # This is folded into the one temperature division already required by
        # the native proposal softmax.  It adds no vocabulary-sized kernel.
        return temperatures * transform.temperature

    return hook


def _make_probs_hook(config: ProposalCalibrationConfig):
    def hook(
        probs: torch.Tensor,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        transform = config.head(depth)
        if transform.kind in {"identity", "temperature"}:
            return probs
        return apply_probability_transform(probs, transform)

    return hook


def install_proposal_calibration() -> ProposalCalibrationConfig:
    """Install target-free Q calibration and optional compact trace capture."""
    global _DIAGNOSTIC_EMITTED
    from remtp.probabilistic_mtp import (
        set_draft_probs_hook,
        set_draft_temperature_hook,
    )

    config_path = os.getenv("REMTP_PC_CONFIG", "").strip()
    heads = int(os.getenv("MTP_TOKENS", "6"))
    config = load_config(config_path or None, heads=heads)
    set_draft_temperature_hook(_make_temperature_hook(config))
    if any(
        item.kind
        in {"topk_redistribute", "topk_concentrate", "topk_uniform_mix"}
        for item in config.heads
    ):
        set_draft_probs_hook(_make_probs_hook(config))
    else:
        # The selected static configuration is temperature-only. Avoid even a
        # Python probability-hook dispatch on the hot proposal path.
        set_draft_probs_hook(None)

    trace_path = os.getenv("REMTP_PC_TRACE_PATH", "").strip()
    if trace_path:
        module = importlib.import_module(_REJECTION_MODULE)
        current = module.rejection_sample
        if not getattr(current, "_remtp_pc_trace", False):
            _traced_rejection_sample._remtp_pc_trace = True
            _traced_rejection_sample._remtp_original = current
            module.rejection_sample = _traced_rejection_sample
        runner_module = importlib.import_module(_GPU_RUNNER_MODULE)
        runner_cls = runner_module.GPUModelRunner
        current_sample = runner_cls._sample
        if not getattr(current_sample, "_remtp_pc_trace_request", False):
            _trace_runner_sample._remtp_pc_trace_request = True
            _trace_runner_sample._remtp_original = current_sample
            runner_cls._sample = _trace_runner_sample

    if not _DIAGNOSTIC_EMITTED:
        print(
            "[ReMTP][ProposalCalibration] "
            f"config={config_path or 'identity'} "
            f"heads={[item.name for item in config.heads]} "
            f"trace={trace_path or 'off'}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True
    return config


def transforms_by_name(
    rows: Iterable[HeadTransform] | None = None,
) -> dict[str, HeadTransform]:
    values = default_search_transforms() if rows is None else tuple(rows)
    return {item.name: item for item in values}
