"""Minimal request-agnostic gate between Cactus and Block Shield.

The gate is deliberately limited to one logistic unit over block scalars that
are already available during verification. Calibration labels are
counterfactual: Shield is positive only when it reduces target-opposed mass
while spending no more than a fixed fraction of the acceptance-length surplus
created by joint Block Verification on the same MTP block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from remtp.block_verification import (
    block_verification_state,
    expected_longest_accepted_prefix,
    prefix_joint_probability,
)


DEFAULT_DRAFT_DEPTH = 6


def _feature_names(depth: int = DEFAULT_DRAFT_DEPTH) -> tuple[str, ...]:
    names: list[str] = []
    for group in (
        "strict_acceptance",
        "target_log_gap_scaled",
        "cactus_prefix_survival",
        "shield_prefix_survival",
        "cactus_repair_indicator",
        "risk_pruned_mass",
        "active_head",
    ):
        names.extend(f"{group}_{index + 1}" for index in range(depth))
    names.extend(
        (
            "mean_strict_acceptance",
            "min_strict_acceptance",
            "max_target_log_gap_scaled",
            "repair_fraction",
            "mean_risk_pruned_mass",
            "mean_prefix_survival_loss",
            "max_prefix_survival_loss",
            "mean_cactus_over_q_mass",
        )
    )
    return tuple(names)


FEATURE_NAMES = _feature_names()


def _pad(values: torch.Tensor, depth: int) -> torch.Tensor:
    if values.shape[0] > depth:
        raise ValueError("block is deeper than the configured gate")
    if values.shape[0] == depth:
        return values
    return torch.cat(
        (
            values,
            torch.zeros(
                depth - values.shape[0],
                device=values.device,
                dtype=values.dtype,
            ),
        )
    )


def block_gate_features(
    target_candidate_probs: torch.Tensor,
    draft_candidate_probs: torch.Tensor,
    target_log_gaps: torch.Tensor,
    cactus_candidate_probs: torch.Tensor,
    shield_candidate_probs: torch.Tensor,
    *,
    expected_depth: int = DEFAULT_DRAFT_DEPTH,
) -> torch.Tensor:
    """Build the fixed, vocabulary-free logistic-gate feature vector."""
    vectors = (
        target_candidate_probs,
        draft_candidate_probs,
        target_log_gaps,
        cactus_candidate_probs,
        shield_candidate_probs,
    )
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("gate inputs must have shape [draft_depth]")
    if len({vector.shape[0] for vector in vectors}) != 1:
        raise ValueError("gate inputs must have the same draft depth")
    rows = target_candidate_probs.shape[0]
    if rows < 1 or rows > expected_depth:
        raise ValueError("unexpected draft depth")

    q_y = draft_candidate_probs.clamp_min(1e-30)
    strict = torch.minimum(
        torch.ones_like(target_candidate_probs),
        target_candidate_probs / q_y,
    )
    cactus_prefix = prefix_joint_probability(cactus_candidate_probs, q_y)
    shield_prefix = prefix_joint_probability(shield_candidate_probs, q_y)
    previous_cactus = torch.cat(
        (torch.ones_like(cactus_prefix[:1]), cactus_prefix[:-1])
    )
    repair = (
        (cactus_candidate_probs > q_y)
        & (previous_cactus < 1.0 - 1e-12)
    ).to(torch.float32)
    risk_pruned = (
        cactus_candidate_probs - shield_candidate_probs
    ).clamp_min(0.0)
    prefix_loss = (cactus_prefix - shield_prefix).clamp_min(0.0)
    over_q = (cactus_candidate_probs - q_y).clamp_min(0.0)
    gap_scaled = target_log_gaps.clamp(0.0, 10.0) / 10.0
    active = torch.ones_like(strict)

    per_head = torch.cat(
        tuple(
            _pad(vector.to(torch.float32), expected_depth)
            for vector in (
                strict,
                gap_scaled,
                cactus_prefix,
                shield_prefix,
                repair,
                risk_pruned,
                active,
            )
        )
    )
    aggregate = torch.stack(
        (
            strict.mean(),
            strict.amin(),
            gap_scaled.amax(),
            repair.mean(),
            risk_pruned.mean(),
            prefix_loss.mean(),
            prefix_loss.amax(),
            over_q.mean(),
        )
    ).to(torch.float32)
    features = torch.cat((per_head, aggregate))
    expected_features = 7 * expected_depth + 8
    if features.shape != (expected_features,):
        raise RuntimeError("unexpected block-gate feature shape")
    return features


def candidate_shift_distribution(
    target_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    candidate_probs: torch.Tensor,
) -> torch.Tensor:
    """Raise each drafted token and proportionally shrink all other tokens."""
    if target_probs.ndim != 2:
        raise ValueError("target_probs must have shape [rows,vocab]")
    ids = draft_token_ids.to(device=target_probs.device, dtype=torch.int64)
    rows = torch.arange(ids.shape[0], device=target_probs.device)
    original = target_probs[rows, ids]
    candidate = torch.maximum(candidate_probs, original).clamp(max=1.0)
    scale = ((1.0 - candidate) / (1.0 - original)).nan_to_num(
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    shifted = target_probs.to(torch.float32) * scale.unsqueeze(-1)
    shifted.scatter_(1, ids.unsqueeze(1), candidate.unsqueeze(1))
    return shifted / shifted.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def counterfactual_gate_record(
    *,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    draft_candidate_probs: torch.Tensor,
    target_log_gaps: torch.Tensor,
    cactus_candidate_probs: torch.Tensor,
    shield_candidate_probs: torch.Tensor,
    safe_tolerance: float = 1e-6,
    surplus_spend_fraction: float = 0.5,
) -> dict[str, Any]:
    """Construct one exact counterfactual training record."""
    if not 0.0 <= surplus_spend_fraction <= 1.0:
        raise ValueError("surplus_spend_fraction must be in [0,1]")
    features = block_gate_features(
        target_candidate_probs,
        draft_candidate_probs,
        target_log_gaps,
        cactus_candidate_probs,
        shield_candidate_probs,
    )
    cactus_probs = candidate_shift_distribution(
        target_probs,
        draft_token_ids,
        cactus_candidate_probs,
    )
    shield_probs = candidate_shift_distribution(
        target_probs,
        draft_token_ids,
        shield_candidate_probs,
    )
    cactus_state = block_verification_state(
        cactus_probs,
        draft_probs,
        draft_token_ids,
        assume_normalized=True,
    )
    shield_state = block_verification_state(
        shield_probs,
        draft_probs,
        draft_token_ids,
        assume_normalized=True,
    )
    cactus_expected = expected_longest_accepted_prefix(
        cactus_state.subblock_acceptance_probability
    )
    shield_expected = expected_longest_accepted_prefix(
        shield_state.subblock_acceptance_probability
    )
    local_cactus_acceptance = torch.minimum(
        torch.ones_like(cactus_candidate_probs),
        cactus_candidate_probs / draft_candidate_probs.clamp_min(1e-30),
    )
    tokenwise_expected = torch.cumprod(
        local_cactus_acceptance,
        dim=0,
    ).sum()
    block_surplus = (cactus_expected - tokenwise_expected).clamp_min(0.0)
    allowed_loss = surplus_spend_fraction * block_surplus
    target_opposition = target_log_gaps.clamp(0.0, 10.0) / 10.0
    risk_reduction = (
        (cactus_candidate_probs - shield_candidate_probs).clamp_min(0.0)
        * target_opposition
    ).sum()
    shield_safe = (
        (shield_expected + allowed_loss + safe_tolerance >= cactus_expected)
        & (risk_reduction > 1e-12)
    )
    return {
        "feature_names": list(FEATURE_NAMES),
        "features": features.detach().cpu().tolist(),
        "label": int(shield_safe.item()),
        "cactus_expected_length": float(cactus_expected.item()),
        "shield_expected_length": float(shield_expected.item()),
        "expected_length_delta": float(
            (shield_expected - cactus_expected).item()
        ),
        "tokenwise_cactus_expected_length": float(tokenwise_expected.item()),
        "block_verification_surplus": float(block_surplus.item()),
        "allowed_surplus_spend": float(allowed_loss.item()),
        "surplus_spend_fraction": surplus_spend_fraction,
        "risk_reduction": float(risk_reduction.item()),
    }


@dataclass(frozen=True)
class LogisticBlockGate:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weight: tuple[float, ...]
    bias: float
    threshold: float

    def validate(self) -> None:
        width = len(FEATURE_NAMES)
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("block-gate feature schema does not match runtime")
        if not (
            len(self.mean) == len(self.scale) == len(self.weight) == width
        ):
            raise ValueError("block-gate vector width does not match runtime")
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("block-gate scales must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("block-gate threshold must be in [0,1]")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticBlockGate":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            feature_names=tuple(payload["feature_names"]),
            mean=tuple(float(value) for value in payload["mean"]),
            scale=tuple(float(value) for value in payload["scale"]),
            weight=tuple(float(value) for value in payload["weight"]),
            bias=float(payload["bias"]),
            threshold=float(payload["threshold"]),
        )
        model.validate()
        return model

    def tensors(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate()
        return (
            torch.tensor(self.mean, device=device, dtype=torch.float32),
            torch.tensor(self.scale, device=device, dtype=torch.float32),
            torch.tensor(self.weight, device=device, dtype=torch.float32),
            torch.tensor(self.bias, device=device, dtype=torch.float32),
        )


def logistic_gate_probability(
    features: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    standardized = (features - mean) / scale.clamp_min(1e-12)
    return torch.sigmoid((standardized * weight).sum() + bias)
