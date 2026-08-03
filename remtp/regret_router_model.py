"""Tiny low-rank controller for cross-block regret feedback.

The target model and native MTP remain frozen.  This module only maps a
previous-block regret state plus causal context features to bounded controls
for the next six MTP heads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class RegretRouterArchitecture:
    hidden_size: int
    num_heads: int = 6
    rank: int = 8
    width: int = 32
    max_direction_strength: float = 0.03
    max_logit_scale: float = 0.08
    max_budget_reduction: float = 0.50

    def validate(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.rank < 1 or self.width < 1:
            raise ValueError("rank and width must be positive")
        if self.max_direction_strength < 0.0:
            raise ValueError("max_direction_strength must be non-negative")
        if self.max_logit_scale < 0.0:
            raise ValueError("max_logit_scale must be non-negative")
        if not 0.0 <= self.max_budget_reduction <= 1.0:
            raise ValueError("max_budget_reduction must be in [0,1]")


@dataclass
class RegretRouterControls:
    direction_strength: torch.Tensor
    logit_scale: torch.Tensor
    budget_scale: torch.Tensor


class RegretRouter(nn.Module):
    """Low-rank, per-head router with strictly bounded outputs."""

    scalar_feature_count = 6

    def __init__(self, architecture: RegretRouterArchitecture) -> None:
        super().__init__()
        architecture.validate()
        self.architecture = architecture
        self.root_projection = nn.Linear(
            architecture.hidden_size,
            architecture.rank,
            bias=False,
        )
        self.direction_projection = nn.Linear(
            architecture.hidden_size,
            architecture.rank,
            bias=False,
        )
        input_size = 2 * architecture.rank + self.scalar_feature_count
        self.mlp = nn.Sequential(
            nn.Linear(input_size, architecture.width),
            nn.SiLU(),
            nn.Linear(architecture.width, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.root_projection.weight, std=0.02)
        nn.init.normal_(self.direction_projection.weight, std=0.02)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        # Direction is nearly disabled and budget is nearly one at start.
        self.mlp[2].bias.data.copy_(torch.tensor([-4.0, 0.0, -4.0]))

    @staticmethod
    def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
        return value / torch.sqrt(
            value.to(torch.float32).square().mean(dim=-1, keepdim=True)
        ).clamp_min(1e-8).to(value.dtype)

    def forward(
        self,
        root_hidden: torch.Tensor,
        regret_direction: torch.Tensor,
        regret_debt: torch.Tensor,
        entropy: torch.Tensor,
        margin: torch.Tensor,
        head_reliability: torch.Tensor,
    ) -> RegretRouterControls:
        """Return controls with shape ``[batch,num_heads]``.

        ``entropy`` and ``margin`` may be either one value per block or one
        value per head. The proposal path uses the last causally available
        target statistics; verification may call the same router again with
        the newly observed per-head target statistics for budget control.
        """
        if root_hidden.ndim != 2 or regret_direction.shape != root_hidden.shape:
            raise ValueError("root_hidden and regret_direction must share [B,H]")
        if root_hidden.shape[-1] != self.architecture.hidden_size:
            raise ValueError("router hidden size mismatch")
        batch = root_hidden.shape[0]
        heads = self.architecture.num_heads

        def expand_feature(value: torch.Tensor) -> torch.Tensor:
            value = value.to(device=root_hidden.device, dtype=torch.float32)
            if value.ndim == 1:
                if value.shape[0] != batch:
                    raise ValueError("block feature batch mismatch")
                return value.unsqueeze(1).expand(batch, heads)
            if value.shape != (batch, heads):
                raise ValueError("head feature must have shape [B,num_heads]")
            return value

        debt = expand_feature(regret_debt)
        entropy = expand_feature(entropy)
        margin = expand_feature(margin)
        reliability = head_reliability.to(
            device=root_hidden.device,
            dtype=torch.float32,
        )
        if reliability.ndim == 1:
            if reliability.shape[0] != heads:
                raise ValueError("head reliability length mismatch")
            reliability = reliability.unsqueeze(0).expand(batch, heads)
        elif reliability.shape != (batch, heads):
            raise ValueError("head reliability must be [H] or [B,H]")

        root = self._rms_normalize(root_hidden.to(torch.float32))
        direction = self._rms_normalize(regret_direction.to(torch.float32))
        root_low = self.root_projection(root).unsqueeze(1).expand(-1, heads, -1)
        direction_low = self.direction_projection(direction).unsqueeze(1).expand(
            -1, heads, -1
        )
        depth = torch.linspace(
            0.0,
            1.0,
            heads,
            device=root_hidden.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(batch, heads)
        debt_gate = (debt / (1.0 + debt)).clamp(0.0, 1.0)
        scalar = torch.stack(
            (debt, debt_gate, entropy, margin, reliability, depth),
            dim=-1,
        )
        raw = self.mlp(torch.cat((root_low, direction_low, scalar), dim=-1))

        direction_strength = (
            self.architecture.max_direction_strength
            * torch.sigmoid(raw[..., 0])
            * debt_gate
        )
        logit_scale = (
            self.architecture.max_logit_scale
            * torch.tanh(raw[..., 1])
            * debt_gate
        )
        budget_reduction = (
            self.architecture.max_budget_reduction
            * torch.sigmoid(raw[..., 2])
            * debt_gate
        )
        return RegretRouterControls(
            direction_strength=direction_strength,
            logit_scale=logit_scale,
            budget_scale=(1.0 - budget_reduction).clamp(0.0, 1.0),
        )


def save_regret_router_checkpoint(
    path: str | Path,
    model: RegretRouter,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "architecture": asdict(model.architecture),
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        Path(path),
    )


def load_regret_router_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RegretRouter, dict[str, Any]]:
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    if payload.get("format_version") != 1:
        raise ValueError("unsupported regret-router checkpoint format")
    architecture = RegretRouterArchitecture(**payload["architecture"])
    model = RegretRouter(architecture)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, dict(payload.get("metadata", {}))
