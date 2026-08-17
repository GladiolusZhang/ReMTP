"""Entropy-aware dynamic-chain fallback for Scheme 3.

vLLM 0.18 can build TreeAttention draft metadata, but its installed rejection
sampler does not select a valid path through a flattened tree. Qwen3.5 also
has recurrent GDN state that cannot safely be branched by that path. This
module implements the explicitly allowed fallback from the proposal: retain
the native MTP chain and dynamically omit target-verification nodes.

The next block uses one of three chain lengths selected from the previous
target root distribution and MTP confidence. The MTP proposer still computes
its configured maximum depth, but the scheduler sends only the selected
prefix to the target model, so the target node budget never exceeds the
native MTP baseline.
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


@dataclass(frozen=True)
class AdaptiveChainConfig:
    max_depth: int = 6
    medium_depth: int = 4
    short_depth: int = 2
    top_m: int = 4
    high_margin: float = 1.50
    medium_margin: float = 0.50
    low_entropy: float = 0.45
    high_entropy: float = 0.75
    high_mtp_confidence: float = 0.30
    medium_mtp_confidence: float = 0.10
    diagnostics: bool = False

    @classmethod
    def from_env(cls) -> "AdaptiveChainConfig":
        config = cls(
            max_depth=int(os.getenv("REMTP_ADAPTIVE_MAX_DEPTH", "6")),
            medium_depth=int(os.getenv("REMTP_ADAPTIVE_MEDIUM_DEPTH", "4")),
            short_depth=int(os.getenv("REMTP_ADAPTIVE_SHORT_DEPTH", "2")),
            top_m=int(os.getenv("REMTP_ADAPTIVE_TOP_M", "4")),
            high_margin=float(os.getenv("REMTP_ADAPTIVE_HIGH_MARGIN", "1.50")),
            medium_margin=float(
                os.getenv("REMTP_ADAPTIVE_MEDIUM_MARGIN", "0.50")
            ),
            low_entropy=float(os.getenv("REMTP_ADAPTIVE_LOW_ENTROPY", "0.45")),
            high_entropy=float(
                os.getenv("REMTP_ADAPTIVE_HIGH_ENTROPY", "0.75")
            ),
            high_mtp_confidence=float(
                os.getenv("REMTP_ADAPTIVE_HIGH_MTP_CONF", "0.30")
            ),
            medium_mtp_confidence=float(
                os.getenv("REMTP_ADAPTIVE_MEDIUM_MTP_CONF", "0.10")
            ),
            diagnostics=os.getenv("REMTP_ADAPTIVE_DIAGNOSTICS", "0") == "1",
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 1 <= self.short_depth <= self.medium_depth <= self.max_depth:
            raise ValueError("adaptive depths must satisfy 1 <= short <= medium <= max")
        if self.top_m < 2:
            raise ValueError("adaptive top-m must be at least two")
        if self.high_margin < self.medium_margin or self.medium_margin < 0.0:
            raise ValueError("high margin must be no smaller than medium margin")
        if not 0.0 <= self.low_entropy <= self.high_entropy <= 1.0:
            raise ValueError("entropy thresholds must lie in [0,1]")
        if self.high_mtp_confidence < self.medium_mtp_confidence:
            raise ValueError("high MTP confidence must exceed medium confidence")


@dataclass
class _AdaptiveState:
    next_depth: int = 6
    request_id: str | None = None
    diagnostic_emitted: bool = False


_CONFIG: AdaptiveChainConfig | None = None
_STATE = _AdaptiveState()


def select_chain_depth(
    target_root_probs: torch.Tensor,
    target_root_logits: torch.Tensor,
    mtp_candidate_prob: torch.Tensor,
    config: AdaptiveChainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected depth, target compact entropy and target margin."""

    if target_root_probs.ndim != 1 or target_root_logits.ndim != 1:
        raise ValueError("root probabilities and logits must be one-dimensional")
    if target_root_probs.shape != target_root_logits.shape:
        raise ValueError("root probability and logit layouts must match")
    top_m = min(config.top_m, target_root_probs.shape[0])
    top_values, top_ids = torch.topk(
        target_root_logits, k=top_m, largest=True, sorted=True
    )
    top_probs = target_root_probs[top_ids]
    tail = (1.0 - top_probs.sum()).clamp_min(0.0).reshape(1)
    buckets = torch.cat((top_probs, tail))
    entropy = -(
        buckets * torch.log(buckets.clamp_min(1e-30))
    ).sum() / math.log(top_m + 1.0)
    margin = (top_values[0] - top_values[1]).to(torch.float32).clamp_min(0.0)
    q_conf = mtp_candidate_prob.to(torch.float32).reshape(())

    high = (
        (margin >= config.high_margin)
        & (entropy <= config.low_entropy)
        & (q_conf >= config.high_mtp_confidence)
    )
    medium = (
        (margin >= config.medium_margin)
        & (entropy <= config.high_entropy)
        & (q_conf >= config.medium_mtp_confidence)
    )
    depth = torch.where(
        high,
        torch.as_tensor(config.max_depth, device=margin.device),
        torch.where(
            medium,
            torch.as_tensor(config.medium_depth, device=margin.device),
            torch.as_tensor(config.short_depth, device=margin.device),
        ),
    )
    return depth.to(torch.int64), entropy.clamp(0.0, 1.0), margin


def _adaptive_observer_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    original = getattr(_adaptive_observer_rejection_sample, "_remtp_original")
    if (
        _CONFIG is not None
        and target_logits.shape[0] > 0
        and draft_probs is not None
        and len(num_draft_tokens) == 1
    ):
        root_probs = target_logits[0].softmax(dim=-1, dtype=torch.float32)
        root_id = draft_token_ids[0].to(torch.int64)
        q_root = draft_probs[0, root_id]
        depth, entropy, margin = select_chain_depth(
            root_probs, target_logits[0], q_root, _CONFIG
        )
        # One scalar synchronization is intentionally placed after target
        # verification. The scheduler already performs D2H synchronization to
        # receive draft IDs, so no vocabulary tensor is moved to the CPU.
        _STATE.next_depth = int(depth.item())
        if _CONFIG.diagnostics and not _STATE.diagnostic_emitted:
            print(
                "[ReMTP][AdaptiveChain][diagnostic] "
                f"next_depth={_STATE.next_depth} "
                f"root_entropy={entropy.item():.4f} "
                f"root_margin={margin.item():.4f} q(D0)={q_root.item():.4f}",
                flush=True,
            )
            _STATE.diagnostic_emitted = True
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


def _adaptive_get_draft_token_ids_cpu(
    self: Any,
) -> tuple[list[list[int]], list[str]]:
    original = getattr(_adaptive_get_draft_token_ids_cpu, "_remtp_original")
    draft_rows, request_ids = original(self)
    if _CONFIG is None or not draft_rows or not request_ids:
        return draft_rows, request_ids
    if len(draft_rows) != len(request_ids):
        raise RuntimeError("draft rows and request IDs have different lengths")
    if len(request_ids) != 1:
        raise RuntimeError("adaptive-chain MTP requires --max-num-seqs 1")

    request_id = request_ids[0]
    if request_id != _STATE.request_id:
        _STATE.request_id = request_id
        _STATE.next_depth = _CONFIG.max_depth
    depth = max(1, min(_STATE.next_depth, _CONFIG.max_depth))
    return [list(draft_rows[0][:depth])], request_ids


def install_adaptive_chain_mtp() -> None:
    """Install the target-node adaptive-chain Scheme 3 fallback."""

    global _CONFIG

    config = AdaptiveChainConfig.from_env()
    _CONFIG = config
    _STATE.next_depth = config.max_depth
    _STATE.request_id = None
    _STATE.diagnostic_emitted = False

    rejection_module = importlib.import_module(_REJECTION_MODULE)
    current_rejection = rejection_module.rejection_sample
    if not getattr(current_rejection, "_remtp_adaptive_chain", False):
        _adaptive_observer_rejection_sample._remtp_adaptive_chain = True
        _adaptive_observer_rejection_sample._remtp_original = current_rejection
        rejection_module.rejection_sample = _adaptive_observer_rejection_sample

    runner_module = importlib.import_module(_GPU_RUNNER_MODULE)
    runner_cls = runner_module.GPUModelRunner
    current_get = runner_cls._get_draft_token_ids_cpu
    if not getattr(current_get, "_remtp_adaptive_chain", False):
        _adaptive_get_draft_token_ids_cpu._remtp_adaptive_chain = True
        _adaptive_get_draft_token_ids_cpu._remtp_original = current_get
        runner_cls._get_draft_token_ids_cpu = _adaptive_get_draft_token_ids_cpu

    print(
        "[ReMTP][AdaptiveChain] scheme3=adaptive-chain-fallback "
        f"depths={config.short_depth}/{config.medium_depth}/{config.max_depth} "
        "target_nodes<=native-MTP tree_claim=disabled",
        flush=True,
    )
