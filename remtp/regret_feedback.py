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


def _parse_vocab_head_map(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(","))
    if not result or any(item < 0 for item in result):
        raise ValueError(
            "REMTP_REGRET_VOCAB_HEAD_MAP must contain non-negative indices"
        )
    return result


@dataclass(frozen=True)
class RegretFeedbackConfig:
    """Configuration for Budget-Induced Regret Feedback."""

    direction: str = "token"
    expected_draft_tokens: int = 6
    regret_top_k: int = 16
    compatibility_top_k: int = 32
    compatibility_mode: str = "relative"
    compatibility_source: str = "cached"
    alpha: float = 0.03
    strength_reference: float = 0.0
    scale_learning_rate: float = 0.50
    scale_decay: float = 0.90
    scale_gap_reference: float = 0.05
    scale_newton_clip: float = 0.20
    scale_min: float = 0.75
    scale_max: float = 1.25
    scale_initial: float = 1.0
    scale_initial_by_reliability: bool = False
    scale_causal_only: bool = False
    top1_bias_initial: float = 0.25
    top1_bias_learning_rate: float = 0.25
    top1_bias_decay: float = 0.90
    top1_bias_clip: float = 0.75
    repetition_ngram_size: int = 4
    repetition_penalty: float = 1.0
    repetition_debt_reference: float = 0.25
    adaptive_min_depth: int = 4
    adaptive_depth_blocks: int = 2
    adaptive_depth_trigger: float = 0.25
    root_anchor_mix: float = 0.25
    root_anchor_reference: float = 0.50
    root_anchor_min_confidence: float = 0.50
    vocab_bias_scale: float = 0.05
    vocab_bias_clip: float = 2.0
    vocab_bias_top_k: int = 32
    vocab_head_map: tuple[int, ...] = (0, 0, 4, 2, 3, 1)
    vocab_route_decay: float = 0.50
    vocab_route_min_gain: float = 0.0
    vocab_route_prior_gain: float = 0.0
    vocab_route_gain_clip: float = 0.001
    fusion_gate_threshold: float = 0.50
    hidden_consistency_threshold: float = 0.0
    hidden_confirmation_blocks: int = 2
    hidden_unconfirmed_scale: float = 0.50
    verifier_risk_strength: float = 2.0
    verifier_risk_reference: float = 0.25
    verifier_policy: str = "heuristic"
    verifier_cactus_mix: float = 1.0
    verifier_budget_slope: float = 0.4
    verifier_tail_threshold: float = 0.8
    token_decay: float = 0.90
    rejection_reset: float = 0.25
    max_idle_blocks: int = 2
    incompatible_decay: float = 0.50
    head_reliability: tuple[float, ...] = _DEFAULT_HEAD_RELIABILITY
    audit_interval: int = 0
    diagnostics: bool = False

    def validate(self) -> None:
        if self.direction not in {
            "none",
            "token",
            "expected_token",
            "expected_fusion",
            "expected_token_adaptive",
            "expected_token_preconditioned",
            "expected_hidden_fusion",
            "expected_hidden_consistent",
            "expected_hidden_adaptive",
            "expected_hidden_calibrated",
            "expected_hidden_compatible",
            "expected_root_anchor",
            "expected_root_exact",
            "expected_residual",
            "expected_transport",
            "boundary_residual",
            "expected_trust",
            "expected_logit",
            "expected_pq_gradient",
            "expected_vocab",
            "expected_vocab_audit",
            "expected_vocab_negative",
            "expected_vocab_headmap",
            "expected_vocab_online",
            "hidden",
            "hidden_output",
            "headwise_output",
            "expected_scale",
            "expected_scale_adaptive_depth",
            "expected_scale_counterfactual",
            "expected_scale_repetition",
            "expected_scale_tv",
            "expected_top1_bias",
            "adaptive_depth",
            "verifier_risk",
            "hybrid",
        }:
            raise ValueError(
                "direction must be one of: none, token, hidden, "
                "expected_token, expected_fusion, expected_hidden_fusion, "
                "expected_token_adaptive, "
                "expected_token_preconditioned, "
                "expected_hidden_consistent, "
                "expected_hidden_adaptive, "
                "expected_hidden_calibrated, "
                "expected_hidden_compatible, "
                "expected_root_anchor, "
                "expected_root_exact, "
                "expected_residual, "
                "boundary_residual, expected_trust, hidden_output, "
                "expected_logit, expected_pq_gradient, headwise_output, "
                "expected_vocab, expected_vocab_audit, "
                "expected_vocab_negative, expected_vocab_headmap, "
                "expected_vocab_online, expected_scale, "
                "expected_scale_adaptive_depth, "
                "expected_scale_counterfactual, expected_scale_repetition, "
                "expected_scale_tv, expected_top1_bias, adaptive_depth, "
                "verifier_risk, hybrid"
            )
        if self.compatibility_mode not in {"relative", "positive_lift"}:
            raise ValueError(
                "compatibility_mode must be relative or positive_lift"
            )
        if self.compatibility_source not in {"cached", "fresh_hidden"}:
            raise ValueError(
                "compatibility_source must be cached or fresh_hidden"
            )
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if self.regret_top_k < 1 or self.compatibility_top_k < 2:
            raise ValueError("regret/compatibility top-k values are invalid")
        if self.alpha < 0.0 or self.alpha > 0.25:
            raise ValueError("alpha must be in [0, 0.25]")
        if self.strength_reference < 0.0:
            raise ValueError("strength_reference must be non-negative")
        if not 0.0 <= self.scale_learning_rate <= 2.0:
            raise ValueError("scale_learning_rate must be in [0,2]")
        if not 0.0 <= self.scale_decay <= 1.0:
            raise ValueError("scale_decay must be in [0,1]")
        if self.scale_gap_reference <= 0.0:
            raise ValueError("scale_gap_reference must be positive")
        if not 0.0 < self.scale_newton_clip < 1.0:
            raise ValueError("scale_newton_clip must be in (0,1)")
        if not 0.0 < self.scale_min <= 1.0 <= self.scale_max:
            raise ValueError(
                "scale bounds must satisfy 0 < min <= 1 <= max"
            )
        if not self.scale_min <= self.scale_initial <= self.scale_max:
            raise ValueError("initial scale must lie within scale bounds")
        if not 0.0 <= self.top1_bias_initial <= self.top1_bias_clip:
            raise ValueError("initial top-1 bias must lie in [0, clip]")
        if not 0.0 <= self.top1_bias_learning_rate <= 2.0:
            raise ValueError("top-1 bias learning rate must be in [0,2]")
        if not 0.0 <= self.top1_bias_decay <= 1.0:
            raise ValueError("top-1 bias decay must be in [0,1]")
        if self.top1_bias_clip <= 0.0:
            raise ValueError("top-1 bias clip must be positive")
        if self.repetition_ngram_size < 2:
            raise ValueError("repetition ngram size must be at least two")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition penalty must be positive")
        if self.repetition_debt_reference <= 0.0:
            raise ValueError("repetition debt reference must be positive")
        if not 1 <= self.adaptive_min_depth < self.expected_draft_tokens:
            raise ValueError("adaptive min depth must be in [1, draft_tokens)")
        if self.adaptive_depth_blocks < 1:
            raise ValueError("adaptive depth blocks must be positive")
        if self.adaptive_depth_trigger <= 0.0:
            raise ValueError("adaptive depth trigger must be positive")
        if not 0.0 <= self.root_anchor_mix <= 1.0:
            raise ValueError("root anchor mix must be in [0,1]")
        if self.root_anchor_reference <= 0.0:
            raise ValueError("root anchor reference must be positive")
        if not 0.0 <= self.root_anchor_min_confidence < 1.0:
            raise ValueError("root anchor minimum confidence must be in [0,1)")
        if not 0.0 <= self.vocab_bias_scale <= 1.0:
            raise ValueError("vocab_bias_scale must be in [0,1]")
        if self.vocab_bias_clip <= 0.0:
            raise ValueError("vocab_bias_clip must be positive")
        if self.vocab_bias_top_k < 1:
            raise ValueError("vocab_bias_top_k must be positive")
        if (
            len(self.vocab_head_map) < self.expected_draft_tokens
            or any(
                item >= self.expected_draft_tokens
                for item in self.vocab_head_map[: self.expected_draft_tokens]
            )
        ):
            raise ValueError(
                "vocab head map must cover and index every MTP position"
            )
        if not 0.0 <= self.vocab_route_decay < 1.0:
            raise ValueError("vocab route decay must be in [0,1)")
        if self.vocab_route_min_gain < 0.0:
            raise ValueError("vocab route minimum gain must be non-negative")
        if self.vocab_route_prior_gain < 0.0:
            raise ValueError("vocab route prior gain must be non-negative")
        if self.vocab_route_gain_clip <= 0.0:
            raise ValueError("vocab route gain clip must be positive")
        if not 0.0 <= self.fusion_gate_threshold <= 1.0:
            raise ValueError("fusion_gate_threshold must be in [0,1]")
        if not -1.0 <= self.hidden_consistency_threshold <= 1.0:
            raise ValueError("hidden consistency threshold must be in [-1,1]")
        if self.hidden_confirmation_blocks < 1:
            raise ValueError("hidden confirmation blocks must be positive")
        if not 0.0 <= self.hidden_unconfirmed_scale <= 1.0:
            raise ValueError("hidden unconfirmed scale must be in [0,1]")
        if self.verifier_risk_strength < 0.0:
            raise ValueError("verifier_risk_strength must be non-negative")
        if self.verifier_risk_reference <= 0.0:
            raise ValueError("verifier_risk_reference must be positive")
        if self.verifier_policy not in {
            "heuristic",
            "cactus_anchor",
            "cactus_tail",
        }:
            raise ValueError(
                "verifier_policy must be heuristic, cactus_anchor, "
                "or cactus_tail"
            )
        if not 0.0 <= self.verifier_cactus_mix <= 1.0:
            raise ValueError("verifier_cactus_mix must be in [0,1]")
        if not 0.0 <= self.verifier_budget_slope <= 1.0:
            raise ValueError("verifier_budget_slope must be in [0,1]")
        if not 0.0 <= self.verifier_tail_threshold < 1.0:
            raise ValueError("verifier_tail_threshold must be in [0,1)")
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
        default_vocab_head_map = (
            "0,0,4,2,3,1"
            if expected == 6
            else ",".join(str(index) for index in range(expected))
        )
        config = cls(
            direction=os.getenv("REMTP_REGRET_DIRECTION", "token"),
            expected_draft_tokens=expected,
            regret_top_k=int(os.getenv("REMTP_REGRET_TOP_K", "16")),
            compatibility_top_k=int(
                os.getenv("REMTP_REGRET_COMPATIBILITY_TOP_K", "32")
            ),
            compatibility_mode=os.getenv(
                "REMTP_REGRET_COMPATIBILITY_MODE",
                "relative",
            ),
            compatibility_source=os.getenv(
                "REMTP_REGRET_COMPATIBILITY_SOURCE",
                "cached",
            ),
            alpha=float(os.getenv("REMTP_REGRET_ALPHA", "0.03")),
            strength_reference=float(
                os.getenv("REMTP_REGRET_STRENGTH_REFERENCE", "0.0")
            ),
            scale_learning_rate=float(
                os.getenv("REMTP_REGRET_SCALE_LEARNING_RATE", "0.50")
            ),
            scale_decay=float(
                os.getenv("REMTP_REGRET_SCALE_DECAY", "0.90")
            ),
            scale_gap_reference=float(
                os.getenv("REMTP_REGRET_SCALE_GAP_REFERENCE", "0.05")
            ),
            scale_newton_clip=float(
                os.getenv("REMTP_REGRET_SCALE_NEWTON_CLIP", "0.20")
            ),
            scale_min=float(
                os.getenv("REMTP_REGRET_SCALE_MIN", "0.75")
            ),
            scale_max=float(
                os.getenv("REMTP_REGRET_SCALE_MAX", "1.25")
            ),
            scale_initial=float(
                os.getenv("REMTP_REGRET_SCALE_INITIAL", "1.0")
            ),
            scale_initial_by_reliability=_env_flag(
                "REMTP_REGRET_SCALE_INITIAL_BY_RELIABILITY",
                False,
            ),
            scale_causal_only=_env_flag(
                "REMTP_REGRET_SCALE_CAUSAL_ONLY",
                False,
            ),
            top1_bias_initial=float(
                os.getenv("REMTP_REGRET_TOP1_BIAS_INITIAL", "0.25")
            ),
            top1_bias_learning_rate=float(
                os.getenv("REMTP_REGRET_TOP1_BIAS_LEARNING_RATE", "0.25")
            ),
            top1_bias_decay=float(
                os.getenv("REMTP_REGRET_TOP1_BIAS_DECAY", "0.90")
            ),
            top1_bias_clip=float(
                os.getenv("REMTP_REGRET_TOP1_BIAS_CLIP", "0.75")
            ),
            repetition_ngram_size=int(
                os.getenv("REMTP_REGRET_REPETITION_NGRAM_SIZE", "4")
            ),
            repetition_penalty=float(
                os.getenv("REMTP_REGRET_REPETITION_PENALTY", "1.0")
            ),
            repetition_debt_reference=float(
                os.getenv("REMTP_REGRET_REPETITION_DEBT_REFERENCE", "0.25")
            ),
            adaptive_min_depth=int(
                os.getenv("REMTP_REGRET_ADAPTIVE_MIN_DEPTH", "4")
            ),
            adaptive_depth_blocks=int(
                os.getenv("REMTP_REGRET_ADAPTIVE_DEPTH_BLOCKS", "2")
            ),
            adaptive_depth_trigger=float(
                os.getenv("REMTP_REGRET_ADAPTIVE_DEPTH_TRIGGER", "0.25")
            ),
            root_anchor_mix=float(
                os.getenv("REMTP_REGRET_ROOT_ANCHOR_MIX", "0.25")
            ),
            root_anchor_reference=float(
                os.getenv("REMTP_REGRET_ROOT_ANCHOR_REFERENCE", "0.50")
            ),
            root_anchor_min_confidence=float(
                os.getenv("REMTP_REGRET_ROOT_ANCHOR_MIN_CONFIDENCE", "0.50")
            ),
            vocab_bias_scale=float(
                os.getenv("REMTP_REGRET_VOCAB_BIAS_SCALE", "0.05")
            ),
            vocab_bias_clip=float(
                os.getenv("REMTP_REGRET_VOCAB_BIAS_CLIP", "2.0")
            ),
            vocab_bias_top_k=int(
                os.getenv("REMTP_REGRET_VOCAB_BIAS_TOP_K", "32")
            ),
            vocab_head_map=_parse_vocab_head_map(
                os.getenv(
                    "REMTP_REGRET_VOCAB_HEAD_MAP",
                    default_vocab_head_map,
                )
            ),
            vocab_route_decay=float(
                os.getenv("REMTP_REGRET_VOCAB_ROUTE_DECAY", "0.50")
            ),
            vocab_route_min_gain=float(
                os.getenv("REMTP_REGRET_VOCAB_ROUTE_MIN_GAIN", "0.0")
            ),
            vocab_route_prior_gain=float(
                os.getenv("REMTP_REGRET_VOCAB_ROUTE_PRIOR_GAIN", "0.0")
            ),
            vocab_route_gain_clip=float(
                os.getenv("REMTP_REGRET_VOCAB_ROUTE_GAIN_CLIP", "0.001")
            ),
            fusion_gate_threshold=float(
                os.getenv("REMTP_REGRET_FUSION_GATE_THRESHOLD", "0.50")
            ),
            hidden_consistency_threshold=float(
                os.getenv("REMTP_REGRET_HIDDEN_CONSISTENCY_THRESHOLD", "0.0")
            ),
            hidden_confirmation_blocks=int(
                os.getenv("REMTP_REGRET_HIDDEN_CONFIRMATION_BLOCKS", "2")
            ),
            hidden_unconfirmed_scale=float(
                os.getenv("REMTP_REGRET_HIDDEN_UNCONFIRMED_SCALE", "0.50")
            ),
            verifier_risk_strength=float(
                os.getenv("REMTP_REGRET_VERIFIER_RISK_STRENGTH", "2.0")
            ),
            verifier_risk_reference=float(
                os.getenv("REMTP_REGRET_VERIFIER_RISK_REFERENCE", "0.25")
            ),
            verifier_policy=os.getenv(
                "REMTP_REGRET_VERIFIER_POLICY",
                "heuristic",
            ),
            verifier_cactus_mix=float(
                os.getenv("REMTP_REGRET_VERIFIER_CACTUS_MIX", "1.0")
            ),
            verifier_budget_slope=float(
                os.getenv("REMTP_REGRET_VERIFIER_BUDGET_SLOPE", "0.4")
            ),
            verifier_tail_threshold=float(
                os.getenv("REMTP_REGRET_VERIFIER_TAIL_THRESHOLD", "0.8")
            ),
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
    head_memory: torch.Tensor | None = None
    head_log_scales: torch.Tensor | None = None
    head_top1_bias: torch.Tensor | None = None
    head_vocab_bias: torch.Tensor | None = None
    head_vocab_positive_bias: torch.Tensor | None = None
    head_vocab_negative_bias: torch.Tensor | None = None
    head_vocab_candidate_bias: torch.Tensor | None = None
    vocab_route_scores: torch.Tensor | None = None
    vocab_head_route: tuple[int, ...] | None = None
    hidden_confirmation_count: torch.Tensor | None = None
    hidden_last_alignment: torch.Tensor | None = None
    source_target_hidden: torch.Tensor | None = None
    current_base_probs: list[torch.Tensor] = field(default_factory=list)
    tv_debt: torch.Tensor | None = None
    idle_blocks: torch.Tensor | None = None
    output_weight: torch.Tensor | None = None
    compatibility_ids: torch.Tensor | None = None
    compatibility_probs: torch.Tensor | None = None
    bonus_ids: torch.Tensor | None = None
    bonus_probs: torch.Tensor | None = None
    root_target_logits: torch.Tensor | None = None
    root_anchor_gate: torch.Tensor | None = None
    root_base_probs: torch.Tensor | None = None
    root_exact_applied: torch.Tensor | None = None
    suppress_scale_for_block: torch.Tensor | None = None
    repetition_penalty_ids: torch.Tensor | None = None
    repetition_gate: torch.Tensor | None = None
    adaptive_depth_cooldown: int = 0

    def reset_request(self) -> None:
        self.memory = None
        self.head_memory = None
        self.head_log_scales = None
        self.head_top1_bias = None
        self.head_vocab_bias = None
        self.head_vocab_positive_bias = None
        self.head_vocab_negative_bias = None
        self.head_vocab_candidate_bias = None
        self.vocab_route_scores = None
        self.vocab_head_route = None
        self.hidden_confirmation_count = None
        self.hidden_last_alignment = None
        self.source_target_hidden = None
        self.current_base_probs.clear()
        self.tv_debt = None
        self.idle_blocks = None
        self.compatibility_ids = None
        self.compatibility_probs = None
        self.bonus_ids = None
        self.bonus_probs = None
        self.root_target_logits = None
        self.root_anchor_gate = None
        self.root_base_probs = None
        self.root_exact_applied = None
        self.suppress_scale_for_block = None
        self.repetition_penalty_ids = None
        self.repetition_gate = None
        self.adaptive_depth_cooldown = 0


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
    pq_kl_sum: torch.Tensor | None = None
    pq_kl_rows: torch.Tensor | None = None
    expected_regret_sum: torch.Tensor | None = None
    abs_logit_scale_sum: torch.Tensor | None = None
    counterfactual_kl_gain_sum: torch.Tensor | None = None
    counterfactual_kl_rows: torch.Tensor | None = None
    counterfactual_tv_gain_sum: torch.Tensor | None = None
    counterfactual_tv_rows: torch.Tensor | None = None
    repetition_escape_blocks: torch.Tensor | None = None
    reduced_depth_blocks: torch.Tensor | None = None
    vocab_full_map_gain: torch.Tensor | None = None
    vocab_positive_map_gain: torch.Tensor | None = None
    vocab_negative_map_gain: torch.Tensor | None = None
    vocab_candidate_map_gain: torch.Tensor | None = None
    vocab_context_cosine: torch.Tensor | None = None
    vocab_draft_context_cosine: torch.Tensor | None = None
    vocab_context_gain_sum: torch.Tensor | None = None
    vocab_draft_context_gain_sum: torch.Tensor | None = None
    vocab_oracle_gain_sum: torch.Tensor | None = None
    vocab_audit_blocks: torch.Tensor | None = None

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
        self.pq_kl_sum = scalar.clone()
        self.pq_kl_rows = scalar.clone()
        self.expected_regret_sum = scalar.clone()
        self.abs_logit_scale_sum = scalar.clone()
        self.counterfactual_kl_gain_sum = scalar.clone()
        self.counterfactual_kl_rows = scalar.clone()
        self.counterfactual_tv_gain_sum = scalar.clone()
        self.counterfactual_tv_rows = scalar.clone()
        self.repetition_escape_blocks = scalar.clone()
        self.reduced_depth_blocks = scalar.clone()
        matrix = torch.zeros(
            (rows, rows),
            device=device,
            dtype=torch.float64,
        )
        self.vocab_full_map_gain = matrix.clone()
        self.vocab_positive_map_gain = matrix.clone()
        self.vocab_negative_map_gain = matrix.clone()
        self.vocab_candidate_map_gain = matrix.clone()
        self.vocab_context_cosine = matrix.clone()
        self.vocab_draft_context_cosine = matrix.clone()
        self.vocab_context_gain_sum = scalar.clone()
        self.vocab_draft_context_gain_sum = scalar.clone()
        self.vocab_oracle_gain_sum = scalar.clone()
        self.vocab_audit_blocks = scalar.clone()

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
            "pq_kl_sum",
            "pq_kl_rows",
            "expected_regret_sum",
            "abs_logit_scale_sum",
            "counterfactual_kl_gain_sum",
            "counterfactual_kl_rows",
            "counterfactual_tv_gain_sum",
            "counterfactual_tv_rows",
            "repetition_escape_blocks",
            "reduced_depth_blocks",
            "vocab_full_map_gain",
            "vocab_positive_map_gain",
            "vocab_negative_map_gain",
            "vocab_candidate_map_gain",
            "vocab_context_cosine",
            "vocab_draft_context_cosine",
            "vocab_context_gain_sum",
            "vocab_draft_context_gain_sum",
            "vocab_oracle_gain_sum",
            "vocab_audit_blocks",
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


def pad_head_values(values: torch.Tensor, expected_rows: int) -> torch.Tensor:
    """Right-pad one-dimensional per-head audit values."""
    if values.ndim != 1 or values.shape[0] > expected_rows:
        raise ValueError("invalid per-head audit shape")
    if values.shape[0] == expected_rows:
        return values
    return torch.cat(
        [
            values,
            torch.zeros(
                expected_rows - values.shape[0],
                device=values.device,
                dtype=values.dtype,
            ),
        ]
    )


def pad_head_rows(values: torch.Tensor, expected_rows: int) -> torch.Tensor:
    """Right-pad a two-dimensional per-head state tensor."""
    if values.ndim != 2 or values.shape[0] > expected_rows:
        raise ValueError("invalid per-head state shape")
    if values.shape[0] == expected_rows:
        return values
    return torch.cat(
        [
            values,
            torch.zeros(
                (expected_rows - values.shape[0], values.shape[1]),
                device=values.device,
                dtype=values.dtype,
            ),
        ],
        dim=0,
    )


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


def expected_causal_responsibility(
    draft_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    accepted_mask: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
) -> torch.Tensor:
    """Posterior probability that relaxation caused an observed acceptance."""
    rows = draft_probs.shape[0]
    if candidate_ids.shape != (rows,) or accepted_mask.shape != (rows,):
        raise ValueError("candidate and acceptance rows must match draft_probs")
    draft = draft_probs.to(torch.float32).clamp_min(1e-30)
    draft = draft / draft.sum(dim=-1, keepdim=True)
    ids = candidate_ids.to(device=draft.device, dtype=torch.int64)
    row_ids = torch.arange(rows, device=draft.device)
    q_y = draft[row_ids, ids]
    strict = torch.minimum(
        torch.ones_like(q_y),
        target_candidate_probs.to(torch.float32) / q_y,
    )
    relaxed = torch.minimum(
        torch.ones_like(q_y),
        boosted_candidate_probs.to(torch.float32) / q_y,
    )
    return (
        accepted_mask.to(device=draft.device, dtype=torch.float32)
        * (relaxed - strict).clamp_min(0.0)
        / relaxed.clamp_min(1e-30)
    )


def expected_regret_scale_update(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    accepted_mask: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
    old_log_scales: torch.Tensor,
    config: RegretFeedbackConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update per-head draft logit scales from expected relaxed residual.

    For an actually accepted candidate, ``1-A_strict/A_relaxed`` is the
    posterior probability that relaxation was causally necessary under the
    verifier's shared-uniform coupling.  It replaces the noisy binary causal
    event while remaining conditioned on the prefix that was really committed.

    The signed correction is a clipped Newton step for minimizing
    ``KL(P_i || softmax(s_i * log Q_i))`` at ``s_i=1``.  Only future draft
    distributions are calibrated; target distributions and verification are
    untouched.
    """
    if target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must have equal shape")
    rows = target_probs.shape[0]
    if candidate_ids.shape != (rows,):
        raise ValueError("candidate_ids must have one ID per probability row")
    if accepted_mask.shape != (rows,):
        raise ValueError("accepted_mask must have one value per draft head")
    if old_log_scales.shape != (rows,):
        raise ValueError("old_log_scales must have one value per draft head")

    target = target_probs.to(torch.float32).clamp_min(1e-30)
    draft = draft_probs.to(torch.float32).clamp_min(1e-30)
    target = target / target.sum(dim=-1, keepdim=True)
    draft = draft / draft.sum(dim=-1, keepdim=True)
    expected_residual = expected_causal_responsibility(
        draft,
        candidate_ids,
        accepted_mask,
        target_candidate_probs,
        boosted_candidate_probs,
    )
    update_weight = (
        expected_residual / config.scale_gap_reference
    ).clamp(0.0, 1.0)

    log_q = torch.log(draft)
    mean_log_q = (draft * log_q).sum(dim=-1)
    target_mean_log_q = (target * log_q).sum(dim=-1)
    gradient = mean_log_q - target_mean_log_q
    variance = (
        draft * (log_q - mean_log_q.unsqueeze(1)).square()
    ).sum(dim=-1).clamp_min(1e-6)
    newton_delta = (-gradient / variance).clamp(
        -config.scale_newton_clip,
        config.scale_newton_clip,
    )

    observation = torch.log1p(newton_delta)
    updated = (
        old_log_scales.to(device=draft.device, dtype=torch.float32)
        * config.scale_decay
        + config.scale_learning_rate * update_weight * observation
    )
    updated = updated.clamp(
        math.log(config.scale_min),
        math.log(config.scale_max),
    )
    pq_kl = (
        target * (torch.log(target) - log_q)
    ).sum(dim=-1).clamp_min(0.0)
    return updated, expected_residual, pq_kl, newton_delta


def expected_top1_bias_update(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    accepted_mask: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
    old_bias: torch.Tensor,
    config: RegretFeedbackConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calibrate only each head's preferred-token margin from regret.

    The observed target-vs-draft log-odds residual is weighted by the
    posterior probability that relaxation caused the acceptance. The scalar
    is applied to the *next block's* current top-1 token, so no token identity
    or task-specific feature is carried across blocks.
    """
    if target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must have equal shape")
    rows = target_probs.shape[0]
    if candidate_ids.shape != (rows,) or accepted_mask.shape != (rows,):
        raise ValueError("candidate and acceptance rows must match probabilities")
    if old_bias.shape != (rows,):
        raise ValueError("old_bias must have one value per draft head")

    target = target_probs.to(torch.float32).clamp_min(1e-30)
    draft = draft_probs.to(torch.float32).clamp_min(1e-30)
    target = target / target.sum(dim=-1, keepdim=True)
    draft = draft / draft.sum(dim=-1, keepdim=True)
    expected_residual = expected_causal_responsibility(
        draft,
        candidate_ids,
        accepted_mask,
        target_candidate_probs,
        boosted_candidate_probs,
    )
    row_ids = torch.arange(rows, device=draft.device)
    ids = candidate_ids.to(device=draft.device, dtype=torch.int64)
    p_y = target[row_ids, ids].clamp(1e-6, 1.0 - 1e-6)
    q_y = draft[row_ids, ids].clamp(1e-6, 1.0 - 1e-6)
    target_log_odds = torch.log(p_y) - torch.log1p(-p_y)
    draft_log_odds = torch.log(q_y) - torch.log1p(-q_y)
    observation = (target_log_odds - draft_log_odds).clamp(
        -config.top1_bias_clip,
        config.top1_bias_clip,
    )
    updated = (
        old_bias.to(device=draft.device, dtype=torch.float32)
        * config.top1_bias_decay
        + config.top1_bias_learning_rate * expected_residual * observation
    ).clamp(0.0, config.top1_bias_clip)
    pq_kl = (
        target * (torch.log(target) - torch.log(draft))
    ).sum(dim=-1).clamp_min(0.0)
    return updated, expected_residual, pq_kl


def recover_unscaled_draft_probs(
    scaled_draft_probs: torch.Tensor,
    applied_log_scales: torch.Tensor,
) -> torch.Tensor:
    """Recover ``softmax(z)`` exactly from ``softmax(scale * z)``.

    The unknown additive logit normalizer cancels during renormalization.  This
    gives the no-feedback counterfactual for every realized MTP prefix without
    another drafter forward pass.
    """
    if scaled_draft_probs.ndim != 2:
        raise ValueError("scaled_draft_probs must be rank two")
    rows = scaled_draft_probs.shape[0]
    if applied_log_scales.shape != (rows,):
        raise ValueError("applied_log_scales must have one value per row")
    scaled = scaled_draft_probs.to(torch.float32).clamp_min(1e-30)
    scaled = scaled / scaled.sum(dim=-1, keepdim=True)
    inverse_scale = torch.exp(
        -applied_log_scales.to(device=scaled.device, dtype=torch.float32)
    )
    return torch.softmax(
        torch.log(scaled) * inverse_scale.unsqueeze(1),
        dim=-1,
    )


def repeated_ngram_continuation_ids(
    token_ids: list[int] | tuple[int, ...],
    ngram_size: int,
) -> tuple[int, ...]:
    """Return tokens that would repeat an existing suffix n-gram.

    This is the minimal no-repeat-n-gram signal: only continuations previously
    observed after the current ``n-1`` token suffix are returned.  The runtime
    uses it as a soft MTP-only penalty, never as a target-model ban.
    """
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least two")
    history = tuple(int(token_id) for token_id in token_ids)
    prefix_size = ngram_size - 1
    if len(history) < ngram_size:
        return ()
    suffix = history[-prefix_size:]
    continuations: set[int] = set()
    for start in range(len(history) - ngram_size + 1):
        if history[start : start + prefix_size] == suffix:
            continuations.add(history[start + prefix_size])
    return tuple(sorted(continuations))


def exact_root_anchor_gate(
    tv_debt: torch.Tensor | None,
    reference: float,
) -> torch.Tensor | None:
    """Enable exact head-1 anchoring only after material relaxed TV debt."""
    if reference <= 0.0:
        raise ValueError("root anchor reference must be positive")
    if tv_debt is None:
        return None
    return (tv_debt.to(torch.float32) >= reference).to(torch.float32)


def counterfactual_regret_scale_update(
    target_probs: torch.Tensor,
    scaled_draft_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    accepted_mask: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
    applied_log_scales: torch.Tensor,
    config: RegretFeedbackConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Learn a next-block scale whose counterfactual beats ``scale=1``.

    For each realized MTP prefix, the scaled distribution and applied scale
    determine the unscaled counterfactual exactly.  A positive
    ``KL(P||Q_base)-KL(P||Q_scaled)`` audits whether the action used for the
    current block helped.  Separately, a clipped Newton proposal is evaluated
    against the same exact base distribution.  It is used for the next block
    only if its own KL is strictly lower than the base KL.  This permits a
    harmful sharpening action to become a validated softening action instead
    of merely returning to one.  Updates remain gated by posterior
    causal-relaxation responsibility, so only future MTP proposals change.
    """
    if target_probs.shape != scaled_draft_probs.shape:
        raise ValueError("target_probs and draft_probs must have equal shape")
    rows = target_probs.shape[0]
    if candidate_ids.shape != (rows,):
        raise ValueError("candidate_ids must have one ID per probability row")
    if accepted_mask.shape != (rows,):
        raise ValueError("accepted_mask must have one value per draft head")
    if applied_log_scales.shape != (rows,):
        raise ValueError("applied_log_scales must have one value per draft head")

    target = target_probs.to(torch.float32).clamp_min(1e-30)
    scaled = scaled_draft_probs.to(torch.float32).clamp_min(1e-30)
    target = target / target.sum(dim=-1, keepdim=True)
    scaled = scaled / scaled.sum(dim=-1, keepdim=True)
    base = recover_unscaled_draft_probs(scaled, applied_log_scales)

    expected_residual = expected_causal_responsibility(
        scaled,
        candidate_ids,
        accepted_mask,
        target_candidate_probs,
        boosted_candidate_probs,
    )
    update_weight = (
        expected_residual / config.scale_gap_reference
    ).clamp(0.0, 1.0)

    log_target = torch.log(target)
    log_scaled = torch.log(scaled)
    log_base = torch.log(base.clamp_min(1e-30))
    scaled_kl = (target * (log_target - log_scaled)).sum(dim=-1)
    base_kl = (target * (log_target - log_base)).sum(dim=-1)
    counterfactual_gain = base_kl - scaled_kl

    mean_log_base = (base * log_base).sum(dim=-1)
    target_mean_log_base = (target * log_base).sum(dim=-1)
    gradient = mean_log_base - target_mean_log_base
    variance = (
        base * (log_base - mean_log_base.unsqueeze(1)).square()
    ).sum(dim=-1).clamp_min(1e-6)
    optimal_delta = (-gradient / variance).clamp(
        -config.scale_newton_clip,
        config.scale_newton_clip,
    )
    candidate_scale = (1.0 + optimal_delta).clamp(
        config.scale_min,
        config.scale_max,
    )
    candidate_log_scale = torch.log(candidate_scale)
    candidate = torch.softmax(
        log_base * candidate_scale.unsqueeze(1),
        dim=-1,
    ).clamp_min(1e-30)
    candidate_kl = (
        target * (log_target - torch.log(candidate))
    ).sum(dim=-1)
    candidate_gain = base_kl - candidate_kl

    # Validate the proposed action exactly.  A tiny tolerance prevents
    # floating-point ties from being treated as evidence for steering.
    helpful = candidate_gain > 1e-7
    desired = torch.where(
        helpful,
        candidate_log_scale,
        torch.zeros_like(candidate_log_scale),
    )
    retained = (
        applied_log_scales.to(device=target.device, dtype=torch.float32)
        * config.scale_decay
    )
    step = (config.scale_learning_rate * update_weight).clamp(0.0, 1.0)
    updated = retained + step * (desired - retained)
    updated = updated.clamp(
        math.log(config.scale_min),
        math.log(config.scale_max),
    )
    return (
        updated,
        expected_residual,
        scaled_kl.clamp_min(0.0),
        optimal_delta,
        counterfactual_gain,
    )


def counterfactual_tv_scale_update(
    target_probs: torch.Tensor,
    scaled_draft_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    accepted_mask: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
    applied_log_scales: torch.Tensor,
    config: RegretFeedbackConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Choose a next-block scale by exact target--draft TV reduction.

    Since strict speculative acceptance has expectation ``1-TV(P,Q)``, this
    objective is directly aligned with reducing dependence on relaxed
    acceptance.  The three-point scale set is intentionally small and bounded;
    the selected action is applied only after exact retrospective validation
    and only when causal-relaxation responsibility is nonzero.
    """
    if target_probs.shape != scaled_draft_probs.shape:
        raise ValueError("target_probs and draft_probs must have equal shape")
    rows = target_probs.shape[0]
    if candidate_ids.shape != (rows,):
        raise ValueError("candidate_ids must have one ID per probability row")
    if accepted_mask.shape != (rows,):
        raise ValueError("accepted_mask must have one value per draft head")
    if applied_log_scales.shape != (rows,):
        raise ValueError("applied_log_scales must have one value per draft head")

    target = target_probs.to(torch.float32).clamp_min(1e-30)
    scaled = scaled_draft_probs.to(torch.float32).clamp_min(1e-30)
    target = target / target.sum(dim=-1, keepdim=True)
    scaled = scaled / scaled.sum(dim=-1, keepdim=True)
    base = recover_unscaled_draft_probs(scaled, applied_log_scales)
    expected_residual = expected_causal_responsibility(
        scaled,
        candidate_ids,
        accepted_mask,
        target_candidate_probs,
        boosted_candidate_probs,
    )
    update_weight = (
        expected_residual / config.scale_gap_reference
    ).clamp(0.0, 1.0)

    scale_options = torch.as_tensor(
        [config.scale_min, 1.0, config.scale_max],
        device=target.device,
        dtype=torch.float32,
    )
    log_base = torch.log(base.clamp_min(1e-30))
    options = torch.softmax(
        log_base.unsqueeze(1) * scale_options.view(1, 3, 1),
        dim=-1,
    )
    option_tv = 0.5 * (
        options - target.unsqueeze(1)
    ).abs().sum(dim=-1)
    best_tv, best_indices = option_tv.min(dim=1)
    base_tv = 0.5 * (base - target).abs().sum(dim=-1)
    scaled_tv = 0.5 * (scaled - target).abs().sum(dim=-1)
    validated_gain = base_tv - best_tv
    applied_gain = base_tv - scaled_tv
    best_scales = scale_options.index_select(0, best_indices)
    desired = torch.where(
        validated_gain > 1e-7,
        torch.log(best_scales),
        torch.zeros_like(best_scales),
    )
    retained = (
        applied_log_scales.to(device=target.device, dtype=torch.float32)
        * config.scale_decay
    )
    step = (config.scale_learning_rate * update_weight).clamp(0.0, 1.0)
    updated = retained + step * (desired - retained)
    updated = updated.clamp(
        math.log(config.scale_min),
        math.log(config.scale_max),
    )
    pq_kl = (
        target * (torch.log(target) - torch.log(scaled))
    ).sum(dim=-1).clamp_min(0.0)
    return (
        updated,
        expected_residual,
        pq_kl,
        best_scales - 1.0,
        applied_gain,
    )


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
    event_trust: torch.Tensor | None = None,
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
    preferred_rows = output_weight[alt_ids].to(torch.float32)
    candidate_rows = output_weight[candidate_ids].to(torch.float32)
    if config.direction != "expected_residual":
        preferred_rows = F.normalize(
            preferred_rows,
            dim=-1,
            eps=1e-8,
        )
        candidate_rows = F.normalize(
            candidate_rows,
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
    if config.direction == "expected_trust":
        head_risk = (0.5 + 0.5 * reliability).clamp(0.5, 1.0)
    else:
        head_risk = (
            0.5
            + (reliability.amax() - reliability) / reliability_range
        ).clamp(0.5, 1.5)
    if event_trust is None:
        event_trust = torch.ones_like(allocated_tv)
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
        * event_trust.to(device=allocated_tv.device, dtype=allocated_tv.dtype)
        * distance_weight
        * causal_relaxed.to(allocated_tv.dtype)
    )
    raw_vector = (event_weight.unsqueeze(1) * directions).sum(dim=0)
    total_strength = event_weight.sum()
    vector_rms = torch.sqrt(raw_vector.square().mean()).clamp_min(1e-8)
    block_vector = raw_vector / vector_rms * total_strength
    return block_vector, total_strength, concentration


def hidden_residual_block_vector(
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    allocated_tv: torch.Tensor,
    causal_relaxed: torch.Tensor,
    accepted_count: torch.Tensor,
    hidden_similarity: torch.Tensor,
    config: RegretFeedbackConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a feature-space prediction-error residual for one block.

    Each row is the target verification hidden minus the MTP hidden that
    produced the draft distribution at the same position.  Unlike the token
    residual, this retains the full native-MTP feature error rather than
    treating a local vocabulary substitution as a future reasoning gradient.
    """
    row_vectors, event_weight = hidden_residual_block_rows(
        draft_hidden,
        target_hidden,
        allocated_tv,
        causal_relaxed,
        accepted_count,
        hidden_similarity,
        config,
    )
    raw_vector = row_vectors.sum(dim=0)
    total_strength = event_weight.sum()
    vector_rms = torch.sqrt(raw_vector.square().mean()).clamp_min(1e-8)
    block_vector = raw_vector / vector_rms * total_strength
    return block_vector, total_strength


def hidden_residual_block_rows(
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    allocated_tv: torch.Tensor,
    causal_relaxed: torch.Tensor,
    accepted_count: torch.Tensor,
    hidden_similarity: torch.Tensor,
    config: RegretFeedbackConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep native-MTP prediction errors separate for every MTP head."""
    if draft_hidden.shape != target_hidden.shape or draft_hidden.ndim != 2:
        raise ValueError("aligned draft/target hidden states are required")
    rows = draft_hidden.shape[0]
    residual = (
        target_hidden.to(torch.float32) - draft_hidden.to(torch.float32)
    )
    residual = residual / torch.sqrt(
        residual.square().mean(dim=-1, keepdim=True)
    ).clamp_min(1e-8)
    reliability = torch.as_tensor(
        config.head_reliability[:rows],
        device=residual.device,
        dtype=torch.float32,
    )
    reliability_range = (
        reliability.amax() - reliability.amin()
    ).clamp_min(1e-8)
    head_risk = (
        0.5
        + (reliability.amax() - reliability) / reliability_range
    ).clamp(0.5, 1.5)
    # A residual is useful only when the two states remain sufficiently
    # comparable.  Low cosine means "large error", but also makes the direction
    # itself less transferable to the next root state.
    feature_trust = (0.25 + 0.75 * hidden_similarity).clamp(0.25, 1.0)
    depth = torch.arange(rows, device=residual.device)
    distance = (
        accepted_count.to(depth.dtype) - 1 - depth
    ).clamp_min(0)
    distance_weight = torch.pow(
        torch.full_like(allocated_tv, config.token_decay),
        distance.to(allocated_tv.dtype),
    )
    event_weight = (
        allocated_tv
        * head_risk
        * feature_trust
        * distance_weight
        * causal_relaxed.to(allocated_tv.dtype)
    )
    return event_weight.unsqueeze(1) * residual, event_weight


def boundary_hidden_residual(
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    allocated_tv: torch.Tensor,
    expected_causal: torch.Tensor,
    accepted_count: torch.Tensor,
    rejected: torch.Tensor,
    hidden_similarity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use probability regret to gate the block-end feature correction.

    The signal is emitted only when the full draft block was committed.  This
    avoids transporting a feature error across a target correction token and
    keeps the residual adjacent to the next MTP block.
    """
    if draft_hidden.shape != target_hidden.shape or draft_hidden.ndim != 2:
        raise ValueError("aligned draft/target hidden states are required")
    rows = draft_hidden.shape[0]
    index = (accepted_count - 1).clamp(0, rows - 1).to(torch.int64)
    residual = (
        target_hidden[index].to(torch.float32)
        - draft_hidden[index].to(torch.float32)
    )
    residual = residual / torch.sqrt(
        residual.square().mean()
    ).clamp_min(1e-8)
    regret_mass = (
        allocated_tv.to(torch.float32)
        * expected_causal.to(torch.float32)
    ).sum()
    feature_trust = (
        0.25
        + 0.75
        * hidden_similarity[index].to(torch.float32)
    ).clamp(0.25, 1.0)
    full_block = (~rejected).to(torch.float32)
    strength = regret_mass * feature_trust * full_block
    return residual * strength, strength


def probability_residual_head_rows(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    output_weight: torch.Tensor,
    allocated_tv: torch.Tensor,
    expected_causal: torch.Tensor,
    head_reliability: tuple[float, ...],
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project each head's largest P-Q residuals into shared hidden space."""
    if target_probs.shape != draft_probs.shape or target_probs.ndim != 2:
        raise ValueError("aligned target/draft distributions are required")
    rows = target_probs.shape[0]
    k = min(top_k, target_probs.shape[1])
    probability_residual = (
        target_probs.to(torch.float32) - draft_probs.to(torch.float32)
    )
    _, ids = probability_residual.abs().topk(k, dim=-1)
    signed_mass = probability_residual.gather(1, ids)
    output_rows = output_weight[ids].to(torch.float32)
    gradient_rows = (
        signed_mass.unsqueeze(-1) * output_rows
    ).sum(dim=1)
    gradient_rows = gradient_rows / torch.sqrt(
        gradient_rows.square().mean(dim=-1, keepdim=True)
    ).clamp_min(1e-8)

    reliability = torch.as_tensor(
        head_reliability[:rows],
        device=target_probs.device,
        dtype=torch.float32,
    )
    event_weight = (
        allocated_tv.to(torch.float32)
        * expected_causal.to(torch.float32)
        * (0.5 + 0.5 * reliability)
    )
    return event_weight.unsqueeze(1) * gradient_rows, event_weight


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


def update_confirmed_hidden_memory(
    old_memory: torch.Tensor,
    old_confirmations: torch.Tensor,
    idle_blocks: torch.Tensor,
    block_vector: torch.Tensor,
    block_strength: torch.Tensor,
    committed_tokens: torch.Tensor,
    rejected: torch.Tensor,
    config: RegretFeedbackConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain a hidden residual only after consistent cross-block evidence."""
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
    retained_norm = torch.linalg.vector_norm(retained.to(torch.float32))
    block_norm = torch.linalg.vector_norm(block_vector.to(torch.float32))
    has_old = retained_norm > 1e-8
    has_new = (block_strength > 0.0) & (block_norm > 1e-8)
    alignment = (
        (retained.to(torch.float32) * block_vector.to(torch.float32)).sum()
        / (retained_norm * block_norm).clamp_min(1e-8)
    )
    aligned = (
        has_old
        & has_new
        & (alignment >= config.hidden_consistency_threshold)
        & ~rejected
    )
    restart = has_new & ~aligned

    memory = torch.where(
        aligned,
        retained + block_vector,
        torch.where(restart, block_vector, retained),
    )
    confirmations = torch.where(
        aligned,
        old_confirmations + 1,
        torch.where(
            restart,
            torch.ones_like(old_confirmations),
            old_confirmations,
        ),
    )
    confirmations = torch.where(
        rejected,
        torch.zeros_like(confirmations),
        confirmations,
    )
    next_idle = torch.where(
        has_new,
        torch.zeros_like(idle_blocks),
        idle_blocks + 1,
    )
    expired = (~has_new) & (next_idle >= config.max_idle_blocks)
    memory = torch.where(expired, torch.zeros_like(memory), memory)
    confirmations = torch.where(
        expired,
        torch.zeros_like(confirmations),
        confirmations,
    )
    return memory, confirmations, next_idle, alignment


def compatibility_gate(
    memory: torch.Tensor,
    output_weight: torch.Tensor,
    top_ids: torch.Tensor,
    top_probs: torch.Tensor,
    *,
    mode: str = "relative",
    current_hidden: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure whether regret preferentially lifts current high-P tokens.

    ``relative`` preserves the original rank-correlation-only behavior.
    ``positive_lift`` additionally requires the actual injected direction to
    increase both the target top-1 row and the probability-weighted target
    head.  This rejects a subtle failure mode where every supported token is
    pushed down, but the highest-probability token is pushed down less.
    """
    rows = F.normalize(
        output_weight[top_ids].to(torch.float32),
        dim=-1,
        eps=1e-8,
    )
    memory_direction = memory.to(torch.float32)
    if current_hidden is not None:
        source = current_hidden.to(
            device=memory_direction.device,
            dtype=torch.float32,
        )
        if source.ndim > 1:
            source = source.mean(dim=0)
        projection = (
            (source * memory_direction).sum()
            / source.square().sum().clamp_min(1e-8)
        )
        memory_direction = memory_direction - projection * source
    memory_direction = memory_direction / torch.sqrt(
        memory_direction.square().mean()
    ).clamp_min(1e-8)
    pushes = rows @ memory_direction
    weights = top_probs.to(torch.float32)
    weights = weights / weights.sum().clamp_min(1e-30)
    weighted_lift = (weights * pushes).sum()
    correlation = weighted_lift - pushes.mean()
    normalized = correlation / pushes.std(unbiased=False).clamp_min(1e-8)
    gate = normalized.clamp(0.0, 1.0)
    if mode == "positive_lift":
        positive = (weighted_lift > 0.0) & (pushes[0] > 0.0)
        gate = gate * positive.to(gate.dtype)
    elif mode != "relative":
        raise ValueError(f"unknown compatibility mode: {mode}")
    return gate, correlation


def transport_regret_to_target_head(
    memory: torch.Tensor,
    output_weight: torch.Tensor,
    top_ids: torch.Tensor,
    top_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-anchor stale regret in the current target-supported subspace.

    The previous residual is used only as a query over the current target
    top-k rows.  The returned direction therefore boosts target-supported
    continuations that remain aligned with the residual, instead of injecting
    a token direction tied literally to the previous position.
    """
    rows = F.normalize(
        output_weight[top_ids].to(torch.float32),
        dim=-1,
        eps=1e-8,
    )
    query = memory.to(device=rows.device, dtype=torch.float32)
    query = F.normalize(query, dim=0, eps=1e-8)
    scores = rows @ query
    probs = top_probs.to(device=rows.device, dtype=torch.float32)
    probs = probs / probs.sum().clamp_min(1e-30)
    mean_score = (probs * scores).sum()
    centered = scores - mean_score
    positive = centered.clamp_min(0.0)
    attention = probs * positive
    positive_mass = attention.sum()
    score_std = torch.sqrt(
        (probs * centered.square()).sum()
    ).clamp_min(1e-8)
    gate = (positive_mass / score_std).clamp(0.0, 1.0)

    preferred = (
        attention.unsqueeze(1) * rows
    ).sum(dim=0) / positive_mass.clamp_min(1e-8)
    target_center = (probs.unsqueeze(1) * rows).sum(dim=0)
    direction = preferred - target_center
    direction = torch.where(
        positive_mass > 1e-8,
        direction,
        torch.zeros_like(direction),
    )
    return direction, gate, positive_mass


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


def fusion_aligned_regret(
    embedding_regret: torch.Tensor,
    fc_weight: torch.Tensor,
) -> torch.Tensor:
    """Map an embedding-side direction into the MTP hidden-side FC input.

    Qwen3.5 MTP forms ``fc([norm(embedding), norm(target_hidden)])``.
    This transpose-Jacobian map produces a hidden-input direction whose
    first-order post-FC effect aligns with the desired embedding-side effect.
    """
    if embedding_regret.ndim != 1:
        raise ValueError("embedding regret must be one-dimensional")
    hidden_size = embedding_regret.shape[0]
    if fc_weight.ndim != 2 or fc_weight.shape != (
        hidden_size,
        2 * hidden_size,
    ):
        raise ValueError(
            "MTP fc weight must have shape [hidden,2*hidden]"
        )
    weight = fc_weight.to(
        device=embedding_regret.device,
        dtype=torch.float32,
    )
    regret = embedding_regret.to(torch.float32)
    embedding_weight = weight[:, :hidden_size]
    hidden_weight = weight[:, hidden_size:]
    post_fc_direction = embedding_weight @ regret
    return hidden_weight.transpose(0, 1) @ post_fc_direction


def fusion_preconditioned_regret(
    embedding_regret: torch.Tensor,
    fc_weight: torch.Tensor,
) -> torch.Tensor:
    """Jacobi-precondition the token residual before hidden-side injection.

    The plain transpose Jacobian overweights high-norm rows of the hidden
    channel.  Dividing the desired post-fusion feature change by the diagonal
    of ``W_hidden W_hidden.T`` is a cheap approximate least-squares inverse.
    """
    if embedding_regret.ndim != 1:
        raise ValueError("embedding regret must be one-dimensional")
    hidden_size = embedding_regret.shape[0]
    if fc_weight.ndim != 2 or fc_weight.shape != (
        hidden_size,
        2 * hidden_size,
    ):
        raise ValueError(
            "MTP fc weight must have shape [hidden,2*hidden]"
        )
    weight = fc_weight.to(
        device=embedding_regret.device,
        dtype=torch.float32,
    )
    embedding_weight = weight[:, :hidden_size]
    hidden_weight = weight[:, hidden_size:]
    desired_feature = embedding_weight @ embedding_regret.to(torch.float32)
    row_energy = hidden_weight.square().sum(dim=1).clamp_min(1e-8)
    return hidden_weight.transpose(0, 1) @ (desired_feature / row_energy)


def fusion_backproject_residual(
    feature_residual: torch.Tensor,
    fc_weight: torch.Tensor,
) -> torch.Tensor:
    """Back-project an MTP output error into the target-hidden FC input.

    Qwen3.5 MTP fuses ``[embedding, target_hidden]`` with one linear layer.
    For an observed post-fusion feature error ``e``, ``W_hidden.T @ e`` is the
    first-order target-hidden correction that most directly reduces that error.
    """
    if feature_residual.ndim != 1:
        raise ValueError("feature residual must be one-dimensional")
    hidden_size = feature_residual.shape[0]
    if fc_weight.ndim != 2 or fc_weight.shape != (
        hidden_size,
        2 * hidden_size,
    ):
        raise ValueError(
            "MTP fc weight must have shape [hidden,2*hidden]"
        )
    weight = fc_weight.to(
        device=feature_residual.device,
        dtype=torch.float32,
    )
    hidden_weight = weight[:, hidden_size:]
    return hidden_weight.transpose(0, 1) @ feature_residual.to(torch.float32)


def fusion_forward_direction(
    hidden_direction: torch.Tensor,
    fc_weight: torch.Tensor,
) -> torch.Tensor:
    """Project a target-hidden input direction to the MTP feature space."""
    if hidden_direction.ndim != 1:
        raise ValueError("hidden direction must be one-dimensional")
    hidden_size = hidden_direction.shape[0]
    if fc_weight.ndim != 2 or fc_weight.shape != (
        hidden_size,
        2 * hidden_size,
    ):
        raise ValueError(
            "MTP fc weight must have shape [hidden,2*hidden]"
        )
    weight = fc_weight.to(
        device=hidden_direction.device,
        dtype=torch.float32,
    )
    hidden_weight = weight[:, hidden_size:]
    return hidden_weight @ hidden_direction.to(torch.float32)


def regret_strength_gate(
    memory: torch.Tensor,
    reference: float,
) -> torch.Tensor:
    """Map retained TV-weighted regret magnitude to an injection multiplier.

    ``reference == 0`` preserves the original uncalibrated behavior.  A
    positive reference keeps the magnitude information that is encoded in the
    memory RMS instead of erasing it during direction normalization.
    """
    if reference <= 0.0:
        return torch.ones((), device=memory.device, dtype=torch.float32)
    memory_rms = torch.sqrt(memory.to(torch.float32).square().mean())
    return (memory_rms / reference).clamp(0.0, 1.0)


def _allocate_preserving_budget(
    capacities: torch.Tensor,
    priorities: torch.Tensor,
    total_budget: torch.Tensor,
) -> torch.Tensor:
    """Capped water filling that preserves every feasible scalar TV budget."""
    allocated = torch.zeros_like(capacities)
    remaining = torch.minimum(
        total_budget.clamp_min(0.0),
        capacities.sum(),
    )
    for _ in range(capacities.shape[0]):
        residual = (capacities - allocated).clamp_min(0.0)
        active = torch.where(
            residual > 1e-12,
            priorities.clamp_min(1e-12),
            torch.zeros_like(priorities),
        )
        proposed = remaining * active / active.sum().clamp_min(1e-30)
        step = torch.minimum(proposed, residual)
        allocated = allocated + step
        remaining = (remaining - step.sum()).clamp_min(0.0)
    return allocated


def regret_risk_redistribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    base_boosted: torch.Tensor,
    hidden_similarity: torch.Tensor,
    tv_debt: torch.Tensor,
    config: RegretFeedbackConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move, but never add, Exact-TV mass away from risky current positions.

    Zero regret is an exact identity. With regret, early candidates that have
    weak target support, low strict acceptance, and weak feature agreement lose
    priority. The same feasible per-block TV mass is water-filled into safer
    positions, so quality cannot improve merely by shrinking the relaxation.
    """
    rows = candidate_ids.shape[0]
    row_ids = torch.arange(rows, device=target_probs.device)
    ids = candidate_ids.to(device=target_probs.device, dtype=torch.int64)
    target = target_probs.to(torch.float32)
    draft = draft_probs.to(torch.float32)
    p_y = target[row_ids, ids]
    q_y = draft[row_ids, ids]
    base_tv = (base_boosted.to(torch.float32) - p_y).clamp_min(0.0)
    capacities = (q_y - p_y).clamp_min(0.0)
    risk_level = (
        tv_debt.to(device=p_y.device, dtype=torch.float32)
        / config.verifier_risk_reference
    ).clamp(0.0, 1.0)
    if config.verifier_risk_strength == 0.0:
        return base_boosted.to(torch.float32), risk_level, torch.zeros_like(p_y)

    top_p = target.amax(dim=-1)
    log_gap = (
        torch.log(top_p.clamp_min(1e-30))
        - torch.log(p_y.clamp_min(1e-30))
    ).clamp_min(0.0)
    target_support = torch.exp(-log_gap / 2.0)
    strict_acceptance = torch.minimum(
        torch.ones_like(p_y),
        p_y / q_y.clamp_min(1e-30),
    )
    hidden_trust = (
        0.25
        + 0.75
        * hidden_similarity.to(device=p_y.device, dtype=torch.float32)
    ).clamp(0.25, 1.0)
    prefix_impact = torch.linspace(
        1.0,
        0.35,
        rows,
        device=p_y.device,
        dtype=torch.float32,
    )
    risk = (
        prefix_impact
        * (1.0 - target_support)
        * (1.0 - strict_acceptance)
        * (1.25 - 0.25 * hidden_trust)
    ).clamp(0.0, 1.0)
    if config.verifier_policy in {"cactus_anchor", "cactus_tail"}:
        cactus_increment = torch.minimum(
            torch.sqrt(
                (
                    2.0
                    * p_y
                    * (1.0 - p_y)
                ).clamp_min(0.0)
            ),
            (1.0 - p_y).clamp_min(0.0),
        )
        cactus_cap = torch.minimum(cactus_increment, capacities)
        if config.verifier_policy == "cactus_tail":
            tail_level = (
                (risk_level - config.verifier_tail_threshold)
                / (1.0 - config.verifier_tail_threshold)
            ).clamp(0.0, 1.0)
            mix = (
                config.verifier_cactus_mix
                * config.verifier_risk_strength
                * tail_level
            ).clamp(0.0, 1.0)
            protected = (1.0 - mix) * base_tv + mix * cactus_cap
            return p_y + protected, risk_level, risk

        mix = (
            config.verifier_cactus_mix
            * config.verifier_risk_strength
            * risk_level
        ).clamp(0.0, 1.0)
        priorities = (
            (1.0 - mix) * base_tv
            + mix * cactus_cap
        ).clamp_min(1e-12)
        budget_scale = (
            1.0
            + config.verifier_budget_slope * (0.5 - risk_level)
        ).clamp(0.5, 1.5)
        target_budget = base_tv.sum() * budget_scale
        redistributed = _allocate_preserving_budget(
            capacities,
            priorities,
            target_budget,
        )
        redistributed = torch.where(
            risk_level > 0.0,
            redistributed,
            base_tv,
        )
        return p_y + redistributed, risk_level, risk

    suppression = torch.exp(
        -config.verifier_risk_strength * risk_level * risk
    )
    safe_bonus = (
        target_support
        * hidden_trust
        * (0.5 + 0.5 * strict_acceptance)
    )
    priorities = (
        base_tv.clamp_min(1e-12)
        * suppression
        * (1.0 + risk_level * safe_bonus)
    )
    redistributed = _allocate_preserving_budget(
        capacities,
        priorities,
        base_tv.sum(),
    )
    redistributed = torch.where(
        risk_level > 0.0,
        redistributed,
        base_tv,
    )
    return p_y + redistributed, risk_level, risk


def _capture_uniform_probs(*args: Any, **kwargs: Any) -> torch.Tensor:
    global _LAST_UNIFORM_PROBS

    original = getattr(_capture_uniform_probs, "_remtp_original")
    result = original(*args, **kwargs)
    _LAST_UNIFORM_PROBS = result
    return result


def build_sparse_vocab_residuals(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    event_gate: torch.Tensor,
    temperature: torch.Tensor,
    *,
    top_k: int,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build full, positive, negative, and candidate-only logit residuals.

    The tensors are raw-logit corrections. Dividing them by the sampling
    temperature therefore recovers the clipped ``log(P/Q)`` direction.
    """
    if target_probs.shape != draft_probs.shape or target_probs.ndim != 2:
        raise ValueError("target/draft probabilities must share [rows,vocab]")
    rows, vocab = target_probs.shape
    if candidate_ids.shape != (rows,) or event_gate.shape != (rows,):
        raise ValueError("candidate IDs/event gates must have one value per row")
    target = target_probs.to(torch.float32).clamp_min(1e-30)
    draft = draft_probs.to(torch.float32).clamp_min(1e-30)
    target = target / target.sum(dim=-1, keepdim=True)
    draft = draft / draft.sum(dim=-1, keepdim=True)
    log_ratio = (torch.log(target) - torch.log(draft)).clamp(-clip, clip)
    scale = temperature.to(
        device=target.device,
        dtype=torch.float32,
    ) * event_gate.to(device=target.device, dtype=torch.float32)
    scale = scale.unsqueeze(1)
    k = min(top_k, vocab)

    def sparse_from_score(score: torch.Tensor) -> torch.Tensor:
        _, ids = score.topk(k, dim=-1)
        values = log_ratio.gather(1, ids) * scale
        result = torch.zeros_like(target)
        result.scatter_(1, ids, values)
        return result

    full = sparse_from_score(log_ratio.abs())
    positive = sparse_from_score(log_ratio.clamp_min(0.0))
    negative = sparse_from_score((-log_ratio).clamp_min(0.0))
    candidate = torch.zeros_like(target)
    candidate_values = log_ratio.gather(
        1,
        candidate_ids.to(device=target.device, dtype=torch.int64).unsqueeze(1),
    ).clamp_max(0.0)
    candidate.scatter_(
        1,
        candidate_ids.to(device=target.device, dtype=torch.int64).unsqueeze(1),
        candidate_values * scale,
    )
    return full, positive, negative, candidate


def counterfactual_vocab_kl_gain(
    target_probs: torch.Tensor,
    base_draft_probs: torch.Tensor,
    source_bias: torch.Tensor,
    temperature: torch.Tensor,
    bias_scale: float,
) -> torch.Tensor:
    """Return [source,current] KL gains without changing sampled drafts."""
    if target_probs.shape != base_draft_probs.shape:
        raise ValueError("target/base draft probabilities must have equal shape")
    if source_bias.ndim != 2 or source_bias.shape[1] != target_probs.shape[1]:
        raise ValueError("source bias must have shape [source,vocab]")
    target = target_probs.to(torch.float32).clamp_min(1e-30)
    draft = base_draft_probs.to(torch.float32).clamp_min(1e-30)
    target = target / target.sum(dim=-1, keepdim=True)
    draft = draft / draft.sum(dim=-1, keepdim=True)
    base_kl = (
        target * (torch.log(target) - torch.log(draft))
    ).sum(dim=-1)
    safe_temperature = temperature.to(
        device=target.device,
        dtype=torch.float32,
    ).clamp_min(1e-5)
    result = torch.empty(
        (source_bias.shape[0], target.shape[0]),
        device=target.device,
        dtype=torch.float32,
    )
    base_log = torch.log(draft)
    for source_index in range(source_bias.shape[0]):
        adjusted = torch.softmax(
            base_log
            + bias_scale
            * source_bias[source_index].to(
                device=target.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            / safe_temperature,
            dim=-1,
        )
        adjusted_kl = (
            target * (torch.log(target) - torch.log(adjusted.clamp_min(1e-30)))
        ).sum(dim=-1)
        result[source_index] = base_kl - adjusted_kl
    return result


def update_vocab_head_route(
    previous_scores: torch.Tensor | None,
    route_gain: torch.Tensor,
    decay: float,
    min_gain: float,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Update a request-local source-to-head routing table."""
    if route_gain.ndim != 2 or route_gain.shape[0] != route_gain.shape[1]:
        raise ValueError("route gain must be a square [source,target] matrix")
    if not 0.0 <= decay < 1.0:
        raise ValueError("route decay must be in [0,1)")
    if min_gain < 0.0:
        raise ValueError("minimum route gain must be non-negative")
    if previous_scores is None:
        previous_scores = torch.zeros_like(route_gain)
    elif previous_scores.shape != route_gain.shape:
        raise ValueError("previous route scores must match route gain")
    updated = decay * previous_scores + (1.0 - decay) * route_gain
    best_gain, best_source = updated.max(dim=0)
    active_source = torch.where(
        best_gain > min_gain,
        best_source,
        torch.full_like(best_source, -1),
    )
    return updated, tuple(int(item) for item in active_source.tolist())


def _capture_bonus_logits(
    bonus_logits: torch.Tensor,
    sampling_metadata: Any,
) -> None:
    config = _require_config()
    if (
        config.direction == "none"
        or config.compatibility_source == "fresh_hidden"
        or config.direction in {
            "hidden_output",
            "headwise_output",
            "expected_scale",
            "expected_scale_adaptive_depth",
            "expected_scale_counterfactual",
            "expected_scale_repetition",
            "expected_scale_tv",
            "expected_top1_bias",
            "adaptive_depth",
            "expected_fusion",
            "expected_token_adaptive",
            "expected_token_preconditioned",
            "expected_hidden_fusion",
            "expected_hidden_consistent",
            "expected_hidden_adaptive",
            "expected_hidden_calibrated",
            "expected_root_anchor",
            "expected_root_exact",
            "expected_transport",
            "boundary_residual",
            "expected_logit",
            "expected_pq_gradient",
            "expected_vocab",
            "expected_vocab_audit",
            "expected_vocab_negative",
            "expected_vocab_headmap",
            "expected_vocab_online",
            "verifier_risk",
        }
    ):
        _STATE.bonus_ids = None
        _STATE.bonus_probs = None
        return
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


def _pre_verification(
    *,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    boosted_candidate_probs: torch.Tensor,
    hidden_similarity: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    del target_candidate_probs, sampling_metadata
    config = _require_config()
    if config.direction != "verifier_risk":
        return boosted_candidate_probs
    if _STATE.tv_debt is None:
        _STATE.tv_debt = torch.zeros(
            (),
            device=target_probs.device,
            dtype=torch.float32,
        )
    adjusted, risk_level, _ = regret_risk_redistribution(
        target_probs,
        draft_probs,
        draft_token_ids,
        boosted_candidate_probs,
        hidden_similarity,
        _STATE.tv_debt,
        config,
    )
    _AUDIT.ensure(config.expected_draft_tokens, target_probs.device)
    assert _AUDIT.injected is not None
    assert _AUDIT.gate_sum is not None
    _AUDIT.injected += (risk_level > 0.0).to(torch.float64)
    _AUDIT.gate_sum += risk_level.to(torch.float64)
    return adjusted


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
    if not 1 <= rows <= config.expected_draft_tokens:
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
    if _STATE.tv_debt is None:
        _STATE.tv_debt = torch.zeros(
            (),
            device=target_probs.device,
            dtype=torch.float32,
        )
    committed_tokens = accepted_count + 1
    debt_decay = torch.pow(
        torch.as_tensor(
            config.token_decay,
            device=target_probs.device,
            dtype=torch.float32,
        ),
        committed_tokens.to(torch.float32),
    )
    retained_debt = _STATE.tv_debt * debt_decay
    retained_debt = torch.where(
        rejected,
        retained_debt * config.rejection_reset,
        retained_debt,
    )
    new_debt = (
        allocated_tv
        * causal_relaxed.to(allocated_tv.dtype)
    ).sum()
    _STATE.tv_debt = retained_debt + new_debt

    expected_residual = torch.zeros_like(allocated_tv)
    pq_kl = torch.zeros_like(allocated_tv)
    scale_delta = torch.zeros_like(allocated_tv)
    counterfactual_tv_gain = torch.zeros_like(allocated_tv)
    if config.direction in {
        "expected_token",
        "expected_fusion",
        "expected_token_adaptive",
        "expected_token_preconditioned",
        "expected_hidden_fusion",
        "expected_hidden_consistent",
        "expected_hidden_adaptive",
        "expected_hidden_compatible",
        "expected_root_anchor",
        "expected_root_exact",
        "adaptive_depth",
        "expected_residual",
        "expected_transport",
        "boundary_residual",
        "expected_trust",
        "expected_logit",
        "expected_pq_gradient",
        "expected_vocab",
        "expected_vocab_audit",
        "expected_vocab_negative",
        "expected_vocab_headmap",
        "expected_vocab_online",
    }:
        expected_residual = expected_causal_responsibility(
            draft_probs,
            draft_token_ids,
            accepted,
            target_candidate_probs,
            boosted_candidate_probs,
        )
    elif config.direction == "expected_top1_bias":
        if _STATE.head_top1_bias is None:
            _STATE.head_top1_bias = torch.full(
                (config.expected_draft_tokens,),
                config.top1_bias_initial,
                device=target_probs.device,
                dtype=torch.float32,
            )
        updated_bias, expected_residual, pq_kl = expected_top1_bias_update(
            target_probs,
            draft_probs,
            draft_token_ids,
            accepted,
            target_candidate_probs,
            boosted_candidate_probs,
            _STATE.head_top1_bias[:rows],
            config,
        )
        next_bias = _STATE.head_top1_bias.clone()
        next_bias[:rows] = updated_bias
        _STATE.head_top1_bias = next_bias
    elif config.direction in {
        "expected_scale",
        "expected_scale_adaptive_depth",
        "expected_scale_counterfactual",
        "expected_scale_repetition",
        "expected_scale_tv",
        "expected_hidden_calibrated",
    }:
        if _STATE.head_log_scales is None:
            if config.scale_initial_by_reliability:
                reliability = torch.as_tensor(
                    config.head_reliability[:rows],
                    device=target_probs.device,
                    dtype=torch.float32,
                )
                initial_scales = 1.0 + (
                    config.scale_initial - 1.0
                ) * reliability
            else:
                initial_scales = torch.full(
                    (rows,),
                    config.scale_initial,
                    device=target_probs.device,
                    dtype=torch.float32,
                )
            _STATE.head_log_scales = torch.log(initial_scales).to(
                device=target_probs.device,
                dtype=torch.float32,
            )
        previous_log_scales = _STATE.head_log_scales
        active_log_scales = previous_log_scales[:rows]
        if config.direction == "expected_scale_counterfactual":
            (
                updated_log_scales,
                expected_residual,
                pq_kl,
                scale_delta,
                counterfactual_kl_gain,
            ) = counterfactual_regret_scale_update(
                target_probs,
                draft_probs,
                draft_token_ids,
                causal_relaxed if config.scale_causal_only else accepted,
                target_candidate_probs,
                boosted_candidate_probs,
                active_log_scales,
                config,
            )
        elif config.direction == "expected_scale_tv":
            (
                updated_log_scales,
                expected_residual,
                pq_kl,
                scale_delta,
                counterfactual_tv_gain,
            ) = counterfactual_tv_scale_update(
                target_probs,
                draft_probs,
                draft_token_ids,
                causal_relaxed if config.scale_causal_only else accepted,
                target_candidate_probs,
                boosted_candidate_probs,
                active_log_scales,
                config,
            )
        else:
            (
                updated_log_scales,
                expected_residual,
                pq_kl,
                scale_delta,
            ) = expected_regret_scale_update(
                target_probs,
                draft_probs,
                draft_token_ids,
                causal_relaxed if config.scale_causal_only else accepted,
                target_candidate_probs,
                boosted_candidate_probs,
                active_log_scales,
                config,
            )
        if previous_log_scales.shape[0] == rows:
            _STATE.head_log_scales = updated_log_scales
        else:
            merged_log_scales = previous_log_scales.clone()
            merged_log_scales[:rows] = updated_log_scales
            _STATE.head_log_scales = merged_log_scales
    if config.direction in {
        "adaptive_depth",
        "expected_scale_adaptive_depth",
    }:
        # A depth veto should be caused by evidence from the tail it removes,
        # not by unrelated residuals in early heads. Include the last retained
        # head as a boundary warning and every head that would be suppressed.
        tail_start = min(rows, max(0, config.adaptive_min_depth - 1))
        regret_strength = float(expected_residual[tail_start:].sum().item())
        if regret_strength >= config.adaptive_depth_trigger:
            _STATE.adaptive_depth_cooldown = max(
                _STATE.adaptive_depth_cooldown,
                config.adaptive_depth_blocks,
            )
    if config.direction in {
        "none",
        "expected_token",
        "expected_fusion",
        "expected_token_adaptive",
        "expected_token_preconditioned",
        "expected_hidden_fusion",
        "expected_hidden_consistent",
        "expected_hidden_adaptive",
        "expected_hidden_compatible",
        "expected_root_anchor",
        "expected_root_exact",
        "adaptive_depth",
        "expected_residual",
        "expected_transport",
        "boundary_residual",
        "expected_trust",
        "expected_logit",
        "expected_pq_gradient",
        "expected_vocab",
        "expected_vocab_audit",
        "expected_vocab_negative",
        "expected_vocab_headmap",
        "expected_vocab_online",
    }:
        target = target_probs.to(torch.float32).clamp_min(1e-30)
        draft = draft_probs.to(torch.float32).clamp_min(1e-30)
        target = target / target.sum(dim=-1, keepdim=True)
        draft = draft / draft.sum(dim=-1, keepdim=True)
        pq_kl = (
            target * (torch.log(target) - torch.log(draft))
        ).sum(dim=-1).clamp_min(0.0)

    if config.direction != "expected_scale_counterfactual":
        counterfactual_kl_gain = torch.zeros_like(allocated_tv)
    root_exact_rows = torch.zeros(
        (), device=target_probs.device, dtype=torch.float32
    )
    if config.direction == "expected_root_exact":
        if (
            _STATE.root_base_probs is not None
            and _STATE.root_exact_applied is not None
        ):
            target_root = target_probs[:1].to(torch.float32).clamp_min(1e-30)
            applied_root = draft_probs[:1].to(torch.float32).clamp_min(1e-30)
            base_root = _STATE.root_base_probs.to(
                device=target_probs.device,
                dtype=torch.float32,
            ).clamp_min(1e-30)
            target_root = target_root / target_root.sum(dim=-1, keepdim=True)
            applied_root = applied_root / applied_root.sum(
                dim=-1, keepdim=True
            )
            base_root = base_root / base_root.sum(dim=-1, keepdim=True)
            base_kl = (
                target_root
                * (torch.log(target_root) - torch.log(base_root))
            ).sum()
            applied_kl = (
                target_root
                * (torch.log(target_root) - torch.log(applied_root))
            ).sum()
            base_tv = 0.5 * (target_root - base_root).abs().sum()
            applied_tv = 0.5 * (target_root - applied_root).abs().sum()
            root_exact_rows = _STATE.root_exact_applied.to(
                device=target_probs.device,
                dtype=torch.float32,
            )
            counterfactual_kl_gain[0] = (
                base_kl - applied_kl
            ) * root_exact_rows
            counterfactual_tv_gain[0] = (
                base_tv - applied_tv
            ) * root_exact_rows
        _STATE.root_base_probs = None
        _STATE.root_exact_applied = None
    vocab_full_map_gain: torch.Tensor | None = None
    vocab_positive_map_gain: torch.Tensor | None = None
    vocab_negative_map_gain: torch.Tensor | None = None
    vocab_candidate_map_gain: torch.Tensor | None = None
    vocab_context_cosine: torch.Tensor | None = None
    vocab_draft_context_cosine: torch.Tensor | None = None
    vocab_context_gain = torch.zeros(
        (),
        device=target_probs.device,
        dtype=torch.float32,
    )
    vocab_draft_context_gain = torch.zeros_like(vocab_context_gain)
    vocab_oracle_gain = torch.zeros_like(vocab_context_gain)
    if config.direction == "expected_vocab":
        if len(_STATE.current_base_probs) != rows:
            raise RuntimeError(
                "expected-vocab feedback did not capture every base q row"
            )
        base_draft = torch.cat(_STATE.current_base_probs, dim=0).to(
            device=target_probs.device,
            dtype=torch.float32,
        )
        _STATE.current_base_probs.clear()
        target = target_probs.to(torch.float32).clamp_min(1e-30)
        applied_draft = draft_probs.to(torch.float32).clamp_min(1e-30)
        base_draft = base_draft.clamp_min(1e-30)
        target = target / target.sum(dim=-1, keepdim=True)
        applied_draft = (
            applied_draft / applied_draft.sum(dim=-1, keepdim=True)
        )
        base_draft = base_draft / base_draft.sum(dim=-1, keepdim=True)
        applied_kl = (
            target * (torch.log(target) - torch.log(applied_draft))
        ).sum(dim=-1)
        base_kl = (
            target * (torch.log(target) - torch.log(base_draft))
        ).sum(dim=-1)
        counterfactual_kl_gain = base_kl - applied_kl
        controller = (counterfactual_kl_gain >= 0.0).to(torch.float32)

        k = min(config.vocab_bias_top_k, target_probs.shape[-1])
        _, bias_ids = (target - applied_draft).abs().topk(k, dim=-1)
        log_ratio = (
            torch.log(target.gather(1, bias_ids))
            - torch.log(applied_draft.gather(1, bias_ids))
        ).clamp(-config.vocab_bias_clip, config.vocab_bias_clip)
        temperature = sampling_metadata.temperature[0].to(
            device=target_probs.device,
            dtype=torch.float32,
        )
        event_gate = (
            expected_residual / config.scale_gap_reference
        ).clamp(0.0, 1.0)
        bias_values = (
            temperature
            * log_ratio
            * event_gate.unsqueeze(1)
            * controller.unsqueeze(1)
        )
        _STATE.head_vocab_bias = torch.zeros_like(
            target_probs,
            dtype=torch.float32,
        )
        _STATE.head_vocab_bias.scatter_(1, bias_ids, bias_values)
    elif config.direction in {
        "expected_vocab_negative",
        "expected_vocab_headmap",
        "expected_vocab_online",
    }:
        if len(_STATE.current_base_probs) != rows:
            raise RuntimeError(
                "expected-vocab-negative feedback did not capture every q row"
            )
        base_draft = torch.cat(_STATE.current_base_probs, dim=0).to(
            device=target_probs.device,
            dtype=torch.float32,
        )
        _STATE.current_base_probs.clear()
        target = target_probs.to(torch.float32).clamp_min(1e-30)
        applied_draft = draft_probs.to(torch.float32).clamp_min(1e-30)
        base_draft = base_draft.clamp_min(1e-30)
        target = target / target.sum(dim=-1, keepdim=True)
        applied_draft = (
            applied_draft / applied_draft.sum(dim=-1, keepdim=True)
        )
        base_draft = base_draft / base_draft.sum(dim=-1, keepdim=True)
        applied_kl = (
            target * (torch.log(target) - torch.log(applied_draft))
        ).sum(dim=-1)
        base_kl = (
            target * (torch.log(target) - torch.log(base_draft))
        ).sum(dim=-1)
        counterfactual_kl_gain = base_kl - applied_kl
        temperature = sampling_metadata.temperature[0].to(
            device=target_probs.device,
            dtype=torch.float32,
        )
        previous_negative_bias = _STATE.head_vocab_negative_bias
        if (
            config.direction == "expected_vocab_online"
            and previous_negative_bias is not None
        ):
            route_gain = counterfactual_vocab_kl_gain(
                target,
                base_draft,
                previous_negative_bias,
                temperature,
                config.vocab_bias_scale,
            ).clamp(
                -config.vocab_route_gain_clip,
                config.vocab_route_gain_clip,
            )
            if (
                _STATE.vocab_route_scores is None
                and config.vocab_route_prior_gain > 0.0
            ):
                _STATE.vocab_route_scores = torch.zeros_like(route_gain)
                target_heads = torch.arange(
                    rows,
                    device=route_gain.device,
                )
                source_heads = torch.as_tensor(
                    config.vocab_head_map[:rows],
                    device=route_gain.device,
                    dtype=torch.int64,
                )
                _STATE.vocab_route_scores[
                    source_heads,
                    target_heads,
                ] = config.vocab_route_prior_gain
            (
                _STATE.vocab_route_scores,
                _STATE.vocab_head_route,
            ) = update_vocab_head_route(
                _STATE.vocab_route_scores,
                route_gain,
                config.vocab_route_decay,
                config.vocab_route_min_gain,
            )
        event_gate = (
            expected_residual / config.scale_gap_reference
        ).clamp(0.0, 1.0)
        _, _, negative_bias, _ = build_sparse_vocab_residuals(
            target,
            applied_draft,
            draft_token_ids.to(torch.int64),
            event_gate,
            temperature,
            top_k=config.vocab_bias_top_k,
            clip=config.vocab_bias_clip,
        )
        # Match the audit estimator exactly: inactive source heads contribute
        # zero and remain in the six-head mean instead of amplifying the few
        # active events.
        if config.direction == "expected_vocab_negative":
            _STATE.head_vocab_bias = negative_bias.mean(
                dim=0,
                keepdim=True,
            )
        else:
            _STATE.head_vocab_bias = negative_bias
            _STATE.head_vocab_negative_bias = negative_bias
            if (
                config.direction == "expected_vocab_online"
                and _STATE.vocab_head_route is None
            ):
                _STATE.vocab_head_route = tuple(
                    config.vocab_head_map[:rows]
                )
    elif config.direction == "expected_vocab_audit":
        if len(_STATE.current_base_probs) != rows:
            raise RuntimeError(
                "expected-vocab audit did not capture every base q row"
            )
        base_draft = torch.cat(_STATE.current_base_probs, dim=0).to(
            device=target_probs.device,
            dtype=torch.float32,
        )
        _STATE.current_base_probs.clear()
        target = target_probs.to(torch.float32).clamp_min(1e-30)
        target = target / target.sum(dim=-1, keepdim=True)
        base_draft = base_draft.clamp_min(1e-30)
        base_draft = base_draft / base_draft.sum(dim=-1, keepdim=True)
        temperature = sampling_metadata.temperature[0].to(
            device=target_probs.device,
            dtype=torch.float32,
        )

        previous_biases = (
            _STATE.head_vocab_bias,
            _STATE.head_vocab_positive_bias,
            _STATE.head_vocab_negative_bias,
            _STATE.head_vocab_candidate_bias,
        )
        if all(item is not None for item in previous_biases):
            assert _STATE.head_vocab_bias is not None
            assert _STATE.head_vocab_positive_bias is not None
            assert _STATE.head_vocab_negative_bias is not None
            assert _STATE.head_vocab_candidate_bias is not None
            vocab_full_map_gain = counterfactual_vocab_kl_gain(
                target,
                base_draft,
                _STATE.head_vocab_bias,
                temperature,
                config.vocab_bias_scale,
            )
            vocab_positive_map_gain = counterfactual_vocab_kl_gain(
                target,
                base_draft,
                _STATE.head_vocab_positive_bias,
                temperature,
                config.vocab_bias_scale,
            )
            vocab_negative_map_gain = counterfactual_vocab_kl_gain(
                target,
                base_draft,
                _STATE.head_vocab_negative_bias,
                temperature,
                config.vocab_bias_scale,
            )
            vocab_candidate_map_gain = counterfactual_vocab_kl_gain(
                target,
                base_draft,
                _STATE.head_vocab_candidate_bias,
                temperature,
                config.vocab_bias_scale,
            )

        from remtp.probabilistic_mtp import get_last_aligned_hidden_states

        current_draft_hidden, current_target_hidden = (
            get_last_aligned_hidden_states(rows)
        )
        if current_draft_hidden is None or current_target_hidden is None:
            raise RuntimeError(
                "expected-vocab audit requires aligned draft/target hidden states"
            )
        current_draft_hidden = current_draft_hidden.to(torch.float32)
        current_target_hidden = current_target_hidden.to(torch.float32)
        if (
            vocab_full_map_gain is not None
            and _STATE.source_target_hidden is not None
        ):
            source = torch.nn.functional.normalize(
                _STATE.source_target_hidden.to(
                    device=current_target_hidden.device,
                    dtype=torch.float32,
                ),
                dim=-1,
            )
            current = torch.nn.functional.normalize(
                current_target_hidden,
                dim=-1,
            )
            vocab_context_cosine = source @ current.transpose(0, 1)
            source_for_current = vocab_context_cosine.argmax(dim=0)
            current_indices = torch.arange(
                rows,
                device=target_probs.device,
            )
            vocab_context_gain = vocab_full_map_gain[
                source_for_current,
                current_indices,
            ].sum()
            draft_current = torch.nn.functional.normalize(
                current_draft_hidden,
                dim=-1,
            )
            vocab_draft_context_cosine = (
                source @ draft_current.transpose(0, 1)
            )
            draft_source_for_current = (
                vocab_draft_context_cosine.argmax(dim=0)
            )
            vocab_draft_context_gain = vocab_negative_map_gain[
                draft_source_for_current,
                current_indices,
            ].sum()
            vocab_oracle_gain = vocab_full_map_gain.max(dim=0).values.sum()

        event_gate = (
            expected_residual / config.scale_gap_reference
        ).clamp(0.0, 1.0)
        (
            _STATE.head_vocab_bias,
            _STATE.head_vocab_positive_bias,
            _STATE.head_vocab_negative_bias,
            _STATE.head_vocab_candidate_bias,
        ) = build_sparse_vocab_residuals(
            target,
            base_draft,
            draft_token_ids.to(torch.int64),
            event_gate,
            temperature,
            top_k=config.vocab_bias_top_k,
            clip=config.vocab_bias_clip,
        )
        _STATE.source_target_hidden = current_target_hidden.contiguous()

    top_probs: torch.Tensor | None = None
    top_ids: torch.Tensor | None = None
    concentration = torch.ones_like(allocated_tv)
    token_vector = torch.zeros(
        output_weight.shape[1],
        device=target_probs.device,
        dtype=torch.float32,
    )
    token_strength = torch.zeros(
        (),
        device=target_probs.device,
        dtype=torch.float32,
    )
    hidden_vector = torch.zeros_like(token_vector)
    hidden_strength = torch.zeros_like(token_strength)
    if config.direction in {
        "token",
        "expected_token",
        "expected_fusion",
        "expected_token_adaptive",
        "expected_token_preconditioned",
        "expected_residual",
        "expected_transport",
        "expected_trust",
        "expected_logit",
        "hidden",
        "hybrid",
        "expected_hidden_compatible",
    }:
        search_k = min(
            max(config.regret_top_k + 1, config.compatibility_top_k),
            target_probs.shape[-1],
        )
        top_probs, top_ids = target_probs.topk(search_k, dim=-1)
    if config.direction in {
        "token",
        "expected_token",
        "expected_fusion",
        "expected_token_adaptive",
        "expected_token_preconditioned",
        "expected_residual",
        "expected_transport",
        "expected_trust",
        "expected_logit",
        "hybrid",
    }:
        assert top_probs is not None and top_ids is not None
        token_event_weight = (
            expected_residual
            if config.direction in {
                "expected_token",
                "expected_fusion",
                "expected_token_adaptive",
                "expected_token_preconditioned",
                "expected_residual",
                "expected_transport",
                "expected_trust",
                "expected_logit",
            }
            else causal_relaxed
        )
        token_vector, token_strength, concentration = regret_block_vector(
            target_probs,
            draft_token_ids.to(torch.int64),
            allocated_tv,
            token_event_weight,
            accepted_count,
            output_weight,
            config,
            top_probs=top_probs,
            top_ids=top_ids,
            event_trust=(
                hidden_similarity.to(torch.float32).clamp(0.0, 1.0).square()
                if config.direction == "expected_trust"
                else None
            ),
        )
    hidden_rows = torch.zeros(
        (rows, output_weight.shape[1]),
        device=target_probs.device,
        dtype=torch.float32,
    )
    if config.direction == "expected_pq_gradient":
        hidden_rows, hidden_event_weight = probability_residual_head_rows(
            target_probs,
            draft_probs,
            output_weight,
            allocated_tv,
            expected_residual,
            config.head_reliability,
            config.regret_top_k,
        )
        hidden_strength = hidden_event_weight.sum()
        hidden_vector = hidden_rows.sum(dim=0)
        hidden_vector = hidden_vector / torch.sqrt(
            hidden_vector.square().mean()
        ).clamp_min(1e-8) * hidden_strength
    elif config.direction in {
        "hidden",
        "hidden_output",
        "headwise_output",
        "boundary_residual",
        "expected_hidden_fusion",
        "expected_hidden_consistent",
        "expected_hidden_adaptive",
        "expected_hidden_calibrated",
        "expected_hidden_compatible",
        "hybrid",
    }:
        from remtp.probabilistic_mtp import get_last_aligned_hidden_states

        draft_hidden, target_hidden = get_last_aligned_hidden_states(rows)
        if draft_hidden is None or target_hidden is None:
            raise RuntimeError(
                "hidden regret requires aligned MTP/target hidden capture"
            )
        if config.direction in {
            "boundary_residual",
            "expected_hidden_fusion",
            "expected_hidden_consistent",
            "expected_hidden_adaptive",
            "expected_hidden_calibrated",
            "expected_hidden_compatible",
        }:
            hidden_vector, hidden_strength = boundary_hidden_residual(
                draft_hidden,
                target_hidden,
                allocated_tv,
                expected_residual,
                accepted_count,
                rejected,
                hidden_similarity,
            )
        else:
            hidden_rows, hidden_event_weight = hidden_residual_block_rows(
                draft_hidden,
                target_hidden,
                allocated_tv,
                causal_relaxed,
                accepted_count,
                hidden_similarity,
                config,
            )
            hidden_strength = hidden_event_weight.sum()
            hidden_vector = hidden_rows.sum(dim=0)
            hidden_vector = hidden_vector / torch.sqrt(
                hidden_vector.square().mean()
            ).clamp_min(1e-8) * hidden_strength
    if config.direction in {
        "token",
        "expected_token",
        "expected_fusion",
        "expected_token_adaptive",
        "expected_token_preconditioned",
        "expected_residual",
        "expected_transport",
        "expected_trust",
        "expected_logit",
    }:
        block_vector, block_strength = token_vector, token_strength
    elif config.direction in {
        "hidden",
        "hidden_output",
        "headwise_output",
        "boundary_residual",
        "expected_hidden_fusion",
        "expected_hidden_consistent",
        "expected_hidden_adaptive",
        "expected_hidden_calibrated",
        "expected_hidden_compatible",
        "expected_pq_gradient",
    }:
        block_vector, block_strength = hidden_vector, hidden_strength
    elif config.direction == "hybrid":
        block_vector = 0.25 * token_vector + 0.75 * hidden_vector
        block_strength = 0.25 * token_strength + 0.75 * hidden_strength
        block_vector = block_vector / torch.sqrt(
            block_vector.square().mean()
        ).clamp_min(1e-8) * block_strength
    elif config.direction in {
        "expected_scale",
        "expected_scale_adaptive_depth",
        "expected_scale_counterfactual",
        "expected_scale_repetition",
        "expected_scale_tv",
        "expected_top1_bias",
    }:
        block_vector = token_vector
        block_strength = expected_residual.sum()
    else:
        block_vector, block_strength = token_vector, token_strength

    if _STATE.memory is None:
        _STATE.memory = torch.zeros_like(block_vector)
        _STATE.head_memory = torch.zeros_like(
            pad_head_rows(hidden_rows, config.expected_draft_tokens)
        )
        _STATE.idle_blocks = torch.zeros(
            (),
            device=block_vector.device,
            dtype=torch.int64,
        )
        _STATE.hidden_confirmation_count = torch.zeros(
            (),
            device=block_vector.device,
            dtype=torch.int64,
        )
    assert _STATE.idle_blocks is not None
    assert _STATE.head_memory is not None
    assert _STATE.hidden_confirmation_count is not None
    previous_idle = _STATE.idle_blocks
    if config.direction in {
        "expected_hidden_consistent",
        "expected_hidden_adaptive",
        "expected_hidden_calibrated",
        "expected_hidden_compatible",
        "expected_token_adaptive",
        "expected_token_preconditioned",
    }:
        (
            _STATE.memory,
            _STATE.hidden_confirmation_count,
            _STATE.idle_blocks,
            _STATE.hidden_last_alignment,
        ) = update_confirmed_hidden_memory(
            _STATE.memory,
            _STATE.hidden_confirmation_count,
            previous_idle,
            block_vector,
            block_strength,
            committed_tokens,
            rejected,
            config,
        )
    else:
        _STATE.memory, _STATE.idle_blocks = update_regret_memory(
            _STATE.memory,
            previous_idle,
            block_vector,
            block_strength,
            committed_tokens,
            rejected,
            config,
        )
    padded_hidden_rows = pad_head_rows(
        hidden_rows, config.expected_draft_tokens
    )
    _STATE.head_memory, _ = update_regret_memory(
        _STATE.head_memory,
        previous_idle,
        padded_hidden_rows,
        hidden_strength,
        committed_tokens,
        rejected,
        config,
    )

    if top_ids is not None and top_probs is not None:
        reference_index = accepted_count.clamp(max=rows - 1)
        target_reference_ids = top_ids[
            reference_index, : config.compatibility_top_k
        ]
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
    else:
        _STATE.compatibility_ids = None
        _STATE.compatibility_probs = None
    _STATE.bonus_ids = None
    _STATE.bonus_probs = None

    _AUDIT.ensure(config.expected_draft_tokens, target_probs.device)
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
    assert _AUDIT.pq_kl_sum is not None
    assert _AUDIT.pq_kl_rows is not None
    assert _AUDIT.expected_regret_sum is not None
    assert _AUDIT.abs_logit_scale_sum is not None
    assert _AUDIT.counterfactual_kl_gain_sum is not None
    assert _AUDIT.counterfactual_kl_rows is not None
    assert _AUDIT.counterfactual_tv_gain_sum is not None
    assert _AUDIT.counterfactual_tv_rows is not None
    assert _AUDIT.repetition_escape_blocks is not None
    assert _AUDIT.reduced_depth_blocks is not None
    assert _AUDIT.vocab_full_map_gain is not None
    assert _AUDIT.vocab_positive_map_gain is not None
    assert _AUDIT.vocab_negative_map_gain is not None
    assert _AUDIT.vocab_candidate_map_gain is not None
    assert _AUDIT.vocab_context_cosine is not None
    assert _AUDIT.vocab_draft_context_cosine is not None
    assert _AUDIT.vocab_context_gain_sum is not None
    assert _AUDIT.vocab_draft_context_gain_sum is not None
    assert _AUDIT.vocab_oracle_gain_sum is not None
    assert _AUDIT.vocab_audit_blocks is not None
    _AUDIT.rounds += 1
    _AUDIT.accepted += accepted.sum().to(torch.float64)
    _AUDIT.strict_accepted += strict_accepted.sum().to(torch.float64)
    _AUDIT.causal_relaxed += causal_relaxed.sum().to(torch.float64)
    _AUDIT.accepted_tv += (
        allocated_tv * accepted.to(allocated_tv.dtype)
    ).sum().to(torch.float64)
    _AUDIT.memory_strength += block_strength.to(torch.float64)
    _AUDIT.head_reached += pad_head_values(
        reached.to(torch.float64), config.expected_draft_tokens
    )
    _AUDIT.head_strict += pad_head_values(
        (reached & strict_accepted).to(torch.float64),
        config.expected_draft_tokens,
    )
    _AUDIT.head_causal += pad_head_values(
        causal_relaxed.to(torch.float64), config.expected_draft_tokens
    )
    _AUDIT.head_target_prob += (
        pad_head_values(
            (target_candidate_probs * reached).to(torch.float64),
            config.expected_draft_tokens,
        )
    )
    _AUDIT.head_hidden_cos += (
        pad_head_values(
            (hidden_similarity * reached).to(torch.float64),
            config.expected_draft_tokens,
        )
    )
    if config.direction in {
        "none",
        "expected_token",
        "expected_fusion",
        "expected_token_adaptive",
        "expected_token_preconditioned",
        "expected_hidden_fusion",
        "expected_hidden_consistent",
        "expected_hidden_adaptive",
        "expected_hidden_compatible",
        "expected_residual",
        "expected_transport",
        "boundary_residual",
        "expected_trust",
        "expected_logit",
        "expected_pq_gradient",
        "expected_vocab",
        "expected_vocab_audit",
        "expected_vocab_negative",
        "expected_vocab_headmap",
        "expected_vocab_online",
        "expected_scale",
        "expected_scale_adaptive_depth",
        "expected_scale_counterfactual",
        "expected_scale_repetition",
        "expected_scale_tv",
        "expected_top1_bias",
        "expected_hidden_calibrated",
        "expected_root_anchor",
        "expected_root_exact",
        "adaptive_depth",
    }:
        _AUDIT.pq_kl_sum += pq_kl.sum().to(torch.float64)
        _AUDIT.pq_kl_rows += torch.as_tensor(
            rows,
            device=target_probs.device,
            dtype=torch.float64,
        )
    _AUDIT.expected_regret_sum += expected_residual.sum().to(torch.float64)
    if _STATE.head_log_scales is not None:
        _AUDIT.abs_logit_scale_sum += (
            _STATE.head_log_scales.abs().mean().to(torch.float64)
        )
    if config.direction in {
        "expected_vocab",
        "expected_vocab_negative",
        "expected_vocab_headmap",
        "expected_vocab_online",
    }:
        _AUDIT.counterfactual_kl_gain_sum += (
            counterfactual_kl_gain.sum().to(torch.float64)
        )
        _AUDIT.counterfactual_kl_rows += torch.as_tensor(
            rows,
            device=target_probs.device,
            dtype=torch.float64,
        )
    elif config.direction == "expected_scale_counterfactual":
        reached_rows = reached.to(counterfactual_kl_gain.dtype)
        _AUDIT.counterfactual_kl_gain_sum += (
            counterfactual_kl_gain * reached_rows
        ).sum().to(torch.float64)
        _AUDIT.counterfactual_kl_rows += reached_rows.sum().to(torch.float64)
    elif config.direction == "expected_scale_tv":
        reached_rows = reached.to(counterfactual_tv_gain.dtype)
        _AUDIT.counterfactual_tv_gain_sum += (
            counterfactual_tv_gain * reached_rows
        ).sum().to(torch.float64)
        _AUDIT.counterfactual_tv_rows += reached_rows.sum().to(torch.float64)
    elif config.direction == "expected_root_exact":
        _AUDIT.counterfactual_kl_gain_sum += (
            counterfactual_kl_gain[0].to(torch.float64)
        )
        _AUDIT.counterfactual_kl_rows += root_exact_rows.to(torch.float64)
        _AUDIT.counterfactual_tv_gain_sum += (
            counterfactual_tv_gain[0].to(torch.float64)
        )
        _AUDIT.counterfactual_tv_rows += root_exact_rows.to(torch.float64)
    elif (
        config.direction == "expected_vocab_audit"
        and vocab_full_map_gain is not None
        and vocab_positive_map_gain is not None
        and vocab_negative_map_gain is not None
        and vocab_candidate_map_gain is not None
        and vocab_context_cosine is not None
        and vocab_draft_context_cosine is not None
    ):
        _AUDIT.vocab_full_map_gain += vocab_full_map_gain.to(torch.float64)
        _AUDIT.vocab_positive_map_gain += (
            vocab_positive_map_gain.to(torch.float64)
        )
        _AUDIT.vocab_negative_map_gain += (
            vocab_negative_map_gain.to(torch.float64)
        )
        _AUDIT.vocab_candidate_map_gain += (
            vocab_candidate_map_gain.to(torch.float64)
        )
        _AUDIT.vocab_context_cosine += (
            vocab_context_cosine.to(torch.float64)
        )
        _AUDIT.vocab_draft_context_cosine += (
            vocab_draft_context_cosine.to(torch.float64)
        )
        _AUDIT.vocab_context_gain_sum += (
            vocab_context_gain.to(torch.float64)
        )
        _AUDIT.vocab_draft_context_gain_sum += (
            vocab_draft_context_gain.to(torch.float64)
        )
        _AUDIT.vocab_oracle_gain_sum += vocab_oracle_gain.to(torch.float64)
        _AUDIT.vocab_audit_blocks += torch.ones(
            (),
            device=target_probs.device,
            dtype=torch.float64,
        )

    if config.diagnostics and not _DIAGNOSTIC_EMITTED:
        print(
            "[ReMTP][Regret][diagnostic] "
            f"accepted={accepted.tolist()} "
            f"strict={strict_accepted.tolist()} "
            f"causal={causal_relaxed.tolist()} "
            f"p={target_candidate_probs.tolist()} "
            f"h={boosted_candidate_probs.tolist()} "
            f"allocated_TV={allocated_tv.tolist()} "
            f"expected_residual={expected_residual.tolist()} "
            f"scale_delta={scale_delta.tolist()} "
            f"head_scales="
            f"{None if _STATE.head_log_scales is None else torch.exp(_STATE.head_log_scales).tolist()} "
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
    assert _AUDIT.pq_kl_sum is not None
    assert _AUDIT.pq_kl_rows is not None
    assert _AUDIT.expected_regret_sum is not None
    assert _AUDIT.abs_logit_scale_sum is not None
    assert _AUDIT.counterfactual_kl_gain_sum is not None
    assert _AUDIT.counterfactual_kl_rows is not None
    assert _AUDIT.counterfactual_tv_gain_sum is not None
    assert _AUDIT.counterfactual_tv_rows is not None
    assert _AUDIT.repetition_escape_blocks is not None
    assert _AUDIT.reduced_depth_blocks is not None
    assert _AUDIT.vocab_full_map_gain is not None
    assert _AUDIT.vocab_positive_map_gain is not None
    assert _AUDIT.vocab_negative_map_gain is not None
    assert _AUDIT.vocab_candidate_map_gain is not None
    assert _AUDIT.vocab_context_cosine is not None
    assert _AUDIT.vocab_draft_context_cosine is not None
    assert _AUDIT.vocab_context_gain_sum is not None
    assert _AUDIT.vocab_draft_context_gain_sum is not None
    assert _AUDIT.vocab_oracle_gain_sum is not None
    assert _AUDIT.vocab_audit_blocks is not None
    accepted = _AUDIT.accepted.item()
    strict_ratio = _AUDIT.strict_accepted.item() / max(accepted, 1.0)
    tv_per_accepted = _AUDIT.accepted_tv.item() / max(accepted, 1.0)
    injection_count = _AUDIT.injected.item()
    vocab_blocks = max(_AUDIT.vocab_audit_blocks.item(), 1.0)
    vocab_rows = vocab_blocks * _AUDIT.vocab_full_map_gain.shape[1]
    online_route = ""
    if _STATE.vocab_head_route is not None:
        route_scores = (
            None
            if _STATE.vocab_route_scores is None
            else _STATE.vocab_route_scores.max(dim=0).values.tolist()
        )
        online_route = (
            f" online_route={list(_STATE.vocab_head_route)}"
            f" online_route_score={route_scores}"
        )
    hidden_confirmation = ""
    if _STATE.hidden_confirmation_count is not None:
        last_alignment = (
            None
            if _STATE.hidden_last_alignment is None
            else _STATE.hidden_last_alignment.item()
        )
        hidden_confirmation = (
            f" hidden_confirmations="
            f"{_STATE.hidden_confirmation_count.item()}"
            f" hidden_last_alignment={last_alignment}"
        )
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
        f"mean_pq_kl="
        f"{_AUDIT.pq_kl_sum.item()/max(_AUDIT.pq_kl_rows.item(),1.0):.6f} "
        f"expected_regret_per_round="
        f"{_AUDIT.expected_regret_sum.item()/window_rounds:.6f} "
        f"mean_abs_logit_scale="
        f"{_AUDIT.abs_logit_scale_sum.item()/window_rounds:.6f} "
        f"counterfactual_kl_gain="
        f"{_AUDIT.counterfactual_kl_gain_sum.item()/max(_AUDIT.counterfactual_kl_rows.item(),1.0):.6f} "
        f"counterfactual_tv_gain="
        f"{_AUDIT.counterfactual_tv_gain_sum.item()/max(_AUDIT.counterfactual_tv_rows.item(),1.0):.6f} "
        f"repetition_escape_blocks="
        f"{_AUDIT.repetition_escape_blocks.item():.0f} "
        f"reduced_depth_blocks={_AUDIT.reduced_depth_blocks.item():.0f} "
        f"head_reached={_AUDIT.head_reached.tolist()} "
        f"head_strict_count={_AUDIT.head_strict.tolist()} "
        f"head_causal_count={_AUDIT.head_causal.tolist()} "
        f"head_target_p_sum={_AUDIT.head_target_prob.tolist()} "
        f"head_hidden_cos_sum={_AUDIT.head_hidden_cos.tolist()}"
        f"{online_route}{hidden_confirmation}",
        flush=True,
    )
    if _AUDIT.vocab_audit_blocks.item() > 0.0:
        print(
            "[ReMTP][Regret][vocab-audit] "
            f"blocks={_AUDIT.vocab_audit_blocks.item():.0f} "
            f"context_gain="
            f"{_AUDIT.vocab_context_gain_sum.item()/vocab_rows:.8f} "
            f"draft_context_negative_gain="
            f"{_AUDIT.vocab_draft_context_gain_sum.item()/vocab_rows:.8f} "
            f"oracle_gain="
            f"{_AUDIT.vocab_oracle_gain_sum.item()/vocab_rows:.8f} "
            f"full_map="
            f"{(_AUDIT.vocab_full_map_gain/vocab_blocks).tolist()} "
            f"positive_map="
            f"{(_AUDIT.vocab_positive_map_gain/vocab_blocks).tolist()} "
            f"negative_map="
            f"{(_AUDIT.vocab_negative_map_gain/vocab_blocks).tolist()} "
            f"candidate_map="
            f"{(_AUDIT.vocab_candidate_map_gain/vocab_blocks).tolist()} "
            f"context_cosine="
            f"{(_AUDIT.vocab_context_cosine/vocab_blocks).tolist()} "
            f"draft_context_cosine="
            f"{(_AUDIT.vocab_draft_context_cosine/vocab_blocks).tolist()}",
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
    if config.direction == "expected_hidden_calibrated":
        _STATE.suppress_scale_for_block = torch.zeros(
            (),
            device=getattr(self, "device", None),
            dtype=torch.bool,
        )

    lm_head = getattr(getattr(self, "model", None), "lm_head", None)
    output_weight = getattr(lm_head, "weight", None)
    if isinstance(output_weight, torch.Tensor):
        _STATE.output_weight = output_weight.detach()

    if config.direction == "expected_scale_repetition":
        _STATE.repetition_penalty_ids = None
        _STATE.repetition_gate = None
        sampling_metadata = kwargs.get("sampling_metadata")
        if sampling_metadata is None and len(args) >= 7:
            sampling_metadata = args[6]
        if sampling_metadata is None:
            raise RuntimeError(
                "repetition feedback requires sampling metadata"
            )
        if _STATE.tv_debt is not None:
            input_batch = getattr(
                getattr(self, "runner", None),
                "input_batch",
                None,
            )
            histories = getattr(input_batch, "req_output_token_ids", None)
            if histories is None:
                histories = sampling_metadata.output_token_ids
            if not histories:
                histories = [[]]
            continuation_ids = repeated_ngram_continuation_ids(
                histories[0],
                config.repetition_ngram_size,
            )
            if continuation_ids:
                _STATE.repetition_penalty_ids = torch.as_tensor(
                    continuation_ids,
                    device=getattr(self, "device", None),
                    dtype=torch.int64,
                )
                _STATE.repetition_gate = (
                    _STATE.tv_debt.to(torch.float32)
                    / config.repetition_debt_reference
                ).clamp(0.0, 1.0)
                _AUDIT.ensure(
                    config.expected_draft_tokens,
                    self.device,
                )
                assert _AUDIT.repetition_escape_blocks is not None
                _AUDIT.repetition_escape_blocks += torch.ones(
                    (),
                    device=_AUDIT.repetition_escape_blocks.device,
                    dtype=torch.float64,
                )

    if "target_hidden_states" in kwargs:
        target_hidden = kwargs["target_hidden_states"]
        hidden_location: tuple[str, int] = ("kwargs", -1)
    elif len(args) >= 3:
        target_hidden = args[2]
        hidden_location = ("args", 2)
    else:
        raise RuntimeError("EagleProposer.propose target hidden argument changed")

    if config.direction in {"expected_root_anchor", "expected_root_exact"}:
        _STATE.root_target_logits = None
        _STATE.root_anchor_gate = None
        _STATE.root_base_probs = None
        _STATE.root_exact_applied = None
        exact_gate = (
            exact_root_anchor_gate(
                _STATE.tv_debt,
                config.root_anchor_reference,
            )
            if config.direction == "expected_root_exact"
            else None
        )
        if config.direction == "expected_root_exact" and exact_gate is None:
            pass
        else:
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
            if selected.shape[0] != 1:
                raise RuntimeError("root anchor requires --max-num-seqs 1")
            _STATE.root_target_logits = self.model.compute_logits(
                selected
            ).detach()
        if config.direction == "expected_root_exact":
            _STATE.root_anchor_gate = exact_gate
    if (
        config.direction == "expected_root_anchor"
        and _STATE.root_target_logits is not None
    ):
        sampling_metadata = kwargs.get("sampling_metadata")
        if sampling_metadata is None:
            raise RuntimeError("root anchor requires sampling metadata")
        temperature = sampling_metadata.temperature[0].to(torch.float32)
        safe_temperature = torch.where(
            temperature < 1e-5,
            torch.ones_like(temperature),
            temperature,
        )
        target_confidence = torch.softmax(
            _STATE.root_target_logits[0].to(torch.float32)
            / safe_temperature,
            dim=-1,
        ).amax()
        confidence_gate = (
            (target_confidence - config.root_anchor_min_confidence)
            / (1.0 - config.root_anchor_min_confidence)
        ).clamp(0.0, 1.0)
        if _STATE.tv_debt is None:
            _STATE.root_anchor_gate = torch.zeros(
                (),
                device=target_hidden.device,
                dtype=torch.float32,
            )
        else:
            _STATE.root_anchor_gate = (
                _STATE.tv_debt.to(
                    device=target_hidden.device,
                    dtype=torch.float32,
                )
                / config.root_anchor_reference
            ).clamp(0.0, 1.0) * confidence_gate

    if (
        config.direction in {
            "expected_fusion",
            "expected_token_adaptive",
            "expected_token_preconditioned",
            "expected_hidden_fusion",
            "expected_hidden_consistent",
            "expected_hidden_adaptive",
            "expected_hidden_calibrated",
            "expected_hidden_compatible",
        }
        and _STATE.memory is not None
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
        mtp_core = getattr(getattr(self, "model", None), "model", None)
        fc = getattr(mtp_core, "fc", None)
        fc_weight = getattr(fc, "weight", None)
        if not isinstance(fc_weight, torch.Tensor):
            raise RuntimeError("Qwen3.5 MTP fusion FC weight is unavailable")
        if config.direction in {
            "expected_hidden_fusion",
            "expected_hidden_consistent",
            "expected_hidden_adaptive",
            "expected_hidden_calibrated",
            "expected_hidden_compatible",
        }:
            steering_memory = fusion_backproject_residual(
                _STATE.memory,
                fc_weight,
            )
        elif config.direction == "expected_token_preconditioned":
            steering_memory = fusion_preconditioned_regret(
                _STATE.memory,
                fc_weight,
            )
        else:
            steering_memory = fusion_aligned_regret(
                _STATE.memory,
                fc_weight,
            )
        gate = regret_strength_gate(
            _STATE.memory,
            config.strength_reference,
        )
        gate = torch.where(
            gate >= config.fusion_gate_threshold,
            gate,
            torch.zeros_like(gate),
        )
        if config.direction in {
            "expected_hidden_consistent",
            "expected_hidden_adaptive",
            "expected_hidden_calibrated",
            "expected_hidden_compatible",
            "expected_token_adaptive",
            "expected_token_preconditioned",
        }:
            if _STATE.hidden_confirmation_count is None:
                gate = torch.zeros_like(gate)
            else:
                confirmed = (
                    _STATE.hidden_confirmation_count
                    >= config.hidden_confirmation_blocks
                )
                if config.direction in {
                    "expected_hidden_adaptive",
                    "expected_hidden_calibrated",
                    "expected_hidden_compatible",
                    "expected_token_adaptive",
                    "expected_token_preconditioned",
                }:
                    observed = _STATE.hidden_confirmation_count > 0
                    confirmation_scale = torch.where(
                        confirmed,
                        torch.ones_like(gate),
                        torch.where(
                            observed,
                            torch.full_like(
                                gate,
                                config.hidden_unconfirmed_scale,
                            ),
                            torch.zeros_like(gate),
                        ),
                    )
                else:
                    confirmation_scale = confirmed.to(gate.dtype)
                gate = gate * confirmation_scale
        if config.direction == "expected_hidden_compatible":
            if not isinstance(output_weight, torch.Tensor):
                gate = torch.zeros_like(gate)
            else:
                if config.compatibility_source == "fresh_hidden":
                    if selected.shape[0] != 1:
                        raise RuntimeError(
                            "fresh hidden compatibility requires "
                            "--max-num-seqs 1"
                        )
                    current_logits = self.model.compute_logits(selected)[0]
                    k = min(
                        config.compatibility_top_k,
                        current_logits.shape[-1],
                    )
                    values, compatibility_ids = current_logits.topk(k)
                    sampling_metadata = kwargs.get("sampling_metadata")
                    if sampling_metadata is None:
                        raise RuntimeError(
                            "fresh hidden compatibility requires sampling metadata"
                        )
                    temperature = sampling_metadata.temperature[0].to(
                        torch.float32
                    )
                    safe_temperature = torch.where(
                        temperature < 1e-5,
                        torch.ones_like(temperature),
                        temperature,
                    )
                    compatibility_probs = torch.softmax(
                        values.to(torch.float32) / safe_temperature,
                        dim=-1,
                    )
                else:
                    compatibility_ids = _STATE.compatibility_ids
                    compatibility_probs = _STATE.compatibility_probs
                if compatibility_ids is None or compatibility_probs is None:
                    gate = torch.zeros_like(gate)
                    compatibility = torch.zeros_like(gate)
                    correlation = torch.zeros_like(gate)
                else:
                    post_fusion_direction = fusion_forward_direction(
                        steering_memory,
                        fc_weight,
                    )
                    compatibility, correlation = compatibility_gate(
                        post_fusion_direction,
                        output_weight,
                        compatibility_ids,
                        compatibility_probs,
                        mode="positive_lift",
                    )
                    gate = gate * compatibility
                compatible = correlation > 0.0
                _STATE.memory = torch.where(
                    compatible,
                    _STATE.memory,
                    _STATE.memory * config.incompatible_decay,
                )
                if _STATE.hidden_confirmation_count is not None:
                    _STATE.hidden_confirmation_count = torch.where(
                        compatible,
                        _STATE.hidden_confirmation_count,
                        torch.zeros_like(_STATE.hidden_confirmation_count),
                    )
        if config.direction == "expected_hidden_calibrated":
            _STATE.suppress_scale_for_block = gate > 0.0
        modified = inject_regret(
            selected,
            steering_memory,
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

    fresh_compatibility = (
        config.compatibility_source == "fresh_hidden"
        or config.direction == "expected_transport"
    )
    has_compatibility = (
        config.direction == "boundary_residual"
        or fresh_compatibility
        or (
            _STATE.compatibility_ids is not None
            and _STATE.compatibility_probs is not None
        )
    )
    if (
        config.direction not in {
            "hidden_output",
            "headwise_output",
            "expected_scale",
            "expected_scale_adaptive_depth",
            "expected_scale_counterfactual",
            "expected_scale_repetition",
            "expected_scale_tv",
            "expected_top1_bias",
            "adaptive_depth",
            "expected_logit",
            "expected_pq_gradient",
            "expected_vocab",
            "expected_vocab_audit",
            "expected_vocab_negative",
            "expected_vocab_headmap",
            "expected_vocab_online",
            "expected_fusion",
            "expected_token_adaptive",
            "expected_token_preconditioned",
            "expected_hidden_fusion",
            "expected_hidden_consistent",
            "expected_hidden_adaptive",
            "expected_hidden_calibrated",
            "expected_hidden_compatible",
            "expected_root_anchor",
            "expected_root_exact",
            "verifier_risk",
        }
        and
        _STATE.memory is not None
        and _STATE.output_weight is not None
        and has_compatibility
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
        if selected.shape[0] != 1:
            raise RuntimeError(
                "fresh regret compatibility requires --max-num-seqs 1"
            )
        if config.direction == "boundary_residual":
            steering_memory = _STATE.memory
            gate = regret_strength_gate(
                _STATE.memory,
                config.strength_reference,
            )
            correlation = gate
        else:
            if fresh_compatibility:
                current_logits = self.model.compute_logits(selected)[0]
                k = min(config.compatibility_top_k, current_logits.shape[-1])
                values, compatibility_ids = current_logits.topk(k)
                sampling_metadata = kwargs.get("sampling_metadata")
                if sampling_metadata is None:
                    raise RuntimeError(
                        "fresh regret compatibility requires sampling metadata"
                    )
                temperature = sampling_metadata.temperature[0].to(
                    torch.float32
                )
                safe_temperature = torch.where(
                    temperature < 1e-5,
                    torch.ones_like(temperature),
                    temperature,
                )
                compatibility_probs = torch.softmax(
                    values.to(torch.float32) / safe_temperature,
                    dim=-1,
                )
            else:
                assert _STATE.compatibility_ids is not None
                assert _STATE.compatibility_probs is not None
                compatibility_ids = _STATE.compatibility_ids
                compatibility_probs = _STATE.compatibility_probs
            if config.direction == "expected_transport":
                steering_memory, gate, correlation = (
                    transport_regret_to_target_head(
                        _STATE.memory,
                        _STATE.output_weight,
                        compatibility_ids,
                        compatibility_probs,
                    )
                )
            else:
                steering_memory = _STATE.memory
                gate, correlation = compatibility_gate(
                    _STATE.memory,
                    _STATE.output_weight,
                    compatibility_ids,
                    compatibility_probs,
                    mode=config.compatibility_mode,
                    current_hidden=selected,
                )
            gate = gate * regret_strength_gate(
                _STATE.memory,
                config.strength_reference,
            )
        positive = correlation > 0.0
        _STATE.memory = torch.where(
            positive,
            _STATE.memory,
            _STATE.memory * config.incompatible_decay,
        )
        modified = inject_regret(
            selected,
            steering_memory,
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

    if (
        config.direction in {"adaptive_depth", "expected_scale_adaptive_depth"}
        and _STATE.adaptive_depth_cooldown > 0
    ):
        original_depth = self.num_speculative_tokens
        self.num_speculative_tokens = config.adaptive_min_depth
        _STATE.adaptive_depth_cooldown -= 1
        _AUDIT.ensure(config.expected_draft_tokens, target_hidden.device)
        assert _AUDIT.reduced_depth_blocks is not None
        _AUDIT.reduced_depth_blocks += torch.ones(
            (),
            device=_AUDIT.reduced_depth_blocks.device,
            dtype=torch.float64,
        )
        try:
            return original(self, *args, **kwargs)
        finally:
            self.num_speculative_tokens = original_depth
    return original(self, *args, **kwargs)


def _calibrate_draft_logits(
    logits: torch.Tensor,
    proposal_depth: int,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Apply the previous block's scalar calibration to one MTP head."""
    config = _require_config()
    if config.direction in {"expected_root_anchor", "expected_root_exact"}:
        if (
            proposal_depth != 0
            or _STATE.root_target_logits is None
            or _STATE.root_anchor_gate is None
        ):
            return logits
        if config.direction == "expected_root_exact":
            mix = _STATE.root_anchor_gate.to(
                device=logits.device,
                dtype=torch.float32,
            )
            temperature = temperatures[0].to(
                device=logits.device,
                dtype=torch.float32,
            )
            safe_temperature = torch.where(
                temperature < 1e-5,
                torch.ones_like(temperature),
                temperature,
            )
            _STATE.root_base_probs = torch.softmax(
                logits.to(torch.float32) / safe_temperature,
                dim=-1,
            ).contiguous()
            _STATE.root_exact_applied = mix > 0.0
        else:
            mix = (
                config.root_anchor_mix
                * _STATE.root_anchor_gate.to(
                    device=logits.device,
                    dtype=torch.float32,
                )
            ).clamp(0.0, 1.0)
        target_logits = _STATE.root_target_logits.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        anchored = torch.lerp(logits, target_logits, mix.to(logits.dtype))
        _STATE.root_target_logits = None
        _STATE.root_anchor_gate = None
        _AUDIT.ensure(config.expected_draft_tokens, logits.device)
        assert _AUDIT.injected is not None
        assert _AUDIT.gate_sum is not None
        _AUDIT.injected += (mix > 0.0).to(torch.float64)
        _AUDIT.gate_sum += mix.to(torch.float64)
        return anchored
    if config.direction in {
        "expected_vocab",
        "expected_vocab_audit",
        "expected_vocab_negative",
        "expected_vocab_headmap",
        "expected_vocab_online",
    }:
        temperature = temperatures[0].to(
            device=logits.device,
            dtype=torch.float32,
        )
        safe_temperature = torch.where(
            temperature < 1e-5,
            torch.ones_like(temperature),
            temperature,
        )
        base_probs = torch.softmax(
            logits.to(torch.float32) / safe_temperature,
            dim=-1,
        )
        _STATE.current_base_probs.append(base_probs.contiguous())
        if config.direction == "expected_vocab_audit":
            return logits
        if config.direction == "expected_vocab_negative":
            bias_index = 0
        elif config.direction == "expected_vocab_headmap":
            bias_index = config.vocab_head_map[proposal_depth]
        elif config.direction == "expected_vocab_online":
            if _STATE.vocab_head_route is None:
                return logits
            bias_index = _STATE.vocab_head_route[proposal_depth]
            if bias_index < 0:
                return logits
        else:
            bias_index = proposal_depth
        if (
            _STATE.head_vocab_bias is None
            or bias_index >= _STATE.head_vocab_bias.shape[0]
        ):
            return logits
        bias = _STATE.head_vocab_bias[bias_index].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        active = bias.abs().amax() > 0.0
        _AUDIT.ensure(config.expected_draft_tokens, logits.device)
        assert _AUDIT.injected is not None
        assert _AUDIT.gate_sum is not None
        _AUDIT.injected += active.to(torch.float64)
        _AUDIT.gate_sum += (
            bias.abs().amax().to(torch.float64)
            * config.vocab_bias_scale
        )
        return logits + config.vocab_bias_scale * bias.unsqueeze(0)
    if config.direction == "expected_top1_bias":
        if _STATE.head_top1_bias is None:
            _STATE.head_top1_bias = torch.full(
                (config.expected_draft_tokens,),
                config.top1_bias_initial,
                device=logits.device,
                dtype=torch.float32,
            )
        if proposal_depth >= _STATE.head_top1_bias.shape[0]:
            return logits
        bias = _STATE.head_top1_bias[proposal_depth].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        top_ids = logits.argmax(dim=-1, keepdim=True)
        calibrated = logits.clone()
        calibrated.scatter_add_(
            1,
            top_ids,
            bias.expand(logits.shape[0], 1),
        )
        _AUDIT.ensure(config.expected_draft_tokens, logits.device)
        assert _AUDIT.injected is not None
        assert _AUDIT.gate_sum is not None
        _AUDIT.injected += (bias.abs() > 1e-8).to(torch.float64)
        _AUDIT.gate_sum += bias.abs().to(torch.float64)
        return calibrated
    if config.direction not in {
        "expected_scale",
        "expected_scale_adaptive_depth",
        "expected_scale_counterfactual",
        "expected_scale_repetition",
        "expected_scale_tv",
        "expected_hidden_calibrated",
    }:
        return logits
    if (
        config.direction == "expected_scale_counterfactual"
        and _STATE.head_log_scales is None
    ):
        if config.scale_initial_by_reliability:
            reliability = torch.as_tensor(
                config.head_reliability[: config.expected_draft_tokens],
                device=logits.device,
                dtype=torch.float32,
            )
            initial_scales = 1.0 + (
                config.scale_initial - 1.0
            ) * reliability
        else:
            initial_scales = torch.full(
                (config.expected_draft_tokens,),
                config.scale_initial,
                device=logits.device,
                dtype=torch.float32,
            )
        _STATE.head_log_scales = torch.log(initial_scales)
    if (
        _STATE.head_log_scales is None
        or proposal_depth >= _STATE.head_log_scales.shape[0]
    ):
        return logits
    log_scale = _STATE.head_log_scales[proposal_depth].to(
        device=logits.device,
        dtype=torch.float32,
    )
    scale = torch.exp(log_scale).to(dtype=logits.dtype)
    if (
        config.direction == "expected_hidden_calibrated"
        and _STATE.suppress_scale_for_block is not None
    ):
        scale = torch.where(
            _STATE.suppress_scale_for_block.to(device=logits.device),
            torch.ones_like(scale),
            scale,
        )
    _AUDIT.ensure(config.expected_draft_tokens, logits.device)
    assert _AUDIT.injected is not None
    assert _AUDIT.gate_sum is not None
    active = log_scale.abs() > 1e-8
    _AUDIT.injected += active.to(torch.float64)
    _AUDIT.gate_sum += log_scale.abs().to(torch.float64)
    calibrated = logits * scale
    if (
        config.direction == "expected_scale_repetition"
        and _STATE.repetition_penalty_ids is not None
        and _STATE.repetition_gate is not None
    ):
        penalty_ids = _STATE.repetition_penalty_ids.to(logits.device)
        penalty_ids = penalty_ids[
            (penalty_ids >= 0) & (penalty_ids < logits.shape[-1])
        ]
        if penalty_ids.numel() > 0:
            gate = _STATE.repetition_gate.to(
                device=logits.device,
                dtype=logits.dtype,
            )
            calibrated = calibrated.clone()
            calibrated[:, penalty_ids] -= config.repetition_penalty * gate
            _AUDIT.injected += torch.ones(
                (), device=logits.device, dtype=torch.float64
            )
            _AUDIT.gate_sum += gate.to(torch.float64)
    return calibrated


def _correct_first_draft_hidden(
    hidden_states: torch.Tensor,
    proposal_depth: int,
) -> torch.Tensor:
    """Apply one-step feature-error feedback at the MTP output interface."""
    config = _require_config()
    if config.direction == "hidden_output":
        if proposal_depth != 0 or _STATE.memory is None:
            return hidden_states
        memory = _STATE.memory
    elif config.direction == "headwise_output":
        if (
            _STATE.head_memory is None
            or proposal_depth >= _STATE.head_memory.shape[0]
        ):
            return hidden_states
        memory = _STATE.head_memory[proposal_depth]
    elif config.direction == "expected_pq_gradient":
        if (
            _STATE.head_memory is None
            or proposal_depth >= _STATE.head_memory.shape[0]
        ):
            return hidden_states
        memory = _STATE.head_memory[proposal_depth]
    elif config.direction == "expected_logit":
        if (
            _STATE.memory is None
            or proposal_depth >= config.expected_draft_tokens
        ):
            return hidden_states
        memory = _STATE.memory
    else:
        return hidden_states
    gate = regret_strength_gate(
        memory,
        config.strength_reference,
    )
    if config.direction in {"expected_logit", "expected_pq_gradient"}:
        head_trust = torch.as_tensor(
            config.head_reliability[proposal_depth],
            device=hidden_states.device,
            dtype=torch.float32,
        )
        gate = gate * (0.5 + 0.5 * head_trust)
    corrected = inject_regret(
        hidden_states,
        memory,
        gate,
        config.alpha,
    )
    _AUDIT.ensure(config.expected_draft_tokens, hidden_states.device)
    assert _AUDIT.injected is not None
    assert _AUDIT.gate_sum is not None
    _AUDIT.injected += (gate > 0.0).to(torch.float64)
    _AUDIT.gate_sum += gate.to(torch.float64)
    return corrected


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

    from remtp.probabilistic_mtp import (
        set_bonus_logits_hook,
        set_draft_hidden_hook,
        set_draft_logits_hook,
    )
    from remtp.target_anchored_mtp import (
        set_post_verification_hook,
        set_pre_verification_hook,
    )

    set_bonus_logits_hook(_capture_bonus_logits)
    set_draft_hidden_hook(_correct_first_draft_hidden)
    set_draft_logits_hook(_calibrate_draft_logits)
    set_pre_verification_hook(_pre_verification)
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
        f"direction={config.direction} "
        f"draft_tokens={config.expected_draft_tokens} "
        f"top_k={config.regret_top_k} "
        f"compat_top_k={config.compatibility_top_k} "
        f"compat_mode={config.compatibility_mode} "
        f"compat_source={config.compatibility_source} "
        f"alpha={config.alpha:g} "
        f"strength_ref={config.strength_reference:g} "
        f"scale_lr={config.scale_learning_rate:g} "
        f"scale_decay={config.scale_decay:g} "
        f"scale_gap_ref={config.scale_gap_reference:g} "
        f"scale_newton_clip={config.scale_newton_clip:g} "
        f"scale_bounds={config.scale_min:g}:{config.scale_max:g} "
        f"scale_initial={config.scale_initial:g} "
        f"scale_initial_by_reliability="
        f"{int(config.scale_initial_by_reliability)} "
        f"scale_causal_only={int(config.scale_causal_only)} "
        f"top1_bias_initial={config.top1_bias_initial:g} "
        f"top1_bias_lr={config.top1_bias_learning_rate:g} "
        f"top1_bias_decay={config.top1_bias_decay:g} "
        f"top1_bias_clip={config.top1_bias_clip:g} "
        f"repetition_ngram={config.repetition_ngram_size} "
        f"repetition_penalty={config.repetition_penalty:g} "
        f"repetition_debt_ref={config.repetition_debt_reference:g} "
        f"adaptive_depth={config.adaptive_min_depth} "
        f"adaptive_blocks={config.adaptive_depth_blocks} "
        f"adaptive_trigger={config.adaptive_depth_trigger:g} "
        f"root_anchor_mix={config.root_anchor_mix:g} "
        f"root_anchor_reference={config.root_anchor_reference:g} "
        f"root_anchor_min_confidence="
        f"{config.root_anchor_min_confidence:g} "
        f"vocab_bias_scale={config.vocab_bias_scale:g} "
        f"vocab_bias_clip={config.vocab_bias_clip:g} "
        f"vocab_bias_top_k={config.vocab_bias_top_k} "
        f"vocab_head_map={','.join(str(x) for x in config.vocab_head_map)} "
        f"vocab_route_decay={config.vocab_route_decay:g} "
        f"vocab_route_min_gain={config.vocab_route_min_gain:g} "
        f"vocab_route_prior_gain={config.vocab_route_prior_gain:g} "
        f"vocab_route_gain_clip={config.vocab_route_gain_clip:g} "
        f"fusion_gate_threshold={config.fusion_gate_threshold:g} "
        f"hidden_consistency_threshold={config.hidden_consistency_threshold:g} "
        f"hidden_confirmation_blocks={config.hidden_confirmation_blocks} "
        f"hidden_unconfirmed_scale={config.hidden_unconfirmed_scale:g} "
        f"verifier_risk_strength={config.verifier_risk_strength:g} "
        f"verifier_risk_ref={config.verifier_risk_reference:g} "
        f"verifier_policy={config.verifier_policy} "
        f"verifier_cactus_mix={config.verifier_cactus_mix:g} "
        f"verifier_budget_slope={config.verifier_budget_slope:g} "
        f"verifier_tail_threshold={config.verifier_tail_threshold:g} "
        f"token_decay={config.token_decay:g} "
        f"rejection_reset={config.rejection_reset:g} "
        f"max_idle_blocks={config.max_idle_blocks} "
        f"causal_only=1 verifier_budget_control="
        f"{int(config.direction == 'verifier_risk')} "
        f"root_injection="
        f"{int(config.direction not in {'none', 'hidden_output', 'headwise_output', 'expected_logit', 'expected_pq_gradient', 'expected_vocab', 'expected_vocab_audit', 'expected_vocab_negative', 'expected_vocab_headmap', 'expected_vocab_online', 'expected_fusion', 'expected_token_adaptive', 'expected_token_preconditioned', 'expected_hidden_fusion', 'expected_hidden_consistent', 'expected_hidden_adaptive', 'expected_hidden_calibrated', 'expected_hidden_compatible', 'expected_root_anchor', 'expected_root_exact', 'expected_scale', 'expected_scale_adaptive_depth', 'expected_scale_counterfactual', 'expected_scale_repetition', 'expected_scale_tv', 'expected_top1_bias', 'adaptive_depth', 'verifier_risk'})} "
        f"fusion_injection="
        f"{int(config.direction in {'expected_fusion', 'expected_token_adaptive', 'expected_token_preconditioned', 'expected_hidden_fusion', 'expected_hidden_consistent', 'expected_hidden_adaptive', 'expected_hidden_calibrated', 'expected_hidden_compatible'})} "
        f"output_hidden_injection="
        f"{int(config.direction in {'hidden_output', 'headwise_output', 'expected_logit', 'expected_pq_gradient'})} "
        f"draft_logit_calibration="
        f"{int(config.direction in {'expected_scale', 'expected_scale_adaptive_depth', 'expected_scale_counterfactual', 'expected_scale_repetition', 'expected_scale_tv', 'expected_top1_bias', 'expected_hidden_calibrated', 'expected_root_anchor', 'expected_root_exact', 'expected_vocab', 'expected_vocab_audit', 'expected_vocab_negative', 'expected_vocab_headmap', 'expected_vocab_online'})}",
        flush=True,
    )
