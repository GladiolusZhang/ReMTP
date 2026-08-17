"""Native-MTP prefix-credit relaxation with exact joint verification.

The six Qwen3.5 MTP proposals form one ordered conditional path.  A later
candidate probability above its local draft probability can therefore repay
an earlier joint-prefix deficit.  Token-wise relaxed verifiers discard this
mass once ``h_i(y_i) == q_i(y_i)``; this module instead derives the useful
cap from the current prefix survival probability.

Only confident target top-1 candidates receive redistributed probability
mass.  The target/MTP weights, proposal path, bonus distribution, and caches
remain unchanged.  All new work is a fixed-length scalar recurrence on GPU.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch

from remtp.block_verification import (
    block_verification_state,
    longest_accepted_prefix,
    prefix_joint_probability,
)


_REJECTION_MODULE = "vllm.v1.sample.rejection_sampler"
_CONFIG: PrefixCreditConfig | None = None
_FAST_PATH: Any | None = None
_VLLM_REJECTION: Any | None = None
_DIAGNOSTIC_EMITTED = False
_AUDIT_ROUNDS = 0
_AUDIT_TOTAL: torch.Tensor | None = None


@dataclass(frozen=True)
class PrefixCreditConfig:
    """Runtime settings for prefix-credit allocation."""

    expected_draft_tokens: int = 6
    min_top1_margin: float = 0.10
    per_token_tv_cap: float = 0.49
    block_tv_cap: float = 0.75
    allocation_mode: str = "prefix_credit"
    min_prefix_gain_per_tv: float = 0.50
    compile_fast_path: bool = True
    diagnostics: bool = False
    audit_interval: int = 0

    @classmethod
    def from_env(cls) -> "PrefixCreditConfig":
        return cls(
            expected_draft_tokens=int(os.getenv("MTP_TOKENS", "6")),
            min_top1_margin=float(
                os.getenv("REMTP_PREFIX_CREDIT_MIN_TOP1_MARGIN", "0.10")
            ),
            per_token_tv_cap=float(
                os.getenv("REMTP_PREFIX_CREDIT_PER_TOKEN_TV", "0.49")
            ),
            block_tv_cap=float(
                os.getenv("REMTP_PREFIX_CREDIT_BLOCK_TV", "0.75")
            ),
            allocation_mode=os.getenv(
                "REMTP_PREFIX_CREDIT_ALLOCATION",
                "prefix_credit",
            ),
            min_prefix_gain_per_tv=float(
                os.getenv("REMTP_PREFIX_CREDIT_MIN_GAIN_PER_TV", "0.50")
            ),
            compile_fast_path=(
                os.getenv("REMTP_PREFIX_CREDIT_COMPILE", "1") == "1"
            ),
            diagnostics=(
                os.getenv("REMTP_PREFIX_CREDIT_DIAGNOSTICS", "0") == "1"
            ),
            audit_interval=int(
                os.getenv("REMTP_PREFIX_CREDIT_AUDIT_INTERVAL", "0")
            ),
        )

    def validate(self) -> None:
        if self.expected_draft_tokens < 1:
            raise ValueError("expected_draft_tokens must be positive")
        if self.min_top1_margin < 0.0:
            raise ValueError("min_top1_margin must be non-negative")
        if not 0.0 <= self.per_token_tv_cap < 1.0:
            raise ValueError("per_token_tv_cap must be in [0,1)")
        if not 0.0 <= self.block_tv_cap < 1.0:
            raise ValueError("block_tv_cap must be in [0,1)")
        if self.allocation_mode not in {
            "prefix_credit",
            "atomic_credit",
            "token_cap",
        }:
            raise ValueError(
                "allocation_mode must be prefix_credit, atomic_credit, "
                "or token_cap"
            )
        if self.min_prefix_gain_per_tv < 0.0:
            raise ValueError("min_prefix_gain_per_tv must be non-negative")
        if self.audit_interval < 0:
            raise ValueError("audit_interval must be non-negative")


@dataclass(frozen=True)
class PrefixCreditResult:
    """Temporary verifier distribution and block-level audit quantities."""

    probs: torch.Tensor
    target_candidate_probs: torch.Tensor
    draft_candidate_probs: torch.Tensor
    boosted_candidate_probs: torch.Tensor
    allocated_tv: torch.Tensor
    eligibility: torch.Tensor
    prefix_survival_before: torch.Tensor
    prefix_survival_after: torch.Tensor
    over_q_credit: torch.Tensor


def target_top1_margin_stats(
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target top-1 IDs and top1-runner-up logit margins."""

    if target_logits.ndim != 2 or target_logits.shape[-1] < 2:
        raise ValueError("target logits require shape [rows,vocab>=2]")
    values, ids = target_logits.topk(2, dim=-1)
    return ids[:, 0], (values[:, 0] - values[:, 1]).to(torch.float32)


def prefix_credit_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_top1_ids: torch.Tensor,
    target_top1_margins: torch.Tensor,
    config: PrefixCreditConfig,
) -> PrefixCreditResult:
    """Construct ``H`` with prefix-aware rather than token-wise saturation.

    Raising one candidate probability from ``p`` to ``h`` while scaling all
    other probabilities proportionally has exact total variation ``h-p``.
    The recurrence therefore spends an exact block TV budget.
    """

    config.validate()
    if target_probs.ndim != 2 or target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must share [rows,vocab]")
    rows_count = target_probs.shape[0]
    expected_vector = (rows_count,)
    for name, value in (
        ("draft_token_ids", draft_token_ids),
        ("target_top1_ids", target_top1_ids),
        ("target_top1_margins", target_top1_margins),
    ):
        if value.shape != expected_vector:
            raise ValueError(f"{name} must have shape [rows]")
    if rows_count < 1:
        raise ValueError("at least one MTP proposal is required")

    target = target_probs.to(torch.float32)
    draft = draft_probs.to(device=target.device, dtype=torch.float32)
    ids = draft_token_ids.to(device=target.device, dtype=torch.int64)
    top1_ids = target_top1_ids.to(device=target.device, dtype=torch.int64)
    margins = target_top1_margins.to(
        device=target.device,
        dtype=torch.float32,
    )
    row_ids = torch.arange(rows_count, device=target.device)
    p_y = target[row_ids, ids].clamp(0.0, 1.0)
    q_y = draft[row_ids, ids].clamp(0.0, 1.0)
    eligible = (ids == top1_ids) & (margins >= config.min_top1_margin)

    remaining = p_y.new_tensor(config.block_tv_cap)
    survival = p_y.new_ones(())
    allocations: list[torch.Tensor] = []
    before_values: list[torch.Tensor] = []
    after_values: list[torch.Tensor] = []
    max_candidate = 1.0 - 1e-6

    # D is fixed to six in the benchmark. torch.compile statically unrolls
    # this recurrence; there are no scalar reads or host synchronizations.
    for depth in range(rows_count):
        before_values.append(survival)
        capacity = torch.minimum(
            p_y.new_tensor(config.per_token_tv_cap),
            (max_candidate - p_y[depth]).clamp_min(0.0),
        )
        local_request = (
            q_y[depth] - p_y[depth]
        ).clamp_min(0.0)
        local_allocation = torch.minimum(
            local_request,
            torch.minimum(capacity, remaining),
        )
        local_allocation = torch.where(
            eligible[depth],
            local_allocation,
            torch.zeros_like(local_allocation),
        )

        # Over-q mass is useful only through the ordered prefix recurrence.
        # The atomic ablation requires full repair; the main method permits a
        # partial payment only when its exact marginal prefix gain is high.
        local_h = p_y[depth] + local_allocation
        desired_h = q_y[depth] / survival.clamp_min(1e-30)
        extra_needed = (desired_h - local_h).clamp_min(0.0)
        extra_capacity = torch.minimum(
            (capacity - local_allocation).clamp_min(0.0),
            (remaining - local_allocation).clamp_min(0.0),
        )
        extra_capacity = torch.minimum(
            extra_capacity,
            (max_candidate - local_h).clamp_min(0.0),
        )
        repairable = (
            eligible[depth]
            & (config.allocation_mode == "atomic_credit")
            & (survival < 1.0 - 1e-7)
            & (extra_needed > 1e-12)
            & (extra_needed <= extra_capacity + 1e-7)
            & (desired_h <= max_candidate)
        )
        extra_allocation = torch.where(
            repairable,
            extra_needed,
            torch.zeros_like(extra_needed),
        )
        if config.allocation_mode == "prefix_credit":
            # Before prefix saturation, d(a_i)/d(h_i)=a_(i-1)/q_i.
            # This is the exact immediate joint-prefix gain obtained per unit
            # TV at this position. Permit partial credit only when that gain
            # clears the configured efficiency floor.
            gain_per_tv = survival / q_y[depth].clamp_min(1e-30)
            efficient = (
                eligible[depth]
                & (survival < 1.0 - 1e-7)
                & (extra_needed > 1e-12)
                & (gain_per_tv >= config.min_prefix_gain_per_tv)
            )
            extra_allocation = torch.where(
                efficient,
                torch.minimum(extra_needed, extra_capacity),
                torch.zeros_like(extra_needed),
            )
        allocation = local_allocation + extra_allocation
        allocations.append(allocation)
        remaining = (remaining - allocation).clamp_min(0.0)

        candidate_probability = p_y[depth] + allocation
        ratio = candidate_probability / q_y[depth].clamp_min(1e-30)
        survival = torch.minimum(
            survival * ratio,
            torch.ones_like(survival),
        )
        after_values.append(survival)

    allocated = torch.stack(allocations)
    boosted = (p_y + allocated).clamp(max=max_candidate)
    scale = ((1.0 - boosted) / (1.0 - p_y)).nan_to_num(
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    verification = target * scale.unsqueeze(-1)
    verification.scatter_(1, ids.unsqueeze(1), boosted.unsqueeze(1))

    return PrefixCreditResult(
        probs=verification.contiguous(),
        target_candidate_probs=p_y,
        draft_candidate_probs=q_y,
        boosted_candidate_probs=boosted,
        allocated_tv=allocated,
        eligibility=eligible,
        prefix_survival_before=torch.stack(before_values),
        prefix_survival_after=torch.stack(after_values),
        # Count only probability mass newly allocated beyond both the
        # original target and local draft candidate probabilities. Natural
        # target surplus p>q is useful to joint verification, but is not a
        # contribution of prefix-credit relaxation.
        over_q_credit=(boosted - torch.maximum(p_y, q_y)).clamp_min(0.0),
    )


def _prefix_credit_fast_tensors(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_top1_ids: torch.Tensor,
    target_top1_margins: torch.Tensor,
    config: PrefixCreditConfig,
) -> tuple[torch.Tensor, ...]:
    result = prefix_credit_distribution(
        target_probs,
        draft_probs,
        draft_token_ids,
        target_top1_ids,
        target_top1_margins,
        config,
    )
    state = block_verification_state(
        result.probs,
        draft_probs,
        draft_token_ids,
        assume_normalized=True,
    )
    token_capped_prefix = prefix_joint_probability(
        torch.minimum(
            result.boosted_candidate_probs,
            torch.maximum(
                result.target_candidate_probs,
                result.draft_candidate_probs,
            ),
        ),
        result.draft_candidate_probs,
    )
    return (
        result.probs,
        state.prefix_joint_probability,
        state.subblock_acceptance_probability,
        state.correction_scale,
        result.target_candidate_probs,
        result.draft_candidate_probs,
        result.boosted_candidate_probs,
        result.allocated_tv,
        result.eligibility.to(torch.float32),
        result.prefix_survival_before,
        result.over_q_credit,
        (state.prefix_joint_probability - token_capped_prefix).clamp_min(0.0),
    )


def _prefix_credit_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Construct ``H`` and commit the exact jointly verified prefix."""

    global _AUDIT_ROUNDS, _AUDIT_TOTAL, _DIAGNOSTIC_EMITTED
    global _FAST_PATH

    original = getattr(_prefix_credit_rejection_sample, "_remtp_original")
    if (
        sampling_metadata.all_greedy
        or target_logits.shape[0] == 0
        or draft_probs is None
    ):
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
        raise RuntimeError("prefix-credit MTP requires --max-num-seqs 1")
    if _CONFIG is None or _VLLM_REJECTION is None or _FAST_PATH is None:
        raise RuntimeError("prefix-credit MTP has not been installed")
    num_drafts = num_draft_tokens[0]
    if not 1 <= num_drafts <= _CONFIG.expected_draft_tokens:
        raise RuntimeError(
            f"expected at most {_CONFIG.expected_draft_tokens} drafts, "
            f"received {num_drafts}"
        )
    if draft_token_ids.shape[0] != num_drafts:
        raise RuntimeError("unexpected flattened draft-token layout")

    module = _VLLM_REJECTION
    target_top1_ids, target_top1_margins = target_top1_margin_stats(
        target_logits
    )
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    (
        verification_probs,
        prefix_joint,
        subblock_acceptance,
        correction_scale,
        p_y,
        q_y,
        h_y,
        allocated_tv,
        eligibility,
        prefix_before,
        over_q_credit,
        prefix_repair,
    ) = _FAST_PATH(
        target_probs,
        draft_probs,
        draft_token_ids,
        target_top1_ids,
        target_top1_margins,
    )

    uniforms = module.generate_uniform_probs(
        num_drafts,
        num_draft_tokens,
        sampling_metadata.generators,
        target_logits.device,
    )
    accepted_length = longest_accepted_prefix(
        subblock_acceptance,
        uniforms,
    )

    # Exponential-race recovery does not require normalized rows.  Scaling H
    # by the committed-prefix survival gives the exact joint residual.
    scaled_verification = (
        correction_scale.unsqueeze(-1) * verification_probs
    ).contiguous()
    recovered_token_ids = module.sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        scaled_verification,
        sampling_metadata,
        target_logits.device,
    )

    output = torch.full(
        (1, max_spec_len + 1),
        module.PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=target_logits.device,
    )
    output[0, :num_drafts] = draft_token_ids.to(torch.int32)
    positions = torch.arange(
        max_spec_len + 1,
        device=target_logits.device,
        dtype=torch.int64,
    )
    output[0] = torch.where(
        positions < accepted_length,
        output[0],
        torch.full_like(output[0], module.PLACEHOLDER_TOKEN_ID),
    )
    recovery_index = accepted_length.clamp(max=num_drafts - 1)
    correction = torch.where(
        accepted_length == num_drafts,
        bonus_token_ids.reshape(-1)[0].to(torch.int32),
        recovered_token_ids[recovery_index].to(torch.int32),
    )
    output[0].scatter_(
        0,
        accepted_length.reshape(1),
        correction.reshape(1),
    )

    if _CONFIG.audit_interval > 0:
        if _AUDIT_TOTAL is None or _AUDIT_TOTAL.device != target_logits.device:
            _AUDIT_TOTAL = torch.zeros(
                9,
                device=target_logits.device,
                dtype=torch.float64,
            )
        _AUDIT_ROUNDS += 1
        _AUDIT_TOTAL += torch.stack(
            (
                allocated_tv.sum(),
                over_q_credit.sum(),
                eligibility.sum(),
                (_CONFIG.block_tv_cap - allocated_tv.sum()).clamp_min(0.0),
                prefix_repair.sum(),
                prefix_repair[-1],
                prefix_joint[-1],
                accepted_length.to(torch.float32),
                torch.where(
                    over_q_credit.sum() > 1e-12,
                    prefix_repair.sum() / over_q_credit.sum(),
                    torch.zeros_like(over_q_credit.sum()),
                ),
            )
        ).to(torch.float64)
        if _AUDIT_ROUNDS % _CONFIG.audit_interval == 0:
            values = (_AUDIT_TOTAL / _CONFIG.audit_interval).tolist()
            print(
                "[ReMTP][PrefixCredit][audit] "
                f"rounds={_AUDIT_ROUNDS-_CONFIG.audit_interval+1}-"
                f"{_AUDIT_ROUNDS} "
                f"mean_tv={values[0]:.6f} "
                f"mean_over_q_credit={values[1]:.6f} "
                f"mean_eligible={values[2]:.6f} "
                f"mean_unused_tv={values[3]:.6f} "
                f"mean_prefix_repair_sum={values[4]:.6f} "
                f"mean_final_prefix_repair={values[5]:.6f} "
                f"mean_final_survival={values[6]:.6f} "
                f"mean_committed_drafts={values[7]:.6f} "
                f"mean_credit_efficiency={values[8]:.6f}",
                flush=True,
            )
            _AUDIT_TOTAL.zero_()

    if _CONFIG.diagnostics and not _DIAGNOSTIC_EMITTED:
        print(
            "[ReMTP][PrefixCredit][diagnostic] "
            f"mode={_CONFIG.allocation_mode} "
            f"p={p_y.tolist()} q={q_y.tolist()} h={h_y.tolist()} "
            f"eligible={eligibility.tolist()} "
            f"prefix_before={prefix_before.tolist()} "
            f"tv={allocated_tv.tolist()} "
            f"over_q={over_q_credit.tolist()} "
            f"prefix_joint={prefix_joint.tolist()} "
            f"prefix_repair={prefix_repair.tolist()} "
            f"subblock_A={subblock_acceptance.tolist()} "
            f"u={uniforms.tolist()} accepted={accepted_length.item()}",
            flush=True,
        )
        _DIAGNOSTIC_EMITTED = True
    return output


def install_prefix_credit_mtp() -> None:
    """Install prefix-credit allocation below probabilistic native MTP."""

    global _CONFIG, _FAST_PATH, _VLLM_REJECTION

    config = PrefixCreditConfig.from_env()
    config.validate()
    module = importlib.import_module(_REJECTION_MODULE)
    _CONFIG = config
    _VLLM_REJECTION = module

    def fast_path(
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_top1_ids: torch.Tensor,
        target_top1_margins: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return _prefix_credit_fast_tensors(
            target_probs,
            draft_probs,
            draft_token_ids,
            target_top1_ids,
            target_top1_margins,
            config,
        )

    _FAST_PATH = fast_path
    if config.compile_fast_path:
        _FAST_PATH = torch.compile(fast_path, fullgraph=True, dynamic=False)

    current = module.rejection_sample
    if not getattr(current, "_remtp_prefix_credit", False):
        _prefix_credit_rejection_sample._remtp_prefix_credit = True
        _prefix_credit_rejection_sample._remtp_original = current
        module.rejection_sample = _prefix_credit_rejection_sample

    print(
        "[ReMTP][PrefixCredit] native-MTP joint prefix relaxation enabled "
        f"mode={config.allocation_mode} "
        f"min_top1_margin={config.min_top1_margin:g} "
        f"per_token_tv={config.per_token_tv_cap:g} "
        f"block_tv={config.block_tv_cap:g} "
        f"min_gain_per_tv={config.min_prefix_gain_per_tv:g} "
        f"compiled={int(config.compile_fast_path)}",
        flush=True,
    )
