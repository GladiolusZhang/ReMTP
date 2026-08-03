"""Train a selective, logit-only regret Router from verifier traces.

The target model, native MTP, Exact-TV allocation, and hidden states remain
unchanged.  For every actionable trace position, an offline compact-support
oracle chooses the bounded MTP logit scale that best aligns Q with P while
penalizing intervention and entropy drift.  The Router distils that action
from causally available regret state and current-Q statistics.  A checkpoint
is enabled only if it beats the exact no-op policy on both model-selection
and untouched audit requests.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from remtp.regret_router_model import (
    RegretRouter,
    RegretRouterArchitecture,
    save_regret_router_checkpoint,
)
from remtp.regret_router_train import (
    RouterTraceDataset,
    _calibrate_q,
    _load_records,
    split_records_by_request,
)


def compact_objective(
    target: torch.Tensor,
    draft: torch.Tensor,
    candidate: torch.Tensor,
    *,
    intervention_weight: float,
    entropy_weight: float,
) -> torch.Tensor:
    """Return one compact-support objective per batch/head/action."""
    p = target
    q = draft
    while p.ndim < candidate.ndim:
        p = p.unsqueeze(-2)
        q = q.unsqueeze(-2)
    p_log = torch.log(p.clamp_min(1e-30))
    q_log = torch.log(q.clamp_min(1e-30))
    candidate_log = torch.log(candidate.clamp_min(1e-30))
    alignment = 0.5 * (
        (p * (p_log - candidate_log)).sum(dim=-1)
        + (candidate * (candidate_log - p_log)).sum(dim=-1)
    )
    intervention = 0.5 * (
        (q * (q_log - candidate_log)).sum(dim=-1)
        + (candidate * (candidate_log - q_log)).sum(dim=-1)
    )

    def entropy(value: torch.Tensor) -> torch.Tensor:
        return -(value * torch.log(value.clamp_min(1e-30))).sum(dim=-1)

    entropy_drift = (entropy(candidate) - entropy(q)).square()
    return (
        alignment
        + intervention_weight * intervention
        + entropy_weight * entropy_drift
    )


@torch.no_grad()
def oracle_logit_targets(
    batch: dict[str, torch.Tensor],
    *,
    debt_reference: float,
    max_logit_scale: float,
    grid_size: int,
    min_oracle_gain: float,
    intervention_weight: float,
    entropy_weight: float,
) -> dict[str, torch.Tensor]:
    """Build bounded oracle actions and abstention labels for one batch."""
    p = batch["p_compact"]
    q = batch["q_compact"]
    support = batch["support_mask"]
    scales = torch.linspace(
        -max_logit_scale,
        max_logit_scale,
        grid_size,
        device=q.device,
        dtype=torch.float32,
    )
    logits = torch.log(q.clamp_min(1e-30)).unsqueeze(-2)
    logits = logits * torch.exp(scales).view(1, 1, grid_size, 1)
    candidates = torch.softmax(
        logits.masked_fill(~support.unsqueeze(-2), float("-inf")),
        dim=-1,
    )
    candidate_objective = compact_objective(
        p,
        q,
        candidates,
        intervention_weight=intervention_weight,
        entropy_weight=entropy_weight,
    )
    best_objective, best_index = candidate_objective.min(dim=-1)
    baseline = compact_objective(
        p,
        q,
        q,
        intervention_weight=intervention_weight,
        entropy_weight=entropy_weight,
    )
    gain = (baseline - best_objective).clamp_min(0.0)
    normalized_debt = batch["regret_debt"] / debt_reference
    debt_gate = (normalized_debt / (1.0 + normalized_debt)).clamp(0.0, 1.0)
    actionable_block = normalized_debt > 0.0
    eligible = actionable_block.unsqueeze(1) & (gain >= min_oracle_gain)
    raw_target = scales[best_index]
    target_scale = torch.where(
        eligible,
        raw_target * debt_gate.unsqueeze(1),
        torch.zeros_like(raw_target),
    )
    return {
        "target_scale": target_scale,
        "baseline_objective": baseline,
        "oracle_objective": best_objective,
        "oracle_gain": gain,
        "actionable": actionable_block.unsqueeze(1).expand_as(gain),
        "eligible": eligible,
    }


def selective_batch_loss(
    model: RegretRouter,
    batch: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    oracle = oracle_logit_targets(
        batch,
        debt_reference=args.debt_reference,
        max_logit_scale=args.max_logit_scale,
        grid_size=args.oracle_grid_size,
        min_oracle_gain=args.min_oracle_gain,
        intervention_weight=args.intervention_weight,
        entropy_weight=args.entropy_weight,
    )
    normalized_debt = batch["regret_debt"] / args.debt_reference
    proposal = model(
        batch["root_hidden"],
        batch["regret_direction"],
        normalized_debt,
        batch["source_entropy"],
        batch["source_margin"],
        batch["head_reliability"],
        batch["draft_features"],
    )
    predicted_scale = proposal.logit_scale
    target_scale = oracle["target_scale"]
    actionable = oracle["actionable"]
    if actionable.any():
        normalized_error = (
            (predicted_scale - target_scale) / args.max_logit_scale
        ).square()
        loss = normalized_error[actionable].mean()
    else:
        loss = predicted_scale.sum() * 0.0

    applied_scale = torch.where(
        predicted_scale.abs() >= args.min_action_scale,
        predicted_scale,
        torch.zeros_like(predicted_scale),
    )
    calibrated = _calibrate_q(
        batch["q_compact"],
        batch["direction_delta"],
        batch["support_mask"],
        torch.zeros_like(applied_scale),
        applied_scale,
    )
    model_objective = compact_objective(
        batch["p_compact"],
        batch["q_compact"],
        calibrated,
        intervention_weight=args.intervention_weight,
        entropy_weight=args.entropy_weight,
    )
    base = oracle["baseline_objective"]
    if actionable.any():
        model_gain = (base - model_objective)[actionable].mean()
        oracle_gain = oracle["oracle_gain"][actionable].mean()
        mae = (predicted_scale - target_scale).abs()[actionable].mean()
        identity_mae = target_scale.abs()[actionable].mean()
        predicted_action_rate = (
            applied_scale.abs() >= args.min_action_scale
        )[actionable].to(torch.float32).mean()
        oracle_action_rate = (
            target_scale.abs() >= args.min_action_scale
        )[actionable].to(torch.float32).mean()
        sign_mask = oracle["eligible"] & (
            target_scale.abs() >= args.min_action_scale
        )
        if sign_mask.any():
            sign_accuracy = (
                torch.sign(predicted_scale[sign_mask])
                == torch.sign(target_scale[sign_mask])
            ).to(torch.float32).mean()
        else:
            sign_accuracy = torch.ones((), device=loss.device)
    else:
        zero = torch.zeros((), device=loss.device)
        model_gain = oracle_gain = mae = identity_mae = zero
        predicted_action_rate = oracle_action_rate = zero
        sign_accuracy = torch.ones((), device=loss.device)
    metrics = {
        "loss": loss.detach(),
        "model_gain": model_gain.detach(),
        "oracle_gain": oracle_gain.detach(),
        "logit_mae": mae.detach(),
        "identity_mae": identity_mae.detach(),
        "predicted_action_rate": predicted_action_rate.detach(),
        "oracle_action_rate": oracle_action_rate.detach(),
        "sign_accuracy": sign_accuracy.detach(),
    }
    return loss, metrics


def run_epoch(
    model: RegretRouter,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    for source in loader:
        batch = {key: value.to(device) for key, value in source.items()}
        with torch.set_grad_enabled(training):
            loss, metrics = selective_batch_loss(model, batch, args=args)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item())
        batches += 1
    if batches == 0:
        raise ValueError("empty training or validation split")
    return {key: value / batches for key, value in totals.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("stage=load_traces", flush=True)
    records = _load_records(
        args.data_dir,
        progress_every=args.load_progress_every,
        max_shards=args.max_shards,
    )
    (
        development_records,
        audit_records,
        development_request_ids,
        audit_request_ids,
    ) = split_records_by_request(
        records,
        validation_fraction=args.audit_fraction,
        seed=args.seed + 1,
    )
    validation_within_development = args.validation_fraction / (
        1.0 - args.audit_fraction
    )
    (
        training_records,
        validation_records,
        training_request_ids,
        validation_request_ids,
    ) = split_records_by_request(
        development_records,
        validation_fraction=validation_within_development,
        seed=args.seed,
    )
    print("stage=build_compact_support", flush=True)
    training_dataset = RouterTraceDataset(
        training_records,
        progress_every=args.build_progress_every,
        split_name="training",
    )
    validation_dataset = RouterTraceDataset(
        validation_records,
        progress_every=args.build_progress_every,
        split_name="validation",
    )
    audit_dataset = RouterTraceDataset(
        audit_records,
        progress_every=args.build_progress_every,
        split_name="audit",
    )
    record_count = len(records)
    training_record_count = len(training_records)
    validation_record_count = len(validation_records)
    audit_record_count = len(audit_records)
    del (
        records,
        development_records,
        training_records,
        validation_records,
        audit_records,
    )
    gc.collect()

    example = training_dataset[0]
    architecture = RegretRouterArchitecture(
        hidden_size=int(example["root_hidden"].numel()),
        num_heads=int(example["target_entropy"].numel()),
        rank=args.rank,
        width=args.width,
        max_direction_strength=0.0,
        max_logit_scale=args.max_logit_scale,
        max_budget_reduction=0.0,
        draft_feature_count=2,
    )
    device = torch.device(args.device)
    model = RegretRouter(architecture).to(device)
    if not args.use_high_dimensional_context:
        nn.init.zeros_(model.root_projection.weight)
        nn.init.zeros_(model.direction_projection.weight)
        model.root_projection.weight.requires_grad_(False)
        model.direction_projection.weight.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    audit_loader = DataLoader(
        audit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("stage=train_selective_logit_router", flush=True)
    history: list[dict[str, Any]] = []
    best_gain = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, float] | None = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(
            model,
            training_loader,
            device=device,
            optimizer=optimizer,
            args=args,
        )
        validation = run_epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            args=args,
        )
        history.append(
            {"epoch": epoch, "train": training, "validation": validation}
        )
        print(
            f"epoch={epoch:03d} train_loss={training['loss']:.6f} "
            f"validation_loss={validation['loss']:.6f} "
            f"validation_model_gain={validation['model_gain']:.6f} "
            f"validation_oracle_gain={validation['oracle_gain']:.6f} "
            f"validation_sign_accuracy={validation['sign_accuracy']:.4f} "
            f"validation_action_rate={validation['predicted_action_rate']:.4f}",
            flush=True,
        )
        gain = validation["model_gain"]
        if gain > best_gain + args.early_stop_delta:
            best_gain = gain
            best_epoch = epoch
            best_validation = validation
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    assert best_state is not None and best_validation is not None
    model.load_state_dict(best_state)
    audit = run_epoch(
        model,
        audit_loader,
        device=device,
        optimizer=None,
        args=args,
    )

    def passes_held_out(metrics: dict[str, float]) -> bool:
        return (
            metrics["model_gain"] >= args.min_validation_gain
            and metrics["logit_mae"] < metrics["identity_mae"]
            and metrics["sign_accuracy"] >= args.min_sign_accuracy
        )

    enabled = passes_held_out(best_validation) and passes_held_out(audit)
    policy = {
        "version": 2,
        "enabled": enabled,
        "action_mode": "contextual_logit",
        "min_abs_logit_scale": args.min_action_scale,
        "zero_debt_fast_path": True,
        "hidden_steering": False,
        "budget_scaling": False,
    }
    metadata = {
        "records": record_count,
        "requests": len(development_request_ids | audit_request_ids),
        "training_records": training_record_count,
        "validation_records": validation_record_count,
        "audit_records": audit_record_count,
        "training_requests": len(training_request_ids),
        "validation_requests": len(validation_request_ids),
        "audit_requests": len(audit_request_ids),
        "split_unit": "request_id",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_model_gain": best_gain,
        "audit": audit,
        "objective": "selective compact-support oracle logit distillation",
        "policy": policy,
        "uses_high_dimensional_context": args.use_high_dimensional_context,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_regret_router_checkpoint(args.output, model.cpu(), metadata=metadata)
    report = {
        "architecture": asdict(architecture),
        "metadata": metadata,
        "history": history,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"policy_enabled={enabled} best_epoch={best_epoch} "
        f"validation_model_gain={best_gain:.6f} "
        f"audit_model_gain={audit['model_gain']:.6f} "
        f"audit_sign_accuracy={audit['sign_accuracy']:.4f}",
        flush=True,
    )
    print(f"saved checkpoint: {args.output}", flush=True)
    print(f"saved metrics: {report_path}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--audit-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-logit-scale", type=float, default=0.08)
    parser.add_argument("--oracle-grid-size", type=int, default=17)
    parser.add_argument("--min-oracle-gain", type=float, default=1e-4)
    parser.add_argument("--min-validation-gain", type=float, default=1e-4)
    parser.add_argument("--min-sign-accuracy", type=float, default=0.55)
    parser.add_argument("--min-action-scale", type=float, default=0.01)
    parser.add_argument("--intervention-weight", type=float, default=0.25)
    parser.add_argument("--entropy-weight", type=float, default=0.10)
    parser.add_argument("--debt-reference", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--early-stop-delta", type=float, default=1e-5)
    parser.add_argument("--load-progress-every", type=int, default=100)
    parser.add_argument("--build-progress-every", type=int, default=10000)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--use-high-dimensional-context", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive")
    if args.oracle_grid_size < 3 or args.oracle_grid_size % 2 == 0:
        parser.error("oracle-grid-size must be an odd integer >= 3")
    if (
        not 0.0 < args.validation_fraction < 1.0
        or not 0.0 < args.audit_fraction < 1.0
        or args.validation_fraction + args.audit_fraction >= 1.0
    ):
        parser.error(
            "validation and audit fractions must be positive and sum to < 1"
        )
    if args.max_logit_scale <= 0.0 or args.debt_reference <= 0.0:
        parser.error("max-logit-scale and debt-reference must be positive")
    return args


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
