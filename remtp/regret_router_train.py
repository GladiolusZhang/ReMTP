"""Train the tiny regret router from label-free verifier traces.

The target model and native MTP stay frozen.  Training uses compact top-k
target/draft distributions plus Exact-TV verifier scalars collected from
ordinary generation traces; task answers are never consumed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from remtp.regret_router_model import (
    RegretRouter,
    RegretRouterArchitecture,
    save_regret_router_checkpoint,
)


def _load_records(
    directory: Path,
    *,
    progress_every: int = 0,
    max_shards: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(directory.glob("router_shard_*.pt"))
    pending = directory / "router_pending.pt"
    if max_shards is not None:
        paths = paths[:max_shards]
    elif pending.exists():
        paths.append(pending)
    if not paths:
        raise FileNotFoundError(f"no router_shard_*.pt files in {directory}")
    for index, path in enumerate(paths, 1):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError(f"unsupported collector shard: {path}")
        shard = payload.get("records")
        if not isinstance(shard, list):
            raise ValueError(f"invalid collector records: {path}")
        records.extend(shard)
        if progress_every > 0 and (
            index == 1 or index % progress_every == 0 or index == len(paths)
        ):
            print(
                f"loaded_shards={index}/{len(paths)} records={len(records)}",
                flush=True,
            )
    if not records:
        raise ValueError("collector shards contain no records")
    return records


def _compact_support(record: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    """Build exact probabilities on top-k(P) union top-k(Q) plus tail."""
    p_ids = record["p_top_ids"].to(torch.int64)
    q_ids = record["q_top_ids"].to(torch.int64)
    p_top = record["p_top_probs"].to(torch.float32)
    q_top = record["q_top_probs"].to(torch.float32)
    p_on_q = record["p_on_q_top"].to(torch.float32)
    q_on_p = record["q_on_p_top"].to(torch.float32)
    p_delta = record["p_direction_delta"].to(torch.float32)
    q_delta = record["q_direction_delta"].to(torch.float32)
    heads, top_k = p_ids.shape
    support = 2 * top_k + 1
    p_result = torch.zeros(heads, support, dtype=torch.float32)
    q_result = torch.zeros_like(p_result)
    delta_result = torch.zeros_like(p_result)
    mask = torch.zeros(heads, support, dtype=torch.bool)
    for head in range(heads):
        p_lookup = {
            int(token): (float(prob), float(delta))
            for token, prob, delta in zip(
                p_ids[head], p_top[head], p_delta[head], strict=True
            )
        }
        q_lookup = {
            int(token): (float(prob), float(delta))
            for token, prob, delta in zip(
                q_ids[head], q_top[head], q_delta[head], strict=True
            )
        }
        p_cross = {
            int(token): float(prob)
            for token, prob in zip(q_ids[head], p_on_q[head], strict=True)
        }
        q_cross = {
            int(token): float(prob)
            for token, prob in zip(p_ids[head], q_on_p[head], strict=True)
        }
        union = list(dict.fromkeys([*p_lookup, *q_lookup]))
        for index, token in enumerate(union):
            if token in p_lookup:
                p_result[head, index] = p_lookup[token][0]
                delta_result[head, index] = p_lookup[token][1]
            else:
                p_result[head, index] = p_cross[token]
                delta_result[head, index] = q_lookup[token][1]
            if token in q_lookup:
                q_result[head, index] = q_lookup[token][0]
            else:
                q_result[head, index] = q_cross[token]
            mask[head, index] = True
        tail_index = 2 * top_k
        p_result[head, tail_index] = (
            1.0 - p_result[head, :tail_index].sum()
        ).clamp_min(0.0)
        q_result[head, tail_index] = (
            1.0 - q_result[head, :tail_index].sum()
        ).clamp_min(0.0)
        mask[head, tail_index] = True
    p_result /= p_result.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    q_result /= q_result.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return p_result, q_result, delta_result.clamp(-20.0, 20.0), mask


class RouterTraceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        progress_every: int = 0,
        split_name: str = "dataset",
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        for index, record in enumerate(records, 1):
            p_compact, q_compact, direction_delta, support_mask = (
                _compact_support(record)
            )
            q_top = record["q_top_probs"][:, :8].to(torch.float32)
            q_top_conditional = q_top / q_top.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-30)
            q_entropy = -(
                q_top_conditional
                * torch.log(q_top_conditional.clamp_min(1e-30))
            ).sum(dim=-1) / math.log(q_top.shape[-1])
            q_margin = (
                torch.log(q_top[:, 0].clamp_min(1e-30))
                - torch.log(q_top[:, 1].clamp_min(1e-30))
            ).clamp(0.0, 8.0) / 8.0
            draft_features = torch.stack(
                (
                    q_entropy,
                    q_margin,
                ),
                dim=-1,
            )
            self.examples.append(
                {
                    "root_hidden": record["root_hidden"].to(torch.float32),
                    "regret_direction": record["regret_direction"].to(
                        torch.float32
                    ),
                    "regret_debt": record["regret_debt"].reshape(()).to(
                        torch.float32
                    ),
                    "source_entropy": record["source_entropy"].reshape(()).to(
                        torch.float32
                    ),
                    "source_margin": record["source_margin"].reshape(()).to(
                        torch.float32
                    ),
                    "target_entropy": record["target_entropy"].to(torch.float32),
                    "target_margin": record["target_margin"].to(torch.float32),
                    "head_reliability": record["head_reliability"].to(
                        torch.float32
                    ),
                    "draft_features": draft_features,
                    "p_compact": p_compact,
                    "q_compact": q_compact,
                    "direction_delta": direction_delta,
                    "support_mask": support_mask,
                    "p_y": record["target_candidate_probs"].to(torch.float32),
                    "q_y": record["draft_candidate_probs"].to(torch.float32),
                    "allocated_tv": record["allocated_tv"].to(torch.float32),
                    "strict_acceptance": record["strict_acceptance"].to(
                        torch.float32
                    ),
                    "expected_regret": record["expected_regret"].to(
                        torch.float32
                    ),
                }
            )
            if progress_every > 0 and (
                index == 1
                or index % progress_every == 0
                or index == len(records)
            ):
                print(
                    f"built_{split_name}_examples={index}/{len(records)}",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1)


def _calibrate_q(
    q: torch.Tensor,
    direction_delta: torch.Tensor,
    support_mask: torch.Tensor,
    direction_strength: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    logits = torch.log(q.clamp_min(1e-30))
    logits = logits * torch.exp(logit_scale).unsqueeze(-1)
    logits = logits + direction_strength.unsqueeze(-1) * direction_delta
    logits = logits.masked_fill(~support_mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def _symmetric_kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_log = torch.log(first.clamp_min(1e-30))
    second_log = torch.log(second.clamp_min(1e-30))
    return 0.5 * (
        (first * (first_log - second_log)).sum(dim=-1)
        + (second * (second_log - first_log)).sum(dim=-1)
    )


def _batch_loss(
    model: RegretRouter,
    batch: dict[str, torch.Tensor],
    *,
    intervention_weight: float,
    entropy_weight: float,
    acceptance_weight: float,
    risk_weight: float,
    debt_reference: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    normalized_debt = batch["regret_debt"] / debt_reference
    proposal = model(
        batch["root_hidden"],
        batch["regret_direction"],
        normalized_debt,
        batch["source_entropy"],
        batch["source_margin"],
        batch["head_reliability"],
    )
    calibrated_q = _calibrate_q(
        batch["q_compact"],
        batch["direction_delta"],
        batch["support_mask"],
        proposal.direction_strength,
        proposal.logit_scale,
    )
    alignment = _symmetric_kl(batch["p_compact"], calibrated_q).mean()
    intervention = _symmetric_kl(batch["q_compact"], calibrated_q).mean()
    entropy_drift = (
        _entropy(calibrated_q) - _entropy(batch["q_compact"])
    ).square().mean()

    verification = model(
        batch["root_hidden"],
        batch["regret_direction"],
        normalized_debt,
        batch["target_entropy"],
        batch["target_margin"],
        batch["head_reliability"],
    )
    scaled_h = batch["p_y"] + verification.budget_scale * batch["allocated_tv"]
    scaled_acceptance = torch.minimum(
        torch.ones_like(scaled_h),
        scaled_h / batch["q_y"].clamp_min(1e-30),
    )
    acceptance_gain = (
        scaled_acceptance - batch["strict_acceptance"]
    ).clamp_min(0.0).mean()
    risk = (
        verification.budget_scale
        * batch["expected_regret"]
        * (0.5 + 0.5 * batch["target_margin"])
    ).mean()
    loss = (
        alignment
        + intervention_weight * intervention
        + entropy_weight * entropy_drift
        - acceptance_weight * acceptance_gain
        + risk_weight * risk
    )
    metrics = {
        "loss": loss.detach(),
        "alignment": alignment.detach(),
        "intervention": intervention.detach(),
        "entropy_drift": entropy_drift.detach(),
        "acceptance_gain": acceptance_gain.detach(),
        "risk": risk.detach(),
        "budget_scale": verification.budget_scale.mean().detach(),
        "direction_strength": proposal.direction_strength.mean().detach(),
        "abs_logit_scale": proposal.logit_scale.abs().mean().detach(),
    }
    return loss, metrics


def _run_epoch(
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
            loss, metrics = _batch_loss(
                model,
                batch,
                intervention_weight=args.intervention_weight,
                entropy_weight=args.entropy_weight,
                acceptance_weight=args.acceptance_weight,
                risk_weight=args.risk_weight,
                debt_reference=args.debt_reference,
            )
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


def split_records_by_request(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    missing_request_ids = sum("request_id" not in record for record in records)
    if missing_request_ids:
        raise ValueError(
            f"{missing_request_ids} trace blocks have no request_id; "
            "recollect them to prevent block-level validation leakage"
        )
    request_ids = sorted({str(record["request_id"]) for record in records})
    if len(request_ids) < 2:
        raise ValueError("at least two collected requests are required")
    random.Random(seed).shuffle(request_ids)
    validation_request_count = max(
        1,
        round(len(request_ids) * validation_fraction),
    )
    validation_request_count = min(
        validation_request_count,
        len(request_ids) - 1,
    )
    validation_request_ids = set(request_ids[:validation_request_count])
    training_request_ids = set(request_ids[validation_request_count:])
    training_records = [
        record
        for record in records
        if str(record["request_id"]) in training_request_ids
    ]
    validation_records = [
        record
        for record in records
        if str(record["request_id"]) in validation_request_ids
    ]
    return (
        training_records,
        validation_records,
        training_request_ids,
        validation_request_ids,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    records = _load_records(args.data_dir)
    (
        training_records,
        validation_records,
        training_request_ids,
        validation_request_ids,
    ) = split_records_by_request(
        records,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    training_dataset = RouterTraceDataset(training_records)
    validation_dataset = RouterTraceDataset(validation_records)
    example = training_dataset[0]
    hidden_size = int(example["root_hidden"].numel())
    num_heads = int(example["target_entropy"].numel())
    architecture = RegretRouterArchitecture(
        hidden_size=hidden_size,
        num_heads=num_heads,
        rank=args.rank,
        width=args.width,
        max_direction_strength=args.max_direction_strength,
        max_logit_scale=args.max_logit_scale,
        max_budget_reduction=args.max_budget_reduction,
    )
    device = torch.device(args.device)
    model = RegretRouter(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
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
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        training = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            args=args,
        )
        validation = _run_epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            args=args,
        )
        row = {"epoch": epoch, "train": training, "validation": validation}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={training['loss']:.6f} "
            f"validation_loss={validation['loss']:.6f} "
            f"validation_alignment={validation['alignment']:.6f} "
            f"validation_budget_scale={validation['budget_scale']:.6f}",
            flush=True,
        )
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    metadata = {
        "records": len(records),
        "requests": len(training_request_ids | validation_request_ids),
        "training_records": len(training_records),
        "validation_records": len(validation_records),
        "training_requests": len(training_request_ids),
        "validation_requests": len(validation_request_ids),
        "split_unit": "request_id",
        "seed": args.seed,
        "best_validation_loss": best_validation,
        "objective": "label-free target alignment + TV efficiency",
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
    print(f"saved checkpoint: {args.output}")
    print(f"saved metrics: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-direction-strength", type=float, default=0.03)
    parser.add_argument("--max-logit-scale", type=float, default=0.08)
    parser.add_argument("--max-budget-reduction", type=float, default=0.50)
    parser.add_argument("--intervention-weight", type=float, default=0.25)
    parser.add_argument("--entropy-weight", type=float, default=0.10)
    parser.add_argument("--acceptance-weight", type=float, default=1.0)
    parser.add_argument("--risk-weight", type=float, default=2.0)
    parser.add_argument("--debt-reference", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation-fraction must be in (0,1)")
    if args.debt_reference <= 0.0:
        parser.error("debt-reference must be positive")
    return args


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
