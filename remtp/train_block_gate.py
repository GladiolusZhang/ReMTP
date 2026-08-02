"""Train the one-unit counterfactual Block Shield router."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from remtp.block_gate import FEATURE_NAMES


def load_records(
    path: Path,
    *,
    drop_first_groups: int = 1,
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("block-gate calibration data is empty")
    groups = sorted({int(record["request_group"]) for record in records})
    dropped = set(groups[:drop_first_groups])
    kept = [
        record
        for record in records
        if int(record["request_group"]) not in dropped
    ]
    if not kept:
        raise ValueError("no calibration records remain after dropping warmup")
    for record in kept:
        if tuple(record["feature_names"]) != FEATURE_NAMES:
            raise ValueError("calibration feature schema does not match runtime")
        if len(record["features"]) != len(FEATURE_NAMES):
            raise ValueError("calibration feature width does not match runtime")
    return kept


def _split_groups(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    groups = sorted({int(record["request_group"]) for record in records})
    if len(groups) < 2:
        raise ValueError("at least two request groups are required")
    random.Random(seed).shuffle(groups)
    validation_count = min(
        len(groups) - 1,
        max(1, round(len(groups) * validation_fraction)),
    )
    validation_groups = set(groups[:validation_count])
    train_indices = [
        index
        for index, record in enumerate(records)
        if int(record["request_group"]) not in validation_groups
    ]
    validation_indices = [
        index
        for index, record in enumerate(records)
        if int(record["request_group"]) in validation_groups
    ]
    return train_indices, validation_indices


def _select_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    expected_deltas: torch.Tensor,
    block_surpluses: torch.Tensor,
    risk_reductions: torch.Tensor,
    *,
    max_false_positive_rate: float,
    min_expected_delta: float,
    min_remaining_surplus: float,
) -> tuple[float, dict[str, float]]:
    candidates = sorted(
        set(float(value) for value in probabilities.tolist()) | {1.0}
    )
    negatives = (labels < 0.5).sum().item()
    best: tuple[float, float, float, float, float, float] | None = None
    for threshold in candidates:
        routed = probabilities >= threshold
        false_positives = (routed & (labels < 0.5)).sum().item()
        false_positive_rate = false_positives / max(negatives, 1)
        expected_delta = (routed * expected_deltas).mean().item()
        remaining_surplus = (
            block_surpluses + routed * expected_deltas
        ).mean().item()
        risk_gain = (routed * risk_reductions).mean().item()
        coverage = routed.to(torch.float32).mean().item()
        if (
            false_positive_rate <= max_false_positive_rate
            and expected_delta >= min_expected_delta
            and remaining_surplus >= min_remaining_surplus
        ):
            candidate = (
                risk_gain,
                coverage,
                -false_positive_rate,
                expected_delta,
                remaining_surplus,
                threshold,
            )
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("no validation threshold satisfies gate constraints")
    (
        risk_gain,
        coverage,
        negative_fpr,
        expected_delta,
        remaining_surplus,
        threshold,
    ) = best
    return threshold, {
        "validation_risk_gain": risk_gain,
        "validation_route_fraction": coverage,
        "validation_false_positive_rate": -negative_fpr,
        "validation_expected_length_delta": expected_delta,
        "validation_remaining_block_surplus": remaining_surplus,
    }


def train_logistic_gate(
    records: list[dict[str, Any]],
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    max_false_positive_rate: float = 0.05,
    min_expected_delta: float = -1.0,
    min_remaining_surplus: float = 0.10,
) -> dict[str, Any]:
    train_indices, validation_indices = _split_groups(
        records,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    features = torch.tensor(
        [record["features"] for record in records],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [record["label"] for record in records],
        dtype=torch.float32,
    )
    expected_deltas = torch.tensor(
        [record["expected_length_delta"] for record in records],
        dtype=torch.float32,
    )
    risk_reductions = torch.tensor(
        [record["risk_reduction"] for record in records],
        dtype=torch.float32,
    )
    block_surpluses = torch.tensor(
        [record["block_verification_surplus"] for record in records],
        dtype=torch.float32,
    )
    train_x = features[train_indices]
    train_y = labels[train_indices]
    if train_y.sum().item() == 0 or train_y.sum().item() == train_y.numel():
        raise ValueError("training split must contain both gate labels")
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = (train_x - mean) / scale

    torch.manual_seed(seed)
    weight = torch.zeros(len(FEATURE_NAMES), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    positive = train_y.sum()
    negative = train_y.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).detach()
    optimizer = torch.optim.LBFGS(
        (weight, bias),
        max_iter=200,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = standardized @ weight + bias
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_y,
            pos_weight=positive_weight,
        )
        # Small L2 keeps a tiny calibration set from producing extreme gates.
        loss = loss + 1e-4 * weight.square().mean()
        loss.backward()
        return loss

    final_loss = float(optimizer.step(closure).item())
    validation_x = (features[validation_indices] - mean) / scale
    validation_probabilities = torch.sigmoid(
        validation_x @ weight.detach() + bias.detach()
    )
    threshold, threshold_metrics = _select_threshold(
        validation_probabilities,
        labels[validation_indices],
        expected_deltas[validation_indices],
        block_surpluses[validation_indices],
        risk_reductions[validation_indices],
        max_false_positive_rate=max_false_positive_rate,
        min_expected_delta=min_expected_delta,
        min_remaining_surplus=min_remaining_surplus,
    )
    train_probability = torch.sigmoid(
        standardized @ weight.detach() + bias.detach()
    )
    payload: dict[str, Any] = {
        "version": 1,
        "model_type": "logistic_block_gate",
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weight": weight.detach().tolist(),
        "bias": float(bias.detach().item()),
        "threshold": threshold,
        "training": {
            "seed": seed,
            "records": len(records),
            "request_groups": len(
                {int(record["request_group"]) for record in records}
            ),
            "train_records": len(train_indices),
            "validation_records": len(validation_indices),
            "train_positive_rate": float(train_y.mean().item()),
            "validation_positive_rate": float(
                labels[validation_indices].mean().item()
            ),
            "train_loss": final_loss,
            "train_accuracy_at_half": float(
                ((train_probability >= 0.5) == (train_y >= 0.5))
                .to(torch.float32)
                .mean()
                .item()
            ),
            **threshold_metrics,
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--drop-first-groups", type=int, default=1)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.05)
    parser.add_argument("--min-expected-delta", type=float, default=-1.0)
    parser.add_argument("--min-remaining-surplus", type=float, default=0.10)
    args = parser.parse_args()
    records = load_records(
        args.data,
        drop_first_groups=args.drop_first_groups,
    )
    payload = train_logistic_gate(
        records,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        max_false_positive_rate=args.max_false_positive_rate,
        min_expected_delta=args.min_expected_delta,
        min_remaining_surplus=args.min_remaining_surplus,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["training"], indent=2, sort_keys=True))
    print(f"Saved logistic block gate: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
