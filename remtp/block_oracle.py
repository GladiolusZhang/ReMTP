"""Structured ReMTP block traces and offline acceptance-length oracles.

Collection is opt-in and intentionally synchronizes tiny K<=6 tensors to the
CPU.  It must never be enabled for reported throughput runs.  The replay tool
uses the recorded target/draft probabilities and the same verification
uniforms to separate eligibility, budget, control and allocation bottlenecks.
All counterfactuals are within-round estimates: a changed rejection/recovery
would alter later real prefixes, which the recorded target forward cannot
represent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import torch

if TYPE_CHECKING:
    from remtp.risk_entropy_mtp import RiskEntropyConfig, RiskEntropyResult


_AUDIT_REQUEST_INDEX = -1
_AUDIT_LAST_OUTPUT_LENGTH: int | None = None
_AUDIT_LAST_REQUEST_ID: str | None = None
_AUDIT_ROUND_INDEX = 0


def _floats(value: torch.Tensor) -> list[float]:
    return value.detach().to(device="cpu", dtype=torch.float32).tolist()


def _ints(value: torch.Tensor) -> list[int]:
    return value.detach().to(device="cpu", dtype=torch.int64).tolist()


def _bools(value: torch.Tensor) -> list[bool]:
    return value.detach().to(device="cpu", dtype=torch.bool).tolist()


def _request_position(
    sampling_metadata: Any,
    request_id: str | None,
    output_length_before_round: int,
) -> tuple[int, int]:
    global _AUDIT_REQUEST_INDEX, _AUDIT_LAST_OUTPUT_LENGTH, _AUDIT_LAST_REQUEST_ID

    if request_id is not None:
        if request_id != _AUDIT_LAST_REQUEST_ID:
            _AUDIT_REQUEST_INDEX += 1
            _AUDIT_LAST_REQUEST_ID = request_id
        _AUDIT_LAST_OUTPUT_LENGTH = output_length_before_round
        return _AUDIT_REQUEST_INDEX, output_length_before_round

    rows = getattr(sampling_metadata, "output_token_ids", None)
    output_length = len(rows[0]) if rows else 0
    if (
        _AUDIT_LAST_OUTPUT_LENGTH is None
        or output_length <= _AUDIT_LAST_OUTPUT_LENGTH
    ):
        _AUDIT_REQUEST_INDEX += 1
    _AUDIT_LAST_OUTPUT_LENGTH = output_length
    return _AUDIT_REQUEST_INDEX, output_length


def append_audit_round(
    *,
    path: str,
    dataset: str,
    result: "RiskEntropyResult",
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor | None,
    output_token_ids: torch.Tensor,
    sampling_metadata: Any,
    request_id: str | None,
    output_length_before_round: int,
    previous_debt: torch.Tensor,
    config: "RiskEntropyConfig",
    placeholder_token_id: int,
) -> None:
    """Append one synchronization-heavy audit record."""

    global _AUDIT_ROUND_INDEX
    _AUDIT_ROUND_INDEX += 1
    request_index, output_length = _request_position(
        sampling_metadata,
        request_id,
        output_length_before_round,
    )

    rows = draft_token_ids.shape[0]
    requested_top_count = max(2, int(os.getenv("REMTP_ORACLE_TOPK", "3")))
    top_count = min(requested_top_count, draft_probs.shape[-1])
    q_top_values, q_top_ids = torch.topk(
        draft_probs.to(torch.float32),
        k=top_count,
        dim=-1,
        largest=True,
        sorted=True,
    )
    p_at_q_top = target_probs.to(torch.float32).gather(1, q_top_ids)
    p_top_values, p_top_ids = torch.topk(
        target_probs.to(torch.float32),
        k=top_count,
        dim=-1,
        largest=True,
        sorted=True,
    )

    valid_count = int(
        (output_token_ids[0] != placeholder_token_id).sum().item()
    )
    accepted_drafts = max(0, min(rows, valid_count - 1))
    committed_tokens = _ints(output_token_ids[0, :valid_count])
    rejected_index = accepted_drafts if accepted_drafts < rows else None
    uniforms = (
        []
        if uniform_probs is None
        else _floats(uniform_probs[:rows])
    )
    decisions: list[str] = []
    acceptance_sources: list[str] = []
    strict_values = _floats(result.strict_acceptance)
    relaxed_values = _floats(result.relaxed_acceptance)
    for index in range(rows):
        if index < accepted_drafts:
            decisions.append("accepted")
            if uniforms and uniforms[index] > strict_values[index]:
                acceptance_sources.append("relaxed-only")
            else:
                acceptance_sources.append("strict")
        elif index == rejected_index:
            decisions.append("rejected")
            acceptance_sources.append("rejected")
        else:
            decisions.append("skipped")
            acceptance_sources.append("skipped")
    debt = float(previous_debt.detach().to(torch.float32).item())
    debt_factor = 1.0 + config.debt_scale * debt
    no_debt_threshold = (
        config.remtp_gap_floor
        + (config.base_log_gap - config.remtp_gap_floor)
        * torch.exp(
            -result.target_margin / config.remtp_margin_temperature
        )
    )

    record = {
        "version": 1,
        "dataset": dataset,
        "request_id": request_id,
        "request_index": request_index,
        "round_index": _AUDIT_ROUND_INDEX,
        "output_length_before_round": output_length,
        "draft_tokens": _ints(draft_token_ids),
        "accepted_drafts": accepted_drafts,
        "decisions": decisions,
        "acceptance_sources": acceptance_sources,
        "rejected_head": (
            None if rejected_index is None else rejected_index + 1
        ),
        "committed_tokens": committed_tokens,
        "anchor_kind": "bonus" if rejected_index is None else "recovery",
        "previous_debt": debt,
        "p_y": _floats(result.target_candidate_probs),
        "q_y": _floats(result.draft_candidate_probs),
        "strict_acceptance": _floats(result.strict_acceptance),
        "relaxed_acceptance": _floats(result.relaxed_acceptance),
        "allocated_tv": _floats(result.allocated_tv),
        "desired_tv": _floats(result.desired_tv),
        "local_risk": _floats(result.local_risk),
        "candidate_rank": _ints(result.candidate_rank),
        "candidate_gap": _floats(result.candidate_log_gap),
        "target_margin": _floats(result.target_margin),
        "threshold": _floats(result.threshold),
        "threshold_no_debt": _floats(no_debt_threshold),
        "target_head_mass": _floats(result.target_head_mass),
        "local_eligible": _bools(result.local_eligible),
        "sentinel_pass": _bools(result.sentinel_pass),
        "sentinel_checks": _ints(result.sentinel_checks),
        "sentinel_strength": _floats(result.sentinel_strength),
        "continuation_support": _floats(result.continuation_support),
        "continuation_regret": _floats(result.continuation_regret),
        "positive_certificate": _bools(result.positive_certificate),
        "uniforms": uniforms,
        "q_top_ids": q_top_ids.detach().to(device="cpu").tolist(),
        "q_top_probs": q_top_values.detach().to(device="cpu").tolist(),
        "p_at_q_top": p_at_q_top.detach().to(device="cpu").tolist(),
        "recorded_q_top_k": top_count,
        "p_top_ids": p_top_ids.detach().to(device="cpu").tolist(),
        "p_top_probs": p_top_values.detach().to(device="cpu").tolist(),
        "config": {
            "max_rank": config.max_target_rank,
            "rank_limit": config.scheme2_rank_limit,
            "base_gap": config.base_log_gap,
            "gap_floor": config.remtp_gap_floor,
            "margin_temperature": config.remtp_margin_temperature,
            "risky_rank": config.remtp_risky_rank,
            "risky_gap": config.remtp_risky_log_gap,
            "risky_min_checks": config.remtp_risky_min_checks,
            "sentinel_min_checks": config.sentinel_min_checks,
            "sentinel_soft_floor": config.sentinel_soft_floor,
            "cactus_delta": config.cactus_delta,
            "per_token_tv": config.per_token_tv_cap,
            "block_tv": config.block_tv_cap,
            "risk_budget": config.risk_budget,
            "debt_scale": config.debt_scale,
            "continuation_gap": config.remtp_continuation_gap,
            "continuation_support": config.remtp_continuation_support,
            "continuation_regret": config.remtp_continuation_regret,
            "continuation_horizon": config.remtp_continuation_horizon,
            "effective_block_tv": config.block_tv_cap / debt_factor,
            "effective_risk_budget": config.risk_budget / debt_factor,
        },
    }

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def expected_mal(acceptance: list[float]) -> float:
    reach = 1.0
    total = 1.0
    for probability in acceptance:
        reach *= min(1.0, max(0.0, probability))
        total += reach
    return total


def replay_mal(acceptance: list[float], uniforms: list[float]) -> float:
    if len(uniforms) != len(acceptance):
        return expected_mal(acceptance)
    accepted = 0
    for probability, uniform in zip(acceptance, uniforms):
        if uniform > probability:
            break
        accepted += 1
    return 1.0 + accepted


def _acceptance(
    p_y: list[float], q_y: list[float], allocation: list[float]
) -> list[float]:
    return [
        min(1.0, (p + tv) / max(q, 1e-30))
        for p, q, tv in zip(p_y, q_y, allocation)
    ]


def _reach(acceptance: list[float]) -> list[float]:
    values: list[float] = []
    prefix = 1.0
    for probability in acceptance:
        values.append(prefix)
        prefix *= probability
    return values


def _suffix_value(acceptance: list[float]) -> list[float]:
    values = [1.0] * len(acceptance)
    for index in range(len(acceptance) - 2, -1, -1):
        values[index] = 1.0 + acceptance[index + 1] * values[index + 1]
    return values


def _reach_weighted_cost(
    allocation: list[float], p_y: list[float], q_y: list[float]
) -> float:
    acceptance = _acceptance(p_y, q_y, allocation)
    return sum(
        reach * tv for reach, tv in zip(_reach(acceptance), allocation)
    )


def dynamic_mal_allocation(
    *,
    p_y: list[float],
    q_y: list[float],
    capacity: list[float],
    eligible: list[bool],
    block_budget: float,
    local_risk: list[float],
    risk_budget: float,
    reach_budget: float | None = None,
    steps: int = 256,
) -> list[float]:
    """Offline coordinate solver for the six-dimensional MAL objective."""

    allocation = [0.0] * len(p_y)
    step_tv = block_budget / max(steps, 1)
    for _ in range(steps + len(p_y)):
        acceptance = _acceptance(p_y, q_y, allocation)
        reach = _reach(acceptance)
        suffix = _suffix_value(acceptance)
        raw_spent = sum(allocation)
        risk_spent = sum(
            value * risk for value, risk in zip(allocation, local_risk)
        )
        raw_remaining = max(0.0, block_budget - raw_spent)
        risk_remaining = max(0.0, risk_budget - risk_spent)
        if raw_remaining <= 1e-12:
            break

        candidates: list[tuple[float, int, float]] = []
        current_reach_cost = _reach_weighted_cost(allocation, p_y, q_y)
        for index in range(len(p_y)):
            remaining_capacity = max(0.0, capacity[index] - allocation[index])
            if not eligible[index] or remaining_capacity <= 1e-12:
                continue
            amount = min(step_tv, remaining_capacity, raw_remaining)
            if local_risk[index] > 0.0:
                amount = min(amount, risk_remaining / local_risk[index])
            if amount <= 1e-12:
                continue

            trial = allocation.copy()
            trial[index] += amount
            reach_delta = (
                _reach_weighted_cost(trial, p_y, q_y)
                - current_reach_cost
            )
            if reach_budget is not None:
                reach_remaining = max(0.0, reach_budget - current_reach_cost)
                if reach_delta > reach_remaining + 1e-12:
                    low, high = 0.0, amount
                    for _ in range(20):
                        middle = 0.5 * (low + high)
                        trial[index] = allocation[index] + middle
                        cost = _reach_weighted_cost(trial, p_y, q_y)
                        if cost <= reach_budget:
                            low = middle
                        else:
                            high = middle
                    amount = low
                    if amount <= 1e-12:
                        continue
                    reach_delta = reach_remaining

            marginal_gain = (
                reach[index]
                * suffix[index]
                / max(q_y[index], 1e-30)
            )
            normalized_cost = max(
                1.0 / max(raw_remaining, 1e-30),
                local_risk[index] / max(risk_remaining, 1e-30),
                (
                    reach_delta / max(amount, 1e-30)
                    / max(
                        (reach_budget or math.inf) - current_reach_cost,
                        1e-30,
                    )
                    if reach_budget is not None
                    else 0.0
                ),
            )
            candidates.append((marginal_gain / normalized_cost, index, amount))

        if not candidates:
            break
        _, selected, amount = max(candidates)
        allocation[selected] += amount
    return allocation


def _legacy_ultra(record: dict[str, Any]) -> list[float]:
    p_y = list(map(float, record["p_y"]))
    q_y = list(map(float, record["q_y"]))
    strict = list(map(float, record["strict_acceptance"]))
    gaps = list(map(float, record["candidate_gap"]))
    ranks = list(map(int, record["candidate_rank"]))
    head_mass = list(map(float, record["target_head_mass"]))
    desired = [
        min(
            math.sqrt(max(0.0, 2.0 * p * (1.0 - p))),
            max(0.0, q - p),
            0.55,
        )
        for p, q in zip(p_y, q_y)
    ]
    checks = [3] * len(p_y)
    strength = [1.0] * len(p_y)
    sentinel = [True] * len(p_y)
    for index in range(len(p_y) - 1):
        count = int(gaps[index + 1] <= 2.40)
        count += int(head_mass[index + 1] >= 0.03)
        count += int(q_y[index + 1] >= 0.003)
        checks[index] = count
        strength[index] = count / 3.0
        sentinel[index] = count >= 1
    sentinel[-1] = gaps[-1] <= 2.0
    strength[-1] = float(sentinel[-1])

    allocation = [0.0] * len(p_y)
    active = True
    spent = 0.0
    for index in range(len(p_y)):
        local = ranks[index] <= 8 and gaps[index] <= 2.0
        allow = active and (strict[index] >= 1.0 - 1e-7 or (local and sentinel[index]))
        if allow and strict[index] < 1.0 - 1e-7:
            soft = 0.65 + 0.35 * strength[index]
            allocation[index] = min(desired[index], max(0.0, 1.85 - spent)) * soft
            spent += allocation[index]
        active = allow
    return _acceptance(p_y, q_y, allocation)


def _scenario(
    record: dict[str, Any], acceptance: list[float]
) -> tuple[float, float]:
    return (
        expected_mal(acceptance),
        replay_mal(acceptance, list(map(float, record["uniforms"]))),
    )


def analyze_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scenario_names = (
        "current",
        "scheme2_ultra_replay",
        "single_successor_check",
        "fixed_gap_2p00",
        "fixed_gap_2p15",
        "local_without_sentinel",
        "gate_ceiling",
        "budget_ceiling",
        "control_ceiling",
        "joint_optimizer",
        "reach_matched_joint",
        "reach_budget_ceiling",
        "positive_continuation",
    )
    totals = {
        name: {"expected_mal": 0.0, "same_uniform_mal": 0.0}
        for name in scenario_names
    }
    rounds = 0
    max_heads = 0
    head: list[dict[str, float]] = []
    first_failure: list[int] = []
    rejection_reasons = {"rank": 0, "gap": 0, "sentinel": 0}
    top2 = {"failures": 0, "alternative_accepts": 0, "by_head": []}
    budget_usage = {
        "allocated_tv": 0.0,
        "desired_eligible_tv": 0.0,
        "configured_block_tv": 0.0,
        "allocated_risk_mass": 0.0,
        "configured_risk_budget": 0.0,
    }

    for record in records:
        rounds += 1
        p_y = list(map(float, record["p_y"]))
        q_y = list(map(float, record["q_y"]))
        strict = list(map(float, record["strict_acceptance"]))
        relaxed = list(map(float, record["relaxed_acceptance"]))
        desired = list(map(float, record["desired_tv"]))
        allocated = list(map(float, record["allocated_tv"]))
        local_risk = list(map(float, record["local_risk"]))
        ranks = list(map(int, record["candidate_rank"]))
        gaps = list(map(float, record["candidate_gap"]))
        threshold = list(map(float, record["threshold"]))
        threshold_no_debt = list(map(float, record["threshold_no_debt"]))
        local = list(map(bool, record["local_eligible"]))
        sentinel = list(map(bool, record["sentinel_pass"]))
        checks = list(map(int, record["sentinel_checks"]))
        uniforms = list(map(float, record["uniforms"]))
        config = record["config"]
        count = len(p_y)
        max_heads = max(max_heads, count)
        while len(head) < count:
            head.append(
                {
                    "rounds": 0.0,
                    "reached": 0.0,
                    "strict_probability": 0.0,
                    "relaxed_probability": 0.0,
                    "allocated_tv": 0.0,
                    "eligible": 0.0,
                    "sentinel": 0.0,
                    "relaxed_only": 0.0,
                    "force_to_one_tv": 0.0,
                    "marginal_mal_per_tv": 0.0,
                }
            )
        while len(first_failure) <= count:
            first_failure.append(0)
        while len(top2["by_head"]) < count:
            top2["by_head"].append({"failures": 0, "alternative_accepts": 0})

        eligible = [a and b for a, b in zip(local, sentinel)]
        current = relaxed
        ultra = _legacy_ultra(record)
        gate = [
            1.0 if allowed and value < 1.0 - 1e-7 else value
            for value, allowed in zip(strict, eligible)
        ]
        budget = _acceptance(
            p_y,
            q_y,
            [tv if allowed else 0.0 for tv, allowed in zip(desired, eligible)],
        )

        block_budget = float(config["effective_block_tv"])
        risk_budget = float(config["effective_risk_budget"])

        def allocate_for(candidate_eligible: list[bool]) -> list[float]:
            candidate_allocation = dynamic_mal_allocation(
                p_y=p_y,
                q_y=q_y,
                capacity=desired,
                eligible=candidate_eligible,
                block_budget=block_budget,
                local_risk=local_risk,
                risk_budget=risk_budget,
            )
            return _acceptance(p_y, q_y, candidate_allocation)

        single_check_eligible = [
            is_local and check >= 1
            for is_local, check in zip(local, checks)
        ]
        fixed_2p00_eligible = [
            rank <= int(config["max_rank"])
            and gap <= 2.0
            and is_sentinel
            for rank, gap, is_sentinel in zip(ranks, gaps, sentinel)
        ]
        fixed_2p15_eligible = [
            rank <= int(config["max_rank"])
            and gap <= 2.15
            and is_sentinel
            for rank, gap, is_sentinel in zip(ranks, gaps, sentinel)
        ]
        single_check = allocate_for(single_check_eligible)
        fixed_2p00 = allocate_for(fixed_2p00_eligible)
        fixed_2p15 = allocate_for(fixed_2p15_eligible)
        without_sentinel = allocate_for(local)

        budget_usage["allocated_tv"] += sum(allocated)
        budget_usage["desired_eligible_tv"] += sum(
            value for value, allowed in zip(desired, eligible) if allowed
        )
        budget_usage["configured_block_tv"] += block_budget
        budget_usage["allocated_risk_mass"] += sum(
            value * risk for value, risk in zip(allocated, local_risk)
        )
        budget_usage["configured_risk_budget"] += risk_budget
        joint_allocation = dynamic_mal_allocation(
            p_y=p_y,
            q_y=q_y,
            capacity=desired,
            eligible=eligible,
            block_budget=block_budget,
            local_risk=local_risk,
            risk_budget=risk_budget,
        )
        joint = _acceptance(p_y, q_y, joint_allocation)

        current_reach_budget = _reach_weighted_cost(allocated, p_y, q_y)
        reach_matched_allocation = dynamic_mal_allocation(
            p_y=p_y,
            q_y=q_y,
            capacity=desired,
            eligible=eligible,
            block_budget=block_budget,
            local_risk=local_risk,
            risk_budget=risk_budget,
            reach_budget=current_reach_budget,
        )
        reach_ceiling_allocation = dynamic_mal_allocation(
            p_y=p_y,
            q_y=q_y,
            capacity=desired,
            eligible=eligible,
            block_budget=block_budget,
            local_risk=local_risk,
            risk_budget=risk_budget,
            reach_budget=block_budget,
        )
        reach_matched = _acceptance(p_y, q_y, reach_matched_allocation)
        reach_ceiling = _acceptance(p_y, q_y, reach_ceiling_allocation)

        control_eligible = [
            rank <= int(config["rank_limit"]) and gap <= no_debt_gap
            for rank, gap, no_debt_gap in zip(ranks, gaps, threshold_no_debt)
        ]
        control_allocation = dynamic_mal_allocation(
            p_y=p_y,
            q_y=q_y,
            capacity=desired,
            eligible=control_eligible,
            block_budget=float(config["block_tv"]),
            local_risk=local_risk,
            risk_budget=float(config["risk_budget"]),
        )
        control = _acceptance(p_y, q_y, control_allocation)

        continuation = [0.0] * count
        for index in range(count - 1):
            product = 1.0
            for future in range(index + 1, min(count, index + 4)):
                product *= strict[future]
            continuation[index] = product
        positive_eligible = eligible.copy()
        for index in range(count - 1):
            if (
                not positive_eligible[index]
                and ranks[index] <= int(config["rank_limit"])
                and gaps[index] <= 2.50
                and continuation[index] >= 0.65
            ):
                positive_eligible[index] = True
        positive_allocation = dynamic_mal_allocation(
            p_y=p_y,
            q_y=q_y,
            capacity=desired,
            eligible=positive_eligible,
            block_budget=block_budget,
            local_risk=local_risk,
            risk_budget=risk_budget,
        )
        positive = _acceptance(p_y, q_y, positive_allocation)

        scenarios = {
            "current": current,
            "scheme2_ultra_replay": ultra,
            "single_successor_check": single_check,
            "fixed_gap_2p00": fixed_2p00,
            "fixed_gap_2p15": fixed_2p15,
            "local_without_sentinel": without_sentinel,
            "gate_ceiling": gate,
            "budget_ceiling": budget,
            "control_ceiling": control,
            "joint_optimizer": joint,
            "reach_matched_joint": reach_matched,
            "reach_budget_ceiling": reach_ceiling,
            "positive_continuation": positive,
        }
        for name, acceptance in scenarios.items():
            expected, replayed = _scenario(record, acceptance)
            totals[name]["expected_mal"] += expected
            totals[name]["same_uniform_mal"] += replayed

        accepted = int(record["accepted_drafts"])
        first_failure[min(accepted, count)] += 1
        strict_reach = _reach(strict)
        strict_suffix = _suffix_value(strict)
        for index in range(count):
            values = head[index]
            values["rounds"] += 1.0
            values["reached"] += float(index <= accepted)
            values["strict_probability"] += strict[index]
            values["relaxed_probability"] += relaxed[index]
            values["allocated_tv"] += allocated[index]
            values["eligible"] += float(eligible[index])
            values["sentinel"] += float(sentinel[index])
            values["force_to_one_tv"] += max(0.0, q_y[index] - p_y[index])
            values["marginal_mal_per_tv"] += (
                strict_reach[index]
                * strict_suffix[index]
                / max(q_y[index], 1e-30)
            )
            if (
                len(uniforms) == count
                and index < accepted
                and uniforms[index] > strict[index]
            ):
                values["relaxed_only"] += 1.0

            if strict[index] < 1.0 - 1e-7 and not eligible[index]:
                rejection_reasons["rank"] += int(
                    ranks[index] > int(config["rank_limit"])
                )
                rejection_reasons["gap"] += int(gaps[index] > threshold[index])
                rejection_reasons["sentinel"] += int(not sentinel[index])

        if accepted < count and len(uniforms) == count:
            top2["failures"] += 1
            top2["by_head"][accepted]["failures"] += 1
            actual_id = int(record["draft_tokens"][accepted])
            alternatives = []
            for token_id, q_prob, p_prob in zip(
                record["q_top_ids"][accepted],
                record["q_top_probs"][accepted],
                record["p_at_q_top"][accepted],
            ):
                if int(token_id) == actual_id:
                    continue
                alternatives.append(min(1.0, float(p_prob) / max(float(q_prob), 1e-30)))
            if alternatives and uniforms[accepted] <= max(alternatives):
                top2["alternative_accepts"] += 1
                top2["by_head"][accepted]["alternative_accepts"] += 1

    if rounds == 0:
        raise ValueError("audit contains no records")

    for values in totals.values():
        values["expected_mal"] /= rounds
        values["same_uniform_mal"] /= rounds
    for values in head:
        denominator = max(values["rounds"], 1.0)
        for key in tuple(values):
            if key != "rounds":
                values[key] /= denominator
    for key in budget_usage:
        budget_usage[key] /= rounds
    budget_usage["block_tv_utilization"] = (
        budget_usage["allocated_tv"]
        / max(budget_usage["configured_block_tv"], 1e-30)
    )
    budget_usage["risk_budget_utilization"] = (
        budget_usage["allocated_risk_mass"]
        / max(budget_usage["configured_risk_budget"], 1e-30)
    )
    top2["alternative_accept_rate"] = (
        top2["alternative_accepts"] / max(top2["failures"], 1)
    )
    for values in top2["by_head"]:
        values["alternative_accept_rate"] = (
            values["alternative_accepts"] / max(values["failures"], 1)
        )

    current_expected = totals["current"]["expected_mal"]
    current_replay = totals["current"]["same_uniform_mal"]
    for values in totals.values():
        values["expected_delta_vs_current"] = (
            values["expected_mal"] - current_expected
        )
        values["same_uniform_delta_vs_current"] = (
            values["same_uniform_mal"] - current_replay
        )
    return {
        "rounds": rounds,
        "max_heads": max_heads,
        "scenarios": totals,
        "per_head": head,
        "first_failure_counts": first_failure,
        "rejection_reasons": rejection_reasons,
        "top2_candidate_oracle": top2,
        "budget_usage": budget_usage,
        "caveat": (
            "Counterfactuals replay one already-verified block. They do not "
            "model changed recovery tokens or subsequent target states."
        ),
    }


def render_report(result: dict[str, Any], source: Path) -> str:
    lines = [
        "# ReMTP block-oracle audit",
        "",
        f"- source: `{source.resolve()}`",
        f"- rounds: {result['rounds']}",
        f"- caveat: {result['caveat']}",
        "",
        "## MAL ceilings and counterfactuals",
        "",
        "| scenario | expected MAL | delta | same-u MAL | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in result["scenarios"].items():
        lines.append(
            f"| {name} | {values['expected_mal']:.4f} | "
            f"{values['expected_delta_vs_current']:+.4f} | "
            f"{values['same_uniform_mal']:.4f} | "
            f"{values['same_uniform_delta_vs_current']:+.4f} |"
        )
    usage = result["budget_usage"]
    lines.extend(
        [
            "",
            "## Budget utilization",
            "",
            f"- mean allocated TV: {usage['allocated_tv']:.4f} / "
            f"{usage['configured_block_tv']:.4f} "
            f"({100.0 * usage['block_tv_utilization']:.1f}%)",
            f"- mean eligible saturation capacity: "
            f"{usage['desired_eligible_tv']:.4f}",
            f"- mean allocated risk mass: "
            f"{usage['allocated_risk_mass']:.4f} / "
            f"{usage['configured_risk_budget']:.4f} "
            f"({100.0 * usage['risk_budget_utilization']:.1f}%)",
        ]
    )
    lines.extend(
        [
            "",
            "## Per-head diagnostics",
            "",
            "| head | reach | strict A | relaxed A | eligibility | sentinel | "
            "allocated TV | relaxed-only | force-to-1 TV | marginal MAL/TV |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, values in enumerate(result["per_head"], start=1):
        lines.append(
            f"| {index} | {values['reached']:.3f} | "
            f"{values['strict_probability']:.3f} | "
            f"{values['relaxed_probability']:.3f} | "
            f"{values['eligible']:.3f} | {values['sentinel']:.3f} | "
            f"{values['allocated_tv']:.4f} | "
            f"{values['relaxed_only']:.3f} | "
            f"{values['force_to_one_tv']:.4f} | "
            f"{values['marginal_mal_per_tv']:.3f} |"
        )
    top2 = result["top2_candidate_oracle"]
    lines.extend(
        [
            "",
            "## Failure and candidate coverage",
            "",
            f"- first-failure counts (0..K accepted): "
            f"`{result['first_failure_counts']}`",
            f"- rejection reasons: `{result['rejection_reasons']}`",
            f"- top-2 alternative has local target support at the observed "
            f"first failure: "
            f"{top2['alternative_accepts']}/{top2['failures']} "
            f"({100.0 * top2['alternative_accept_rate']:.1f}%)",
            "- this top-2 statistic is a branch-coverage signal, not a MAL "
            "gain: a real tree must also draft and target-verify the "
            "alternative-conditioned suffix.",
            "",
            "## Decision",
            "",
        ]
    )
    joint_gain = result["scenarios"]["joint_optimizer"][
        "expected_delta_vs_current"
    ]
    positive_gain = result["scenarios"]["positive_continuation"][
        "expected_delta_vs_current"
    ]
    candidate_rate = top2["alternative_accept_rate"]
    if joint_gain >= 0.10:
        lines.append(
            f"- Joint allocation is worth an online implementation "
            f"(expected MAL gain {joint_gain:+.3f})."
        )
    else:
        lines.append(
            f"- Joint allocation alone is below the 0.10 GSM8K bar "
            f"(expected MAL gain {joint_gain:+.3f})."
        )
    lines.append(
        f"- Positive continuation certificate gain: {positive_gain:+.3f}."
    )
    lines.append(
        f"- Top-2 local candidate coverage at failures: "
        f"{100.0 * candidate_rate:.1f}%."
    )
    lines.append("")
    return "\n".join(lines)


def load_records(
    path: Path, *, skip_requests: int = 0
) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if int(record.get("request_index", 0)) < skip_requests:
                    continue
                records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--skip-requests",
        type=int,
        default=0,
        help="Ignore leading warmup requests recorded by the benchmark.",
    )
    args = parser.parse_args()

    if args.skip_requests < 0:
        parser.error("--skip-requests must be non-negative")
    result = analyze_records(
        load_records(args.input, skip_requests=args.skip_requests)
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_report(result, args.input), encoding="utf-8"
    )
    print(f"Audit JSON: {args.output_json.resolve()}")
    print(f"Audit report: {args.output_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
