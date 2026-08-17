"""Offline analysis and static search for Proposal-Calibrated MTP traces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from remtp.proposal_calibration import (
    ProposalCalibrationConfig,
    transforms_by_name,
)


def expected_mal(acceptance: Iterable[float]) -> float:
    reach = 1.0
    total = 1.0
    for value in acceptance:
        reach *= min(1.0, max(0.0, float(value)))
        total += reach
    return total


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.fmean(rows) if rows else 0.0


def _strategy_mal(record: dict[str, Any], config: list[str]) -> float:
    grid = record["strategy_overlap"]
    return expected_mal(grid[name][head] for head, name in enumerate(config))


def _dataset_score(records: list[dict[str, Any]], config: list[str]) -> float:
    return _mean(_strategy_mal(row, config) for row in records)


def _split_requests(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in records if int(row.get("request_index", 0)) % 5 != 0]
    heldout = [row for row in records if int(row.get("request_index", 0)) % 5 == 0]
    if not train or not heldout:
        split = max(1, int(0.8 * len(records)))
        train, heldout = records[:split], records[split:]
    return train, heldout or train


def _joint_objective(
    datasets: dict[str, list[dict[str, Any]]],
    config: list[str],
) -> float:
    return _mean(_dataset_score(records, config) for records in datasets.values())


def coordinate_search(
    datasets: dict[str, list[dict[str, Any]]],
    strategy_names: list[str],
    heads: int,
    passes: int = 4,
) -> tuple[list[str], float]:
    """Small exact-evaluation coordinate search over a six-head config."""
    selected = ["identity"] * heads
    best_score = _joint_objective(datasets, selected)
    for _ in range(passes):
        changed = False
        for head in range(heads):
            local_name = selected[head]
            local_score = best_score
            for name in strategy_names:
                trial = selected.copy()
                trial[head] = name
                score = _joint_objective(datasets, trial)
                if score > local_score + 1e-12:
                    local_name, local_score = name, score
            if local_name != selected[head]:
                selected[head] = local_name
                best_score = local_score
                changed = True
        if not changed:
            break
    return selected, best_score


def _quantile_bins(values: list[float], bins: int = 4) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    return [
        ordered[min(len(ordered) - 1, int(len(ordered) * index / bins))]
        for index in range(1, bins)
    ]


def _bucket(value: float, boundaries: list[float]) -> int:
    return sum(value > boundary for boundary in boundaries)


def analyze_head_statistics(records: list[dict[str, Any]], heads: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for head in range(heads):
        entropy_values = [float(row["q_entropy"][head]) for row in records]
        margin_values = [float(row["q_log_margin"][head]) for row in records]
        entropy_boundaries = _quantile_bins(entropy_values)
        margin_boundaries = _quantile_bins(margin_values)
        entropy_bins: dict[int, list[float]] = defaultdict(list)
        margin_bins: dict[int, list[float]] = defaultdict(list)
        miss = top2 = top3 = 0
        contribution: list[float] = []
        for row in records:
            q_ids = row["q_top_ids"][head]
            target_top = int(row["p_top1_ids"][head])
            if int(q_ids[0]) != target_top:
                miss += 1
                top2 += int(target_top in list(map(int, q_ids[:2])))
                top3 += int(target_top in list(map(int, q_ids[:3])))
            overlap = float(row["native_overlap"][head])
            entropy_bins[_bucket(float(row["q_entropy"][head]), entropy_boundaries)].append(overlap)
            margin_bins[_bucket(float(row["q_log_margin"][head]), margin_boundaries)].append(overlap)
            reach = 1.0
            for index in range(head + 1):
                reach *= float(row["native_overlap"][index])
            contribution.append(reach)
        output.append(
            {
                "head": head + 1,
                "rows": len(records),
                "mean_overlap": _mean(float(row["native_overlap"][head]) for row in records),
                "mean_sampled_strict_acceptance": _mean(
                    float(row["sampled_strict_acceptance"][head]) for row in records
                ),
                "mean_q_entropy": _mean(entropy_values),
                "mean_q_log_margin": _mean(margin_values),
                "top1_misses": miss,
                "target_top1_in_q_top2_given_miss": top2 / miss if miss else 0.0,
                "target_top1_in_q_top3_given_miss": top3 / miss if miss else 0.0,
                "weighted_mal_contribution": _mean(contribution),
                "entropy_quartile_overlap": [
                    _mean(entropy_bins[index]) for index in range(4)
                ],
                "margin_quartile_overlap": [
                    _mean(margin_bins[index]) for index in range(4)
                ],
            }
        )
    return output


def five_plus_one_upper_bound(records: list[dict[str, Any]]) -> dict[str, float]:
    chain6: list[float] = []
    oracle: list[float] = []
    for row in records:
        acceptance = list(map(float, row["native_overlap"]))
        chain6.append(expected_mal(acceptance))
        chain5 = expected_mal(acceptance[:5])
        alt_target_mass = float(row["p_at_q_top"][0][1])
        # Optimistic upper bound: the spare root node rescues as much of the
        # primary-root failure mass as target P assigns to Q's rank-2 token.
        # It ignores branch-state/kernel cost, so failure to clear the gate is
        # strong evidence against implementing this tree.
        rescue = min(max(0.0, 1.0 - acceptance[0]), alt_target_mass)
        oracle.append(chain5 + rescue)
    baseline = _mean(chain6)
    upper = _mean(oracle)
    return {
        "chain6_expected_mal": baseline,
        "five_plus_one_optimistic_upper_mal": upper,
        "upper_bound_delta": upper - baseline,
    }


def analyze_legacy(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract only defensible metrics from pre-calibration top-2 traces."""
    heads = max((len(row.get("q_y", [])) for row in records), default=0)
    per_head = []
    for head in range(heads):
        valid = [row for row in records if len(row.get("q_y", [])) > head]
        miss = hit2 = 0
        for row in valid:
            q_ids = row.get("q_top_ids", [[]] * heads)[head]
            p_ids = row.get("p_top_ids", [[]] * heads)[head]
            if q_ids and p_ids and int(q_ids[0]) != int(p_ids[0]):
                miss += 1
                hit2 += int(int(p_ids[0]) in list(map(int, q_ids[:2])))
        per_head.append(
            {
                "head": head + 1,
                "rows": len(valid),
                "mean_sampled_strict_acceptance": _mean(
                    float(row["strict_acceptance"][head]) for row in valid
                ),
                "top1_misses": miss,
                "target_top1_in_q_top2_given_miss": hit2 / miss if miss else 0.0,
            }
        )
    return {
        "records": len(records),
        "head_statistics": per_head,
        "limitations": [
            "full P/Q rows were not recorded, so exact sum_x min(P,Q) is unavailable",
            "Q entropy and exact temperature/top-k replay are unavailable",
            "only top-2 candidate coverage can be audited",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Proposal-Calibrated MTP offline analysis",
        "",
        "This report uses strict probability overlap. No relaxed target distribution is used.",
        "",
    ]
    if payload.get("legacy"):
        lines.extend(["## Existing-log audit", ""])
        for dataset, section in payload["legacy"].items():
            lines.append(f"### {dataset}")
            lines.append("")
            lines.append("| head | sampled strict acceptance | top-2 rescue among top-1 misses |")
            lines.append("|---:|---:|---:|")
            for row in section["head_statistics"]:
                lines.append(
                    f"| {row['head']} | {row['mean_sampled_strict_acceptance']:.4f} | "
                    f"{100*row['target_top1_in_q_top2_given_miss']:.1f}% |"
                )
            lines.append("")
        lines.append("Old logs do not contain full P/Q, so exact overlap and calibration replay require the compact trace below.")
        lines.append("")

    lines.extend(["## Compact full-distribution statistics", ""])
    for dataset, section in payload["datasets"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| head | overlap | Q entropy | Q log-margin | top-2 rescue | top-3 rescue | MAL contribution |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in section["head_statistics"]:
            lines.append(
                f"| {row['head']} | {row['mean_overlap']:.4f} | "
                f"{row['mean_q_entropy']:.3f} | {row['mean_q_log_margin']:.3f} | "
                f"{100*row['target_top1_in_q_top2_given_miss']:.1f}% | "
                f"{100*row['target_top1_in_q_top3_given_miss']:.1f}% | "
                f"{row['weighted_mal_contribution']:.4f} |"
            )
        lines.extend(
            [
                "",
                "| strategy | held-out expected MAL | delta vs native |",
                "|---|---:|---:|",
            ]
        )
        for row in section["uniform_strategies"][:12]:
            lines.append(
                f"| {row['strategy']} | {row['heldout_mal']:.4f} | {row['delta']:+.4f} |"
            )
        oracle = section["five_plus_one_oracle"]
        lines.extend(
            [
                "",
                f"5+1 optimistic oracle delta: **{oracle['upper_bound_delta']:+.4f} MAL**.",
                "",
            ]
        )

    selected = payload["selection"]
    lines.extend(
        [
            "## Selected static head-wise calibration",
            "",
            "```json",
            json.dumps(selected["config"], ensure_ascii=False, indent=2),
            "```",
            "",
            "| dataset | native held-out MAL | calibrated held-out MAL | delta | gate |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for dataset, row in selected["heldout"].items():
        lines.append(
            f"| {dataset} | {row['native_mal']:.4f} | {row['calibrated_mal']:.4f} | "
            f"{row['delta']:+.4f} | {'PASS' if row['pass'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            f"**Conclusion: {payload['conclusion']}**",
            "",
            "The offline estimate freezes the recorded conditional prefixes. A small online pilot is still required because changing an early proposal changes later MTP states.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(
    traces: dict[str, list[dict[str, Any]]],
    legacy: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    traces = {
        name: [row for row in rows if int(row.get("num_heads", 0)) == 6]
        for name, rows in traces.items()
    }
    if not all(traces.values()):
        raise ValueError("every dataset needs at least one six-head compact trace")
    train: dict[str, list[dict[str, Any]]] = {}
    heldout: dict[str, list[dict[str, Any]]] = {}
    for name, rows in traces.items():
        train[name], heldout[name] = _split_requests(rows)

    available_sets = [set(rows[0]["strategy_overlap"]) for rows in traces.values()]
    available = sorted(set.intersection(*available_sets))
    selected_names, train_score = coordinate_search(train, available, heads=6)
    transform_lookup = transforms_by_name()
    config = ProposalCalibrationConfig(
        tuple(transform_lookup[name] for name in selected_names)
    )

    datasets_payload: dict[str, Any] = {}
    heldout_payload: dict[str, Any] = {}
    for name, rows in traces.items():
        _, eval_rows = _split_requests(rows)
        native = _dataset_score(eval_rows, ["identity"] * 6)
        calibrated = _dataset_score(eval_rows, selected_names)
        threshold = 0.10 if name.lower() == "gsm8k" else 0.03
        uniform_rows = []
        for strategy in available:
            mal = _dataset_score(eval_rows, [strategy] * 6)
            uniform_rows.append(
                {"strategy": strategy, "heldout_mal": mal, "delta": mal - native}
            )
        uniform_rows.sort(key=lambda row: row["heldout_mal"], reverse=True)
        datasets_payload[name] = {
            "records": len(rows),
            "train_records": len(train[name]),
            "heldout_records": len(eval_rows),
            "head_statistics": analyze_head_statistics(rows, 6),
            "uniform_strategies": uniform_rows,
            "five_plus_one_oracle": five_plus_one_upper_bound(rows),
        }
        heldout_payload[name] = {
            "native_mal": native,
            "calibrated_mal": calibrated,
            "delta": calibrated - native,
            "required_delta": threshold,
            "pass": calibrated - native >= threshold,
        }

    pass_offline = all(row["pass"] for row in heldout_payload.values())
    tree_revisit = all(
        section["five_plus_one_oracle"]["upper_bound_delta"] >= 0.10
        for section in datasets_payload.values()
    )
    conclusion = (
        "GO to a small online pilot; do not run formal datasets yet"
        if pass_offline
        else "NO-GO for formal datasets; revise static Q calibration first"
    )
    return {
        "version": 1,
        "objective": "mean[1 + a1 + a1*a2 + ... + a1*...*a6]",
        "datasets": datasets_payload,
        "legacy": {
            name: analyze_legacy(rows) for name, rows in (legacy or {}).items()
        },
        "selection": {
            "train_joint_score": train_score,
            "head_strategies": selected_names,
            "config": config.to_dict(),
            "heldout": heldout_payload,
            "offline_gate_pass": pass_offline,
        },
        "strictness": {
            "target_distribution_modified": False,
            "proposal_sampling_uses_q_tilde": True,
            "acceptance_uses_q_tilde": True,
            "residual_uses_q_tilde": True,
        },
        "five_plus_one": {
            "revisit_gpu_tree": tree_revisit,
            "future_overhead_requirement": "<10%",
        },
        "conclusion": conclusion,
    }


def _named_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected DATASET=PATH, got: {value}")
        name, path = value.split("=", 1)
        output[name] = Path(path)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", default=[], help="DATASET=PATH")
    parser.add_argument("--legacy", action="append", default=[], help="DATASET=PATH")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    trace_paths = _named_paths(args.trace)
    legacy_paths = _named_paths(args.legacy)
    payload = analyze(
        {name: load_jsonl(path) for name, path in trace_paths.items()},
        {name: load_jsonl(path) for name, path in legacy_paths.items()},
    )
    for path in (args.output_json, args.output_md, args.output_config):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    args.output_config.write_text(
        json.dumps(payload["selection"]["config"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_md)
    print(args.output_config)
    print(payload["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
