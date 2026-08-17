"""Summarize MiMo tree JSONL into tree-native acceptance metrics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("tree audit is empty")
    rounds = len(records)
    accepted = sum(int(row.get("accepted_drafts", 0)) for row in records)
    target_calls = sum(int(row.get("target_forward_calls", 0)) for row in records)
    target_nodes = sum(int(row.get("target_validation_nodes", 0)) for row in records)
    forward_times = [
        float(row.get("target_forward_seconds", 0.0))
        for row in records
        if float(row.get("target_forward_seconds", 0.0)) > 0
    ]
    branches: dict[int, list[int]] = defaultdict(list)
    terminals = Counter(str(row.get("terminal", "unknown")) for row in records)
    depth_evaluated: Counter[int] = Counter()
    depth_accepted: Counter[int] = Counter()
    depth_rejected: Counter[int] = Counter()
    depth_survived: Counter[int] = Counter()
    branching_total: Counter[int] = Counter()
    branching_parents: Counter[int] = Counter()
    coverage_values: list[float] = []
    surviving_paths_total = 0
    selected_node_total = 0
    output_token_total = 0
    rescue_attempts = 0
    rescue_accepts = 0
    selected_rescue_tokens = 0
    selected_rescue_rounds = 0
    rescue_extension_tokens = 0
    rescue_unlocked_tokens = 0
    selected_rescue_target_margins: list[float] = []
    selected_rescue_margin_multipliers: list[float] = []
    for row in records:
        branch = row.get("selected_branch")
        if branch is not None:
            branches[int(branch)].append(int(row.get("accepted_drafts", 0)))
        surviving_paths_total += int(row.get("surviving_paths", 0))
        selected_node_total += int(row.get("accepted_drafts", 0))
        output_token_total += len(row.get("output_token_ids", []))
        round_selected_rescues = 0
        seen_parent_coverage: set[tuple[int, int | None]] = set()
        for node in row.get("nodes", []):
            status = str(node.get("status"))
            depth = int(node.get("depth", 0))
            if status in {"ACCEPT", "REJECT", "SURVIVE", "SELECTED", "PRUNE"}:
                depth_evaluated[depth] += 1
            if status in {"ACCEPT", "SELECTED"}:
                depth_accepted[depth] += 1
            elif status in {"REJECT", "PRUNE"}:
                depth_rejected[depth] += 1
            if status in {"SURVIVE", "SELECTED"}:
                depth_survived[depth] += 1
            if node.get("rescue_eligible"):
                rescue_attempts += 1
            if node.get("rescued"):
                rescue_accepts += 1
                if status == "SELECTED":
                    selected_rescue_tokens += 1
                    round_selected_rescues += 1
                    if node.get("target_log_margin") is not None:
                        selected_rescue_target_margins.append(
                            float(node["target_log_margin"])
                        )
                    if node.get("rescue_margin_multiplier") is not None:
                        selected_rescue_margin_multipliers.append(
                            float(node["rescue_margin_multiplier"])
                        )
                    rescue_extension_tokens += int(
                        node.get("rescue_extension_tokens", 0)
                    )
                    rescue_unlocked_tokens += int(
                        node.get(
                            "rescue_unlocked_tokens",
                            node.get("rescue_extension_tokens", 0),
                        )
                    )
            if "coverage" in node:
                key = (int(row.get("round", 0)), node.get("parent"))
                if key not in seen_parent_coverage:
                    seen_parent_coverage.add(key)
                    coverage_values.append(float(node["coverage"]))
        construction = row.get("dynamic_construction") or {}
        for group in construction.get("branching", []):
            depth = int(group.get("depth", 0))
            branching_total[depth] += int(group.get("retained", 0))
            branching_parents[depth] += 1
        if round_selected_rescues:
            selected_rescue_rounds += 1

    return {
        "topology": records[-1].get("topology", "unknown"),
        "verification_mode": records[-1].get("verification_mode", "unknown"),
        "rounds": rounds,
        "accepted_draft_tokens": accepted,
        "mean_accepted_depth": accepted / rounds,
        "mean_acceptance_length": (
            output_token_total / rounds if output_token_total else 1.0 + accepted / rounds
        ),
        "root_selection_rate": sum(len(values) for values in branches.values()) / rounds,
        "target_forward_calls_per_round": target_calls / rounds,
        "target_nodes_per_round": target_nodes / rounds,
        "mean_target_forward_ms": (
            1000.0 * sum(forward_times) / len(forward_times) if forward_times else None
        ),
        "average_surviving_paths": surviving_paths_total / rounds,
        "frontier_rescue_attempts": rescue_attempts,
        "frontier_rescue_accepts": rescue_accepts,
        "selected_rescue_tokens": selected_rescue_tokens,
        "selected_rescue_round_rate": selected_rescue_rounds / rounds,
        "ordinary_accepted_draft_tokens": accepted - rescue_extension_tokens,
        "ordinary_mean_accepted_depth": (
            accepted - rescue_extension_tokens
        ) / rounds,
        "rescue_extension_tokens": rescue_extension_tokens,
        "rescue_extension_mal_gain": rescue_extension_tokens / rounds,
        "rescue_unlocked_tokens": rescue_unlocked_tokens,
        "rescue_unlocked_mal_gain": rescue_unlocked_tokens / rounds,
        "pre_rescue_mean_accepted_depth": (
            accepted - rescue_unlocked_tokens
        ) / rounds,
        "mean_selected_rescue_target_log_margin": (
            sum(selected_rescue_target_margins)
            / len(selected_rescue_target_margins)
            if selected_rescue_target_margins
            else None
        ),
        "mean_selected_rescue_margin_multiplier": (
            sum(selected_rescue_margin_multipliers)
            / len(selected_rescue_margin_multipliers)
            if selected_rescue_margin_multipliers
            else None
        ),
        "useful_node_ratio": selected_node_total / target_nodes if target_nodes else 0.0,
        "mean_target_candidate_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else None
        ),
        "average_branching_factor_by_depth": {
            str(depth): branching_total[depth] / branching_parents[depth]
            for depth in sorted(branching_parents)
            if branching_parents[depth]
        },
        "terminals": dict(sorted(terminals.items())),
        "branches": {
            str(branch): {
                "selected_rounds": len(depths),
                "selection_rate": len(depths) / rounds,
                "mean_accepted_depth_when_selected": sum(depths) / len(depths),
            }
            for branch, depths in sorted(branches.items())
        },
        "depths": {
            str(depth): {
                "evaluated": depth_evaluated[depth],
                "accepted": depth_accepted[depth],
                "rejected": depth_rejected[depth],
                "survived": depth_survived[depth],
                "acceptance_rate_when_evaluated": (
                    depth_accepted[depth] / depth_evaluated[depth]
                    if depth_evaluated[depth]
                    else None
                ),
            }
            for depth in sorted(depth_evaluated)
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    forward = summary["mean_target_forward_ms"]
    lines = [
        "# MiMo tree acceptance report",
        "",
        f"- Topology: `{summary['topology']}`",
        f"- Verification: `{summary['verification_mode']}`",
        f"- Rounds: {summary['rounds']}",
        f"- Mean accepted depth: **{summary['mean_accepted_depth']:.3f}** draft tokens/round",
        f"- Tree mean acceptance length: **{summary['mean_acceptance_length']:.3f}** tokens/round",
        f"- Root selection rate: {summary['root_selection_rate']:.1%}",
        f"- Target forward calls/round: {summary['target_forward_calls_per_round']:.3f}",
        f"- Target validation nodes/round: {summary['target_nodes_per_round']:.3f}",
        f"- Selected frontier-rescue rounds: {summary['selected_rescue_round_rate']:.1%}",
        f"- Frontier rescue attempts/accepts/selected tokens: "
        f"{summary['frontier_rescue_attempts']}/"
        f"{summary['frontier_rescue_accepts']}/"
        f"{summary['selected_rescue_tokens']}",
        "- Estimated accepted depth before selected rescue effects: "
        f"{summary['pre_rescue_mean_accepted_depth']:.3f}",
        "- Legacy one-token rescue extensions / direct MAL contribution: "
        f"{summary['rescue_extension_tokens']} / "
        f"+{summary['rescue_extension_mal_gain']:.3f}",
        "- Rescue-unlocked selected tokens / MAL contribution: "
        f"{summary['rescue_unlocked_tokens']} / "
        f"+{summary['rescue_unlocked_mal_gain']:.3f}",
    ]
    if forward is not None:
        lines.append(f"- Mean profiled target forward: {forward:.3f} ms")
    if summary.get("mean_selected_rescue_target_log_margin") is not None:
        lines.append(
            "- Selected-rescue target log-margin / threshold multiplier: "
            f"{summary['mean_selected_rescue_target_log_margin']:.3f} / "
            f"{summary['mean_selected_rescue_margin_multiplier']:.3f}x"
        )
    if summary.get("mean_target_candidate_coverage") is not None:
        lines.extend(
            [
                f"- Average surviving paths: {summary['average_surviving_paths']:.3f}",
                f"- Useful-node ratio: {summary['useful_node_ratio']:.1%}",
                "- Mean target candidate-set coverage: "
                f"{summary['mean_target_candidate_coverage']:.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "`Mean accepted depth` counts only accepted draft nodes. Tree MAL adds the",
            "one correction/bonus token that terminates every speculative round.",
            "",
            "## Per-depth verification",
            "",
            "| depth | evaluated | selected/accepted | survived incl. unselected | pruned/rejected | selected/evaluated |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for depth, row in summary["depths"].items():
        rate = row["acceptance_rate_when_evaluated"]
        rate_text = "n/a" if rate is None else f"{rate:.1%}"
        lines.append(
            f"| {depth} | {row['evaluated']} | {row['accepted']} | "
            f"{row.get('survived', 0)} | {row['rejected']} | {rate_text} |"
        )
    if summary.get("average_branching_factor_by_depth"):
        lines.extend(["", "## Dynamic branching", "", "| depth | mean retained children |", "|---:|---:|"])
        for depth, value in summary["average_branching_factor_by_depth"].items():
            lines.append(f"| {depth} | {value:.3f} |")
    lines.extend(
        [
            "",
            "## Branch selection",
            "",
            "| branch | selected rounds | selection rate | mean accepted depth when selected |",
            "|---:|---:|---:|---:|",
        ]
    )
    for branch, row in summary["branches"].items():
        lines.append(
            f"| {branch} | {row['selected_rounds']} | {row['selection_rate']:.1%} | "
            f"{row['mean_accepted_depth_when_selected']:.3f} |"
        )
    lines.extend(["", "## Terminal events", ""])
    for terminal, count in summary["terminals"].items():
        lines.append(f"- `{terminal}`: {count}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = summarize(load_records(args.audit))
    text = markdown(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote: {args.output}")
    else:
        print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
