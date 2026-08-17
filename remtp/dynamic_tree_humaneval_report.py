"""Combine HumanEval throughput with tree-native acceptance statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from remtp.mimo_tree_report import load_records, summarize


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def build_report(run_dir: Path, audit: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    result = dict(payload["results"][0])
    tree = summarize(load_records(audit))
    combined = {
        "method": "Dynamic MTP Tree + target-dominant relaxation",
        "samples": int(result["samples"]),
        "pass_at_1": result.get("pass_at_1"),
        "decode_tok_s": result.get("decode_tok_s"),
        "e2e_tok_s": result.get("e2e_output_tok_s"),
        "mean_acceptance_length": tree["mean_acceptance_length"],
        "mean_accepted_depth": tree["mean_accepted_depth"],
        "average_tree_nodes": tree["target_nodes_per_round"],
        "target_forward_calls_per_round": tree["target_forward_calls_per_round"],
        "mean_target_forward_ms": tree["mean_target_forward_ms"],
        "average_surviving_paths": tree.get("average_surviving_paths"),
        "useful_node_ratio": tree.get("useful_node_ratio"),
        "mean_target_candidate_coverage": tree.get("mean_target_candidate_coverage"),
        "tree": tree,
        "humaneval": result,
        "config": payload.get("config", {}),
    }
    return combined


def markdown(report: dict[str, Any]) -> str:
    quality = report.get("pass_at_1")
    quality_text = "pending" if quality is None else f"{100.0 * quality:.1f}%"
    lines = [
        "# HumanEval 50: Dynamic MTP Tree",
        "",
        "| method | samples | pass@1 | mean acceptance length | decode tok/s | e2e tok/s | avg tree nodes | target forwards/round |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {report['method']} | {report['samples']} | {quality_text} | "
        f"{_display(report['mean_acceptance_length'])} | "
        f"{_display(report['decode_tok_s'])} | "
        f"{_display(report['e2e_tok_s'])} | "
        f"{_display(report['average_tree_nodes'])} | "
        f"{_display(report['target_forward_calls_per_round'])} |",
        "",
        "## Tree diagnostics",
        "",
        f"- Mean accepted draft depth: {_display(report['mean_accepted_depth'])}",
        f"- Average surviving paths: {_display(report['average_surviving_paths'])}",
        f"- Useful-node ratio: {100.0 * float(report.get('useful_node_ratio') or 0.0):.1f}%",
        f"- Mean target candidate-set coverage: {_display(report['mean_target_candidate_coverage'])}",
        f"- Mean target forward: {_display(report['mean_target_forward_ms'])} ms",
        "",
        "`Mean acceptance length` is tree-native: selected draft-path length plus",
        "the unmodified target correction/bonus anchor committed in that round.",
        "It does not use vLLM's linear per-position acceptance display.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = build_report(args.run_dir, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(report), encoding="utf-8")
    if args.json_output:
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
