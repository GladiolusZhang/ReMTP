"""Combine one FastMTP D=3 run and one D=4 run into a single report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINES = ("native", "cactus", "spec_cascade")


def _load(root: Path) -> dict[str, list[dict[str, Any]]]:
    path = root / "comparison.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"gsm8k", "humaneval"}:
        raise ValueError(f"unexpected comparison datasets: {path}")
    return payload


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["method_id"]): dict(row) for row in rows}


def _check_same_baseline(
    dataset: str,
    method: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> None:
    keys = (
        "samples",
        "quality",
        "decode_tok_s",
        "e2e_tok_s",
        "truncation_rate",
        "mal",
        "draft_acceptance",
        "nodes",
    )
    mismatch = [key for key in keys if left.get(key) != right.get(key)]
    if mismatch:
        raise ValueError(
            f"D=3/D=4 baseline mismatch for {dataset}/{method}: "
            + ", ".join(mismatch)
        )


def build_report(
    d3_root: Path,
    d4_root: Path,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    d3 = _load(d3_root)
    d4 = _load(d4_root)
    combined: dict[str, list[dict[str, Any]]] = {}
    markdown = [
        "# FastMTP relaxed-tree depth comparison",
        "",
        "D=3 and D=4 use the same node cap, proposal guards, verifier, tasks,",
        "sampling parameters and target-forward protocol. Baselines are generated",
        "once and reused by the D=4 run.",
    ]
    for dataset in ("gsm8k", "humaneval"):
        left = _by_id(d3[dataset])
        right = _by_id(d4[dataset])
        if "dynamic_tree" not in left or "dynamic_tree" not in right:
            raise ValueError(f"dynamic_tree is missing for {dataset}")
        rows: list[dict[str, Any]] = []
        for method in BASELINES:
            if method not in left or method not in right:
                raise ValueError(f"baseline {method} is missing for {dataset}")
            _check_same_baseline(dataset, method, left[method], right[method])
            rows.append(dict(left[method]))
        d3_tree = dict(left["dynamic_tree"])
        d3_tree.update(method_id="dynamic_tree_d3", method="Relaxed tree D=3")
        d4_tree = dict(right["dynamic_tree"])
        d4_tree.update(method_id="dynamic_tree_d4", method="Relaxed tree D=4")
        rows.extend((d3_tree, d4_tree))
        combined[dataset] = rows

        quality_label = "accuracy" if dataset == "gsm8k" else "pass@1"
        native_quality = float(left["native"]["quality"])
        markdown.extend(
            [
                "",
                f"## {'GSM8K' if dataset == 'gsm8k' else 'HumanEval'} "
                f"({int(rows[0]['samples'])} tasks)",
                "",
                f"| method | {quality_label} | vs Native | MAL | nodes/round | "
                "accepted/nodes | rescue rounds | truncation | e2e tok/s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            quality = float(row["quality"])
            rescue = row.get("rescue_rate")
            markdown.append(
                "| {method} | {quality:.1f}% | {delta:+.1f} pp | {mal:.3f} | "
                "{nodes:.3f} | {accepted:.1f}% | {rescue} | {trunc:.1f}% | "
                "{e2e:.3f} |".format(
                    method=row["method"],
                    quality=100 * quality,
                    delta=100 * (quality - native_quality),
                    mal=float(row["mal"]),
                    nodes=float(row["nodes"]),
                    accepted=100 * float(row["draft_acceptance"]),
                    rescue=("-" if rescue is None else f"{100 * float(rescue):.1f}%"),
                    trunc=100 * float(row["truncation_rate"]),
                    e2e=float(row["e2e_tok_s"]),
                )
            )
    markdown.extend(
        [
            "",
            "## Configuration semantics",
            "",
            "- The tree node cap is a maximum, not a fill target.",
            "- Guarded top-2/top-3 backups must pass both absolute-Q and relative-Q floors.",
            "- Both tree depths use one target forward per speculative round.",
            "- D=4 recursively reuses FastMTP's one trained physical MTP layer one extra time.",
            "- Tree verification remains approximate and must be judged by quality and MAL together.",
            "",
        ]
    )
    return "\n".join(markdown), combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3-root", type=Path, required=True)
    parser.add_argument("--d4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    markdown, payload = build_report(args.d3_root, args.d4_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "depth34_comparison.md").write_text(
        markdown, encoding="utf-8"
    )
    (args.output_dir / "depth34_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "depth34_comparison.md")


if __name__ == "__main__":
    main()
