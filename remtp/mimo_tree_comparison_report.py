"""Combine protocol-locked MiMo chain baselines with a dynamic-tree run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_chain(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return dict(payload["results"][0])


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_report(
    native: dict[str, Any],
    cactus: dict[str, Any],
    dynamic: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        {"method": "Native probabilistic MTP", **native},
        {"method": "Cactus + MTP", **cactus},
        {
            "method": "Dynamic MTP Tree + target-dominant relaxation",
            "samples": dynamic.get("samples"),
            "pass_at_1": dynamic.get("pass_at_1"),
            "decode_tok_s": dynamic.get("decode_tok_s"),
            "e2e_output_tok_s": dynamic.get("e2e_tok_s"),
            "mean_acceptance_length": dynamic.get("mean_acceptance_length"),
            "draft_token_acceptance_rate": (
                dynamic.get("humaneval") or {}
            ).get("draft_token_acceptance_rate"),
            "truncation_rate": (dynamic.get("humaneval") or {}).get(
                "truncation_rate"
            ),
            "average_tree_nodes": dynamic.get("average_tree_nodes"),
        },
    ]
    baseline = rows[0]
    for row in rows:
        for key in ("mean_acceptance_length", "decode_tok_s", "e2e_output_tok_s"):
            lhs = row.get(key)
            rhs = baseline.get(key)
            row[f"delta_{key}_vs_native"] = (
                float(lhs) - float(rhs)
                if lhs is not None and rhs is not None
                else None
            )
    return {"rows": rows}


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# HumanEval 50: MiMo MTP versus dynamic tree",
        "",
        "All rows use MiMo-7B-Base-MTP3, MTP depth 3, the same sampled tasks, "
        "temperature 0.7, seed 42, max_tokens 512, eager execution and "
        "synchronous scheduling.",
        "",
        "| method | pass@1 | MAL | ΔMAL vs native | decode tok/s | Δdecode | e2e tok/s | Δe2e | draft acceptance | truncation | avg tree nodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        pass_at_1 = row.get("pass_at_1")
        accept = row.get("draft_token_acceptance_rate")
        truncation = row.get("truncation_rate")
        lines.append(
            f"| {row['method']} | "
            f"{'n/a' if pass_at_1 is None else f'{100 * float(pass_at_1):.1f}%'} | "
            f"{_fmt(row.get('mean_acceptance_length'))} | "
            f"{_fmt(row.get('delta_mean_acceptance_length_vs_native'))} | "
            f"{_fmt(row.get('decode_tok_s'))} | "
            f"{_fmt(row.get('delta_decode_tok_s_vs_native'))} | "
            f"{_fmt(row.get('e2e_output_tok_s'))} | "
            f"{_fmt(row.get('delta_e2e_output_tok_s_vs_native'))} | "
            f"{'n/a' if accept is None else f'{100 * float(accept):.1f}%'} | "
            f"{'n/a' if truncation is None else f'{100 * float(truncation):.1f}%'} | "
            f"{_fmt(row.get('average_tree_nodes'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--cactus", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    dynamic = json.loads(args.dynamic.read_text(encoding="utf-8"))
    report = build_report(_load_chain(args.native), _load_chain(args.cactus), dynamic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(report), encoding="utf-8")
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
