"""Combine protocol-locked GSM8K and HumanEval comparisons into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_FOCUS = ("remtp",)


def _load_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "comparison.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"comparison must be a non-empty list: {path}")
    rows = [dict(row) for row in payload]
    directories = [row.get("directory") for row in rows]
    if len(set(directories)) != len(directories):
        raise ValueError(f"duplicate comparison directory in {path}")
    return rows


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["directory"]): row for row in rows}


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _full_table(
    rows: list[dict[str, Any]], *, quality_key: str, quality_label: str
) -> list[str]:
    lines = [
        f"| method | {quality_label} | decode tok/s | e2e tok/s | MAL | "
        "draft acceptance | truncation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {_pct(float(row[quality_key]))} | "
            f"{float(row['decode_tok_s']):.3f} | "
            f"{float(row['e2e_tok_s']):.3f} | "
            f"{float(row['mean_acceptance_length']):.3f} | "
            f"{_pct(float(row['draft_acceptance']))} | "
            f"{_pct(float(row['truncation_rate']))} |"
        )
    return lines


def build_report(
    gsm_root: Path,
    humaneval_root: Path,
    *,
    focus: list[str] | tuple[str, ...] = DEFAULT_FOCUS,
) -> str:
    gsm_rows = _load_rows(gsm_root)
    humaneval_rows = _load_rows(humaneval_root)
    gsm = _index(gsm_rows)
    humaneval = _index(humaneval_rows)
    missing = [
        profile
        for profile in focus
        if profile not in gsm or profile not in humaneval
    ]
    if missing:
        raise ValueError(
            "focused profiles are missing from a comparison: "
            + ", ".join(missing)
        )

    lines = [
        "# GSM8K + HumanEval relaxed-MTP comparison",
        "",
        "Both tables use their protocol-locked datasets, prompts, sampling "
        "parameters and metrics. Historical rows are reused only after "
        "protocol fingerprint validation.",
        "",
        "## Main method across both datasets",
        "",
        "| method | GSM8K accuracy | GSM8K e2e | GSM8K MAL | "
        "HumanEval pass@1 | HumanEval e2e | HumanEval MAL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in focus:
        gsm_row = gsm[profile]
        human_row = humaneval[profile]
        lines.append(
            f"| {gsm_row['method']} | "
            f"{_pct(float(gsm_row['accuracy']))} | "
            f"{float(gsm_row['e2e_tok_s']):.3f} | "
            f"{float(gsm_row['mean_acceptance_length']):.3f} | "
            f"{_pct(float(human_row['pass_at_1']))} | "
            f"{float(human_row['e2e_tok_s']):.3f} | "
            f"{float(human_row['mean_acceptance_length']):.3f} |"
        )

    lines.extend(["", "## GSM8K: all methods", ""])
    lines.extend(_full_table(gsm_rows, quality_key="accuracy", quality_label="accuracy"))
    lines.extend(["", "## HumanEval: all methods", ""])
    lines.extend(
        _full_table(
            humaneval_rows,
            quality_key="pass_at_1",
            quality_label="pass@1",
        )
    )
    lines.extend(
        [
            "",
            "## Source comparisons",
            "",
            f"- GSM8K: `{(gsm_root / 'comparison.md').resolve()}`",
            f"- HumanEval: `{(humaneval_root / 'comparison.md').resolve()}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm-root", type=Path, required=True)
    parser.add_argument("--humaneval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus", nargs="+", default=list(DEFAULT_FOCUS))
    args = parser.parse_args()

    report = build_report(
        args.gsm_root,
        args.humaneval_root,
        focus=args.focus,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Combined report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
