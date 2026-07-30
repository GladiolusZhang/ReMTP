"""Compare local GSM8K outputs from the block-feature ablation runner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHODS = (
    ("cactus", "Cactus + MTP"),
    ("token_only", "Token support only"),
    ("distribution", "Token + current target-led JS"),
    ("full", "Token + current/future target-led JS"),
)


def load_summary(run_root: Path, directory: str) -> dict[str, Any]:
    path = run_root / directory / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one result in {path}")
    return dict(results[0])


def comparison_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory, label in METHODS:
        summary = load_summary(run_root, directory)
        rows.append(
            {
                "method": label,
                "samples": summary["samples"],
                "accuracy": summary["accuracy"],
                "decode_tok_s": summary["decode_tok_s"],
                "e2e_output_tok_s": summary["e2e_output_tok_s"],
                "mean_acceptance_length": summary[
                    "mean_acceptance_length"
                ],
                "draft_token_acceptance_rate": summary[
                    "draft_token_acceptance_rate"
                ],
                "truncation_rate": summary["truncation_rate"],
            }
        )
    return rows


def write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
    (run_root / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (run_root / "comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# GSM8K block-feature ablation",
        "",
        "| method | accuracy | decode tok/s | e2e tok/s | "
        "mean acceptance length | draft acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {100.0 * row['accuracy']:.1f}% | "
            f"{row['decode_tok_s']:.3f} | "
            f"{row['e2e_output_tok_s']:.3f} | "
            f"{row['mean_acceptance_length']:.3f} | "
            f"{100.0 * row['draft_token_acceptance_rate']:.1f}% |"
        )
    (run_root / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    rows = comparison_rows(run_root)
    write_outputs(run_root, rows)
    print((run_root / "comparison.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
