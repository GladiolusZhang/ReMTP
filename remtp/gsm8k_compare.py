"""Combine three local GSM8K benchmark runs into one local summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


RUNS = (
    ("standard_mtp", "标准概率 MTP"),
    ("tokenv3", "SpecCascade [TokenV3]"),
    ("cactus", "Cactus + MTP"),
)


def _percent_delta(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0)


def load_comparison(run_root: Path) -> tuple[list[dict[str, Any]], str]:
    summaries: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    manifest_hashes: set[str] = set()

    for directory, method in RUNS:
        run_dir = run_root / directory
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "sample_manifest.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"incomplete run {run_dir}: summary or manifest missing"
            )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        if len(results) != 1:
            raise ValueError(f"{summary_path} must contain one GSM8K result")
        summaries.append((directory, method, payload["config"], results[0]))
        manifest_hashes.add(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        )

    if len(manifest_hashes) != 1:
        raise ValueError("sample manifests differ across the three methods")

    comparison_keys = (
        "samples",
        "sample_seed",
        "temperature",
        "generation_seed",
        "max_tokens",
        "mtp_tokens",
        "data_sha256",
        "answer_metric",
    )
    baseline_config = summaries[0][2]
    for _, method, config, _ in summaries[1:]:
        mismatched = [
            key
            for key in comparison_keys
            if config.get(key) != baseline_config.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"{method} config differs for: {', '.join(mismatched)}"
            )

    baseline = summaries[0][3]
    rows: list[dict[str, Any]] = []
    for directory, method, _, result in summaries:
        rows.append(
            {
                "run": directory,
                "method": method,
                "samples": result["samples"],
                "correct": result["correct"],
                "accuracy": result["accuracy"],
                "accuracy_delta_pp": 100.0
                * (result["accuracy"] - baseline["accuracy"]),
                "decode_tok_s": result["decode_tok_s"],
                "decode_delta_pct": _percent_delta(
                    result["decode_tok_s"],
                    baseline["decode_tok_s"],
                ),
                "e2e_tok_s": result["e2e_output_tok_s"],
                "e2e_delta_pct": _percent_delta(
                    result["e2e_output_tok_s"],
                    baseline["e2e_output_tok_s"],
                ),
                "mean_acceptance_length": result[
                    "mean_acceptance_length"
                ],
                "acceptance_length_delta_pct": _percent_delta(
                    result["mean_acceptance_length"],
                    baseline["mean_acceptance_length"],
                ),
                "draft_acceptance_rate": result[
                    "draft_token_acceptance_rate"
                ],
                "draft_acceptance_delta_pp": 100.0
                * (
                    result["draft_token_acceptance_rate"]
                    - baseline["draft_token_acceptance_rate"]
                ),
                "truncation_rate": result["truncation_rate"],
                "output_tokens": result["output_tokens"],
            }
        )
    return rows, manifest_hashes.pop()


def write_comparison(
    run_root: Path,
    rows: list[dict[str, Any]],
    manifest_sha256: str,
) -> None:
    (run_root / "comparison.json").write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha256,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
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
        "# GSM8K three-way local comparison",
        "",
        f"- samples per method: `{rows[0]['samples']}`",
        f"- shared manifest SHA-256: `{manifest_sha256}`",
        "",
        "| method | accuracy | delta | decode tok/s | delta | "
        "e2e tok/s | mean acceptance length | draft acceptance | truncation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | "
            f"{100.0 * row['accuracy']:.1f}% "
            f"({row['correct']}/{row['samples']}) | "
            f"{row['accuracy_delta_pp']:+.1f} pp | "
            f"{row['decode_tok_s']:.3f} | "
            f"{row['decode_delta_pct']:+.2f}% | "
            f"{row['e2e_tok_s']:.3f} | "
            f"{row['mean_acceptance_length']:.3f} | "
            f"{100.0 * row['draft_acceptance_rate']:.2f}% | "
            f"{100.0 * row['truncation_rate']:.1f}% |"
        )
    (run_root / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine standard MTP, TokenV3, and Cactus GSM8K runs."
    )
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    rows, manifest_sha256 = load_comparison(run_root)
    write_comparison(run_root, rows, manifest_sha256)

    print("\nGSM8K three-way comparison")
    print(
        "method                         accuracy   decode tok/s   "
        "mean acceptance"
    )
    for row in rows:
        print(
            f"{row['method']:<30} "
            f"{100.0 * row['accuracy']:>6.1f}%   "
            f"{row['decode_tok_s']:>10.3f}   "
            f"{row['mean_acceptance_length']:>10.3f}"
        )
    print(f"\nLocal comparison: {run_root / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
