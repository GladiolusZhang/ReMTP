"""Aggregate the formal MTP-depth comparison into one local report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "native_mtp": "Native probabilistic MTP",
    "cactus": "Cactus + MTP",
    "spec_cascade": "SpecCascade TokenV3 + MTP",
    "proposal_calibrated": "Proposal-Calibrated MTP (ours)",
}


def _load_summary(directory: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    path = directory / "summary.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    config = payload.get("config")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(config, dict):
        raise ValueError(f"invalid summary: {path}")
    return dict(config), dict(results[0])


def _manifest_hash(directory: Path) -> str | None:
    path = directory / "sample_manifest.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _check_protocol(
    entries: list[tuple[str, int, str, dict[str, Any], dict[str, Any], Path]],
) -> None:
    keys = {
        "gsm8k": (
            "data_sha256",
            "samples",
            "sample_seed",
            "temperature",
            "generation_seed",
            "max_tokens",
        ),
        "humaneval": (
            "data_sha256",
            "samples",
            "sample_seed",
            "temperature",
            "generation_seed",
            "max_tokens",
            "answer_metric",
        ),
    }
    for dataset in keys:
        subset = [entry for entry in entries if entry[0] == dataset]
        if not subset:
            continue
        reference = subset[0]
        for entry in subset[1:]:
            mismatched = [
                key for key in keys[dataset] if entry[3].get(key) != reference[3].get(key)
            ]
            if mismatched:
                raise ValueError(
                    f"{dataset} protocol mismatch for MTP={entry[1]} {entry[2]}: "
                    + ", ".join(mismatched)
                )
        manifests = {_manifest_hash(entry[5]) for entry in subset}
        manifests.discard(None)
        if len(manifests) > 1:
            raise ValueError(f"{dataset} sample manifests differ across runs")


def collect(
    root: Path,
    depths: list[int],
    methods: list[str],
    datasets: list[str] | None = None,
) -> list[dict[str, Any]]:
    datasets = datasets or ["gsm8k", "humaneval"]
    entries: list[tuple[str, int, str, dict[str, Any], dict[str, Any], Path]] = []
    for depth in depths:
        for method in methods:
            for dataset in datasets:
                directory = root / f"mtp_{depth}" / method / dataset
                loaded = _load_summary(directory)
                if loaded is None:
                    continue
                config, summary = loaded
                if int(config.get("mtp_tokens", -1)) != depth:
                    raise ValueError(f"wrong mtp_tokens in {directory}")
                if dataset == "humaneval" and summary.get("evaluation_status") != "complete":
                    continue
                entries.append((dataset, depth, method, config, summary, directory))
    _check_protocol(entries)

    rows: list[dict[str, Any]] = []
    for dataset, depth, method, config, summary, directory in entries:
        quality = summary.get("accuracy")
        if dataset == "humaneval":
            quality = summary.get("pass_at_1")
        rows.append(
            {
                "dataset": dataset,
                "mtp_tokens": depth,
                "method_id": method,
                "method": METHOD_LABELS.get(method, method),
                "samples": int(summary["samples"]),
                "quality": float(quality) if quality is not None else None,
                "decode_tok_s": float(summary["decode_tok_s"]),
                "e2e_tok_s": float(summary["e2e_output_tok_s"]),
                "mean_acceptance_length": float(summary["mean_acceptance_length"]),
                "draft_acceptance": float(summary["draft_token_acceptance_rate"]),
                "truncation_rate": float(summary.get("truncation_rate", 0.0)),
                "timeouts": int(summary.get("evaluation_counts", {}).get("timeout", 0)),
                "result_dir": str(directory),
            }
        )
    return rows


def _with_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["mtp_tokens"]), {})[
            row["method_id"]
        ] = row
    output = []
    for row in rows:
        baseline = grouped[(row["dataset"], row["mtp_tokens"])].get("native_mtp")
        cactus = grouped[(row["dataset"], row["mtp_tokens"])].get("cactus")
        enriched = dict(row)
        enriched["mal_delta_vs_native"] = (
            row["mean_acceptance_length"] - baseline["mean_acceptance_length"]
            if baseline
            else None
        )
        enriched["e2e_delta_vs_native_pct"] = (
            100.0 * (row["e2e_tok_s"] / baseline["e2e_tok_s"] - 1.0)
            if baseline and baseline["e2e_tok_s"] > 0
            else None
        )
        enriched["quality_delta_vs_cactus_pp"] = (
            100.0 * (row["quality"] - cactus["quality"])
            if cactus and row["quality"] is not None and cactus["quality"] is not None
            else None
        )
        output.append(enriched)
    return output


def _number(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _percent(value: float | None, digits: int = 2, suffix: str = "%") -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}{suffix}"


def write_outputs(
    root: Path,
    rows: list[dict[str, Any]],
    depths: list[int],
    methods: list[str],
    datasets: list[str] | None = None,
) -> None:
    datasets = datasets or ["gsm8k", "humaneval"]
    rows = _with_deltas(rows)
    (root / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    completed = {(row["dataset"], row["mtp_tokens"], row["method_id"]) for row in rows}
    lines = [
        "# Formal Proposal-Calibrated MTP depth comparison",
        "",
        "Methods: Native probabilistic MTP, Cactus, SpecCascade TokenV3, and Proposal-Calibrated MTP.",
        "All methods within a dataset use identical prompts, sample subset, temperature, generation seed, and output limit.",
        "HumanEval quality is isolated official-test `pass@1`; evaluation time is excluded from generation throughput.",
        "",
    ]
    quality_labels = {"gsm8k": "accuracy", "humaneval": "pass@1"}
    for dataset in datasets:
        quality_label = quality_labels[dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"| MTP | method | {quality_label} | decode tok/s | e2e tok/s | MAL | "
                "draft acceptance | MAL vs Native | E2E vs Native | quality vs Cactus | truncation |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        lookup = {(row["mtp_tokens"], row["method_id"]): row for row in dataset_rows}
        for depth in depths:
            for method in methods:
                row = lookup.get((depth, method))
                if row is None:
                    lines.append(
                        f"| {depth} | {METHOD_LABELS.get(method, method)} | PENDING | — | — | — | — | — | — | — | — |"
                    )
                    continue
                lines.append(
                    f"| {depth} | {row['method']} | {100*row['quality']:.1f}% | "
                    f"{row['decode_tok_s']:.3f} | {row['e2e_tok_s']:.3f} | "
                    f"{row['mean_acceptance_length']:.3f} | {100*row['draft_acceptance']:.1f}% | "
                    f"{_number(row['mal_delta_vs_native'], signed=True)} | "
                    f"{_percent(row['e2e_delta_vs_native_pct'])} | "
                    f"{_percent(row['quality_delta_vs_cactus_pp'], 1, ' pp')} | "
                    f"{100*row['truncation_rate']:.1f}% |"
                )
        lines.append("")

    expected = len(datasets) * len(depths) * len(methods)
    lines.extend(
        [
            "## Completion",
            "",
            f"Completed rows: **{len(completed)}/{expected}**.",
            "",
            "The report is regenerated after every method/depth run, so interrupted experiments leave a readable partial table.",
        ]
    )
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--depths", nargs="+", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=("gsm8k", "humaneval"), default=("gsm8k", "humaneval")
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    rows = collect(args.root, args.depths, args.methods, list(args.datasets))
    write_outputs(args.root, rows, args.depths, args.methods, list(args.datasets))
    print((args.root / "comparison.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
