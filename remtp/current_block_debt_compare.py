"""Compare relaxed-MTP experiments against the Cactus reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


_PROTOCOL_KEYS = (
    "samples",
    "sample_seed",
    "temperature",
    "generation_seed",
    "max_tokens",
    "mtp_tokens",
    "data_sha256",
    "answer_metric",
)


def _load_run(directory: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = directory / "summary.json"
    manifest_path = directory / "sample_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing sample manifest: {manifest_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one result in {path}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"missing benchmark config in {path}")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return dict(config), dict(results[0]), manifest_hash


def _profile_label(directory: str) -> str:
    labels = {
        "native_mtp": "Native probabilistic MTP",
        "cactus": "Cactus + MTP",
        "debt_conservative": "Current-block debt (conservative)",
        "debt_balanced": "Current-block debt (balanced)",
        "debt_cactus_fallback": "Current-block debt (Cactus fallback)",
        "debt_scale": "Current-block debt + posterior scale",
        "debt_top1": "Current-block debt + nonnegative top-1 bias",
        "top1_surplus": "Cactus-dominant top-1 surplus",
        "target_surplus": "Cactus-dominant target surplus",
        "risk_swap": "Target-anchored Cactus risk swap",
        "target_recovery": "Target-surplus + target recovery",
        "native_block": "Native MTP + Block Verification",
        "cactus_block": "Cactus + Block Verification",
        "risk_swap_block": "Target-anchored risk swap + Block Verification",
        "block_shield": "Block-surplus-shielded risk control",
        "sparse_checkpoint": "Sparse target-risk checkpoint + Block Verification",
    }
    return labels.get(directory, directory)


def compare(run_root: Path, profiles: list[str]) -> list[dict[str, Any]]:
    baseline_config, baseline, baseline_manifest = _load_run(
        run_root / "cactus"
    )
    rows: list[dict[str, Any]] = []
    for directory in ["cactus", *profiles]:
        config, summary, manifest = _load_run(run_root / directory)
        if manifest != baseline_manifest:
            raise ValueError(
                f"sample manifest differs between cactus and {directory}"
            )
        mismatched = [
            key
            for key in _PROTOCOL_KEYS
            if config.get(key) != baseline_config.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"benchmark protocol differs for {directory}: "
                + ", ".join(mismatched)
            )
        accuracy_delta = summary["accuracy"] - baseline["accuracy"]
        mal_delta = (
            summary["mean_acceptance_length"]
            - baseline["mean_acceptance_length"]
        )
        e2e_delta = (
            summary["e2e_output_tok_s"]
            - baseline["e2e_output_tok_s"]
        )
        pareto_pass = (
            directory not in {"cactus", "native_mtp"}
            and accuracy_delta > 0.0
            and mal_delta > 0.0
            and e2e_delta >= 0.0
        )
        rows.append(
            {
                "directory": directory,
                "manifest_sha256": manifest,
                "method": _profile_label(directory),
                "samples": summary["samples"],
                "accuracy": summary["accuracy"],
                "accuracy_delta_pp": 100.0 * accuracy_delta,
                "decode_tok_s": summary["decode_tok_s"],
                "e2e_tok_s": summary["e2e_output_tok_s"],
                "e2e_delta_pct": (
                    100.0
                    * e2e_delta
                    / max(baseline["e2e_output_tok_s"], 1e-30)
                ),
                "mean_acceptance_length": summary[
                    "mean_acceptance_length"
                ],
                "mal_delta": mal_delta,
                "draft_acceptance": summary[
                    "draft_token_acceptance_rate"
                ],
                "truncation_rate": summary["truncation_rate"],
                "pareto_pass": pareto_pass,
            }
        )
    return rows


def _write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
    (run_root / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (run_root / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Relaxed-MTP Pareto gate",
        "",
        "Pass condition: Accuracy > Cactus, MAL > Cactus, and E2E >= Cactus.",
        "",
        "| method | accuracy | delta | e2e tok/s | delta | MAL | delta | "
        "draft acceptance | truncation | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {100.0*row['accuracy']:.1f}% | "
            f"{row['accuracy_delta_pp']:+.1f} pp | "
            f"{row['e2e_tok_s']:.3f} | {row['e2e_delta_pct']:+.2f}% | "
            f"{row['mean_acceptance_length']:.3f} | "
            f"{row['mal_delta']:+.3f} | "
            f"{100.0*row['draft_acceptance']:.1f}% | "
            f"{100.0*row['truncation_rate']:.1f}% | "
            f"{'YES' if row['pareto_pass'] else 'NO'} |"
        )
    (run_root / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["pareto_pass"]]
    winner = None
    if feasible:
        winner = max(
            feasible,
            key=lambda row: (
                row["accuracy"],
                row["mean_acceptance_length"],
                row["e2e_tok_s"],
            ),
        )
    return {
        "pareto_pass": winner is not None,
        "winner": None if winner is None else winner["directory"],
        "selection_order": ["accuracy", "mean_acceptance_length", "e2e_tok_s"],
        "criterion": {
            "accuracy": "strictly greater than Cactus",
            "mean_acceptance_length": "strictly greater than Cactus",
            "e2e_tok_s": "greater than or equal to Cactus",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("profiles", nargs="+")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    rows = compare(run_root, args.profiles)
    _write_outputs(run_root, rows)
    selection = _select(rows)
    (run_root / "gate_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print((run_root / "comparison.md").read_text(encoding="utf-8"))
    print("GATE_SELECTION=" + json.dumps(selection, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
