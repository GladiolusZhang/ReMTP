"""Build a protocol-checked MiMo quality/MAL comparison for two benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


METHODS = (
    ("native", "Native probabilistic MTP"),
    ("cactus", "Cactus + MTP"),
    ("spec_cascade", "SpecCascade TokenV3 + MTP"),
    ("dynamic_tree", "Dynamic MTP Tree + target relaxation"),
)

PROTOCOL_KEYS = (
    "samples",
    "sample_seed",
    "temperature",
    "generation_seed",
    "max_tokens",
    "mtp_tokens",
    "data_sha256",
    "model",
)


def _wilson_interval(correct: int, samples: int) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    z = 1.959963984540054
    rate = correct / samples
    denominator = 1.0 + z * z / samples
    center = (rate + z * z / (2.0 * samples)) / denominator
    half = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / samples
            + z * z / (4.0 * samples * samples)
        )
        / denominator
    )
    return center - half, center + half


def _read_summary(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = directory / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one result in {path}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"missing config in {path}")
    return dict(config), dict(results[0])


def _manifest_hash(directory: Path) -> str:
    return hashlib.sha256(
        (directory / "sample_manifest.json").read_bytes()
    ).hexdigest()


def _load_dataset(root: Path, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_config: dict[str, Any] | None = None
    baseline_manifest: str | None = None
    baseline_quality: float | None = None
    baseline_mal: float | None = None

    for directory_name, label in METHODS:
        directory = root / dataset / directory_name
        config, result = _read_summary(directory)
        manifest = _manifest_hash(directory)
        if dataset == "humaneval":
            if result.get("evaluation_status") != "complete":
                raise ValueError(f"HumanEval evaluation is incomplete: {directory}")
            quality = result.get("pass_at_1")
        else:
            quality = result.get("accuracy")
        if quality is None:
            raise ValueError(f"quality is missing from {directory}")

        if baseline_config is None:
            baseline_config = config
            baseline_manifest = manifest
        else:
            assert baseline_manifest is not None
            if manifest != baseline_manifest:
                raise ValueError(f"sample manifest differs for {dataset}/{directory_name}")
            mismatched = [
                key
                for key in PROTOCOL_KEYS
                if config.get(key) != baseline_config.get(key)
            ]
            if mismatched:
                raise ValueError(
                    f"protocol differs for {dataset}/{directory_name}: "
                    + ", ".join(mismatched)
                )

        mal = float(result["mean_acceptance_length"])
        accepted_per_round = float(result["accepted_draft_tokens_per_round"])
        draft_rounds = float(result["draft_rounds"])
        tree_nodes: float | None = (
            float(result["draft_tokens"]) / draft_rounds
            if draft_rounds > 0
            else None
        )
        if directory_name == "dynamic_tree":
            tree_path = directory / "tree_metrics.json"
            tree_payload = json.loads(tree_path.read_text(encoding="utf-8"))
            tree = tree_payload.get("tree")
            if not isinstance(tree, dict):
                raise ValueError(f"missing tree metrics in {tree_path}")
            mal = float(tree["mean_acceptance_length"])
            accepted_per_round = float(tree["mean_accepted_depth"])
            tree_nodes = float(tree["target_nodes_per_round"])

        correct = int(result["correct"])
        samples = int(result["samples"])
        ci_low, ci_high = _wilson_interval(correct, samples)
        row = {
            "directory": directory_name,
            "method": label,
            "samples": samples,
            "correct": correct,
            "quality": float(quality),
            "quality_ci_low": ci_low,
            "quality_ci_high": ci_high,
            "mean_acceptance_length": mal,
            "accepted_draft_tokens_per_round": accepted_per_round,
            "average_tree_nodes": tree_nodes,
            "manifest_sha256": manifest,
        }
        if baseline_quality is None:
            baseline_quality = float(quality)
            baseline_mal = mal
        assert baseline_mal is not None
        row["quality_delta_pp_vs_native"] = 100.0 * (
            float(quality) - baseline_quality
        )
        row["mal_delta_vs_native"] = mal - baseline_mal
        rows.append(row)
    return rows


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _dataset_table(rows: list[dict[str, Any]], quality_label: str) -> list[str]:
    lines = [
        f"| method | {quality_label} | 95% CI | vs Native | MAL | "
        "vs Native | accepted drafts/round | avg candidate nodes/round |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        nodes = (
            "-"
            if row["average_tree_nodes"] is None
            else f"{row['average_tree_nodes']:.3f}"
        )
        lines.append(
            f"| {row['method']} | {_percent(row['quality'])} "
            f"({row['correct']}/{row['samples']}) | "
            f"[{_percent(row['quality_ci_low'])}, "
            f"{_percent(row['quality_ci_high'])}] | "
            f"{row['quality_delta_pp_vs_native']:+.1f} pp | "
            f"{row['mean_acceptance_length']:.3f} | "
            f"{row['mal_delta_vs_native']:+.3f} | "
            f"{row['accepted_draft_tokens_per_round']:.3f} | {nodes} |"
        )
    return lines


def build_report(root: Path) -> tuple[str, dict[str, Any]]:
    gsm8k = _load_dataset(root, "gsm8k")
    humaneval = _load_dataset(root, "humaneval")
    lines = [
        "# MiMo MTP algorithm-correctness comparison",
        "",
        "The primary metrics are task quality and tree/chain-native mean "
        "acceptance length (MAL). Throughput is deliberately omitted from "
        "the decision table.",
        "",
        f"## GSM8K ({gsm8k[0]['samples']} sampled test problems)",
        "",
    ]
    lines.extend(_dataset_table(gsm8k, "accuracy"))
    lines.extend(
        [
            "",
            f"## HumanEval ({humaneval[0]['samples']} sampled problems)",
            "",
        ]
    )
    lines.extend(_dataset_table(humaneval, "pass@1"))
    lines.extend(
        [
            "",
            "## Metric notes",
            "",
            "- `MAL = accepted draft tokens per round + one target correction/bonus`.",
            "- Native, Cactus and SpecCascade use the same three-layer MTP "
            "chain. Their node column is the measured drafted tokens/round.",
            "- Dynamic-tree MAL is recomputed from its selected paths, not from "
            "vLLM's linear speculative-position metric.",
            "- More tree nodes are a candidate-coverage budget, not accepted "
            "tokens; only a connected selected path contributes to MAL.",
            "- The 95% intervals are Wilson binomial intervals over the sampled "
            "tasks. A second generation seed is still required for a strong "
            "stochastic-decoding claim.",
            "",
        ]
    )
    payload = {"gsm8k": gsm8k, "humaneval": humaneval}
    return "\n".join(lines), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = args.output or (root / "comparison.md")
    json_output = args.json_output or (root / "comparison.json")
    report, payload = build_report(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
