"""Combine direct-Cactus-tree and target-dominant guided-tree results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASETS = ("gsm8k", "humaneval")
BASELINE_IDS = ("native", "cactus", "spec_cascade")
REQUIRED_IDS = frozenset((*BASELINE_IDS, "dynamic_tree"))
PROTOCOL_KEYS = (
    "samples",
    "sample_seed",
    "temperature",
    "generation_seed",
    "max_tokens",
    "data_sha256",
    "model",
    "system_prompt",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(root: Path, dataset: str) -> dict[str, dict[str, Any]]:
    payload = _load_json(root / "comparison.json")
    rows = payload.get(dataset)
    if not isinstance(rows, list):
        raise ValueError(f"missing {dataset} rows: {root / 'comparison.json'}")
    indexed = {str(row["method_id"]): dict(row) for row in rows}
    missing = REQUIRED_IDS - indexed.keys()
    if missing:
        raise ValueError(
            f"missing methods in {root}/{dataset}: {', '.join(sorted(missing))}"
        )
    return indexed


def _protocol_signature(root: Path, dataset: str) -> tuple[Any, ...]:
    directory = root / dataset / "dynamic_tree"
    payload = _load_json(directory / "summary.json")
    config = payload.get("config") or {}
    manifest_hash = hashlib.sha256(
        (directory / "sample_manifest.json").read_bytes()
    ).hexdigest()
    return tuple(config.get(key) for key in PROTOCOL_KEYS) + (manifest_hash,)


def _same_baseline(
    direct: dict[str, Any], guided: dict[str, Any], *, dataset: str, method: str
) -> None:
    keys = (
        "samples",
        "quality",
        "mal",
        "draft_acceptance",
        "nodes",
        "truncation_rate",
    )
    mismatched = [key for key in keys if direct.get(key) != guided.get(key)]
    if mismatched:
        raise ValueError(
            f"baseline differs for {dataset}/{method}: {', '.join(mismatched)}"
        )


def _verified_variant(root: Path, expected_mode: str) -> None:
    path = root / "tree_variant.json"
    payload = _load_json(path)
    if (
        payload.get("support_mode") != expected_mode
        or payload.get("marker_verified") is not True
    ):
        raise ValueError(f"unverified tree variant: {path}")


def _number(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}%"


def _table(rows: list[dict[str, Any]], quality_name: str) -> list[str]:
    lines = [
        f"| method | {quality_name} | vs Native | MAL | ΔMAL vs Native | "
        "MAL gap to chain Cactus | nodes/round | draft acceptance | e2e tok/s | truncation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {_percent(row['quality'])} | "
            f"{float(row['quality_delta_vs_native_pp']):+.1f} pp | "
            f"{_number(row['mal'])} | {_number(row['mal_delta_vs_native'])} | "
            f"{_number(row['mal_gap_to_cactus'])} | {_number(row['nodes'])} | "
            f"{_percent(row['draft_acceptance'])} | {_number(row['e2e_tok_s'])} | "
            f"{_percent(row['truncation_rate'])} |"
        )
    return lines


def build_report(
    direct_root: Path, guided_root: Path
) -> tuple[str, dict[str, Any]]:
    direct_root = direct_root.resolve()
    guided_root = guided_root.resolve()
    _verified_variant(direct_root, "cactus")
    _verified_variant(guided_root, "cactus_guided")
    combined: dict[str, list[dict[str, Any]]] = {}

    for dataset in DATASETS:
        direct = _rows(direct_root, dataset)
        guided = _rows(guided_root, dataset)
        if _protocol_signature(direct_root, dataset) != _protocol_signature(
            guided_root, dataset
        ):
            raise ValueError(f"tree protocols or sample manifests differ: {dataset}")
        for method in BASELINE_IDS:
            _same_baseline(
                direct[method], guided[method], dataset=dataset, method=method
            )

        rows = [dict(direct[method]) for method in BASELINE_IDS]
        direct_tree = dict(direct["dynamic_tree"])
        direct_tree.update(
            method_id="direct_cactus_tree",
            method="Direct Cactus + Dynamic Tree",
        )
        guided_tree = dict(guided["dynamic_tree"])
        guided_tree.update(
            method_id="cactus_guided_tree",
            method="Cactus-guided Dynamic Tree (ours)",
        )
        rows.extend((direct_tree, guided_tree))

        native = direct["native"]
        cactus = direct["cactus"]
        for row in rows:
            row["quality_delta_vs_native_pp"] = 100.0 * (
                float(row["quality"]) - float(native["quality"])
            )
            row["mal_delta_vs_native"] = (
                None
                if row.get("mal") is None or native.get("mal") is None
                else float(row["mal"]) - float(native["mal"])
            )
            row["mal_gap_to_cactus"] = (
                None
                if row.get("mal") is None or cactus.get("mal") is None
                else float(row["mal"]) - float(cactus["mal"])
            )
        combined[dataset] = rows

    lines = [
        "# FastMTP tree relaxation ablation",
        "",
        "This report separates direct Cactus verification on every tree node from "
        "the proposed target-dominant tree that uses Cactus only to calibrate a "
        "dead frontier. Both runs use the same checkpoint, sampled tasks, generation "
        "protocol, Q-driven soft-reach tree, D=3, maximum 10 nodes and one target "
        "forward per round.",
        "",
        f"## GSM8K ({combined['gsm8k'][0]['samples']} tasks)",
        "",
    ]
    lines.extend(_table(combined["gsm8k"], "accuracy"))
    lines.extend(
        [
            "",
            f"## HumanEval ({combined['humaneval'][0]['samples']} tasks)",
            "",
        ]
    )
    lines.extend(_table(combined["humaneval"], "pass@1"))
    lines.extend(
        [
            "",
            "## Method distinction",
            "",
            "- **Direct Cactus + Dynamic Tree**: every candidate node survives with "
            "the chain Cactus probability; the dynamic tree only changes candidate "
            "coverage and path selection. This is the Cactus+tree ablation.",
            "- **Cactus-guided Dynamic Tree (ours)**: normal survival remains target-"
            "relative. Cactus is used only when a surviving parent has no normal "
            "child, requires tree/target evidence, and can rescue at most one node "
            "per selected path.",
            "- Both tree variants remain approximate and must be judged jointly by "
            "quality and MAL.",
            "",
            f"Direct run: `{direct_root}`",
            "",
            f"Guided run: `{guided_root}`",
            "",
        ]
    )
    payload: dict[str, Any] = {
        "direct_root": str(direct_root),
        "guided_root": str(guided_root),
        **combined,
    }
    return "\n".join(lines), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--guided-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    markdown, payload = build_report(args.direct_root, args.guided_root)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison.md").write_text(markdown, encoding="utf-8")
    (output_root / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_root / "comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
