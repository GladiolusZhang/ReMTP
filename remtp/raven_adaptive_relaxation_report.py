"""Aggregate the uniform-control and adaptive-TV RAVEN comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from remtp.fastmtp_verified_report import _load_dataset


BASELINE_IDS = ("native", "cactus", "spec_cascade")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}%"


def _number(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _audit(path: Path) -> dict[str, Any]:
    depths: dict[int, dict[str, Any]] = {}
    blocks: dict[str, list[float]] = {
        "baseline_tv": [],
        "adaptive_tv": [],
        "recovered_tv": [],
        "spent_tv": [],
    }
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        round_row = json.loads(raw)
        primary = [
            node
            for node in round_row.get("nodes") or []
            if node.get("proposal_role") == "sampled_cactus_primary"
        ]
        if primary and primary[0].get("adaptive_block_baseline_tv") is not None:
            blocks["baseline_tv"].append(
                float(primary[0]["adaptive_block_baseline_tv"])
            )
            blocks["adaptive_tv"].append(
                float(primary[0]["adaptive_block_tv"])
            )
            blocks["recovered_tv"].append(
                float(primary[0]["adaptive_block_recovered_tv"])
            )
            blocks["spent_tv"].append(
                float(primary[0]["adaptive_block_spent_tv"])
            )
        for node in primary:
            depth = int(node["trunk_position"])
            current = depths.setdefault(
                depth,
                {
                    "seen": 0,
                    "accepted": 0,
                    "weights": [],
                    "deltas": [],
                    "tv": [],
                    "baseline_tv": [],
                    "saturation_tv": [],
                },
            )
            current["seen"] += 1
            current["accepted"] += int(bool(node.get("trunk_accepted")))
            for source, destination in (
                ("relaxation_depth_weight", "weights"),
                ("effective_relaxation_delta", "deltas"),
                ("relaxation_tv", "tv"),
                ("baseline_relaxation_tv", "baseline_tv"),
                ("saturation_relaxation_tv", "saturation_tv"),
            ):
                if node.get(source) is not None:
                    current[destination].append(float(node[source]))
    depth_result: dict[str, Any] = {}
    for depth, row in sorted(depths.items()):
        seen = int(row["seen"])
        depth_result[str(depth)] = {
            "seen": seen,
            "acceptance": int(row["accepted"]) / seen if seen else None,
            "mean_weight": _mean(row["weights"]),
            "mean_delta": _mean(row["deltas"]),
            "mean_tv": _mean(row["tv"]),
            "mean_baseline_tv": _mean(row["baseline_tv"]),
            "mean_saturation_tv": _mean(row["saturation_tv"]),
        }
    return {
        "depths": depth_result,
        "blocks": {f"mean_{key}": _mean(values) for key, values in blocks.items()},
    }


def _profile_row(root: Path, profile: str, dataset: str) -> dict[str, Any]:
    rows = _load_dataset(root / profile, dataset)
    dynamic = dict(next(row for row in rows if row["method_id"] == "dynamic_tree"))
    manifest = _read_json(root / profile / "profile_manifest.json")
    dynamic.update(
        {
            "method": manifest["label"],
            "profile": profile,
            "manifest": manifest,
            "audit": _audit(
                root / profile / dataset / "dynamic_tree" / "tree_rounds.jsonl"
            ),
        }
    )
    return dynamic


def _dataset_rows(
    root: Path,
    profiles: list[str],
    dataset: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = _load_dataset(root / profiles[0], dataset)
    baseline = [row for row in reference if row["method_id"] in BASELINE_IDS]
    variants = [_profile_row(root, profile, dataset) for profile in profiles]
    return baseline, variants


def _main_table(
    baseline: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    quality_name: str,
) -> list[str]:
    lines = [
        f"| method | {quality_name} | MAL | e2e tok/s | draft acceptance | nodes/round |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in (*baseline, *variants):
        lines.append(
            f"| {row['method']} | {_percent(row['quality'])} | "
            f"{_number(row['mal'])} | {_number(row['e2e_tok_s'])} | "
            f"{_percent(row['draft_acceptance'])} | {_number(row['nodes'])} |"
        )
    return lines


def _audit_table(variants: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| profile | depth | mean dynamic weight | mean delta | mean TV | "
        "primary acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        for depth, row in variant["audit"]["depths"].items():
            lines.append(
                f"| {variant['profile']} | {depth} | "
                f"{_number(row['mean_weight'])} | {_number(row['mean_delta'])} | "
                f"{_number(row['mean_tv'], 4)} | {_percent(row['acceptance'])} |"
            )
    return lines


def _block_table(variants: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| profile | baseline block TV | realized block TV | recoverable TV | "
        "reallocated TV |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in variants:
        block = variant["audit"]["blocks"]
        lines.append(
            f"| {variant['profile']} | {_number(block['mean_baseline_tv'], 4)} | "
            f"{_number(block['mean_adaptive_tv'], 4)} | "
            f"{_number(block['mean_recovered_tv'], 4)} | "
            f"{_number(block['mean_spent_tv'], 4)} |"
        )
    return lines


def build_report(root: Path) -> tuple[str, dict[str, Any]]:
    suite = _read_json(root / "suite_manifest.json")
    profiles = list(suite["profiles"])
    gsm_base, gsm_variants = _dataset_rows(root, profiles, "gsm8k")
    he_base, he_variants = _dataset_rows(root, profiles, "humaneval")
    lines = [
        "# RAVEN saturation-aware adaptive relaxation",
        "",
        "The adaptive profile starts from the uniform candidate-conditioned "
        "transform, caps positions whose acceptance probability has reached one, "
        "and reallocates part of the recovered actual TV according to current-block "
        "expected-MAL marginal utility. The target forward, proposal tree, residual "
        "correction draw, and exact residual-hit reuse are unchanged.",
        "",
        f"## GSM8K ({gsm_base[0]['samples']} tasks)",
        "",
    ]
    lines.extend(_main_table(gsm_base, gsm_variants, "accuracy"))
    lines.extend(["", "### GSM8K depth audit", ""])
    lines.extend(_audit_table(gsm_variants))
    lines.extend(["", "### GSM8K block-TV audit", ""])
    lines.extend(_block_table(gsm_variants))
    lines.extend(["", f"## HumanEval ({he_base[0]['samples']} tasks)", ""])
    lines.extend(_main_table(he_base, he_variants, "pass@1"))
    lines.extend(["", "### HumanEval depth audit", ""])
    lines.extend(_audit_table(he_variants))
    lines.extend(["", "### HumanEval block-TV audit", ""])
    lines.extend(_block_table(he_variants))
    payload = {
        "suite": suite,
        "gsm8k": {"baseline": gsm_base, "variants": gsm_variants},
        "humaneval": {"baseline": he_base, "variants": he_variants},
    }
    return "\n".join(lines) + "\n", payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.suite_root.resolve()
    markdown, payload = build_report(root)
    (root / "comparison.md").write_text(markdown, encoding="utf-8")
    (root / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print((root / "comparison.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
