"""Aggregate the RAVEN native-MTP depth-relaxation sweep."""

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


def _depth_audit(path: Path) -> dict[str, dict[str, float | int | None]]:
    values: dict[int, dict[str, list[float] | int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        round_row = json.loads(raw)
        for node in round_row.get("nodes") or []:
            if node.get("proposal_role") != "sampled_cactus_primary":
                continue
            depth = int(node["trunk_position"])
            current = values.setdefault(
                depth,
                {
                    "seen": 0,
                    "accepted": 0,
                    "weights": [],
                    "deltas": [],
                    "tv": [],
                },
            )
            current["seen"] = int(current["seen"]) + 1
            current["accepted"] = int(current["accepted"]) + int(
                bool(node.get("trunk_accepted"))
            )
            if node.get("relaxation_depth_weight") is not None:
                cast_weights = current["weights"]
                assert isinstance(cast_weights, list)
                cast_weights.append(float(node["relaxation_depth_weight"]))
            if node.get("effective_relaxation_delta") is not None:
                cast_deltas = current["deltas"]
                assert isinstance(cast_deltas, list)
                cast_deltas.append(float(node["effective_relaxation_delta"]))
            if node.get("relaxation_tv") is not None:
                cast_tv = current["tv"]
                assert isinstance(cast_tv, list)
                cast_tv.append(float(node["relaxation_tv"]))
    result: dict[str, dict[str, float | int | None]] = {}
    for depth, row in sorted(values.items()):
        seen = int(row["seen"])
        accepted = int(row["accepted"])
        weights = row["weights"]
        deltas = row["deltas"]
        tv = row["tv"]
        assert (
            isinstance(weights, list)
            and isinstance(deltas, list)
            and isinstance(tv, list)
        )
        result[str(depth)] = {
            "seen": seen,
            "accepted": accepted,
            "acceptance": accepted / seen if seen else None,
            "mean_weight": sum(weights) / len(weights) if weights else None,
            "mean_effective_delta": (
                sum(deltas) / len(deltas) if deltas else None
            ),
            "mean_relaxation_tv": sum(tv) / len(tv) if tv else None,
        }
    return result


def _load_profile(
    suite_root: Path,
    profile: str,
    dataset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = suite_root / profile
    rows = _load_dataset(root, dataset)
    dynamic = next(row for row in rows if row["method_id"] == "dynamic_tree")
    manifest = _read_json(root / "profile_manifest.json")
    dynamic = dict(dynamic)
    dynamic["method"] = manifest["label"]
    dynamic["profile"] = profile
    dynamic["base_delta"] = float(manifest["base_delta"])
    dynamic["depth_weights"] = [
        float(value) for value in manifest["depth_weights"]
    ]
    depth_audit = _depth_audit(
        root / dataset / "dynamic_tree" / "tree_rounds.jsonl"
    )
    for depth, weight in enumerate(dynamic["depth_weights"], start=1):
        row = depth_audit.setdefault(
            str(depth),
            {
                "seen": 0,
                "accepted": 0,
                "acceptance": None,
                "mean_weight": None,
                "mean_effective_delta": None,
                "mean_relaxation_tv": None,
            },
        )
        if row["mean_weight"] is None:
            row["mean_weight"] = weight
        if row["mean_effective_delta"] is None:
            row["mean_effective_delta"] = dynamic["base_delta"] * weight
    dynamic["depth_audit"] = depth_audit
    baseline = [row for row in rows if row["method_id"] in BASELINE_IDS]
    return baseline, dynamic


def _collect_dataset(
    suite_root: Path,
    profiles: list[str],
    dataset: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference: list[dict[str, Any]] | None = None
    variants: list[dict[str, Any]] = []
    for profile in profiles:
        baseline, variant = _load_profile(suite_root, profile, dataset)
        if reference is None:
            reference = baseline
        else:
            expected = {
                row["method_id"]: (row["samples"], row["quality"], row["mal"])
                for row in reference
            }
            actual = {
                row["method_id"]: (row["samples"], row["quality"], row["mal"])
                for row in baseline
            }
            if actual != expected:
                raise ValueError(
                    f"baseline results differ across profiles: {profile}/{dataset}"
                )
        variants.append(variant)
    if reference is None:
        raise ValueError("depth-relaxation sweep has no profiles")
    return reference, variants


def _main_table(
    baseline: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    quality_name: str,
) -> list[str]:
    lines = [
        f"| method | schedule $w_1,w_2,w_3$ | {quality_name} | MAL | "
        "e2e tok/s | draft acceptance | nodes/round | truncation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline:
        lines.append(
            f"| {row['method']} | - | {_percent(row['quality'])} | "
            f"{_number(row['mal'])} | {_number(row['e2e_tok_s'])} | "
            f"{_percent(row['draft_acceptance'])} | {_number(row['nodes'])} | "
            f"{_percent(row['truncation_rate'])} |"
        )
    for row in variants:
        schedule = ", ".join(f"{value:.3f}" for value in row["depth_weights"])
        lines.append(
            f"| {row['method']} | {schedule} | {_percent(row['quality'])} | "
            f"{_number(row['mal'])} | {_number(row['e2e_tok_s'])} | "
            f"{_percent(row['draft_acceptance'])} | {_number(row['nodes'])} | "
            f"{_percent(row['truncation_rate'])} |"
        )
    return lines


def _depth_table(variants: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| profile | depth | effective delta | mean TV | primary acceptance | seen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in variants:
        for depth, audit in row["depth_audit"].items():
            lines.append(
                f"| {row['profile']} | {depth} | "
                f"{_number(audit['mean_effective_delta'])} | "
                f"{_number(audit['mean_relaxation_tv'], 4)} | "
                f"{_percent(audit['acceptance'])} | {audit['seen']} |"
            )
    return lines


def build_report(suite_root: Path) -> tuple[str, dict[str, Any]]:
    suite = _read_json(suite_root / "suite_manifest.json")
    profiles = list(suite["profiles"])
    gsm_base, gsm_variants = _collect_dataset(suite_root, profiles, "gsm8k")
    he_base, he_variants = _collect_dataset(suite_root, profiles, "humaneval")
    lines = [
        "# RAVEN native-MTP depth-relaxation sweep",
        "",
        "RAVEN uses the same base candidate-conditioned transformation in every "
        "row, with $\\delta_i=\\delta w_i$. The calibrated schedules have mean "
        "weight one, so their nominal depth budget matches the uniform control; "
        "actual TV remains nonlinear and is audited separately. The chain Cactus "
        "baseline retains a uniform delta. Baselines are generated once and shared "
        "read-only across profiles.",
        "",
        f"## GSM8K ({gsm_base[0]['samples']} tasks)",
        "",
    ]
    lines.extend(_main_table(gsm_base, gsm_variants, "accuracy"))
    lines.extend(["", "### GSM8K depth audit", ""])
    lines.extend(_depth_table(gsm_variants))
    lines.extend(["", f"## HumanEval ({he_base[0]['samples']} tasks)", ""])
    lines.extend(_main_table(he_base, he_variants, "pass@1"))
    lines.extend(["", "### HumanEval depth audit", ""])
    lines.extend(_depth_table(he_variants))
    lines.extend(
        [
            "",
            "The `uniform` profile reproduces the former RAVEN primary verifier. "
            "The `empirical` profile uses the normalized depth-wise strict "
            "acceptance prior. The `squared` profile applies a stronger monotone "
            "decay to the same prior.",
            "",
        ]
    )
    payload = {
        "suite": suite,
        "gsm8k": {"baseline": gsm_base, "variants": gsm_variants},
        "humaneval": {"baseline": he_base, "variants": he_variants},
    }
    return "\n".join(lines), payload


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
