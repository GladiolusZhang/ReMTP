"""Aggregate the exact-residual-hit FastMTP route suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_IDS = ("native", "cactus", "spec_cascade")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _route_manifest(route_root: Path) -> dict[str, Any]:
    path = route_root / "route_manifest.json"
    if not path.is_file():
        raise ValueError(f"route manifest is missing: {path}")
    return dict(_read_json(path))


def _audit_stats(path: Path) -> dict[str, float | int | None]:
    rounds = 0
    correction_opportunities = 0
    correction_hits = 0
    reused_hits = 0
    unlocked_tokens = 0.0
    strict_continuations = 0
    cactus_continuations = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        rounds += 1
        nodes = row.get("nodes") or []
        if any(node.get("trunk_rejected") for node in nodes):
            correction_opportunities += 1
        correction_rows = [
            node for node in nodes if "correction_hit" in node
        ]
        hit_rows = [node for node in correction_rows if node.get("correction_hit")]
        if hit_rows:
            correction_hits += 1
        reused = [node for node in hit_rows if node.get("rescued")]
        if reused:
            reused_hits += 1
            unlocked_tokens += max(
                float(node.get("rescue_unlocked_tokens") or 1.0)
                for node in reused
            )
        strict_continuations += sum(
            node.get("verification_rule") == "strict" for node in nodes
        )
        cactus_continuations += sum(
            node.get("verification_rule") == "cactus"
            and node.get("proposal_role") == "sampled_recovery_continuation"
            for node in nodes
        )
    return {
        "rounds": rounds,
        "correction_opportunities": correction_opportunities,
        "correction_hits": correction_hits,
        "correction_hit_rate": (
            correction_hits / correction_opportunities
            if correction_opportunities
            else None
        ),
        "reused_hit_rounds": reused_hits,
        "reused_hit_rate": reused_hits / rounds if rounds else None,
        "mean_unlocked_per_reused_hit": (
            unlocked_tokens / reused_hits if reused_hits else None
        ),
        "strict_continuation_nodes": strict_continuations,
        "cactus_continuation_nodes": cactus_continuations,
    }


def _number(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}%"


def _route_rows(
    suite_root: Path,
    route_names: list[str],
    dataset: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline: list[dict[str, Any]] | None = None
    routes: list[dict[str, Any]] = []
    for route_name in route_names:
        route_root = suite_root / route_name
        manifest = _route_manifest(route_root)
        comparison = _read_json(route_root / "comparison.json")
        dataset_rows = comparison[dataset]
        current_baseline = [
            dict(row) for row in dataset_rows if row["method_id"] in BASELINE_IDS
        ]
        if {row["method_id"] for row in current_baseline} != set(BASELINE_IDS):
            raise ValueError(f"incomplete baseline set: {route_root}/{dataset}")
        if baseline is None:
            baseline = current_baseline
        else:
            reference = {
                row["method_id"]: (row["samples"], row["quality"], row["mal"])
                for row in baseline
            }
            current = {
                row["method_id"]: (row["samples"], row["quality"], row["mal"])
                for row in current_baseline
            }
            if current != reference:
                raise ValueError(
                    f"baseline results differ across routes: {route_root}/{dataset}"
                )
        dynamic = next(
            dict(row) for row in dataset_rows if row["method_id"] == "dynamic_tree"
        )
        dynamic["method"] = manifest["label"]
        dynamic["route"] = route_name
        dynamic["mode"] = manifest["support_mode"]
        dynamic["max_nodes"] = manifest["max_nodes"]
        dynamic["max_children"] = manifest["max_children"]
        dynamic["audit"] = _audit_stats(
            route_root / dataset / "dynamic_tree" / "tree_rounds.jsonl"
        )
        routes.append(dynamic)
    assert baseline is not None
    return baseline, routes


def _table(
    baseline: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    quality_name: str,
) -> list[str]:
    cactus = next(row for row in baseline if row["method_id"] == "cactus")
    lines = [
        f"| method | {quality_name} | MAL | vs Cactus MAL | e2e tok/s | "
        "nodes/round | correction hit | reused rounds | unlocked/hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline:
        lines.append(
            f"| {row['method']} | {_percent(row['quality'])} | "
            f"{_number(row['mal'])} | "
            f"{float(row['mal']) - float(cactus['mal']):+.3f} | "
            f"{_number(row['e2e_tok_s'])} | {_number(row['nodes'])} | - | - | - |"
        )
    for row in routes:
        audit = row["audit"]
        lines.append(
            f"| {row['method']} | {_percent(row['quality'])} | "
            f"{_number(row['mal'])} | "
            f"{float(row['mal']) - float(cactus['mal']):+.3f} | "
            f"{_number(row['e2e_tok_s'])} | {_number(row['nodes'])} | "
            f"{_percent(audit['correction_hit_rate'])} | "
            f"{_percent(audit['reused_hit_rate'])} | "
            f"{_number(audit['mean_unlocked_per_reused_hit'])} |"
        )
    return lines


def build_report(suite_root: Path) -> tuple[str, dict[str, Any]]:
    suite = _read_json(suite_root / "suite_manifest.json")
    route_names = list(suite["routes"])
    gsm_base, gsm_routes = _route_rows(suite_root, route_names, "gsm8k")
    he_base, he_routes = _route_rows(suite_root, route_names, "humaneval")
    lines = [
        "# Exact-residual-hit tree exploration",
        "",
        "The sampled primary path uses the same Cactus rule as the chain baseline. "
        "At its first rejection, the correction token is sampled from the exact "
        "Cactus residual before the tree is consulted. A backup branch can be reused "
        "only when its token ID equals that already sampled correction.",
        "",
        f"## GSM8K ({gsm_base[0]['samples']} tasks)",
        "",
    ]
    lines.extend(_table(gsm_base, gsm_routes, "accuracy"))
    lines.extend(["", f"## HumanEval ({he_base[0]['samples']} tasks)", ""])
    lines.extend(_table(he_base, he_routes, "pass@1"))
    lines.extend(
        [
            "",
            "## Route semantics",
            "",
            "- `shadow`: exact Cactus correction; the tree records hits but never changes output.",
            "- `anchor`: on an exact hit, reuse the verified branch state and append one original target token.",
            "- `strict`: after an exact hit, continue through independently sampled branch Q with strict `p/q` verification.",
            "- `cactus`: after an exact hit, continue with the same Cactus relaxation.",
            "- `*_wide`: the same verifier with a larger candidate/node cap; this isolates residual-coverage effects.",
            "",
            "`correction hit` is conditional on a primary rejection. `reused rounds` "
            "uses all speculative rounds as denominator.",
            "",
        ]
    )
    payload = {
        "suite": suite,
        "gsm8k": {"baseline": gsm_base, "routes": gsm_routes},
        "humaneval": {"baseline": he_base, "routes": he_routes},
    }
    return "\n".join(lines), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    args = parser.parse_args()
    suite_root = args.suite_root.resolve()
    markdown, payload = build_report(suite_root)
    (suite_root / "comparison.md").write_text(markdown, encoding="utf-8")
    (suite_root / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print((suite_root / "comparison.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
