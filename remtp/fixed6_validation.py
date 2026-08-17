"""Validate Fixed-6 audit invariants and greedy reference equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from remtp.fixed6_microtree import FIXED_NODE_BUDGET, get_topology


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_audit(
    records: Iterable[dict[str, Any]], topology_name: str
) -> dict[str, Any]:
    topology = get_topology(topology_name)
    rows = list(records)
    errors: list[str] = []
    observed_branches: set[int] = set()
    for offset, record in enumerate(rows, start=1):
        if record.get("topology") != topology_name:
            errors.append(f"round {offset}: topology mismatch")
        if int(record.get("target_validation_nodes", -1)) != FIXED_NODE_BUDGET:
            errors.append(f"round {offset}: target node budget is not 6")
        if int(record.get("target_forward_calls", -1)) != 1:
            errors.append(f"round {offset}: target forward count is not 1")

        selected = tuple(map(int, record.get("selected_nodes", [])))
        branch = record.get("selected_branch")
        if selected:
            if branch is None:
                errors.append(f"round {offset}: selected path has no branch")
                continue
            branch = int(branch)
            observed_branches.add(branch)
            expected = topology.path_to(selected[-1])
            if selected != expected:
                errors.append(
                    f"round {offset}: {selected} is not a valid prefix path; "
                    f"expected {expected}"
                )
            if topology.nodes[selected[-1]].branch != branch:
                errors.append(f"round {offset}: selected branch is inconsistent")
        elif branch is not None:
            errors.append(f"round {offset}: empty path has a selected branch")

    return {
        "topology": topology_name,
        "rounds": len(rows),
        "target_validation_nodes_per_round": FIXED_NODE_BUDGET,
        "target_forward_calls_per_round": 1,
        "observed_branches": sorted(observed_branches),
        "audit_invariants_passed": bool(rows) and not errors,
        "errors": errors,
    }


def response_content(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["choices"][0]["message"]["content"])


def compare_responses(reference: Path, candidate: Path) -> dict[str, Any]:
    expected = response_content(reference)
    actual = response_content(candidate)
    first_mismatch = None
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            first_mismatch = index
            break
    if first_mismatch is None and len(expected) != len(actual):
        first_mismatch = min(len(expected), len(actual))
    return {
        "exact_text_match": expected == actual,
        "reference_characters": len(expected),
        "candidate_characters": len(actual),
        "first_character_mismatch": first_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--topology", choices=("6-chain", "4+2", "3+2+1"), required=True)
    parser.add_argument("--reference-response", type=Path)
    parser.add_argument("--candidate-response", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.reference_response is None) != (args.candidate_response is None):
        parser.error("provide both response files or neither")

    result = validate_audit(load_jsonl(args.audit), args.topology)
    if args.reference_response is not None:
        result["reference_equivalence"] = compare_responses(
            args.reference_response, args.candidate_response
        )
    equivalence = result.get("reference_equivalence", {})
    result["passed"] = result["audit_invariants_passed"] and bool(
        equivalence.get("exact_text_match", True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
