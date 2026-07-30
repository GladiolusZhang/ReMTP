"""Append a regret-feedback result to an existing local five-way table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from remtp.target_anchored_compare import comparison_rows


AUDIT_PREFIX = "[ReMTP][Regret][audit] "
SCALAR_PATTERN = re.compile(r"([a-z_]+)=([0-9.eE+-]+)")
ARRAY_PATTERN = re.compile(r"([a-z_]+)=(\[[^\]]*\])")


def load_single_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one result in {path}")
    return dict(results[0])


def parse_audit(path: Path) -> dict[str, Any]:
    totals = {
        "accepted": 0.0,
        "strict_accepted": 0.0,
        "causal_relaxed": 0.0,
        "accepted_tv": 0.0,
        "injections": 0.0,
        "gate_sum": 0.0,
        "rounds": 0,
    }
    vector_totals: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = line.find(AUDIT_PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(AUDIT_PREFIX) :]
        scalars = {
            key: float(value)
            for key, value in SCALAR_PATTERN.findall(payload)
            if key != "rounds"
        }
        round_range = re.search(r"rounds=(\d+)-(\d+)", payload)
        if round_range is None:
            continue
        window_rounds = int(round_range.group(2)) - int(round_range.group(1)) + 1
        accepted = scalars.get("accepted", 0.0)
        totals["rounds"] += window_rounds
        totals["accepted"] += accepted
        totals["strict_accepted"] += scalars.get("strict_accepted", 0.0)
        totals["causal_relaxed"] += scalars.get("causal_relaxed", 0.0)
        totals["accepted_tv"] += scalars.get("tv_per_accepted", 0.0) * accepted
        injections = scalars.get("injections", 0.0)
        totals["injections"] += injections
        totals["gate_sum"] += scalars.get("mean_gate", 0.0) * injections
        for key, value in ARRAY_PATTERN.findall(payload):
            numbers = [float(item) for item in json.loads(value)]
            if key not in vector_totals:
                vector_totals[key] = [0.0] * len(numbers)
            vector_totals[key] = [
                left + right
                for left, right in zip(vector_totals[key], numbers)
            ]

    accepted = max(totals["accepted"], 1.0)
    injections = max(totals["injections"], 1.0)
    result: dict[str, Any] = {
        "audited_rounds": totals["rounds"],
        "accepted_draft_tokens": totals["accepted"],
        "strict_acceptable_ratio": totals["strict_accepted"] / accepted,
        "causal_relaxed_acceptances": totals["causal_relaxed"],
        "tv_per_accepted_token": totals["accepted_tv"] / accepted,
        "feedback_injections": totals["injections"],
        "mean_compatibility_gate": totals["gate_sum"] / injections,
    }
    reached = vector_totals.get("head_reached")
    if reached is not None:
        denominator = [max(item, 1.0) for item in reached]
        result["per_head_strict_acceptance"] = [
            value / denom
            for value, denom in zip(
                vector_totals.get("head_strict_count", []),
                denominator,
            )
        ]
        result["per_head_causal_relaxed_rate"] = [
            value / denom
            for value, denom in zip(
                vector_totals.get("head_causal_count", []),
                denominator,
            )
        ]
        result["per_head_target_candidate_prob"] = [
            value / denom
            for value, denom in zip(
                vector_totals.get("head_target_p_sum", []),
                denominator,
            )
        ]
        result["per_head_hidden_cosine"] = [
            value / denom
            for value, denom in zip(
                vector_totals.get("head_hidden_cos_sum", []),
                denominator,
            )
        ]
    return result


def write_comparison(
    output_root: Path,
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "regret_mechanism.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# GSM8K MTP=6 comparison with regret feedback",
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
    lines.extend(
        (
            "",
            "Regret mechanism audit:",
            "",
            f"- strict-acceptable ratio: "
            f"{100.0 * audit.get('strict_acceptable_ratio', 0.0):.2f}%",
            f"- causal relaxed acceptances: "
            f"{audit.get('causal_relaxed_acceptances', 0.0):.0f}",
            f"- TV per accepted draft token: "
            f"{audit.get('tv_per_accepted_token', 0.0):.6f}",
            f"- feedback injections: "
            f"{audit.get('feedback_injections', 0.0):.0f}",
        )
    )
    (output_root / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("regret_summary", type=Path)
    parser.add_argument("server_log", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    rows = comparison_rows(args.baseline_root.resolve())
    summary = load_single_summary(args.regret_summary.resolve())
    rows.append(
        {
            "method": "Exact-TV + regret feedback",
            "samples": summary["samples"],
            "accuracy": summary["accuracy"],
            "decode_tok_s": summary["decode_tok_s"],
            "e2e_output_tok_s": summary["e2e_output_tok_s"],
            "mean_acceptance_length": summary["mean_acceptance_length"],
            "draft_token_acceptance_rate": summary[
                "draft_token_acceptance_rate"
            ],
            "truncation_rate": summary["truncation_rate"],
        }
    )
    audit = parse_audit(args.server_log.resolve())
    write_comparison(args.output_root.resolve(), rows, audit)
    print(
        (args.output_root.resolve() / "comparison.md").read_text(
            encoding="utf-8"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
