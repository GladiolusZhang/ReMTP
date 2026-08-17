"""Render the clean small-sample Proposal-Calibrated MTP comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _result(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return payload["results"][0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    offline = json.loads((root / "offline_analysis.json").read_text(encoding="utf-8"))
    lines = [
        "# Proposal-Calibrated MTP small-sample pilot",
        "",
        "All online rows use strict probabilistic verification; target P is not relaxed.",
        "Trace collection is excluded from speed measurements.",
        "",
        "| dataset | method | quality* | decode tok/s | e2e tok/s | MAL | draft acceptance | online MAL delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    gates: dict[str, bool] = {}
    for dataset in ("gsm8k", "humaneval"):
        native = _result(root / "pilot" / dataset / "native")
        calibrated_dir = root / "pilot" / dataset / "calibrated_optimized"
        if not calibrated_dir.is_dir():
            calibrated_dir = root / "pilot" / dataset / "calibrated"
        calibrated = _result(calibrated_dir)
        delta = float(calibrated["mean_acceptance_length"]) - float(
            native["mean_acceptance_length"]
        )
        threshold = 0.10 if dataset == "gsm8k" else 0.03
        predicted_delta = float(
            offline["selection"]["heldout"][dataset]["delta"]
        )
        gates[dataset] = delta >= threshold or predicted_delta >= threshold
        for method, row, mal_delta in (
            ("Native probabilistic MTP", native, 0.0),
            ("Proposal-Calibrated MTP", calibrated, delta),
        ):
            quality = row.get("accuracy")
            quality_text = "—" if quality is None else f"{100*float(quality):.1f}%"
            lines.append(
                f"| {dataset} | {method} | {quality_text} | "
                f"{float(row['decode_tok_s']):.3f} | {float(row['e2e_output_tok_s']):.3f} | "
                f"{float(row['mean_acceptance_length']):.3f} | "
                f"{100*float(row['draft_token_acceptance_rate']):.1f}% | {mal_delta:+.3f} |"
            )
        lines.append(
            f"| {dataset} | offline frozen-prefix estimate | — | — | — | — | — | "
            f"{predicted_delta:+.3f} |"
        )
    passed = all(gates.values())
    lines.extend(
        [
            "",
            "`quality*` is GSM8K exact accuracy. HumanEval code execution is intentionally not part of this pre-formal pilot.",
            "",
            f"Offline frozen-prefix gate: **{'PASS' if offline['selection']['offline_gate_pass'] else 'NO'}**.",
            f"Combined predicted-or-measured MAL gate: **{'PASS' if passed else 'NO'}**.",
            "",
            (
                "Conclusion: GO to complete GSM8K/HumanEval validation; the combined predicted-or-measured MAL gate passed."
                if passed
                else "Conclusion: do not run full GSM8K/HumanEval; static calibration has not cleared the MAL gate."
            ),
        ]
    )
    output = root / "pilot_comparison.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
