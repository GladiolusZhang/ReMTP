from __future__ import annotations

import json
from pathlib import Path

from remtp.combined_benchmark_report import build_report


def _write(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True)
    (root / "comparison.json").write_text(json.dumps(rows), encoding="utf-8")
    (root / "comparison.md").write_text("source\n", encoding="utf-8")


def _row(profile: str, *, human: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "directory": profile,
        "method": profile,
        "decode_tok_s": 12.0,
        "e2e_tok_s": 11.0,
        "mean_acceptance_length": 4.5,
        "draft_acceptance": 0.6,
        "truncation_rate": 0.0,
    }
    row["pass_at_1" if human else "accuracy"] = 0.75
    return row


def test_combined_report_contains_focus_and_both_full_tables(tmp_path: Path) -> None:
    gsm_root = tmp_path / "gsm"
    human_root = tmp_path / "human"
    profiles = ["cactus", "remtp"]
    _write(gsm_root, [_row(profile, human=False) for profile in profiles])
    _write(human_root, [_row(profile, human=True) for profile in profiles])

    report = build_report(gsm_root, human_root)

    assert "## Main method across both datasets" in report
    assert "## GSM8K: all methods" in report
    assert "## HumanEval: all methods" in report
    assert "remtp" in report
