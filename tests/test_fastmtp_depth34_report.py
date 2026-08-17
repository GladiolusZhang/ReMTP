from __future__ import annotations

import json
from pathlib import Path

import pytest

from remtp.fastmtp_depth34_report import build_report


def _rows(depth: int) -> dict[str, list[dict[str, object]]]:
    methods = (
        ("native", "Native FastMTP", 3.0),
        ("cactus", "Cactus + FastMTP", 3.2),
        ("spec_cascade", "SpecCascade", 3.1),
        ("dynamic_tree", "Dynamic tree", 3.3 if depth == 3 else 3.5),
    )
    payload: dict[str, list[dict[str, object]]] = {}
    for dataset in ("gsm8k", "humaneval"):
        payload[dataset] = [
            {
                "method_id": method_id,
                "method": label,
                "samples": 10,
                "quality": 0.8,
                "decode_tok_s": 10.0,
                "e2e_tok_s": 9.0,
                "truncation_rate": 0.0,
                "mal": mal,
                "draft_acceptance": 0.5,
                "nodes": 3.0 if method_id != "dynamic_tree" else float(depth + 4),
                "rescue_rate": None if method_id != "dynamic_tree" else 0.02,
            }
            for method_id, label, mal in methods
        ]
    return payload


def _write(root: Path, payload: dict[str, object]) -> None:
    root.mkdir()
    (root / "comparison.json").write_text(json.dumps(payload), encoding="utf-8")


def test_depth34_report_reuses_baselines_and_keeps_both_trees(tmp_path: Path) -> None:
    d3, d4 = tmp_path / "d3", tmp_path / "d4"
    _write(d3, _rows(3))
    _write(d4, _rows(4))
    markdown, payload = build_report(d3, d4)
    assert "Relaxed tree D=3" in markdown
    assert "Relaxed tree D=4" in markdown
    assert len(payload["gsm8k"]) == 5
    assert payload["gsm8k"][-1]["mal"] == 3.5


def test_depth34_report_rejects_changed_shared_baseline(tmp_path: Path) -> None:
    d3, d4 = tmp_path / "d3", tmp_path / "d4"
    left, right = _rows(3), _rows(4)
    right["gsm8k"][0]["quality"] = 0.7
    _write(d3, left)
    _write(d4, right)
    with pytest.raises(ValueError, match="baseline mismatch"):
        build_report(d3, d4)
