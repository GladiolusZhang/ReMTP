from __future__ import annotations

import json
from pathlib import Path

import pytest

from remtp.fastmtp_tree_relaxation_report import build_report


def _write_run(
    root: Path, *, tree_quality: float, tree_mal: float, support_mode: str
) -> None:
    rows_by_dataset: dict[str, list[dict[str, object]]] = {}
    for dataset in ("gsm8k", "humaneval"):
        rows: list[dict[str, object]] = []
        for method, quality, mal in (
            ("native", 0.8, 3.0),
            ("cactus", 0.8, 3.4),
            ("spec_cascade", 0.8, 3.2),
            ("dynamic_tree", tree_quality, tree_mal),
        ):
            rows.append(
                {
                    "method_id": method,
                    "method": method,
                    "samples": 2,
                    "quality": quality,
                    "mal": mal,
                    "draft_acceptance": 0.5,
                    "nodes": 3.0 if method != "dynamic_tree" else 7.0,
                    "e2e_tok_s": 10.0,
                    "truncation_rate": 0.0,
                    "quality_delta_vs_native_pp": 0.0,
                }
            )
        rows_by_dataset[dataset] = rows
        directory = root / dataset / "dynamic_tree"
        directory.mkdir(parents=True)
        config = {
            "samples": 2,
            "sample_seed": 7,
            "temperature": 0.6,
            "generation_seed": 42,
            "max_tokens": 128,
            "data_sha256": f"{dataset}-sha",
            "model": "TencentBAC/FastMTP",
            "system_prompt": "",
        }
        (directory / "summary.json").write_text(
            json.dumps({"config": config, "results": [{}]}), encoding="utf-8"
        )
        (directory / "sample_manifest.json").write_text("[]", encoding="utf-8")
    (root / "comparison.json").write_text(
        json.dumps(rows_by_dataset), encoding="utf-8"
    )
    (root / "tree_variant.json").write_text(
        json.dumps(
            {
                "support_mode": support_mode,
                "marker_verified": True,
                "server_log": "/tmp/server.log",
            }
        ),
        encoding="utf-8",
    )


def test_combined_report_separates_direct_and_guided_tree(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    guided = tmp_path / "guided"
    _write_run(
        direct, tree_quality=0.79, tree_mal=3.25, support_mode="cactus"
    )
    _write_run(
        guided,
        tree_quality=0.81,
        tree_mal=3.30,
        support_mode="cactus_guided",
    )
    markdown, payload = build_report(direct, guided)
    assert "Direct Cactus + Dynamic Tree" in markdown
    assert "Cactus-guided Dynamic Tree (ours)" in markdown
    guided_row = next(
        row
        for row in payload["gsm8k"]
        if row["method_id"] == "cactus_guided_tree"
    )
    assert guided_row["mal_delta_vs_native"] == pytest.approx(0.30)
    assert guided_row["mal_gap_to_cactus"] == pytest.approx(-0.10)


def test_combined_report_rejects_different_tree_protocols(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    guided = tmp_path / "guided"
    _write_run(
        direct, tree_quality=0.79, tree_mal=3.25, support_mode="cactus"
    )
    _write_run(
        guided,
        tree_quality=0.81,
        tree_mal=3.30,
        support_mode="cactus_guided",
    )
    summary = guided / "gsm8k" / "dynamic_tree" / "summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["config"]["generation_seed"] = 99
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocols or sample manifests differ"):
        build_report(direct, guided)


def test_combined_report_rejects_wrong_variant_marker(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    guided = tmp_path / "guided"
    _write_run(
        direct, tree_quality=0.79, tree_mal=3.25, support_mode="cactus"
    )
    _write_run(
        guided, tree_quality=0.81, tree_mal=3.30, support_mode="cactus"
    )
    with pytest.raises(ValueError, match="unverified tree variant"):
        build_report(direct, guided)
