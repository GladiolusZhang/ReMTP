from __future__ import annotations

import json
from pathlib import Path

import pytest

from remtp.fastmtp_verified_report import METHODS, build_report


def _write_run(root: Path, dataset: str, method: str) -> None:
    directory = root / dataset / method
    directory.mkdir(parents=True)
    config = {
        "samples": 2,
        "sample_seed": 7,
        "temperature": 0.6,
        "generation_seed": 42,
        "max_tokens": 128,
        "data_sha256": "abc",
        "model": "TencentBAC/FastMTP",
        "system_prompt": "",
        "mtp_tokens": 0 if method == "target" else 3,
    }
    result = {
        "samples": 2,
        "accuracy": 1.0 if dataset == "gsm8k" else None,
        "pass_at_1": 1.0 if dataset == "humaneval" else None,
        "evaluation_status": "complete" if dataset == "humaneval" else None,
        "decode_tok_s": 10.0,
        "e2e_output_tok_s": 9.0,
        "mean_acceptance_length": None if method == "target" else 2.5,
        "draft_token_acceptance_rate": None if method == "target" else 0.5,
        "draft_rounds": 4 if method != "target" else 0,
        "draft_tokens": 12 if method != "target" else 0,
        "truncation_rate": 0.0,
    }
    (directory / "summary.json").write_text(
        json.dumps({"config": config, "results": [result]}), encoding="utf-8"
    )
    (directory / "sample_manifest.json").write_text("[]", encoding="utf-8")
    (directory / "method_runtime.json").write_text(
        json.dumps(
            {
                "method": method,
                "marker_verified": True,
                "adapter_verified": True,
                "marker": method,
            }
        ),
        encoding="utf-8",
    )
    if method == "dynamic_tree":
        (directory / "tree_metrics.json").write_text(
            json.dumps(
                {
                    "tree": {
                        "verification_mode": "residual_hit_anchor",
                        "mean_acceptance_length": 3.0,
                        "mean_accepted_depth": 2.0,
                        "ordinary_mean_accepted_depth": 1.75,
                        "rescue_extension_mal_gain": 0.25,
                        "pre_rescue_mean_accepted_depth": 1.50,
                        "rescue_unlocked_mal_gain": 0.50,
                        "target_nodes_per_round": 8.0,
                    }
                }
            ),
            encoding="utf-8",
        )


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    for dataset in ("gsm8k", "humaneval"):
        for method, _ in METHODS:
            _write_run(root, dataset, method)
    return root


def _fixture_without_target(tmp_path: Path) -> Path:
    root = tmp_path / "run_without_target"
    for dataset in ("gsm8k", "humaneval"):
        for method, _ in METHODS:
            if method != "target":
                _write_run(root, dataset, method)
    return root


def test_report_reads_real_benchmark_key_names_and_tree_metrics(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    markdown, payload = build_report(root)
    assert "Cactus + FastMTP" in markdown
    tree = next(row for row in payload["gsm8k"] if row["method_id"] == "dynamic_tree")
    assert tree["mal"] == 3.0
    assert tree["draft_acceptance"] == 0.25
    assert tree["ordinary_depth"] == 1.50
    assert tree["extension_mal_gain"] == 0.50
    assert tree["verification_mode"] == "residual_hit_anchor"
    assert tree["speedup_vs_native"] == 1.0
    assert "exact residual-hit anchoring" in markdown


def test_report_allows_target_to_be_omitted(tmp_path: Path) -> None:
    root = _fixture_without_target(tmp_path)
    markdown, payload = build_report(root)
    assert "Target only" not in markdown
    assert "speed vs Native" in markdown
    assert len(payload["gsm8k"]) == 4
    native = next(row for row in payload["gsm8k"] if row["method_id"] == "native")
    assert native["speedup_vs_native"] == 1.0


def test_report_rejects_unverified_worker_marker(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    runtime = root / "gsm8k" / "cactus" / "method_runtime.json"
    runtime.write_text(
        json.dumps({"method": "native", "marker_verified": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unverified runtime"):
        build_report(root)
