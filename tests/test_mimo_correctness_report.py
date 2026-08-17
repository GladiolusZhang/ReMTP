from __future__ import annotations

import json
from pathlib import Path

import pytest

from remtp.mimo_correctness_report import METHODS, build_report


def _write_run(
    root: Path,
    dataset: str,
    method: str,
    *,
    correct: int,
    mal: float,
    manifest: str = "same manifest\n",
) -> None:
    directory = root / dataset / method
    directory.mkdir(parents=True)
    samples = 100
    result = {
        "samples": samples,
        "correct": correct,
        "accuracy": correct / samples,
        "pass_at_1": correct / samples,
        "evaluation_status": "complete",
        "mean_acceptance_length": mal,
        "accepted_draft_tokens_per_round": mal - 1.0,
        "draft_rounds": 20,
        "draft_tokens": 60,
    }
    config = {
        "samples": samples,
        "sample_seed": 7 if dataset == "gsm8k" else 8,
        "temperature": 0.7,
        "generation_seed": 42,
        "max_tokens": 384 if dataset == "gsm8k" else 512,
        "mtp_tokens": 3,
        "data_sha256": f"{dataset}-data",
        "model": "XiaomiMiMo/MiMo-7B-Base",
    }
    (directory / "summary.json").write_text(
        json.dumps({"config": config, "results": [result]}),
        encoding="utf-8",
    )
    (directory / "sample_manifest.json").write_text(manifest, encoding="utf-8")
    if method == "dynamic_tree":
        (directory / "tree_metrics.json").write_text(
            json.dumps(
                {
                    "tree": {
                        "mean_acceptance_length": mal + 0.25,
                        "mean_accepted_depth": mal - 0.75,
                        "target_nodes_per_round": 5.5,
                    }
                }
            ),
            encoding="utf-8",
        )


def _write_complete_root(root: Path) -> None:
    for dataset in ("gsm8k", "humaneval"):
        for index, (method, _) in enumerate(METHODS):
            _write_run(
                root,
                dataset,
                method,
                correct=70 + index,
                mal=2.0 + 0.1 * index,
            )


def test_report_focuses_on_quality_and_tree_native_mal(tmp_path: Path) -> None:
    _write_complete_root(tmp_path)
    markdown, payload = build_report(tmp_path)
    assert "SpecCascade TokenV3 + MTP" in markdown
    assert "GSM8K (100 sampled test problems)" in markdown
    assert "decode tok/s" not in markdown
    assert "e2e tok/s" not in markdown
    dynamic = payload["gsm8k"][-1]
    assert dynamic["mean_acceptance_length"] == pytest.approx(2.55)
    assert dynamic["accepted_draft_tokens_per_round"] == pytest.approx(1.55)
    assert dynamic["average_tree_nodes"] == pytest.approx(5.5)
    assert payload["gsm8k"][0]["average_tree_nodes"] == pytest.approx(3.0)


def test_report_rejects_different_sample_manifests(tmp_path: Path) -> None:
    _write_complete_root(tmp_path)
    (tmp_path / "gsm8k" / "cactus" / "sample_manifest.json").write_text(
        "different\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sample manifest differs"):
        build_report(tmp_path)
