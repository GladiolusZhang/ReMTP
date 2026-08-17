from __future__ import annotations

import json
from pathlib import Path

from remtp.mimo_rl_route_report import METHODS, build_report


def _write(root: Path, dataset: str, method: str, index: int) -> None:
    directory = root / dataset / method
    directory.mkdir(parents=True)
    target = method == "target_only"
    tree = method.startswith("tree_")
    result = {
        "samples": 20,
        "correct": 10 + index,
        "accuracy": (10 + index) / 20,
        "pass_at_1": (10 + index) / 20,
        "evaluation_status": "complete",
        "truncation_rate": 0.05,
        "decode_tok_s": 100 + index,
        "e2e_output_tok_s": 90 + index,
        "mean_acceptance_length": None if target else 2.0,
        "accepted_draft_tokens_per_round": None if target else 1.0,
        "draft_rounds": 10,
        "draft_tokens": 30,
    }
    config = {
        "samples": 20,
        "sample_seed": 1 if dataset == "gsm8k" else 2,
        "temperature": 0.6,
        "generation_seed": 42,
        "max_tokens": 2048,
        "mtp_tokens": 0 if target else 3,
        "data_sha256": dataset,
        "model": "XiaomiMiMo/MiMo-7B-RL-0530",
        "system_prompt": "",
    }
    (directory / "summary.json").write_text(
        json.dumps({"config": config, "results": [result]}), encoding="utf-8"
    )
    (directory / "sample_manifest.json").write_text("same", encoding="utf-8")
    if tree:
        (directory / "tree_metrics.json").write_text(
            json.dumps(
                {
                    "tree": {
                        "mean_acceptance_length": 2.5,
                        "mean_accepted_depth": 1.5,
                        "target_nodes_per_round": 8.0,
                    }
                }
            ),
            encoding="utf-8",
        )


def test_route_report_accepts_target_mtp_token_count_difference(tmp_path: Path) -> None:
    for dataset in ("gsm8k", "humaneval"):
        for index, (method, _) in enumerate(METHODS):
            _write(tmp_path, dataset, method, index)
    markdown, payload = build_report(tmp_path)
    assert "explicit empty system message" in markdown
    assert "physical 0-1-2" in markdown
    assert payload["gsm8k"][0]["mal"] is None
    assert payload["humaneval"][-1]["mal"] == 2.5
