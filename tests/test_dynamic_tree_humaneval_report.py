from __future__ import annotations

import json

from remtp.dynamic_tree_humaneval_report import build_report, markdown


def test_combines_tree_native_mal_with_humaneval_speed(tmp_path) -> None:
    run = tmp_path / "humaneval"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "config": {"samples": 50},
                "results": [
                    {
                        "samples": 50,
                        "pass_at_1": 0.7,
                        "decode_tok_s": 100.0,
                        "e2e_output_tok_s": 90.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "rounds.jsonl"
    audit.write_text(
        json.dumps(
            {
                "topology": "dynamic-mtp",
                "verification_mode": "target_dynamic_relaxed",
                "round": 1,
                "accepted_drafts": 2,
                "target_forward_calls": 1,
                "target_validation_nodes": 5,
                "target_forward_seconds": 0.01,
                "selected_branch": 0,
                "terminal": "relaxed-bonus",
                "output_token_ids": [1, 2, 3],
                "surviving_paths": 3,
                "nodes": [
                    {"depth": 1, "status": "SELECTED", "parent": None, "coverage": 0.8},
                    {"depth": 2, "status": "SELECTED", "parent": 0, "coverage": 0.7},
                    {"depth": 1, "status": "PRUNE", "parent": None, "coverage": 0.8},
                ],
                "dynamic_construction": {
                    "branching": [
                        {"depth": 1, "retained": 2},
                        {"depth": 2, "retained": 3},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_report(run, audit)
    assert report["mean_acceptance_length"] == 3.0
    assert report["decode_tok_s"] == 100.0
    assert report["e2e_tok_s"] == 90.0
    assert report["average_tree_nodes"] == 5.0
    text = markdown(report)
    assert "70.0%" in text
    assert "100.000" in text
