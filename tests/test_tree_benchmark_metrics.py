from __future__ import annotations

import json
from pathlib import Path

from remtp.tree_benchmark_metrics import write_metrics


def test_write_metrics_uses_committed_tree_path_length(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    audit = tmp_path / "rounds.jsonl"
    records = [
        {
            "topology": "dynamic",
            "verification_mode": "target_dynamic_relaxed",
            "accepted_drafts": 2,
            "output_token_ids": [11, 12, 13],
            "target_forward_calls": 1,
            "target_validation_nodes": 5,
            "selected_branch": 0,
            "terminal": "bonus",
            "nodes": [],
        },
        {
            "topology": "dynamic",
            "verification_mode": "target_dynamic_relaxed",
            "accepted_drafts": 0,
            "output_token_ids": [14],
            "target_forward_calls": 1,
            "target_validation_nodes": 3,
            "selected_branch": None,
            "terminal": "correction",
            "nodes": [],
        },
    ]
    audit.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    payload = write_metrics(run_dir, audit)
    tree = payload["tree"]
    assert tree["mean_accepted_depth"] == 1.0
    assert tree["mean_acceptance_length"] == 2.0
    assert tree["target_nodes_per_round"] == 4.0
    assert (run_dir / "tree_metrics.md").is_file()
