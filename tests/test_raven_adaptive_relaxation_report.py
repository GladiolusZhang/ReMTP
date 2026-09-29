from __future__ import annotations

import json

import pytest

from remtp.raven_adaptive_relaxation_report import _audit


def test_adaptive_audit_counts_block_budget_once_per_round(tmp_path) -> None:
    path = tmp_path / "tree_rounds.jsonl"
    rows = [
        {
            "nodes": [
                {
                    "proposal_role": "sampled_cactus_primary",
                    "trunk_position": 1,
                    "trunk_accepted": True,
                    "relaxation_depth_weight": 0.4,
                    "effective_relaxation_delta": 0.4,
                    "relaxation_tv": 0.10,
                    "baseline_relaxation_tv": 0.20,
                    "saturation_relaxation_tv": 0.10,
                    "adaptive_block_baseline_tv": 0.60,
                    "adaptive_block_tv": 0.45,
                    "adaptive_block_recovered_tv": 0.20,
                    "adaptive_block_spent_tv": 0.05,
                },
                {
                    "proposal_role": "sampled_cactus_primary",
                    "trunk_position": 2,
                    "trunk_accepted": False,
                    "relaxation_depth_weight": 1.2,
                    "effective_relaxation_delta": 1.2,
                    "relaxation_tv": 0.25,
                    "baseline_relaxation_tv": 0.20,
                    "saturation_relaxation_tv": 0.30,
                    "adaptive_block_baseline_tv": 0.60,
                    "adaptive_block_tv": 0.45,
                    "adaptive_block_recovered_tv": 0.20,
                    "adaptive_block_spent_tv": 0.05,
                },
            ]
        }
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    audit = _audit(path)

    assert audit["depths"]["1"]["acceptance"] == pytest.approx(1.0)
    assert audit["depths"]["2"]["acceptance"] == pytest.approx(0.0)
    assert audit["blocks"]["mean_baseline_tv"] == pytest.approx(0.60)
    assert audit["blocks"]["mean_adaptive_tv"] == pytest.approx(0.45)
    assert audit["blocks"]["mean_recovered_tv"] == pytest.approx(0.20)
    assert audit["blocks"]["mean_spent_tv"] == pytest.approx(0.05)
