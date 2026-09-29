from __future__ import annotations

import json
from pathlib import Path

import pytest

from remtp.raven_depth_relaxation_report import _depth_audit
from remtp.dynamic_tree_vllm import _parse_relaxation_depth_weights


def test_depth_audit_uses_primary_nodes_and_records_effective_delta(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "tree_rounds.jsonl"
    rounds = [
        {
            "nodes": [
                {
                    "proposal_role": "sampled_cactus_primary",
                    "trunk_position": 1,
                    "trunk_accepted": True,
                    "relaxation_depth_weight": 1.0,
                    "effective_relaxation_delta": 1.0,
                    "relaxation_tv": 0.2,
                },
                {
                    "proposal_role": "sampled_recovery_continuation",
                    "trunk_position": 2,
                    "normal_survival": True,
                    "relaxation_depth_weight": 0.75,
                    "effective_relaxation_delta": 0.75,
                    "relaxation_tv": 0.1,
                },
            ]
        },
        {
            "nodes": [
                {
                    "proposal_role": "sampled_cactus_primary",
                    "trunk_position": 1,
                    "trunk_accepted": False,
                    "relaxation_depth_weight": 1.0,
                    "effective_relaxation_delta": 1.0,
                    "relaxation_tv": 0.1,
                }
            ]
        },
    ]
    audit.write_text(
        "\n".join(json.dumps(row) for row in rounds) + "\n",
        encoding="utf-8",
    )

    result = _depth_audit(audit)

    assert set(result) == {"1"}
    assert result["1"]["seen"] == 2
    assert result["1"]["acceptance"] == pytest.approx(0.5)
    assert result["1"]["mean_effective_delta"] == pytest.approx(1.0)
    assert result["1"]["mean_relaxation_tv"] == pytest.approx(0.15)


def test_depth_weight_environment_parser_is_explicit() -> None:
    assert _parse_relaxation_depth_weights("") == ()
    assert _parse_relaxation_depth_weights("1.0, 0.862,0.763") == (
        1.0,
        0.862,
        0.763,
    )
    with pytest.raises(ValueError, match="comma-separated"):
        _parse_relaxation_depth_weights("1.0,broken")
