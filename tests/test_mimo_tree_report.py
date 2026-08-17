from __future__ import annotations

from remtp.mimo_tree_report import summarize


def test_tree_native_summary_separates_depth_and_mal() -> None:
    records = [
        {
            "topology": "3+2+1",
            "verification_mode": "strict",
            "accepted_drafts": 2,
            "selected_branch": 0,
            "terminal": "correction",
            "target_forward_calls": 1,
            "target_validation_nodes": 6,
            "target_forward_seconds": 0.01,
            "nodes": [
                {"depth": 1, "status": "ACCEPT"},
                {"depth": 2, "status": "ACCEPT"},
                {"depth": 3, "status": "REJECT"},
            ],
        },
        {
            "topology": "3+2+1",
            "verification_mode": "strict",
            "accepted_drafts": 0,
            "selected_branch": None,
            "terminal": "correction",
            "target_forward_calls": 1,
            "target_validation_nodes": 6,
            "target_forward_seconds": 0.02,
            "nodes": [{"depth": 1, "status": "REJECT"}],
        },
    ]
    result = summarize(records)
    assert result["mean_accepted_depth"] == 1.0
    assert result["mean_acceptance_length"] == 2.0
    assert result["target_forward_calls_per_round"] == 1.0
    assert result["target_nodes_per_round"] == 6.0


def test_summary_reports_one_token_rescue_mal_contribution() -> None:
    records = [
        {
            "topology": "dynamic",
            "verification_mode": "target_path_rescue",
            "accepted_drafts": 3,
            "output_token_ids": [10, 11, 12, 13],
            "selected_branch": 0,
            "terminal": "target-path-extension-bonus",
            "target_forward_calls": 1,
            "target_validation_nodes": 7,
            "nodes": [
                {"depth": 1, "status": "SELECTED"},
                {"depth": 2, "status": "SELECTED"},
                {
                    "depth": 3,
                    "status": "SELECTED",
                    "rescue_eligible": True,
                    "rescued": True,
                    "rescue_extension_tokens": 1,
                },
            ],
        }
    ]
    result = summarize(records)
    assert result["ordinary_mean_accepted_depth"] == 2.0
    assert result["rescue_extension_tokens"] == 1
    assert result["rescue_extension_mal_gain"] == 1.0
    assert result["rescue_unlocked_tokens"] == 1
    assert result["rescue_unlocked_mal_gain"] == 1.0


def test_summary_reports_descendants_unlocked_by_one_rescue() -> None:
    records = [
        {
            "topology": "dynamic",
            "verification_mode": "prefix_reopen_rescue",
            "accepted_drafts": 3,
            "output_token_ids": [10, 11, 12, 13],
            "selected_branch": 1,
            "terminal": "prefix-reopen-rescue-bonus",
            "target_forward_calls": 1,
            "target_validation_nodes": 8,
            "nodes": [
                {
                    "depth": 1,
                    "status": "SELECTED",
                    "rescue_eligible": True,
                    "rescued": True,
                    "rescue_unlocked_tokens": 3,
                    "target_log_margin": 0.4,
                    "rescue_margin_multiplier": 1.2,
                },
                {"depth": 2, "status": "SELECTED"},
                {"depth": 3, "status": "SELECTED"},
            ],
        }
    ]
    result = summarize(records)
    assert result["mean_accepted_depth"] == 3.0
    assert result["pre_rescue_mean_accepted_depth"] == 0.0
    assert result["rescue_unlocked_tokens"] == 3
    assert result["rescue_unlocked_mal_gain"] == 3.0
    assert result["mean_selected_rescue_target_log_margin"] == 0.4
    assert result["mean_selected_rescue_margin_multiplier"] == 1.2
