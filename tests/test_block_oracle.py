from __future__ import annotations

import unittest

from remtp.block_oracle import (
    analyze_records,
    dynamic_mal_allocation,
    expected_mal,
    replay_mal,
)


def _record() -> dict[str, object]:
    return {
        "dataset": "toy",
        "draft_tokens": [10, 11, 12],
        "accepted_drafts": 0,
        "previous_debt": 0.0,
        "p_y": [0.20, 0.40, 0.40],
        "q_y": [0.50, 0.50, 0.50],
        "strict_acceptance": [0.40, 0.80, 0.80],
        "relaxed_acceptance": [0.60, 0.80, 0.80],
        "allocated_tv": [0.10, 0.00, 0.00],
        "desired_tv": [0.30, 0.10, 0.10],
        "local_risk": [0.5, 0.2, 0.2],
        "candidate_rank": [2, 1, 1],
        "candidate_gap": [0.5, 0.0, 0.0],
        "target_margin": [0.1, 0.8, 0.8],
        "threshold": [1.9, 1.3, 1.3],
        "threshold_no_debt": [1.9, 1.3, 1.3],
        "target_head_mass": [0.5, 0.6, 0.6],
        "local_eligible": [True, True, True],
        "sentinel_pass": [True, True, True],
        "sentinel_checks": [3, 3, 3],
        "sentinel_strength": [1.0, 1.0, 1.0],
        "uniforms": [0.50, 0.50, 0.50],
        "q_top_ids": [[10, 20], [11, 21], [12, 22]],
        "q_top_probs": [[0.50, 0.25], [0.50, 0.25], [0.50, 0.25]],
        "p_at_q_top": [[0.20, 0.20], [0.40, 0.10], [0.40, 0.10]],
        "p_top_ids": [[30, 10], [11, 31], [12, 32]],
        "p_top_probs": [[0.30, 0.20], [0.40, 0.30], [0.40, 0.30]],
        "config": {
            "max_rank": 8,
            "rank_limit": 8,
            "base_gap": 2.15,
            "gap_floor": 1.15,
            "margin_temperature": 0.75,
            "risky_rank": 4,
            "risky_gap": 1.2,
            "risky_min_checks": 2,
            "sentinel_min_checks": 1,
            "sentinel_soft_floor": 0.7,
            "cactus_delta": 1.25,
            "per_token_tv": 0.65,
            "block_tv": 0.50,
            "risk_budget": 5.50,
            "debt_scale": 0.20,
            "effective_block_tv": 0.50,
            "effective_risk_budget": 5.50,
        },
    }


class BlockOracleTest(unittest.TestCase):
    def test_expected_and_same_uniform_mal(self) -> None:
        acceptance = [0.5, 0.8, 1.0]
        self.assertAlmostEqual(expected_mal(acceptance), 2.3)
        self.assertEqual(replay_mal(acceptance, [0.4, 0.9, 0.1]), 2.0)

    def test_dynamic_solver_targets_the_prefix_bottleneck(self) -> None:
        allocation = dynamic_mal_allocation(
            p_y=[0.20, 0.40, 0.40],
            q_y=[0.50, 0.50, 0.50],
            capacity=[0.30, 0.10, 0.10],
            eligible=[True, True, True],
            block_budget=0.20,
            local_risk=[0.5, 0.2, 0.2],
            risk_budget=1.0,
            steps=128,
        )

        self.assertGreater(allocation[0], allocation[2])
        self.assertLessEqual(sum(allocation), 0.20 + 1e-9)

    def test_analysis_reports_candidate_and_allocation_oracles(self) -> None:
        result = analyze_records([_record()])

        self.assertEqual(result["rounds"], 1)
        self.assertIn("joint_optimizer", result["scenarios"])
        self.assertIn("gate_ceiling", result["scenarios"])
        self.assertEqual(
            result["top2_candidate_oracle"]["alternative_accepts"], 1
        )


if __name__ == "__main__":
    unittest.main()
