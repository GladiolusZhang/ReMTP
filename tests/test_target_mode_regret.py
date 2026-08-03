from __future__ import annotations

import unittest

import torch

from remtp.target_mode_regret import (
    TargetModeRegretConfig,
    target_mode_regret_distribution,
)


class TargetModeRegretTest(unittest.TestCase):
    def config(self, **overrides: object) -> TargetModeRegretConfig:
        values: dict[str, object] = {
            "min_target_prob": 0.5,
            "per_token_tv_cap": 0.2,
            "block_tv_cap": 0.35,
            "debt_confidence_slope": 0.0,
            "depth_confidence_slope": 0.0,
        }
        values.update(overrides)
        return TargetModeRegretConfig(**values)

    def test_high_confidence_target_mode_saturates_at_q(self) -> None:
        target = torch.tensor([[0.70, 0.30]])
        draft = torch.tensor([[0.90, 0.10]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([0]),
            self.config(),
        )

        torch.testing.assert_close(result.probs[0, 0], torch.tensor(0.90))
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.20]))

    def test_low_confidence_candidate_is_strict_identity(self) -> None:
        target = torch.tensor([[0.49, 0.51]])
        draft = torch.tensor([[0.80, 0.20]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([0]),
            self.config(),
        )

        torch.testing.assert_close(result.probs, target)
        torch.testing.assert_close(result.allocated_tv, torch.zeros(1))

    def test_no_budget_is_spent_when_strict_acceptance_is_one(self) -> None:
        target = torch.tensor([[0.80, 0.20]])
        draft = torch.tensor([[0.60, 0.40]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([0]),
            self.config(),
        )
        torch.testing.assert_close(result.allocated_tv, torch.zeros(1))
        torch.testing.assert_close(result.boosted_candidate_probs, torch.tensor([0.8]))

    def test_zero_block_budget_is_exact_target_identity(self) -> None:
        target = torch.tensor([[0.70, 0.20, 0.10]])
        draft = torch.tensor([[0.99, 0.005, 0.005]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([0]),
            self.config(block_tv_cap=0.0),
        )
        torch.testing.assert_close(result.probs, target)
        torch.testing.assert_close(result.allocated_tv, torch.zeros(1))

    def test_block_cap_reclaims_excess_tail_budget(self) -> None:
        target = torch.tensor([[0.70, 0.30], [0.70, 0.30]])
        draft = torch.tensor([[0.95, 0.05], [0.95, 0.05]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([0, 0]),
            self.config(block_tv_cap=0.25),
        )
        torch.testing.assert_close(
            result.allocated_tv,
            torch.tensor([0.20, 0.05]),
        )

    def test_regret_debt_tightens_later_confidence(self) -> None:
        target = torch.tensor([[0.70, 0.30], [0.60, 0.40]])
        draft = torch.tensor([[0.90, 0.10], [0.90, 0.10]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([0, 0]),
            self.config(debt_confidence_slope=1.0),
        )
        self.assertAlmostEqual(result.confidence_thresholds[1].item(), 0.70)
        self.assertEqual(result.allocated_tv[1].item(), 0.0)

    def test_invalid_non_mode_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.config(min_target_prob=0.49).validate()


if __name__ == "__main__":
    unittest.main()
