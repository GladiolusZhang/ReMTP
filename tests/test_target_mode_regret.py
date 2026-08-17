from __future__ import annotations

import unittest

import torch

from remtp.target_mode_regret import (
    TargetModeRegretConfig,
    target_band_stats,
    target_top1_margin_stats,
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

    def test_top1_margin_stats_are_one_batched_reduction(self) -> None:
        logits = torch.tensor(
            [
                [1.0, 3.0, 2.0],
                [4.0, 1.0, 3.5],
            ]
        )
        top1_ids, margins = target_top1_margin_stats(logits)
        torch.testing.assert_close(top1_ids, torch.tensor([1, 0]))
        torch.testing.assert_close(margins, torch.tensor([1.0, 0.5]))

    def test_top1_margin_rescues_mode_below_half(self) -> None:
        target = torch.tensor([[0.45, 0.30, 0.25]])
        draft = torch.tensor([[0.65, 0.20, 0.15]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([0]),
            self.config(
                eligibility_mode="top1_margin",
                min_top1_margin=0.20,
                debt_margin_slope=0.0,
            ),
            target_top1_ids=torch.tensor([0]),
            target_top1_margins=torch.tensor([0.40]),
        )
        torch.testing.assert_close(result.boosted_candidate_probs, torch.tensor([0.65]))
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.20]))
        self.assertTrue(result.eligibility.item())

    def test_top1_margin_rejects_non_top1_candidate(self) -> None:
        target = torch.tensor([[0.40, 0.35, 0.25]])
        draft = torch.tensor([[0.20, 0.65, 0.15]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([1]),
            self.config(
                eligibility_mode="top1_margin",
                min_top1_margin=0.0,
                debt_margin_slope=0.0,
            ),
            target_top1_ids=torch.tensor([0]),
            target_top1_margins=torch.tensor([0.20]),
        )
        torch.testing.assert_close(result.probs, target)
        torch.testing.assert_close(result.allocated_tv, torch.zeros(1))
        self.assertFalse(result.eligibility.item())

    def test_margin_debt_tightens_later_top1_rescue(self) -> None:
        target = torch.tensor([[0.45, 0.30, 0.25], [0.45, 0.30, 0.25]])
        draft = torch.tensor([[0.65, 0.20, 0.15], [0.65, 0.20, 0.15]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([0, 0]),
            self.config(
                eligibility_mode="top1_margin",
                min_top1_margin=0.10,
                debt_margin_slope=1.0,
            ),
            target_top1_ids=torch.tensor([0, 0]),
            target_top1_margins=torch.tensor([0.40, 0.25]),
        )
        torch.testing.assert_close(
            result.confidence_thresholds,
            torch.tensor([0.10, 0.30]),
        )
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.20, 0.0]))

    def test_target_band_stats_find_rank_and_candidate_gap(self) -> None:
        logits = torch.tensor(
            [
                [3.0, 2.8, 1.0, 0.0],
                [1.0, 4.0, 3.5, 2.0],
            ]
        )
        ranks, gaps, margins = target_band_stats(
            logits,
            torch.tensor([1, 3]),
            max_rank=3,
        )
        torch.testing.assert_close(ranks, torch.tensor([2, 3]))
        torch.testing.assert_close(gaps, torch.tensor([0.2, 2.0]))
        torch.testing.assert_close(margins, torch.tensor([0.2, 0.5]))

    def test_target_band_rescues_close_second_choice(self) -> None:
        target = torch.tensor([[0.40, 0.35, 0.25]])
        draft = torch.tensor([[0.20, 0.65, 0.15]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([1]),
            self.config(
                eligibility_mode="target_band",
                max_target_rank=2,
                max_candidate_log_gap=0.30,
                band_debt_slope=0.0,
            ),
            target_candidate_ranks=torch.tensor([2]),
            target_candidate_log_gaps=torch.tensor([0.13]),
        )
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.20]))
        torch.testing.assert_close(result.boosted_candidate_probs, torch.tensor([0.55]))
        self.assertTrue(result.eligibility.item())

    def test_target_band_rejects_distant_or_low_rank_candidate(self) -> None:
        target = torch.tensor([[0.40, 0.35, 0.25], [0.40, 0.35, 0.25]])
        draft = torch.tensor([[0.20, 0.65, 0.15], [0.20, 0.15, 0.65]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([1, 2]),
            self.config(
                eligibility_mode="target_band",
                max_target_rank=2,
                max_candidate_log_gap=0.30,
                band_debt_slope=0.0,
            ),
            target_candidate_ranks=torch.tensor([2, 3]),
            target_candidate_log_gaps=torch.tensor([0.40, 0.10]),
        )
        torch.testing.assert_close(result.allocated_tv, torch.zeros(2))

    def test_target_band_debt_narrows_later_allowed_gap(self) -> None:
        target = torch.tensor([[0.40, 0.35, 0.25], [0.40, 0.35, 0.25]])
        draft = torch.tensor([[0.20, 0.65, 0.15], [0.20, 0.65, 0.15]])
        result = target_mode_regret_distribution(
            target,
            draft,
            torch.tensor([1, 1]),
            self.config(
                eligibility_mode="target_band",
                max_target_rank=2,
                max_candidate_log_gap=0.30,
                band_debt_slope=1.0,
            ),
            target_candidate_ranks=torch.tensor([2, 2]),
            target_candidate_log_gaps=torch.tensor([0.10, 0.20]),
        )
        torch.testing.assert_close(
            result.confidence_thresholds,
            torch.tensor([0.30, 0.10]),
        )
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.20, 0.0]))


if __name__ == "__main__":
    unittest.main()
