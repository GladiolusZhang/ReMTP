from __future__ import annotations

import unittest

import torch

from remtp.prefix_credit_mtp import (
    PrefixCreditConfig,
    prefix_credit_distribution,
    target_top1_margin_stats,
)


class PrefixCreditMTPTest(unittest.TestCase):
    def config(self, **overrides: object) -> PrefixCreditConfig:
        values: dict[str, object] = {
            "expected_draft_tokens": 2,
            "min_top1_margin": 0.0,
            "per_token_tv_cap": 0.60,
            "block_tv_cap": 0.60,
            "compile_fast_path": False,
        }
        values.update(overrides)
        return PrefixCreditConfig(**values)

    @staticmethod
    def example() -> tuple[torch.Tensor, ...]:
        # Row 0 creates prefix survival 0.5 and is intentionally ineligible.
        # Row 1 is a confident target top-1 candidate. Token-wise saturation
        # needs only 0.05 TV, while prefix saturation needs 0.50 TV to repair
        # the earlier deficit completely.
        target = torch.tensor(
            [
                [0.40, 0.60, 0.00],
                [0.40, 0.35, 0.25],
            ]
        )
        draft = torch.tensor(
            [
                [0.80, 0.20, 0.00],
                [0.45, 0.30, 0.25],
            ]
        )
        ids = torch.tensor([0, 0])
        top1 = torch.tensor([1, 0])
        margins = torch.tensor([0.20, 0.10])
        return target, draft, ids, top1, margins

    def test_prefix_credit_uses_over_q_mass_to_repair_earlier_debt(self) -> None:
        target, draft, ids, top1, margins = self.example()
        result = prefix_credit_distribution(
            target,
            draft,
            ids,
            top1,
            margins,
            self.config(),
        )
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.0, 0.50]))
        torch.testing.assert_close(result.boosted_candidate_probs, torch.tensor([0.40, 0.90]))
        torch.testing.assert_close(result.over_q_credit, torch.tensor([0.0, 0.45]))
        torch.testing.assert_close(result.prefix_survival_after, torch.tensor([0.5, 1.0]))

    def test_token_cap_cannot_repair_earlier_prefix_debt(self) -> None:
        target, draft, ids, top1, margins = self.example()
        result = prefix_credit_distribution(
            target,
            draft,
            ids,
            top1,
            margins,
            self.config(allocation_mode="token_cap"),
        )
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.0, 0.05]))
        torch.testing.assert_close(result.boosted_candidate_probs, torch.tensor([0.40, 0.45]))
        torch.testing.assert_close(result.prefix_survival_after, torch.tensor([0.5, 0.5]))

    def test_exact_block_tv_cap_is_respected(self) -> None:
        target, draft, ids, top1, margins = self.example()
        result = prefix_credit_distribution(
            target,
            draft,
            ids,
            top1,
            margins,
            self.config(block_tv_cap=0.20, allocation_mode="atomic_credit"),
        )
        # The 0.20 budget cannot pay the 0.45 over-q repair in full. The
        # method spends only the useful 0.05 local token saturation and does
        # not partially fund an ineffective prefix repair.
        torch.testing.assert_close(result.allocated_tv.sum(), torch.tensor(0.05))
        torch.testing.assert_close(result.prefix_survival_after[-1], torch.tensor(0.5))

    def test_per_position_cap_prevents_partial_prefix_repair(self) -> None:
        target, draft, ids, top1, margins = self.example()
        result = prefix_credit_distribution(
            target,
            draft,
            ids,
            top1,
            margins,
            self.config(
                per_token_tv_cap=0.30,
                allocation_mode="atomic_credit",
            ),
        )
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.0, 0.05]))
        torch.testing.assert_close(result.over_q_credit, torch.zeros(2))

    def test_efficiency_floor_allows_useful_partial_credit(self) -> None:
        target, draft, ids, top1, margins = self.example()
        result = prefix_credit_distribution(
            target,
            draft,
            ids,
            top1,
            margins,
            self.config(per_token_tv_cap=0.30, min_prefix_gain_per_tv=1.0),
        )
        # a/q=0.5/0.45 > 1, so the available partial payment is useful.
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.0, 0.30]))
        torch.testing.assert_close(result.over_q_credit, torch.tensor([0.0, 0.25]))
        torch.testing.assert_close(
            result.prefix_survival_after[-1],
            torch.tensor(7.0 / 9.0),
        )

    def test_efficiency_floor_rejects_low_gain_partial_credit(self) -> None:
        target, draft, ids, top1, margins = self.example()
        result = prefix_credit_distribution(
            target,
            draft,
            ids,
            top1,
            margins,
            self.config(per_token_tv_cap=0.30, min_prefix_gain_per_tv=1.2),
        )
        torch.testing.assert_close(result.allocated_tv, torch.tensor([0.0, 0.05]))
        torch.testing.assert_close(result.over_q_credit, torch.zeros(2))

    def test_only_confident_target_top1_is_a_credit_destination(self) -> None:
        target, draft, ids, top1, margins = self.example()
        result = prefix_credit_distribution(
            target,
            draft,
            ids,
            top1,
            torch.tensor([0.20, 0.09]),
            self.config(min_top1_margin=0.10),
        )
        torch.testing.assert_close(result.allocated_tv, torch.zeros(2))

    def test_distribution_is_normalized_and_preserves_other_ratios(self) -> None:
        target, draft, ids, top1, margins = self.example()
        result = prefix_credit_distribution(
            target,
            draft,
            ids,
            top1,
            margins,
            self.config(),
        )
        torch.testing.assert_close(result.probs.sum(dim=-1), torch.ones(2))
        original_ratio = target[1, 1] / target[1, 2]
        relaxed_ratio = result.probs[1, 1] / result.probs[1, 2]
        torch.testing.assert_close(relaxed_ratio, original_ratio)

    def test_zero_budget_is_target_identity(self) -> None:
        target, draft, ids, top1, margins = self.example()
        result = prefix_credit_distribution(
            target,
            draft,
            ids,
            top1,
            margins,
            self.config(block_tv_cap=0.0),
        )
        torch.testing.assert_close(result.probs, target)

    def test_natural_target_surplus_is_not_reported_as_new_credit(self) -> None:
        result = prefix_credit_distribution(
            torch.tensor([[0.80, 0.20]]),
            torch.tensor([[0.60, 0.40]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1.0]),
            self.config(expected_draft_tokens=1),
        )
        torch.testing.assert_close(result.allocated_tv, torch.zeros(1))
        torch.testing.assert_close(result.over_q_credit, torch.zeros(1))

    def test_target_top1_margin_stats(self) -> None:
        ids, margins = target_top1_margin_stats(
            torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 3.5]])
        )
        torch.testing.assert_close(ids, torch.tensor([1, 0]))
        torch.testing.assert_close(margins, torch.tensor([1.0, 0.5]))

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.config(allocation_mode="unknown").validate()

    def test_negative_efficiency_floor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.config(min_prefix_gain_per_tv=-0.1).validate()


if __name__ == "__main__":
    unittest.main()
