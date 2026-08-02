import unittest

import torch

from remtp.target_anchored_mtp import (
    TargetAnchoredConfig,
    aligned_hidden_cosine,
    boost_candidate_logits,
    cactus_tv_increment,
    target_anchored_candidate_probs,
    target_anchored_distribution,
)


class TargetAnchoredMTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = torch.tensor(
            [
                [0.55, 0.30, 0.10, 0.05],
                [0.50, 0.35, 0.10, 0.05],
                [0.45, 0.35, 0.15, 0.05],
                [0.40, 0.35, 0.20, 0.05],
            ],
            dtype=torch.float32,
        )
        self.draft = torch.tensor(
            [
                [0.40, 0.45, 0.10, 0.05],
                [0.35, 0.50, 0.10, 0.05],
                [0.30, 0.50, 0.15, 0.05],
                [0.25, 0.50, 0.20, 0.05],
            ],
            dtype=torch.float32,
        )
        self.ids = torch.tensor([1, 1, 1, 1])

    @staticmethod
    def config(
        variant: str = "tv_head",
        **overrides: object,
    ) -> TargetAnchoredConfig:
        values: dict[str, object] = {
            "variant": variant,
            "cactus_delta": 0.01,
            "expected_draft_tokens": 4,
            "head_reliability": (1.0, 0.85, 0.70, 0.55),
            "max_target_log_gap": 100.0,
        }
        values.update(overrides)
        return TargetAnchoredConfig(**values)

    def test_cactus_tv_is_exact_candidate_mass_shift(self) -> None:
        selected = torch.tensor([0.08, 0.20, 0.01, 0.06])
        shift = cactus_tv_increment(selected, delta=1.0)
        boosted = selected + shift
        self.assertTrue(torch.all(boosted <= 1.0).item())
        torch.testing.assert_close(shift, boosted - selected)

    def test_cactus_cap_stops_at_draft_probability(self) -> None:
        result = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config("cactus_cap", cactus_delta=1.0),
        )
        torch.testing.assert_close(
            result.boosted_candidate_probs,
            result.draft_candidate_probs,
        )
        torch.testing.assert_close(
            result.allocated_tv,
            result.useful_tv_capacity,
        )

    def test_no_boost_when_strict_acceptance_is_saturated(self) -> None:
        draft = self.draft.clone()
        draft[0, 1] = 0.20
        draft[0, 0] += 0.25
        result = target_anchored_distribution(
            self.target,
            draft,
            self.ids,
            self.config("tv_head"),
        )
        self.assertEqual(result.strict_acceptance[0].item(), 1.0)
        self.assertEqual(result.useful_tv_capacity[0].item(), 0.0)
        self.assertEqual(result.allocated_tv[0].item(), 0.0)

    def test_block_allocation_never_exceeds_cactus_tv_budget(self) -> None:
        result = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config("tv_head"),
        )
        self.assertLessEqual(
            result.allocated_tv.sum().item(),
            result.cactus_tv.sum().item() + 1e-6,
        )
        self.assertTrue(
            torch.all(
                result.allocated_tv
                <= result.useful_tv_capacity + 1e-6
            ).item()
        )

    def test_redistribution_recovers_capped_cactus_waste(self) -> None:
        target = torch.tensor(
            [
                [0.80, 0.20],
                [0.80, 0.20],
                [0.80, 0.20],
                [0.80, 0.20],
            ]
        )
        draft = torch.tensor(
            [
                [0.79, 0.21],
                [0.60, 0.40],
                [0.60, 0.40],
                [0.60, 0.40],
            ]
        )
        capped = target_anchored_distribution(
            target,
            draft,
            self.ids,
            self.config("cactus_cap", cactus_delta=0.01),
        )
        redistributed = target_anchored_distribution(
            target,
            draft,
            self.ids,
            self.config("tv_head", cactus_delta=0.01),
        )
        self.assertGreater(
            redistributed.allocated_tv.sum().item(),
            capped.allocated_tv.sum().item(),
        )

    def test_head_reliability_changes_budget_priority(self) -> None:
        config = self.config(
            "tv_head",
            cactus_delta=0.001,
            head_reliability=(1.0, 0.5, 0.25, 0.1),
            use_prefix_value=False,
        )
        result = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            config,
        )
        self.assertGreater(result.priority[0], result.priority[-1])
        self.assertGreater(result.allocated_tv[0], result.allocated_tv[-1])

    def test_future_signal_is_veto_only(self) -> None:
        target = self.target.clone()
        target[2] = torch.tensor([0.989, 0.001, 0.005, 0.005])
        result = target_anchored_distribution(
            target,
            self.draft,
            self.ids,
            self.config("tv_hidden_veto", max_target_log_gap=100.0),
            hidden_similarity=torch.ones(4),
        )
        self.assertTrue(torch.all(result.future_veto <= 1.0).item())
        self.assertEqual(result.future_veto[-1].item(), 1.0)
        self.assertLess(result.future_veto[0].item(), 0.5)
        self.assertLess(result.future_veto[1].item(), 0.5)

    def test_future_good_support_never_adds_a_reward(self) -> None:
        result = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config("tv_hidden_veto"),
            hidden_similarity=torch.ones(4),
        )
        base = (
            result.strict_acceptance.sqrt()
            * (1.0 - result.strict_acceptance)
            * result.prefix_value
            * result.target_support
            * result.head_reliability
        )
        self.assertTrue(torch.all(result.priority <= base + 1e-6).item())

    def test_hidden_cosine_downweights_inconsistent_position(self) -> None:
        result = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(
                "tv_hidden_veto",
                cactus_delta=0.001,
                head_reliability=(1.0, 1.0, 1.0, 1.0),
                use_prefix_value=False,
            ),
            hidden_similarity=torch.tensor([0.0, 1.0, 1.0, 1.0]),
        )
        self.assertLess(result.priority[0], result.priority[1])

    def test_debt_control_bounds_position_and_block_acceptance_uplift(
        self,
    ) -> None:
        result = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(
                "tv_debt_control",
                cactus_delta=1.0,
                debt_position_limit=0.15,
                debt_block_limit=0.30,
                debt_soft_log_gap=10.0,
                debt_hard_log_gap=20.0,
                debt_max_position_tv=1.0,
                debt_max_cactus_ratio=10.0,
            ),
        )
        self.assertTrue(
            torch.all(result.acceptance_residual <= 0.15 + 1e-6).item()
        )
        self.assertLessEqual(result.cumulative_debt[-1].item(), 0.30 + 1e-6)
        self.assertTrue(
            torch.all(result.allocated_tv <= result.risk_capacity + 1e-6).item()
        )
        self.assertTrue(
            torch.all(
                result.allocated_tv <= result.raw_allocated_tv + 1e-6
            ).item()
        )

    def test_debt_control_recycles_budget_after_risk_caps(self) -> None:
        result = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(
                "tv_debt_control",
                cactus_delta=0.01,
                debt_position_limit=0.20,
                debt_block_limit=1.0,
                debt_soft_log_gap=10.0,
                debt_hard_log_gap=20.0,
                debt_max_position_tv=1.0,
                debt_max_cactus_ratio=10.0,
            ),
        )
        self.assertGreater(result.allocated_tv.sum().item(), 0.0)
        self.assertLessEqual(
            result.allocated_tv.sum().item(),
            result.cactus_tv.sum().item() + 1e-6,
        )

    def test_hard_target_opposition_stops_suffix_relaxation(self) -> None:
        target = self.target.clone()
        target[1] = torch.tensor([0.999, 0.0001, 0.0005, 0.0004])
        result = target_anchored_distribution(
            target,
            self.draft,
            self.ids,
            self.config(
                "tv_debt_control",
                cactus_delta=1.0,
                max_target_log_gap=100.0,
                debt_soft_log_gap=2.0,
                debt_hard_log_gap=4.0,
                debt_max_position_tv=1.0,
                debt_max_cactus_ratio=10.0,
                debt_fallback="strict",
            ),
        )
        self.assertEqual(result.allocated_tv[1].item(), 0.0)
        self.assertTrue(torch.all(result.allocated_tv[2:] == 0.0).item())
        self.assertTrue(result.fallback_mask[1].item())
        self.assertTrue(torch.all(result.stopped_mask[2:]).item())

    def test_cactus_fallback_is_available_as_an_ablation(self) -> None:
        target = self.target.clone()
        target[1] = torch.tensor([0.999, 0.0001, 0.0005, 0.0004])
        result = target_anchored_distribution(
            target,
            self.draft,
            self.ids,
            self.config(
                "tv_debt_control",
                cactus_delta=0.01,
                max_target_log_gap=100.0,
                debt_soft_log_gap=2.0,
                debt_hard_log_gap=4.0,
                debt_max_position_tv=1.0,
                debt_max_cactus_ratio=10.0,
                debt_fallback="cactus_cap",
            ),
        )
        expected = torch.minimum(result.cactus_tv, result.useful_tv_capacity)
        torch.testing.assert_close(result.allocated_tv[1:], expected[1:])
        self.assertTrue(torch.all(result.fallback_mask[1:]).item())

    def test_loose_debt_limits_recover_tv_head_allocation(self) -> None:
        baseline = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config("tv_head", cactus_delta=0.01),
        )
        debt = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(
                "tv_debt_control",
                cactus_delta=0.01,
                debt_position_limit=1.0,
                debt_block_limit=4.0,
                debt_soft_log_gap=100.0,
                debt_hard_log_gap=101.0,
                debt_max_position_tv=1.0,
                debt_max_cactus_ratio=100.0,
            ),
        )
        torch.testing.assert_close(
            debt.allocated_tv,
            baseline.allocated_tv,
        )

    def test_aligned_hidden_cosine_and_fallback(self) -> None:
        draft = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        cosine, available = aligned_hidden_cosine(
            draft,
            target,
            2,
            device=torch.device("cpu"),
        )
        self.assertTrue(available)
        torch.testing.assert_close(cosine, torch.tensor([1.0, 0.0]))
        fallback, available = aligned_hidden_cosine(
            None,
            None,
            2,
            device=torch.device("cpu"),
        )
        self.assertFalse(available)
        torch.testing.assert_close(fallback, torch.ones(2))

    def test_extreme_target_gap_gets_no_budget(self) -> None:
        target = self.target.clone()
        target[0] = torch.tensor([0.999, 0.0001, 0.0005, 0.0004])
        result = target_anchored_distribution(
            target,
            self.draft,
            self.ids,
            self.config("tv_head", max_target_log_gap=4.0),
        )
        self.assertEqual(result.target_support[0].item(), 0.0)
        self.assertEqual(result.priority[0].item(), 0.0)
        self.assertEqual(result.allocated_tv[0].item(), 0.0)

    def test_non_candidate_ratios_are_preserved(self) -> None:
        result = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config("tv_head"),
        )
        original_ratio = self.target[0, 0] / self.target[0, 2]
        relaxed_ratio = result.probs[0, 0] / result.probs[0, 2]
        torch.testing.assert_close(original_ratio, relaxed_ratio)
        torch.testing.assert_close(result.probs.sum(dim=-1), torch.ones(4))

    def test_logit_update_reconstructs_full_distribution(self) -> None:
        result = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config("tv_head"),
        )
        logits = boost_candidate_logits(
            torch.log(self.target),
            self.ids,
            result.target_candidate_probs,
            result.boosted_candidate_probs,
        )
        torch.testing.assert_close(torch.softmax(logits, dim=-1), result.probs)

    def test_candidate_fast_path_matches_detailed_path(self) -> None:
        config = self.config("tv_hidden_veto")
        hidden = torch.tensor([0.8, 0.7, 0.6, 0.5])
        detailed = target_anchored_distribution(
            self.target,
            self.draft,
            self.ids,
            config,
            hidden_similarity=hidden,
            assume_normalized=True,
            construct_probs=False,
        )
        p_y, h_y = target_anchored_candidate_probs(
            self.target,
            self.draft,
            self.ids,
            hidden,
            config,
        )
        self.assertIsNone(detailed.probs)
        torch.testing.assert_close(p_y, detailed.target_candidate_probs)
        torch.testing.assert_close(h_y, detailed.boosted_candidate_probs)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            TargetAnchoredConfig(variant="js").validate()
        with self.assertRaises(ValueError):
            TargetAnchoredConfig(head_reliability=(1.0,)).validate()
        with self.assertRaises(ValueError):
            TargetAnchoredConfig(debt_position_limit=1.1).validate()
        with self.assertRaises(ValueError):
            TargetAnchoredConfig(
                debt_soft_log_gap=2.0,
                debt_hard_log_gap=2.0,
            ).validate()
        with self.assertRaises(ValueError):
            TargetAnchoredConfig(debt_fallback="unsafe").validate()
    def test_default_config_covers_six_mtp_heads(self) -> None:
        config = TargetAnchoredConfig()
        config.validate()
        self.assertEqual(config.expected_draft_tokens, 6)
        self.assertEqual(len(config.head_reliability), 6)

        target = torch.softmax(torch.randn(6, 8), dim=-1)
        draft = torch.softmax(torch.randn(6, 8), dim=-1)
        result = target_anchored_distribution(
            target,
            draft,
            torch.arange(6),
            config,
            hidden_similarity=torch.ones(6),
        )
        self.assertEqual(result.allocated_tv.shape[0], 6)
        self.assertTrue(
            torch.all(
                result.boosted_candidate_probs
                <= result.draft_candidate_probs.maximum(
                    result.target_candidate_probs
                )
                + 1e-6
            ).item()
        )


if __name__ == "__main__":
    unittest.main()
