import unittest

import torch

from remtp.gated_depth_kl_mtp import (
    GatedDepthKLConfig,
    bernoulli_kl,
    boost_candidate_logits,
    gated_depth_kl_distribution,
    greedy_verification_ids,
)


class GatedDepthKLMTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GatedDepthKLConfig()

    def test_zero_epsilon_recovers_standard_target(self) -> None:
        target = torch.tensor([[0.6, 0.3, 0.1]])
        result = gated_depth_kl_distribution(
            target,
            torch.tensor([1]),
            torch.tensor([1]),
            GatedDepthKLConfig(epsilon_0=0.0),
        )
        torch.testing.assert_close(result.probs, target)
        self.assertEqual(result.epsilon.item(), 0.0)

    def test_safe_top_candidate_is_boosted_within_budget(self) -> None:
        target = torch.tensor([[0.6, 0.3, 0.1]])
        result = gated_depth_kl_distribution(
            target,
            torch.tensor([0]),
            torch.tensor([1]),
            self.config,
        )
        self.assertTrue(result.safe.item())
        self.assertGreater(
            result.boosted_candidate_probs.item(),
            result.original_candidate_probs.item(),
        )
        self.assertLessEqual(
            result.realized_kl.item(),
            result.epsilon.item() + 1e-6,
        )

    def test_depth_decay_applies_to_maximum_budget(self) -> None:
        target = torch.tensor(
            [
                [0.6, 0.3, 0.1],
                [0.6, 0.3, 0.1],
            ]
        )
        result = gated_depth_kl_distribution(
            target,
            torch.tensor([0, 0]),
            torch.tensor([1, 2]),
            self.config,
        )
        torch.testing.assert_close(
            result.epsilon,
            torch.tensor([0.02, 0.014]),
        )

    def test_each_safety_gate_can_disable_relaxation(self) -> None:
        target = torch.tensor(
            [
                [0.7, 0.2, 0.09, 0.01],
                [0.7, 0.2, 0.099, 0.001],
                [0.9, 0.05, 0.03, 0.02],
            ]
        )
        candidates = torch.tensor([3, 3, 2])
        configs = (
            GatedDepthKLConfig(max_rank=3),
            GatedDepthKLConfig(min_target_prob=0.005),
            GatedDepthKLConfig(max_log_gap=1.0),
        )
        for row, config in enumerate(configs):
            result = gated_depth_kl_distribution(
                target[row : row + 1],
                candidates[row : row + 1],
                torch.tensor([1]),
                config,
            )
            self.assertFalse(result.safe.item())
            torch.testing.assert_close(
                result.probs,
                target[row : row + 1],
            )

    def test_non_candidate_probabilities_keep_relative_ratios(self) -> None:
        target = torch.tensor([[0.5, 0.3, 0.2]])
        result = gated_depth_kl_distribution(
            target,
            torch.tensor([1]),
            torch.tensor([1]),
            self.config,
        )
        original_ratio = target[0, 0] / target[0, 2]
        relaxed_ratio = result.probs[0, 0] / result.probs[0, 2]
        torch.testing.assert_close(relaxed_ratio, original_ratio)
        torch.testing.assert_close(
            result.probs.sum(dim=-1),
            torch.ones(1),
        )

    def test_logit_boost_constructs_the_same_relaxed_distribution(self) -> None:
        target = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.7, 0.2, 0.1],
            ]
        )
        candidates = torch.tensor([1, 0])
        result = gated_depth_kl_distribution(
            target,
            candidates,
            torch.tensor([1, 2]),
            self.config,
        )
        relaxed_logits = boost_candidate_logits(
            torch.log(target),
            candidates,
            result.original_candidate_probs,
            result.boosted_candidate_probs,
        )
        torch.testing.assert_close(
            torch.softmax(relaxed_logits, dim=-1),
            result.probs,
        )

    def test_probability_ratio_cap_is_respected(self) -> None:
        target = torch.tensor([[0.99, 0.01]])
        config = GatedDepthKLConfig(
            epsilon_0=10.0,
            max_prob_ratio=2.0,
            max_log_gap=10.0,
        )
        result = gated_depth_kl_distribution(
            target,
            torch.tensor([1]),
            torch.tensor([1]),
            config,
        )
        self.assertLessEqual(
            result.boosted_candidate_probs.item(),
            0.02 + 1e-6,
        )

    def test_bernoulli_kl_matches_full_distribution_kl(self) -> None:
        target = torch.tensor([[0.5, 0.3, 0.2]])
        result = gated_depth_kl_distribution(
            target,
            torch.tensor([1]),
            torch.tensor([1]),
            self.config,
        )
        full_kl = (
            result.probs
            * (
                torch.log(result.probs)
                - torch.log(target)
            )
        ).sum()
        binary = bernoulli_kl(
            result.boosted_candidate_probs,
            result.original_candidate_probs,
        ).sum()
        torch.testing.assert_close(full_kl, binary)

    def test_bernoulli_kl_is_finite_at_probability_one(self) -> None:
        value = bernoulli_kl(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
        )
        self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(value, torch.zeros(1))

    def test_greedy_rejects_to_original_target_top1(self) -> None:
        target = torch.tensor(
            [
                [0.6, 0.3, 0.1],
                [0.5, 0.4, 0.1],
            ]
        )
        relaxed = torch.tensor(
            [
                [0.4, 0.5, 0.1],
                [0.5, 0.4, 0.1],
            ]
        )
        ids = greedy_verification_ids(
            target,
            relaxed,
            torch.tensor([1, 1]),
        )
        torch.testing.assert_close(ids, torch.tensor([1, 0]))

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GatedDepthKLConfig(depth_decay=1.1).validate()


if __name__ == "__main__":
    unittest.main()
