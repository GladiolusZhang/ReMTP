import unittest

import torch

from remtp.block_feature_mtp import (
    BlockFeatureConfig,
    bernoulli_kl,
    block_feature_distribution,
    boost_candidate_logits,
    greedy_verification_ids,
    projected_logit_consistency,
)


class BlockFeatureMTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = torch.tensor(
            [
                [0.55, 0.30, 0.10, 0.05],
                [0.50, 0.35, 0.10, 0.05],
                [0.45, 0.35, 0.15, 0.05],
                [0.40, 0.35, 0.20, 0.05],
            ]
        )
        self.draft = torch.tensor(
            [
                [0.40, 0.45, 0.10, 0.05],
                [0.35, 0.50, 0.10, 0.05],
                [0.30, 0.50, 0.15, 0.05],
                [0.25, 0.50, 0.20, 0.05],
            ]
        )
        self.ids = torch.tensor([1, 1, 1, 1])

    def test_zero_budget_recovers_standard_target(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            BlockFeatureConfig(block_kl_budget=0.0),
        )
        torch.testing.assert_close(result.probs, self.target)
        torch.testing.assert_close(
            result.allocated_kl,
            torch.zeros(4),
        )

    def test_total_kl_respects_single_block_budget(self) -> None:
        config = BlockFeatureConfig(
            block_kl_budget=0.02,
            max_normalized_surprisal=100.0,
            min_feature_consistency=0.0,
        )
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            config,
        )
        self.assertLessEqual(result.realized_kl.sum().item(), 0.020001)
        self.assertGreater(result.realized_kl.sum().item(), 0.0)

    def test_boost_never_exceeds_draft_candidate_probability(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            BlockFeatureConfig(
                block_kl_budget=100.0,
                max_normalized_surprisal=100.0,
                min_feature_consistency=0.0,
            ),
        )
        self.assertTrue(
            torch.all(
                result.boosted_candidate_probs
                <= result.draft_candidate_probs + 1e-6
            )
        )
        torch.testing.assert_close(
            result.boosted_candidate_probs,
            result.draft_candidate_probs,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_candidate_with_q_below_p_is_not_modified(self) -> None:
        draft = self.draft.clone()
        draft[0] = torch.tensor([0.70, 0.20, 0.05, 0.05])
        result = block_feature_distribution(
            self.target,
            draft,
            self.ids,
            BlockFeatureConfig(
                block_kl_budget=1.0,
                min_feature_consistency=0.0,
            ),
        )
        self.assertEqual(result.relaxation_need[0].item(), 0.0)
        self.assertAlmostEqual(
            result.boosted_candidate_probs[0].item(),
            self.target[0, 1].item(),
            places=6,
        )

    def test_prefix_value_favors_earlier_positions(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            BlockFeatureConfig(
                block_kl_budget=0.01,
                use_future_support=False,
                use_feature_consistency=False,
                max_normalized_surprisal=100.0,
            ),
        )
        self.assertEqual(result.prefix_reach_probability[0].item(), 1.0)
        self.assertTrue(
            torch.all(
                result.prefix_reach_probability[1:]
                <= result.prefix_reach_probability[:-1]
            )
        )
        self.assertGreater(result.priority[0], result.priority[3])

    def test_unreachable_suffix_does_not_consume_block_budget(self) -> None:
        target = self.target.clone()
        draft = self.draft.clone()
        target[0, 1] = 1e-6
        target[0, 0] += 0.30 - 1e-6
        result = block_feature_distribution(
            target,
            draft,
            self.ids,
            BlockFeatureConfig(
                block_kl_budget=1.0,
                min_prefix_reach_probability=0.01,
                max_normalized_surprisal=100.0,
                min_feature_consistency=0.0,
            ),
        )
        self.assertLess(result.prefix_reach_probability[1].item(), 0.01)
        torch.testing.assert_close(
            result.allocated_kl[1:],
            torch.zeros(3),
        )

    def test_future_support_uses_the_rest_of_the_block(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            BlockFeatureConfig(min_feature_consistency=0.0),
        )
        self.assertGreater(result.future_support[0].item(), 0.0)
        self.assertGreater(result.future_support[1].item(), 0.0)
        self.assertEqual(result.future_support[-1].item(), 0.0)

    def test_projected_consistency_distinguishes_aligned_logits(self) -> None:
        target = torch.tensor(
            [
                [0.70, 0.20, 0.08, 0.02],
                [0.70, 0.20, 0.08, 0.02],
            ]
        )
        draft = torch.tensor(
            [
                [0.69, 0.21, 0.08, 0.02],
                [0.02, 0.08, 0.20, 0.70],
            ]
        )
        consistency = projected_logit_consistency(
            target,
            draft,
            torch.tensor([0, 1]),
        )
        self.assertGreater(consistency[0].item(), consistency[1].item())
        self.assertGreater(consistency[0].item(), 0.99)

    def test_state_consistency_is_shifted_to_the_next_position(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            BlockFeatureConfig(min_feature_consistency=0.0),
        )
        torch.testing.assert_close(
            result.state_feature_consistency[:-1],
            result.row_feature_consistency[1:],
        )

    def test_extreme_surprisal_gate_is_the_only_probability_gate(self) -> None:
        target = self.target.clone()
        target[0] = torch.tensor([0.999997, 0.000001, 0.000001, 0.000001])
        result = block_feature_distribution(
            target,
            self.draft,
            self.ids,
            BlockFeatureConfig(
                max_normalized_surprisal=2.0,
                use_feature_consistency=False,
            ),
        )
        self.assertFalse(result.safe[0].item())
        self.assertEqual(result.allocated_kl[0].item(), 0.0)

    def test_non_candidate_probabilities_keep_relative_ratios(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            BlockFeatureConfig(
                block_kl_budget=0.02,
                min_feature_consistency=0.0,
            ),
        )
        original_ratio = self.target[0, 0] / self.target[0, 2]
        relaxed_ratio = result.probs[0, 0] / result.probs[0, 2]
        torch.testing.assert_close(relaxed_ratio, original_ratio)
        torch.testing.assert_close(
            result.probs.sum(dim=-1),
            torch.ones(4),
        )

    def test_logit_boost_reconstructs_relaxed_distribution(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            BlockFeatureConfig(
                block_kl_budget=0.02,
                min_feature_consistency=0.0,
            ),
        )
        logits = boost_candidate_logits(
            torch.log(self.target),
            self.ids,
            result.target_candidate_probs,
            result.boosted_candidate_probs,
        )
        torch.testing.assert_close(
            torch.softmax(logits, dim=-1),
            result.probs,
        )

    def test_runtime_path_skips_full_relaxed_probability_tensor(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            BlockFeatureConfig(
                block_kl_budget=0.02,
                min_feature_consistency=0.0,
            ),
            assume_normalized=True,
            construct_probs=False,
        )
        self.assertIsNone(result.probs)
        self.assertLessEqual(result.realized_kl.sum().item(), 0.020001)

    def test_bernoulli_kl_matches_full_distribution_kl(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            BlockFeatureConfig(
                block_kl_budget=0.02,
                min_feature_consistency=0.0,
            ),
        )
        full_kl = (
            result.probs
            * (
                torch.log(result.probs.clamp_min(1e-30))
                - torch.log(self.target.clamp_min(1e-30))
            )
        ).sum(dim=-1)
        binary = bernoulli_kl(
            result.boosted_candidate_probs,
            result.target_candidate_probs,
        )
        torch.testing.assert_close(full_kl, binary)

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
            BlockFeatureConfig(expected_draft_tokens=0).validate()


if __name__ == "__main__":
    unittest.main()
