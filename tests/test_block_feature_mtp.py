import math
import unittest

import torch

from remtp.block_feature_mtp import (
    BlockFeatureConfig,
    bernoulli_kl,
    block_feature_distribution,
    boost_candidate_logits,
    greedy_verification_ids,
    topk_distribution_agreement,
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

    @staticmethod
    def config(**overrides: object) -> BlockFeatureConfig:
        values: dict[str, object] = {
            "distribution_top_k": 2,
            "min_reliability": 0.0,
            "max_normalized_surprisal": 100.0,
        }
        values.update(overrides)
        return BlockFeatureConfig(**values)

    def test_zero_budget_recovers_standard_target(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(block_kl_budget=0.0),
        )
        torch.testing.assert_close(result.probs, self.target)
        torch.testing.assert_close(result.allocated_kl, torch.zeros(4))

    def test_total_kl_respects_single_block_budget(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(block_kl_budget=0.02),
        )
        self.assertLessEqual(result.realized_kl.sum().item(), 0.020001)
        self.assertGreater(result.realized_kl.sum().item(), 0.0)

    def test_boost_never_exceeds_draft_candidate_probability(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(block_kl_budget=100.0),
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

    def test_strictly_accepted_candidate_needs_no_relaxation(self) -> None:
        draft = self.draft.clone()
        draft[0] = torch.tensor([0.70, 0.20, 0.05, 0.05])
        result = block_feature_distribution(
            self.target,
            draft,
            self.ids,
            self.config(block_kl_budget=1.0),
        )
        self.assertEqual(result.strict_acceptance[0].item(), 1.0)
        self.assertEqual(
            result.strict_rejection_probability[0].item(),
            0.0,
        )
        self.assertEqual(result.priority[0].item(), 0.0)
        self.assertAlmostEqual(
            result.boosted_candidate_probs[0].item(),
            self.target[0, 1].item(),
            places=6,
        )

    def test_candidate_rank_and_target_top1_gap_are_explicit(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(),
        )
        self.assertEqual(result.target_candidate_ranks[0].item(), 2)
        self.assertAlmostEqual(
            result.target_candidate_log_gaps[0].item(),
            math.log(0.55) - math.log(0.30),
            places=6,
        )
        self.assertGreater(result.token_support[0].item(), 0.0)
        self.assertLess(result.token_support[0].item(), 1.0)

    def test_candidate_rank_is_exact_beyond_distribution_topk(self) -> None:
        target = torch.tensor([[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]])
        draft = torch.tensor([[0.10, 0.10, 0.10, 0.10, 0.10, 0.50]])
        result = block_feature_distribution(
            target,
            draft,
            torch.tensor([5]),
            self.config(
                expected_draft_tokens=1,
                distribution_top_k=2,
            ),
        )
        self.assertEqual(result.target_candidate_ranks[0].item(), 6)

    def test_prefix_value_favors_earlier_reachable_positions(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(
                block_kl_budget=0.01,
                use_current_distribution=False,
                use_future_distribution=False,
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
        target[0, 1] = 1e-6
        target[0, 0] += 0.30 - 1e-6
        result = block_feature_distribution(
            target,
            self.draft,
            self.ids,
            self.config(
                block_kl_budget=1.0,
                min_prefix_reach_probability=0.01,
            ),
        )
        self.assertLess(result.prefix_reach_probability[1].item(), 0.01)
        torch.testing.assert_close(
            result.allocated_kl[1:],
            torch.zeros(3),
        )

    def test_topk_agreement_uses_full_distribution_not_only_q_y(self) -> None:
        target = torch.tensor(
            [
                [0.60, 0.25, 0.10, 0.04, 0.01],
                [0.60, 0.25, 0.10, 0.04, 0.01],
            ]
        )
        draft = torch.tensor(
            [
                [0.58, 0.27, 0.10, 0.04, 0.01],
                [0.05, 0.05, 0.10, 0.35, 0.45],
            ]
        )
        agreement = topk_distribution_agreement(
            target,
            draft,
            top_k=2,
            overlap_weight=0.4,
            probability_cosine_weight=0.4,
            entropy_consistency_weight=0.2,
        )
        # q(y=2) is identical, but the rest of Q points in opposite directions.
        self.assertEqual(draft[0, 2].item(), draft[1, 2].item())
        self.assertGreater(
            agreement.topk_overlap[0].item(),
            agreement.topk_overlap[1].item(),
        )
        self.assertGreater(
            agreement.probability_cosine[0].item(),
            agreement.probability_cosine[1].item(),
        )
        self.assertGreater(
            agreement.combined[0].item(),
            agreement.combined[1].item(),
        )

    def test_full_q_changes_relaxation_with_same_candidate_probability(
        self,
    ) -> None:
        target = torch.tensor([[0.60, 0.25, 0.10, 0.04, 0.01]])
        coherent_q = torch.tensor([[0.55, 0.20, 0.20, 0.04, 0.01]])
        drifting_q = torch.tensor([[0.05, 0.05, 0.20, 0.35, 0.35]])
        config = self.config(
            expected_draft_tokens=1,
            block_kl_budget=0.1,
            min_reliability=0.5,
            use_future_distribution=False,
        )
        coherent = block_feature_distribution(
            target,
            coherent_q,
            torch.tensor([2]),
            config,
        )
        drifting = block_feature_distribution(
            target,
            drifting_q,
            torch.tensor([2]),
            config,
        )
        self.assertEqual(
            coherent.draft_candidate_probs[0].item(),
            drifting.draft_candidate_probs[0].item(),
        )
        self.assertTrue(coherent.safe[0].item())
        self.assertFalse(drifting.safe[0].item())
        self.assertGreater(
            coherent.boosted_candidate_probs[0].item(),
            coherent.target_candidate_probs[0].item(),
        )
        self.assertEqual(
            drifting.boosted_candidate_probs[0].item(),
            drifting.target_candidate_probs[0].item(),
        )

    def test_future_distribution_uses_only_later_positions(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(future_decay=0.5),
        )
        expected_first = (
            result.current_distribution_consistency[1]
            + 0.5 * result.current_distribution_consistency[2]
            + 0.25 * result.current_distribution_consistency[3]
        ) / 1.75
        torch.testing.assert_close(
            result.future_distribution_consistency[0],
            expected_first,
        )
        self.assertEqual(
            result.future_distribution_consistency[-1].item(),
            0.0,
        )

    def test_future_distribution_can_rescue_a_coherent_path(self) -> None:
        target = torch.tensor(
            [
                [0.45, 0.30, 0.15, 0.07, 0.03],
                [0.55, 0.30, 0.10, 0.04, 0.01],
                [0.55, 0.30, 0.10, 0.04, 0.01],
                [0.55, 0.30, 0.10, 0.04, 0.01],
            ]
        )
        draft = torch.tensor(
            [
                [0.20, 0.40, 0.10, 0.15, 0.15],
                [0.53, 0.34, 0.08, 0.04, 0.01],
                [0.53, 0.34, 0.08, 0.04, 0.01],
                [0.53, 0.34, 0.08, 0.04, 0.01],
            ]
        )
        ids = torch.tensor([1, 1, 1, 1])
        without_future = block_feature_distribution(
            target,
            draft,
            ids,
            self.config(use_future_distribution=False),
        )
        with_future = block_feature_distribution(
            target,
            draft,
            ids,
            self.config(use_future_distribution=True),
        )
        self.assertGreater(
            with_future.future_distribution_consistency[0].item(),
            with_future.current_distribution_consistency[0].item(),
        )
        self.assertGreater(
            with_future.reliability[0].item(),
            without_future.reliability[0].item(),
        )

    def test_last_position_is_not_penalized_for_missing_future(self) -> None:
        with_future = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(use_future_distribution=True),
        )
        without_future = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(use_future_distribution=False),
        )
        torch.testing.assert_close(
            with_future.reliability[-1],
            without_future.reliability[-1],
        )

    def test_all_low_signals_disable_relaxation(self) -> None:
        target = torch.tensor(
            [[0.97, 0.01, 0.01, 0.005, 0.005]] * 4
        )
        draft = torch.tensor(
            [[0.05, 0.50, 0.05, 0.20, 0.20]] * 4
        )
        result = block_feature_distribution(
            target,
            draft,
            torch.tensor([1, 1, 1, 1]),
            self.config(
                min_reliability=0.8,
                max_normalized_surprisal=1000.0,
            ),
        )
        self.assertFalse(result.safe.any().item())
        torch.testing.assert_close(result.allocated_kl, torch.zeros(4))

    def test_non_candidate_probabilities_keep_relative_ratios(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(block_kl_budget=0.02),
        )
        original_ratio = self.target[0, 0] / self.target[0, 2]
        relaxed_ratio = result.probs[0, 0] / result.probs[0, 2]
        torch.testing.assert_close(relaxed_ratio, original_ratio)
        torch.testing.assert_close(result.probs.sum(dim=-1), torch.ones(4))

    def test_logit_boost_reconstructs_relaxed_distribution(self) -> None:
        result = block_feature_distribution(
            self.target,
            self.draft,
            self.ids,
            self.config(block_kl_budget=0.02),
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
            self.config(block_kl_budget=0.02),
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
            self.config(block_kl_budget=0.02),
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
            BlockFeatureConfig(distribution_top_k=0).validate()
        with self.assertRaises(ValueError):
            topk_distribution_agreement(
                self.target,
                self.draft,
                top_k=2,
                overlap_weight=0.0,
                probability_cosine_weight=0.0,
                entropy_consistency_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
