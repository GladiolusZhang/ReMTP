import unittest

import torch

from remtp.regret_router import (
    RegretRouterConfig,
    RegretRouterState,
    _STATE,
    _router_hidden_hook,
    accepted_prefix_mask,
    apply_budget_scale,
    apply_direction_to_hidden,
    expected_regret_debt,
    _pad_head_feature,
    regret_direction_from_target_mass,
    target_head_statistics,
)


class RegretRouterTest(unittest.TestCase):
    def test_config_requires_learned_checkpoint_and_collection_directory(self) -> None:
        RegretRouterConfig().validate()
        with self.assertRaises(ValueError):
            RegretRouterConfig(mode="learned").validate()
        with self.assertRaises(ValueError):
            RegretRouterConfig(mode="collect").validate()

    def test_prefix_mask_stops_at_first_recovery(self) -> None:
        mask = accepted_prefix_mask(
            torch.tensor([[10, 20, 99, 40]]),
            torch.tensor([10, 20, 30, 40]),
        )
        torch.testing.assert_close(mask, torch.tensor([True, True, False, False]))

    def test_expected_debt_uses_acceptance_uplift_not_sampled_uniform(self) -> None:
        weight, responsibility = expected_regret_debt(
            torch.tensor([True, True, False]),
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0.5, 0.6, 0.2]),
            torch.tensor([0.8, 0.6, 0.9]),
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([0.0, 0.5, 1.0]),
        )
        torch.testing.assert_close(responsibility, torch.tensor([0.3, 0.0, 0.7]))
        self.assertAlmostEqual(weight[0].item(), 0.03, places=6)
        self.assertEqual(weight[1].item(), 0.0)
        self.assertEqual(weight[2].item(), 0.0)

    def test_target_statistics_are_bounded(self) -> None:
        entropy, margin, values, ids = target_head_statistics(
            torch.tensor([[0.7, 0.2, 0.1]]),
            top_k=2,
        )
        self.assertTrue(0.0 <= entropy.item() <= 1.0)
        self.assertTrue(0.0 <= margin.item() <= 1.0)
        self.assertEqual(values.shape, (1, 2))
        self.assertEqual(ids.shape, (1, 2))

    def test_partial_head_features_are_padded(self) -> None:
        value = _pad_head_feature(torch.tensor([0.2, 0.4]), 4)
        torch.testing.assert_close(value, torch.tensor([0.2, 0.4, 0.4, 0.4]))

    def test_direction_points_away_from_relaxed_candidate(self) -> None:
        direction = regret_direction_from_target_mass(
            torch.tensor([[0.2, 0.7, 0.1]]),
            torch.tensor([0]),
            torch.tensor([0.5]),
            torch.eye(3),
            top_k=2,
        )
        self.assertLess(direction[0].item(), 0.0)
        self.assertGreater(direction[1].item(), 0.0)

    def test_hidden_correction_preserves_rms(self) -> None:
        hidden = torch.tensor([[1.0, 2.0, 3.0]])
        corrected = apply_direction_to_hidden(
            hidden,
            torch.tensor([-1.0, 1.0, 0.0]),
            torch.tensor(0.03),
        )
        torch.testing.assert_close(
            hidden.square().mean().sqrt(),
            corrected.square().mean().sqrt(),
        )
        self.assertFalse(torch.equal(hidden, corrected))

    def test_budget_scale_can_only_reduce_exact_tv(self) -> None:
        target = torch.tensor([0.1, 0.2, 0.3])
        boosted = torch.tensor([0.4, 0.5, 0.35])
        scaled = apply_budget_scale(
            target,
            boosted,
            torch.tensor([1.0, 0.5, 0.0]),
        )
        torch.testing.assert_close(scaled, torch.tensor([0.4, 0.35, 0.3]))
        self.assertTrue(torch.all(scaled >= target))
        self.assertTrue(torch.all(scaled <= boosted))

    def test_request_reset_preserves_policy_but_clears_feedback(self) -> None:
        state = RegretRouterState(
            request_id="old",
            policy_enabled=True,
            action_mode="logit_only",
            min_abs_logit_scale=0.01,
            uses_high_dimensional_context=False,
            uses_regret_direction=False,
            router_head=lambda *args: args,
            feedback_active=True,
            debt=torch.tensor([0.2]),
        )
        state.reset_request("new")
        self.assertEqual(state.request_id, "new")
        self.assertEqual(state.action_mode, "logit_only")
        self.assertEqual(state.min_abs_logit_scale, 0.01)
        self.assertFalse(state.uses_high_dimensional_context)
        self.assertFalse(state.uses_regret_direction)
        self.assertIsNotNone(state.router_head)
        self.assertFalse(state.feedback_active)
        self.assertIsNone(state.debt)

    def test_logit_only_policy_bypasses_hidden_exactly(self) -> None:
        previous = _STATE.action_mode
        try:
            _STATE.action_mode = "logit_only"
            hidden = torch.randn(3, 8, dtype=torch.bfloat16)
            output = _router_hidden_hook(hidden, 0)
            self.assertIs(output, hidden)
            self.assertTrue(torch.equal(output, hidden))
        finally:
            _STATE.action_mode = previous


if __name__ == "__main__":
    unittest.main()
