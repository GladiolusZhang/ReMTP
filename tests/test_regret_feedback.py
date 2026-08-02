import unittest

import torch

from remtp.regret_feedback import (
    RegretFeedbackConfig,
    accepted_prefix_mask,
    boundary_hidden_residual,
    build_sparse_vocab_residuals,
    compatibility_gate,
    counterfactual_regret_scale_update,
    counterfactual_tv_scale_update,
    counterfactual_vocab_kl_gain,
    expected_causal_responsibility,
    expected_regret_scale_update,
    expected_top1_bias_update,
    exact_root_anchor_gate,
    fusion_backproject_residual,
    fusion_aligned_regret,
    fusion_forward_direction,
    fusion_preconditioned_regret,
    hidden_residual_block_vector,
    hidden_residual_block_rows,
    inject_regret,
    pad_head_rows,
    pad_head_values,
    probability_residual_head_rows,
    recover_unscaled_draft_probs,
    repeated_ngram_continuation_ids,
    regret_strength_gate,
    regret_block_vector,
    regret_risk_redistribution,
    strict_and_causal_masks,
    transport_regret_to_target_head,
    update_confirmed_hidden_memory,
    update_regret_memory,
    update_vocab_head_route,
)


class RegretFeedbackTest(unittest.TestCase):
    @staticmethod
    def config(**overrides: object) -> RegretFeedbackConfig:
        values: dict[str, object] = {
            "expected_draft_tokens": 3,
            "regret_top_k": 2,
            "compatibility_top_k": 3,
            "head_reliability": (1.0, 0.7, 0.3),
            "vocab_head_map": (0, 1, 2),
            "adaptive_min_depth": 2,
        }
        values.update(overrides)
        return RegretFeedbackConfig(**values)

    def test_exact_causal_relaxed_event_uses_same_uniform(self) -> None:
        draft = torch.tensor([4, 5, 6])
        output = torch.tensor([[4, 5, 9, -1]])
        accepted = accepted_prefix_mask(output, draft)
        reached, strict, causal = strict_and_causal_masks(
            accepted,
            torch.tensor([0.8, 0.2, 0.5]),
            uniform_probs=torch.tensor([0.3, 0.4, 0.9]),
        )
        torch.testing.assert_close(
            accepted,
            torch.tensor([True, True, False]),
        )
        torch.testing.assert_close(
            reached,
            torch.tensor([True, True, True]),
        )
        torch.testing.assert_close(
            strict,
            torch.tensor([True, False, False]),
        )
        torch.testing.assert_close(
            causal,
            torch.tensor([False, True, False]),
        )

    def test_only_causal_positions_create_regret(self) -> None:
        target = torch.tensor(
            [
                [0.50, 0.30, 0.15, 0.05],
                [0.45, 0.30, 0.20, 0.05],
                [0.40, 0.30, 0.20, 0.10],
            ]
        )
        candidates = torch.tensor([1, 1, 1])
        output_weight = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        vector, strength, _ = regret_block_vector(
            target,
            candidates,
            torch.tensor([0.05, 0.10, 0.20]),
            torch.tensor([False, True, False]),
            torch.tensor(3),
            output_weight,
            self.config(),
        )
        self.assertGreater(strength.item(), 0.0)
        self.assertGreater(vector.square().sum().item(), 0.0)

        zero_vector, zero_strength, _ = regret_block_vector(
            target,
            candidates,
            torch.tensor([0.05, 0.10, 0.20]),
            torch.zeros(3, dtype=torch.bool),
            torch.tensor(3),
            output_weight,
            self.config(),
        )
        self.assertEqual(zero_strength.item(), 0.0)
        torch.testing.assert_close(zero_vector, torch.zeros_like(zero_vector))

    def test_expected_trust_can_suppress_unreliable_residuals(self) -> None:
        target = torch.tensor(
            [
                [0.50, 0.30, 0.15, 0.05],
                [0.45, 0.30, 0.20, 0.05],
                [0.40, 0.30, 0.20, 0.10],
            ]
        )
        output_weight = torch.eye(4, 3)
        vector, strength, _ = regret_block_vector(
            target,
            torch.tensor([1, 1, 1]),
            torch.tensor([0.05, 0.10, 0.20]),
            torch.ones(3),
            torch.tensor(3),
            output_weight,
            self.config(direction="expected_trust"),
            event_trust=torch.zeros(3),
        )
        self.assertEqual(strength.item(), 0.0)
        torch.testing.assert_close(vector, torch.zeros_like(vector))

    def test_hidden_residual_points_from_mtp_prediction_to_target_fact(
        self,
    ) -> None:
        draft_hidden = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        target_hidden = torch.tensor(
            [[2.0, 0.0], [0.0, 3.0], [3.0, 1.0]]
        )
        vector, strength = hidden_residual_block_vector(
            draft_hidden,
            target_hidden,
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([False, True, False]),
            torch.tensor(3),
            torch.tensor([1.0, 1.0, 1.0]),
            self.config(),
        )
        self.assertGreater(strength.item(), 0.0)
        self.assertGreater(vector[1].item(), 0.0)
        self.assertAlmostEqual(vector[0].item(), 0.0, places=6)

    def test_headwise_hidden_residual_keeps_events_separate(self) -> None:
        draft_hidden = torch.zeros((3, 2))
        target_hidden = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
        )
        rows, weights = hidden_residual_block_rows(
            draft_hidden,
            target_hidden,
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([True, False, True]),
            torch.tensor(3),
            torch.ones(3),
            self.config(),
        )
        self.assertGreater(weights[0].item(), 0.0)
        self.assertEqual(weights[1].item(), 0.0)
        self.assertGreater(weights[2].item(), 0.0)
        torch.testing.assert_close(rows[1], torch.zeros(2))

    def test_boundary_residual_requires_a_fully_committed_block(self) -> None:
        draft_hidden = torch.zeros((3, 2))
        target_hidden = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
        )
        full, full_strength = boundary_hidden_residual(
            draft_hidden,
            target_hidden,
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0.0, 0.5, 0.5]),
            torch.tensor(3),
            torch.tensor(False),
            torch.ones(3),
        )
        self.assertGreater(full_strength.item(), 0.0)
        self.assertGreater(full.square().sum().item(), 0.0)

        rejected, rejected_strength = boundary_hidden_residual(
            draft_hidden,
            target_hidden,
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0.0, 0.5, 0.0]),
            torch.tensor(2),
            torch.tensor(True),
            torch.ones(3),
        )
        self.assertEqual(rejected_strength.item(), 0.0)
        torch.testing.assert_close(rejected, torch.zeros(2))

    def test_probability_residual_gradient_points_from_q_toward_p(
        self,
    ) -> None:
        target = torch.tensor(
            [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]
        )
        draft = torch.tensor(
            [[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]]
        )
        rows, weights = probability_residual_head_rows(
            target,
            draft,
            torch.eye(2),
            torch.tensor([0.1, 0.1, 0.1]),
            torch.tensor([1.0, 1.0, 0.0]),
            (1.0, 1.0, 1.0),
            top_k=2,
        )
        self.assertGreater(rows[0, 0].item(), 0.0)
        self.assertLess(rows[0, 1].item(), 0.0)
        self.assertLess(rows[1, 0].item(), 0.0)
        self.assertGreater(rows[1, 1].item(), 0.0)
        self.assertEqual(weights[2].item(), 0.0)
        torch.testing.assert_close(rows[2], torch.zeros(2))

    def test_memory_decay_rejection_reset_and_expiry(self) -> None:
        config = self.config(
            token_decay=0.9,
            rejection_reset=0.25,
            max_idle_blocks=2,
        )
        old = torch.ones(4)
        memory, age = update_regret_memory(
            old,
            torch.tensor(0),
            torch.zeros(4),
            torch.tensor(0.0),
            torch.tensor(2),
            torch.tensor(True),
            config,
        )
        torch.testing.assert_close(
            memory,
            torch.full((4,), 0.9**2 * 0.25),
        )
        self.assertEqual(age.item(), 1)

        memory, age = update_regret_memory(
            memory,
            age,
            torch.zeros(4),
            torch.tensor(0.0),
            torch.tensor(1),
            torch.tensor(False),
            config,
        )
        torch.testing.assert_close(memory, torch.zeros(4))
        self.assertEqual(age.item(), 2)

    def test_compatibility_gate_is_positive_for_target_aligned_memory(self) -> None:
        output_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        gate, correlation = compatibility_gate(
            torch.tensor([1.0, 0.0]),
            output_weight,
            torch.tensor([0, 1, 2]),
            torch.tensor([0.8, 0.15, 0.05]),
        )
        self.assertGreater(correlation.item(), 0.0)
        self.assertGreater(gate.item(), 0.0)

    def test_positive_lift_gate_rejects_relative_but_absolute_downshift(
        self,
    ) -> None:
        output_weight = torch.tensor(
            [
                [-0.1, 0.995],
                [-0.5, 0.866],
                [-1.0, 0.0],
            ]
        )
        memory = torch.tensor([1.0, 0.0])
        top_ids = torch.tensor([0, 1, 2])
        top_probs = torch.tensor([0.8, 0.15, 0.05])
        relative_gate, correlation = compatibility_gate(
            memory,
            output_weight,
            top_ids,
            top_probs,
            mode="relative",
        )
        positive_gate, _ = compatibility_gate(
            memory,
            output_weight,
            top_ids,
            top_probs,
            mode="positive_lift",
        )
        self.assertGreater(correlation.item(), 0.0)
        self.assertGreater(relative_gate.item(), 0.0)
        self.assertEqual(positive_gate.item(), 0.0)

    def test_transport_reanchors_regret_in_current_target_head(self) -> None:
        output_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        direction, gate, mass = transport_regret_to_target_head(
            torch.tensor([1.0, 0.0]),
            output_weight,
            torch.tensor([0, 1, 2]),
            torch.tensor([0.80, 0.15, 0.05]),
        )
        self.assertGreater(direction[0].item(), 0.0)
        self.assertLess(direction[1].item(), 0.0)
        self.assertGreater(gate.item(), 0.0)
        self.assertGreater(mass.item(), 0.0)

    def test_injection_changes_direction_but_preserves_rms(self) -> None:
        hidden = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
        memory = torch.tensor([0.0, 1.0, 1.0, 0.0])
        steered = inject_regret(
            hidden,
            memory,
            torch.tensor(1.0),
            alpha=0.05,
        )
        self.assertFalse(torch.equal(hidden, steered))
        torch.testing.assert_close(
            torch.sqrt(hidden.square().mean(dim=-1)),
            torch.sqrt(steered.square().mean(dim=-1)),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_regret_strength_gate_preserves_tv_magnitude(self) -> None:
        memory = torch.full((4,), 0.1)
        self.assertEqual(regret_strength_gate(memory, 0.0).item(), 1.0)
        self.assertAlmostEqual(
            regret_strength_gate(memory, 0.2).item(),
            0.5,
            places=6,
        )
        self.assertEqual(regret_strength_gate(memory, 0.05).item(), 1.0)

    def test_verifier_risk_preserves_budget_and_zero_regret_identity(
        self,
    ) -> None:
        target = torch.tensor(
            [
                [0.80, 0.05, 0.10, 0.05],
                [0.40, 0.25, 0.25, 0.10],
                [0.30, 0.35, 0.20, 0.15],
            ]
        )
        draft = torch.tensor(
            [
                [0.40, 0.40, 0.10, 0.10],
                [0.30, 0.40, 0.20, 0.10],
                [0.20, 0.50, 0.20, 0.10],
            ]
        )
        ids = torch.tensor([1, 1, 1])
        p_y = target[torch.arange(3), ids]
        base = p_y + torch.tensor([0.10, 0.05, 0.05])
        config = self.config(
            verifier_risk_strength=3.0,
            verifier_risk_reference=0.25,
        )
        identity, zero_level, _ = regret_risk_redistribution(
            target,
            draft,
            ids,
            base,
            torch.ones(3),
            torch.tensor(0.0),
            config,
        )
        torch.testing.assert_close(identity, base)
        self.assertEqual(zero_level.item(), 0.0)

        adjusted, risk_level, risk = regret_risk_redistribution(
            target,
            draft,
            ids,
            base,
            torch.ones(3),
            torch.tensor(0.25),
            config,
        )
        adjusted_tv = adjusted - p_y
        base_tv = base - p_y
        torch.testing.assert_close(adjusted_tv.sum(), base_tv.sum())
        self.assertTrue(torch.all(adjusted_tv >= 0.0))
        self.assertTrue(
            torch.all(adjusted_tv <= draft[torch.arange(3), ids] - p_y)
        )
        self.assertEqual(risk_level.item(), 1.0)
        self.assertGreater(risk[0].item(), risk[-1].item())
        self.assertLess(adjusted_tv[0].item(), base_tv[0].item())

    def test_cactus_anchor_trades_budget_by_regret_level(self) -> None:
        target = torch.tensor(
            [
                [0.70, 0.10, 0.10, 0.10],
                [0.40, 0.30, 0.20, 0.10],
                [0.30, 0.40, 0.20, 0.10],
            ]
        )
        draft = torch.tensor(
            [
                [0.40, 0.40, 0.10, 0.10],
                [0.30, 0.50, 0.10, 0.10],
                [0.20, 0.60, 0.10, 0.10],
            ]
        )
        ids = torch.tensor([1, 1, 1])
        p_y = target[torch.arange(3), ids]
        base = p_y + torch.tensor([0.10, 0.05, 0.05])
        config = self.config(
            verifier_policy="cactus_anchor",
            verifier_risk_strength=1.0,
            verifier_risk_reference=0.25,
            verifier_budget_slope=0.4,
        )
        low, _, _ = regret_risk_redistribution(
            target,
            draft,
            ids,
            base,
            torch.ones(3),
            torch.tensor(0.025),
            config,
        )
        high, _, _ = regret_risk_redistribution(
            target,
            draft,
            ids,
            base,
            torch.ones(3),
            torch.tensor(0.25),
            config,
        )
        self.assertGreater((low - p_y).sum().item(), 0.20)
        self.assertLess((high - p_y).sum().item(), 0.20)

    def test_expected_regret_scale_softens_an_overconfident_head(self) -> None:
        target = torch.tensor(
            [[0.50, 0.50], [0.50, 0.50], [0.50, 0.50]]
        )
        draft = torch.tensor(
            [[0.90, 0.10], [0.90, 0.10], [0.90, 0.10]]
        )
        config = self.config(
            direction="expected_scale",
            scale_learning_rate=1.0,
            scale_decay=1.0,
            scale_gap_reference=0.01,
        )
        updated, residual, pq_kl, delta = expected_regret_scale_update(
            target,
            draft,
            torch.tensor([0, 0, 0]),
            torch.tensor([True, False, False]),
            torch.tensor([0.50, 0.50, 0.50]),
            torch.tensor([0.70, 0.70, 0.70]),
            torch.zeros(3),
            config,
        )
        self.assertLess(updated[0].item(), 0.0)
        self.assertLess(delta[0].item(), 0.0)
        self.assertGreater(residual[0].item(), 0.0)
        self.assertEqual(residual[1].item(), 0.0)
        self.assertGreater(pq_kl[0].item(), 0.0)

    def test_expected_causal_responsibility_is_conditioned_on_commit(
        self,
    ) -> None:
        weights = expected_causal_responsibility(
            torch.tensor(
                [[0.50, 0.50], [0.50, 0.50], [0.50, 0.50]]
            ),
            torch.tensor([0, 0, 0]),
            torch.tensor([True, False, True]),
            torch.tensor([0.20, 0.20, 0.20]),
            torch.tensor([0.40, 0.40, 0.20]),
        )
        self.assertAlmostEqual(weights[0].item(), 0.5, places=6)
        self.assertEqual(weights[1].item(), 0.0)
        self.assertEqual(weights[2].item(), 0.0)

    def test_top1_bias_uses_causal_probability_residual(self) -> None:
        config = self.config(
            direction="expected_top1_bias",
            top1_bias_initial=0.25,
            top1_bias_learning_rate=0.25,
            top1_bias_decay=0.9,
            top1_bias_clip=0.75,
        )
        target = torch.tensor([[0.2, 0.8], [0.8, 0.2]])
        draft = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
        updated, residual, pq_kl = expected_top1_bias_update(
            target,
            draft,
            torch.tensor([0, 0]),
            torch.tensor([True, False]),
            torch.tensor([0.2, 0.8]),
            torch.tensor([0.6, 0.8]),
            torch.tensor([0.25, 0.25]),
            config,
        )
        self.assertGreater(residual[0].item(), 0.0)
        self.assertEqual(residual[1].item(), 0.0)
        self.assertLess(updated[0].item(), 0.25 * 0.9)
        self.assertAlmostEqual(updated[1].item(), 0.25 * 0.9, places=6)
        self.assertGreater(pq_kl[0].item(), 0.0)

    def test_expected_regret_scale_sharpens_an_underconfident_head(self) -> None:
        target = torch.tensor(
            [[0.90, 0.10], [0.90, 0.10], [0.90, 0.10]]
        )
        draft = torch.tensor(
            [[0.60, 0.40], [0.60, 0.40], [0.60, 0.40]]
        )
        config = self.config(
            direction="expected_scale",
            scale_learning_rate=1.0,
            scale_decay=1.0,
            scale_gap_reference=0.01,
        )
        updated, residual, _, delta = expected_regret_scale_update(
            target,
            draft,
            torch.tensor([0, 0, 0]),
            torch.tensor([True, False, False]),
            torch.tensor([0.20, 0.20, 0.20]),
            torch.tensor([0.40, 0.40, 0.40]),
            torch.zeros(3),
            config,
        )
        self.assertGreater(updated[0].item(), 0.0)
        self.assertGreater(delta[0].item(), 0.0)
        self.assertGreater(residual[0].item(), 0.0)

    def test_expected_regret_has_no_update_without_relaxation_residual(
        self,
    ) -> None:
        probs = torch.tensor(
            [[0.70, 0.30], [0.60, 0.40], [0.55, 0.45]]
        )
        old = torch.tensor([0.10, -0.10, 0.05])
        config = self.config(
            direction="expected_scale",
            scale_decay=0.8,
        )
        updated, residual, _, _ = expected_regret_scale_update(
            probs,
            probs,
            torch.tensor([0, 0, 0]),
            torch.tensor([True, True, True]),
            torch.tensor([0.70, 0.60, 0.55]),
            torch.tensor([0.70, 0.60, 0.55]),
            old,
            config,
        )
        torch.testing.assert_close(updated, old * 0.8)
        torch.testing.assert_close(residual, torch.zeros(3))

    def test_expected_regret_scale_respects_bounds(self) -> None:
        target = torch.tensor(
            [[0.90, 0.10], [0.50, 0.50], [0.50, 0.50]]
        )
        draft = torch.tensor(
            [[0.60, 0.40], [0.90, 0.10], [0.90, 0.10]]
        )
        config = self.config(
            direction="expected_scale",
            scale_learning_rate=2.0,
            scale_decay=1.0,
            scale_gap_reference=0.01,
            scale_min=0.90,
            scale_max=1.10,
        )
        updated, _, _, _ = expected_regret_scale_update(
            target,
            draft,
            torch.tensor([0, 0, 0]),
            torch.tensor([True, True, True]),
            torch.tensor([0.20, 0.20, 0.20]),
            torch.tensor([0.40, 0.40, 0.40]),
            torch.tensor([1.0, -1.0, -1.0]),
            config,
        )
        self.assertTrue(torch.all(torch.exp(updated) <= 1.10 + 1e-6))
        self.assertTrue(torch.all(torch.exp(updated) >= 0.90 - 1e-6))

    def test_unscaled_distribution_is_recovered_from_scaled_q(self) -> None:
        logits = torch.tensor(
            [[1.2, -0.4, 0.3], [0.1, 0.7, -0.8]],
            dtype=torch.float32,
        )
        log_scales = torch.log(torch.tensor([1.2, 0.8]))
        scaled = torch.softmax(
            logits * torch.exp(log_scales).unsqueeze(1),
            dim=-1,
        )
        recovered = recover_unscaled_draft_probs(scaled, log_scales)
        torch.testing.assert_close(
            recovered,
            torch.softmax(logits, dim=-1),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_repeated_ngram_finds_only_loop_continuations(self) -> None:
        history = [10, 11, 12, 13, 10, 11, 12]
        self.assertEqual(
            repeated_ngram_continuation_ids(history, 4),
            (13,),
        )
        self.assertEqual(
            repeated_ngram_continuation_ids([10, 11, 12], 4),
            (),
        )
        with self.assertRaises(ValueError):
            repeated_ngram_continuation_ids(history, 1)

    def test_exact_root_anchor_requires_material_tv_debt(self) -> None:
        self.assertIsNone(exact_root_anchor_gate(None, 0.05))
        self.assertEqual(
            exact_root_anchor_gate(torch.tensor(0.049), 0.05).item(),
            0.0,
        )
        self.assertEqual(
            exact_root_anchor_gate(torch.tensor(0.05), 0.05).item(),
            1.0,
        )
        with self.assertRaises(ValueError):
            exact_root_anchor_gate(torch.tensor(0.1), 0.0)

    def test_variable_depth_audit_padding_preserves_prefix(self) -> None:
        padded = pad_head_values(torch.tensor([1.0, 2.0]), 4)
        torch.testing.assert_close(
            padded,
            torch.tensor([1.0, 2.0, 0.0, 0.0]),
        )
        with self.assertRaises(ValueError):
            pad_head_values(torch.ones(5), 4)

        padded_rows = pad_head_rows(torch.tensor([[1.0], [2.0]]), 4)
        torch.testing.assert_close(
            padded_rows,
            torch.tensor([[1.0], [2.0], [0.0], [0.0]]),
        )
        with self.assertRaises(ValueError):
            pad_head_rows(torch.ones((5, 1)), 4)

        self.config(direction="expected_scale_adaptive_depth").validate()

    def test_counterfactual_scale_keeps_a_helpful_action(self) -> None:
        base_logits = torch.tensor(
            [[0.4, 0.0], [0.4, 0.0], [0.4, 0.0]],
            dtype=torch.float32,
        )
        old = torch.log(torch.full((3,), 1.10))
        scaled = torch.softmax(
            base_logits * torch.exp(old).unsqueeze(1),
            dim=-1,
        )
        target = torch.tensor(
            [[0.80, 0.20], [0.80, 0.20], [0.80, 0.20]]
        )
        config = self.config(
            direction="expected_scale_counterfactual",
            scale_learning_rate=1.0,
            scale_decay=1.0,
            scale_gap_reference=0.01,
            scale_min=0.75,
            scale_max=1.25,
        )
        updated, residual, _, delta, gain = (
            counterfactual_regret_scale_update(
                target,
                scaled,
                torch.tensor([0, 0, 0]),
                torch.tensor([True, False, False]),
                torch.tensor([0.20, 0.20, 0.20]),
                torch.tensor([0.40, 0.40, 0.40]),
                old,
                config,
            )
        )
        self.assertGreater(gain[0].item(), 0.0)
        self.assertGreater(delta[0].item(), 0.0)
        self.assertGreater(updated[0].item(), 0.0)
        self.assertGreater(residual[0].item(), 0.0)

    def test_counterfactual_scale_rolls_back_a_harmful_action(self) -> None:
        base_logits = torch.tensor(
            [[1.2, 0.0], [1.2, 0.0], [1.2, 0.0]],
            dtype=torch.float32,
        )
        old = torch.log(torch.full((3,), 1.20))
        scaled = torch.softmax(
            base_logits * torch.exp(old).unsqueeze(1),
            dim=-1,
        )
        target = torch.tensor(
            [[0.50, 0.50], [0.50, 0.50], [0.50, 0.50]]
        )
        config = self.config(
            direction="expected_scale_counterfactual",
            scale_learning_rate=1.0,
            scale_decay=1.0,
            scale_gap_reference=0.01,
            scale_min=0.75,
            scale_max=1.25,
        )
        updated, _, _, _, gain = counterfactual_regret_scale_update(
            target,
            scaled,
            torch.tensor([0, 0, 0]),
            torch.tensor([True, False, False]),
            torch.tensor([0.20, 0.20, 0.20]),
            torch.tensor([0.40, 0.40, 0.40]),
            old,
            config,
        )
        self.assertLess(gain[0].item(), 0.0)
        self.assertLess(updated[0].item(), 0.0)
        self.assertAlmostEqual(updated[1].item(), old[1].item(), places=6)

    def test_tv_scale_selects_only_exact_tv_improvements(self) -> None:
        draft = torch.tensor(
            [[0.60, 0.40], [0.80, 0.20], [0.50, 0.50]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[0.80, 0.20], [0.50, 0.50], [0.50, 0.50]],
            dtype=torch.float32,
        )
        config = self.config(
            direction="expected_scale_tv",
            scale_learning_rate=1.0,
            scale_decay=1.0,
            scale_gap_reference=0.01,
            scale_min=0.90,
            scale_max=1.10,
        )
        updated, residual, _, selected_delta, applied_gain = (
            counterfactual_tv_scale_update(
                target,
                draft,
                torch.tensor([0, 0, 0]),
                torch.tensor([True, True, False]),
                torch.tensor([0.20, 0.20, 0.20]),
                torch.tensor([0.40, 0.40, 0.20]),
                torch.zeros(3),
                config,
            )
        )
        self.assertGreater(updated[0].item(), 0.0)
        self.assertGreater(selected_delta[0].item(), 0.0)
        self.assertLess(updated[1].item(), 0.0)
        self.assertLess(selected_delta[1].item(), 0.0)
        self.assertEqual(updated[2].item(), 0.0)
        self.assertGreater(residual[0].item(), 0.0)
        self.assertGreater(residual[1].item(), 0.0)
        torch.testing.assert_close(applied_gain, torch.zeros(3))

    def test_sparse_vocab_residual_separates_positive_and_negative_mass(
        self,
    ) -> None:
        target = torch.tensor(
            [[0.70, 0.20, 0.10], [0.20, 0.60, 0.20]]
        )
        draft = torch.tensor(
            [[0.40, 0.50, 0.10], [0.50, 0.30, 0.20]]
        )
        full, positive, negative, candidate = (
            build_sparse_vocab_residuals(
                target,
                draft,
                torch.tensor([1, 0]),
                torch.tensor([1.0, 0.5]),
                torch.tensor(0.7),
                top_k=2,
                clip=2.0,
            )
        )
        self.assertGreater(positive[0, 0].item(), 0.0)
        self.assertLess(negative[0, 1].item(), 0.0)
        self.assertLess(candidate[0, 1].item(), 0.0)
        self.assertLess(candidate[1, 0].item(), 0.0)
        self.assertGreater(full.abs().sum().item(), 0.0)

    def test_same_context_vocab_residual_reduces_pq_kl(self) -> None:
        target = torch.tensor(
            [[0.70, 0.20, 0.10], [0.20, 0.60, 0.20]]
        )
        draft = torch.tensor(
            [[0.40, 0.50, 0.10], [0.50, 0.30, 0.20]]
        )
        full, _, _, _ = build_sparse_vocab_residuals(
            target,
            draft,
            torch.tensor([1, 0]),
            torch.ones(2),
            torch.tensor(0.7),
            top_k=3,
            clip=2.0,
        )
        gain = counterfactual_vocab_kl_gain(
            target,
            draft,
            full,
            torch.tensor(0.7),
            0.05,
        )
        self.assertGreater(gain[0, 0].item(), 0.0)
        self.assertGreater(gain[1, 1].item(), 0.0)

    def test_fusion_aligned_regret_respects_fc_channel_geometry(self) -> None:
        regret = torch.tensor([1.0, -2.0, 0.5])
        identity = torch.eye(3)
        fc_weight = torch.cat([identity, identity], dim=1)
        torch.testing.assert_close(
            fusion_aligned_regret(regret, fc_weight),
            regret,
        )
        with self.assertRaises(ValueError):
            fusion_aligned_regret(regret, torch.eye(3))

    def test_fusion_backproject_residual_uses_hidden_channel(self) -> None:
        residual = torch.tensor([1.0, -2.0, 0.5])
        identity = torch.eye(3)
        fc_weight = torch.cat([2.0 * identity, identity], dim=1)
        torch.testing.assert_close(
            fusion_backproject_residual(residual, fc_weight),
            residual,
        )
        with self.assertRaises(ValueError):
            fusion_backproject_residual(residual, torch.eye(3))

    def test_fusion_forward_direction_uses_hidden_channel(self) -> None:
        direction = torch.tensor([1.0, -2.0, 0.5])
        identity = torch.eye(3)
        fc_weight = torch.cat([2.0 * identity, 3.0 * identity], dim=1)
        torch.testing.assert_close(
            fusion_forward_direction(direction, fc_weight),
            3.0 * direction,
        )
        with self.assertRaises(ValueError):
            fusion_forward_direction(direction, torch.eye(3))

    def test_fusion_preconditioned_regret_corrects_row_scale(self) -> None:
        regret = torch.tensor([1.0, -2.0, 0.5])
        identity = torch.eye(3)
        hidden = torch.diag(torch.tensor([1.0, 2.0, 4.0]))
        fc_weight = torch.cat([identity, hidden], dim=1)
        expected = torch.tensor([1.0, -1.0, 0.125])
        torch.testing.assert_close(
            fusion_preconditioned_regret(regret, fc_weight),
            expected,
        )
        with self.assertRaises(ValueError):
            fusion_preconditioned_regret(regret, torch.eye(3))

    def test_expected_vocab_audit_is_a_valid_direction(self) -> None:
        self.config(direction="expected_vocab_audit").validate()
        self.config(direction="expected_vocab_negative").validate()
        self.config(direction="expected_vocab_headmap").validate()
        self.config(direction="expected_vocab_online").validate()
        self.config(direction="expected_fusion").validate()
        self.config(direction="expected_token_adaptive").validate()
        self.config(direction="expected_token_preconditioned").validate()
        self.config(direction="expected_hidden_fusion").validate()
        self.config(direction="expected_hidden_consistent").validate()
        self.config(direction="expected_hidden_adaptive").validate()
        self.config(direction="expected_hidden_calibrated").validate()
        self.config(direction="expected_hidden_compatible").validate()
        self.config(direction="expected_root_anchor").validate()

    def test_online_vocab_route_uses_only_positive_ema_gain(self) -> None:
        gain = torch.tensor(
            [
                [0.2, -0.1, 0.0],
                [0.1, 0.3, -0.2],
                [-0.4, 0.1, -0.1],
            ]
        )
        scores, route = update_vocab_head_route(
            None,
            gain,
            decay=0.5,
            min_gain=0.0,
        )
        torch.testing.assert_close(scores, gain * 0.5)
        self.assertEqual(route, (0, 1, -1))

        updated, route = update_vocab_head_route(
            scores,
            torch.tensor(
                [
                    [-0.4, 0.0, 0.4],
                    [0.0, 0.1, 0.0],
                    [0.3, -0.2, 0.2],
                ]
            ),
            decay=0.5,
            min_gain=0.051,
        )
        self.assertEqual(route, (-1, 1, 0))
        self.assertEqual(updated.shape, (3, 3))

    def test_hidden_memory_requires_consecutive_aligned_blocks(self) -> None:
        config = self.config(
            token_decay=1.0,
            rejection_reset=0.25,
            hidden_consistency_threshold=0.0,
        )
        memory = torch.zeros(3)
        confirmations = torch.tensor(0)
        idle = torch.tensor(0)
        memory, confirmations, idle, _ = update_confirmed_hidden_memory(
            memory,
            confirmations,
            idle,
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor(0.1),
            torch.tensor(1),
            torch.tensor(False),
            config,
        )
        self.assertEqual(confirmations.item(), 1)
        memory, confirmations, idle, alignment = (
            update_confirmed_hidden_memory(
                memory,
                confirmations,
                idle,
                torch.tensor([0.5, 0.1, 0.0]),
                torch.tensor(0.1),
                torch.tensor(1),
                torch.tensor(False),
                config,
            )
        )
        self.assertEqual(confirmations.item(), 2)
        self.assertGreater(alignment.item(), 0.0)
        _, confirmations, _, alignment = update_confirmed_hidden_memory(
            memory,
            confirmations,
            idle,
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor(0.1),
            torch.tensor(1),
            torch.tensor(False),
            config,
        )
        self.assertEqual(confirmations.item(), 1)
        self.assertLess(alignment.item(), 0.0)

    def test_config_rejects_unsafe_injection_strength(self) -> None:
        with self.assertRaises(ValueError):
            self.config(alpha=0.5).validate()
        with self.assertRaises(ValueError):
            self.config(direction="unknown").validate()
        with self.assertRaises(ValueError):
            self.config(vocab_head_map=(0, 1, 3)).validate()
        with self.assertRaises(ValueError):
            self.config(vocab_route_decay=1.0).validate()
        with self.assertRaises(ValueError):
            self.config(vocab_route_prior_gain=-0.1).validate()
        with self.assertRaises(ValueError):
            self.config(vocab_route_gain_clip=0.0).validate()
        with self.assertRaises(ValueError):
            self.config(hidden_consistency_threshold=1.1).validate()
        with self.assertRaises(ValueError):
            self.config(hidden_confirmation_blocks=0).validate()
        with self.assertRaises(ValueError):
            self.config(hidden_unconfirmed_scale=1.1).validate()
        with self.assertRaises(ValueError):
            self.config(root_anchor_mix=1.1).validate()
        with self.assertRaises(ValueError):
            self.config(compatibility_mode="unknown").validate()
        with self.assertRaises(ValueError):
            self.config(compatibility_source="unknown").validate()
        with self.assertRaises(ValueError):
            self.config(strength_reference=-0.1).validate()
        with self.assertRaises(ValueError):
            self.config(scale_gap_reference=0.0).validate()
        with self.assertRaises(ValueError):
            self.config(scale_min=1.1).validate()
        with self.assertRaises(ValueError):
            self.config(scale_min=0.75, scale_max=1.25, scale_initial=1.5).validate()
        with self.assertRaises(ValueError):
            self.config(repetition_ngram_size=1).validate()
        with self.assertRaises(ValueError):
            self.config(repetition_penalty=0.0).validate()
        with self.assertRaises(ValueError):
            self.config(repetition_debt_reference=0.0).validate()
        with self.assertRaises(ValueError):
            self.config(adaptive_min_depth=3).validate()
        with self.assertRaises(ValueError):
            self.config(adaptive_depth_blocks=0).validate()
        with self.assertRaises(ValueError):
            self.config(adaptive_depth_trigger=0.0).validate()
        with self.assertRaises(ValueError):
            self.config(vocab_bias_scale=1.1).validate()
        with self.assertRaises(ValueError):
            self.config(vocab_bias_top_k=0).validate()
        with self.assertRaises(ValueError):
            self.config(fusion_gate_threshold=1.1).validate()
        with self.assertRaises(ValueError):
            self.config(verifier_risk_strength=-0.1).validate()
        with self.assertRaises(ValueError):
            self.config(verifier_risk_reference=0.0).validate()
        with self.assertRaises(ValueError):
            self.config(verifier_policy="unknown").validate()
        with self.assertRaises(ValueError):
            self.config(verifier_cactus_mix=1.1).validate()
        with self.assertRaises(ValueError):
            self.config(verifier_budget_slope=1.1).validate()
        with self.assertRaises(ValueError):
            self.config(verifier_tail_threshold=1.0).validate()


if __name__ == "__main__":
    unittest.main()
