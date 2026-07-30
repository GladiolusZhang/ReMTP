import unittest

import torch

from remtp.regret_feedback import (
    RegretFeedbackConfig,
    accepted_prefix_mask,
    compatibility_gate,
    inject_regret,
    regret_block_vector,
    strict_and_causal_masks,
    update_regret_memory,
)


class RegretFeedbackTest(unittest.TestCase):
    @staticmethod
    def config(**overrides: object) -> RegretFeedbackConfig:
        values: dict[str, object] = {
            "expected_draft_tokens": 3,
            "regret_top_k": 2,
            "compatibility_top_k": 3,
            "head_reliability": (1.0, 0.7, 0.3),
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

    def test_config_rejects_unsafe_injection_strength(self) -> None:
        with self.assertRaises(ValueError):
            self.config(alpha=0.5).validate()


if __name__ == "__main__":
    unittest.main()
