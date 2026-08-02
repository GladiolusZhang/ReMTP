import types
import unittest
from unittest import mock

import torch

import remtp.cactus_mtp as cactus_module
from remtp.cactus_mtp import cactus_target_distribution
from remtp.cactus_regret_feedback import (
    CactusRegretConfig,
    accepted_prefix_mask,
    cactus_residual_direction,
    cactus_hidden_residual_direction,
    cactus_boundary_hidden_residual_direction,
    classify_cactus_causal_acceptance,
    inject_cactus_regret,
    inject_cactus_regret_at_root,
    posterior_causal_responsibility,
)


class CactusRegretFeedbackTest(unittest.TestCase):
    def test_config_accepts_zero_alpha_identity_ablation(self) -> None:
        CactusRegretConfig(alpha=0.0).validate()
        with self.assertRaises(ValueError):
            CactusRegretConfig(alpha=0.11).validate()
        with self.assertRaises(ValueError):
            CactusRegretConfig(strength_reference=0.0).validate()
        with self.assertRaises(ValueError):
            CactusRegretConfig(injection_site="other").validate()
        with self.assertRaises(ValueError):
            CactusRegretConfig(residual_space="other").validate()

    def test_prefix_mask_stops_at_first_non_draft_token(self) -> None:
        output = torch.tensor([[10, 20, 99, 40, 50]])
        drafts = torch.tensor([10, 20, 30, 40])
        torch.testing.assert_close(
            accepted_prefix_mask(output, drafts),
            torch.tensor([True, True, False, False]),
        )

    def test_shared_uniform_identifies_only_cactus_caused_acceptance(self) -> None:
        target = torch.tensor(
            [
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
            ]
        )
        cactus = torch.tensor(
            [
                [0.6, 0.4],
                [0.6, 0.4],
                [0.6, 0.4],
            ]
        )
        draft = torch.tensor(
            [
                [0.8, 0.2],
                [0.8, 0.2],
                [0.8, 0.2],
            ]
        )
        strict, causal, strict_a, cactus_a = (
            classify_cactus_causal_acceptance(
                torch.tensor([True, True, False]),
                target,
                cactus,
                draft,
                torch.tensor([0, 0, 0]),
                torch.tensor([0.10, 0.50, 0.50]),
            )
        )
        torch.testing.assert_close(strict_a, torch.tensor([0.25, 0.25, 0.25]))
        torch.testing.assert_close(cactus_a, torch.tensor([0.75, 0.75, 0.75]))
        torch.testing.assert_close(strict, torch.tensor([True, False, False]))
        torch.testing.assert_close(causal, torch.tensor([False, True, False]))

    def test_posterior_causal_responsibility_is_continuous_and_unbiased(self) -> None:
        responsibility = posterior_causal_responsibility(
            torch.tensor([True, True, False]),
            torch.tensor([0.25, 0.50, 0.25]),
            torch.tensor([0.75, 0.75, 0.75]),
        )
        torch.testing.assert_close(
            responsibility,
            torch.tensor([2.0 / 3.0, 1.0 / 3.0, 0.0]),
        )
        # Multiplying the posterior responsibility by P(Cactus accepts)
        # recovers the causal acceptance mass A_cactus - A_strict.
        torch.testing.assert_close(
            responsibility[:2] * torch.tensor([0.75, 0.75]),
            torch.tensor([0.50, 0.25]),
        )

    def test_residual_direction_points_from_candidate_to_target_preference(self) -> None:
        target = torch.tensor([[0.2, 0.7, 0.1]])
        cactus = torch.tensor([[0.6, 0.35, 0.05]])
        output_weight = torch.eye(3)
        direction, strength, transferred = cactus_residual_direction(
            target,
            cactus,
            torch.tensor([0]),
            torch.tensor([True]),
            torch.tensor(1),
            output_weight,
            top_k=2,
            depth_decay=0.9,
        )
        self.assertAlmostEqual(strength.item(), 0.4, places=6)
        self.assertAlmostEqual(transferred.item(), 0.4, places=6)
        self.assertLess(direction[0].item(), 0.0)
        self.assertGreater(direction[1].item(), 0.0)

    def test_no_causal_event_clears_direction(self) -> None:
        target = torch.tensor([[0.2, 0.8]])
        cactus = torch.tensor([[0.6, 0.4]])
        direction, strength, _ = cactus_residual_direction(
            target,
            cactus,
            torch.tensor([0]),
            torch.tensor([False]),
            torch.tensor(1),
            torch.eye(2),
            top_k=1,
            depth_decay=0.9,
        )
        torch.testing.assert_close(direction, torch.zeros(2))
        self.assertEqual(strength.item(), 0.0)

    def test_hidden_residual_points_from_mtp_prediction_to_target_fact(self) -> None:
        direction, strength, transferred = cactus_hidden_residual_direction(
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.tensor([[0.2, 0.8]]),
            torch.tensor([[0.6, 0.4]]),
            torch.tensor([0]),
            torch.tensor([0.5]),
            torch.tensor(1),
            depth_decay=0.9,
        )
        self.assertAlmostEqual(strength.item(), 0.2, places=6)
        self.assertAlmostEqual(transferred.item(), 0.4, places=6)
        self.assertLess(direction[0].item(), 0.0)
        self.assertGreater(direction[1].item(), 0.0)

    def test_boundary_hidden_residual_requires_full_draft_commit(self) -> None:
        args = (
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            torch.tensor([[0.2, 0.8], [0.2, 0.8]]),
            torch.tensor([[0.6, 0.4], [0.6, 0.4]]),
            torch.tensor([0, 0]),
            torch.tensor([0.5, 0.5]),
        )
        direction, strength, _ = cactus_boundary_hidden_residual_direction(
            *args,
            torch.tensor(2),
            depth_decay=0.9,
        )
        self.assertGreater(strength.item(), 0.0)
        self.assertLess(direction[0].item(), 0.0)
        self.assertGreater(direction[1].item(), 0.0)

        rejected_direction, rejected_strength, _ = (
            cactus_boundary_hidden_residual_direction(
                *args,
                torch.tensor(1),
                depth_decay=0.9,
            )
        )
        self.assertEqual(rejected_strength.item(), 0.0)
        torch.testing.assert_close(rejected_direction, torch.zeros(2))

    def test_injection_preserves_rms_and_zero_alpha_is_exact_identity(self) -> None:
        hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        direction = torch.tensor([-1.0, 1.0, 0.5, -0.5])
        identity = inject_cactus_regret(
            hidden,
            direction,
            torch.tensor(1.0),
            alpha=0.0,
        )
        self.assertIs(identity, hidden)
        steered = inject_cactus_regret(
            hidden,
            direction,
            torch.tensor(1.0),
            alpha=0.03,
        )
        torch.testing.assert_close(
            steered.square().mean().sqrt(),
            hidden.square().mean().sqrt(),
        )
        self.assertFalse(torch.equal(steered, hidden))

    def test_root_injection_changes_only_selected_request_end_rows(self) -> None:
        hidden = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )
        corrected = inject_cactus_regret_at_root(
            hidden,
            torch.tensor([1, 3]),
            torch.tensor([-1.0, 1.0]),
            torch.tensor(1.0),
            alpha=0.03,
        )
        torch.testing.assert_close(corrected[0], hidden[0])
        torch.testing.assert_close(corrected[2], hidden[2])
        self.assertFalse(torch.equal(corrected[1], hidden[1]))
        self.assertFalse(torch.equal(corrected[3], hidden[3]))

    def test_cactus_observer_does_not_change_verifier_or_output(self) -> None:
        drafts = torch.tensor([0])
        draft_probs = torch.tensor([[0.8, 0.2]])
        target_logits = torch.log(torch.tensor([[0.2, 0.8]]))
        expected = torch.tensor([[123, 456]])
        observed: dict[str, torch.Tensor] = {}

        def original(*args):
            observed["verifier"] = torch.softmax(args[5], dim=-1)
            return expected

        def hook(**kwargs):
            observed["hook_cactus"] = kwargs["cactus_probs"]

        wrapper = cactus_module._cactus_rejection_sample_v1
        previous_original = getattr(wrapper, "_remtp_original", None)
        previous_hook = cactus_module._POST_VERIFICATION_HOOK
        wrapper._remtp_original = original
        cactus_module.set_post_verification_hook(hook)
        try:
            with mock.patch.dict(
                "os.environ", {"REMTP_CACTUS_DIAGNOSTICS": "0"}
            ):
                result = wrapper(
                    drafts,
                    [1],
                    1,
                    torch.tensor([0, 1]),
                    draft_probs,
                    target_logits,
                    torch.tensor([[1]]),
                    types.SimpleNamespace(all_greedy=False),
                )
        finally:
            cactus_module.set_post_verification_hook(previous_hook)
            if previous_original is None:
                delattr(wrapper, "_remtp_original")
            else:
                wrapper._remtp_original = previous_original
        expected_cactus = cactus_target_distribution(
            target_logits.softmax(dim=-1), drafts, delta=1.0
        )
        self.assertIs(result, expected)
        torch.testing.assert_close(observed["verifier"], expected_cactus)
        torch.testing.assert_close(observed["hook_cactus"], expected_cactus)


if __name__ == "__main__":
    unittest.main()
