import unittest

import torch

from remtp.speculative_cascade import (
    recover_unscaled_probs,
    target_distribution,
)


class SpeculativeCascadeTest(unittest.TestCase):
    def test_opt_selects_target_or_draft_distribution(self) -> None:
        draft = torch.tensor([[0.5, 0.5]])
        target = torch.tensor([[0.9, 0.1]])

        deferred = target_distribution(
            draft,
            target,
            draft,
            target,
            rule="opt",
            alpha=0.5,
        )
        kept = target_distribution(
            draft,
            target,
            draft,
            target,
            rule="opt",
            alpha=2.0,
        )
        torch.testing.assert_close(deferred, target)
        torch.testing.assert_close(kept, draft)

    def test_token_v3_redistributes_rejected_draft_mass(self) -> None:
        draft = torch.tensor([[0.7, 0.3]])
        target = torch.tensor([[0.4, 0.6]])

        strict = target_distribution(
            draft,
            target,
            draft,
            target,
            rule="token_v3",
            alpha=0.0,
        )
        lenient = target_distribution(
            draft,
            target,
            draft,
            target,
            rule="token_v3",
            alpha=1.0,
        )
        torch.testing.assert_close(
            strict,
            torch.tensor([[0.28, 0.72]]),
        )
        torch.testing.assert_close(lenient, draft)
        torch.testing.assert_close(strict.sum(dim=-1), torch.ones(1))

    def test_token_v3_with_deterministic_mtp_proposal(self) -> None:
        target = torch.tensor([[0.6, 0.3, 0.1]])
        deterministic_draft = torch.tensor([[0.0, 1.0, 0.0]])

        accepted = target_distribution(
            deterministic_draft,
            target,
            deterministic_draft,
            target,
            rule="token_v3",
            alpha=0.5,
        )
        deferred = target_distribution(
            deterministic_draft,
            target,
            deterministic_draft,
            target,
            rule="token_v3",
            alpha=0.4,
        )
        torch.testing.assert_close(accepted, deterministic_draft)
        torch.testing.assert_close(deferred, target)

    def test_temperature_unscaling(self) -> None:
        raw_logits = torch.tensor([[2.0, 0.0]])
        temperature = torch.tensor([0.5])
        processed_logits = raw_logits / temperature
        recovered = recover_unscaled_probs(
            processed_logits,
            temperature,
        )
        torch.testing.assert_close(
            recovered,
            torch.softmax(raw_logits, dim=-1),
        )

    def test_invalid_token_v3_alpha(self) -> None:
        probs = torch.tensor([[0.5, 0.5]])
        with self.assertRaises(ValueError):
            target_distribution(
                probs,
                probs,
                probs,
                probs,
                rule="token_v3",
                alpha=1.1,
            )


if __name__ == "__main__":
    unittest.main()
