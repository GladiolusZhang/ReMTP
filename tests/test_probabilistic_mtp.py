import unittest

import torch

from remtp.probabilistic_mtp import sample_mtp_logits


class ProbabilisticMTPTest(unittest.TestCase):
    def test_returns_normalized_full_distribution(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        token_ids, probs = sample_mtp_logits(
            logits,
            torch.tensor([0.7]),
            {},
        )
        self.assertEqual(token_ids.shape, (1,))
        self.assertEqual(probs.shape, logits.shape)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1))
        self.assertGreater(torch.count_nonzero(probs).item(), 1)

    def test_seeded_sampling_is_reproducible(self) -> None:
        logits = torch.tensor([[0.2, 0.1, 0.0]])
        first_generator = torch.Generator().manual_seed(42)
        second_generator = torch.Generator().manual_seed(42)

        first_token, first_probs = sample_mtp_logits(
            logits,
            torch.tensor([0.7]),
            {0: first_generator},
        )
        second_token, second_probs = sample_mtp_logits(
            logits,
            torch.tensor([0.7]),
            {0: second_generator},
        )
        torch.testing.assert_close(first_token, second_token)
        torch.testing.assert_close(first_probs, second_probs)

    def test_zero_temperature_remains_greedy(self) -> None:
        logits = torch.tensor([[0.1, 2.0, 0.5]])
        token_ids, _ = sample_mtp_logits(
            logits,
            torch.tensor([0.0]),
            {},
        )
        self.assertEqual(token_ids.item(), 1)


if __name__ == "__main__":
    unittest.main()
