import unittest

import torch

from remtp.prefix_saturation_trim import (
    cactus_candidate_probabilities,
    candidate_shift_distribution,
    trim_prefix_saturation,
)


class PrefixSaturationTrimTest(unittest.TestCase):
    def test_trim_preserves_repaired_prefix(self) -> None:
        p_y = torch.tensor([0.1, 0.1])
        q_y = torch.tensor([0.4, 0.4])
        cactus_h = torch.tensor([0.3, 0.8])
        trimmed, cactus_prefix, trimmed_prefix = trim_prefix_saturation(
            p_y,
            q_y,
            cactus_h,
        )
        torch.testing.assert_close(cactus_prefix, torch.tensor([0.75, 1.0]))
        torch.testing.assert_close(trimmed_prefix, cactus_prefix)
        self.assertAlmostEqual(trimmed[0].item(), 0.3, places=6)
        self.assertAlmostEqual(trimmed[1].item(), 0.4 / 0.75, places=6)
        self.assertLess(trimmed[1].item(), cactus_h[1].item())

    def test_trim_never_moves_below_target(self) -> None:
        p_y = torch.tensor([0.7])
        q_y = torch.tensor([0.2])
        cactus_h = torch.tensor([0.9])
        trimmed, _, _ = trim_prefix_saturation(p_y, q_y, cactus_h)
        self.assertGreaterEqual(trimmed.item(), p_y.item())

    def test_random_blocks_preserve_every_prefix(self) -> None:
        generator = torch.Generator().manual_seed(7)
        for _ in range(20):
            p_y = 0.01 + 0.8 * torch.rand(6, generator=generator)
            q_y = 0.01 + 0.8 * torch.rand(6, generator=generator)
            cactus_h = p_y + (1.0 - p_y) * torch.rand(
                6,
                generator=generator,
            )
            trimmed, cactus_prefix, trimmed_prefix = trim_prefix_saturation(
                p_y,
                q_y,
                cactus_h,
            )
            torch.testing.assert_close(
                trimmed_prefix,
                cactus_prefix,
                rtol=1e-5,
                atol=1e-6,
            )
            self.assertTrue(torch.all(trimmed >= p_y - 1e-7))
            self.assertTrue(torch.all(trimmed <= cactus_h + 1e-7))

    def test_cactus_candidate_formula(self) -> None:
        candidate = cactus_candidate_probabilities(torch.tensor([0.5]), 0.5)
        self.assertAlmostEqual(candidate.item(), 1.0, places=6)

    def test_shifted_distribution_is_normalized(self) -> None:
        shifted = candidate_shift_distribution(
            torch.tensor([[0.2, 0.3, 0.5]]),
            torch.tensor([0]),
            torch.tensor([0.6]),
        )
        torch.testing.assert_close(shifted, torch.tensor([[0.6, 0.15, 0.25]]))


if __name__ == "__main__":
    unittest.main()
