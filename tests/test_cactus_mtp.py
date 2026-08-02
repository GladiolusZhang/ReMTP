import unittest

import torch

from remtp.cactus_mtp import (
    cactus_target_distribution,
    sparse_checkpoint_target_distribution,
)


class CactusMTPTest(unittest.TestCase):
    def test_zero_delta_recovers_standard_target(self) -> None:
        target = torch.tensor([[0.2, 0.3, 0.5]])
        cactus = cactus_target_distribution(
            target,
            torch.tensor([0]),
            delta=0.0,
        )
        torch.testing.assert_close(cactus, target)

    def test_selected_probability_gets_paper_bonus(self) -> None:
        target = torch.tensor([[0.2, 0.3, 0.5]])
        cactus = cactus_target_distribution(
            target,
            torch.tensor([0]),
            delta=0.5,
        )
        torch.testing.assert_close(
            cactus,
            torch.tensor([[0.6, 0.15, 0.25]]),
        )

    def test_distribution_is_normalized_when_gamma_clips(self) -> None:
        target = torch.tensor([[0.4, 0.35, 0.25]])
        cactus = cactus_target_distribution(
            target,
            torch.tensor([1]),
            delta=100.0,
        )
        torch.testing.assert_close(cactus, torch.tensor([[0.0, 1.0, 0.0]]))
        torch.testing.assert_close(cactus.sum(dim=-1), torch.ones(1))

    def test_small_delta_respects_local_kl_budget(self) -> None:
        target = torch.tensor([[0.1, 0.4, 0.5]])
        delta = 0.01
        cactus = cactus_target_distribution(
            target,
            torch.tensor([0]),
            delta=delta,
        )
        kl = (
            cactus
            * (
                torch.log(cactus.clamp_min(1e-30))
                - torch.log(target.clamp_min(1e-30))
            )
        ).sum()
        self.assertLessEqual(kl.item(), delta + 1e-6)

    def test_negative_delta_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cactus_target_distribution(
                torch.tensor([[0.5, 0.5]]),
                torch.tensor([0]),
                delta=-0.1,
            )

    def test_sparse_checkpoint_changes_only_target_opposed_rows(self) -> None:
        target = torch.tensor(
            [
                [0.2, 0.3, 0.5],
                [0.01, 0.09, 0.90],
            ]
        )
        strict, checkpoint = sparse_checkpoint_target_distribution(
            target,
            torch.tensor([0, 0]),
            delta=0.5,
            max_log_gap=2.0,
        )
        expected_safe = cactus_target_distribution(
            target[:1],
            torch.tensor([0]),
            delta=0.5,
        )
        torch.testing.assert_close(strict[:1], expected_safe)
        torch.testing.assert_close(strict[1:], target[1:])
        torch.testing.assert_close(
            checkpoint,
            torch.tensor([False, True]),
        )

    def test_sparse_checkpoint_rejects_negative_gap(self) -> None:
        with self.assertRaises(ValueError):
            sparse_checkpoint_target_distribution(
                torch.tensor([[0.5, 0.5]]),
                torch.tensor([0]),
                delta=1.0,
                max_log_gap=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
