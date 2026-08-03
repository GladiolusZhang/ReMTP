import unittest

import torch

from remtp.regret_router_selective_train import oracle_logit_targets


class SelectiveRegretRouterTrainTest(unittest.TestCase):
    def batch(self, debt: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "p_compact": torch.tensor([[[0.90, 0.10]], [[0.60, 0.40]]]),
            "q_compact": torch.tensor([[[0.60, 0.40]], [[0.90, 0.10]]]),
            "support_mask": torch.ones(2, 1, 2, dtype=torch.bool),
            "regret_debt": debt,
        }

    def test_oracle_sharpens_or_softens_toward_target(self) -> None:
        result = oracle_logit_targets(
            self.batch(torch.tensor([0.1, 0.1])),
            debt_reference=0.05,
            max_logit_scale=0.8,
            grid_size=33,
            min_oracle_gain=1e-6,
            intervention_weight=0.0,
            entropy_weight=0.0,
        )
        self.assertGreater(result["target_scale"][0, 0].item(), 0.0)
        self.assertLess(result["target_scale"][1, 0].item(), 0.0)
        self.assertTrue(torch.all(result["oracle_gain"] > 0.0).item())

    def test_zero_debt_is_an_exact_abstention_label(self) -> None:
        result = oracle_logit_targets(
            self.batch(torch.zeros(2)),
            debt_reference=0.05,
            max_logit_scale=0.8,
            grid_size=33,
            min_oracle_gain=1e-6,
            intervention_weight=0.0,
            entropy_weight=0.0,
        )
        torch.testing.assert_close(
            result["target_scale"], torch.zeros(2, 1)
        )
        self.assertFalse(result["actionable"].any().item())


if __name__ == "__main__":
    unittest.main()
