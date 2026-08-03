import tempfile
import unittest
from pathlib import Path

import torch

from remtp.regret_router_model import (
    RegretRouter,
    RegretRouterArchitecture,
    load_regret_router_checkpoint,
    save_regret_router_checkpoint,
)


class RegretRouterModelTest(unittest.TestCase):
    def architecture(self) -> RegretRouterArchitecture:
        return RegretRouterArchitecture(
            hidden_size=8,
            num_heads=3,
            rank=2,
            width=6,
        )

    def test_controls_are_bounded_and_zero_debt_is_identity(self) -> None:
        model = RegretRouter(self.architecture())
        controls = model(
            torch.randn(2, 8),
            torch.randn(2, 8),
            torch.zeros(2),
            torch.tensor([0.4, 0.5]),
            torch.tensor([0.2, 0.3]),
            torch.tensor([1.0, 0.6, 0.3]),
        )
        torch.testing.assert_close(
            controls.direction_strength,
            torch.zeros(2, 3),
        )
        torch.testing.assert_close(controls.logit_scale, torch.zeros(2, 3))
        torch.testing.assert_close(controls.budget_scale, torch.ones(2, 3))

    def test_per_head_target_features_are_supported(self) -> None:
        model = RegretRouter(self.architecture())
        controls = model(
            torch.randn(1, 8),
            torch.randn(1, 8),
            torch.ones(1),
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.3, 0.2, 0.1]]),
            torch.tensor([1.0, 0.6, 0.3]),
        )
        self.assertEqual(controls.direction_strength.shape, (1, 3))
        self.assertTrue(torch.all(controls.budget_scale <= 1.0).item())
        self.assertTrue(torch.all(controls.budget_scale >= 0.5).item())

    def test_checkpoint_roundtrip(self) -> None:
        model = RegretRouter(self.architecture())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.pt"
            save_regret_router_checkpoint(path, model, metadata={"steps": 3})
            loaded, metadata = load_regret_router_checkpoint(path)
        self.assertEqual(loaded.architecture, model.architecture)
        self.assertEqual(metadata["steps"], 3)
        for expected, actual in zip(model.parameters(), loaded.parameters()):
            torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
