import json
import tempfile
import unittest
from pathlib import Path

import torch

from remtp.block_gate import (
    FEATURE_NAMES,
    LogisticBlockGate,
    block_gate_features,
    candidate_shift_distribution,
    counterfactual_gate_record,
    logistic_gate_probability,
)


class BlockGateTest(unittest.TestCase):
    def test_features_have_frozen_schema_and_finite_values(self) -> None:
        features = block_gate_features(
            torch.tensor([0.4, 0.2]),
            torch.tensor([0.5, 0.4]),
            torch.tensor([0.2, 1.5]),
            torch.tensor([0.6, 0.5]),
            torch.tensor([0.5, 0.4]),
        )
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertTrue(torch.isfinite(features).all())

    def test_candidate_shift_preserves_normalization_and_target_ratios(self) -> None:
        target = torch.tensor([[0.2, 0.3, 0.5]])
        shifted = candidate_shift_distribution(
            target,
            torch.tensor([0]),
            torch.tensor([0.6]),
        )
        torch.testing.assert_close(shifted.sum(dim=-1), torch.ones(1))
        torch.testing.assert_close(shifted, torch.tensor([[0.6, 0.15, 0.25]]))

    def test_counterfactual_record_is_serializable(self) -> None:
        target = torch.tensor([[0.4, 0.6], [0.7, 0.3]])
        draft = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        record = counterfactual_gate_record(
            target_probs=target,
            draft_probs=draft,
            draft_token_ids=torch.tensor([0, 0]),
            target_candidate_probs=torch.tensor([0.4, 0.7]),
            draft_candidate_probs=torch.tensor([0.5, 0.5]),
            target_log_gaps=torch.tensor([0.4, 0.0]),
            cactus_candidate_probs=torch.tensor([0.7, 0.9]),
            shield_candidate_probs=torch.tensor([0.6, 0.9]),
        )
        self.assertEqual(len(record["features"]), len(FEATURE_NAMES))
        self.assertIn(record["label"], (0, 1))
        self.assertGreaterEqual(record["block_verification_surplus"], 0.0)
        self.assertAlmostEqual(
            record["allowed_surplus_spend"],
            0.5 * record["block_verification_surplus"],
        )
        json.dumps(record)

    def test_logistic_model_round_trip(self) -> None:
        width = len(FEATURE_NAMES)
        payload = {
            "feature_names": list(FEATURE_NAMES),
            "mean": [0.0] * width,
            "scale": [1.0] * width,
            "weight": [0.0] * width,
            "bias": 0.0,
            "threshold": 0.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            model = LogisticBlockGate.load(path)
        mean, scale, weight, bias = model.tensors(device=torch.device("cpu"))
        probability = logistic_gate_probability(
            torch.ones(width),
            mean,
            scale,
            weight,
            bias,
        )
        self.assertEqual(probability.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
