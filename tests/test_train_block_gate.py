import unittest

from remtp.block_gate import FEATURE_NAMES, LogisticBlockGate
from remtp.train_block_gate import train_logistic_gate


class TrainBlockGateTest(unittest.TestCase):
    def test_synthetic_grouped_training_exports_runtime_model(self) -> None:
        records = []
        for group in range(10):
            for offset in range(4):
                positive = (group + offset) % 2 == 0
                features = [0.0] * len(FEATURE_NAMES)
                features[0] = 1.0 if positive else -1.0
                records.append(
                    {
                        "request_group": group,
                        "feature_names": list(FEATURE_NAMES),
                        "features": features,
                        "label": int(positive),
                        "expected_length_delta": 0.01 if positive else -0.1,
                        "block_verification_surplus": 0.2,
                        "risk_reduction": 0.1 if positive else 0.0,
                    }
                )
        payload = train_logistic_gate(
            records,
            seed=7,
            validation_fraction=0.2,
            max_false_positive_rate=0.0,
        )
        model = LogisticBlockGate(
            feature_names=tuple(payload["feature_names"]),
            mean=tuple(payload["mean"]),
            scale=tuple(payload["scale"]),
            weight=tuple(payload["weight"]),
            bias=payload["bias"],
            threshold=payload["threshold"],
        )
        model.validate()
        self.assertGreater(payload["training"]["validation_route_fraction"], 0)
        self.assertEqual(
            payload["training"]["validation_false_positive_rate"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
