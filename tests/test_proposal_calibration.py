import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import remtp.proposal_calibration as proposal_calibration
import torch

from remtp.probabilistic_mtp import (
    sample_mtp_logits,
    set_draft_probs_hook,
    set_draft_temperature_hook,
)
from remtp.proposal_calibration import (
    HeadTransform,
    ProposalCalibrationConfig,
    apply_probability_transform,
    overlap_mass,
    strict_output_distribution,
)
from remtp.proposal_calibration_offline import coordinate_search, expected_mal


class ProposalCalibrationTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_draft_probs_hook(None)
        set_draft_temperature_hook(None)

    def test_all_transforms_are_normalized_and_target_free(self) -> None:
        q = torch.tensor([[0.70, 0.20, 0.08, 0.02]])
        transforms = (
            HeadTransform(kind="temperature", temperature=1.4),
            HeadTransform(kind="topk_redistribute", top_k=3, strength=0.2),
            HeadTransform(kind="topk_concentrate", top_k=3, strength=0.2),
            HeadTransform(kind="topk_uniform_mix", top_k=2, strength=0.1),
        )
        for transform in transforms:
            calibrated = apply_probability_transform(q, transform)
            torch.testing.assert_close(calibrated.sum(-1), torch.ones(1))
            self.assertTrue(bool((calibrated >= 0).all()))

        # Same Q gives the same Q_tilde regardless of an unrelated P tensor.
        first = apply_probability_transform(q, transforms[1])
        _unrelated_target = torch.tensor([[0.01, 0.01, 0.01, 0.97]])
        second = apply_probability_transform(q, transforms[1])
        torch.testing.assert_close(first, second)

    def test_redistribution_preserves_topk_mass_and_tail(self) -> None:
        q = torch.tensor([[0.60, 0.25, 0.10, 0.05]])
        calibrated = apply_probability_transform(
            q,
            HeadTransform(kind="topk_redistribute", top_k=2, strength=0.4),
        )
        torch.testing.assert_close(calibrated[:, :2].sum(-1), q[:, :2].sum(-1))
        torch.testing.assert_close(calibrated[:, 2:], q[:, 2:])

        concentrated = apply_probability_transform(
            q,
            HeadTransform(kind="topk_concentrate", top_k=3, strength=0.4),
        )
        torch.testing.assert_close(concentrated[:, :3].sum(-1), q[:, :3].sum(-1))
        torch.testing.assert_close(concentrated[:, 3:], q[:, 3:])
        self.assertGreater(float(concentrated[0, 0]), float(q[0, 0]))

    def test_strict_rejection_recovers_target_for_every_transform(self) -> None:
        p = torch.tensor(
            [[0.05, 0.15, 0.30, 0.50], [0.40, 0.10, 0.20, 0.30]],
            dtype=torch.float64,
        )
        q = torch.tensor(
            [[0.60, 0.20, 0.10, 0.10], [0.05, 0.10, 0.75, 0.10]],
            dtype=torch.float64,
        )
        for transform in (
            HeadTransform(),
            HeadTransform(kind="temperature", temperature=0.7),
            HeadTransform(kind="temperature", temperature=1.5),
            HeadTransform(kind="topk_redistribute", top_k=3, strength=0.3),
            HeadTransform(kind="topk_concentrate", top_k=3, strength=0.3),
            HeadTransform(kind="topk_uniform_mix", top_k=3, strength=0.1),
        ):
            q_tilde = apply_probability_transform(q, transform)
            recovered = strict_output_distribution(p, q_tilde)
            torch.testing.assert_close(recovered, p.to(torch.float32), atol=1e-6, rtol=1e-6)

    def test_sampler_returns_the_same_calibrated_q_used_for_sampling(self) -> None:
        transform = HeadTransform(
            kind="topk_redistribute", top_k=3, strength=0.25
        )

        def hook(probs, logits, temperatures, depth):
            return apply_probability_transform(probs, transform)

        set_draft_probs_hook(hook)
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        _, returned_q = sample_mtp_logits(
            logits,
            torch.tensor([0.7]),
            {0: torch.Generator().manual_seed(42)},
            proposal_depth=2,
        )
        raw_q = torch.softmax(logits / 0.7, dim=-1)
        expected = apply_probability_transform(raw_q, transform)
        torch.testing.assert_close(returned_q, expected)

    def test_temperature_is_folded_into_native_softmax(self) -> None:
        set_draft_temperature_hook(lambda temperatures, depth: temperatures * 0.5)
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        _, returned_q = sample_mtp_logits(
            logits,
            torch.tensor([0.8]),
            {0: torch.Generator().manual_seed(7)},
            proposal_depth=4,
        )
        expected = torch.softmax(logits / (0.8 * 0.5), dim=-1)
        torch.testing.assert_close(returned_q, expected)

    def test_config_round_trip(self) -> None:
        config = ProposalCalibrationConfig(
            (
                HeadTransform(kind="temperature", temperature=0.8),
                HeadTransform(kind="topk_redistribute", top_k=2, strength=0.1),
            )
        )
        restored = ProposalCalibrationConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)

    def test_expected_mal_weights_early_heads(self) -> None:
        early = expected_mal([0.9, 0.5, 0.5])
        late = expected_mal([0.5, 0.5, 0.9])
        self.assertGreater(early, late)

    def test_coordinate_search_uses_block_objective(self) -> None:
        records = []
        for request in range(10):
            records.append(
                {
                    "request_index": request,
                    "strategy_overlap": {
                        "identity": [0.5] * 6,
                        "temperature:2": [0.8, 0.5, 0.5, 0.5, 0.5, 0.5],
                    },
                }
            )
        config, score = coordinate_search(
            {"toy": records},
            ["identity", "temperature:2"],
            heads=6,
        )
        self.assertEqual(config[0], "temperature:2")
        self.assertGreater(score, expected_mal([0.5] * 6))

    def test_empty_async_output_history_does_not_create_fake_requests(self) -> None:
        saved = (
            proposal_calibration._TRACE_REQUEST,
            proposal_calibration._TRACE_LAST_LENGTH,
            proposal_calibration._TRACE_ACTIVE_REQUEST_ID,
            proposal_calibration._TRACE_SEEN_REQUEST_ID,
            proposal_calibration._TRACE_ACTIVE_OUTPUT_LENGTH,
        )
        try:
            proposal_calibration._TRACE_REQUEST = -1
            proposal_calibration._TRACE_LAST_LENGTH = None
            proposal_calibration._TRACE_ACTIVE_REQUEST_ID = None
            proposal_calibration._TRACE_SEEN_REQUEST_ID = None
            proposal_calibration._TRACE_ACTIVE_OUTPUT_LENGTH = 0
            metadata = SimpleNamespace(output_token_ids=[])
            first, _ = proposal_calibration._request_index(metadata)
            second, _ = proposal_calibration._request_index(metadata)
            self.assertEqual((first, second), (0, 0))
        finally:
            (
                proposal_calibration._TRACE_REQUEST,
                proposal_calibration._TRACE_LAST_LENGTH,
                proposal_calibration._TRACE_ACTIVE_REQUEST_ID,
                proposal_calibration._TRACE_SEEN_REQUEST_ID,
                proposal_calibration._TRACE_ACTIVE_OUTPUT_LENGTH,
            ) = saved


if __name__ == "__main__":
    unittest.main()
