from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from remtp.checkpoint import missing_checkpoint_files
from remtp.trace import (
    _trace_legacy_rejection_result,
    format_round,
    format_stochastic_round,
)


class FormatRoundTest(unittest.TestCase):
    @patch("remtp.trace.token_label", side_effect=lambda token_id: str(token_id))
    def test_all_drafts_accepted(self, _label) -> None:
        output = format_round(
            round_number=1,
            request_id="req-1",
            previous_token=10,
            draft_tokens=[20, 30],
            target_tokens=[20, 30, 40],
            emitted_tokens=[20, 30, 40],
            accepted_count=2,
        )

        self.assertIn("TARGET previous : 10", output)
        self.assertIn("D0=20  D1=30", output)
        self.assertIn("T0=20  T1=30  bonus=40", output)
        self.assertIn("D0==T0 ✓  D1==T1 ✓", output)
        self.assertIn("accepted 2/2", output)
        self.assertIn("COMMIT          : 20 30 40", output)

    @patch("remtp.trace.token_label", side_effect=lambda token_id: str(token_id))
    def test_first_mismatch_stops_verification(self, _label) -> None:
        output = format_round(
            round_number=2,
            request_id="req-2",
            previous_token=10,
            draft_tokens=[20, 31],
            target_tokens=[20, 30, 99],
            emitted_tokens=[20, 30],
            accepted_count=1,
        )

        self.assertIn("D0==T0 ✓  D1!=T1 ✗", output)
        self.assertIn("accepted 1/2", output)
        self.assertIn("COMMIT          : 20 30", output)

    @patch("remtp.trace.token_label", side_effect=lambda token_id: str(token_id))
    def test_tokens_after_mismatch_are_skipped(self, _label) -> None:
        output = format_round(
            round_number=3,
            request_id="req-3",
            previous_token=None,
            draft_tokens=[21, 31, 41],
            target_tokens=[20, 30, 40],
            emitted_tokens=[20],
            accepted_count=0,
        )

        self.assertIn("D0!=T0 ✗  D1 skipped  D2 skipped", output)
        self.assertNotIn("TARGET previous", output)


class FormatStochasticRoundTest(unittest.TestCase):
    @patch("remtp.trace.token_label", side_effect=lambda token_id: str(token_id))
    def test_accept_then_recover(self, _label) -> None:
        output = format_stochastic_round(
            round_number=4,
            request_id="req-random",
            draft_tokens=[20, 30],
            target_probabilities=[0.8, 0.35],
            draft_probabilities=[1.0, 1.0],
            uniform_probabilities=[0.3, 0.6],
            emitted_tokens=[20, 31],
            accepted_count=1,
            recovery_token=31,
            bonus_token=None,
            deterministic_draft=True,
        )

        self.assertIn("[stochastic]", output)
        self.assertIn("p(D0)=0.8000  p(D1)=0.3500", output)
        self.assertIn("u=0.3000  -> ACCEPT ✓", output)
        self.assertIn("u=0.6000  -> REJECT ✗", output)
        self.assertIn("RECOVER         : 31", output)
        self.assertIn("COMMIT          : 20 31", output)

    @patch("remtp.trace.token_label", side_effect=lambda token_id: str(token_id))
    def test_all_accepted_adds_bonus(self, _label) -> None:
        output = format_stochastic_round(
            round_number=5,
            request_id="req-random",
            draft_tokens=[20, 30],
            target_probabilities=[0.8, 0.7],
            draft_probabilities=[1.0, 1.0],
            uniform_probabilities=[0.3, 0.6],
            emitted_tokens=[20, 30, 40],
            accepted_count=2,
            recovery_token=None,
            bonus_token=40,
            deterministic_draft=True,
        )

        self.assertIn("VERIFY D1", output)
        self.assertIn("ACCEPT ✓", output)
        self.assertIn("BONUS           : 40", output)
        self.assertIn("COMMIT          : 20 30 40", output)

    @patch("remtp.trace._emit_stochastic_round")
    def test_probability_inputs_match_rejection_sampler(self, emit) -> None:
        sampling_metadata = SimpleNamespace(
            output_token_ids=[[]],
            all_greedy=False,
            all_random=True,
            temperature=torch.tensor([0.8]),
        )
        _trace_legacy_rejection_result(
            draft_token_ids=torch.tensor([0, 1]),
            num_draft_tokens=[2],
            draft_probs=None,
            target_logits=torch.log(
                torch.tensor(
                    [
                        [0.8, 0.2],
                        [0.65, 0.35],
                    ]
                )
            ),
            bonus_token_ids=torch.tensor([[1]]),
            sampling_metadata=sampling_metadata,
            output_token_ids=torch.tensor([[0, 0, -1]]),
            uniform_probs=torch.tensor([0.3, 0.6]),
        )

        args = emit.call_args.args
        self.assertEqual(args[1], [0, 1])
        self.assertAlmostEqual(args[2][0], 0.8, places=5)
        self.assertAlmostEqual(args[2][1], 0.35, places=5)
        self.assertEqual(args[3], [1.0, 1.0])
        self.assertEqual(args[6], 1)
        self.assertEqual(args[7], 0)
        self.assertIsNone(args[8])
        self.assertTrue(args[9])


class CheckpointTest(unittest.TestCase):
    def test_reports_missing_weight_shard(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory)
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model_path / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"model-1.safetensors",'
                '"b":"model-2.safetensors"}}',
                encoding="utf-8",
            )
            (model_path / "model-1.safetensors").touch()

            self.assertEqual(
                missing_checkpoint_files(model_path),
                ["model-2.safetensors"],
            )


if __name__ == "__main__":
    unittest.main()
