from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from remtp.checkpoint import missing_checkpoint_files
from remtp.trace import format_round


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
