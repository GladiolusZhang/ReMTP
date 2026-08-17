import json
import tempfile
import unittest
from pathlib import Path

from remtp.humaneval_reuse import protocol_mismatches, reuse_profiles


class HumanEvalReuseTest(unittest.TestCase):
    @staticmethod
    def _config(samples: int = 164) -> dict:
        return {
            "samples": samples,
            "sample_seed": 20260802,
            "temperature": 0.7,
            "generation_seed": 42,
            "max_tokens": 512,
            "mtp_tokens": 6,
            "data_sha256": "data",
            "answer_metric": "pass@1",
            "evaluation_image": "python:3-slim",
            "evaluation_image_id": "sha256:test",
            "evaluation_timeout_seconds": 8.0,
        }

    @staticmethod
    def _write_run(root: Path, profile: str, config: dict) -> Path:
        directory = root / profile
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "results": [{"evaluation_status": "complete"}],
                }
            ),
            encoding="utf-8",
        )
        (directory / "sample_manifest.json").write_text(
            "manifest\n", encoding="utf-8"
        )
        (directory / "requests.jsonl").write_text("{}\n", encoding="utf-8")
        return directory

    def test_reuses_compatible_result_as_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "old"
            destination_root = root / "new"
            source = self._write_run(source_root, "cactus", self._config())
            reused = reuse_profiles(
                source_roots=[source_root],
                destination_root=destination_root,
                profiles=["cactus"],
                expected=self._config(),
                required=True,
            )
            destination = destination_root / "cactus"
            self.assertTrue(destination.is_symlink())
            self.assertEqual(reused["cactus"], source.resolve())

    def test_rejects_missing_compatible_result_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "old"
            self._write_run(source_root, "cactus", self._config(samples=40))
            with self.assertRaisesRegex(ValueError, "cactus"):
                reuse_profiles(
                    source_roots=[source_root],
                    destination_root=root / "new",
                    profiles=["cactus"],
                    expected=self._config(samples=164),
                    required=True,
                )

    def test_reports_protocol_difference(self) -> None:
        actual = self._config()
        expected = self._config()
        actual["temperature"] = 0.8
        self.assertEqual(protocol_mismatches(actual, expected), ["temperature"])


if __name__ == "__main__":
    unittest.main()
