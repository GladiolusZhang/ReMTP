import json
import tempfile
import unittest
from pathlib import Path

from remtp.humaneval_compare import compare


class HumanEvalCompareTest(unittest.TestCase):
    @staticmethod
    def _write(
        root: Path,
        profile: str,
        *,
        pass_at_1: float,
        e2e: float,
        mal: float,
    ) -> None:
        directory = root / profile
        directory.mkdir()
        config = {
            "samples": 10,
            "sample_seed": 7,
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
        summary = {
            "samples": 10,
            "pass_at_1": pass_at_1,
            "decode_tok_s": e2e + 10,
            "e2e_output_tok_s": e2e,
            "mean_acceptance_length": mal,
            "draft_token_acceptance_rate": (mal - 1) / 6,
            "truncation_rate": 0.0,
            "evaluation_status": "complete",
            "evaluation_counts": {"passed": int(10 * pass_at_1)},
        }
        (directory / "summary.json").write_text(
            json.dumps({"config": config, "results": [summary]}),
            encoding="utf-8",
        )
        (directory / "sample_manifest.json").write_text(
            "shared\n", encoding="utf-8"
        )

    @staticmethod
    def _write_requests(root: Path, profile: str, raw_output: str) -> None:
        row = {
            "task_id": "HumanEval/0",
            "seed": 42,
            "output_tokens": 2,
            "finish_reason": "stop",
            "candidate": raw_output,
            "raw_output": raw_output,
        }
        (root / profile / "requests.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

    def test_reports_quality_and_speed_deltas_without_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "cactus", pass_at_1=0.7, e2e=180, mal=4.9)
            self._write(root, "native_mtp", pass_at_1=0.8, e2e=140, mal=3.8)
            rows = compare(root, ["cactus", "native_mtp"])
            native = rows[1]
            self.assertEqual(native["result_source"], "new")
            self.assertAlmostEqual(native["pass_at_1_delta_pp"], 10.0)
            self.assertLess(native["e2e_delta_pct"], 0)
            self.assertNotIn("pareto_pass", native)

    def test_rejects_different_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "cactus", pass_at_1=0.7, e2e=180, mal=4.9)
            self._write(root, "native_mtp", pass_at_1=0.8, e2e=140, mal=3.8)
            (root / "native_mtp" / "sample_manifest.json").write_text(
                "different\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                compare(root, ["cactus", "native_mtp"])

    def test_identity_control_must_match_native_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "cactus", pass_at_1=0.7, e2e=180, mal=4.9)
            self._write(root, "native_mtp", pass_at_1=0.8, e2e=140, mal=3.8)
            self._write(
                root,
                "target_mode_identity",
                pass_at_1=0.8,
                e2e=140,
                mal=3.8,
            )
            self._write_requests(root, "native_mtp", "same")
            self._write_requests(root, "target_mode_identity", "different")
            with self.assertRaisesRegex(ValueError, "identity control"):
                compare(
                    root,
                    ["native_mtp", "target_mode_identity", "cactus"],
                )

            self._write_requests(root, "target_mode_identity", "same")
            rows = compare(
                root,
                ["native_mtp", "target_mode_identity", "cactus"],
            )
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
