import json
import tempfile
import unittest
from pathlib import Path

from remtp.current_block_debt_compare import compare


class CurrentBlockDebtCompareTest(unittest.TestCase):
    @staticmethod
    def _write(root: Path, name: str, *, acc: float, e2e: float, mal: float) -> None:
        directory = root / name
        directory.mkdir()
        payload = {
            "config": {
                "samples": 40,
                "sample_seed": 7,
                "temperature": 0.7,
                "generation_seed": 42,
                "max_tokens": 384,
                "mtp_tokens": 6,
                "data_sha256": "data",
                "answer_metric": "exact",
            },
            "results": [
                {
                    "samples": 40,
                    "accuracy": acc,
                    "decode_tok_s": e2e + 10.0,
                    "e2e_output_tok_s": e2e,
                    "mean_acceptance_length": mal,
                    "draft_token_acceptance_rate": (mal - 1.0) / 6.0,
                    "truncation_rate": 0.0,
                }
            ]
        }
        (directory / "summary.json").write_text(json.dumps(payload))
        (directory / "sample_manifest.json").write_text("shared\n")

    def test_gate_requires_all_three_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "cactus", acc=0.80, e2e=180.0, mal=4.9)
            self._write(root, "pass", acc=0.825, e2e=181.0, mal=5.0)
            self._write(root, "quality_fail", acc=0.80, e2e=190.0, mal=5.2)
            self._write(root, "speed_fail", acc=0.825, e2e=179.0, mal=5.2)
            rows = compare(root, ["pass", "quality_fail", "speed_fail"])
            status = {row["directory"]: row["pareto_pass"] for row in rows}
            self.assertTrue(status["pass"])
            self.assertFalse(status["quality_fail"])
            self.assertFalse(status["speed_fail"])

    def test_rejects_different_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "cactus", acc=0.80, e2e=180.0, mal=4.9)
            self._write(root, "candidate", acc=0.81, e2e=181.0, mal=5.0)
            (root / "candidate" / "sample_manifest.json").write_text(
                "different\n"
            )
            with self.assertRaises(ValueError):
                compare(root, ["candidate"])


if __name__ == "__main__":
    unittest.main()
