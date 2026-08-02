import json
import tempfile
import unittest
from pathlib import Path

from remtp.current_block_debt_compare import (
    compare,
    paired_bootstrap_intervals,
)


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

    @staticmethod
    def _write_requests(root: Path, name: str) -> None:
        path = root / name / "requests.jsonl"
        records = []
        for index in range(4):
            records.append(
                {
                    "task": "gsm8k",
                    "question_id": index,
                    "sample_index": index,
                    "seed": 42 + index,
                    "correct": index % 2 == 0,
                    "output_tokens": 10 + index,
                    "client_seconds": 0.1 + index / 100.0,
                    "metrics": {
                        "vllm:spec_decode_num_accepted_tokens_total": 8 + index,
                        "vllm:spec_decode_num_drafts_total": 2 + index,
                    },
                }
            )
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )

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

    def test_native_mtp_is_reported_but_never_selected_as_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "cactus", acc=0.80, e2e=180.0, mal=4.9)
            self._write(root, "native_mtp", acc=0.90, e2e=190.0, mal=5.2)
            self._write(root, "candidate", acc=0.81, e2e=181.0, mal=5.0)
            rows = compare(root, ["native_mtp", "candidate"])
            status = {row["directory"]: row["pareto_pass"] for row in rows}
            self.assertFalse(status["native_mtp"])
            self.assertTrue(status["candidate"])

    def test_explicit_eligibility_limits_selection_to_our_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "cactus", acc=0.80, e2e=180.0, mal=4.9)
            self._write(root, "paper", acc=0.90, e2e=190.0, mal=5.2)
            self._write(root, "ours", acc=0.81, e2e=181.0, mal=5.0)
            rows = compare(
                root,
                ["paper", "ours"],
                eligible_profiles={"ours"},
            )
            status = {row["directory"]: row["pareto_pass"] for row in rows}
            self.assertFalse(status["paper"])
            self.assertTrue(status["ours"])

    def test_paired_bootstrap_is_zero_for_identical_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "cactus", acc=0.5, e2e=100.0, mal=4.0)
            self._write(root, "candidate", acc=0.5, e2e=100.0, mal=4.0)
            self._write_requests(root, "cactus")
            self._write_requests(root, "candidate")
            intervals = paired_bootstrap_intervals(
                root,
                ["candidate"],
                bootstrap_samples=100,
                bootstrap_seed=7,
            )
            for interval in intervals["candidate"].values():
                self.assertEqual(interval, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
