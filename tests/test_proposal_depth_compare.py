import json
import tempfile
import unittest
from pathlib import Path

from remtp.proposal_depth_compare import collect, write_outputs


class ProposalDepthCompareTest(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        depth: int,
        method: str,
        dataset: str,
    ) -> None:
        directory = root / f"mtp_{depth}" / method / dataset
        directory.mkdir(parents=True)
        config = {
            "data_sha256": f"{dataset}-data",
            "samples": 2,
            "sample_seed": 7,
            "temperature": 0.7,
            "generation_seed": 42,
            "max_tokens": 64,
            "mtp_tokens": depth,
        }
        result = {
            "samples": 2,
            "accuracy": 0.5,
            "pass_at_1": 0.5,
            "decode_tok_s": 100.0 + depth,
            "e2e_output_tok_s": 90.0 + depth,
            "mean_acceptance_length": 2.0 + depth / 10,
            "draft_token_acceptance_rate": 0.5,
            "truncation_rate": 0.0,
        }
        if dataset == "humaneval":
            config["answer_metric"] = "official_tests_pass_at_1"
            result["evaluation_status"] = "complete"
            result["evaluation_counts"] = {"timeout": 0}
        (directory / "summary.json").write_text(
            json.dumps({"config": config, "results": [result]}), encoding="utf-8"
        )
        (directory / "sample_manifest.json").write_text(
            f"same-{dataset}", encoding="utf-8"
        )

    def test_partial_and_complete_matrix_share_one_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for depth in (2, 4):
                for method in ("native_mtp", "cactus"):
                    for dataset in ("gsm8k", "humaneval"):
                        self._write_run(root, depth, method, dataset)
            rows = collect(root, [2, 4], ["native_mtp", "cactus", "spec_cascade"])
            self.assertEqual(len(rows), 8)
            write_outputs(
                root,
                rows,
                [2, 4],
                ["native_mtp", "cactus", "spec_cascade"],
            )
            report = (root / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("Completed rows: **8/12**", report)
            self.assertIn("PENDING", report)
            self.assertTrue((root / "comparison.json").is_file())
            self.assertTrue((root / "comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
