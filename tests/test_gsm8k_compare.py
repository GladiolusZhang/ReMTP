import json
import tempfile
import unittest
from pathlib import Path

from remtp.gsm8k_compare import load_comparison, write_comparison


class Gsm8kCompareTest(unittest.TestCase):
    def test_three_way_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            for index, run_name in enumerate(
                ("standard_mtp", "tokenv3", "cactus")
            ):
                run_dir = run_root / run_name
                run_dir.mkdir()
                config = {
                    "samples": 10,
                    "sample_seed": 1,
                    "temperature": 0.7,
                    "generation_seed": 42,
                    "max_tokens": 384,
                    "mtp_tokens": 2,
                    "data_sha256": "data",
                    "answer_metric": "exact",
                }
                result = {
                    "samples": 10,
                    "correct": 8 + index,
                    "accuracy": 0.8 + index * 0.1,
                    "decode_tok_s": 100.0 + index * 10.0,
                    "e2e_output_tok_s": 90.0 + index * 10.0,
                    "mean_acceptance_length": 2.0 + index * 0.1,
                    "draft_token_acceptance_rate": 0.5 + index * 0.1,
                    "truncation_rate": 0.0,
                    "output_tokens": 1000,
                }
                (run_dir / "summary.json").write_text(
                    json.dumps({"config": config, "results": [result]}),
                    encoding="utf-8",
                )
                (run_dir / "sample_manifest.json").write_text(
                    "same manifest\n",
                    encoding="utf-8",
                )

            rows, manifest_hash = load_comparison(run_root)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["accuracy_delta_pp"], 0.0)
            self.assertAlmostEqual(rows[1]["decode_delta_pct"], 10.0)
            write_comparison(run_root, rows, manifest_hash)
            self.assertTrue((run_root / "comparison.json").is_file())
            self.assertTrue((run_root / "comparison.csv").is_file())
            self.assertTrue((run_root / "comparison.md").is_file())


if __name__ == "__main__":
    unittest.main()
