import json
import tempfile
import unittest
from pathlib import Path

from remtp.router_decontaminate import decontaminate, load_rows


class RouterDecontaminationTest(unittest.TestCase):
    def test_exact_ngram_and_humaneval_artifacts_are_removed(self) -> None:
        try:
            import datasketch  # noqa: F401
        except ImportError:
            self.skipTest("datasketch is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "humaneval.jsonl"
            benchmark_rows = [
                {
                    "prompt": "def special_humaneval_function(values):\n"
                    "    \"\"\"Return every positive integer from the input "
                    "sequence while preserving its original order.\"\"\""
                },
                {
                    "prompt": "alpha beta gamma delta epsilon zeta eta theta "
                    "iota kappa lambda mu nu xi omicron"
                },
            ]
            benchmark.write_text(
                "".join(json.dumps(row) + "\n" for row in benchmark_rows),
                encoding="utf-8",
            )
            corpus = root / "corpus.jsonl"
            corpus_rows = [
                {"prompt": benchmark_rows[1]["prompt"], "source": "exact"},
                {
                    "prompt": "Please implement def special_humaneval_function(values): "
                    "with a different explanation and behavior.",
                    "source": "function",
                },
                {
                    "prompt": "alpha beta gamma delta epsilon zeta eta theta "
                    "and now this request changes direction entirely",
                    "source": "ngram",
                },
                {
                    "prompt": "Explain how ocean currents influence regional weather.",
                    "source": "clean",
                },
            ]
            corpus.write_text(
                "".join(json.dumps(row) + "\n" for row in corpus_rows),
                encoding="utf-8",
            )
            output = root / "filtered.jsonl"
            report = decontaminate(
                corpus=corpus,
                output=output,
                report_path=root / "report.json",
                benchmarks=[("humaneval", benchmark)],
                ngram_size=5,
                shingle_size=3,
                lsh_threshold=0.5,
                jaccard_threshold=0.7,
                num_perm=32,
            )
            kept = load_rows(output)
            self.assertEqual([row["source"] for row in kept], ["clean"])
            self.assertEqual(report["removed_records"], 3)
            self.assertIn("normalized_exact", report["reason_counts"])
            self.assertIn("humaneval_function_name", report["reason_counts"])
            self.assertIn("5_gram", report["reason_counts"])


if __name__ == "__main__":
    unittest.main()
