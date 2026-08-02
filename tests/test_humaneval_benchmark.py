import gzip
import json
import tempfile
import unittest
from pathlib import Path

from remtp.humaneval_benchmark import (
    add_pending_quality_fields,
    load_humaneval,
    sample_humaneval,
    sanitize_generation,
)


def _task(index: int) -> dict[str, str]:
    return {
        "task_id": f"HumanEval/{index}",
        "prompt": f"def f{index}(x):\n    \"\"\"Return x.\"\"\"\n",
        "canonical_solution": "    return x\n",
        "test": f"def check(candidate):\n    assert candidate(1) == 1\n",
        "entry_point": f"f{index}",
    }


class HumanEvalBenchmarkTest(unittest.TestCase):
    def test_loads_plain_and_gzip_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = "".join(json.dumps(_task(i)) + "\n" for i in range(3))
            plain = root / "HumanEval.jsonl"
            plain.write_text(text, encoding="utf-8")
            compressed = root / "download.tmp"
            with gzip.open(compressed, "wt", encoding="utf-8") as handle:
                handle.write(text)
            self.assertEqual(len(load_humaneval(plain)), 3)
            self.assertEqual(len(load_humaneval(compressed)), 3)

    def test_sampling_is_reproducible_and_sorted(self) -> None:
        rows = [_task(i) for i in range(20)]
        first = sample_humaneval(rows, 5, 42)
        second = sample_humaneval(rows, 5, 42)
        self.assertEqual(first, second)
        ids = [int(row["task_id"].split("/")[-1]) for row in first]
        self.assertEqual(ids, sorted(ids))

    def test_sanitizes_fenced_complete_function(self) -> None:
        output = "Here is the code:\n```python\ndef add(a, b):\n    return a+b\n```"
        candidate, mode = sanitize_generation(output, "add")
        self.assertEqual(mode, "full_function")
        self.assertTrue(candidate.startswith("def add"))
        self.assertNotIn("```", candidate)

    def test_preserves_body_completion(self) -> None:
        candidate, mode = sanitize_generation("    return x + 1", "inc")
        self.assertEqual(mode, "completion")
        self.assertEqual(candidate, "    return x + 1")

    def test_preserves_imports_for_valid_full_module(self) -> None:
        output = "import re\n\ndef clean(x):\n    return re.sub('x', '', x)"
        candidate, mode = sanitize_generation(output, "clean")
        self.assertEqual(mode, "full_function")
        self.assertTrue(candidate.startswith("import re"))

    def test_pending_summary_has_no_fake_accuracy(self) -> None:
        summary = {}
        records = [
            {"finish_reason": "stop"},
            {"finish_reason": "length"},
        ]
        add_pending_quality_fields(summary, records)
        self.assertIsNone(summary["pass_at_1"])
        self.assertEqual(summary["evaluation_status"], "pending_docker_evaluation")
        self.assertEqual(summary["truncation_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
