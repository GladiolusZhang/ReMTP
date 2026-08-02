import unittest

from remtp.gsm8k_benchmark import (
    add_accuracy,
    extract_gold_answer,
    extract_model_answer,
    normalize_number,
    sample_gsm8k,
)


class Gsm8kBenchmarkTest(unittest.TestCase):
    def test_number_normalization(self) -> None:
        self.assertEqual(normalize_number("1,234"), "1234")
        self.assertEqual(normalize_number("18.00"), "18")
        self.assertEqual(normalize_number("-0.0"), "0")
        self.assertIsNone(normalize_number("not-a-number"))

    def test_gold_answer_requires_hash_suffix(self) -> None:
        self.assertEqual(
            extract_gold_answer("work\n#### 1,234"),
            "1234",
        )
        with self.assertRaises(ValueError):
            extract_gold_answer("work only")

    def test_model_answer_extraction_precedence(self) -> None:
        self.assertEqual(
            extract_model_answer("2 + 2 = 4\n#### 4"),
            ("4", "hash"),
        )
        self.assertEqual(
            extract_model_answer(r"Therefore \\boxed{3.0}"),
            ("3", "boxed"),
        )
        self.assertEqual(
            extract_model_answer("The final value is 1,024."),
            ("1024", "last_number"),
        )
        self.assertEqual(extract_model_answer("No number"), (None, None))

    def test_sampling_is_reproducible(self) -> None:
        rows = [{"question_id": i} for i in range(1, 21)]
        first = sample_gsm8k(rows, samples=5, seed=42)
        second = sample_gsm8k(rows, samples=5, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(
            [row["question_id"] for row in first],
            sorted(row["question_id"] for row in first),
        )

    def test_sampling_honors_exclusions(self) -> None:
        rows = [
            {"question_id": index, "question": str(index)}
            for index in range(10)
        ]
        selected = sample_gsm8k(
            rows,
            samples=5,
            seed=42,
            excluded_question_ids={0, 1, 2, 3},
        )
        self.assertTrue(
            all(row["question_id"] not in {0, 1, 2, 3} for row in selected)
        )

    def test_accuracy_fields(self) -> None:
        summary = {}
        records = [
            {
                "correct": True,
                "predicted_answer": "1",
                "answer_source": "hash",
                "finish_reason": "stop",
            },
            {
                "correct": False,
                "predicted_answer": None,
                "answer_source": None,
                "finish_reason": "length",
            },
        ]
        add_accuracy(summary, records)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["parse_rate"], 0.5)
        self.assertEqual(summary["format_compliance_rate"], 0.5)
        self.assertEqual(summary["truncation_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
