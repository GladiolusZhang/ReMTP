import unittest

from remtp.benchmark import (
    aggregate_records,
    official_task,
    parse_metrics,
    sample_questions,
)


class BenchmarkTest(unittest.TestCase):
    def test_official_task_boundaries(self) -> None:
        self.assertEqual(official_task(81), "multi_turn")
        self.assertEqual(official_task(160), "multi_turn")
        self.assertEqual(official_task(161), "translation")
        self.assertEqual(official_task(560), "rag")
        with self.assertRaises(ValueError):
            official_task(80)

    def test_sampling_is_balanced_and_reproducible(self) -> None:
        questions = []
        for question_id in range(161, 241):
            questions.append(
                {
                    "question_id": question_id,
                    "turns": ["translate"],
                    "_spec_bench_task": "translation",
                }
            )
        for question_id in range(241, 321):
            questions.append(
                {
                    "question_id": question_id,
                    "turns": ["summarize"],
                    "_spec_bench_task": "summarization",
                }
            )

        first = sample_questions(
            questions,
            ["translation", "summarization"],
            samples_per_task=20,
            seed=42,
        )
        second = sample_questions(
            questions,
            ["translation", "summarization"],
            samples_per_task=20,
            seed=42,
        )
        self.assertEqual(
            [row["question_id"] for row in first],
            [row["question_id"] for row in second],
        )
        self.assertEqual(len(first), 40)
        self.assertEqual(
            sum(row["_spec_bench_task"] == "translation" for row in first),
            20,
        )

    def test_parse_metrics_sums_labelled_series(self) -> None:
        text = """
# HELP vllm:generation_tokens_total Number of generation tokens.
vllm:generation_tokens_total{model_name="a"} 12
vllm:generation_tokens_total{model_name="b"} 3
vllm:spec_decode_num_drafts_total{model_name="a"} 5
unrelated_metric 999
"""
        metrics = parse_metrics(text)
        self.assertEqual(metrics["vllm:generation_tokens_total"], 15.0)
        self.assertEqual(metrics["vllm:spec_decode_num_drafts_total"], 5.0)
        self.assertEqual(
            metrics["vllm:spec_decode_num_accepted_tokens_total"],
            0.0,
        )

    def test_aggregate_acceptance_and_throughput(self) -> None:
        record = {
            "output_tokens": 30,
            "client_seconds": 1.5,
            "metrics": {
                "vllm:generation_tokens_total": 30.0,
                "vllm:request_decode_time_seconds_sum": 0.5,
                "vllm:request_prefill_time_seconds_sum": 0.1,
                "vllm:spec_decode_num_drafts_total": 10.0,
                "vllm:spec_decode_num_draft_tokens_total": 20.0,
                "vllm:spec_decode_num_accepted_tokens_total": 15.0,
            },
        }
        summary = aggregate_records([record], "translation", 1)
        self.assertEqual(summary["decode_tok_s"], 60.0)
        self.assertEqual(summary["e2e_output_tok_s"], 20.0)
        self.assertEqual(summary["mean_acceptance_length"], 2.5)
        self.assertEqual(summary["draft_token_acceptance_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
