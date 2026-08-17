import unittest

from remtp.proposal_decision_report import build_report


class ProposalDecisionReportTest(unittest.TestCase):
    def test_report_distinguishes_rejection_skip_and_recovery(self) -> None:
        trace = [
            {
                "version": 2,
                "request_index": 0,
                "round_index": 7,
                "draft_tokens": [10, 11, 12],
                "accepted_drafts": 1,
                "decisions": ["accepted", "rejected", "skipped"],
                "rejected_head": 2,
                "committed_tokens": [10, 99],
                "anchor_kind": "recovery",
                "p_y": [0.4, 0.1, 0.2],
                "q_y": [0.5, 0.5, 0.3],
                "sampled_strict_acceptance": [0.8, 0.2, 2 / 3],
                "p_top_ids": [[10, 20], [21, 11], [12, 22]],
                "p_top_probs": [[0.4, 0.3], [0.6, 0.1], [0.2, 0.2]],
                "q_top_ids": [[10, 30], [11, 31], [12, 32]],
                "q_top_probs": [[0.5, 0.2], [0.5, 0.2], [0.3, 0.2]],
            }
        ]
        requests = [
            {
                "task": "gsm8k",
                "question_id": 3,
                "output_tokens": 2,
                "finish_reason": "stop",
                "gold_answer": "9",
                "predicted_answer": "9",
                "correct": True,
                "output": "work\n#### 9",
            }
        ]
        markdown, payload = build_report(
            trace,
            requests,
            {"3": {"question": "What is 3 times 3?"}},
            lambda token_id: f"tok{token_id}",
        )
        self.assertIn("first rejection at head 2, then recovery", markdown)
        self.assertIn("| 2 | rejected |", markdown)
        self.assertIn("| 3 | skipped |", markdown)
        self.assertIn("99 `\"tok99\"`", markdown)
        self.assertIn("Final generated output", markdown)
        self.assertIn("#### 9", markdown)
        self.assertEqual(payload["requests"][0]["first_rejection_heads"], [2])

    def test_report_labels_all_accepted_bonus_round(self) -> None:
        trace = [
            {
                "version": 2,
                "request_index": 0,
                "round_index": 1,
                "draft_tokens": [1, 2],
                "decisions": ["accepted", "accepted"],
                "rejected_head": None,
                "committed_tokens": [1, 2, 3],
                "anchor_kind": "bonus",
                "p_y": [0.5, 0.5],
                "q_y": [0.5, 0.5],
                "sampled_strict_acceptance": [1.0, 1.0],
                "p_top_ids": [[1, 5], [2, 6]],
                "p_top_probs": [[0.5, 0.2], [0.5, 0.2]],
                "q_top_ids": [[1, 7], [2, 8]],
                "q_top_probs": [[0.5, 0.2], [0.5, 0.2]],
            }
        ]
        markdown, _ = build_report(
            trace,
            [{"task": "humaneval", "task_id": "HumanEval/0", "raw_output": "x"}],
            {"HumanEval/0": {"prompt": "def f(): pass"}},
            str,
        )
        self.assertIn("all 2 drafts accepted, then target bonus", markdown)
        self.assertIn("Final committed token kind: `bonus`", markdown)

    def test_report_renders_relaxed_only_acceptance(self) -> None:
        trace = [
            {
                "request_index": 0,
                "round_index": 1,
                "draft_tokens": [4],
                "accepted_drafts": 1,
                "decisions": ["accepted"],
                "acceptance_sources": ["relaxed-only"],
                "rejected_head": None,
                "committed_tokens": [4, 5],
                "anchor_kind": "bonus",
                "p_y": [0.1],
                "q_y": [0.5],
                "strict_acceptance": [0.2],
                "relaxed_acceptance": [0.7],
                "allocated_tv": [0.25],
                "p_top_ids": [[5, 4]],
                "p_top_probs": [[0.6, 0.1]],
                "q_top_ids": [[4, 5]],
                "q_top_probs": [[0.5, 0.2]],
            }
        ]
        markdown, _ = build_report(
            trace,
            [{"task": "gsm8k", "question_id": 1, "output": "answer"}],
            {"1": {"question": "question"}},
            str,
        )
        self.assertIn("accepted/relaxed-only", markdown)
        self.assertIn("| TV | relaxed alpha |", markdown)
        self.assertIn("| 0.250000 | 0.700000 |", markdown)


if __name__ == "__main__":
    unittest.main()
