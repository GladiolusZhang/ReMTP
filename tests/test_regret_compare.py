import json
import tempfile
import unittest
from pathlib import Path

from remtp.regret_compare import parse_audit


class RegretCompareTest(unittest.TestCase):
    def test_audit_windows_are_aggregated_by_counts(self) -> None:
        line = (
            "(EngineCore pid=1) [ReMTP][Regret][audit] "
            "rounds=1-2 accepted=10 strict_accepted=8 causal_relaxed=2 "
            "strict_ratio=0.8 tv_per_accepted=0.05 injections=2 "
            "mean_gate=0.5 mean_regret_strength=0.1 "
            "head_reached=[2.0, 1.0] "
            "head_strict_count=[1.0, 1.0] "
            "head_causal_count=[1.0, 0.0] "
            "head_target_p_sum=[1.0, 0.25] "
            "head_hidden_cos_sum=[1.5, 0.75]\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(line + line, encoding="utf-8")
            result = parse_audit(path)
        self.assertEqual(result["audited_rounds"], 4)
        self.assertEqual(result["accepted_draft_tokens"], 20)
        self.assertAlmostEqual(result["strict_acceptable_ratio"], 0.8)
        self.assertAlmostEqual(result["tv_per_accepted_token"], 0.05)
        self.assertAlmostEqual(result["mean_compatibility_gate"], 0.5)
        self.assertEqual(result["per_head_strict_acceptance"], [0.5, 1.0])
        self.assertEqual(
            result["per_head_causal_relaxed_rate"],
            [0.5, 0.0],
        )

    def test_empty_log_returns_zero_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text("", encoding="utf-8")
            result = parse_audit(path)
        self.assertEqual(result["audited_rounds"], 0)
        self.assertEqual(result["causal_relaxed_acceptances"], 0)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
