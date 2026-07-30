import unittest

from remtp.head_reliability import estimate_head_reliability


class HeadReliabilityTest(unittest.TestCase):
    def test_estimates_conditional_head_rates(self) -> None:
        text = """
Drafted: 400 tokens, Per-position acceptance rate: 0.900, 0.720, 0.540, 0.360, Avg Draft acceptance rate: 63.0%
Drafted: 800 tokens, Per-position acceptance rate: 0.600, 0.420, 0.252, 0.126, Avg Draft acceptance rate: 35.0%
"""
        reliability, prefix, rounds = estimate_head_reliability(text, 4)
        self.assertEqual(rounds, 300)
        self.assertAlmostEqual(prefix[0], 0.7)
        self.assertAlmostEqual(prefix[1], 0.52)
        self.assertAlmostEqual(reliability[0], 0.7)
        self.assertAlmostEqual(reliability[1], 0.52 / 0.7)
        self.assertAlmostEqual(reliability[2], 0.348 / 0.52)
        self.assertAlmostEqual(reliability[3], 0.204 / 0.348)

    def test_rejects_missing_metrics(self) -> None:
        with self.assertRaises(ValueError):
            estimate_head_reliability("no metrics", 4)


if __name__ == "__main__":
    unittest.main()
