from __future__ import annotations

import unittest

from remtp.fixed6_validation import validate_audit


class Fixed6ValidationTests(unittest.TestCase):
    def test_valid_branch_prefix_and_fixed_budget(self) -> None:
        result = validate_audit(
            [
                {
                    "topology": "4+2",
                    "target_validation_nodes": 6,
                    "target_forward_calls": 1,
                    "selected_nodes": [0, 2, 4],
                    "selected_branch": 0,
                },
                {
                    "topology": "4+2",
                    "target_validation_nodes": 6,
                    "target_forward_calls": 1,
                    "selected_nodes": [1, 3],
                    "selected_branch": 1,
                },
            ],
            "4+2",
        )
        self.assertTrue(result["audit_invariants_passed"])
        self.assertEqual(result["observed_branches"], [0, 1])

    def test_flattened_cross_branch_path_fails(self) -> None:
        result = validate_audit(
            [
                {
                    "topology": "3+2+1",
                    "target_validation_nodes": 6,
                    "target_forward_calls": 1,
                    "selected_nodes": [0, 4],
                    "selected_branch": 0,
                }
            ],
            "3+2+1",
        )
        self.assertFalse(result["audit_invariants_passed"])


if __name__ == "__main__":
    unittest.main()
