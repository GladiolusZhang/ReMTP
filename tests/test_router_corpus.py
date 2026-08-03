import random
import unittest

from remtp.router_corpus import normalize_prompt, select_source


class RouterCorpusTest(unittest.TestCase):
    def test_normalize_prompt_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_prompt("  hello\n  world  "), "hello world")

    def test_selection_filters_and_deduplicates(self) -> None:
        dataset = [
            {"prompt": "short"},
            {"prompt": "A sufficiently long and useful first router prompt."},
            {"prompt": "A sufficiently long and useful first router prompt."},
            {"prompt": "A separate sufficiently long second router prompt."},
        ]
        selected = select_source(
            dataset,
            source="test",
            field="prompt",
            count=2,
            rng=random.Random(42),
            seen=set(),
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["source"] for row in selected}, {"test"})
        self.assertEqual(len({row["prompt"] for row in selected}), 2)


if __name__ == "__main__":
    unittest.main()
