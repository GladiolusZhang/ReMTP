import json
import tempfile
import unittest
from pathlib import Path

from remtp.regret_router_collect import load_corpus


class RegretRouterCollectTest(unittest.TestCase):
    def test_loads_supported_jsonl_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            rows = [
                {"prompt": "first"},
                {"instruction": "second"},
                {"messages": [{"role": "user", "content": "third"}]},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            loaded = load_corpus(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["messages"][0]["content"], "first")


if __name__ == "__main__":
    unittest.main()
