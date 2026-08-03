import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from remtp.router_benchmark_download import (
    BenchmarkSpec,
    prepare_benchmarks,
    validate_blob,
)


def encoded_rows(rows: list[dict], *, compressed: bool = False) -> bytes:
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    return gzip.compress(content) if compressed else content


class RouterBenchmarkDownloadTest(unittest.TestCase):
    def test_validate_blob_checks_rows_prompts_and_identity(self) -> None:
        blob = encoded_rows(
            [
                {"task_id": "a", "prompt": "first prompt"},
                {"task_id": "b", "prompt": "second prompt"},
            ],
            compressed=True,
        )
        spec = BenchmarkSpec(
            name="sample",
            filename="sample.jsonl.gz",
            url="https://example.invalid/sample.jsonl.gz",
            revision="test",
            sha256=hashlib.sha256(blob).hexdigest(),
            expected_rows=2,
            prompt_fields=("prompt",),
            identity_field="task_id",
        )
        record = validate_blob(spec, blob)
        self.assertEqual(record["rows"], 2)

        duplicate = encoded_rows(
            [
                {"task_id": "a", "prompt": "first prompt"},
                {"task_id": "a", "prompt": "second prompt"},
            ],
            compressed=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_blob(spec, duplicate, verify_checksum=False)

    def test_prepare_is_idempotent_after_download(self) -> None:
        blob = encoded_rows([{"prompt": "a prompt"}])
        spec = BenchmarkSpec(
            name="sample",
            filename="sample.jsonl",
            url="https://example.invalid/sample.jsonl",
            revision="test",
            sha256=hashlib.sha256(blob).hexdigest(),
            expected_rows=1,
            prompt_fields=("prompt",),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "remtp.router_benchmark_download.BENCHMARK_SPECS", (spec,)
        ), mock.patch(
            "remtp.router_benchmark_download._download", return_value=blob
        ) as download:
            root = Path(directory)
            first = prepare_benchmarks(root)
            second = prepare_benchmarks(root)
            self.assertEqual(first["datasets"][0]["action"], "downloaded")
            self.assertEqual(second["datasets"][0]["action"], "verified")
            download.assert_called_once()
            self.assertTrue((root / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
