"""Download and validate the benchmark prompts used for router decontamination."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    filename: str
    url: str
    revision: str
    sha256: str
    expected_rows: int
    prompt_fields: tuple[str, ...]
    identity_field: str | None = None


GOOGLE_RESEARCH_REVISION = "8dadc6c56e2c2e51a9dd7e0d4bf2840922b4b6c0"

BENCHMARK_SPECS = (
    BenchmarkSpec(
        name="humaneval",
        filename="HumanEval.jsonl.gz",
        url=(
            "https://raw.githubusercontent.com/openai/human-eval/"
            "6d43fb980f9fee3c892a914eda09951f772ad10d/data/"
            "HumanEval.jsonl.gz"
        ),
        revision="6d43fb980f9fee3c892a914eda09951f772ad10d",
        sha256="b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
        expected_rows=164,
        prompt_fields=("prompt",),
        identity_field="task_id",
    ),
    BenchmarkSpec(
        name="humaneval_plus",
        filename="HumanEvalPlus.jsonl.gz",
        url=(
            "https://raw.githubusercontent.com/evalplus/"
            "humanevalplus_release/"
            "200defce9e3429d28ca215b6dd061c0f7f31c18b/"
            "HumanEvalPlus.jsonl.gz"
        ),
        revision="200defce9e3429d28ca215b6dd061c0f7f31c18b",
        sha256="272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
        expected_rows=164,
        prompt_fields=("prompt",),
        identity_field="task_id",
    ),
    BenchmarkSpec(
        name="mbpp",
        filename="mbpp.jsonl",
        url=(
            "https://raw.githubusercontent.com/google-research/"
            f"google-research/{GOOGLE_RESEARCH_REVISION}/mbpp/mbpp.jsonl"
        ),
        revision=GOOGLE_RESEARCH_REVISION,
        sha256="ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f",
        expected_rows=974,
        prompt_fields=("text",),
        identity_field="task_id",
    ),
    BenchmarkSpec(
        name="mbpp_plus",
        filename="MbppPlus.jsonl.gz",
        url=(
            "https://raw.githubusercontent.com/evalplus/mbppplus_release/"
            "dadf43da556a00f7bacd71cb154f2932757d9144/"
            "MbppPlus.jsonl.gz"
        ),
        revision="dadf43da556a00f7bacd71cb154f2932757d9144",
        sha256="af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
        expected_rows=378,
        prompt_fields=("prompt",),
        identity_field="task_id",
    ),
    BenchmarkSpec(
        name="gsm8k",
        filename="gsm8k_test.jsonl",
        url=(
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            "3101c7d5072418e28b9008a6636bde82a006892c/"
            "grade_school_math/data/test.jsonl"
        ),
        revision="3101c7d5072418e28b9008a6636bde82a006892c",
        sha256="3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        expected_rows=1319,
        prompt_fields=("question",),
    ),
    BenchmarkSpec(
        name="ifeval",
        filename="ifeval_input_data.jsonl",
        url=(
            "https://raw.githubusercontent.com/google-research/"
            f"google-research/{GOOGLE_RESEARCH_REVISION}/"
            "instruction_following_eval/data/input_data.jsonl"
        ),
        revision=GOOGLE_RESEARCH_REVISION,
        sha256="67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
        expected_rows=541,
        prompt_fields=("prompt",),
        identity_field="key",
    ),
)


def _decode_rows(filename: str, blob: bytes) -> list[dict[str, Any]]:
    if filename.endswith(".gz") or blob.startswith(b"\x1f\x8b"):
        blob = gzip.decompress(blob)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(blob.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{filename}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def validate_blob(
    spec: BenchmarkSpec,
    blob: bytes,
    *,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    digest = hashlib.sha256(blob).hexdigest()
    if verify_checksum and digest != spec.sha256:
        raise ValueError(
            f"checksum mismatch for {spec.name}: expected {spec.sha256}, got {digest}"
        )
    rows = _decode_rows(spec.filename, blob)
    if len(rows) != spec.expected_rows:
        raise ValueError(
            f"row-count mismatch for {spec.name}: expected "
            f"{spec.expected_rows}, got {len(rows)}"
        )
    missing_prompt = [
        index
        for index, row in enumerate(rows)
        if not any(
            isinstance(row.get(field), str) and row[field].strip()
            for field in spec.prompt_fields
        )
    ]
    if missing_prompt:
        raise ValueError(
            f"{spec.name} has {len(missing_prompt)} rows without a prompt; "
            f"first index={missing_prompt[0]}"
        )
    if spec.identity_field is not None:
        identities = [row.get(spec.identity_field) for row in rows]
        if any(value is None for value in identities):
            raise ValueError(
                f"{spec.name} is missing identity field {spec.identity_field!r}"
            )
        if len(set(identities)) != len(identities):
            raise ValueError(
                f"{spec.name} contains duplicate {spec.identity_field!r} values"
            )
    return {
        "name": spec.name,
        "filename": spec.filename,
        "rows": len(rows),
        "sha256": digest,
        "prompt_fields": list(spec.prompt_fields),
        "revision": spec.revision,
        "source_url": spec.url,
    }


def _download(url: str, *, timeout: float, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ReMTP-router-benchmark-downloader/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"Download failed (attempt {attempt}/{retries}); "
                f"retrying in {delay}s: {exc}"
            )
            time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from last_error


def _atomic_write(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def prepare_benchmarks(
    output_dir: Path,
    *,
    force: bool = False,
    check_only: bool = False,
    timeout: float = 120.0,
    retries: int = 3,
) -> dict[str, Any]:
    if force and check_only:
        raise ValueError("force and check_only cannot be enabled together")
    if retries < 1:
        raise ValueError("retries must be at least 1")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[dict[str, Any]] = []
    for spec in BENCHMARK_SPECS:
        destination = output_dir / spec.filename
        action = "verified"
        if destination.is_file() and not force:
            blob = destination.read_bytes()
            try:
                record = validate_blob(spec, blob)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Existing file is invalid: {destination}\n{exc}\n"
                    "Run again with FORCE=1 to replace it."
                ) from exc
        elif check_only:
            raise FileNotFoundError(
                f"required benchmark is missing in check-only mode: {destination}"
            )
        else:
            action = "downloaded"
            blob = _download(spec.url, timeout=timeout, retries=retries)
            record = validate_blob(spec, blob)
            _atomic_write(destination, blob)
        record["path"] = str(destination)
        record["action"] = action
        datasets.append(record)
        print(
            f"[{action:10}] {spec.name:14} rows={record['rows']:4d} "
            f"sha256={record['sha256'][:12]}…"
        )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "datasets": datasets,
        "specifications": [asdict(spec) for spec in BENCHMARK_SPECS],
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    print(f"Manifest: {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/regret_router/benchmarks"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_benchmarks(
        args.output_dir,
        force=args.force,
        check_only=args.check_only,
        timeout=args.timeout,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
