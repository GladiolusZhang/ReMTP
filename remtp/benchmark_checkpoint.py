"""Durable per-request checkpoints for long benchmark generation runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CHECKPOINT_NAME = "requests.checkpoint.jsonl"


def initialize_benchmark_output(
    output_dir: Path,
    manifest: list[dict[str, Any]],
    *,
    resume: bool,
    identity_key: str,
) -> tuple[list[dict[str, Any]], Path]:
    """Create an output directory or validate and load its checkpoint.

    Checkpoint records must be an exact prefix of the deterministic sample
    manifest. This prevents a changed seed/dataset/protocol from being mixed
    into an interrupted run.
    """
    manifest_path = output_dir / "sample_manifest.json"
    checkpoint_path = output_dir / CHECKPOINT_NAME
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        if (output_dir / "summary.json").exists():
            raise ValueError(f"benchmark is already complete: {output_dir}")
        if not manifest_path.is_file():
            raise ValueError(f"partial run has no sample manifest: {output_dir}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(
                f"partial run manifest differs from requested samples: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    records: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        for line_number, raw in enumerate(
            checkpoint_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid checkpoint JSON at {checkpoint_path}:{line_number}"
                ) from exc
    if len(records) > len(manifest):
        raise ValueError(f"checkpoint is longer than manifest: {checkpoint_path}")
    expected = [row[identity_key] for row in manifest[: len(records)]]
    actual = [row.get(identity_key) for row in records]
    if actual != expected:
        raise ValueError(
            f"checkpoint is not a prefix of the sample manifest: {checkpoint_path}"
        )
    return records, checkpoint_path


def append_benchmark_checkpoint(path: Path, record: dict[str, Any]) -> None:
    """Durably append one completed request before starting the next one."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
