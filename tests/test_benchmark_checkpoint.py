from __future__ import annotations

import json
from pathlib import Path

import pytest

from remtp.benchmark_checkpoint import (
    append_benchmark_checkpoint,
    initialize_benchmark_output,
)


def test_checkpoint_resumes_exact_manifest_prefix(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest = [{"task_id": "a"}, {"task_id": "b"}]
    records, checkpoint = initialize_benchmark_output(
        output,
        manifest,
        resume=False,
        identity_key="task_id",
    )
    assert records == []
    append_benchmark_checkpoint(checkpoint, {"task_id": "a", "value": 1})
    records, _ = initialize_benchmark_output(
        output,
        manifest,
        resume=True,
        identity_key="task_id",
    )
    assert records == [{"task_id": "a", "value": 1}]


def test_checkpoint_refuses_changed_manifest_or_nonprefix(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest = [{"question_id": 1}, {"question_id": 2}]
    _, checkpoint = initialize_benchmark_output(
        output,
        manifest,
        resume=False,
        identity_key="question_id",
    )
    checkpoint.write_text(
        json.dumps({"question_id": 2}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a prefix"):
        initialize_benchmark_output(
            output,
            manifest,
            resume=True,
            identity_key="question_id",
        )
    with pytest.raises(ValueError, match="manifest differs"):
        initialize_benchmark_output(
            output,
            [{"question_id": 9}],
            resume=True,
            identity_key="question_id",
        )
