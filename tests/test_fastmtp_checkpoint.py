from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from remtp.fastmtp_checkpoint import (
    FastMTPCheckpointError,
    build_logical_view,
    check_fastmtp,
)


def _checkpoint(path: Path, *, layers: int = 1) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiMoForCausalLM"],
                "num_nextn_predict_layers": layers,
                "hidden_size": 8,
                "vocab_size": 16,
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "model.layers.0.weight": torch.zeros(1),
            "model.mtp_layers.0.weight": torch.ones(1),
        },
        path / "model.safetensors",
    )


def test_logical_view_keeps_one_physical_trained_head(tmp_path: Path) -> None:
    source = tmp_path / "FastMTP"
    output = tmp_path / "FastMTP-Tree32"
    _checkpoint(source)
    checked = build_logical_view(source, output, 32)
    assert checked["logical_mtp_width"] == 32
    assert checked["physical_mtp_layers"] == 1
    assert checked["mtp_layers"] == [0]
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["num_nextn_predict_layers"] == 32
    assert config["remtp_physical_mtp_layers"] == 1
    assert (output / "model.safetensors").is_symlink()


def test_fastmtp_rejects_more_than_one_physical_head(tmp_path: Path) -> None:
    source = tmp_path / "bad"
    _checkpoint(source, layers=2)
    with pytest.raises(FastMTPCheckpointError, match="exactly one"):
        check_fastmtp(source)
