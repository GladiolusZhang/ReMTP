from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from remtp.mimo_checkpoint import (
    MiMoCheckpointError,
    build_logical_view,
    build_overlay,
    check_overlay,
)


def _config(architecture: str, layers: int) -> dict[str, object]:
    return {
        "architectures": [architecture],
        "hidden_size": 8,
        "vocab_size": 16,
        "num_attention_heads": 2,
        "num_nextn_predict_layers": layers,
    }


def _write_checkpoint(
    path: Path,
    config: dict[str, object],
    tensors: dict[str, torch.Tensor],
) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    save_file(tensors, path / "model.safetensors")


def test_build_overlay_links_weights_and_exposes_three_layers(tmp_path: Path) -> None:
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    output = tmp_path / "overlay"
    _write_checkpoint(
        base,
        _config("MiMoForCausalLM", 1),
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "model.mtp_layers.0.input_proj.weight": torch.ones(1),
        },
    )
    _write_checkpoint(
        extra,
        _config("MiMoMTPModel", 3),
        {
            "model.mtp_layers.1.input_proj.weight": torch.full((1,), 2.0),
            "model.mtp_layers.2.input_proj.weight": torch.full((1,), 3.0),
        },
    )

    manifest = build_overlay(base, extra, output)
    checked = check_overlay(output)

    assert manifest["base_mtp_layers"] == [0]
    assert manifest["extra_mtp_layers"] == [1, 2]
    assert checked["mtp_layers"] == [0, 1, 2]
    assert checked["physical_mtp_layers"] == 3
    assert json.loads((output / "config.json").read_text())[  # type: ignore[index]
        "num_nextn_predict_layers"
    ] == 3
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["model.mtp_layers.0.input_proj.weight"].startswith(
        "base-"
    )
    assert index["weight_map"]["model.mtp_layers.2.input_proj.weight"].startswith(
        "mtp-extra-"
    )
    assert (output / "base-model.safetensors").is_symlink()
    assert (output / "mtp-extra-model.safetensors").is_symlink()


def test_build_overlay_rejects_missing_extra_layer(tmp_path: Path) -> None:
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    _write_checkpoint(
        base,
        _config("MiMoForCausalLM", 1),
        {"model.mtp_layers.0.weight": torch.zeros(1)},
    )
    _write_checkpoint(
        extra,
        _config("MiMoMTPModel", 3),
        {"model.mtp_layers.1.weight": torch.zeros(1)},
    )
    with pytest.raises(MiMoCheckpointError, match="layers 1 and 2"):
        build_overlay(base, extra, tmp_path / "overlay")


def test_build_overlay_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "overlay"
    output.mkdir()
    with pytest.raises(MiMoCheckpointError, match="already exists"):
        build_overlay(tmp_path / "base", tmp_path / "extra", output)


def test_logical_tree_view_keeps_three_physical_layers(tmp_path: Path) -> None:
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    overlay = tmp_path / "overlay"
    tree = tmp_path / "tree14"
    _write_checkpoint(
        base,
        _config("MiMoForCausalLM", 1),
        {"model.mtp_layers.0.weight": torch.zeros(1)},
    )
    _write_checkpoint(
        extra,
        _config("MiMoMTPModel", 3),
        {
            "model.mtp_layers.1.weight": torch.ones(1),
            "model.mtp_layers.2.weight": torch.full((1,), 2.0),
        },
    )
    build_overlay(base, extra, overlay)
    result = build_logical_view(overlay, tree, 14)
    assert result["num_nextn_predict_layers"] == 14
    assert result["physical_mtp_layers"] == 3
    assert result["mtp_layers"] == [0, 1, 2]
    assert (tree / "model.safetensors.index.json").is_symlink()
