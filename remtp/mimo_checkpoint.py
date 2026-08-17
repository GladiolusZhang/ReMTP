"""Build and validate a MiMo-7B checkpoint overlay with three MTP layers.

Xiaomi publishes the target checkpoint and the two additional MTP layers as
separate Hugging Face repositories.  This module creates a lightweight local
overlay: large safetensors files remain in their download directories and are
referenced by symlink, while one combined weight index and a config with
``num_nextn_predict_layers=3`` are written to the output directory.

No model weights are copied or modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

from safetensors import safe_open


BASE_ARCHITECTURE = "MiMoForCausalLM"
EXTRA_ARCHITECTURE = "MiMoMTPModel"
EXPECTED_MTP_LAYERS = 3
MANIFEST_NAME = "remtp_mimo_overlay.json"


class MiMoCheckpointError(RuntimeError):
    """The downloaded MiMo checkpoints cannot form a valid MTP3 overlay."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MiMoCheckpointError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MiMoCheckpointError(f"invalid JSON file: {path}: {exc}") from exc


def _weight_files(checkpoint: Path) -> tuple[Path, ...]:
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _read_json(index).get("weight_map", {})
        names = sorted(set(str(name) for name in weight_map.values()))
        files = tuple(checkpoint / name for name in names)
    else:
        files = tuple(sorted(checkpoint.glob("*.safetensors")))
    if not files:
        raise MiMoCheckpointError(f"no safetensors weights found in {checkpoint}")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise MiMoCheckpointError(f"missing safetensors shards: {missing}")
    return files


def _weight_map(checkpoint: Path) -> tuple[dict[str, str], int]:
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        payload = _read_json(index)
        mapping = {str(key): str(value) for key, value in payload["weight_map"].items()}
        total_size = int(payload.get("metadata", {}).get("total_size", 0))
        return mapping, total_size

    mapping: dict[str, str] = {}
    total_size = 0
    for path in _weight_files(checkpoint):
        total_size += path.stat().st_size
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in mapping:
                    raise MiMoCheckpointError(f"duplicate weight key {key!r}")
                mapping[key] = path.name
    return mapping, total_size


def _validate_configs(base: Path, extra: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base_config = _read_json(base / "config.json")
    extra_config = _read_json(extra / "config.json")
    base_architectures = base_config.get("architectures", [])
    extra_architectures = extra_config.get("architectures", [])
    if BASE_ARCHITECTURE not in base_architectures:
        raise MiMoCheckpointError(
            f"base architecture must include {BASE_ARCHITECTURE}, got {base_architectures}"
        )
    if EXTRA_ARCHITECTURE not in extra_architectures:
        raise MiMoCheckpointError(
            f"MTP architecture must include {EXTRA_ARCHITECTURE}, got {extra_architectures}"
        )
    for field in ("hidden_size", "vocab_size", "num_attention_heads"):
        if base_config.get(field) != extra_config.get(field):
            raise MiMoCheckpointError(
                f"base/MTP config mismatch for {field}: "
                f"{base_config.get(field)} != {extra_config.get(field)}"
            )
    if int(extra_config.get("num_nextn_predict_layers", 0)) != EXPECTED_MTP_LAYERS:
        raise MiMoCheckpointError(
            "the add-on checkpoint must declare num_nextn_predict_layers=3"
        )
    return base_config, extra_config


def _layer_ids(keys: Iterable[str]) -> set[int]:
    result: set[int] = set()
    prefix = "model.mtp_layers."
    for key in keys:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        head = suffix.split(".", 1)[0]
        if head.isdigit():
            result.add(int(head))
    return result


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def build_overlay(base: Path, extra: Path, output: Path) -> dict[str, Any]:
    """Create a new MTP3 overlay and return its manifest."""
    base = base.resolve()
    extra = extra.resolve()
    output = output.resolve()
    if output.exists():
        raise MiMoCheckpointError(
            f"output already exists: {output}; choose a new path or remove it explicitly"
        )
    base_config, extra_config = _validate_configs(base, extra)
    base_map, base_size = _weight_map(base)
    extra_map, extra_size = _weight_map(extra)
    base_layers = _layer_ids(base_map)
    extra_layers = _layer_ids(extra_map)
    if 0 not in base_layers:
        raise MiMoCheckpointError("base checkpoint does not contain MTP layer 0")
    if not {1, 2}.issubset(extra_layers):
        raise MiMoCheckpointError(
            f"add-on checkpoint must contain MTP layers 1 and 2, got {sorted(extra_layers)}"
        )
    duplicate = set(base_map).intersection(extra_map)
    if duplicate:
        preview = sorted(duplicate)[:5]
        raise MiMoCheckpointError(f"base/add-on contain duplicate keys: {preview}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        combined: dict[str, str] = {}
        for origin, mapping, label in (
            (base, base_map, "base"),
            (extra, extra_map, "mtp-extra"),
        ):
            renamed: dict[str, str] = {}
            for filename in sorted(set(mapping.values())):
                destination_name = f"{label}-{filename}"
                _relative_symlink(origin / filename, temporary / destination_name)
                renamed[filename] = destination_name
            for key, filename in mapping.items():
                combined[key] = renamed[filename]

        small_files = (
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "configuration_mimo.py",
            "modeling_mimo.py",
        )
        for name in small_files:
            source = base / name
            if source.is_file():
                shutil.copy2(source, temporary / name)

        config = dict(base_config)
        config["num_nextn_predict_layers"] = EXPECTED_MTP_LAYERS
        config["remtp_physical_mtp_layers"] = EXPECTED_MTP_LAYERS
        (temporary / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        index = {
            "metadata": {"total_size": base_size + extra_size},
            "weight_map": dict(sorted(combined.items())),
        }
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": 1,
            "base": str(base),
            "mtp_extra": str(extra),
            "num_nextn_predict_layers": EXPECTED_MTP_LAYERS,
            "base_mtp_layers": sorted(base_layers),
            "extra_mtp_layers": sorted(extra_layers),
            "weight_count": len(combined),
            "uses_symlinks": True,
            "warning": (
                "MiMo extra MTP layers are pretrained-only; Xiaomi reports that "
                "they have not been validated with post-trained MiMo checkpoints."
            ),
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def check_overlay(path: Path, *, require_three_layers: bool = True) -> dict[str, Any]:
    path = path.resolve()
    config = _read_json(path / "config.json")
    if BASE_ARCHITECTURE not in config.get("architectures", []):
        raise MiMoCheckpointError(f"not a MiMo target checkpoint: {path}")
    logical_layers = int(config.get("num_nextn_predict_layers", 0))
    physical_layers = int(
        config.get("remtp_physical_mtp_layers", logical_layers)
    )
    if require_three_layers and physical_layers != EXPECTED_MTP_LAYERS:
        raise MiMoCheckpointError(
            f"expected three physical MTP layers, found {physical_layers}"
        )
    mapping, _ = _weight_map(path)
    found = _layer_ids(mapping)
    expected = set(range(physical_layers))
    missing = sorted(expected - found)
    if missing:
        raise MiMoCheckpointError(f"checkpoint is missing MTP layers {missing}")
    return {
        "path": str(path),
        "num_nextn_predict_layers": logical_layers,
        "physical_mtp_layers": physical_layers,
        "mtp_layers": sorted(found),
        "weight_count": len(mapping),
    }


def build_logical_view(source: Path, output: Path, logical_width: int) -> dict[str, Any]:
    """Create a no-copy draft view with logical width != physical layers."""
    source = source.resolve()
    output = output.resolve()
    if logical_width < EXPECTED_MTP_LAYERS:
        raise MiMoCheckpointError(
            f"logical width must be at least {EXPECTED_MTP_LAYERS}"
        )
    if output.exists():
        checked = check_overlay(output)
        if int(checked["num_nextn_predict_layers"]) != logical_width:
            raise MiMoCheckpointError(
                f"existing logical view has width {checked['num_nextn_predict_layers']}, "
                f"expected {logical_width}: {output}"
            )
        return checked
    check_overlay(source)
    config = _read_json(source / "config.json")
    config["num_nextn_predict_layers"] = int(logical_width)
    config["remtp_physical_mtp_layers"] = EXPECTED_MTP_LAYERS

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for item in source.iterdir():
            if item.name in {"config.json", MANIFEST_NAME}:
                continue
            _relative_symlink(item, temporary / item.name)
        (temporary / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": 1,
            "kind": "logical-draft-view",
            "source": str(source),
            "num_nextn_predict_layers": int(logical_width),
            "physical_mtp_layers": EXPECTED_MTP_LAYERS,
            "uses_symlinks": True,
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return check_overlay(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build a symlink-based MTP3 overlay")
    build.add_argument("--base", type=Path, required=True)
    build.add_argument("--mtp", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check", help="validate a MiMo checkpoint/overlay")
    check.add_argument("path", type=Path)
    check.add_argument("--allow-single-layer", action="store_true")
    logical = sub.add_parser(
        "logical-view", help="build a no-copy draft view with a logical tree width"
    )
    logical.add_argument("--source", type=Path, required=True)
    logical.add_argument("--output", type=Path, required=True)
    logical.add_argument("--width", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_overlay(args.base, args.mtp, args.output)
    elif args.command == "check":
        result = check_overlay(
            args.path, require_three_layers=not args.allow_single_layer
        )
    else:
        result = build_logical_view(args.source, args.output, args.width)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
