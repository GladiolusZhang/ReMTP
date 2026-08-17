"""Validate FastMTP checkpoints and build no-copy logical tree views.

TencentBAC/FastMTP contains one trained physical MTP layer.  vLLM uses
``num_nextn_predict_layers`` both as a model-construction value and as the
maximum speculative-tree width.  A dynamic tree therefore needs a logical
view that advertises ``N`` nodes while still instantiating and loading only
the single trained physical layer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


MANIFEST_NAME = "remtp_fastmtp_view.json"
EXPECTED_ARCHITECTURE = "MiMoForCausalLM"


class FastMTPCheckpointError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FastMTPCheckpointError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FastMTPCheckpointError(f"invalid JSON file: {path}: {exc}") from exc


def _layer_ids(keys: Iterable[str]) -> set[int]:
    prefix = "model.mtp_layers."
    result: set[int] = set()
    for key in keys:
        if not key.startswith(prefix):
            continue
        part = key[len(prefix) :].split(".", 1)[0]
        if part.isdigit():
            result.add(int(part))
    return result


def _weight_map(checkpoint: Path) -> dict[str, str]:
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        payload = _read_json(index)
        mapping = payload.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise FastMTPCheckpointError(f"invalid weight map: {index}")
        result = {str(key): str(value) for key, value in mapping.items()}
        missing = sorted(
            str(checkpoint / name)
            for name in set(result.values())
            if not (checkpoint / name).is_file()
        )
        if missing:
            raise FastMTPCheckpointError(f"missing safetensors shards: {missing}")
        return result

    files = tuple(sorted(checkpoint.glob("*.safetensors")))
    if not files:
        raise FastMTPCheckpointError(f"no safetensors weights found in {checkpoint}")
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - vLLM env always has it.
        raise FastMTPCheckpointError("safetensors is required") from exc
    result: dict[str, str] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise FastMTPCheckpointError(f"duplicate weight key: {key}")
                result[key] = path.name
    return result


def check_fastmtp(path: Path) -> dict[str, Any]:
    path = path.resolve()
    config = _read_json(path / "config.json")
    architectures = config.get("architectures", [])
    if EXPECTED_ARCHITECTURE not in architectures:
        raise FastMTPCheckpointError(
            f"FastMTP must use {EXPECTED_ARCHITECTURE}, got {architectures}"
        )
    logical_layers = int(config.get("num_nextn_predict_layers", 0))
    physical_layers = int(config.get("remtp_physical_mtp_layers", logical_layers))
    if physical_layers != 1:
        raise FastMTPCheckpointError(
            "TencentBAC/FastMTP must expose exactly one trained physical MTP layer; "
            f"found {physical_layers}"
        )
    mapping = _weight_map(path)
    layers = _layer_ids(mapping)
    if layers != {0}:
        raise FastMTPCheckpointError(
            f"FastMTP checkpoint must contain only trained MTP layer 0, got {sorted(layers)}"
        )
    return {
        "path": str(path),
        "logical_mtp_width": logical_layers,
        "physical_mtp_layers": physical_layers,
        "mtp_layers": sorted(layers),
        "weight_count": len(mapping),
    }


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def build_logical_view(source: Path, output: Path, logical_width: int) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if logical_width < 1:
        raise FastMTPCheckpointError("logical tree width must be positive")
    check_fastmtp(source)
    if output.exists():
        checked = check_fastmtp(output)
        if int(checked["logical_mtp_width"]) != logical_width:
            raise FastMTPCheckpointError(
                f"existing view has logical width {checked['logical_mtp_width']}, "
                f"expected {logical_width}: {output}"
            )
        return checked

    config = _read_json(source / "config.json")
    config["num_nextn_predict_layers"] = int(logical_width)
    config["remtp_physical_mtp_layers"] = 1

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for item in source.iterdir():
            if item.name in {"config.json", MANIFEST_NAME, ".cache"}:
                continue
            _relative_symlink(item, temporary / item.name)
        (temporary / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "format": 1,
                    "kind": "fastmtp-logical-tree-view",
                    "source": str(source),
                    "logical_mtp_width": int(logical_width),
                    "physical_mtp_layers": 1,
                    "physical_route": "repeat-0",
                    "uses_symlinks": True,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return check_fastmtp(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("path", type=Path)
    logical = sub.add_parser("logical-view")
    logical.add_argument("--source", type=Path, required=True)
    logical.add_argument("--output", type=Path, required=True)
    logical.add_argument("--width", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "check":
        result = check_fastmtp(args.path)
    else:
        result = build_logical_view(args.source, args.output, args.width)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
