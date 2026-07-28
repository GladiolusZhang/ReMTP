"""Validate that a local Hugging Face safetensors checkpoint is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)


def missing_checkpoint_files(model_path: Path) -> list[str]:
    missing = [
        filename
        for filename in _REQUIRED_FILES
        if not (model_path / filename).is_file()
    ]

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(index["weight_map"].values()))
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            missing.append("model.safetensors.index.json (invalid)")
        else:
            missing.extend(
                shard
                for shard in shard_names
                if not (model_path / shard).is_file()
            )
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()

    missing = missing_checkpoint_files(args.model_path)
    if missing:
        print(f"Incomplete model checkpoint: {args.model_path}")
        for filename in missing:
            print(f"  missing: {filename}")
        return 1

    print(f"Model checkpoint is complete: {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
