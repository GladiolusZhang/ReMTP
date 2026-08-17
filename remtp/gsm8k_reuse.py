"""Reuse protocol-compatible GSM8K result directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROTOCOL_KEYS = (
    "samples",
    "sample_seed",
    "temperature",
    "generation_seed",
    "max_tokens",
    "mtp_tokens",
    "data_sha256",
    "answer_metric",
)
ANSWER_METRIC = "normalized_numeric_exact_match"


def _load(directory: Path) -> dict[str, Any]:
    for name in ("summary.json", "sample_manifest.json", "requests.jsonl"):
        if not (directory / name).is_file():
            raise ValueError(f"missing {name}")
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    config = payload.get("config")
    results = payload.get("results")
    if not isinstance(config, dict):
        raise ValueError("missing benchmark config")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("expected exactly one benchmark result")
    return dict(config)


def _mismatches(config: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    return [key for key in PROTOCOL_KEYS if config.get(key) != expected.get(key)]


def reuse_profiles(
    source_roots: list[Path],
    destination_root: Path,
    profiles: list[str],
    expected: dict[str, Any],
    *,
    required: bool,
) -> dict[str, Path]:
    destination_root.mkdir(parents=True, exist_ok=True)
    reused: dict[str, Path] = {}
    missing: list[str] = []
    for profile in profiles:
        destination = destination_root / profile
        source: Path | None = None
        for root in source_roots:
            candidate = root / profile
            try:
                config = _load(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not _mismatches(config, expected):
                source = candidate.resolve()
                break
        if source is None:
            missing.append(profile)
            continue
        destination.symlink_to(
            os.path.relpath(source, start=destination_root),
            target_is_directory=True,
        )
        reused[profile] = source
    if missing and required:
        raise ValueError("no compatible historical result for: " + ", ".join(missing))
    return reused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--mtp-tokens", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    expected = {
        "samples": args.samples,
        "sample_seed": args.sample_seed,
        "temperature": args.temperature,
        "generation_seed": args.generation_seed,
        "max_tokens": args.max_tokens,
        "mtp_tokens": args.mtp_tokens,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "answer_metric": ANSWER_METRIC,
    }
    try:
        reused = reuse_profiles(
            args.source_root,
            args.destination_root,
            args.profiles,
            expected,
            required=not args.allow_missing,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for profile in args.profiles:
        source = reused.get(profile)
        print(
            f"[reuse] {profile}: {source}"
            if source is not None
            else f"[run]   {profile}: no compatible historical result"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
