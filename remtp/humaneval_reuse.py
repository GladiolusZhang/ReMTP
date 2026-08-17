"""Reuse protocol-compatible HumanEval runs without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from remtp.humaneval_compare import PROTOCOL_KEYS


ANSWER_METRIC = "HumanEval pass@1 (official tests, one sample per task)"


def _load_complete_config(directory: Path) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    manifest_path = directory / "sample_manifest.json"
    requests_path = directory / "requests.jsonl"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError("missing summary.json or sample_manifest.json")
    if not requests_path.is_file():
        raise ValueError("missing requests.jsonl")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    config = payload.get("config")
    results = payload.get("results")
    if not isinstance(config, dict):
        raise ValueError("missing benchmark config")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("expected exactly one benchmark result")
    if results[0].get("evaluation_status") != "complete":
        raise ValueError("HumanEval evaluation is incomplete")
    return dict(config)


def protocol_mismatches(
    config: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    return [
        key
        for key in PROTOCOL_KEYS
        if config.get(key) != expected.get(key)
    ]


def find_compatible_run(
    roots: list[Path],
    profile: str,
    expected: dict[str, Any],
    *,
    destination_root: Path,
) -> Path | None:
    for root in roots:
        candidate = root / profile
        if candidate.resolve() == (destination_root / profile).resolve():
            continue
        try:
            config = _load_complete_config(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not protocol_mismatches(config, expected):
            return candidate.resolve()
    return None


def reuse_profiles(
    *,
    source_roots: list[Path],
    destination_root: Path,
    profiles: list[str],
    expected: dict[str, Any],
    required: bool,
) -> dict[str, Path]:
    destination_root.mkdir(parents=True, exist_ok=True)
    reused: dict[str, Path] = {}
    missing: list[str] = []
    for profile in profiles:
        destination = destination_root / profile
        if destination.exists() or destination.is_symlink():
            try:
                config = _load_complete_config(destination)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"existing destination for {profile} is not reusable: {exc}"
                ) from exc
            mismatched = protocol_mismatches(config, expected)
            if mismatched:
                raise ValueError(
                    f"existing destination protocol differs for {profile}: "
                    + ", ".join(mismatched)
                )
            reused[profile] = destination.resolve()
            continue

        source = find_compatible_run(
            source_roots,
            profile,
            expected,
            destination_root=destination_root,
        )
        if source is None:
            missing.append(profile)
            continue
        relative_source = os.path.relpath(source, start=destination_root)
        destination.symlink_to(relative_source, target_is_directory=True)
        reused[profile] = source

    if missing and required:
        raise ValueError(
            "no protocol-compatible historical result for: "
            + ", ".join(missing)
        )
    return reused


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--evaluation-image", required=True)
    parser.add_argument("--evaluation-image-id", required=True)
    parser.add_argument("--evaluation-timeout", type=float, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = {
        "samples": args.samples,
        "sample_seed": args.sample_seed,
        "temperature": args.temperature,
        "generation_seed": args.generation_seed,
        "max_tokens": args.max_tokens,
        "mtp_tokens": args.mtp_tokens,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "answer_metric": ANSWER_METRIC,
        "evaluation_image": args.evaluation_image,
        "evaluation_image_id": args.evaluation_image_id,
        "evaluation_timeout_seconds": args.evaluation_timeout,
    }
    try:
        reused = reuse_profiles(
            source_roots=args.source_root,
            destination_root=args.destination_root,
            profiles=args.profiles,
            expected=expected,
            required=not args.allow_missing,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for profile in args.profiles:
        source = reused.get(profile)
        if source is None:
            print(f"[run]   {profile}: no compatible historical result")
        else:
            print(f"[reuse] {profile}: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
