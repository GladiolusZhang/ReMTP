"""Send a generic JSON/JSONL corpus through vLLM to collect router traces."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any, TextIO

from remtp.benchmark import _http_request, chat_completion


def _open_text(path: Path) -> TextIO:
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if path.suffix == ".gz" or compressed:
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Accept JSON arrays or JSONL rows containing messages/prompt/text."""
    with _open_text(path) as handle:
        content = handle.read()
    if not content.strip():
        raise ValueError("router corpus is empty")
    if content.lstrip().startswith("["):
        payload = json.loads(content)
        if not isinstance(payload, list):
            raise ValueError("JSON corpus must be an array")
        rows = payload
    else:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"corpus row {index} is not an object")
        messages = row.get("messages")
        if messages is not None:
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"corpus row {index} has invalid messages")
            normalized.append({"messages": messages})
            continue
        prompt = row.get("prompt", row.get("instruction", row.get("text")))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"corpus row {index} needs messages, prompt, instruction, or text"
            )
        normalized.append(
            {"messages": [{"role": "user", "content": prompt.strip()}]}
        )
    if not normalized:
        raise ValueError("router corpus has no usable rows")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20260803)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.max_tokens < 1 or args.progress_every < 1:
        print("samples, max-tokens, and progress-every must be positive", file=sys.stderr)
        return 2
    if args.temperature <= 0.0:
        print("temperature must be positive", file=sys.stderr)
        return 2
    path = args.data.resolve()
    if not path.is_file():
        print(f"router corpus not found: {path}", file=sys.stderr)
        return 2
    base_url = args.base_url.rstrip("/")
    try:
        _http_request(f"{base_url}/health", timeout=args.timeout)
        rows = load_corpus(path)
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    sample_count = min(args.samples, len(rows))
    selected = random.Random(args.sample_seed).sample(rows, sample_count)
    print(
        f"Collecting regret-router traces: samples={sample_count} "
        f"temperature={args.temperature:g} max_tokens={args.max_tokens}"
    )
    for index, row in enumerate(selected, start=1):
        response = chat_completion(
            base_url,
            args.model,
            row["messages"],
            args.temperature,
            args.generation_seed + index * 10,
            args.max_tokens,
            args.timeout,
        )
        usage = response.get("usage") or {}
        if index == 1 or index % args.progress_every == 0 or index == sample_count:
            print(
                f"  [{index:04d}/{sample_count:04d}] "
                f"output_tokens={int(usage.get('completion_tokens', 0))}"
            )
    # EngineCore may terminate without running Python atexit handlers. Starting
    # one final request crosses a request boundary and checkpoints all records
    # from the final real sample; the sentinel's own trace is intentionally
    # discarded when the server stops.
    chat_completion(
        base_url,
        args.model,
        [{"role": "user", "content": "Reply with OK."}],
        args.temperature,
        args.generation_seed + sample_count * 10 + 1,
        1,
        args.timeout,
    )
    print("Trace requests complete. Stop the server to flush the final shard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
