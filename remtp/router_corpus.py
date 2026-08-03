"""Build the mixed, answer-free corpus used to collect regret-router traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable


PROFILES = {
    "pilot": {
        "ultrachat": 1500,
        "no_robots": 1000,
        "magicoder": 1000,
        "oasst1": 500,
    },
    "full": {
        "ultrachat": 3000,
        "no_robots": 2000,
        "magicoder": 2000,
        "oasst1": 1000,
    },
}

SOURCE_SPECS = {
    "ultrachat": {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "field": "prompt",
        "license": "MIT",
    },
    "no_robots": {
        "dataset": "HuggingFaceH4/no_robots",
        "split": "train",
        "field": "prompt",
        "license": "CC-BY-NC-4.0",
    },
    "dolly": {
        "dataset": "databricks/databricks-dolly-15k",
        "split": "train",
        "field": "instruction",
        "license": "CC-BY-SA-3.0",
    },
    "magicoder": {
        "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "split": "train",
        "field": "problem",
        "license": "MIT",
    },
    "oasst1": {
        "dataset": "OpenAssistant/oasst1",
        "split": "train",
        "field": "text",
        "license": "Apache-2.0",
    },
}


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _oasst_root(row: dict[str, Any]) -> bool:
    return (
        row.get("role") == "prompter"
        and row.get("parent_id") is None
        and row.get("lang") == "en"
        and row.get("review_result") is True
        and not bool(row.get("deleted", False))
    )


def select_source(
    dataset: Any,
    *,
    source: str,
    field: str,
    count: int,
    rng: random.Random,
    seen: set[str],
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, str]]:
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected: list[dict[str, str]] = []
    for index in indices:
        row = dataset[index]
        if predicate is not None and not predicate(row):
            continue
        value = row.get(field)
        if not isinstance(value, str):
            continue
        prompt = normalize_prompt(value)
        if not 24 <= len(prompt) <= 6000:
            continue
        key = prompt.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append({"prompt": prompt, "source": source})
        if len(selected) == count:
            return selected
    raise ValueError(
        f"source {source} yielded {len(selected)} usable prompts; "
        f"requested {count}"
    )


def build_corpus(
    *,
    profile: str,
    output: Path,
    seed: int,
    instruction_source: str = "no_robots",
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing datasets/pyarrow. Run: uv pip install -U datasets pyarrow "
            "'huggingface-hub>=0.34,<1.0'"
        ) from exc
    quotas = PROFILES[profile]
    rng = random.Random(seed)
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    used_sources: set[str] = set()
    for quota_source, count in quotas.items():
        source = (
            instruction_source
            if quota_source == "no_robots"
            else quota_source
        )
        spec = SOURCE_SPECS[source]
        used_sources.add(source)
        print(
            f"Loading {spec['dataset']} split={spec['split']} "
            f"for {count} {source} prompts",
            flush=True,
        )
        dataset = load_dataset(spec["dataset"], split=spec["split"])
        records.extend(
            select_source(
                dataset,
                source=source,
                field=spec["field"],
                count=count,
                rng=rng,
                seen=seen,
                predicate=_oasst_root if source == "oasst1" else None,
            )
        )
    rng.shuffle(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "format_version": 1,
        "profile": profile,
        "instruction_source": instruction_source,
        "seed": seed,
        "records": len(records),
        "counts": dict(sorted(Counter(row["source"] for row in records).items())),
        "sha256": digest,
        "sources": {
            source: SOURCE_SPECS[source]
            for source in sorted(used_sources)
        },
        "contains_answers": False,
        "decontamination_status": "pending",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} prompts to {output}")
    print(f"Manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/regret_router/router_corpus_raw.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--instruction-source",
        choices=("no_robots", "dolly"),
        default="no_robots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_corpus(
        profile=args.profile,
        output=args.output,
        seed=args.seed,
        instruction_source=args.instruction_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
