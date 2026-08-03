"""Remove benchmark overlap from a regret-router prompt corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
FUNCTION_PATTERN = re.compile(
    r"\bdef\s+([A-Za-z_]\w*)\s*(\([^)]*\)(?:\s*->\s*[^:\n]+)?)\s*:",
    re.MULTILINE,
)
DOCSTRING_PATTERN = re.compile(
    r"(?:'''|\"\"\")(.*?)(?:'''|\"\"\")",
    re.DOTALL,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(normalize_text(text)))


def ngrams(tokens: tuple[str, ...], size: int) -> set[str]:
    if len(tokens) < size:
        return set()
    return {
        "\x1f".join(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def _open_text(path: Path) -> TextIO:
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if path.suffix == ".gz" or compressed:
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with _open_text(path) as handle:
        content = handle.read()
    if not content.strip():
        raise ValueError(f"empty dataset: {path}")
    if content.lstrip().startswith("[") or content.lstrip().startswith("{"):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            lists = [value for value in payload.values() if isinstance(value, list)]
            rows = lists[0] if len(lists) == 1 else [payload]
        else:
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    return [row for row in rows if isinstance(row, dict)]


def prompt_from_row(row: dict[str, Any]) -> str | None:
    for key in (
        "prompt",
        "input_prompt",
        "question",
        "instruction",
        "problem",
        "text",
        "input",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = [
            message.get("content", "")
            for message in messages
            if isinstance(message, dict)
            and message.get("role") in {None, "user", "prompter"}
            and isinstance(message.get("content"), str)
        ]
        text = "\n".join(part for part in parts if part.strip())
        if text:
            return text
    return None


def _minhash(shingles: set[str], *, num_perm: int) -> Any:
    from datasketch import MinHash

    result = MinHash(num_perm=num_perm)
    for shingle in sorted(shingles):
        result.update(shingle.encode("utf-8"))
    return result


class BenchmarkIndex:
    def __init__(
        self,
        benchmarks: list[tuple[str, Path]],
        *,
        ngram_size: int,
        shingle_size: int,
        lsh_threshold: float,
        jaccard_threshold: float,
        num_perm: int,
    ) -> None:
        try:
            from datasketch import MinHashLSH
        except ImportError as exc:
            raise RuntimeError(
                "Missing datasketch. Run: uv pip install -U datasketch"
            ) from exc
        self.ngram_size = ngram_size
        self.shingle_size = shingle_size
        self.jaccard_threshold = jaccard_threshold
        self.num_perm = num_perm
        self.exact: dict[str, str] = {}
        self.long_ngrams: dict[str, str] = {}
        self.shingles: dict[str, set[str]] = {}
        self.lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        self.function_names: set[str] = set()
        self.signatures: set[str] = set()
        self.docstrings: set[str] = set()
        self.benchmark_manifest: list[dict[str, Any]] = []
        for name, path in benchmarks:
            rows = load_rows(path)
            loaded = 0
            for index, row in enumerate(rows):
                prompt = prompt_from_row(row)
                if not prompt:
                    continue
                key = f"{name}:{index}"
                normalized = normalize_text(prompt)
                tokens = tokenize(prompt)
                self.exact.setdefault(normalized, key)
                for value in ngrams(tokens, ngram_size):
                    self.long_ngrams.setdefault(value, key)
                shingles = ngrams(tokens, shingle_size)
                if shingles:
                    self.shingles[key] = shingles
                    self.lsh.insert(
                        key,
                        _minhash(shingles, num_perm=num_perm),
                    )
                if "humaneval" in name.casefold():
                    for function_name, signature in FUNCTION_PATTERN.findall(prompt):
                        self.function_names.add(function_name.casefold())
                        self.signatures.add(
                            normalize_text(function_name + signature)
                        )
                    self.docstrings.update(
                        normalize_text(value)
                        for value in DOCSTRING_PATTERN.findall(prompt)
                        if normalize_text(value)
                    )
                loaded += 1
            self.benchmark_manifest.append(
                {
                    "name": name,
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": len(rows),
                    "prompts_indexed": loaded,
                }
            )

    def scan(self, prompt: str) -> tuple[list[str], list[str]]:
        normalized = normalize_text(prompt)
        tokens = tokenize(prompt)
        reasons: list[str] = []
        matches: set[str] = set()
        exact = self.exact.get(normalized)
        if exact is not None:
            reasons.append("normalized_exact")
            matches.add(exact)
        long_overlap = {
            self.long_ngrams[value]
            for value in ngrams(tokens, self.ngram_size)
            if value in self.long_ngrams
        }
        if long_overlap:
            reasons.append(f"{self.ngram_size}_gram")
            matches.update(long_overlap)
        shingles = ngrams(tokens, self.shingle_size)
        if shingles:
            candidates = self.lsh.query(
                _minhash(shingles, num_perm=self.num_perm)
            )
            near = []
            for key in candidates:
                reference = self.shingles[key]
                similarity = len(shingles & reference) / max(
                    len(shingles | reference), 1
                )
                if similarity >= self.jaccard_threshold:
                    near.append(key)
            if near:
                reasons.append("minhash_jaccard")
                matches.update(near)
        candidate_functions = {
            function_name.casefold()
            for function_name, _ in FUNCTION_PATTERN.findall(prompt)
        }
        if candidate_functions & self.function_names:
            reasons.append("humaneval_function_name")
        candidate_signatures = {
            normalize_text(function_name + signature)
            for function_name, signature in FUNCTION_PATTERN.findall(prompt)
        }
        if candidate_signatures & self.signatures:
            reasons.append("humaneval_signature")
        candidate_docstrings = {
            normalize_text(value)
            for value in DOCSTRING_PATTERN.findall(prompt)
            if normalize_text(value)
        }
        if candidate_docstrings & self.docstrings:
            reasons.append("humaneval_docstring")
        return sorted(set(reasons)), sorted(matches)


def parse_benchmarks(values: Iterable[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--benchmark must use NAME=/path/to/data")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"benchmark not found: {path}")
        result.append((name, path))
    if not result:
        raise ValueError("at least one benchmark is required")
    return result


def decontaminate(
    *,
    corpus: Path,
    output: Path,
    report_path: Path,
    benchmarks: list[tuple[str, Path]],
    ngram_size: int = 13,
    shingle_size: int = 5,
    lsh_threshold: float = 0.70,
    jaccard_threshold: float = 0.80,
    num_perm: int = 128,
) -> dict[str, Any]:
    index = BenchmarkIndex(
        benchmarks,
        ngram_size=ngram_size,
        shingle_size=shingle_size,
        lsh_threshold=lsh_threshold,
        jaccard_threshold=jaccard_threshold,
        num_perm=num_perm,
    )
    records = load_rows(corpus)
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    reasons = Counter()
    source_removed = Counter()
    for row_index, row in enumerate(records):
        prompt = prompt_from_row(row)
        if prompt is None:
            raise ValueError(f"corpus row {row_index + 1} has no prompt")
        row_reasons, matches = index.scan(prompt)
        if row_reasons:
            reasons.update(row_reasons)
            source = str(row.get("source", "unknown"))
            source_removed[source] += 1
            removed.append(
                {
                    "row": row_index,
                    "source": source,
                    "prompt_sha256": hashlib.sha256(
                        normalize_text(prompt).encode("utf-8")
                    ).hexdigest(),
                    "reasons": row_reasons,
                    "benchmark_matches": matches[:20],
                }
            )
        else:
            kept.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "format_version": 1,
        "input": str(corpus.resolve()),
        "input_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "input_records": len(records),
        "kept_records": len(kept),
        "removed_records": len(removed),
        "reason_counts": dict(sorted(reasons.items())),
        "removed_by_source": dict(sorted(source_removed.items())),
        "parameters": {
            "ngram_size": ngram_size,
            "shingle_size": shingle_size,
            "lsh_threshold": lsh_threshold,
            "jaccard_threshold": jaccard_threshold,
            "num_perm": num_perm,
        },
        "benchmarks": index.benchmark_manifest,
        "removed": removed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Decontamination: kept={len(kept)} removed={len(removed)} "
        f"output={output}"
    )
    print(f"Audit report: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--lsh-threshold", type=float, default=0.70)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--num-perm", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmarks = parse_benchmarks(args.benchmark)
    decontaminate(
        corpus=args.corpus.resolve(),
        output=args.output.resolve(),
        report_path=args.report.resolve(),
        benchmarks=benchmarks,
        ngram_size=args.ngram_size,
        shingle_size=args.shingle_size,
        lsh_threshold=args.lsh_threshold,
        jaccard_threshold=args.jaccard_threshold,
        num_perm=args.num_perm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
