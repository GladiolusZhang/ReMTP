"""Generate reproducible HumanEval solutions through a vLLM chat server.

This module deliberately does not execute generated code.  Evaluation is a
separate Docker-isolated step implemented by :mod:`remtp.humaneval_evaluator`.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from remtp.benchmark import (
    _display,
    _http_request,
    aggregate_records,
    chat_completion,
    fetch_metrics,
    metric_delta,
)
from remtp.benchmark_checkpoint import (
    append_benchmark_checkpoint,
    initialize_benchmark_output,
)


PROMPT_TEMPLATE = """Complete the Python function below.

Requirements:
- Preserve the function name, signature, and documented behavior.
- Return only valid Python code containing the complete function.
- Do not use Markdown fences and do not include an explanation.

{prompt}"""


def benchmark_messages(
    content: str,
    *,
    empty_system_prompt: bool,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if empty_system_prompt:
        messages.append({"role": "system", "content": ""})
    messages.append({"role": "user", "content": content})
    return messages


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _open_jsonl(path: Path) -> TextIO:
    with path.open("rb") as probe:
        gzip_magic = probe.read(2) == b"\x1f\x8b"
    if path.suffix == ".gz" or gzip_magic:
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_humaneval(path: Path) -> list[dict[str, Any]]:
    """Load and validate the official HumanEval JSONL/JSONL.GZ format."""
    rows: list[dict[str, Any]] = []
    required = ("task_id", "prompt", "canonical_solution", "test", "entry_point")
    seen: set[str] = set()
    with _open_jsonl(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid HumanEval JSONL row {line_number}: {exc}"
                ) from exc
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(
                    f"HumanEval row {line_number} is missing: {', '.join(missing)}"
                )
            task_id = str(row["task_id"])
            if task_id in seen:
                raise ValueError(f"duplicate HumanEval task_id: {task_id}")
            seen.add(task_id)
            rows.append({key: str(row[key]) for key in required})
    if not rows:
        raise ValueError(f"HumanEval data is empty: {path}")
    return rows


def _task_sort_key(row: dict[str, Any]) -> tuple[str, int | str]:
    task_id = str(row["task_id"])
    prefix, _, suffix = task_id.rpartition("/")
    try:
        return prefix, int(suffix)
    except ValueError:
        return prefix, suffix


def sample_humaneval(
    rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if samples > len(rows):
        raise ValueError(
            f"HumanEval has {len(rows)} rows; cannot sample {samples}"
        )
    if samples == len(rows):
        selected = list(rows)
    else:
        selected = random.Random(seed).sample(rows, samples)
    return sorted(selected, key=_task_sort_key)


def sanitize_generation(text: str, entry_point: str) -> tuple[str, str]:
    """Remove chat wrappers while preserving either a full function or body."""
    cleaned = THINK_PATTERN.sub("", text).strip("\r\n")
    fences = FENCE_PATTERN.findall(cleaned)
    if fences:
        matching = [block for block in fences if f"def {entry_point}" in block]
        cleaned = max(matching or fences, key=len).strip("\r\n")
    cleaned = cleaned.replace("```python", "").replace("```py", "")
    cleaned = cleaned.replace("```", "").strip("\r\n")

    definition = re.search(
        rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(",
        cleaned,
    )
    if definition is not None:
        try:
            module = ast.parse(cleaned)
        except SyntaxError:
            module = None
        if module is not None and any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
            for node in module.body
        ):
            return cleaned.rstrip(), "full_function"
        # Drop conversational prose before a complete generated function.
        line_start = cleaned.rfind("\n", 0, definition.start()) + 1
        cleaned = cleaned[line_start:].strip()
        return cleaned, "full_function"
    return cleaned.rstrip(), "completion"


def add_pending_quality_fields(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    truncated = sum(record["finish_reason"] == "length" for record in records)
    summary.update(
        {
            "correct": None,
            "accuracy": None,
            "pass_at_1": None,
            "evaluated": 0,
            "evaluation_status": "pending_docker_evaluation",
            "truncated_outputs": truncated,
            "truncation_rate": truncated / len(records) if records else None,
        }
    )
    return summary


def write_humaneval_summary(
    output_dir: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> None:
    payload = {"config": config, "results": [summary]}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    pass_at_1 = summary.get("pass_at_1")
    quality = "pending" if pass_at_1 is None else f"{100.0 * pass_at_1:.1f}%"
    acceptance = summary.get("draft_token_acceptance_rate")
    acceptance_text = (
        "n/a" if acceptance is None else f"{100.0 * acceptance:.1f}%"
    )
    lines = [
        "# ReMTP HumanEval",
        "",
        f"- samples: `{config['samples']}`",
        f"- sample seed: `{config['sample_seed']}`",
        f"- temperature: `{config['temperature']}`",
        f"- generation seed: `{config['generation_seed']}`",
        f"- MTP draft tokens per round: `{config['mtp_tokens']}`",
        f"- max output tokens: `{config['max_tokens']}`",
        f"- execution: `{summary['evaluation_status']}`",
        "",
        "| method | samples | pass@1 | decode tok/s | e2e tok/s | "
        "mean acceptance length | draft acceptance | truncation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {config['run_name']} | {summary['samples']} | {quality} | "
        f"{_display(summary['decode_tok_s'])} | "
        f"{_display(summary['e2e_output_tok_s'])} | "
        f"{_display(summary['mean_acceptance_length'])} | "
        f"{acceptance_text} | "
        f"{100.0 * summary['truncation_rate']:.1f}% |",
    ]
    if summary.get("evaluation_counts"):
        lines.extend(
            [
                "",
                "Evaluation outcomes: `"
                + json.dumps(summary["evaluation_counts"], sort_keys=True)
                + "`.",
            ]
        )
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HumanEval solutions without executing them."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/humaneval/HumanEval.jsonl.gz"),
    )
    parser.add_argument("--samples", type=int, default=164)
    parser.add_argument("--sample-seed", type=int, default=20260802)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--mtp-tokens", type=int, default=6)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--run-name", default="native_mtp")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an output directory from requests.checkpoint.jsonl",
    )
    parser.add_argument(
        "--empty-system-prompt",
        action="store_true",
        help="prepend the explicit empty system message recommended by MiMo",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        print(
            f"HumanEval data not found: {data_path}\n"
            "Run ./scripts/download_humaneval.sh first.",
            file=sys.stderr,
        )
        return 2
    if args.max_tokens <= 0 or args.progress_every <= 0:
        print("max-tokens and progress-every must be positive", file=sys.stderr)
        return 2
    if args.temperature <= 0:
        print("temperature must be > 0", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    try:
        _http_request(f"{base_url}/health", timeout=args.timeout)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    rows = load_humaneval(data_path)
    selected = sample_humaneval(rows, args.samples, args.sample_seed)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
        output_dir = Path("results") / (
            f"humaneval_{safe_name}_n{args.samples}_t{args.temperature:g}_{stamp}"
        )
    manifest = [
        {
            "task_id": row["task_id"],
            "prompt_sha256": hashlib.sha256(
                row["prompt"].encode("utf-8")
            ).hexdigest(),
            "test_sha256": hashlib.sha256(
                row["test"].encode("utf-8")
            ).hexdigest(),
        }
        for row in selected
    ]
    records, checkpoint_path = initialize_benchmark_output(
        output_dir,
        manifest,
        resume=args.resume,
        identity_key="task_id",
    )

    if not args.skip_warmup:
        warmup = selected[0]
        print(f"Warmup: {warmup['task_id']} (excluded)")
        warmup_content = PROMPT_TEMPLATE.format(prompt=warmup["prompt"])
        chat_completion(
            base_url,
            args.model,
            benchmark_messages(
                warmup_content,
                empty_system_prompt=args.empty_system_prompt,
            ),
            args.temperature,
            args.generation_seed,
            min(96, args.max_tokens),
            args.timeout,
        )

    completed = len(records)
    if completed:
        print(f"Resuming HumanEval from sample {completed + 1}/{len(selected)}")
    print(f"\nTask humaneval: {len(selected)} samples (generation only)")
    for sample_index, row in enumerate(selected, start=1):
        if sample_index <= completed:
            continue
        task_number = _task_sort_key(row)[1]
        seed_offset = task_number if isinstance(task_number, int) else sample_index
        request_seed = args.generation_seed + int(seed_offset) * 10
        messages = benchmark_messages(
            PROMPT_TEMPLATE.format(prompt=row["prompt"]),
            empty_system_prompt=args.empty_system_prompt,
        )
        before = fetch_metrics(base_url, args.timeout)
        started = time.perf_counter()
        response = chat_completion(
            base_url,
            args.model,
            messages,
            args.temperature,
            request_seed,
            args.max_tokens,
            args.timeout,
        )
        client_seconds = time.perf_counter() - started
        after = fetch_metrics(base_url, args.timeout)

        choice = response["choices"][0]
        raw_output = choice["message"]["content"] or ""
        candidate, candidate_mode = sanitize_generation(
            raw_output, row["entry_point"]
        )
        usage = response.get("usage") or {}
        record = {
            "task": "humaneval",
            "task_id": row["task_id"],
            "sample_index": sample_index,
            "seed": request_seed,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "client_seconds": client_seconds,
            "finish_reason": choice.get("finish_reason"),
            "entry_point": row["entry_point"],
            "candidate_mode": candidate_mode,
            "candidate": candidate,
            "raw_output": raw_output,
            "correct": None,
            "evaluation_status": "pending",
            "metrics": metric_delta(before, after),
        }
        records.append(record)
        append_benchmark_checkpoint(checkpoint_path, record)
        if (
            sample_index == 1
            or sample_index % args.progress_every == 0
            or sample_index == len(selected)
        ):
            print(
                f"  [{sample_index:03d}/{len(selected):03d}] "
                f"task={row['task_id']} tokens={record['output_tokens']} "
                f"mode={candidate_mode}"
            )

    summary = add_pending_quality_fields(
        aggregate_records(records, "humaneval", len(selected)), records
    )
    config = {
        "data": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "samples": args.samples,
        "sample_seed": args.sample_seed,
        "temperature": args.temperature,
        "generation_seed": args.generation_seed,
        "max_tokens": args.max_tokens,
        "mtp_tokens": args.mtp_tokens,
        "base_url": base_url,
        "model": args.model,
        "run_name": args.run_name,
        "progress_every": args.progress_every,
        "answer_metric": "HumanEval pass@1 (official tests, one sample per task)",
        "execution_policy": "separate restricted Docker container per task",
        "system_prompt": "" if args.empty_system_prompt else "tokenizer_default",
    }
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_humaneval_summary(output_dir, summary, config)

    print(f"\nGenerated solutions: {output_dir.resolve()}")
    print(
        f"decode={_display(summary['decode_tok_s'])} tok/s, "
        f"e2e={_display(summary['e2e_output_tok_s'])} tok/s, "
        f"MAL={_display(summary['mean_acceptance_length'])}"
    )
    print("Quality is pending isolated evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
