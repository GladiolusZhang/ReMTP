"""Generate MBPP solutions for a fixed test-split subset (2026-11-16).

Data: the pinned ``mbpp.jsonl`` of ``router_benchmark_download`` (974 rows;
the standard test split is task_id 11-510).  The prompt shows the task text
and its three official assertions, as in the MBPP paper; the reply's code
(fenced block or plain text) becomes the candidate.  Execution happens later in
restricted Docker containers (``remtp.mbpp_evaluator``), exactly like HumanEval.
Throughput and acceptance fields come from vLLM's metrics, as in the other
benchmarks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from remtp.benchmark import _display, _http_request, aggregate_records, chat_completion, fetch_metrics, metric_delta
from remtp.benchmark_checkpoint import append_benchmark_checkpoint, initialize_benchmark_output
from remtp.humaneval_benchmark import THINK_PATTERN, FENCE_PATTERN, benchmark_messages

TEST_SPLIT = range(11, 511)
PROMPT_TEMPLATE = """You are an expert Python programmer. Write a Python function for the task below.
Your code should pass these tests:

{tests}

Return only valid Python code. Do not include an explanation.

Task: {text}"""


def load_mbpp(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        for key in ("task_id", "text", "test_list"):
            if key not in row:
                raise ValueError(f"MBPP row without {key}: {row}")
    return rows


def test_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["task_id"]) in TEST_SPLIT]


def sample_mbpp(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    pool = test_split(rows)
    if samples <= 0 or samples > len(pool):
        raise ValueError(f"MBPP test split has {len(pool)} tasks; cannot sample {samples}")
    selected = pool if samples == len(pool) else random.Random(seed).sample(pool, samples)
    return sorted(selected, key=lambda row: int(row["task_id"]))


def build_prompt(row: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(tests="\n".join(row["test_list"]), text=row["text"].strip())


def extract_code(text: str) -> str:
    """Code of the reply: the longest fenced block if any, otherwise the text without fences."""
    cleaned = THINK_PATTERN.sub("", text).strip("\r\n")
    fences = FENCE_PATTERN.findall(cleaned)
    if fences:
        return max(fences, key=len).strip("\r\n")
    return cleaned.replace("```python", "").replace("```py", "").replace("```", "").strip("\r\n")


def build_source(row: dict[str, Any], candidate: str) -> str:
    """Setup code, the candidate, then the official assertions (challenge tests excluded)."""
    setup = str(row.get("test_setup_code") or "").strip()
    parts = [setup] if setup else []
    parts += [candidate.rstrip(), "\n".join(row["test_list"])]
    return "\n\n".join(parts) + "\n"


def write_summary(output_dir: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps({"config": config, "results": [summary]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    pass_at_1 = summary.get("pass_at_1")
    quality = "pending" if pass_at_1 is None else f"{100.0 * pass_at_1:.1f}%"
    lines = [
        "# ReMTP MBPP", "",
        f"- samples: `{config['samples']}` (test split, sample seed `{config['sample_seed']}`)",
        f"- temperature: `{config['temperature']}`, generation seed `{config['generation_seed']}`",
        f"- max output tokens: `{config['max_tokens']}`", "",
        "| method | samples | pass@1 | decode tok/s | e2e tok/s | mean acceptance length | truncation |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {config['run_name']} | {summary['samples']} | {quality} | {_display(summary['decode_tok_s'])} | "
        f"{_display(summary['e2e_output_tok_s'])} | {_display(summary['mean_acceptance_length'])} | "
        f"{100.0 * summary['truncation_rate']:.1f}% |",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MBPP solutions without executing them.")
    parser.add_argument("--data", type=Path, default=Path("data/regret_router/benchmarks/mbpp.jsonl"))
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=20261116)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--mtp-tokens", type=int, default=3)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="TencentBAC/FastMTP")
    parser.add_argument("--run-name", default="mbpp")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--empty-system-prompt", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        print(f"MBPP data not found: {data_path}\nRun python -m remtp.router_benchmark_download first.",
              file=sys.stderr)
        return 2
    base_url = args.base_url.rstrip("/")
    _http_request(f"{base_url}/health", timeout=args.timeout)
    selected = sample_mbpp(load_mbpp(data_path), args.samples, args.sample_seed)
    manifest = [{"task_id": row["task_id"],
                 "prompt_sha256": hashlib.sha256(build_prompt(row).encode("utf-8")).hexdigest()} for row in selected]
    records, checkpoint_path = initialize_benchmark_output(args.output_dir, manifest, resume=args.resume,
                                                           identity_key="task_id")
    if not args.skip_warmup:
        chat_completion(base_url, args.model,
                        benchmark_messages(build_prompt(selected[0]), empty_system_prompt=args.empty_system_prompt),
                        args.temperature, args.generation_seed, min(96, args.max_tokens), args.timeout)
    completed = len(records)
    print(f"\nTask mbpp: {len(selected)} samples (generation only)")
    for sample_index, row in enumerate(selected, start=1):
        if sample_index <= completed:
            continue
        request_seed = args.generation_seed + int(row["task_id"]) * 10
        messages = benchmark_messages(build_prompt(row), empty_system_prompt=args.empty_system_prompt)
        before = fetch_metrics(base_url, args.timeout)
        started = time.perf_counter()
        response = chat_completion(base_url, args.model, messages, args.temperature, request_seed,
                                   args.max_tokens, args.timeout)
        client_seconds = time.perf_counter() - started
        after = fetch_metrics(base_url, args.timeout)
        choice = response["choices"][0]
        raw_output = choice["message"]["content"] or ""
        usage = response.get("usage") or {}
        record = {
            "task": "mbpp", "task_id": row["task_id"], "sample_index": sample_index, "seed": request_seed,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "client_seconds": client_seconds, "finish_reason": choice.get("finish_reason"),
            "candidate": extract_code(raw_output), "raw_output": raw_output,
            "correct": None, "evaluation_status": "pending", "metrics": metric_delta(before, after),
        }
        records.append(record)
        append_benchmark_checkpoint(checkpoint_path, record)
        if sample_index == 1 or sample_index % args.progress_every == 0 or sample_index == len(selected):
            print(f"  [{sample_index:03d}/{len(selected):03d}] task={row['task_id']} tokens={record['output_tokens']}")
    summary = aggregate_records(records, "mbpp", len(selected))
    truncated = sum(record["finish_reason"] == "length" for record in records)
    summary.update(correct=None, accuracy=None, pass_at_1=None, evaluated=0,
                   evaluation_status="pending_docker_evaluation", truncated_outputs=truncated,
                   truncation_rate=truncated / len(records) if records else None)
    config = {
        "data": str(data_path), "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "samples": args.samples, "sample_seed": args.sample_seed, "temperature": args.temperature,
        "generation_seed": args.generation_seed, "max_tokens": args.max_tokens, "mtp_tokens": args.mtp_tokens,
        "base_url": base_url, "model": args.model, "run_name": args.run_name,
        "answer_metric": "MBPP pass@1 (three official assertions, one sample per task)",
        "execution_policy": "separate restricted Docker container per task",
        "system_prompt": "" if args.empty_system_prompt else "tokenizer_default",
    }
    with (args.output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_summary(args.output_dir, summary, config)
    print(f"\nGenerated solutions: {args.output_dir.resolve()}")
    print(f"decode={_display(summary['decode_tok_s'])} tok/s, e2e={_display(summary['e2e_output_tok_s'])} tok/s, "
          f"MAL={_display(summary['mean_acceptance_length'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
