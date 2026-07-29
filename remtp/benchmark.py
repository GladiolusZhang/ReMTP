"""Small, reproducible Spec-Bench runner for a vLLM OpenAI server."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SPEC_BENCH_TASK_RANGES = {
    "multi_turn": range(81, 161),
    "translation": range(161, 241),
    "summarization": range(241, 321),
    "qa": range(321, 401),
    "math_reasoning": range(401, 481),
    "rag": range(481, 561),
}
DEFAULT_TASKS = ("translation", "summarization", "math_reasoning", "rag")

METRIC_NAMES = {
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:e2e_request_latency_seconds_sum",
}
METRIC_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def official_task(question_id: int) -> str:
    for task, question_ids in SPEC_BENCH_TASK_RANGES.items():
        if question_id in question_ids:
            return task
    raise ValueError(f"question_id {question_id} is outside Spec-Bench ranges")


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                question_id = int(row["question_id"])
                turns = row["turns"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSONL row {line_number}: {exc}") from exc
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"row {line_number} has no turns")
            row["_spec_bench_task"] = official_task(question_id)
            questions.append(row)
    return questions


def sample_questions(
    questions: list[dict[str, Any]],
    tasks: list[str],
    samples_per_task: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        grouped[question["_spec_bench_task"]].append(question)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for task in tasks:
        candidates = grouped.get(task, [])
        if len(candidates) < samples_per_task:
            raise ValueError(
                f"task {task!r} has {len(candidates)} rows; "
                f"need {samples_per_task}"
            )
        task_rows = rng.sample(candidates, samples_per_task)
        selected.extend(
            sorted(task_rows, key=lambda row: int(row["question_id"]))
        )
    return selected


def _http_request(
    url: str,
    data: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> tuple[bytes, dict[str, str]]:
    body = None
    headers: dict[str, str] = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc


def parse_metrics(text: str) -> dict[str, float]:
    values = {name: 0.0 for name in METRIC_NAMES}
    for line in text.splitlines():
        match = METRIC_PATTERN.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name in values:
            values[name] += float(match.group("value"))
    return values


def fetch_metrics(base_url: str, timeout: float) -> dict[str, float]:
    body, _ = _http_request(f"{base_url}/metrics", timeout=timeout)
    return parse_metrics(body.decode("utf-8"))


def metric_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    return {
        name: max(0.0, after.get(name, 0.0) - before.get(name, 0.0))
        for name in METRIC_NAMES
    }


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    seed: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    body, _ = _http_request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        timeout=timeout,
    )
    return json.loads(body)


def safe_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def aggregate_records(
    records: list[dict[str, Any]],
    task: str,
    sample_count: int,
) -> dict[str, Any]:
    output_tokens = sum(record["output_tokens"] for record in records)
    client_seconds = sum(record["client_seconds"] for record in records)
    server_output_tokens = sum(
        record["metrics"]["vllm:generation_tokens_total"] for record in records
    )
    decode_seconds = sum(
        record["metrics"]["vllm:request_decode_time_seconds_sum"]
        for record in records
    )
    prefill_seconds = sum(
        record["metrics"]["vllm:request_prefill_time_seconds_sum"]
        for record in records
    )
    draft_rounds = sum(
        record["metrics"]["vllm:spec_decode_num_drafts_total"]
        for record in records
    )
    draft_tokens = sum(
        record["metrics"]["vllm:spec_decode_num_draft_tokens_total"]
        for record in records
    )
    accepted_tokens = sum(
        record["metrics"]["vllm:spec_decode_num_accepted_tokens_total"]
        for record in records
    )

    return {
        "task": task,
        "samples": sample_count,
        "requests": len(records),
        "output_tokens": int(output_tokens),
        "server_output_tokens": int(server_output_tokens),
        "decode_seconds": decode_seconds,
        "prefill_seconds": prefill_seconds,
        "client_seconds": client_seconds,
        "decode_tok_s": safe_rate(server_output_tokens, decode_seconds),
        "e2e_output_tok_s": safe_rate(output_tokens, client_seconds),
        "mean_acceptance_length": (
            1.0 + accepted_tokens / draft_rounds
            if draft_rounds > 0
            else None
        ),
        "accepted_draft_tokens_per_round": safe_rate(
            accepted_tokens,
            draft_rounds,
        ),
        "draft_token_acceptance_rate": safe_rate(
            accepted_tokens,
            draft_tokens,
        ),
        "draft_rounds": int(draft_rounds),
        "draft_tokens": int(draft_tokens),
        "accepted_draft_tokens": int(accepted_tokens),
    }


def _display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"config": config, "results": summaries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fieldnames = list(summaries[0].keys())
    with (output_dir / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# ReMTP Spec-Bench subset",
        "",
        f"- temperature: `{config['temperature']}`",
        f"- MTP draft tokens per round: `{config['mtp_tokens']}`",
        f"- samples per task: `{config['samples_per_task']}`",
        f"- max output tokens: `{config['max_tokens']}`",
        "",
        "| task | samples | output tokens | decode tok/s | "
        "e2e output tok/s | mean acceptance length | draft acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        acceptance_rate = summary["draft_token_acceptance_rate"]
        rate_text = (
            f"{100.0 * acceptance_rate:.1f}%"
            if acceptance_rate is not None
            else "n/a"
        )
        lines.append(
            f"| {summary['task']} | {summary['samples']} | "
            f"{summary['output_tokens']} | "
            f"{_display(summary['decode_tok_s'])} | "
            f"{_display(summary['e2e_output_tok_s'])} | "
            f"{_display(summary['mean_acceptance_length'])} | "
            f"{rate_text} |"
        )
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark native vLLM MTP on a Spec-Bench subset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/spec_bench/question.jsonl"),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(SPEC_BENCH_TASK_RANGES),
        default=list(DEFAULT_TASKS),
    )
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=20260729)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--mtp-tokens", type=int, default=2)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--run-name", default="native_mtp")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        print(
            f"Spec-Bench data not found: {data_path}\n"
            "Place the official question.jsonl there, then rerun.",
            file=sys.stderr,
        )
        return 2
    if args.samples_per_task <= 0 or args.max_tokens <= 0:
        print("samples-per-task and max-tokens must be positive", file=sys.stderr)
        return 2
    if args.temperature <= 0:
        print("temperature must be > 0 for this stochastic benchmark", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    try:
        _http_request(f"{base_url}/health", timeout=args.timeout)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    questions = load_questions(data_path)
    selected = sample_questions(
        questions,
        args.tasks,
        args.samples_per_task,
        args.sample_seed,
    )

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
        output_dir = (
            Path("results")
            / f"spec_bench_{safe_run_name}_t{args.temperature:g}_{stamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = [
        {
            "question_id": int(row["question_id"]),
            "task": row["_spec_bench_task"],
            "turns": len(row["turns"]),
        }
        for row in selected
    ]
    (output_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.skip_warmup:
        warmup = selected[0]
        print(f"Warmup: question {warmup['question_id']} (excluded)")
        chat_completion(
            base_url,
            args.model,
            [{"role": "user", "content": str(warmup["turns"][0])}],
            args.temperature,
            args.generation_seed,
            min(16, args.max_tokens),
            args.timeout,
        )

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task in args.tasks:
        task_rows = [
            row for row in selected if row["_spec_bench_task"] == task
        ]
        task_records: list[dict[str, Any]] = []
        print(f"\nTask {task}: {len(task_rows)} samples")
        for sample_index, row in enumerate(task_rows, start=1):
            messages: list[dict[str, str]] = []
            for turn_index, turn in enumerate(row["turns"]):
                messages.append({"role": "user", "content": str(turn)})
                request_seed = (
                    args.generation_seed
                    + int(row["question_id"]) * 10
                    + turn_index
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
                assistant_text = choice["message"]["content"] or ""
                usage = response.get("usage") or {}
                record = {
                    "task": task,
                    "question_id": int(row["question_id"]),
                    "sample_index": sample_index,
                    "turn_index": turn_index,
                    "seed": request_seed,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "output_tokens": int(usage.get("completion_tokens", 0)),
                    "client_seconds": client_seconds,
                    "finish_reason": choice.get("finish_reason"),
                    "output": assistant_text,
                    "metrics": metric_delta(before, after),
                }
                task_records.append(record)
                all_records.append(record)
                messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
            print(
                f"  [{sample_index:02d}/{len(task_rows):02d}] "
                f"question={row['question_id']} "
                f"output_tokens={sum(r['output_tokens'] for r in task_records if r['question_id'] == int(row['question_id']))}"
            )

        summary = aggregate_records(task_records, task, len(task_rows))
        summaries.append(summary)
        print(
            f"  decode={_display(summary['decode_tok_s'])} tok/s, "
            f"e2e={_display(summary['e2e_output_tok_s'])} tok/s, "
            f"mean_acceptance_length="
            f"{_display(summary['mean_acceptance_length'])}"
        )

    overall = aggregate_records(
        all_records,
        "overall",
        len(selected),
    )
    summaries.append(overall)
    config = {
        "data": str(data_path),
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "sample_seed": args.sample_seed,
        "temperature": args.temperature,
        "generation_seed": args.generation_seed,
        "max_tokens": args.max_tokens,
        "mtp_tokens": args.mtp_tokens,
        "base_url": base_url,
        "model": args.model,
        "run_name": args.run_name,
    }
    with (output_dir / "requests.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_summary(output_dir, summaries, config)

    print(f"\nResults: {output_dir.resolve()}")
    print(
        f"Overall decode={_display(overall['decode_tok_s'])} tok/s, "
        f"e2e={_display(overall['e2e_output_tok_s'])} tok/s, "
        f"mean acceptance length="
        f"{_display(overall['mean_acceptance_length'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
