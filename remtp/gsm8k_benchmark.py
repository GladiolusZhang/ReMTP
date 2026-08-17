"""Reproducible GSM8K subset benchmark for a vLLM OpenAI server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

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


NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
HASH_ANSWER_PATTERN = re.compile(
    rf"####\s*({NUMBER_PATTERN.pattern})"
)
BOXED_ANSWER_PATTERN = re.compile(
    rf"\\boxed\{{\s*({NUMBER_PATTERN.pattern})\s*\}}"
)

PROMPT_TEMPLATE = """Solve this grade-school math problem with a concise,
non-repetitive derivation of no more than 120 words.
At the very end, write the final numeric answer on a separate line exactly as:
#### <number>

Problem:
{question}"""


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


def normalize_number(value: str) -> str | None:
    """Normalize an extracted decimal number for exact comparison."""
    cleaned = value.replace(",", "").strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def extract_gold_answer(answer: str) -> str:
    matches = HASH_ANSWER_PATTERN.findall(answer)
    if not matches:
        raise ValueError("GSM8K gold answer has no '#### <number>' suffix")
    normalized = normalize_number(matches[-1])
    if normalized is None:
        raise ValueError(f"invalid GSM8K gold answer: {matches[-1]!r}")
    return normalized


def extract_model_answer(text: str) -> tuple[str | None, str | None]:
    """Extract final numeric answer, preferring the requested GSM8K format."""
    hash_matches = HASH_ANSWER_PATTERN.findall(text)
    if hash_matches:
        return normalize_number(hash_matches[-1]), "hash"

    boxed_matches = BOXED_ANSWER_PATTERN.findall(text)
    if boxed_matches:
        return normalize_number(boxed_matches[-1]), "boxed"

    number_matches = NUMBER_PATTERN.findall(text)
    if number_matches:
        return normalize_number(number_matches[-1]), "last_number"
    return None, None


def load_gsm8k(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                question = str(row["question"])
                answer = str(row["answer"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid GSM8K row {index}: {exc}") from exc
            rows.append(
                {
                    "question_id": index,
                    "question": question,
                    "answer": answer,
                    "gold_answer": extract_gold_answer(answer),
                }
            )
    return rows


def sample_gsm8k(
    rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples > len(rows):
        raise ValueError(
            f"GSM8K has {len(rows)} rows; cannot sample {samples}"
        )
    rng = random.Random(seed)
    return sorted(
        rng.sample(rows, samples),
        key=lambda row: int(row["question_id"]),
    )


def add_accuracy(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    correct = sum(bool(record["correct"]) for record in records)
    parsed = sum(record["predicted_answer"] is not None for record in records)
    format_compliant = sum(
        record["answer_source"] == "hash" for record in records
    )
    truncated = sum(
        record["finish_reason"] == "length" for record in records
    )
    summary.update(
        {
            "correct": correct,
            "accuracy": correct / len(records) if records else None,
            "parsed_answers": parsed,
            "parse_rate": parsed / len(records) if records else None,
            "format_compliant_answers": format_compliant,
            "format_compliance_rate": (
                format_compliant / len(records) if records else None
            ),
            "truncated_outputs": truncated,
            "truncation_rate": truncated / len(records) if records else None,
        }
    )
    return summary


def write_gsm8k_summary(
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
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    accuracy = summary["accuracy"]
    acceptance_rate = summary["draft_token_acceptance_rate"]
    acceptance_text = (
        "n/a"
        if acceptance_rate is None
        else f"{100.0 * acceptance_rate:.1f}%"
    )
    parse_rate = summary["parse_rate"]
    format_rate = summary["format_compliance_rate"]
    truncation_rate = summary["truncation_rate"]
    lines = [
        "# ReMTP GSM8K subset",
        "",
        f"- samples: `{config['samples']}`",
        f"- sample seed: `{config['sample_seed']}`",
        f"- temperature: `{config['temperature']}`",
        f"- generation seed: `{config['generation_seed']}`",
        f"- MTP draft tokens per round: `{config['mtp_tokens']}`",
        f"- max output tokens: `{config['max_tokens']}`",
        "",
        "| method | samples | accuracy | decode tok/s | e2e tok/s | "
        "mean acceptance length | draft acceptance | parse rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {config['run_name']} | {summary['samples']} | "
        f"{100.0 * accuracy:.1f}% | "
        f"{_display(summary['decode_tok_s'])} | "
        f"{_display(summary['e2e_output_tok_s'])} | "
        f"{_display(summary['mean_acceptance_length'])} | "
        f"{acceptance_text} | "
        f"{100.0 * parse_rate:.1f}% |",
        "",
        f"Requested `#### <number>` format compliance: "
        f"{100.0 * format_rate:.1f}%.",
        f"Length-truncated outputs: {summary['truncated_outputs']} "
        f"({100.0 * truncation_rate:.1f}%).",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MTP variants on a fixed GSM8K test subset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/gsm8k/test.jsonl"),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=20260730)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--mtp-tokens", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--run-name", default="native_mtp")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print one progress row every N samples",
    )
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
            f"GSM8K test data not found: {data_path}\n"
            "Download the official test.jsonl there, then rerun.",
            file=sys.stderr,
        )
        return 2
    if (
        args.samples <= 0
        or args.max_tokens <= 0
        or args.progress_every <= 0
    ):
        print(
            "samples, max-tokens, and progress-every must be positive",
            file=sys.stderr,
        )
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

    rows = load_gsm8k(data_path)
    selected = sample_gsm8k(rows, args.samples, args.sample_seed)

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
        output_dir = (
            Path("results")
            / f"gsm8k_{safe_run_name}_n{args.samples}"
            f"_t{args.temperature:g}_{stamp}"
        )
    manifest = [
        {
            "question_id": int(row["question_id"]),
            "question_sha256": hashlib.sha256(
                row["question"].encode("utf-8")
            ).hexdigest(),
            "gold_answer": row["gold_answer"],
        }
        for row in selected
    ]
    records, checkpoint_path = initialize_benchmark_output(
        output_dir,
        manifest,
        resume=args.resume,
        identity_key="question_id",
    )

    if not args.skip_warmup:
        warmup = selected[0]
        print(f"Warmup: question {warmup['question_id']} (excluded)")
        warmup_content = PROMPT_TEMPLATE.format(question=warmup["question"])
        chat_completion(
            base_url,
            args.model,
            benchmark_messages(
                warmup_content,
                empty_system_prompt=args.empty_system_prompt,
            ),
            args.temperature,
            args.generation_seed,
            min(64, args.max_tokens),
            args.timeout,
        )

    completed = len(records)
    if completed:
        print(f"Resuming GSM8K from sample {completed + 1}/{len(selected)}")
    print(f"\nTask gsm8k: {len(selected)} samples")
    for sample_index, row in enumerate(selected, start=1):
        if sample_index <= completed:
            continue
        request_seed = args.generation_seed + int(row["question_id"]) * 10
        messages = benchmark_messages(
            PROMPT_TEMPLATE.format(question=row["question"]),
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
        assistant_text = choice["message"]["content"] or ""
        usage = response.get("usage") or {}
        predicted, answer_source = extract_model_answer(assistant_text)
        correct = predicted == row["gold_answer"]
        record = {
            "task": "gsm8k",
            "question_id": int(row["question_id"]),
            "sample_index": sample_index,
            "seed": request_seed,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "client_seconds": client_seconds,
            "finish_reason": choice.get("finish_reason"),
            "gold_answer": row["gold_answer"],
            "predicted_answer": predicted,
            "answer_source": answer_source,
            "correct": correct,
            "output": assistant_text,
            "metrics": metric_delta(before, after),
        }
        records.append(record)
        append_benchmark_checkpoint(checkpoint_path, record)
        status = "✓" if correct else "✗"
        if (
            sample_index == 1
            or sample_index % args.progress_every == 0
            or sample_index == len(selected)
        ):
            print(
                f"  [{sample_index:03d}/{len(selected):03d}] "
                f"question={row['question_id']} "
                f"output_tokens={record['output_tokens']} "
                f"answer={predicted!s} gold={row['gold_answer']} {status}"
            )

    summary = add_accuracy(
        aggregate_records(records, "gsm8k", len(selected)),
        records,
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
        "answer_metric": "normalized_numeric_exact_match",
        "answer_extraction": "####, then boxed, then last numeric token",
        "system_prompt": "" if args.empty_system_prompt else "tokenizer_default",
    }
    with (output_dir / "requests.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_gsm8k_summary(output_dir, summary, config)

    print(f"\nResults: {output_dir.resolve()}")
    print(
        f"Overall accuracy={100.0 * summary['accuracy']:.1f}%, "
        f"decode={_display(summary['decode_tok_s'])} tok/s, "
        f"e2e={_display(summary['e2e_output_tok_s'])} tok/s, "
        f"mean acceptance length="
        f"{_display(summary['mean_acceptance_length'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
