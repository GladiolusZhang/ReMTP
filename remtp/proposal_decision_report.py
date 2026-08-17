"""Render a small Proposal-Calibrated MTP decision trace for humans.

The trace is deliberately a debugging artifact.  Capturing top-k candidates
and synchronizing tensors to write JSONL changes latency, so numbers produced
by a traced run must never be used as throughput measurements.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


DecodeToken = Callable[[int], str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL row {line_number} in {path}: {exc}"
                ) from exc
    return rows


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "\\n")


def _token(token_id: int, decode: DecodeToken) -> str:
    text = decode(int(token_id))
    escaped = json.dumps(text, ensure_ascii=False)
    return f"{int(token_id)} `{_md_cell(escaped)}`"


def _candidate_list(
    ids: Iterable[int],
    probs: Iterable[float],
    decode: DecodeToken,
) -> str:
    return ", ".join(
        f"{_token(token_id, decode)} ({float(probability):.4f})"
        for token_id, probability in zip(ids, probs, strict=True)
    )


def _indent_block(text: str) -> list[str]:
    rows = text.splitlines() or [""]
    return [f"    {row}" for row in rows]


def _request_key(record: dict[str, Any]) -> str:
    if record.get("task") == "gsm8k":
        return f"GSM8K question {record.get('question_id', '?')}"
    return str(record.get("task_id", f"request {record.get('sample_index', '?')}"))


def _final_output(record: dict[str, Any]) -> str:
    if record.get("task") == "humaneval":
        return str(record.get("raw_output", ""))
    return str(record.get("output", ""))


def _problem_text(
    record: dict[str, Any],
    source_by_id: dict[str, dict[str, Any]],
) -> str:
    if record.get("task") == "gsm8k":
        source = source_by_id.get(str(record.get("question_id")), {})
        return str(source.get("question", "(question text unavailable)"))
    source = source_by_id.get(str(record.get("task_id")), {})
    return str(source.get("prompt", "(prompt unavailable)"))


def build_report(
    trace_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    decode: DecodeToken,
    *,
    max_rounds_per_request: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Build Markdown and a machine-readable request/round report."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[int(row.get("request_index", -1))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("round_index", 0)))

    lines = [
        "# MTP decision trace",
        "",
        "> Debug trace only: tensor synchronization and JSONL writes make its "
        "speed numbers invalid.",
        "",
        "`accepted` means the active sampled verifier accepted that draft "
        "token; `rejected` is the first rejected draft; later positions are "
        "`skipped` because their logits no longer correspond to the committed "
        "prefix. `recovery` is sampled from the active P-or-H residual; "
        "`bonus` is sampled from the target distribution after all drafts pass.",
        "For a relaxed ReMTP trace, `accepted/relaxed-only` means the same "
        "uniform random draw would fail strict verification but passed after "
        "the temporary TV allocation.",
        "",
        "## Request summary",
        "",
        "| request | rounds shown | rejected rounds | first-rejection heads | "
        "output tokens | finish |",
        "|---|---:|---:|---|---:|---|",
    ]
    enriched_requests: list[dict[str, Any]] = []
    for request_index, record in enumerate(request_rows):
        rounds = grouped.get(request_index, [])
        shown = (
            rounds[:max_rounds_per_request]
            if max_rounds_per_request > 0
            else rounds
        )
        rejected = [
            int(row["rejected_head"])
            for row in shown
            if row.get("rejected_head") is not None
        ]
        rejection_text = ", ".join(map(str, rejected)) or "none"
        lines.append(
            f"| {_md_cell(_request_key(record))} | {len(shown)} | "
            f"{len(rejected)} | {rejection_text} | "
            f"{record.get('output_tokens', '?')} | "
            f"{_md_cell(record.get('finish_reason', '?'))} |"
        )
        enriched_requests.append(
            {
                "request_index": request_index,
                "request": record,
                "rounds": shown,
                "rejected_rounds": len(rejected),
                "first_rejection_heads": rejected,
            }
        )

    for item in enriched_requests:
        request_index = int(item["request_index"])
        record = item["request"]
        rounds = item["rounds"]
        lines.extend(
            [
                "",
                f"## Request {request_index + 1}: {_request_key(record)}",
                "",
                "Input:",
                "",
                *_indent_block(_problem_text(record, source_by_id)),
            ]
        )
        if record.get("task") == "gsm8k":
            lines.extend(
                [
                    "",
                    f"Gold: `{record.get('gold_answer')}`; predicted: "
                    f"`{record.get('predicted_answer')}`; correct: "
                    f"`{record.get('correct')}`.",
                ]
            )
        if not rounds:
            lines.extend(["", "No speculative rounds were captured."])
        for local_round, row in enumerate(rounds, start=1):
            draft_ids = [int(value) for value in row.get("draft_tokens", [])]
            committed_ids = [
                int(value) for value in row.get("committed_tokens", [])
            ]
            accepted_drafts = int(row.get("accepted_drafts", 0))
            rejected_head = row.get("rejected_head")
            if "rejected_head" not in row:
                rejected_head = (
                    accepted_drafts + 1
                    if accepted_drafts < len(draft_ids)
                    else None
                )
            outcome = (
                f"all {len(draft_ids)} drafts accepted, then target bonus"
                if rejected_head is None
                else f"first rejection at head {rejected_head}, then recovery"
            )
            lines.extend(
                [
                    "",
                    f"### Round {local_round} "
                    f"(global {row.get('round_index', '?')})",
                    "",
                    f"- Outcome: {outcome}.",
                    "- Draft: "
                    + " → ".join(_token(token_id, decode) for token_id in draft_ids),
                    "- Commit: "
                    + " → ".join(
                        _token(token_id, decode) for token_id in committed_ids
                    ),
                    f"- Final committed token kind: `{row.get('anchor_kind', '?')}`.",
                    "",
                ]
            )
            decisions = row.get("decisions") or [
                (
                    "accepted"
                    if index < accepted_drafts
                    else "rejected"
                    if index == accepted_drafts
                    else "skipped"
                )
                for index in range(len(draft_ids))
            ]
            sources = row.get("acceptance_sources", [])
            p_y = row.get("p_y", [])
            q_y = row.get("q_y", [])
            strict = row.get("sampled_strict_acceptance") or row.get(
                "strict_acceptance", []
            )
            relaxed = row.get("relaxed_acceptance", [])
            allocated_tv = row.get("allocated_tv", [])
            has_relaxation = bool(relaxed or allocated_tv)
            if has_relaxation:
                lines.extend(
                    [
                        "| head | decision/source | draft y | P(y) | Q(y) | "
                        "strict alpha | TV | relaxed alpha | target top candidates | "
                        "proposal top candidates |",
                        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
                    ]
                )
            else:
                lines.extend(
                    [
                        "| head | decision | draft y | P(y) | Q-tilde(y) | "
                        "strict alpha | target top candidates | "
                        "proposal top candidates |",
                        "|---:|---|---|---:|---:|---:|---|---|",
                    ]
                )
            p_top_ids = row.get("p_top_ids", [])
            p_top_probs = row.get("p_top_probs", [])
            q_top_ids = row.get("q_top_ids", [])
            q_top_probs = row.get("q_top_probs", [])
            for head, token_id in enumerate(draft_ids):
                p_candidates = _candidate_list(
                    p_top_ids[head] if head < len(p_top_ids) else [],
                    p_top_probs[head] if head < len(p_top_probs) else [],
                    decode,
                )
                q_candidates = _candidate_list(
                    q_top_ids[head] if head < len(q_top_ids) else [],
                    q_top_probs[head] if head < len(q_top_probs) else [],
                    decode,
                )
                decision = decisions[head] if head < len(decisions) else "?"
                if head < len(sources) and sources[head] not in {
                    "",
                    decision,
                }:
                    decision = f"{decision}/{sources[head]}"
                if has_relaxation:
                    lines.append(
                        f"| {head + 1} | {_md_cell(decision)} | "
                        f"{_token(token_id, decode)} | "
                        f"{float(p_y[head]):.6f} | {float(q_y[head]):.6f} | "
                        f"{float(strict[head]):.6f} | "
                        f"{float(allocated_tv[head]):.6f} | "
                        f"{float(relaxed[head]):.6f} | {p_candidates} | "
                        f"{q_candidates} |"
                    )
                else:
                    lines.append(
                        f"| {head + 1} | {_md_cell(decision)} | "
                        f"{_token(token_id, decode)} | "
                        f"{float(p_y[head]):.6f} | {float(q_y[head]):.6f} | "
                        f"{float(strict[head]):.6f} | {p_candidates} | "
                        f"{q_candidates} |"
                    )

        lines.extend(
            [
                "",
                "Final generated output:",
                "",
                *_indent_block(_final_output(record)),
            ]
        )

    orphan_indices = sorted(set(grouped) - set(range(len(request_rows))))
    if orphan_indices:
        lines.extend(
            [
                "",
                "## Trace alignment warning",
                "",
                "Trace rows referred to request indices absent from "
                f"`requests.jsonl`: `{orphan_indices}`. Use `--skip-warmup` "
                "when collecting a trace.",
            ]
        )
    payload = {
        "trace_version": 2,
        "requests": enriched_requests,
        "orphan_request_indices": orphan_indices,
    }
    return "\n".join(lines) + "\n", payload


def _load_sources(dataset: str, path: Path) -> dict[str, dict[str, Any]]:
    if dataset == "gsm8k":
        from remtp.gsm8k_benchmark import load_gsm8k

        return {
            str(row["question_id"]): row for row in load_gsm8k(path)
        }
    from remtp.humaneval_benchmark import load_humaneval

    return {str(row["task_id"]): row for row in load_humaneval(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode compact MTP decision traces into one Markdown report."
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--max-rounds-per-request", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_rounds_per_request < 0:
        raise ValueError("max-rounds-per-request must be non-negative")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )

    def decode(token_id: int) -> str:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    markdown, payload = build_report(
        load_jsonl(args.trace),
        load_jsonl(args.requests),
        _load_sources(args.dataset, args.data),
        decode,
        max_rounds_per_request=args.max_rounds_per_request,
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Decision report: {args.output_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
