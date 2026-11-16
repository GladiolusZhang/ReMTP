"""Evaluate MBPP generations in restricted Docker containers (2026-11-16).

Same isolation as ``humaneval_evaluator``: one disposable container per task,
no network, memory/CPU limits and a host timeout.  The executed module is the
task's setup code, the candidate and the three official assertions.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from collections import Counter
from pathlib import Path

from remtp.humaneval_evaluator import docker_available, docker_image_id, evaluate_source_docker
from remtp.mbpp_benchmark import build_source, load_mbpp, write_summary


def evaluate_run(run_dir: Path, data_path: Path, *, image: str, timeout: float, memory: str, cpus: str,
                 workers: int) -> dict:
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config, summary = dict(payload["config"]), dict(payload["results"][0])
    records = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    tasks = {int(row["task_id"]): row for row in load_mbpp(data_path)}
    sources = [build_source(tasks[int(record["task_id"])], record.get("candidate") or "") for record in records]
    results: list[dict | None] = [None] * len(records)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mbpp-docker") as pool:
        pending = {pool.submit(evaluate_source_docker, source, image=image, timeout=timeout, memory=memory,
                               cpus=cpus): index for index, source in enumerate(sources)}
        for future in concurrent.futures.as_completed(pending):
            results[pending[future]] = future.result()
    counts: Counter[str] = Counter()
    evaluations = []
    for record, result in zip(records, results):
        status = result["status"]
        counts[status] += 1
        record["correct"] = status == "passed"
        record["evaluation_status"] = status
        evaluations.append({"task_id": record["task_id"], "correct": record["correct"], "status": status,
                            "detail": result["detail"]})
    correct = counts["passed"]
    summary.update(correct=correct, accuracy=correct / len(records) if records else None,
                   pass_at_1=correct / len(records) if records else None, evaluated=len(records),
                   evaluation_status="complete", evaluation_counts=dict(sorted(counts.items())))
    config.update(evaluation_image=image, evaluation_image_id=docker_image_id(image),
                  evaluation_timeout_seconds=timeout, evaluation_memory=memory, evaluation_cpus=cpus)
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (run_dir / "evaluation.jsonl").open("w", encoding="utf-8") as handle:
        for evaluation in evaluations:
            handle.write(json.dumps(evaluation, ensure_ascii=False) + "\n")
    write_summary(run_dir, summary, config)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/regret_router/benchmarks/mbpp.jsonl"))
    parser.add_argument("--image", default="python:3-slim")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--memory", default="512m")
    parser.add_argument("--cpus", default="1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not docker_available() or docker_image_id(args.image) is None:
        print(f"Docker or image {args.image} unavailable", file=sys.stderr)
        return 2
    summary = evaluate_run(args.run_dir, args.data, image=args.image, timeout=args.timeout, memory=args.memory,
                           cpus=args.cpus, workers=args.workers)
    print(f"pass@1={100.0 * summary['pass_at_1']:.1f}% ({summary['correct']}/{summary['evaluated']}) "
          f"outcomes={summary['evaluation_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
