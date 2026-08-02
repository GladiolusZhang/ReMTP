"""Evaluate generated HumanEval code in restricted Docker containers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from remtp.humaneval_benchmark import (
    load_humaneval,
    write_humaneval_summary,
)


def build_candidate_source(task: dict[str, Any], record: dict[str, Any]) -> str:
    """Construct a complete module containing solution and official tests."""
    prompt = str(task["prompt"])
    candidate = str(record.get("candidate") or "")
    entry_point = str(task["entry_point"])
    if record.get("candidate_mode") == "full_function":
        marker = prompt.find(f"def {entry_point}")
        preamble = prompt[:marker] if marker >= 0 else ""
        solution = preamble + candidate
    else:
        solution = prompt + candidate
    return (
        solution.rstrip()
        + "\n\n"
        + str(task["test"]).rstrip()
        + f"\n\ncheck({entry_point})\n"
    )


def docker_available() -> bool:
    result = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def docker_image_id(image: str) -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _docker_command(
    *,
    name: str,
    image: str,
    runner_path: Path,
    job_path: Path,
    memory: str,
    cpus: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        memory,
        "--cpus",
        cpus,
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--ulimit",
        "nofile=64:64",
        "--ulimit",
        "nproc=64:64",
        "-v",
        f"{runner_path.resolve()}:/runner.py:ro",
        "-v",
        f"{job_path.resolve()}:/job.json:ro",
        image,
        "python",
        "-I",
        "-B",
        "/runner.py",
        "/job.json",
    ]


def evaluate_source_docker(
    source: str,
    *,
    image: str,
    timeout: float,
    memory: str,
    cpus: str,
) -> dict[str, str]:
    """Run one candidate in a named container and enforce a host timeout."""
    runner_path = Path(__file__).with_name("humaneval_container_runner.py")
    container_name = f"remtp-humaneval-{uuid.uuid4().hex[:16]}"
    with tempfile.TemporaryDirectory(prefix="remtp-humaneval-") as tmp:
        job_path = Path(tmp) / "job.json"
        job_path.write_text(
            json.dumps({"source": source}, ensure_ascii=False),
            encoding="utf-8",
        )
        command = _docker_command(
            name=container_name,
            image=image,
            runner_path=runner_path,
            job_path=job_path,
            memory=memory,
            cpus=cpus,
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "kill", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            process.communicate()
            return {"status": "timeout", "detail": f"> {timeout:g}s"}
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    lines = [line for line in stdout.splitlines() if line.strip()]
    if process.returncode != 0 or not lines:
        detail = (stderr or stdout or f"exit code {process.returncode}").strip()
        return {"status": "container_error", "detail": detail[-1000:]}
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {
            "status": "container_error",
            "detail": (stderr + "\n" + stdout)[-1000:].strip(),
        }
    return {
        "status": str(result.get("status", "runner_error")),
        "detail": str(result.get("detail", ""))[:1000],
    }


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one result in {run_dir / 'summary.json'}")
    records: list[dict[str, Any]] = []
    with (run_dir / "requests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return dict(payload["config"]), dict(results[0]), records


def evaluate_run(
    run_dir: Path,
    data_path: Path,
    *,
    image: str,
    timeout: float,
    memory: str,
    cpus: str,
    progress_every: int,
) -> dict[str, Any]:
    config, summary, records = _load_run(run_dir)
    tasks = {row["task_id"]: row for row in load_humaneval(data_path)}
    missing = [record["task_id"] for record in records if record["task_id"] not in tasks]
    if missing:
        raise ValueError(f"tasks missing from HumanEval data: {missing[:3]}")

    counts: Counter[str] = Counter()
    evaluations: list[dict[str, Any]] = []
    print(f"Evaluating {len(records)} candidates in restricted Docker containers")
    for index, record in enumerate(records, start=1):
        task = tasks[record["task_id"]]
        result = evaluate_source_docker(
            build_candidate_source(task, record),
            image=image,
            timeout=timeout,
            memory=memory,
            cpus=cpus,
        )
        status = result["status"]
        correct = status == "passed"
        counts[status] += 1
        record["correct"] = correct
        record["evaluation_status"] = status
        evaluation = {
            "task_id": record["task_id"],
            "correct": correct,
            "status": status,
            "detail": result["detail"],
        }
        evaluations.append(evaluation)
        if index == 1 or index % progress_every == 0 or index == len(records):
            marker = "✓" if correct else "✗"
            print(
                f"  [{index:03d}/{len(records):03d}] "
                f"task={record['task_id']} {status} {marker}"
            )

    correct_count = counts["passed"]
    summary.update(
        {
            "correct": correct_count,
            "accuracy": correct_count / len(records) if records else None,
            "pass_at_1": correct_count / len(records) if records else None,
            "evaluated": len(records),
            "evaluation_status": "complete",
            "evaluation_counts": dict(sorted(counts.items())),
        }
    )
    config.update(
        {
            "evaluation_image": image,
            "evaluation_image_id": docker_image_id(image),
            "evaluation_timeout_seconds": timeout,
            "evaluation_memory": memory,
            "evaluation_cpus": cpus,
        }
    )
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (run_dir / "evaluation.jsonl").open("w", encoding="utf-8") as handle:
        for evaluation in evaluations:
            handle.write(json.dumps(evaluation, ensure_ascii=False) + "\n")
    write_humaneval_summary(run_dir, summary, config)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HumanEval generations in restricted Docker containers."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/humaneval/HumanEval.jsonl.gz"),
    )
    parser.add_argument("--image", default="python:3-slim")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--memory", default="512m")
    parser.add_argument("--cpus", default="1.0")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or args.progress_every <= 0:
        print("timeout and progress-every must be positive", file=sys.stderr)
        return 2
    if not args.run_dir.is_dir():
        print(f"run directory not found: {args.run_dir}", file=sys.stderr)
        return 2
    if not args.data.is_file():
        print(f"HumanEval data not found: {args.data}", file=sys.stderr)
        return 2
    if not docker_available():
        print("Docker daemon is unavailable; refusing unsafe local execution.", file=sys.stderr)
        return 2
    if docker_image_id(args.image) is None:
        print(f"Docker image is not installed: {args.image}", file=sys.stderr)
        return 2
    summary = evaluate_run(
        args.run_dir,
        args.data.resolve(),
        image=args.image,
        timeout=args.timeout,
        memory=args.memory,
        cpus=args.cpus,
        progress_every=args.progress_every,
    )
    print(
        f"pass@1={100.0 * summary['pass_at_1']:.1f}% "
        f"({summary['correct']}/{summary['evaluated']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
