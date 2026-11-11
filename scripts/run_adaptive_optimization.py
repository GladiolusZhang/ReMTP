#!/usr/bin/env python3
"""Finite, resumable Jetson optimization checks with a locked code/protocol.

Profiling and short performance runs never contribute to quality metrics.
Calibration is GSM8K train with full 2048 output tokens and two fixed seeds.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remtp.benchmark import aggregate_records, chat_completion, fetch_metrics, metric_delta
from remtp.gsm8k_benchmark import (
    PROMPT_TEMPLATE, benchmark_messages, extract_model_answer, load_gsm8k,
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class Controller:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.server = None
        self.base_url = f"http://127.0.0.1:{args.port}"
        files = sorted([*ROOT.glob("remtp/*.py"), *ROOT.glob("scripts/*.sh"),
                        *ROOT.glob("configs/*"),
                        Path(__file__).resolve()])
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        rows = load_gsm8k(ROOT / "data/gsm8k/train.jsonl")
        self.performance_rows = rows[:2]
        self.calibration_rows = sorted(random.Random(20261001).sample(rows[2:], 24),
                                       key=lambda row: row["question_id"])
        protocol = {
            "source_sha256": digest.hexdigest(), "source_root": str(ROOT),
            "train_sha256": hashlib.sha256((ROOT / "data/gsm8k/train.jsonl").read_bytes()).hexdigest(),
            "performance_ids": [r["question_id"] for r in self.performance_rows],
            "calibration_ids": [r["question_id"] for r in self.calibration_rows],
            "model": "TencentBAC/FastMTP", "target": "FP4", "draft": "BF16",
            "temperature": 0.6, "max_model_len": 4096, "kv_cache": "1024M",
            "quality_max_tokens": 2048, "performance_max_tokens": 128,
            "quality_seeds": [42, 43], "tree_nodes": 9,
            "candidate": {"prune_anchor_descendants": False,
                          "anchor_only_proposal": args.candidate == "anchor",
                          "max_depth": args.depth, "max_nodes": args.nodes,
                          "max_children": 3,
                          "head_input_rows": 3 if args.candidate == "anchor" else None},
            "baseline": {"prune_anchor_descendants": False, "exact_topk_candidates": False,
                         "max_depth": 3, "max_nodes": 9},
            "cactus_depth": args.depth,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "vllm", "bitsandbytes", "transformers")},
            "model_config_sha256": hashlib.sha256((ROOT / "models/FastMTP/config.json").read_bytes()).hexdigest(),
            "audit": False, "profile_times": "inclusive; synchronized; not throughput",
        }
        protocol["candidate"]["exact_topk_candidates"] = True
        path = self.root / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("Code/protocol changed: use a new run directory")
        write_json(path, protocol)

    def status(self, phase, variant="", completed=0, total=0):
        write_json(self.root / "status.json", {
            "phase": phase, "variant": variant, "completed": completed,
            "total": total, "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "controller_pid": os.getpid(),
        })
        print(f"[{phase}] {variant} {completed}/{total}", flush=True)

    def report(self):
        protocol = json.loads((self.root / "protocol.json").read_text())
        lines = ["# Adaptive RAVEN 优化验证报告", "", "## 协议", "",
                 "FastMTP FP4 Target / BF16 草稿层；temperature=0.6；空 system prompt；",
                 "关闭可见思考；上下文 4096；KV 1 GiB；全部组关闭审计。",
                 f"候选设置：`{json.dumps(protocol['candidate'], ensure_ascii=False)}`。", "",
                 "代码、参数及包版本见 `protocol.json`。profiling 使用同步计时，嵌套组件",
                 "不能重复相加，且不会被计入纯测速。", ""]
        smoke = self.root / "smoke_gate.json"
        if smoke.exists():
            x = json.loads(smoke.read_text())
            lines += ["## 固定短请求测速", "", "| 方法 | E2E token/s |",
                      "| --- | ---: |",
                      f"| 优化前 Adaptive D3（两次合并） | {x['baseline_tok_s']:.4f} |",
                      f"| 优化后 Adaptive D{protocol['candidate']['max_depth']} | {x['candidate_tok_s']:.4f} |",
                      f"| Cactus D{protocol['cactus_depth']} | {x['cactus_tok_s']:.4f} |", "",
                      f"相对优化前吞吐变化 {x['speedup']:+.2%}；性能门槛{'通过' if x['passed'] else '未通过'}。",
                      f"相对 Cactus 吞吐变化 {x['candidate_tok_s']/x['cactus_tok_s']-1:+.2%}。",
                      "两条固定 train 提示、最多 128 token，仅用于筛选性能，不能用于质量结论。", ""]
            for variant in ("baseline", "candidate"):
                path = self.root / "profile" / variant / "components.json"
                if path.exists():
                    components = json.loads(path.read_text())
                    lines += [f"### {variant} 组件计时", "", "| 组件 | 平均 ms/调用 | 调用数 |",
                              "| --- | ---: | ---: |"]
                    lines += [f"| {key} | {value['mean_seconds']*1000:.2f} | {value['calls']} |"
                              for key, value in components.items()]
                    lines.append("")
        quality = self.root / "quality_gate.json"
        if quality.exists():
            x = json.loads(quality.read_text())
            lines += ["## 完整质量校准", "", "24 道独立 train 题，每道使用 42/43 两个生成种子；",
                      "每次最多输出 2048 token。每种方法合计 48 个请求，同题跨种子具有相关性。", "",
                      "| 指标 | 优化前 | 优化后 |", "| --- | ---: | ---: |"]
            labels = {"correct": "原评测判对", "stopped_correct": "正常结束且判对",
                      "truncated": "达输出上限", "output_tokens": "输出 token", "client_seconds": "请求总秒数"}
            lines += [f"| {labels[key]} | {value['baseline']} | {value['candidate']} |"
                      for key, value in x["totals"].items()]
            lines += ["", f"质量门槛{'通过' if x['passed'] else '未通过'}。",
                      "门槛同时要求总体和每个种子的原评测正确数、正常结束且判对数均不下降。",
                      f"正确性或结束方式发生配对变动的案例共 {len(x['paired_changes'])} 个；详见 `quality_gate.json`。",
                      "通过校准也不构成独立正式测试证据；仍需分析配对案例，再决定采用和正式复测。", ""]
        else:
            lines += ["## 质量状态", "", "完整质量校准尚未结束，当前没有新的正确率结论。", ""]
        lines += ["## 分析边界", "", "短请求的吞吐变化不等于整题耗时同比下降；",
                  "不同输出长度、接受长度和截断情况需结合完整校准解读。",
                  "最终论文比较仍需 Native、Cactus、SpecCascade 以及优化前后方法的",
                  "独立 test 题、多种子对照。未完成的实验不能作为成功结果。", ""]
        (self.root / "report.md").write_text("\n".join(lines))

    def stop_server(self):
        if self.server is None:
            return
        process = self.server
        for sig, timeout in [(signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)]:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                continue
        # Kill surviving descendants even if the original launcher has exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.server = None

    def start_server(self, variant, directory, profile=False):
        self.stop_server()
        try:
            urllib.request.urlopen(self.base_url + "/health", timeout=2)
        except Exception:
            pass
        else:
            raise ValueError("Port occupied; refusing to stop an unrelated server")
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("REMTP_", "VLLM_"))}
        env.update({
            "METHOD": "cactus" if variant == "cactus" else "dynamic_tree",
            "MAX_MODEL_LEN": "4096", "KV_CACHE_MEMORY_BYTES": "1024M",
            "PORT": str(self.args.port), "REMTP_TREE_AUDIT_JSONL": "",
            "MTP_TOKENS": str(self.args.depth) if variant == "cactus" else "3",
            "QUANTIZATION": "bitsandbytes", "LOAD_FORMAT": "bitsandbytes",
            "REMTP_TREE_TRACE": "0", "REMTP_DYNAMIC_TREE_ADAPTIVE_RELAXATION": "1",
            "REMTP_DYNAMIC_TREE_PRUNE_ANCHOR_DESCENDANTS": "0",
            "REMTP_DYNAMIC_TREE_EXACT_TOPK": "1" if variant == "candidate" else "0",
            "REMTP_DYNAMIC_TREE_ANCHOR_ONLY_PROPOSAL": "1" if variant == "candidate" and self.args.candidate == "anchor" else "0",
            "REMTP_DYNAMIC_TREE_MAX_DEPTH": str(self.args.depth) if variant == "candidate" else "3",
            "REMTP_DYNAMIC_TREE_MAX_NODES": str(self.args.nodes) if variant == "candidate" else "9",
            "REMTP_RUNTIME_PROFILE_JSONL": str(directory / "profile.jsonl") if profile else "",
            "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1",
        })
        log = (directory / "server.log").open("a")
        self.server = subprocess.Popen(
            [str(ROOT / "scripts/serve_fastmtp_jetson.sh")], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()
        for _ in range(180):
            if self.server.poll() is not None:
                raise RuntimeError(f"Server exited; inspect {directory / 'server.log'}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("Server health timed out")
        # Identical warmup, excluded from all metrics.
        chat_completion(self.base_url, "TencentBAC/FastMTP",
                        benchmark_messages("Calculate 2+2.", empty_system_prompt=True),
                        0.6, 1234, 8, 1800)
        if profile:
            (directory / "profile.jsonl").write_text("")

    def run_requests(self, phase, name, variant, requests, limit, profile=False):
        directory = self.root / phase / name
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "requests.jsonl"
        records = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
        expected = [(row["question_id"], seed + row["question_id"] * 10) for row, seed in requests]
        if [(r["question_id"], r["seed"]) for r in records] != expected[:len(records)]:
            raise ValueError("Checkpoint order mismatch")
        if len(records) < len(requests):
            self.status(phase, name, len(records), len(requests))
            self.start_server(variant, directory, profile)
            for row, seed in requests[len(records):]:
                seed += row["question_id"] * 10
                before = fetch_metrics(self.base_url, 30)
                started = time.perf_counter()
                response = chat_completion(
                    self.base_url, "TencentBAC/FastMTP",
                    benchmark_messages(PROMPT_TEMPLATE.format(question=row["question"]), empty_system_prompt=True),
                    0.6, seed, limit, 1800,
                )
                elapsed = time.perf_counter() - started
                after = fetch_metrics(self.base_url, 30)
                choice = response["choices"][0]
                text = choice["message"]["content"] or ""
                predicted, source = extract_model_answer(text)
                record = {
                    "question_id": row["question_id"], "seed": seed,
                    "gold_answer": row["gold_answer"], "predicted_answer": predicted,
                    "answer_source": source, "correct": predicted == row["gold_answer"],
                    "finish_reason": choice["finish_reason"], "output": text,
                    "output_tokens": response["usage"]["completion_tokens"],
                    "client_seconds": elapsed, "metrics": metric_delta(before, after),
                }
                with output.open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                records.append(record)
                self.status(phase, name, len(records), len(requests))
            self.stop_server()
        summary = aggregate_records(records, "gsm8k_train", len(requests))
        summary.update({
            "correct": sum(r["correct"] for r in records),
            "stopped_correct": sum(r["correct"] and r["finish_reason"] == "stop" for r in records),
            "truncated": sum(r["finish_reason"] == "length" for r in records),
        })
        write_json(directory / "summary.json", summary)
        if profile:
            grouped = collections.defaultdict(list)
            for line in (directory / "profile.jsonl").read_text().splitlines():
                record = json.loads(line)
                grouped[record["component"]].append(record["seconds"])
            write_json(directory / "components.json", {
                key: {"calls": len(values), "total_seconds": sum(values),
                      "mean_seconds": sum(values) / len(values)}
                for key, values in grouped.items()
            })
        return summary

    def run(self):
        smoke_path = self.root / "smoke_gate.json"
        if self.args.phase in ("smoke", "all"):
            for variant in ("baseline", "candidate"):
                self.run_requests("profile", variant, variant,
                                  [(self.performance_rows[0], 42)], 64, True)
            summaries = {}
            for name, variant in (("baseline", "baseline"), ("candidate", "candidate"),
                                  ("cactus", "cactus"), ("baseline_repeat", "baseline")):
                summaries[name] = self.run_requests(
                    "performance", name, variant, [(row, 42) for row in self.performance_rows], 128)
            baseline_rate = (summaries["baseline"]["output_tokens"] + summaries["baseline_repeat"]["output_tokens"]) / (
                summaries["baseline"]["client_seconds"] + summaries["baseline_repeat"]["client_seconds"])
            candidate_rate = summaries["candidate"]["e2e_output_tok_s"]
            write_json(smoke_path, {"passed": candidate_rate > baseline_rate * 1.01,
                                   "baseline_tok_s": baseline_rate, "candidate_tok_s": candidate_rate,
                                   "cactus_tok_s": summaries["cactus"]["e2e_output_tok_s"],
                                   "speedup": candidate_rate / baseline_rate - 1,
                                   "note": "Short speed check; require >1% speedup; no quality conclusion."})
            self.report()
        if self.args.phase in ("calibration", "all"):
            if not smoke_path.exists() or not json.loads(smoke_path.read_text())["passed"]:
                self.status("STOPPED_PERFORMANCE_GATE")
                return 3
            summaries = {}
            for seed in (42, 43):
                for variant in ("baseline", "candidate"):
                    name = f"{variant}_seed{seed}"
                    summaries[name] = self.run_requests(
                        "calibration", name, variant,
                        [(row, seed) for row in self.calibration_rows], 2048)
            baseline = [summaries[f"baseline_seed{seed}"] for seed in (42, 43)]
            candidate = [summaries[f"candidate_seed{seed}"] for seed in (42, 43)]
            values = {key: {"baseline": sum(x[key] for x in baseline),
                            "candidate": sum(x[key] for x in candidate)}
                      for key in ("correct", "stopped_correct", "truncated", "output_tokens", "client_seconds")}
            passed = all(values[key]["candidate"] >= values[key]["baseline"]
                         for key in ("correct", "stopped_correct"))
            per_seed = {}
            differences = []
            for seed in (42, 43):
                old = summaries[f"baseline_seed{seed}"]
                new = summaries[f"candidate_seed{seed}"]
                per_seed[str(seed)] = {
                    key: {"baseline": old[key], "candidate": new[key]}
                    for key in ("correct", "stopped_correct", "truncated")}
                passed = passed and all(new[key] >= old[key] for key in ("correct", "stopped_correct"))
                old_records = [json.loads(line) for line in
                               (self.root / "calibration" / f"baseline_seed{seed}" / "requests.jsonl").read_text().splitlines()]
                new_records = [json.loads(line) for line in
                               (self.root / "calibration" / f"candidate_seed{seed}" / "requests.jsonl").read_text().splitlines()]
                for before, after in zip(old_records, new_records):
                    if before["correct"] != after["correct"] or before["finish_reason"] != after["finish_reason"]:
                        differences.append({"seed": seed, "question_id": before["question_id"],
                                            "baseline": {k: before[k] for k in ("correct", "finish_reason", "predicted_answer")},
                                            "candidate": {k: after[k] for k in ("correct", "finish_reason", "predicted_answer")}})
            write_json(self.root / "quality_gate.json", {
                "passed": passed, "totals": values, "per_seed": per_seed, "paired_changes": differences,
                "note": "24 train questions x 2 seeds; not independent test evidence. Inspect paired errors before adoption."})
            self.report()
            self.status("CALIBRATION_COMPLETE" if passed else "STOPPED_QUALITY_GATE")
            return 0 if passed else 3
        self.status("SMOKE_COMPLETE")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", choices=("smoke", "calibration", "all"), default="all")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--candidate", choices=("topk", "anchor"), default="topk")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=9)
    args = parser.parse_args()
    if args.depth <= 0 or args.nodes < args.depth:
        parser.error("depth must be positive and nodes must cover the primary depth")
    try:
        controller = Controller(args)
    except (ValueError, FileNotFoundError, BlockingIOError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        return controller.run()
    except ValueError as exc:
        controller.status("STOPPED_PROTOCOL_ERROR")
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        controller.status("FAILED_RUNTIME")
        print(str(exc), file=sys.stderr)
        raise
    finally:
        controller.stop_server()


if __name__ == "__main__":
    raise SystemExit(main())
