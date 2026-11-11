"""Overnight run: optimised RAVEN on the frozen GSM8K-100 protocol, then Spec-Bench vs Cactus.

Phases (each starts and stops its own server on port 8000, resumable per request):

  gsm8k_raven_opt   RAVEN D3/N9 with REMTP_CANONICAL_TREE_ATTENTION=0 and the three fast
                    flags, replaying the 100 GSM8K questions of the 100+100 benchmark with
                    the same prompt, seed and 2048-token budget; scored like the benchmark.
  specbench_raven_opt   Spec-Bench subset (6 tasks x N samples, 512-token budget) on the
                    same server configuration, through remtp.benchmark.
  specbench_cactus  The same Spec-Bench subset on the frozen Cactus baseline server.
  specbench_native  Optional, only if the deadline allows.

Everything is written under the output directory: controller.log, status.json,
per-phase requests.jsonl / summary files, and overnight_report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remtp.benchmark import fetch_metrics, metric_delta  # noqa: E402
from remtp.gsm8k_benchmark import PROMPT_TEMPLATE, benchmark_messages, extract_model_answer, load_gsm8k  # noqa: E402

BENCH = ROOT / "results/raven_full_100x100_20261010"
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
SPEC_DATA = Path(os.environ.get("SPEC_BENCH_DATA", "/home/apex/ReMTP/data/spec_bench/question.jsonl"))
SPEC_TASKS = ["translation", "summarization", "math_reasoning", "rag", "qa", "multi_turn"]

RAVEN_OPT_FLAGS = {
    "REMTP_CANONICAL_TREE_ATTENTION": "0",
    "REMTP_DYNAMIC_TREE_FAST_SELECT": "1",
    "REMTP_DYNAMIC_TREE_FAST_VERIFY": "1",
    "REMTP_DYNAMIC_TREE_FAST_COMMIT": "1",
}
MARKERS = {
    "raven_opt": [
        "[ReMTP][FastMTPVerified] method=dynamic_tree worker=active",
        "[ReMTP][TargetNumerics] unified 2D probability/value product FP32 (all methods)",
        "D=3 N_max=9 ",
        "[ReMTP][DynamicTree] fast candidate selection",
        "[ReMTP][DynamicTree] fast residual-hit verification",
        "[ReMTP][DynamicTree] fused KV path commit across layers",
    ],
    "cactus": ["[ReMTP][FastMTPVerified] method=cactus worker=active",
               "[ReMTP][TargetNumerics] unified 2D probability/value product FP32 (all methods)"],
    "native": ["[ReMTP][FastMTPVerified] method=native worker=active",
               "[ReMTP][TargetNumerics] unified 2D probability/value product FP32 (all methods)"],
}
ENV_FILES = {
    "raven_opt": BENCH / "servers/adaptive_original/attempt_2/effective_env.json",
    "cactus": BENCH / "servers/cactus/attempt_1/effective_env.json",
    "native": BENCH / "servers/native/attempt_1/effective_env.json",
}


class Controller:
    def __init__(self, out: Path, deadline: float, smoke: bool):
        self.out = out
        self.deadline = deadline
        self.smoke = smoke
        self.proc = None
        self.started = time.time()
        out.mkdir(parents=True, exist_ok=True)
        self.logf = (out / "controller.log").open("a")

    # ----- infrastructure -------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n"); self.logf.flush()

    def status(self, **fields) -> None:
        fields.update(updated=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_hours=(time.time() - self.started) / 3600,
                      hours_to_deadline=(self.deadline - time.time()) / 3600)
        (self.out / "status.json").write_text(json.dumps(fields, ensure_ascii=False, indent=1))

    def remaining_hours(self) -> float:
        return (self.deadline - time.time()) / 3600

    @staticmethod
    def port_open() -> bool:
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", PORT)) == 0

    def start_server(self, kind: str, directory: Path) -> None:
        self.stop_server()
        if self.port_open():
            raise RuntimeError("port 8000 occupied by an unrelated process")
        directory.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("REMTP_", "VLLM_"))}
        env.update(json.loads(ENV_FILES[kind].read_text()))
        if kind == "raven_opt":
            env.update(RAVEN_OPT_FLAGS)
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        (directory / "effective_env.json").write_text(json.dumps(
            {k: v for k, v in env.items() if k.startswith(("REMTP_", "VLLM_")) or k in
             ("METHOD", "PORT", "MAX_MODEL_LEN", "KV_CACHE_MEMORY_BYTES", "QUANTIZATION", "LOAD_FORMAT",
              "MTP_TOKENS", "TARGET_ATTENTION_BACKEND", "ENABLE_PREFIX_CACHING")}, indent=2))
        logf = (directory / "server.log").open("a")
        self.proc = subprocess.Popen([str(ROOT / "scripts/serve_fastmtp_jetson.sh")], cwd=ROOT, env=env,
                                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 1500
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early ({self.proc.returncode}); see {directory / 'server.log'}")
            try:
                with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(3)
        else:
            raise RuntimeError("server did not become healthy")
        text = (directory / "server.log").read_text()
        missing = [m for m in MARKERS[kind] if m not in text]
        # The chain baselines keep their frozen env (canonical hook installed but inert);
        # only the optimised RAVEN server must run without the canonical layout.
        canonical_installed = "canonical ancestor KV reads" in text
        if missing or (kind == "raven_opt" and canonical_installed):
            raise RuntimeError(f"startup markers wrong for {kind}: missing={missing} canonical={canonical_installed}")
        self.post(dict(model="TencentBAC/FastMTP", messages=[{"role": "user", "content": "Calculate 2 + 2."}],
                       chat_template_kwargs={"enable_thinking": False}, temperature=0.6, seed=1, max_tokens=8), 600)
        self.log(f"server up: {kind} pid={self.proc.pid}")

    def stop_server(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(90):
            if proc.poll() is not None:
                break
            time.sleep(1)
        else:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        for _ in range(60):
            if not self.port_open():
                break
            time.sleep(1)
        self.log("server stopped")

    @staticmethod
    def post(payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    # ----- phase 1: GSM8K-100, frozen protocol --------------------------------
    def gsm8k_phase(self) -> dict:
        directory = self.out / "gsm8k_raven_opt"
        directory.mkdir(parents=True, exist_ok=True)
        reference = [json.loads(l) for l in (BENCH / "gsm8k/adaptive_original/requests.jsonl").open()]
        rows = {r["question_id"]: r for r in load_gsm8k(ROOT / "data/gsm8k/test.jsonl")}
        if self.smoke:
            reference = reference[:1]
        max_tokens = 48 if self.smoke else 2048
        done_path = directory / "requests.jsonl"
        done = {}
        if done_path.exists():
            for line in done_path.open():
                rec = json.loads(line); done[rec["question_id"]] = rec
        self.log(f"gsm8k phase: {len(reference)} questions, {len(done)} already done")
        pending = [r for r in reference if r["question_id"] not in done]
        if pending:
            self.start_server("raven_opt", directory)
        for i, ref in enumerate(pending):
            if self.remaining_hours() < 0:
                self.log("deadline reached inside gsm8k phase; stopping early"); break
            qid, row = ref["question_id"], rows[ref["question_id"]]
            payload = dict(model="TencentBAC/FastMTP",
                           messages=benchmark_messages(PROMPT_TEMPLATE.format(question=row["question"]), empty_system_prompt=True),
                           chat_template_kwargs={"enable_thinking": False}, temperature=0.6, seed=ref["seed"],
                           max_tokens=max_tokens, return_token_ids=True)
            self.status(phase="gsm8k_raven_opt", done=len(done), total=len(reference), current=qid)
            before = fetch_metrics(BASE, 30); t0 = time.perf_counter()
            value = self.post(payload, 5400)
            elapsed = time.perf_counter() - t0; after = fetch_metrics(BASE, 30)
            choice = value["choices"][0]; text = choice["message"]["content"] or ""
            m = metric_delta(before, after); rounds = m["vllm:spec_decode_num_drafts_total"]
            predicted, source = extract_model_answer(text)
            rec = dict(task="gsm8k", method="raven_opt", question_id=qid, seed=ref["seed"],
                       prompt_matches_benchmark=(value.get("prompt_token_ids") == ref["prompt_token_ids"]),
                       output_tokens=int(value["usage"]["completion_tokens"]), finish_reason=choice["finish_reason"],
                       client_seconds=elapsed, metrics=m, rounds=rounds,
                       gold_answer=row["gold_answer"], predicted_answer=predicted, answer_source=source,
                       correct=(predicted == row["gold_answer"]),
                       benchmark_correct=ref.get("correct"), benchmark_output_tokens=ref["output_tokens"],
                       benchmark_client_seconds=ref["client_seconds"],
                       output=text, output_token_ids=choice.get("token_ids"))
            done[qid] = rec
            done_path.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.log(f"gsm8k [{len(done)}/{len(reference)}] q{qid} out={rec['output_tokens']} {elapsed:.0f}s "
                     f"correct={rec['correct']} (bench {rec['benchmark_correct']}) s/round={m['vllm:request_decode_time_seconds_sum']/max(rounds,1):.4f}")
        self.stop_server()
        summary = self.gsm8k_summary(directory, done, len(reference))
        self.log(f"gsm8k summary: {json.dumps(summary['raven_opt'])}")
        return summary

    def gsm8k_summary(self, directory: Path, done: dict, total: int) -> dict:
        recs = list(done.values())
        tot = lambda k: sum(r["metrics"][k] for r in recs)
        gen, dec, rounds = tot("vllm:generation_tokens_total"), tot("vllm:request_decode_time_seconds_sum"), tot("vllm:spec_decode_num_drafts_total")
        acc = tot("vllm:spec_decode_num_accepted_tokens_total"); client = sum(r["client_seconds"] for r in recs)
        out = sum(r["output_tokens"] for r in recs)
        mine = dict(method="raven_opt", requests=len(recs), of=total, correct=sum(bool(r["correct"]) for r in recs),
                    accuracy=sum(bool(r["correct"]) for r in recs) / max(len(recs), 1),
                    e2e_output_tok_s=out / client if client else None, mean_seconds=client / max(len(recs), 1),
                    decode_tok_s=gen / dec if dec else None, mean_acceptance_length=1 + acc / rounds if rounds else None,
                    seconds_per_round=dec / rounds if rounds else None, truncated=sum(r["finish_reason"] == "length" for r in recs),
                    mean_output_tokens=out / max(len(recs), 1))
        # Same questions from the recorded benchmark, restricted to the questions completed here.
        qids = set(done)
        table = {"raven_opt": mine}
        for method in ("native", "cactus", "spec_cascade", "adaptive_original"):
            path = BENCH / f"gsm8k/{method}/requests.jsonl"
            rs = [json.loads(l) for l in path.open()]
            rs = [r for r in rs if r["question_id"] in qids]
            t = lambda k: sum(r["metrics"][k] for r in rs)
            g, d, ro, a = t("vllm:generation_tokens_total"), t("vllm:request_decode_time_seconds_sum"), t("vllm:spec_decode_num_drafts_total"), t("vllm:spec_decode_num_accepted_tokens_total")
            c = sum(r["client_seconds"] for r in rs); o = sum(r["output_tokens"] for r in rs)
            table[method] = dict(method=method, requests=len(rs), of=total, correct=sum(bool(r["correct"]) for r in rs),
                                 accuracy=sum(bool(r["correct"]) for r in rs) / max(len(rs), 1), e2e_output_tok_s=o / c if c else None,
                                 mean_seconds=c / max(len(rs), 1), decode_tok_s=g / d if d else None,
                                 mean_acceptance_length=1 + a / ro if ro else None, seconds_per_round=d / ro if ro else None,
                                 truncated=sum(r["finish_reason"] == "length" for r in rs), mean_output_tokens=o / max(len(rs), 1))
        (directory / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=1))
        with (directory / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mine)); w.writeheader(); w.writerows(table.values())
        lines = ["# GSM8K（与 100+100 基准相同的题目/种子/预算）", "",
                 f"优化后 RAVEN 实跑 {len(recs)}/{total} 题；其他方法为 10 月基准记录中相同题目的数字。", "",
                 "| 方法 | 正确数/正确率 | E2E tok/s | 平均整题秒 | MAL | 每轮秒 | decode tok/s | 截断 | 平均输出 token |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for k in ("native", "cactus", "spec_cascade", "adaptive_original", "raven_opt"):
            r = table[k]
            f2 = lambda v: "n/a" if v is None else f"{v:.4f}"
            lines.append(f"| {k} | {r['correct']}/{r['requests']}（{100*r['accuracy']:.0f}%） | {f2(r['e2e_output_tok_s'])} | "
                         f"{f2(r['mean_seconds'])} | {f2(r['mean_acceptance_length'])} | {f2(r['seconds_per_round'])} | {f2(r['decode_tok_s'])} | {r['truncated']} | {r['mean_output_tokens']:.1f} |")
        (directory / "report.md").write_text("\n".join(lines) + "\n")
        return table

    # ----- phase 2/3: Spec-Bench -----------------------------------------------
    def specbench_phase(self, kind: str, samples: int, max_tokens: int) -> dict | None:
        directory = self.out / f"specbench_{kind}"
        if (directory / "summary.json").exists():
            self.log(f"specbench {kind}: already complete"); return json.loads((directory / "summary.json").read_text())
        if directory.exists():
            # remtp.benchmark refuses to reuse a directory; keep the partial one aside.
            directory.rename(directory.with_name(directory.name + f"_partial_{int(time.time())}"))
        self.start_server(kind, self.out / f"server_{kind}")
        tasks = SPEC_TASKS[:2] if self.smoke else SPEC_TASKS
        cmd = [sys.executable, "-m", "remtp.benchmark", "--data", str(SPEC_DATA), "--tasks", *tasks,
               "--samples-per-task", str(samples), "--temperature", "0.6", "--generation-seed", "42",
               "--max-tokens", str(max_tokens), "--mtp-tokens", "3", "--model", "TencentBAC/FastMTP",
               "--run-name", kind, "--timeout", "5400", "--output-dir", str(directory)]
        self.status(phase=f"specbench_{kind}", tasks=tasks, samples=samples)
        self.log(f"specbench {kind}: {' '.join(cmd[2:])}")
        with (self.out / f"specbench_{kind}.log").open("a") as logf:
            code = subprocess.call(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONPATH=str(ROOT)))
        self.stop_server()
        if code != 0:
            self.log(f"specbench {kind} runner exited {code}"); return None
        summary = json.loads((directory / "summary.json").read_text())
        self.log(f"specbench {kind} overall: {json.dumps(summary['results'][-1])}")
        return summary

    def final_report(self, gsm: dict | None, spec: dict) -> None:
        lines = ["# 通宵实验报告（2026-11-10 夜）", "", f"生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；总耗时 {(time.time()-self.started)/3600:.2f} 小时。", ""]
        if gsm:
            lines += [(self.out / "gsm8k_raven_opt/report.md").read_text(), ""]
        if spec:
            lines += ["# Spec-Bench（6 任务，同一抽样，512 token 预算，temperature 0.6）", "",
                      "| 方法 | 任务 | 条数 | 输出 token | decode tok/s | E2E tok/s | MAL | 草稿接受率 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for kind, summary in spec.items():
                for r in summary["results"]:
                    rate = r["draft_token_acceptance_rate"]
                    lines.append(f"| {kind} | {r['task']} | {r['requests']} | {r['output_tokens']} | {r['decode_tok_s']:.3f} | {r['e2e_output_tok_s']:.3f} | "
                                 f"{r['mean_acceptance_length']:.3f} | {100*rate:.1f}% |")
            lines.append("")
        (self.out / "overnight_report.md").write_text("\n".join(lines) + "\n")
        self.log("overnight_report.md written")

    def run(self, phases: list[str], spec_samples: int, spec_max_tokens: int) -> None:
        gsm = None; spec = {}
        try:
            if "gsm8k" in phases:
                gsm = self.gsm8k_phase()
            for kind in ("raven_opt", "cactus", "native"):
                if f"spec_{kind}" not in phases:
                    continue
                if kind == "native" and self.remaining_hours() < 2.5 and not self.smoke:
                    self.log("skipping specbench native: not enough time before the deadline"); continue
                if self.remaining_hours() < 0 and not self.smoke:
                    self.log(f"deadline reached; skipping specbench {kind}"); continue
                result = self.specbench_phase(kind, spec_samples, spec_max_tokens)
                if result:
                    spec[kind] = result
        finally:
            self.stop_server()
            self.final_report(gsm, spec)
            self.status(phase="finished", gsm8k=bool(gsm), specbench=list(spec))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phases", default="gsm8k,spec_raven_opt,spec_cactus,spec_native")
    parser.add_argument("--deadline-hours", type=float, default=11.0)
    parser.add_argument("--spec-samples", type=int, default=10)
    parser.add_argument("--spec-max-tokens", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    controller = Controller(args.out, time.time() + args.deadline_hours * 3600, args.smoke)
    if args.smoke:
        args.spec_samples, args.spec_max_tokens = 1, 32
    controller.log(f"start phases={args.phases} smoke={args.smoke} deadline={args.deadline_hours}h")
    controller.run(args.phases.split(","), args.spec_samples, args.spec_max_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
