"""Day-long run (2026-11-11): third-layer RAVEN on HumanEval-100 and a fresh Spec-Bench sample.

Phases are ``<dataset>:<server kind>`` and run in the order given by ``--phases``.
Each phase starts and stops its own server on port 8000 and is resumable per
request: re-running the same command skips everything already on disk.

  humaneval:<kind>  The 100 HumanEval tasks of the frozen 100+100 benchmark (same prompt,
                    seed, 2048-token budget), then the pinned Docker evaluator; reported next
                    to the recorded Native / Cactus / SpecCascade / original RAVEN numbers.
  gsm8k:<kind>      The 100 GSM8K questions of the same benchmark, scored like the benchmark.
  spec:<kind>       Spec-Bench, 6 tasks x N samples drawn with --spec-sample-seed (different
                    from the 2026-11-10 overnight sample), through remtp.benchmark.

Server kinds:
  raven_v3            frozen RAVEN D3/N9 env + qq_bias tree attention + fast select/verify
                      + in-place commit + cached tree metadata + dead draft KV skipped.
  raven_v3_pathfirst  raven_v3 + first MTP pass reordered to the accepted path (changes Q).
  cactus, native      frozen baseline servers of the 100+100 benchmark.

A phase is skipped when its estimated duration no longer fits before the deadline.
``day_report.md`` and ``status.json`` are rewritten after every phase and request, so a
partial run still leaves a readable report.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
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
from remtp.gsm8k_benchmark import PROMPT_TEMPLATE as GSM_PROMPT  # noqa: E402
from remtp.gsm8k_benchmark import benchmark_messages, extract_model_answer, load_gsm8k  # noqa: E402
from remtp.humaneval_benchmark import PROMPT_TEMPLATE as HUMAN_PROMPT  # noqa: E402
from remtp.humaneval_benchmark import load_humaneval, sanitize_generation  # noqa: E402

BENCH = ROOT / "results/raven_full_100x100_20261010"
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
SPEC_DATA = Path(os.environ.get("SPEC_BENCH_DATA", "/home/apex/ReMTP/data/spec_bench/question.jsonl"))
SPEC_TASKS = ["translation", "summarization", "math_reasoning", "rag", "qa", "multi_turn"]
RECORDED = ("native", "cactus", "spec_cascade", "adaptive_original")

V3_FLAGS = {
    "REMTP_CANONICAL_TREE_ATTENTION": "0",
    "REMTP_DYNAMIC_TREE_FAST_SELECT": "1",
    "REMTP_DYNAMIC_TREE_FAST_VERIFY": "1",
    "REMTP_DYNAMIC_TREE_FAST_COMMIT": "2",
    "REMTP_DYNAMIC_TREE_FAST_METADATA": "1",
    "REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV": "1",
}
V3_MARKERS = [
    "[ReMTP][FastMTPVerified] method=dynamic_tree worker=active",
    "[ReMTP][TargetNumerics] unified 2D probability/value product FP32 (all methods)",
    "D=3 N_max=9 ",
    "[ReMTP][DynamicTree] fast candidate selection",
    "[ReMTP][DynamicTree] fast residual-hit verification",
    "[ReMTP][DynamicTree] in-place ordered KV path commit",
    "[ReMTP][DynamicTree] cached host-built tree masks",
    "[ReMTP][DynamicTree] dead draft KV skipped",
]
PATHFIRST_MARKER = "[ReMTP][DynamicTree] first MTP pass reordered to the accepted path"
NUMERICS = "[ReMTP][TargetNumerics] unified 2D probability/value product FP32 (all methods)"
KINDS = {
    "raven_v3": dict(env="adaptive_original/attempt_2", flags=V3_FLAGS, markers=V3_MARKERS, absent=[PATHFIRST_MARKER]),
    "raven_v3_pathfirst": dict(env="adaptive_original/attempt_2",
                               flags=dict(V3_FLAGS, REMTP_DYNAMIC_TREE_PATH_ORDERED_FIRST_PASS="1"),
                               markers=V3_MARKERS + [PATHFIRST_MARKER], absent=[]),
    "cactus": dict(env="cactus/attempt_1", flags={}, absent=[],
                   markers=["[ReMTP][FastMTPVerified] method=cactus worker=active", NUMERICS]),
    "native": dict(env="native/attempt_1", flags={}, absent=[],
                   markers=["[ReMTP][FastMTPVerified] method=native worker=active", NUMERICS]),
}
# Rough durations (hours) used only to decide whether a phase still fits before the deadline.
ESTIMATE = {"humaneval": 9.5, "gsm8k": 7.0, "spec": 0.036}   # spec: hours per request (x requests)
EVAL = dict(image="python:3-slim", timeout=8.0, memory="512m", cpus="1.0")


def totals(records: list[dict]) -> dict:
    tot = lambda key: sum(r["metrics"][key] for r in records)
    gen, dec = tot("vllm:generation_tokens_total"), tot("vllm:request_decode_time_seconds_sum")
    rounds, acc = tot("vllm:spec_decode_num_drafts_total"), tot("vllm:spec_decode_num_accepted_tokens_total")
    client = sum(r["client_seconds"] for r in records)
    out = sum(r["output_tokens"] for r in records)
    known = [r for r in records if r.get("correct") is not None]
    return dict(requests=len(records), evaluated=len(known), correct=sum(bool(r["correct"]) for r in known),
                accuracy=sum(bool(r["correct"]) for r in known) / len(known) if known else None,
                e2e_output_tok_s=out / client if client else None, mean_seconds=client / max(len(records), 1),
                decode_tok_s=gen / dec if dec else None, mean_acceptance_length=1 + acc / rounds if rounds else None,
                seconds_per_round=dec / rounds if rounds else None,
                truncated=sum(r["finish_reason"] == "length" for r in records),
                mean_output_tokens=out / max(len(records), 1))


class Controller:
    def __init__(self, out: Path, deadline: float, smoke: bool):
        self.out, self.deadline, self.smoke = out, deadline, smoke
        self.proc = None
        self.started = time.time()
        out.mkdir(parents=True, exist_ok=True)
        self.logf = (out / "controller.log").open("a")
        self.sections: dict[str, str] = {}
        self.notes: list[str] = []

    # ----- infrastructure -------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n"); self.logf.flush()

    def status(self, **fields) -> None:
        fields.update(updated=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_hours=(time.time() - self.started) / 3600,
                      hours_to_deadline=self.remaining_hours(), finished_sections=list(self.sections), notes=self.notes)
        tmp = self.out / "status.json.tmp"
        tmp.write_text(json.dumps(fields, ensure_ascii=False, indent=1)); tmp.replace(self.out / "status.json")

    def remaining_hours(self) -> float:
        return (self.deadline - time.time()) / 3600

    def write_report(self) -> None:
        lines = ["# 全天实验报告（2026-11-11 启动）", "",
                 f"更新时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；已运行 {(time.time() - self.started) / 3600:.2f} 小时。", "",
                 "RAVEN v3 = 冻结的 RAVEN D3/N9 环境 + 原生 qq_bias 树注意力 + 快速选择/验证 + 原地路径提交 + 树元数据缓存 + 跳过死的草稿 KV。"
                 "`raven_v3_pathfirst` 另外把首次 MTP forward 按接受路径重排（会改变提议分布 Q，不改变验收规则）。", ""]
        for name in self.sections:
            lines += [self.sections[name], ""]
        if self.notes:
            lines += ["## 运行备注", ""] + [f"- {note}" for note in self.notes] + [""]
        tmp = self.out / "day_report.md.tmp"
        tmp.write_text("\n".join(lines) + "\n"); tmp.replace(self.out / "day_report.md")

    @staticmethod
    def port_open() -> bool:
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", PORT)) == 0

    def start_server(self, kind: str, directory: Path) -> None:
        self.stop_server()
        if self.port_open():
            raise RuntimeError("port 8000 occupied by an unrelated process")
        spec = KINDS[kind]
        directory.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("REMTP_", "VLLM_"))}
        env.update(json.loads((BENCH / "servers" / spec["env"] / "effective_env.json").read_text()))
        env.update(spec["flags"])
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        (directory / "effective_env.json").write_text(json.dumps(
            {k: v for k, v in env.items() if k.startswith(("REMTP_", "VLLM_")) or k in
             ("METHOD", "PORT", "MAX_MODEL_LEN", "KV_CACHE_MEMORY_BYTES", "QUANTIZATION", "LOAD_FORMAT",
              "MTP_TOKENS", "TARGET_ATTENTION_BACKEND", "ENABLE_PREFIX_CACHING")}, indent=2))
        log_path = directory / "server.log"
        offset = log_path.stat().st_size if log_path.exists() else 0
        logf = log_path.open("a")
        self.proc = subprocess.Popen([str(ROOT / "scripts/serve_fastmtp_jetson.sh")], cwd=ROOT, env=env,
                                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 1500
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early ({self.proc.returncode}); see {log_path}")
            try:
                with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(3)
        else:
            raise RuntimeError("server did not become healthy")
        with log_path.open() as handle:
            handle.seek(offset); text = handle.read()
        missing = [m for m in spec["markers"] if m not in text]
        unexpected = [m for m in spec["absent"] if m in text]
        # Chain baselines keep their frozen env (canonical hook installed but inert).
        if kind.startswith("raven") and "canonical ancestor KV reads" in text:
            unexpected.append("canonical tree attention")
        if missing or unexpected:
            raise RuntimeError(f"startup markers wrong for {kind}: missing={missing} unexpected={unexpected}")
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

    # ----- frozen-protocol replay: GSM8K-100 / HumanEval-100 ---------------
    def replay_phase(self, dataset: str, kind: str) -> None:
        name = f"{dataset}_{kind}"
        directory = self.out / name
        directory.mkdir(parents=True, exist_ok=True)
        reference = [json.loads(l) for l in (BENCH / dataset / "adaptive_original/requests.jsonl").open()]
        if dataset == "gsm8k":
            rows = {r["question_id"]: r for r in load_gsm8k(ROOT / "data/gsm8k/test.jsonl")}
            key = lambda ref: ref["question_id"]
            content = lambda row: GSM_PROMPT.format(question=row["question"])
        else:
            rows = {r["task_id"]: r for r in load_humaneval(BENCH / "frozen_data/humaneval/HumanEval.jsonl.gz")}
            key = lambda ref: ref["task_id"]
            content = lambda row: HUMAN_PROMPT.format(prompt=row["prompt"])
        if self.smoke:
            reference = reference[:2]
        max_tokens = 64 if self.smoke else 2048
        done_path = directory / "requests.jsonl"
        done: dict = {}
        if done_path.exists():
            for line in done_path.open():
                rec = json.loads(line); done[rec["sample_id"]] = rec
        pending = [r for r in reference if r["sample_id"] not in done]
        self.log(f"{name}: {len(reference)} items, {len(done)} already done")
        if pending:
            self.start_server(kind, directory)
        for ref in pending:
            if self.remaining_hours() < 0:
                self.note(f"{name}: 到达截止时间，生成在 {len(done)}/{len(reference)} 处停止"); break
            row = rows[key(ref)]
            payload = dict(model="TencentBAC/FastMTP", messages=benchmark_messages(content(row), empty_system_prompt=True),
                           chat_template_kwargs={"enable_thinking": False}, temperature=0.6, seed=ref["seed"],
                           max_tokens=max_tokens, return_token_ids=True)
            self.status(phase=name, done=len(done), total=len(reference), current=ref["sample_id"])
            before = fetch_metrics(BASE, 30); t0 = time.perf_counter()
            value = self.post(payload, 5400)
            elapsed = time.perf_counter() - t0; after = fetch_metrics(BASE, 30)
            choice = value["choices"][0]; text = choice["message"]["content"] or ""
            m = metric_delta(before, after); rounds = m["vllm:spec_decode_num_drafts_total"]
            rec = dict(task=dataset, method=kind, sample_id=ref["sample_id"], question_id=ref["question_id"], seed=ref["seed"],
                       prompt_matches_benchmark=(value.get("prompt_token_ids") == ref["prompt_token_ids"]),
                       output_tokens=int(value["usage"]["completion_tokens"]), finish_reason=choice["finish_reason"],
                       client_seconds=elapsed, metrics=m, rounds=rounds,
                       benchmark_output_tokens=ref["output_tokens"], benchmark_client_seconds=ref["client_seconds"],
                       output=text, output_token_ids=choice.get("token_ids"))
            if dataset == "gsm8k":
                predicted, source = extract_model_answer(text)
                rec.update(gold_answer=row["gold_answer"], predicted_answer=predicted, answer_source=source,
                           correct=(predicted == row["gold_answer"]))
            else:
                candidate, mode = sanitize_generation(text, row["entry_point"])
                rec.update(task_id=row["task_id"], candidate=candidate, candidate_mode=mode, correct=None)
            done[rec["sample_id"]] = rec
            with done_path.open("a") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.log(f"{name} [{len(done)}/{len(reference)}] {rec['sample_id']} out={rec['output_tokens']} {elapsed:.0f}s "
                     f"s/round={m['vllm:request_decode_time_seconds_sum'] / max(rounds, 1):.4f} "
                     f"tok/round={m['vllm:generation_tokens_total'] / max(rounds, 1):.3f} prompt_ok={rec['prompt_matches_benchmark']}"
                     + (f" correct={rec['correct']}" if dataset == "gsm8k" else ""))
            if len(done) % 10 == 0:
                self.replay_summary(dataset, kind, directory, done, reference, rows)
        self.stop_server()                       # Docker execution never overlaps GPU generation.
        if dataset == "humaneval":
            self.evaluate_humaneval(directory, done, reference, rows)
        self.replay_summary(dataset, kind, directory, done, reference, rows)

    def evaluate_humaneval(self, directory: Path, done: dict, reference: list[dict], rows: dict) -> None:
        from remtp.humaneval_evaluator import build_candidate_source, docker_image_id, evaluate_source_docker
        image_id = docker_image_id(EVAL["image"])
        pinned = json.loads((BENCH / "protocol.json").read_text())["evaluation"]["image_id"]
        if image_id is None:
            self.note("HumanEval 判题跳过：Docker 镜像 python:3-slim 不可用"); return
        if image_id != pinned:
            self.note(f"HumanEval 判题镜像与基准不同（{image_id} 对 {pinned}），判分仅供参考")
        eval_path = directory / "evaluation.jsonl"
        evaluated: dict = {}
        if eval_path.exists():
            for line in eval_path.open():
                rec = json.loads(line); evaluated[rec["sample_id"]] = rec
        for ref in reference:
            rec = done.get(ref["sample_id"])
            if rec is None:
                continue
            digest = hashlib.sha256(rec["output"].encode()).hexdigest()
            outcome = evaluated.get(rec["sample_id"])
            if outcome is None or outcome["output_sha256"] != digest:
                result = evaluate_source_docker(build_candidate_source(rows[rec["task_id"]], rec), image=EVAL["image"],
                                                timeout=EVAL["timeout"], memory=EVAL["memory"], cpus=EVAL["cpus"])
                status = result.get("status")
                if status not in {"passed", "failed", "syntax_error", "runtime_error", "timeout"}:
                    self.note(f"HumanEval 判题基础设施异常（{rec['sample_id']}: {status}），该题未计分"); continue
                outcome = dict(sample_id=rec["sample_id"], task_id=rec["task_id"], seed=rec["seed"],
                               correct=(status == "passed"), output_sha256=digest, image_id=image_id, **result)
                evaluated[rec["sample_id"]] = outcome
                with eval_path.open("a") as handle:
                    handle.write(json.dumps(outcome, ensure_ascii=False) + "\n")
            rec["correct"] = outcome["correct"]
        self.log(f"humaneval evaluation: {sum(bool(v['correct']) for v in evaluated.values())}/{len(evaluated)} passed")

    def replay_summary(self, dataset: str, kind: str, directory: Path, done: dict, reference: list[dict], rows: dict) -> None:
        recs = list(done.values())
        if not recs:
            return
        table = {kind: dict(method=kind, of=len(reference), **totals(recs))}
        ids = set(done)
        for method in RECORDED:
            rs = [json.loads(l) for l in (BENCH / dataset / method / "requests.jsonl").open()]
            if dataset == "humaneval":
                verdict = {json.loads(l)["sample_id"]: json.loads(l)["correct"]
                           for l in (BENCH / dataset / method / "evaluation.jsonl").open()}
                for r in rs:
                    r["correct"] = verdict.get(r["sample_id"])
            rs = [r for r in rs if r["sample_id"] in ids]
            table[method] = dict(method=method, of=len(reference), **totals(rs))
        (directory / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=1))
        label = "GSM8K" if dataset == "gsm8k" else "HumanEval"
        quality = "正确率" if dataset == "gsm8k" else "pass@1"
        lines = [f"## {label}-100（冻结协议：同题/同种子/2048 预算），服务配置 `{kind}`", "",
                 f"`{kind}` 实跑 {len(recs)}/{len(reference)} 题；其余方法为 10 月基准记录中相同题目的数字。", "",
                 f"| 方法 | {quality} | E2E tok/s | 平均整题秒 | MAL | 每轮秒 | decode tok/s | 截断 | 平均输出 token |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
        for method in RECORDED + (kind,):
            r = table[method]
            q = "待判分" if not r["evaluated"] else f"{r['correct']}/{r['evaluated']}（{100 * r['accuracy']:.0f}%）"
            lines.append(f"| {method} | {q} | {fmt(r['e2e_output_tok_s'])} | {fmt(r['mean_seconds'])} | {fmt(r['mean_acceptance_length'])} | "
                         f"{fmt(r['seconds_per_round'])} | {fmt(r['decode_tok_s'])} | {r['truncated']} | {r['mean_output_tokens']:.1f} |")
        mismatched = sum(not r["prompt_matches_benchmark"] for r in recs)
        if mismatched:
            lines += ["", f"注意：{mismatched} 条请求的 prompt token 与基准记录不一致。"]
        self.sections[f"{dataset}:{kind}"] = "\n".join(lines)
        self.write_report()

    # ----- Spec-Bench ------------------------------------------------------
    def specbench_phase(self, kind: str, samples: int, max_tokens: int, sample_seed: int) -> None:
        directory = self.out / f"specbench_{kind}"
        if not (directory / "summary.json").exists():
            # Start the server first: if it cannot start, nothing on disk is touched.
            self.start_server(kind, self.out / f"server_{kind}")
            if directory.exists():
                # remtp.benchmark refuses to reuse a directory; keep the partial one aside.
                directory.rename(directory.with_name(directory.name + f"_partial_{int(time.time())}"))
            tasks = SPEC_TASKS[:2] if self.smoke else SPEC_TASKS
            cmd = [sys.executable, "-m", "remtp.benchmark", "--data", str(SPEC_DATA), "--tasks", *tasks,
                   "--samples-per-task", str(samples), "--sample-seed", str(sample_seed), "--temperature", "0.6",
                   "--generation-seed", "42", "--max-tokens", str(max_tokens), "--mtp-tokens", "3",
                   "--model", "TencentBAC/FastMTP", "--run-name", kind, "--timeout", "5400", "--output-dir", str(directory)]
            self.status(phase=f"specbench_{kind}", tasks=tasks, samples=samples, sample_seed=sample_seed)
            self.log(f"specbench {kind}: {' '.join(cmd[2:])}")
            with (self.out / f"specbench_{kind}.log").open("a") as logf:
                code = subprocess.call(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONPATH=str(ROOT)))
            self.stop_server()
            if code != 0:
                self.note(f"Spec-Bench {kind} 的 runner 以 {code} 退出，见 specbench_{kind}.log"); return
        else:
            self.log(f"specbench {kind}: already complete")
        self.spec_section(samples, max_tokens, sample_seed)

    def spec_section(self, samples: int, max_tokens: int, sample_seed: int) -> None:
        found = {}
        for kind in KINDS:
            path = self.out / f"specbench_{kind}/summary.json"
            if path.exists():
                found[kind] = json.loads(path.read_text())["results"]
        if not found:
            return
        lines = [f"## Spec-Bench（6 任务各 {samples} 条，抽样种子 {sample_seed}，{max_tokens} token 预算，temperature 0.6）", "",
                 "| 方法 | 任务 | 请求数 | 输出 token | decode tok/s | E2E tok/s | MAL | 每轮秒 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for kind, results in found.items():
            for r in results:
                rounds = r.get("draft_rounds") or 0
                per_round = f"{r['decode_seconds'] / rounds:.4f}" if rounds else "n/a"
                lines.append(f"| {kind} | {r['task']} | {r['requests']} | {r['output_tokens']} | {r['decode_tok_s']:.3f} | "
                             f"{r['e2e_output_tok_s']:.3f} | {r['mean_acceptance_length']:.3f} | {per_round} |")
        if "cactus" in found:
            base = found["cactus"][-1]
            lines += ["", "overall 相对 Cactus："]
            for kind, results in found.items():
                if kind != "cactus":
                    o = results[-1]
                    lines.append(f"- `{kind}`：E2E {100 * (o['e2e_output_tok_s'] / base['e2e_output_tok_s'] - 1):+.1f}%，"
                                 f"decode {100 * (o['decode_tok_s'] / base['decode_tok_s'] - 1):+.1f}%，"
                                 f"MAL {100 * (o['mean_acceptance_length'] / base['mean_acceptance_length'] - 1):+.1f}%")
        self.sections["spec"] = "\n".join(lines)
        self.write_report()

    def note(self, text: str) -> None:
        self.notes.append(text); self.log("NOTE " + text); self.write_report()

    def run(self, phases: list[str], spec_samples: int, spec_max_tokens: int, spec_sample_seed: int) -> None:
        try:
            for phase in phases:
                dataset, kind = phase.split(":")
                if kind not in KINDS or dataset not in ("gsm8k", "humaneval", "spec"):
                    raise ValueError(f"unknown phase {phase}")
                if dataset == "spec":
                    complete = (self.out / f"specbench_{kind}/summary.json").exists()
                    estimate = ESTIMATE["spec"] * spec_samples * 7     # multi_turn counts twice
                else:
                    complete = False
                    estimate = ESTIMATE[dataset]
                    done_path = self.out / f"{dataset}_{kind}/requests.jsonl"
                    if done_path.exists():
                        estimate *= max(0.0, 1 - sum(1 for _ in done_path.open()) / 100)
                if not complete and not self.smoke and self.remaining_hours() < estimate:
                    self.note(f"跳过 {phase}：预计需要 {estimate:.1f} 小时，距截止只剩 {self.remaining_hours():.1f} 小时"); continue
                try:
                    if dataset == "spec":
                        self.specbench_phase(kind, spec_samples, spec_max_tokens, spec_sample_seed)
                    else:
                        self.replay_phase(dataset, kind)
                except Exception as error:      # one failed phase must not lose the remaining ones
                    self.stop_server()
                    self.note(f"{phase} 失败：{type(error).__name__}: {error}")
        finally:
            self.stop_server()
            self.write_report()
            self.status(phase="finished")
            self.log("day_report.md written")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phases", default="humaneval:raven_v3,spec:raven_v3,spec:cactus,spec:raven_v3_pathfirst,spec:native")
    parser.add_argument("--deadline-hours", type=float, default=24.0)
    parser.add_argument("--spec-samples", type=int, default=15)
    parser.add_argument("--spec-max-tokens", type=int, default=512)
    parser.add_argument("--spec-sample-seed", type=int, default=20261111)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    # Refuse to run next to another instance or a live server: a second copy must not
    # rename directories or rewrite status.json / day_report.md of a running one.
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".controller.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another run_day controller holds {args.out / '.controller.lock'}; exiting without changes", flush=True)
        return 1
    if Controller.port_open():
        print("port 8000 is already in use (another server or run is active); exiting without changes", flush=True)
        return 1
    controller = Controller(args.out, time.time() + args.deadline_hours * 3600, args.smoke)
    if args.smoke:
        args.spec_samples, args.spec_max_tokens = 1, 32
    controller.log(f"start phases={args.phases} smoke={args.smoke} deadline={args.deadline_hours}h")
    controller.run(args.phases.split(","), args.spec_samples, args.spec_max_tokens, args.spec_sample_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
