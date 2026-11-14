"""All methods on HumanEval-100 with the W4A16 Marlin target in MAXN mode, repeated seeds.

Phases are ``<kind>:<run>``; run ``r`` uses the frozen 100+100 HumanEval prompts and
2048-token budget with per-item seed ``frozen_seed + (r - 1)`` (run 1 reproduces the
frozen seeds ``task*10+42``).  Every phase starts and stops its own server, is
resumable per request, and keeps everything needed for later analysis:

  <phase>/requests.jsonl     one record per request (text, token ids, vLLM metrics,
                             wall-clock start/end for joining with rounds.jsonl)
  <phase>/rounds.jsonl       per-round verifier log (remtp.round_logger), all SD methods
  <phase>/evaluation.jsonl   pinned Docker HumanEval verdicts
  <phase>/server.log, effective_env.json, device.json
  controller.log, status.json, protocol.json, suite_report.md, suite_summary.json

Server kinds (frozen 100+100 envs + W4A16 overrides): native, cactus, spec_cascade,
raven_v3 (current RAVEN: frozen D3/N9 + qq_bias tree attention + all overhead fast
paths), target (autoregressive, no speculation).

The default W4A16 checkpoint is a round-to-nearest GPTQ-format proxy; set
SUITE_W4A16_MODEL=models/FastMTP-W4A16-GPTQ for the calibrated one (2026-11-13).
SUITE_ENGINE_OPT=1 (2026-11-14) runs every method with full CUDA graphs and spin waiting,
RAVEN additionally with capturable tree attention and async host reads (token-identical
to eager).  Round logging adds a small per-round copy for all
speculative methods alike.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_day_20261111 as day  # noqa: E402
from remtp.benchmark import fetch_metrics, metric_delta  # noqa: E402
from remtp.gsm8k_benchmark import benchmark_messages  # noqa: E402
from remtp.humaneval_benchmark import PROMPT_TEMPLATE as HUMAN_PROMPT  # noqa: E402
from remtp.humaneval_benchmark import load_humaneval, sanitize_generation  # noqa: E402

BENCH = day.BENCH
W4A16_MODEL = Path(os.environ.get("SUITE_W4A16_MODEL", ROOT / "models/FastMTP-W4A16-RTN"))
W4A16_ENV = {"MODEL_PATH": str(W4A16_MODEL), "QUANTIZATION": "gptq_marlin", "LOAD_FORMAT": "auto"}
DROP = ("REMTP_TARGET_BNB_ROW_BLOCK", "REMTP_TARGET_BNB_UNIFORM_GEMM")
MARLIN = "Using MarlinLinearKernel for GPTQMarlinLinearMethod"
DRAFT_BF16 = "[ReMTP][Jetson] MiMo draft BF16 constructor installed"
ROUNDLOG = "[ReMTP][RoundLog] per-round JSONL enabled"
SD = ("native", "cactus", "spec_cascade", "raven_v3", "raven_v4safe", "raven_v4gated")
# RAVEN v4 (2026-11-13): target-anchored relaxation.  Relaxation only toward drafted
# tokens with target probability >= 1%; elsewhere exact.  "safe": acceptance never
# above Cactus's on the same token (max ratio 1).  "gated": Cactus's realised TV at
# gated positions moves to supported ones, realised block TV <= Cactus's.
V4_GATE = {"REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_PROB": "0.01"}
# Batched per-level candidate selection: token-identical outputs, fewer host transfers.
V4SAFE_FLAGS = {**day.V3_FLAGS, **V4_GATE, "REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MAX_RATIO": "1.0",
                "REMTP_DYNAMIC_TREE_BATCH_SELECT": "1"}
V4GATED_FLAGS = {**day.V3_FLAGS, **V4_GATE, "REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_POOL": "gated",
                 "REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MIX": "1.0", "REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MAX_RATIO": "1.5"}
KINDS = {
    "native": dict(env="native/attempt_1", flags={},
                   markers=["[ReMTP][FastMTPVerified] method=native worker=active", day.NUMERICS]),
    "cactus": dict(env="cactus/attempt_1", flags={},
                   markers=["[ReMTP][FastMTPVerified] method=cactus worker=active", day.NUMERICS]),
    "spec_cascade": dict(env="spec_cascade/attempt_1", flags={},
                         markers=["[ReMTP][FastMTPVerified] method=spec_cascade worker=active", day.NUMERICS]),
    "raven_v3": dict(env="adaptive_original/attempt_2", flags=day.V3_FLAGS, markers=list(day.V3_MARKERS)),
    "raven_v4safe": dict(env="adaptive_original/attempt_2", flags=V4SAFE_FLAGS,
                         markers=list(day.V3_MARKERS) + ["target_gate_min_p=0.01 ", "adaptive_max_ratio=1 ",
                                                         "adaptive_pool=nominal ",
                                                         "[ReMTP][DynamicTree] batched candidate selection"]),
    "raven_v4gated": dict(env="adaptive_original/attempt_2", flags=V4GATED_FLAGS,
                          markers=list(day.V3_MARKERS) + ["target_gate_min_p=0.01 ", "adaptive_max_ratio=1.5 ",
                                                          "adaptive_pool=gated ", "adaptive_mix=1 "]),
    "target": dict(env="native/attempt_1", flags={"METHOD": "target"},
                   markers=["[ReMTP][FastMTPVerified] method=target speculative_decoding=off", day.NUMERICS]),
}
LABEL = {"native": "Native", "cactus": "Cactus", "spec_cascade": "SpecCascade", "raven_v3": "RAVEN v3",
         "raven_v4safe": "RAVEN v4-safe", "raven_v4gated": "RAVEN v4-gated",
         "target": "AR（无推测）"}


def power_mode() -> str:
    try:
        return subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as error:  # pragma: no cover
        return f"unknown ({error})"


def device_snapshot() -> dict:
    temps = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            temps[(zone / "type").read_text().strip()] = int((zone / "temp").read_text()) / 1000
        except Exception:
            pass
    gpu = next(iter(Path("/sys/class/devfreq").glob("*gpu*")), None)
    freq = (gpu / "cur_freq").read_text().strip() if gpu else None
    return dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), power_mode=power_mode(), temperatures_c=temps, gpu_cur_freq=freq)


def exact_mcnemar(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


class Suite(day.Controller):
    def start_server(self, kind: str, directory: Path) -> None:
        self.stop_server()
        if self.port_open():
            raise RuntimeError("port 8000 occupied by an unrelated process")
        spec = KINDS[kind]
        directory.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("REMTP_", "VLLM_"))}
        env.update(json.loads((BENCH / "servers" / spec["env"] / "effective_env.json").read_text()))
        env.update(spec["flags"])
        env.update(W4A16_ENV)
        for key in DROP:
            env.pop(key, None)
        markers = list(spec["markers"]) + [MARLIN]
        if os.environ.get("SUITE_ENGINE_OPT") == "1":
            # 2026-11-14: same engineering configuration for every method -- full CUDA
            # graphs (no torch.compile), spin waiting; RAVEN additionally its capturable
            # tree attention and async host reads.  Outputs are token-identical to eager.
            tree = kind.startswith("raven")
            sizes = list(range(1, 11)) if tree else [1] if kind == "target" else [4]
            env.update({"ENFORCE_EAGER": "0", "REMTP_CUDA_SPIN_WAIT": "1",
                        "REMTP_COMPILATION_CONFIG": json.dumps({
                            "mode": 0, "cudagraph_mode": "FULL" if tree else "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": sizes})})
            markers += ["Graph capturing finished", "[ReMTP][CudaSpinWait] primary context sched=SPIN"]
            if tree:
                env.update({"REMTP_TREE_CUDAGRAPH": "1", "REMTP_DYNAMIC_TREE_ASYNC_HOST": "1"})
                markers += ["[ReMTP][DynamicTree] tree attention CUDA graphs", "[ReMTP][DynamicTree] async host reads"]
        if kind != "target":
            env["REMTP_JETSON_BNB_DRAFT"] = "1"
            env["REMTP_ROUND_LOG_JSONL"] = str(directory / "rounds.jsonl")
            markers += [DRAFT_BF16, ROUNDLOG]
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        (directory / "effective_env.json").write_text(json.dumps(
            {k: v for k, v in env.items() if k.startswith(("REMTP_", "VLLM_")) or k in
             ("METHOD", "PORT", "MAX_MODEL_LEN", "KV_CACHE_MEMORY_BYTES", "QUANTIZATION", "LOAD_FORMAT",
              "MTP_TOKENS", "TARGET_ATTENTION_BACKEND", "ENABLE_PREFIX_CACHING", "MODEL_PATH")}, indent=2))
        (directory / "device.json").write_text(json.dumps(device_snapshot(), indent=2))
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
                with urllib.request.urlopen(day.BASE + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(3)
        else:
            raise RuntimeError("server did not become healthy")
        with log_path.open() as handle:
            handle.seek(offset)
            text = handle.read()
        missing = [m for m in markers if m not in text]
        unexpected = ["canonical tree attention"] if kind.startswith("raven") and "canonical ancestor KV reads" in text else []
        if missing or unexpected:
            raise RuntimeError(f"startup markers wrong for {kind}: missing={missing} unexpected={unexpected}")
        self.post(dict(model="TencentBAC/FastMTP", messages=[{"role": "user", "content": "Calculate 2 + 2."}],
                       chat_template_kwargs={"enable_thinking": False}, temperature=0.6, seed=1, max_tokens=8), 600)
        self.log(f"server up: {kind} pid={self.proc.pid}")

    def suite_phase(self, kind: str, run: int, limit: int | None) -> None:
        name = f"humaneval_{kind}_run{run}"
        directory = self.out / name
        directory.mkdir(parents=True, exist_ok=True)
        reference = [json.loads(l) for l in (BENCH / "humaneval/adaptive_original/requests.jsonl").open()]
        rows = {r["task_id"]: r for r in load_humaneval(BENCH / "frozen_data/humaneval/HumanEval.jsonl.gz")}
        if limit:
            reference = reference[:limit]
        max_tokens = 64 if self.smoke else getattr(self, "max_tokens", 2048)
        done_path = directory / "requests.jsonl"
        done: dict = {}
        if done_path.exists():
            for line in done_path.open():
                rec = json.loads(line)
                done[rec["sample_id"]] = rec
        pending = [r for r in reference if r["sample_id"] not in done]
        self.log(f"{name}: {len(reference)} items, {len(done)} already done")
        if pending:
            self.start_server(kind, directory)
        for ref in pending:
            if self.remaining_hours() < 0:
                self.note(f"{name}: 到达截止时间，生成在 {len(done)}/{len(reference)} 处停止")
                break
            row = rows[ref["task_id"]]
            seed = ref["seed"] + (run - 1)
            payload = dict(model="TencentBAC/FastMTP",
                           messages=benchmark_messages(HUMAN_PROMPT.format(prompt=row["prompt"]), empty_system_prompt=True),
                           chat_template_kwargs={"enable_thinking": False}, temperature=0.6, seed=seed,
                           max_tokens=max_tokens, return_token_ids=True)
            self.status(phase=name, done=len(done), total=len(reference), current=ref["sample_id"])
            before = fetch_metrics(day.BASE, 30)
            started = time.time()
            t0 = time.perf_counter()
            value = self.post(payload, 5400)
            elapsed = time.perf_counter() - t0
            ended = time.time()
            after = fetch_metrics(day.BASE, 30)
            choice = value["choices"][0]
            text = choice["message"]["content"] or ""
            m = metric_delta(before, after)
            rounds = m["vllm:spec_decode_num_drafts_total"]
            candidate, mode = sanitize_generation(text, row["entry_point"])
            rec = dict(task="humaneval", method=kind, run=run, sample_id=ref["sample_id"], task_id=row["task_id"],
                       question_id=ref["question_id"], seed=seed,
                       prompt_matches_benchmark=(value.get("prompt_token_ids") == ref["prompt_token_ids"]),
                       prompt_tokens=len(value.get("prompt_token_ids") or []),
                       output_tokens=int(value["usage"]["completion_tokens"]), finish_reason=choice["finish_reason"],
                       client_seconds=elapsed, started_at=started, ended_at=ended, metrics=m, rounds=rounds,
                       output=text, output_token_ids=choice.get("token_ids"), candidate=candidate,
                       candidate_mode=mode, correct=None)
            done[rec["sample_id"]] = rec
            with done_path.open("a") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            per_round = m["vllm:request_decode_time_seconds_sum"] / rounds if rounds else None
            self.log(f"{name} [{len(done)}/{len(reference)}] {rec['sample_id']} out={rec['output_tokens']} {elapsed:.0f}s "
                     + (f"s/round={per_round:.4f} tok/round={m['vllm:generation_tokens_total'] / rounds:.3f} " if rounds else "")
                     + f"finish={rec['finish_reason']} prompt_ok={rec['prompt_matches_benchmark']}")
            if len(done) % 10 == 0:
                self.suite_report()
        self.stop_server()                       # Docker execution never overlaps GPU generation.
        (directory / "device_after.json").write_text(json.dumps(device_snapshot(), indent=2))
        self.evaluate_humaneval(directory, done, reference, rows)
        self.suite_report()

    # ----- analysis --------------------------------------------------------
    def load_phase(self, directory: Path) -> list[dict]:
        recs = [json.loads(l) for l in (directory / "requests.jsonl").open()] if (directory / "requests.jsonl").exists() else []
        verdict = {}
        if (directory / "evaluation.jsonl").exists():
            for line in (directory / "evaluation.jsonl").open():
                item = json.loads(line)
                verdict[item["sample_id"]] = (item["correct"], item.get("status"))
        for rec in recs:
            rec["correct"], rec["eval_status"] = verdict.get(rec["sample_id"], (None, None))
        return recs

    def suite_report(self) -> None:
        phases = {}
        for directory in sorted(self.out.glob("humaneval_*_run*")):
            kind, run = directory.name[len("humaneval_"):].rsplit("_run", 1)
            recs = self.load_phase(directory)
            if recs:
                phases[(kind, int(run))] = recs
        if not phases:
            return
        summary = {f"{k}:{r}": day.totals(v) for (k, r), v in phases.items()}
        for (kind, run), recs in phases.items():
            known = [x for x in recs if x["correct"] is not None]
            summary[f"{kind}:{run}"].update(
                run=run, method=kind,
                stop_correct=sum(bool(x["correct"]) and x["finish_reason"] == "stop" for x in known),
                stop=sum(x["finish_reason"] == "stop" for x in recs),
                syntax_error=sum(x["eval_status"] == "syntax_error" for x in known))
        lines = ["# W4A16 + Marlin、MAXN：全部方法 HumanEval-100 重复运行", "",
                 f"更新时间 {time.strftime('%Y-%m-%d %H:%M:%S')}；功耗模式：{power_mode().splitlines()[0] if power_mode() else 'unknown'}。", "",
                 f"目标模型 `{W4A16_MODEL.name}`（GPTQ 格式 W4A16，Marlin），MTP 层 BF16；输出上限 "
                 f"{getattr(self, 'max_tokens', 2048)} token。run 1 使用冻结种子（task×10+42），run r 为冻结种子 + (r−1)。", "",
                 "## 每次运行", "",
                 "| 方法 | run | 完成 | pass@1 | 未截断判对 | 截断 | 语法错误 | E2E tok/s | decode tok/s | MAL | 每轮秒 | 平均输出 token | 平均整题秒 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        fmt = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"
        order = [k for k in ("target",) + SD if any(key[0] == k for key in phases)]
        for kind in order:
            for run in sorted(r for k, r in phases if k == kind):
                s = summary[f"{kind}:{run}"]
                q = "待判分" if not s["evaluated"] else f"{s['correct']}/{s['evaluated']}"
                lines.append(f"| {LABEL[kind]} | {run} | {s['requests']} | {q} | {s['stop_correct']}/{s['stop']} | {s['truncated']} | "
                             f"{s['syntax_error']} | {fmt(s['e2e_output_tok_s'])} | {fmt(s['decode_tok_s'])} | "
                             f"{fmt(s['mean_acceptance_length'])} | {fmt(s['seconds_per_round'], 4)} | {s['mean_output_tokens']:.0f} | {s['mean_seconds']:.1f} |")
        # pooled across runs
        lines += ["", "## 合并各次运行", "",
                  "| 方法 | 运行数 | pass@1（合计） | 平均 pass@1 | E2E tok/s | decode tok/s | MAL | 每轮秒 | 截断合计 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        pooled = {}
        for kind in order:
            recs = [x for (k, _), v in phases.items() if k == kind for x in v]
            t = day.totals(recs)
            runs = sorted(r for k, r in phases if k == kind)
            accs = [summary[f"{kind}:{r}"]["accuracy"] for r in runs if summary[f"{kind}:{r}"]["accuracy"] is not None]
            pooled[kind] = dict(t, runs=runs, mean_accuracy=sum(accs) / len(accs) if accs else None)
            lines.append(f"| {LABEL[kind]} | {len(runs)} | {t['correct']}/{t['evaluated']} | "
                         f"{fmt(pooled[kind]['mean_accuracy'] and 100 * pooled[kind]['mean_accuracy'], 1)}% | {fmt(t['e2e_output_tok_s'])} | "
                         f"{fmt(t['decode_tok_s'])} | {fmt(t['mean_acceptance_length'])} | {fmt(t['seconds_per_round'], 4)} | {t['truncated']} |")
        # paired comparisons per run (same task, same seed)
        lines += ["", "## 配对比较（同题同种子，精确 McNemar）", "",
                  "| 比较 | run | 仅前者对 | 仅后者对 | p 值 |", "| --- | ---: | ---: | ---: | ---: |"]
        pairs = [("raven_v4safe", "cactus"), ("raven_v4gated", "cactus"), ("raven_v4safe", "raven_v3"),
                 ("raven_v4gated", "raven_v3"),
                 ("raven_v3", "cactus"), ("raven_v3", "native"), ("cactus", "native"), ("spec_cascade", "native"),
                 ("raven_v3", "spec_cascade"), ("native", "target"), ("cactus", "target")]
        paired = {}
        for a, b in pairs:
            total_a = total_b = 0
            for run in sorted({r for k, r in phases if k == a} & {r for k, r in phases if k == b}):
                va = {x["sample_id"]: x["correct"] for x in phases[(a, run)] if x["correct"] is not None}
                vb = {x["sample_id"]: x["correct"] for x in phases[(b, run)] if x["correct"] is not None}
                common = set(va) & set(vb)
                if not common:
                    continue
                only_a = sum(va[i] and not vb[i] for i in common)
                only_b = sum(vb[i] and not va[i] for i in common)
                total_a += only_a
                total_b += only_b
                paired[f"{a}-{b}:{run}"] = dict(only_a=only_a, only_b=only_b, n=len(common), p=exact_mcnemar(only_a, only_b))
                lines.append(f"| {LABEL[a]} 对 {LABEL[b]} | {run} | {only_a} | {only_b} | {exact_mcnemar(only_a, only_b):.3f} |")
            if total_a + total_b:
                lines.append(f"| {LABEL[a]} 对 {LABEL[b]} | 合计 | {total_a} | {total_b} | （各次运行同题相关，仅供参考） |")
        # run-to-run agreement of each method with itself
        lines += ["", "## 同一方法两次运行的一致性（衡量采样噪声）", "",
                  "| 方法 | 两次都对 | 仅 run1 对 | 仅 run2 对 | 都错 |", "| --- | ---: | ---: | ---: | ---: |"]
        for kind in order:
            if (kind, 1) in phases and (kind, 2) in phases:
                v1 = {x["sample_id"]: x["correct"] for x in phases[(kind, 1)] if x["correct"] is not None}
                v2 = {x["sample_id"]: x["correct"] for x in phases[(kind, 2)] if x["correct"] is not None}
                common = set(v1) & set(v2)
                if common:
                    lines.append(f"| {LABEL[kind]} | {sum(v1[i] and v2[i] for i in common)} | {sum(v1[i] and not v2[i] for i in common)} | "
                                 f"{sum(v2[i] and not v1[i] for i in common)} | {sum(not v1[i] and not v2[i] for i in common)} |")
        if "target" in pooled or "native" in pooled:
            base = pooled.get("target") or pooled.get("native")
            name = "AR" if "target" in pooled else "Native"
            lines += ["", f"## 相对 {name} 的速度（合并各次运行）", ""]
            for kind in order:
                if base.get("e2e_output_tok_s") and pooled[kind].get("e2e_output_tok_s"):
                    lines.append(f"- {LABEL[kind]}：E2E {pooled[kind]['e2e_output_tok_s'] / base['e2e_output_tok_s']:.2f} 倍，"
                                 f"decode {pooled[kind]['decode_tok_s'] / base['decode_tok_s']:.2f} 倍")
        if self.notes:
            lines += ["", "## 运行备注", ""] + [f"- {n}" for n in self.notes]
        (self.out / "suite_summary.json").write_text(json.dumps(dict(per_run=summary, pooled=pooled, paired=paired),
                                                                ensure_ascii=False, indent=1))
        tmp = self.out / "suite_report.md.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(self.out / "suite_report.md")

    def write_report(self) -> None:          # day.Controller hooks call this
        self.suite_report()

    def run_suite(self, phases: list[str], limit: int | None) -> None:
        try:
            for phase in phases:
                kind, run = phase.split(":")
                if kind not in KINDS:
                    raise ValueError(f"unknown kind {kind}")
                try:
                    self.suite_phase(kind, int(run), limit)
                except Exception as error:
                    self.stop_server()
                    self.note(f"{phase} 失败：{type(error).__name__}: {error}")
        finally:
            self.stop_server()
            self.suite_report()
            self.status(phase="finished")
            self.log("suite_report.md written")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phases", default="native:1,cactus:1,spec_cascade:1,raven_v3:1,"
                                            "raven_v3:2,spec_cascade:2,cactus:2,native:2,target:1")
    parser.add_argument("--deadline-hours", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks (smoke)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-mode", default="MAXN")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        Suite(args.out, time.time(), False).suite_report()
        return 0
    lock = (args.out / ".controller.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another suite controller holds the lock; exiting without changes", flush=True)
        return 1
    mode = power_mode()
    if args.require_mode and f"NV Power Mode: {args.require_mode}" not in mode:
        print(f"power mode is not {args.require_mode}: {mode!r}; exiting", flush=True)
        return 1
    if Suite.port_open():
        print("port 8000 is already in use; exiting without changes", flush=True)
        return 1
    if not (W4A16_MODEL / "config.json").exists():
        print(f"missing W4A16 checkpoint {W4A16_MODEL}", flush=True)
        return 1
    suite = Suite(args.out, time.time() + args.deadline_hours * 3600, args.smoke)
    suite.max_tokens = args.max_tokens
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout
    (args.out / "protocol.json").write_text(json.dumps(dict(
        started=time.strftime("%Y-%m-%d %H:%M:%S"), phases=args.phases.split(","), limit=args.limit, smoke=args.smoke,
        power_mode=mode, w4a16_model=str(W4A16_MODEL), w4a16_quantization=json.loads((W4A16_MODEL / "config.json").read_text())
        .get("quantization_config"), frozen_benchmark=str(BENCH), dataset="HumanEval-100 (frozen 100+100 selection)",
        temperature=0.6, max_tokens=args.max_tokens, seeds="frozen task*10+42 + (run-1)", git_head=commit, git_dirty=dirty.splitlines(),
        device=device_snapshot()), ensure_ascii=False, indent=2))
    suite.log(f"start phases={args.phases} limit={args.limit} smoke={args.smoke} mode={mode.splitlines()[0] if mode else ''}")
    suite.run_suite(args.phases.split(","), args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
