"""A/B: RAVEN (dynamic_tree, frozen 100x100 config) with REMTP_CANONICAL_TREE_ATTENTION=1 vs 0.

Arm A reproduces the benchmark server environment exactly; arm B only flips the
tree-attention layout to vLLM's native single-sequence qq_bias path.  Both arms
replay the same GSM8K requests (same prompt, seed, max_tokens) and record the
/metrics deltas the benchmark used, plus token-level agreement with the
recorded benchmark outputs.  Nothing in the serving code is modified.
"""
from __future__ import annotations

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
from remtp.gsm8k_benchmark import PROMPT_TEMPLATE, benchmark_messages  # noqa: E402

BENCH = ROOT / "results/raven_full_100x100_20261010"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/canonical_attention_ab_20261110"
QIDS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["17", "353", "856"])]
ARMS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["1", "0"]
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
MAX_TOKENS = 2048


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def mem_available_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return -1


def load_reference():
    rows = [json.loads(l) for l in (BENCH / "gsm8k/adaptive_original/requests.jsonl").open()]
    return {r["question_id"]: r for r in rows}


def gsm8k_rows():
    return [json.loads(l) for l in (ROOT / "data/gsm8k/test.jsonl").open()]


def gpu_freq_ms():
    """Cumulative ms per GPU frequency from devfreq trans_stat (clocks are not locked)."""
    path = Path("/sys/class/devfreq/17000000.gpu/trans_stat")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines[2:]:
        parts = line.replace("*", " ").replace(":", " ").split()
        if len(parts) >= 3 and parts[0].isdigit():
            out[int(parts[0]) // 1000000] = int(parts[-1])
    return out


def cpu_freq_ms():
    """Cumulative ms per CPU frequency (MHz) per cpufreq policy (governor: schedutil)."""
    out = {}
    for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        try:
            rows = (policy / "stats/time_in_state").read_text().split()
        except OSError:
            continue
        # time_in_state is in units of 10 ms
        out[policy.name] = {int(f) // 1000: int(t) * 10 for f, t in zip(rows[::2], rows[1::2])}
    return out


def mean_mhz(delta):
    total = sum(delta.values())
    return round(sum(f * t for f, t in delta.items()) / total) if total else None


def port_open():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def start_server(arm: str, directory: Path):
    """``arm`` is the canonical flag ("1"/"0"), optionally followed by "+fast" and/or "+commit"."""
    parts = arm.split("+")
    canonical, fast, commit = parts[0], "fast" in parts, "commit" in parts
    if port_open():
        raise RuntimeError("port 8000 occupied")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REMTP_", "VLLM_"))}
    # "+cactus" (2026-11-12): frozen chain Cactus service instead of RAVEN.
    cactus = "cactus" in parts
    # "+native" / "+target" (2026-11-14): frozen chain Native service / AR target only.
    native, target = "native" in parts, "target" in parts
    chain = cactus or native or target
    frozen_env = ("cactus/attempt_1" if cactus else "native/attempt_1" if (native or target)
                  else "adaptive_original/attempt_2")
    frozen = json.loads((BENCH / f"servers/{frozen_env}/effective_env.json").read_text())
    env.update(frozen)
    if target:
        env["METHOD"] = "target"
    # "+w4a16" (2026-11-12): GPTQ-format W4A16 target through vLLM gptq_marlin;
    # BF16 MTP layer as before. Drops only the two bitsandbytes-specific switches.
    w4a16 = "w4a16" in parts
    if w4a16:
        env["MODEL_PATH"] = os.environ.get("AB_W4A16_MODEL", str(ROOT / "models/FastMTP-W4A16-RTN"))
        env["QUANTIZATION"] = "gptq_marlin"
        env["LOAD_FORMAT"] = "auto"
        env["REMTP_JETSON_BNB_DRAFT"] = "1"
        env.pop("REMTP_TARGET_BNB_ROW_BLOCK", None)
        env.pop("REMTP_TARGET_BNB_UNIFORM_GEMM", None)
    env["REMTP_CANONICAL_TREE_ATTENTION"] = canonical
    if fast:
        env["REMTP_DYNAMIC_TREE_FAST_SELECT"] = "1"
        env["REMTP_DYNAMIC_TREE_FAST_VERIFY"] = "1"
    if commit:
        env["REMTP_DYNAMIC_TREE_FAST_COMMIT"] = "1"
    # Third layer (2026-11-11): "+inplace", "+meta", "+skip"; "+pathfirst" changes proposals.
    if "inplace" in parts:
        env["REMTP_DYNAMIC_TREE_FAST_COMMIT"] = "2"
    if "meta" in parts:
        env["REMTP_DYNAMIC_TREE_FAST_METADATA"] = "1"
    if "skip" in parts:
        env["REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV"] = "1"
    if "pathfirst" in parts:
        env["REMTP_DYNAMIC_TREE_PATH_ORDERED_FIRST_PASS"] = "1"
    # RAVEN v4 (2026-11-13): target-support gate p(y) >= 1%, per-position cap = Cactus.
    if "v4safe" in parts:
        env["REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_PROB"] = "0.01"
        env["REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MAX_RATIO"] = "1.0"
    # One host transfer per tree level in the proposal (2026-11-13).
    if "batch" in parts:
        env["REMTP_DYNAMIC_TREE_BATCH_SELECT"] = "1"
    if "roundlog" in parts:
        # Per-round JSONL observer (2026-11-12); must not change outputs.
        env["REMTP_ROUND_LOG_JSONL"] = str(directory / "rounds.jsonl")
    # "+cg" (2026-11-13): full CUDA graphs for uniform decode/verify steps, no torch.compile
    # (same kernels as eager; prefill stays eager).  Capture sizes from AB_CG_SIZES.
    env["ENFORCE_EAGER"] = "0" if "cg" in parts else "1"
    # Chain (Cactus): every verify step has 1+3 tokens -> FULL_DECODE_ONLY at size 4.
    # Tree (RAVEN): 1+n tokens with n <= N varying per round -> FULL at sizes 1..N+1
    # (one request per batch); remtp.tree_cudagraph makes TREE_ATTN capturable.
    if "cg" in parts:
        if chain:
            mode, default_sizes = "FULL_DECODE_ONLY", ("1" if target else "4")
        else:
            mode = "FULL"
            default_sizes = ",".join(str(x) for x in range(1, int(env["REMTP_DYNAMIC_TREE_MAX_NODES"]) + 2))
            env["REMTP_TREE_CUDAGRAPH"] = "1"
        sizes = [int(x) for x in os.environ.get("AB_CG_SIZES", default_sizes).split(",")]
        env["REMTP_COMPILATION_CONFIG"] = json.dumps(
            {"mode": 0, "cudagraph_mode": os.environ.get("AB_CG_MODE", mode), "cudagraph_capture_sizes": sizes})
    else:
        env.pop("REMTP_COMPILATION_CONFIG", None)
    # Tree sub-metadata from host scalars, no .item() syncs (eager); "+hostcheck" validates.
    if "hostmeta" in parts:
        env["REMTP_TREE_HOST_METADATA"] = "1"
    if "hostcheck" in parts:
        env["REMTP_TREE_HOST_METADATA_CHECK"] = "1"
    # "+spin" (2026-11-14): host waits poll instead of sleeping (keeps schedutil clocks up).
    if "spin" in parts:
        env["REMTP_CUDA_SPIN_WAIT"] = "1"
    # "+async" (2026-11-14): no blocking host reads where the value is on the host already.
    if "async" in parts:
        env["REMTP_DYNAMIC_TREE_ASYNC_HOST"] = "1"
    # "+gpubuild" (2026-11-15): device-side tree construction; "+gpucheck" recomputes
    # every level on the host and logs differences.
    if "gpubuild" in parts:
        env["REMTP_DYNAMIC_TREE_GPU_BUILD"] = "1"
    if "gpucheck" in parts:
        env["REMTP_DYNAMIC_TREE_GPU_BUILD_CHECK"] = str(directory / "gpu_build_check.jsonl")
    # "+unfused" (2026-11-15): device tree with the op-by-op decision step instead of the Triton one.
    if "unfused" in parts:
        env["REMTP_DYNAMIC_TREE_GPU_BUILD_FUSED"] = "0"
    # "+prelaunch" (2026-11-15): trunk verification values queued before the first host read.
    if "prelaunch" in parts:
        env["REMTP_DYNAMIC_TREE_VERIFY_PRELAUNCH"] = "1"
    if os.environ.get("AB_GPU_EVENTS") == "1":
        # CUDA-event GPU time per model component (2026-11-13); no added syncs.
        env["REMTP_GPU_EVENT_TIMING_JSONL"] = str(directory / "gpu_events.jsonl")
        env["REMTP_GPU_EVENT_TIMING_EVERY"] = os.environ.get("AB_GPU_EVENTS_EVERY", "100")
    if os.environ.get("AB_HOST_TIMELINE") == "1":
        # perf_counter probes per decode step (2026-11-14); no syncs, analysis only.
        env["REMTP_HOST_TIMELINE_JSONL"] = str(directory / "host_timeline.jsonl")
        env["REMTP_HOST_TIMELINE_START"] = os.environ.get("AB_HOST_TIMELINE_START", "150")
        env["REMTP_HOST_TIMELINE_STEPS"] = os.environ.get("AB_HOST_TIMELINE_STEPS", "60")
        env["REMTP_HOST_TIMELINE_SYNC"] = os.environ.get("AB_HOST_TIMELINE_SYNC", "0")
        env["REMTP_HOST_TIMELINE_FINE"] = os.environ.get("AB_HOST_TIMELINE_FINE", "0")
    if os.environ.get("AB_TORCH_TRACE") == "1":
        # torch.profiler timeline for GPU-idle attribution (2026-11-13); analysis only.
        env["REMTP_TORCH_TRACE_JSON"] = str(directory / "torch_trace.json")
        env["REMTP_TORCH_TRACE_START"] = os.environ.get("AB_TORCH_TRACE_START", "120")
        env["REMTP_TORCH_TRACE_ROUNDS"] = os.environ.get("AB_TORCH_TRACE_ROUNDS", "24")
    if os.environ.get("AB_PROFILE") == "1":
        # Bounded component profile (32 observed rounds); not a performance run.
        env["REMTP_RUNTIME_PROFILE_JSONL"] = str(directory / "profile.jsonl")
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    (directory / "effective_env.json").write_text(json.dumps(
        {k: v for k, v in env.items() if k in frozen or k in ("PYTHONPATH", "MODEL_PATH") or k.startswith("REMTP_")}, indent=2))
    logf = (directory / "server.log").open("w")
    proc = subprocess.Popen([str(ROOT / "scripts/serve_fastmtp_jetson.sh")], cwd=ROOT, env=env,
                            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 1200
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}")
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(3)
    else:
        raise RuntimeError("server did not become healthy")
    text = (directory / "server.log").read_text()
    markers = ["[ReMTP][FastMTPVerified] method=dynamic_tree worker=active",
               "[ReMTP][TargetNumerics] unified 2D probability/value product FP32 (all methods)",
               "[ReMTP][TargetNumerics] sm87 FP4: uniform dequant + BF16 linear (all methods)",
               "D=3 N_max=9 "]
    if chain:
        name = "cactus worker=active" if cactus else "native worker=active" if native else "target speculative_decoding=off"
        markers = [f"[ReMTP][FastMTPVerified] method={name}",
                   "[ReMTP][TargetNumerics] unified 2D probability/value product FP32 (all methods)",
                   "[ReMTP][TargetNumerics] sm87 FP4: uniform dequant + BF16 linear (all methods)"]
    if w4a16:
        markers = [m for m in markers if "sm87 FP4" not in m]
        if not target:
            markers += ["[ReMTP][Jetson] MiMo draft BF16 constructor installed"]
    if fast:
        markers += ["[ReMTP][DynamicTree] fast candidate selection", "[ReMTP][DynamicTree] fast residual-hit verification"]
    if commit:
        markers += ["[ReMTP][DynamicTree] fused KV path commit across layers"]
    if "inplace" in parts:
        markers += ["[ReMTP][DynamicTree] in-place ordered KV path commit"]
    if "meta" in parts:
        markers += ["[ReMTP][DynamicTree] cached host-built tree masks"]
    if "v4safe" in parts:
        markers += ["target_gate_min_p=0.01 ", "adaptive_max_ratio=1 "]
    if "batch" in parts:
        markers += ["[ReMTP][DynamicTree] batched candidate selection"]
    if "skip" in parts:
        markers += ["[ReMTP][DynamicTree] dead draft KV skipped"]
    if "pathfirst" in parts:
        markers += ["[ReMTP][DynamicTree] first MTP pass reordered to the accepted path"]
    if "roundlog" in parts:
        markers += ["[ReMTP][RoundLog] per-round JSONL enabled"]
    if "cg" in parts:
        markers += ["Graph capturing finished"]
        if not chain:
            markers += ["[ReMTP][DynamicTree] tree attention CUDA graphs"]
    if "hostmeta" in parts or ("cg" in parts and not chain):
        markers += ["[ReMTP][DynamicTree] tree attention sub-metadata from host scalars"]
    if "hostcheck" in parts:
        markers += ["(no .item() syncs) + device check"]
    if "spin" in parts:
        markers += ["[ReMTP][CudaSpinWait] primary context sched=SPIN"]
    if "async" in parts:
        markers += ["[ReMTP][DynamicTree] async host reads"]
    if "gpubuild" in parts:
        markers += ["[ReMTP][DynamicTree] device-side tree construction"]
    if "prelaunch" in parts:
        markers += ["[ReMTP][DynamicTree] trunk verification prelaunched"]
    missing = [m for m in markers if m not in text]
    canon_marker = "[ReMTP][TargetNumerics] canonical ancestor KV reads, zero KV copies" in text
    if missing or canon_marker != (canonical == "1"):
        raise RuntimeError(f"startup markers wrong: missing={missing} canonical_marker={canon_marker}")
    log(f"server up (arm={arm}) pid={proc.pid}")
    return proc


def stop_server(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(60):
        if proc.poll() is not None:
            break
        time.sleep(1)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    for _ in range(60):
        if not port_open():
            break
        time.sleep(1)
    log("server stopped")


def post(payload, timeout):
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_arm(canonical: str, reference, data):
    directory = OUT / f"canonical_{canonical}"
    directory.mkdir(parents=True, exist_ok=True)
    proc = None
    records = []
    try:
        proc = start_server(canonical, directory)
        post(dict(model="TencentBAC/FastMTP", messages=[{"role": "user", "content": "Calculate 2 + 2."}],
                  chat_template_kwargs={"enable_thinking": False}, temperature=0.6, seed=1, max_tokens=8), 600)
        log("warmup done")
        for qid in QIDS:
            row = data[qid - 1]
            ref = reference[qid]
            payload = dict(model="TencentBAC/FastMTP",
                           messages=benchmark_messages(PROMPT_TEMPLATE.format(question=row["question"]), empty_system_prompt=True),
                           chat_template_kwargs={"enable_thinking": False}, temperature=0.6, seed=ref["seed"],
                           max_tokens=MAX_TOKENS, return_token_ids=True)
            mem0 = mem_available_kib()
            freq0 = gpu_freq_ms()
            cpu0 = cpu_freq_ms()
            before = fetch_metrics(BASE, 30)
            t0 = time.perf_counter()
            value = post(payload, 5400)
            elapsed = time.perf_counter() - t0
            after = fetch_metrics(BASE, 30)
            freq1 = gpu_freq_ms()
            cpu1 = cpu_freq_ms()
            cpu_delta = {pol: {f: cpu1[pol][f] - cpu0.get(pol, {}).get(f, 0) for f in cpu1[pol]} for pol in cpu1}
            choice = value["choices"][0]
            m = metric_delta(before, after)
            # AR (target only): one decode step per generated token.
            rounds = m.get("vllm:spec_decode_num_drafts_total") or m["vllm:generation_tokens_total"]
            out_ids = choice.get("token_ids") or []
            rec = dict(question_id=qid, seed=ref["seed"], canonical=canonical,
                       prompt_tokens=len(value.get("prompt_token_ids") or []),
                       prompt_matches_benchmark=(value.get("prompt_token_ids") == ref["prompt_token_ids"]),
                       output_tokens=int(value["usage"]["completion_tokens"]), finish_reason=choice["finish_reason"],
                       client_seconds=elapsed, metrics=m,
                       decode_seconds=m["vllm:request_decode_time_seconds_sum"], rounds=rounds,
                       seconds_per_round=m["vllm:request_decode_time_seconds_sum"] / rounds if rounds else None,
                       tokens_per_round=m["vllm:generation_tokens_total"] / rounds if rounds else None,
                       accepted_per_round=m.get("vllm:spec_decode_num_accepted_tokens_total", 0) / rounds if rounds else None,
                       nodes_per_round=m.get("vllm:spec_decode_num_draft_tokens_total", 0) / rounds if rounds else None,
                       decode_tok_s=m["vllm:generation_tokens_total"] / m["vllm:request_decode_time_seconds_sum"],
                       e2e_tok_s=int(value["usage"]["completion_tokens"]) / elapsed,
                       benchmark_output_tokens=ref["output_tokens"], benchmark_seconds_per_round=ref["metrics"]["vllm:request_decode_time_seconds_sum"] / ref["metrics"]["vllm:spec_decode_num_drafts_total"],
                       output_equals_benchmark=(out_ids == ref["output_token_ids"]),
                       first_divergence_vs_benchmark=first_divergence(out_ids, ref["output_token_ids"]),
                       output_token_ids=out_ids, output=choice["message"]["content"],
                       mem_available_kib_before=mem0, mem_available_kib_after=mem_available_kib(),
                       gpu_freq_ms={f: freq1[f] - freq0.get(f, 0) for f in freq1},
                       cpu_freq_ms=cpu_delta, cpu_mean_mhz={pol: mean_mhz(d) for pol, d in cpu_delta.items()})
            records.append(rec)
            (directory / "requests.jsonl").open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            log(f"q{qid}: out={rec['output_tokens']} rounds={rounds:.0f} s/round={rec['seconds_per_round']:.4f} "
                f"(bench {rec['benchmark_seconds_per_round']:.4f}) tok/round={rec['tokens_per_round']:.3f} "
                f"decode={rec['decode_tok_s']:.3f} tok/s equal_to_bench={rec['output_equals_benchmark']} "
                f"first_div={rec['first_divergence_vs_benchmark']} prompt_ok={rec['prompt_matches_benchmark']} "
                f"gpu_MHz_ms={rec['gpu_freq_ms']} cpu_mean_MHz={rec['cpu_mean_mhz']}")
    finally:
        stop_server(proc)
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    data = gsm8k_rows()
    summary = {}
    for arm in ARMS:
        recs = run_arm(arm, reference, data)
        tot = lambda k: sum(r["metrics"][k] for r in recs)
        gen, dec = tot("vllm:generation_tokens_total"), tot("vllm:request_decode_time_seconds_sum")
        rounds = sum(r["rounds"] for r in recs)
        summary[arm] = dict(requests=len(recs), output_tokens=gen, decode_seconds=dec, rounds=rounds,
                            seconds_per_round=dec / rounds, tokens_per_round=gen / rounds, decode_tok_s=gen / dec,
                            e2e_tok_s=sum(r["output_tokens"] for r in recs) / sum(r["client_seconds"] for r in recs),
                            outputs_equal_benchmark=[r["output_equals_benchmark"] for r in recs],
                            first_divergence=[r["first_divergence_vs_benchmark"] for r in recs])
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"arm canonical={arm}: {json.dumps(summary[arm])}")
    arms = list(summary)
    for first, second in zip(arms, arms[1:]):
        a, b = summary[first], summary[second]
        log(f"RESULT {first} -> {second}: s/round {a['seconds_per_round']:.4f} -> {b['seconds_per_round']:.4f} "
            f"({b['seconds_per_round'] / a['seconds_per_round'] - 1:+.1%}); decode tok/s {a['decode_tok_s']:.3f} -> {b['decode_tok_s']:.3f} "
            f"({b['decode_tok_s'] / a['decode_tok_s'] - 1:+.1%})")


if __name__ == "__main__":
    main()
