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
    frozen = json.loads((BENCH / "servers/adaptive_original/attempt_2/effective_env.json").read_text())
    env.update(frozen)
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
    if os.environ.get("AB_PROFILE") == "1":
        # Bounded component profile (32 observed rounds); not a performance run.
        env["REMTP_RUNTIME_PROFILE_JSONL"] = str(directory / "profile.jsonl")
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    (directory / "effective_env.json").write_text(json.dumps(
        {k: v for k, v in env.items() if k in frozen or k == "PYTHONPATH" or k.startswith("REMTP_")}, indent=2))
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
    if fast:
        markers += ["[ReMTP][DynamicTree] fast candidate selection", "[ReMTP][DynamicTree] fast residual-hit verification"]
    if commit:
        markers += ["[ReMTP][DynamicTree] fused KV path commit across layers"]
    if "inplace" in parts:
        markers += ["[ReMTP][DynamicTree] in-place ordered KV path commit"]
    if "meta" in parts:
        markers += ["[ReMTP][DynamicTree] cached host-built tree masks"]
    if "skip" in parts:
        markers += ["[ReMTP][DynamicTree] dead draft KV skipped"]
    if "pathfirst" in parts:
        markers += ["[ReMTP][DynamicTree] first MTP pass reordered to the accepted path"]
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
            before = fetch_metrics(BASE, 30)
            t0 = time.perf_counter()
            value = post(payload, 5400)
            elapsed = time.perf_counter() - t0
            after = fetch_metrics(BASE, 30)
            choice = value["choices"][0]
            m = metric_delta(before, after)
            rounds = m["vllm:spec_decode_num_drafts_total"]
            out_ids = choice.get("token_ids") or []
            rec = dict(question_id=qid, seed=ref["seed"], canonical=canonical,
                       prompt_tokens=len(value.get("prompt_token_ids") or []),
                       prompt_matches_benchmark=(value.get("prompt_token_ids") == ref["prompt_token_ids"]),
                       output_tokens=int(value["usage"]["completion_tokens"]), finish_reason=choice["finish_reason"],
                       client_seconds=elapsed, metrics=m,
                       decode_seconds=m["vllm:request_decode_time_seconds_sum"], rounds=rounds,
                       seconds_per_round=m["vllm:request_decode_time_seconds_sum"] / rounds if rounds else None,
                       tokens_per_round=m["vllm:generation_tokens_total"] / rounds if rounds else None,
                       accepted_per_round=m["vllm:spec_decode_num_accepted_tokens_total"] / rounds if rounds else None,
                       nodes_per_round=m["vllm:spec_decode_num_draft_tokens_total"] / rounds if rounds else None,
                       decode_tok_s=m["vllm:generation_tokens_total"] / m["vllm:request_decode_time_seconds_sum"],
                       e2e_tok_s=int(value["usage"]["completion_tokens"]) / elapsed,
                       benchmark_output_tokens=ref["output_tokens"], benchmark_seconds_per_round=ref["metrics"]["vllm:request_decode_time_seconds_sum"] / ref["metrics"]["vllm:spec_decode_num_drafts_total"],
                       output_equals_benchmark=(out_ids == ref["output_token_ids"]),
                       first_divergence_vs_benchmark=first_divergence(out_ids, ref["output_token_ids"]),
                       output_token_ids=out_ids, output=choice["message"]["content"],
                       mem_available_kib_before=mem0, mem_available_kib_after=mem_available_kib())
            records.append(rec)
            (directory / "requests.jsonl").open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            log(f"q{qid}: out={rec['output_tokens']} rounds={rounds:.0f} s/round={rec['seconds_per_round']:.4f} "
                f"(bench {rec['benchmark_seconds_per_round']:.4f}) tok/round={rec['tokens_per_round']:.3f} "
                f"decode={rec['decode_tok_s']:.3f} tok/s equal_to_bench={rec['output_equals_benchmark']} "
                f"first_div={rec['first_divergence_vs_benchmark']} prompt_ok={rec['prompt_matches_benchmark']}")
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
        gen, dec, rounds = tot("vllm:generation_tokens_total"), tot("vllm:request_decode_time_seconds_sum"), tot("vllm:spec_decode_num_drafts_total")
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
