#!/usr/bin/env python3
"""Offline analysis of the per-round JSONL logs written by ``remtp/round_logger.py``.

Usage: python scripts/analyze_suite_round_logs.py <suite_dir> [--out <json>]

For every speculative phase directory (``humaneval_<kind>_run<r>``) it joins
rounds to requests by wall-clock time, then measures how much each method's
acceptance rule departs from exact speculative sampling on the *same* drafted
token:

* evaluated position: a drafted token whose accept/reject decision mattered
  (all earlier drafts in the round were accepted);
* ``a_exact = min(1, p/q)``: exact (Native) acceptance probability of that token;
* ``a_rule``: the method's own acceptance probability when it can be computed
  (Native: a_exact; Cactus: min(1, gamma/q) with gamma = min(p + sqrt(2 delta p(1-p)), 1);
  RAVEN: the verifier's logged ``accept_probability``).  SpecCascade has no
  closed form here, so only the realised indicator is used;
* excess mass = mean(a_rule - a_exact) over evaluated positions.  Because the
  drafted token is a sample from q, this is an unbiased estimate of
  sum_y [min(q, h) - min(q, p)], the extra probability the rule moves onto
  draft tokens per position; the realised indicator gives the same estimate
  with more variance and is reported as a cross-check;
* low-support acceptances: accepted drafted tokens with p(y) below a threshold,
  per 1000 emitted tokens, both realised and expected (sum of a_rule), plus the
  part attributable to relaxation (sum of a_rule - a_exact).

Corrections and bonus tokens are not scored: Cactus and RAVEN draw corrections
from a proportionally scaled-down target (and RAVEN's anchor from the exact
target), so the relaxation enters only through draft acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

THRESHOLDS = (1e-3, 1e-2, 1e-1)
DELTA = 1.0
PHASE = re.compile(r"^humaneval_(?P<kind>.+)_run(?P<run>\d+)$")


def cactus_accept(p: float, q: float, delta: float = DELTA) -> float:
    gamma = min(p + math.sqrt(max(2.0 * delta * p * (1.0 - p), 0.0)), 1.0)
    return 1.0 if q <= 0 else min(1.0, gamma / q)


def exact_accept(p: float, q: float) -> float:
    return 1.0 if q <= 0 else min(1.0, p / q)


def load_requests(directory: Path) -> list[dict]:
    verdict = {}
    eval_path = directory / "evaluation.jsonl"
    if eval_path.exists():
        for line in eval_path.open():
            item = json.loads(line)
            verdict[item["sample_id"]] = item
    out = []
    for line in (directory / "requests.jsonl").open():
        rec = json.loads(line)
        ev = verdict.get(rec["sample_id"], {})
        out.append(dict(sample_id=rec["sample_id"], start=float(rec["started_at"]), end=float(rec["ended_at"]),
                        output_tokens=int(rec["output_tokens"]), finish=rec["finish_reason"],
                        correct=ev.get("correct"), status=ev.get("status")))
    out.sort(key=lambda r: r["start"])
    return out


def assign(requests: list[dict], t: float, cursor: int) -> tuple[int | None, int]:
    while cursor < len(requests) and t > requests[cursor]["end"] + 0.5:
        cursor += 1
    if cursor < len(requests) and requests[cursor]["start"] - 0.5 <= t:
        return cursor, cursor
    return None, cursor


def new_counter() -> dict:
    c = dict(positions=0, accepted=0, a_exact=0.0, a_rule=0.0, a_rule_known=0, rounds=0, emitted=0)
    for thr in THRESHOLDS:
        c[f"low_real_{thr:g}"] = 0          # realised accepted drafts with p < thr
        c[f"low_rule_{thr:g}"] = 0.0        # expected accepted drafts with p < thr under the rule
        c[f"low_exact_{thr:g}"] = 0.0       # expected under exact acceptance on the same draft token
        c[f"pos_{thr:g}"] = 0               # evaluated positions with p < thr
    return c


def score_position(c: dict, p: float, q: float, accepted: bool, a_rule: float | None) -> None:
    a_ex = exact_accept(p, q)
    c["positions"] += 1
    c["accepted"] += int(accepted)
    c["a_exact"] += a_ex
    if a_rule is not None:
        c["a_rule"] += a_rule
        c["a_rule_known"] += 1
    for thr in THRESHOLDS:
        if p < thr:
            c[f"pos_{thr:g}"] += 1
            c[f"low_real_{thr:g}"] += int(accepted)
            c[f"low_exact_{thr:g}"] += a_ex
            if a_rule is not None:
                c[f"low_rule_{thr:g}"] += a_rule


def finite(value) -> float | None:
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def analyse_phase(directory: Path, kind: str) -> dict:
    requests = load_requests(directory)
    per_request = [new_counter() for _ in requests]
    total = new_counter()
    unassigned = 0
    raven_terminals: dict = defaultdict(int)
    raven_roles: dict = defaultdict(int)
    cursor = 0
    for line in (directory / "rounds.jsonl").open():
        r = json.loads(line)
        index, cursor = assign(requests, r["t"], cursor)
        if index is None:
            unassigned += 1
            continue
        counters = (per_request[index], total)
        if r["kind"] == "chain":
            k, accepted_count = r["k"], r["accepted"]
            q = r.get("q")
            if q is None:
                continue
            for c in counters:
                c["rounds"] += 1
                c["emitted"] += len(r["emitted"])
            for i in range(min(k, accepted_count + 1)):
                p_i, q_i = float(r["p"][i]), float(q[i])
                acc = i < accepted_count
                a_rule = {"native": exact_accept(p_i, q_i), "cactus": cactus_accept(p_i, q_i)}.get(kind)
                for c in counters:
                    score_position(c, p_i, q_i, acc, a_rule)
        else:
            status = dict((int(n), s) for n, s in r["statuses"])
            raven_terminals[r["terminal"]] += 1
            for c in counters:
                c["rounds"] += 1
                c["emitted"] += len(r["emitted"])
            accepted_nodes = set(r["accepted_nodes"])
            for node in r["nodes"]:
                role = node.get("proposal_role")
                n = int(node["node"])
                if n in accepted_nodes:
                    raven_roles[f"accepted_{role}"] += 1
                if role != "sampled_cactus_primary" or status.get(n) not in {"SELECTED", "REJECT"}:
                    continue
                p_i, q_i = finite(node.get("target_probability")), finite(node.get("draft_probability"))
                a_rule = finite(node.get("accept_probability"))
                if p_i is None or q_i is None:
                    continue
                for c in counters:
                    score_position(c, p_i, q_i, status[n] == "SELECTED", a_rule)
            # Sibling hits: the emitted token came from the (scaled) residual; record its support too.
            for node in r["nodes"]:
                if node.get("proposal_role") == "exact_residual_correction_candidate" and node.get("correction_hit"):
                    p_i = finite(node.get("target_probability"))
                    if p_i is not None:
                        for c in counters:
                            c.setdefault("sibling_hits", 0)
                            c["sibling_hits"] += 1
                            for thr in THRESHOLDS:
                                c.setdefault(f"sibling_low_{thr:g}", 0)
                                c[f"sibling_low_{thr:g}"] += int(p_i < thr)
    out_tokens = sum(r["output_tokens"] for r in requests)
    summary = dict(kind=kind, requests=len(requests), output_tokens=out_tokens, unassigned_rounds=unassigned,
                   **{k: v for k, v in total.items()})
    summary["accept_rate_real"] = total["accepted"] / total["positions"] if total["positions"] else None
    summary["accept_rate_exact"] = total["a_exact"] / total["positions"] if total["positions"] else None
    summary["accept_rate_rule"] = total["a_rule"] / total["a_rule_known"] if total["a_rule_known"] else None
    summary["excess_real"] = (total["accepted"] - total["a_exact"]) / total["positions"] if total["positions"] else None
    summary["excess_rule"] = ((total["a_rule"] - total["a_exact"]) / total["positions"]
                              if total["a_rule_known"] == total["positions"] and total["positions"] else None)
    per_k = 1000.0 / out_tokens if out_tokens else 0.0
    for thr in THRESHOLDS:
        summary[f"low_real_per_1k_{thr:g}"] = total[f"low_real_{thr:g}"] * per_k
        summary[f"low_exact_per_1k_{thr:g}"] = total[f"low_exact_{thr:g}"] * per_k
        if summary["excess_rule"] is not None:
            summary[f"low_rule_per_1k_{thr:g}"] = total[f"low_rule_{thr:g}"] * per_k
            summary[f"low_excess_per_1k_{thr:g}"] = (total[f"low_rule_{thr:g}"] - total[f"low_exact_{thr:g}"]) * per_k
        summary[f"low_excess_real_per_1k_{thr:g}"] = (total[f"low_real_{thr:g}"] - total[f"low_exact_{thr:g}"]) * per_k
    if raven_terminals:
        summary["raven_terminals"] = dict(raven_terminals)
        summary["raven_accepted_roles"] = dict(raven_roles)
    rows = []
    for req, c in zip(requests, per_request):
        rows.append(dict(sample_id=req["sample_id"], correct=req["correct"], status=req["status"],
                         finish=req["finish"], output_tokens=req["output_tokens"], positions=c["positions"],
                         low_real_1e3=c["low_real_0.001"], low_real_1e2=c["low_real_0.01"],
                         excess_real=(c["accepted"] - c["a_exact"]),
                         excess_rule=(c["a_rule"] - c["a_exact"]) if c["a_rule_known"] == c["positions"] else None))
    return dict(summary=summary, requests=rows)


def bootstrap_diff(a: dict, b: dict, reps: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    mean = sum(diffs) / len(diffs)
    stats = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(reps))
    return mean, stats[int(0.025 * reps)], stats[int(0.975 * reps)]


def question_pass(suite: Path) -> dict:
    """{kind: {sample_id: mean correctness over runs}} from evaluation.jsonl files."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for directory in sorted(suite.iterdir()):
        m = PHASE.match(directory.name)
        if not m or not (directory / "evaluation.jsonl").exists():
            continue
        for line in (directory / "evaluation.jsonl").open():
            item = json.loads(line)
            acc[m["kind"]][item["sample_id"]].append(float(bool(item["correct"])))
    return {k: {s: sum(v) / len(v) for s, v in d.items()} for k, d in acc.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    phases = {}
    for directory in sorted(args.suite.iterdir()):
        m = PHASE.match(directory.name)
        if m and (directory / "rounds.jsonl").exists():
            phases[directory.name] = analyse_phase(directory, m["kind"])
            s = phases[directory.name]["summary"]
            print(f"{directory.name}: positions={s['positions']} accept real={s['accept_rate_real']:.4f} "
                  f"exact={s['accept_rate_exact']:.4f} rule={s['accept_rate_rule']} excess_real={s['excess_real']:.4f} "
                  f"excess_rule={s['excess_rule']} low1e-3/1k real={s['low_real_per_1k_0.001']:.2f} "
                  f"exact={s['low_exact_per_1k_0.001']:.2f} unassigned={s['unassigned_rounds']}", flush=True)
    passes = question_pass(args.suite)
    comparisons = {}
    for a, b in [("raven_v3", "cactus"), ("raven_v3", "native"), ("cactus", "native"), ("spec_cascade", "native"),
                 ("raven_v3", "spec_cascade"), ("native", "target"), ("cactus", "target")]:
        if a in passes and b in passes:
            comparisons[f"{a}-{b}"] = bootstrap_diff(passes[a], passes[b])
    result = dict(phases=phases, question_bootstrap=comparisons)
    out = args.out or args.suite / "round_log_analysis.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for name, (mean, lo, hi) in comparisons.items():
        print(f"pass@1 diff {name}: {100 * mean:+.1f} pp (95% CI {100 * lo:+.1f} .. {100 * hi:+.1f})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
