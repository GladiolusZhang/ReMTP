"""Quality gates and paired, question-clustered experiment summaries."""
from __future__ import annotations

import collections
import math
import random
import re
import statistics

from remtp.gsm8k_benchmark import HASH_ANSWER_PATTERN, BOXED_ANSWER_PATTERN, normalize_number


def summarize(records):
    total_seconds = sum(r["client_seconds"] for r in records)
    tokens = sum(r["output_tokens"] for r in records)
    stopped_correct = sum(r["correct"] and r["finish_reason"] == "stop" for r in records)
    times = sorted(r["client_seconds"] for r in records)
    fallback = sum(r.get("answer_source") == "last_number" and r["correct"] for r in records)
    repetition = 0
    credible = 0
    for r in records:
        segments = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"[.!?\n]", r["output"]) if len(x.strip()) >= 25]
        repetition += len(segments) >= 4 and len(segments)-len(set(segments)) >= 3
        # A final marker must actually be at the end, allowing only closing
        # punctuation/formatting. Earlier correct numbers do not qualify.
        matches=list(HASH_ANSWER_PATTERN.finditer(r["output"]))+list(BOXED_ANSWER_PATTERN.finditer(r["output"]))
        if matches:
            final=max(matches,key=lambda m:m.end())
            trailing=r["output"][final.end():]
            final_correct=normalize_number(final.group(1))==r.get("gold_answer",r.get("predicted_answer"))
            credible += r["finish_reason"]=="stop" and final_correct and bool(re.fullmatch(r"[\s.*_`$\\()\[\]{}。!]*",trailing))
    metrics=[r.get('metrics',{}) for r in records]
    decode=sum(x.get('vllm:request_decode_time_seconds_sum',0) for x in metrics)
    server_tokens=sum(x.get('vllm:generation_tokens_total',0) for x in metrics)
    rounds=sum(x.get('vllm:spec_decode_num_drafts_total',0) for x in metrics)
    accepted=sum(x.get('vllm:spec_decode_num_accepted_tokens_total',0) for x in metrics)
    return {"requests": len(records), "correct": sum(r["correct"] for r in records),
            "stopped_correct": stopped_correct, "truncated": sum(r["finish_reason"] == "length" for r in records),
            "stopped": sum(r["finish_reason"] == "stop" for r in records),
            "credible_stopped_correct": credible, "last_number_correct": fallback,
            "repetitive_requests": int(repetition), "output_tokens": tokens, "client_seconds": total_seconds,
            "e2e_tok_s": tokens / total_seconds if total_seconds else 0,
            "decode_tok_s":server_tokens/decode if decode else None,
            "mean_acceptance_length":1+accepted/rounds if rounds else None,
            "decode_seconds_per_round":decode/rounds if rounds else None,
            "mean_seconds": statistics.mean(times) if times else 0,
            "median_seconds": statistics.median(times) if times else 0,
            "p95_seconds": times[max(0, math.ceil(len(times)*0.95)-1)] if times else 0,
            "stopped_correct_per_hour": stopped_correct*3600/total_seconds if total_seconds else 0}


def paired_gate(before, after, require_speed=True):
    def key(r): return r["question_id"], r["seed"]
    if len(set(map(key, before))) != len(before) or set(map(key, before)) != set(map(key, after)) or len(before) != len(after):
        raise ValueError("Quality gate requires unique matching question/seed pairs")
    old, new = summarize(before), summarize(after)
    failures = []
    seeds = sorted({r["seed"]-r["question_id"]*10 for r in before})
    groups = {"all": (old, new)}
    for seed in seeds:
        groups[str(seed)] = tuple(summarize([r for r in data if r["seed"]-r["question_id"]*10 == seed]) for data in (before, after))
    for name, (a, b) in groups.items():
        for metric in ("correct", "stopped_correct"):
            if b[metric] < a[metric]: failures.append(f"{name}:{metric}")
        if b["truncated"] > a["truncated"]: failures.append(f"{name}:truncated")
    if require_speed:
        if new["e2e_tok_s"] <= old["e2e_tok_s"]: failures.append("all:throughput")
        if new["mean_seconds"] >= old["mean_seconds"]: failures.append("all:latency")
    return {"passed": not failures, "failures": failures, "baseline": old, "candidate": new,
            "per_seed": {k: {"baseline": a, "candidate": b} for k,(a,b) in groups.items() if k != "all"}}


def clustered_intervals(before, after, draws=2000):
    """Resample questions together with all their seeds; fixed bootstrap RNG."""
    paired_gate(before, after, require_speed=False)  # validate pairing
    a, b = collections.defaultdict(list), collections.defaultdict(list)
    for r in before: a[r["question_id"]].append(r)
    for r in after: b[r["question_id"]].append(r)
    ids = sorted(a)
    rng = random.Random(20261002)
    rates, correct, stopped = [], [], []
    for _ in range(draws):
        sampled = rng.choices(ids, k=len(ids))
        old = summarize_fast([r for i in sampled for r in a[i]])
        new = summarize_fast([r for i in sampled for r in b[i]])
        rates.append(new[0]/old[0])
        correct.append(new[1]-old[1]); stopped.append(new[2]-old[2])
    def interval(values):
        values.sort()
        return [values[int(draws*.025)], values[min(draws-1,int(draws*.975))]]
    return {"throughput_ratio_95ci": interval(rates), "accuracy_difference_95ci": interval(correct),
            "stopped_correct_difference_95ci": interval(stopped), "unit": "question cluster, all seeds", "draws": draws}


def summarize_fast(records):
    return (sum(r["output_tokens"] for r in records)/sum(r["client_seconds"] for r in records),
            sum(r["correct"] for r in records)/len(records),
            sum(r["correct"] and r["finish_reason"] == "stop" for r in records)/len(records))


def dominance_gate(before, after, require_speed=True):
    """Accuracy and real speed only; completion diagnostics never veto a result.

    The legacy paired_gate intentionally remains unchanged for old experiments.
    Per-seed differences are reported; the registered aggregate is the gate.
    """
    if not before or not after:
        raise ValueError('Dominance gate requires nonempty complete pairs')
    key = lambda r: (r['question_id'], r['seed'])
    a = {key(r): r for r in before}
    b = {key(r): r for r in after}
    if len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys():
        raise ValueError('Dominance gate requires unique matching question/seed pairs')
    for pair in a:
        for field in ('max_tokens', 'prompt_token_ids', 'gold_answer'):
            if field in a[pair] and field in b[pair] and a[pair][field] != b[pair][field]:
                raise ValueError('Comparison protocol mismatch: '+field)
        for record in (a[pair], b[pair]):
            if (not math.isfinite(record['client_seconds']) or record['client_seconds'] <= 0
                    or record['output_tokens'] <= 0):
                raise ValueError('Invalid time or token count')
            if require_speed and (record.get('performance_sample') is False or record.get('observation_enabled')):
                raise ValueError('Observed diagnostic requests cannot establish speed')
    old, new = summarize(before), summarize(after)
    failures = []
    if new['correct'] < old['correct']:
        failures.append('all:accuracy')
    if require_speed:
        if new['e2e_tok_s'] <= old['e2e_tok_s']:
            failures.append('all:throughput')
        if new['mean_seconds'] >= old['mean_seconds']:
            failures.append('all:latency')
    seeds = sorted({r['seed']-r['question_id']*10 for r in before})
    per_seed = {str(seed): {name: summarize([r for r in records if r['seed']-r['question_id']*10 == seed])
        for name, records in (('baseline', before), ('candidate', after))} for seed in seeds}
    losses = [dict(question_id=pair[0], seed=pair[1]) for pair in sorted(a)
        if a[pair]['correct'] and not b[pair]['correct']]
    gains = [dict(question_id=pair[0], seed=pair[1]) for pair in sorted(a)
        if not a[pair]['correct'] and b[pair]['correct']]
    return dict(passed=not failures, failures=failures, baseline=old, candidate=new,
        accuracy_difference=(new['correct']-old['correct'])/len(before),
        throughput_ratio=new['e2e_tok_s']/old['e2e_tok_s'],
        latency_speedup=old['mean_seconds']/new['mean_seconds'],
        per_seed=per_seed, accuracy_losses=losses, accuracy_gains=gains,
        objective='accuracy_nondecrease_and_faster_than_each_baseline',
        diagnostic_only=['stopped_correct', 'truncated', 'repetitive_requests', 'per_seed', 'pairwise_changes'])
