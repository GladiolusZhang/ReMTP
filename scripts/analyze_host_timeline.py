"""Attribute decode-step wall time from ``remtp.host_timeline`` probes.

Usage: python scripts/analyze_host_timeline.py <host_timeline.jsonl> [more ...]

Per step: time inside host reads that wait for the GPU (``Tensor.tolist/item/
cpu``) versus host time, the latter attributed exclusively to the innermost
probed function (``(between)`` = outside every probe, e.g. engine-core code).
With spin waiting enabled, host time is when the GPU may idle.
"""
from __future__ import annotations

import collections
import json
import sys

WAITS = ("Tensor.tolist", "Tensor.item", "Tensor.cpu")


def analyse(path: str) -> None:
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    records = [r for r in records if r["events"]]
    host = collections.Counter()
    wait = collections.Counter()
    wait_calls = collections.Counter()
    syncs = collections.Counter()
    total = 0.0
    for record in records:
        events = record["events"]
        end = record["ms"]
        total += end
        stack: list[str] = []
        last = 0.0
        waiting = 0
        for name, kind, t in events + [("(end)", "in", end)]:
            if kind == "sync":
                syncs[name] += 1
                continue
            span = t - last
            if span > 0:
                if waiting:
                    owner = next((n for n in reversed(stack) if n not in WAITS), "(between)")
                    wait[owner] += span
                else:
                    host[stack[-1] if stack else "(between)"] += span
            last = t
            if name == "(end)":
                break
            if kind == "in":
                stack.append(name)
                if name in WAITS:
                    waiting += 1
                    owner = next((n for n in reversed(stack[:-1]) if n not in WAITS), "(between)")
                    wait_calls[owner] += 1
            else:
                while stack and stack[-1] != name:
                    stack.pop()
                if stack:
                    stack.pop()
                if name in WAITS:
                    waiting -= 1
    n = len(records)
    host_total = sum(host.values()) / n
    wait_total = sum(wait.values()) / n
    print(f"{path}\n  steps {n}  wall {total / n:.1f} ms = host {host_total:.1f} + waiting for GPU {wait_total:.1f}")
    print("  host time by innermost probe (ms/step):")
    for name, value in host.most_common(16):
        print(f"    {name:52s} {value / n:6.2f}")
    if syncs:
        print("  synchronising CUDA calls (per step), innermost first:")
        for name, value in syncs.most_common(30):
            print(f"    {value / n:5.2f}  {name[5:][:150]}")
    print("  waits by caller (ms/step, calls/step):")
    for name, value in wait.most_common(8):
        print(f"    {name:52s} {value / n:6.2f}  {wait_calls[name] / n:4.1f}")


if __name__ == "__main__":
    for item in sys.argv[1:]:
        analyse(item)
