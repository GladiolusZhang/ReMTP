"""Opt-in GPU time of model components from CUDA events (analysis only).

``REMTP_GPU_EVENT_TIMING_JSONL=<path>`` brackets the target forward
(``GPUModelRunner._model_forward``, eager or CUDA-graph replay), the target
LM head, the MTP layer forward and the MTP LM head with CUDA events.  No
synchronisation is added: an event pair is read only once its end event has
completed, which the decode loop's own host reads guarantee by the next step.
CUPTI kernel tracing is unavailable on this Jetson (RmProfilingAdminOnly=1);
events are not.  Every ``REMTP_GPU_EVENT_TIMING_EVERY`` engine steps (default
200) one JSON line with mean milliseconds per call and calls per step, keyed by
component and row count, is appended to the file.  Step wall time is measured
between consecutive ``execute_model`` entries.
"""
from __future__ import annotations

import collections
import functools
import json
import os
import time
from typing import Any

import torch

_INSTALLED = False


def install_gpu_event_timing() -> None:
    global _INSTALLED
    path = os.environ.get("REMTP_GPU_EVENT_TIMING_JSONL")
    if _INSTALLED or not path:
        return
    from vllm.model_executor.models.mimo import MiMoForCausalLM
    from vllm.model_executor.models.mimo_mtp import MiMoMTP
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    every = int(os.environ.get("REMTP_GPU_EVENT_TIMING_EVERY", "200"))
    pending: list[tuple[str, int, torch.cuda.Event, torch.cuda.Event]] = []
    totals: dict[tuple[str, int], list[float]] = collections.defaultdict(lambda: [0.0, 0])
    state = {"steps": 0, "last": None, "wall": 0.0, "window_steps": 0}

    def drain() -> None:
        keep = []
        for item in pending:
            name, rows, start, end = item
            if end.query():
                total = totals[(name, rows)]
                total[0] += start.elapsed_time(end)
                total[1] += 1
            else:
                keep.append(item)
        pending[:] = keep

    def timed(name: str, rows_of: Any, original: Any) -> Any:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if torch.cuda.is_current_stream_capturing():
                return original(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            try:
                rows = int(rows_of(args, kwargs))
            except Exception:
                rows = -1
            pending.append((name, rows, start, end))
            return result
        return wrapper

    def first_rows(args: Any, kwargs: Any) -> int:
        tensor = kwargs.get("input_ids")
        if tensor is None:
            tensor = kwargs.get("hidden_states")
        if tensor is None and len(args) > 1:
            tensor = args[1]
        return tensor.shape[0]

    GPUModelRunner._model_forward = timed("target_forward", first_rows, GPUModelRunner._model_forward)
    MiMoForCausalLM.compute_logits = timed("target_lm_head", lambda a, k: a[1].shape[0], MiMoForCausalLM.compute_logits)
    MiMoMTP.forward = timed("mtp_forward", first_rows, MiMoMTP.forward)
    MiMoMTP.compute_logits = timed("mtp_lm_head", lambda a, k: a[1].shape[0], MiMoMTP.compute_logits)

    original_execute = GPUModelRunner.execute_model

    @functools.wraps(original_execute)
    def execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        now = time.perf_counter()
        if state["last"] is not None and now - state["last"] < 1.0:  # skip idle gaps between requests
            state["wall"] += now - state["last"]
            state["window_steps"] += 1
        state["last"] = now
        drain()
        state["steps"] += 1
        if state["window_steps"] >= every:
            steps = state["window_steps"]
            record = {
                "steps": steps,
                "wall_ms_per_step": 1e3 * state["wall"] / steps,
                "components": {
                    f"{name}[{rows}]": {"ms_per_call": total[0] / total[1], "calls_per_step": total[1] / steps}
                    for (name, rows), total in sorted(totals.items()) if total[1]
                },
            }
            record["gpu_ms_per_step"] = sum(
                v["ms_per_call"] * v["calls_per_step"] for v in record["components"].values())
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            totals.clear()
            state["wall"], state["window_steps"] = 0.0, 0
        return original_execute(self, *args, **kwargs)

    GPUModelRunner.execute_model = execute_model
    _INSTALLED = True
    print(f"[ReMTP][GpuEventTiming] CUDA-event component timing -> {path}", flush=True)
