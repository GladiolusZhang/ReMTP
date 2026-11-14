"""Opt-in torch.profiler timeline of a few decode rounds (analysis only).

``REMTP_TORCH_TRACE_JSON=<path>`` records CPU ops, Python frames and CUDA
kernels for ``REMTP_TORCH_TRACE_ROUNDS`` (default 24) consecutive engine steps,
starting at step ``REMTP_TORCH_TRACE_START`` (default 200, i.e. after warmup and
inside the first long request), then writes a Chrome trace.  The profiler adds
host overhead; use the trace for GPU-idle attribution, never for throughput.
"""
from __future__ import annotations

import functools
import os

_INSTALLED = False


def install_torch_trace() -> None:
    global _INSTALLED
    path = os.environ.get("REMTP_TORCH_TRACE_JSON")
    if _INSTALLED or not path:
        return
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    start = int(os.environ.get("REMTP_TORCH_TRACE_START", "200"))
    rounds = int(os.environ.get("REMTP_TORCH_TRACE_ROUNDS", "24"))
    state = {"step": 0, "prof": None, "done": False}
    original = GPUModelRunner.execute_model

    @functools.wraps(original)
    def traced(self, *args, **kwargs):
        step = state["step"]
        state["step"] += 1
        if not state["done"] and step == start:
            torch.cuda.synchronize()
            state["prof"] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True, record_shapes=False)
            state["prof"].start()
        if state["prof"] is not None and step == start + rounds:
            torch.cuda.synchronize()
            state["prof"].stop()
            state["prof"].export_chrome_trace(path)
            state["prof"], state["done"] = None, True
            print(f"[ReMTP][TorchTrace] wrote {path} ({rounds} steps from step {start})", flush=True)
        return original(self, *args, **kwargs)

    GPUModelRunner.execute_model = traced
    _INSTALLED = True
    print(f"[ReMTP][TorchTrace] enabled: steps {start}..{start + rounds} -> {path}", flush=True)
