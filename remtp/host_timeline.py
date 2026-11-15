"""Opt-in host timeline of decode steps without a profiler (analysis only).

``REMTP_HOST_TIMELINE_JSONL=<path>`` wraps a fixed list of runner, sampler,
proposal and verification functions plus ``Tensor.tolist/item/cpu`` (the
host reads that wait for the GPU) with ``time.perf_counter`` probes: about a
microsecond per probe, no synchronisation.  For engine steps
``REMTP_HOST_TIMELINE_START`` .. ``+REMTP_HOST_TIMELINE_STEPS`` (defaults 150
and 60) one JSON line per step lists ``[name, "in"|"out", ms since step
start]``.  With ``REMTP_CUDA_SPIN_WAIT=1`` the time inside a host read is the
GPU work still queued; everything else is host time during which the GPU may
idle.  ``scripts/analyze_host_timeline.py`` attributes it.
"""
from __future__ import annotations

import functools
import importlib
import json
import os
import time
from typing import Any

import torch

_INSTALLED = False
_TARGETS = (
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", (
        "_prepare_inputs", "_model_forward", "sample_tokens", "propose_draft_token_ids",
        "_bookkeeping_sync", "_update_states_after_model_execute")),
    ("vllm.model_executor.models.mimo", "MiMoForCausalLM", ("compute_logits",)),
    ("vllm.model_executor.models.mimo_mtp", "MiMoMTP", ("forward", "compute_logits")),
    ("vllm.v1.sample.rejection_sampler", "RejectionSampler", ("forward",)),
    ("vllm.v1.spec_decode.eagle", "EagleProposer", ("propose", "prepare_inputs_padded")),
    ("vllm.v1.core.sched.scheduler", "Scheduler", ("schedule", "update_from_output")),
)
_FUNCTIONS = (
    ("remtp.raven_fast_path", ("residual_hit_tree_verify_fast", "batched_trunk_children")),
    ("remtp.dynamic_tree_vllm", ("_run_dynamic_tree_nodes",)),
    ("remtp.fixed6_vllm", ("strict_tree_verify",)),
    ("remtp.raven_fast_commit", ("commit_selected_slots_inplace",)),
    ("remtp.probabilistic_mtp", ("sample_mtp_logits",)),
    ("remtp.gpu_tree", ("propose_tree", "select_children", "retain_level", "run_pass", "branch_noise")),
)
# REMTP_HOST_TIMELINE_FINE=1: helpers, wrapped in the namespace that calls them.
_FINE = (
    ("remtp.dynamic_tree_vllm", ("path_expansion_value", "level_expansion_order", "adaptive_node_limit",
                                 "level_expansion_budget", "_set_builder_topology", "_sample_q_token",
                                 "_sample_branch_q_token", "mtp_layer_for_tree_depth", "force_mimo_mtp_layer",
                                 "_set_target_tree_topology", "_copy_variable_drafts_to_cpu",
                                 "dynamic_tree_config_from_env")),
    ("remtp.fixed6_vllm", ("_draft_tree_metadata", "_compact_draft_tree_bias", "_tree_positions",
                           "_temperature_probs", "_processed_target_probs", "_append_round_audit")),
    ("remtp.raven_fast_path", ("adaptive_trunk_relaxation_plan_host", "_boost_known", "residual_distribution_nosync",
                               "sample_categorical_nosync", "normalize_rows_nosync", "resolve_batched_counts",
                               "cactus_trunk_children_fast")),
    ("remtp.round_logger", ()),
)


def install_host_timeline() -> None:
    global _INSTALLED
    path = os.environ.get("REMTP_HOST_TIMELINE_JSONL")
    if _INSTALLED or not path:
        return
    start = int(os.environ.get("REMTP_HOST_TIMELINE_START", "150"))
    count = int(os.environ.get("REMTP_HOST_TIMELINE_STEPS", "60"))
    clock = time.perf_counter
    state = {"step": -1, "t0": 0.0, "events": None}

    def mark(name: str, kind: str) -> None:
        events = state["events"]
        if events is not None:
            events.append((name, kind, round((clock() - state["t0"]) * 1e3, 4)))

    def wrap(name: str, original: Any) -> Any:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if state["events"] is None:
                return original(*args, **kwargs)
            mark(name, "in")
            try:
                return original(*args, **kwargs)
            finally:
                mark(name, "out")
        wrapper._remtp_host_timeline = True
        return wrapper

    sync_debug = os.environ.get("REMTP_HOST_TIMELINE_SYNC", "0") == "1"
    if sync_debug:
        # torch.cuda.set_sync_debug_mode(1) warns on every synchronising CUDA
        # call (item, implicit scalar-index reads, blocking copies, nonzero ...);
        # record the calling location as a "sync" event.
        import traceback
        import warnings

        warnings.filterwarnings("always", message=".*synchroniz.*")
        previous_show = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
            if state["events"] is None or "synchroniz" not in str(message):
                return previous_show(message, category, filename, lineno, file, line)
            frames = [f for f in traceback.extract_stack()[:-1]
                      if ("/remtp/" in f.filename or "/vllm/" in f.filename) and "host_timeline" not in f.filename]
            where = " < ".join(f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}" for f in frames[-3:][::-1])
            mark("SYNC " + where, "sync")

        warnings.showwarning = show

    for module_name, class_name, methods in _TARGETS:
        cls = getattr(importlib.import_module(module_name), class_name)
        for method in methods:
            setattr(cls, method, wrap(f"{class_name}.{method}", getattr(cls, method)))
    for module_name, names in _FUNCTIONS:
        module = importlib.import_module(module_name)
        for name in names:
            if hasattr(module, name):
                setattr(module, name, wrap(name, getattr(module, name)))
    if os.environ.get("REMTP_HOST_TIMELINE_FINE", "0") == "1":
        for module_name, names in _FINE:
            module = importlib.import_module(module_name)
            for name in names:
                if hasattr(module, name):
                    setattr(module, name, wrap(name, getattr(module, name)))
        from remtp.dynamic_mtp_tree import DynamicTreeTopology
        original_from_paths = DynamicTreeTopology.from_paths.__func__
        DynamicTreeTopology.from_paths = classmethod(wrap("DynamicTreeTopology.from_paths", original_from_paths))
    for method in ("tolist", "item", "cpu"):
        setattr(torch.Tensor, method, wrap(f"Tensor.{method}", getattr(torch.Tensor, method)))

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    original_execute = GPUModelRunner.execute_model

    @functools.wraps(original_execute)
    def execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        now = clock()
        events = state["events"]
        if events is not None:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": state["step"], "ms": round((now - state["t0"]) * 1e3, 4),
                                         "events": events}) + "\n")
        state["step"] += 1
        state["t0"] = now
        state["events"] = [] if start <= state["step"] < start + count else None
        if sync_debug:
            torch.cuda.set_sync_debug_mode(1 if state["events"] is not None else 0)
        mark("GPUModelRunner.execute_model", "in")
        try:
            return original_execute(self, *args, **kwargs)
        finally:
            mark("GPUModelRunner.execute_model", "out")

    GPUModelRunner.execute_model = execute_model
    _INSTALLED = True
    print(f"[ReMTP][HostTimeline] steps {start}..{start + count} -> {path}", flush=True)
