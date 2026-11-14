"""Bounded, opt-in observation of real decode rounds; never benchmark here.

CUDA events measure stream spans, including host launch gaps, not sums of
kernel execution time. CPU wall spans are inclusive and can include waits.
Only one explicit synchronization is added at the end of each observed round.
All nesting is recorded; children must not be added to their parents.
An empty profile path installs no hooks. No extra model/RNG/top-k calls occur.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import time

import torch

_INSTALLED = False


class ProfileRecorder:
    """Single-worker/single-request observer with injectable CUDA for tests."""

    def __init__(self, path, *, max_rounds=32, per_request=16,
                 skip_requests=1, profile_requests=2, cuda=None):
        if not 1 <= max_rounds <= 32 or not 1 <= per_request <= max_rounds:
            raise ValueError("Profile requires 1..32 rounds and a bounded request cap")
        if skip_requests != 1:
            raise ValueError("This profile protocol skips exactly one warmup request")
        if profile_requests != 2:
            raise ValueError("This profile protocol observes exactly two registered requests")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds
        self.per_request = per_request
        self.skip_requests = skip_requests
        self.profile_requests = profile_requests
        self.cuda = torch.cuda if cuda is None else cuda
        self.seen = []
        self.counts = {}
        self.rounds = 0
        self.frame = None
        self.stack = []
        self.spans = []

    def begin(self, request_id, generated, input_rows):
        if self.frame is not None:
            raise RuntimeError("Previous observed round was not flushed by sample_tokens")
        if not request_id or not self.cuda.is_initialized():
            return False
        if request_id not in self.seen:
            self.seen.append(request_id)
        request_index=self.seen.index(request_id)
        if (request_index < self.skip_requests
                or request_index >= self.skip_requests+self.profile_requests or generated == 0
                or self.rounds >= self.max_rounds
                or self.counts.get(request_id, 0) >= self.per_request):
            return False
        self.rounds += 1
        self.counts[request_id] = self.counts.get(request_id, 0) + 1
        start = self.cuda.Event(enable_timing=True)
        start.record()
        self.frame = dict(request=request_id, profile_round=self.rounds,
                          request_round=self.counts[request_id], generated_before=generated,
                          input_rows=input_rows, phase="decode", start=start,
                          wall_start=time.perf_counter())
        self.spans = []
        return True

    @contextmanager
    def span(self, component, extra=None):
        if self.frame is None:
            yield None
            return
        domain = self.stack[-1]["domain"] if self.stack else "common"
        if component in ("target_forward", "target_logits"):
            domain = "target"
        elif component in ("proposal", "tree_proposal", "mtp_forward", "mtp_logits"):
            domain = "draft"
        start = self.cuda.Event(enable_timing=True)
        end = self.cuda.Event(enable_timing=True)
        record = dict(component=component, domain=domain, span_id=len(self.spans)+1,
                      parent_id=self.stack[-1]["span_id"] if self.stack else 0,
                      start=start, end=end, **(extra or {}))
        self.spans.append(record)
        self.stack.append(record)
        start.record()
        began = time.perf_counter()
        try:
            yield record
        except BaseException as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["wall_seconds"] = time.perf_counter()-began
            end.record()
            self.stack.pop()

    def finish(self, generated_after):
        if self.frame is None:
            return
        if self.stack:
            raise RuntimeError("Cannot flush unfinished profile spans")
        frame = self.frame
        end = self.cuda.Event(enable_timing=True)
        end.record()
        wall = time.perf_counter()-frame["wall_start"]
        began = time.perf_counter()
        self.cuda.synchronize()
        sync = time.perf_counter()-began
        common = {k:v for k,v in frame.items() if k not in ("start", "wall_start")}
        common.update(pid=os.getpid(), ended_ns=time.time_ns())
        output = []
        for span in self.spans:
            row = {k:v for k,v in span.items() if k not in ("start", "end")}
            row.update(common, record_type="span",
                       cuda_span_seconds=span["start"].elapsed_time(span["end"])/1000,
                       seconds=span["wall_seconds"])
            output.append(row)
        output.append(dict(common, record_type="round", component="round",
                           parent_id=None, span_id=0, wall_seconds=wall,
                           cuda_span_seconds=frame["start"].elapsed_time(end)/1000,
                           explicit_sync_seconds=sync, generated_after=generated_after,
                           worker_committed_tokens=generated_after-frame["generated_before"],
                           boundary="after_prepare_inputs_to_after_sample_tokens"))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write("".join(json.dumps(row)+"\n" for row in output))
        self.frame = None
        self.spans = []


def install_runtime_profile():
    global _INSTALLED
    path = os.getenv("REMTP_RUNTIME_PROFILE_JSONL", "")
    if not path or _INSTALLED:
        return
    from bitsandbytes.backends.cuda import ops
    from vllm.model_executor.models.mimo import MiMoForCausalLM
    from vllm.model_executor.models.mimo_mtp import MiMoMTP
    from vllm.v1.attention.backends import tree_attn, triton_attn
    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from remtp import dynamic_mtp_tree as algorithm, dynamic_tree_vllm as dynamic

    recorder = ProfileRecorder(path,
        max_rounds=int(os.getenv("REMTP_PROFILE_MAX_ROUNDS", "32")),
        per_request=int(os.getenv("REMTP_PROFILE_MAX_ROUNDS_PER_REQUEST", "16")),
        skip_requests=int(os.getenv("REMTP_PROFILE_SKIP_REQUESTS", "1")))

    def rows(args, kwargs):
        value = next((v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)), None)
        return {"rows": int(value.shape[0])} if value is not None and value.ndim else {}

    hooks = [
        (MiMoForCausalLM, "forward", "target_forward", rows),
        (MiMoForCausalLM, "compute_logits", "target_logits", rows),
        (EagleProposer, "propose", "proposal", None),
        (EagleProposer, "propose_tree", "tree_proposal", None),
        (MiMoMTP, "forward", "mtp_forward", rows),
        (MiMoMTP, "compute_logits", "mtp_logits", rows),
        (GPUModelRunner, "_sample", "sampling", None),
        (RejectionSampler, "forward", "verification", None),
        (GPUModelRunner, "_update_states_after_model_execute", "cache_update", None),
        (GPUModelRunner, "_bookkeeping_sync", "bookkeeping", None),
        (algorithm, "adaptive_trunk_relaxation_plan", "adaptive_plan", None),
        (dynamic, "_run_dynamic_tree_nodes", "draft_tree",
            lambda a,k: {"node_rows": len(k["node_ids"])}),
        (dynamic, "select_dynamic_candidates", "candidate_selection", None),
        (algorithm, "normalize_distribution", "normalization", rows),
        (ops, "_dequantize_4bit_impl", "fp4_dequant", None),
        (torch.nn.functional, "linear", "linear", rows),
        (tree_attn, "unified_attention", "tree_attention", None),
        (triton_attn, "unified_attention", "triton_attention", None),
    ]
    # Finer host-side spans of the dynamic tree (2026-11-13).  Rebound module
    # globals are wrapped where they are looked up at call time.
    import vllm.v1.sample.rejection_sampler as rejection_module
    from remtp import fixed6_vllm as fixed, raven_fast_path as fast
    hooks += [
        (dynamic, "_dynamic_propose_tree", "dynamic_propose", None),
        (dynamic, "_dynamic_verify_dispatch", "verify_dispatch", None),
        (dynamic, "dynamic_tree_config_from_env", "config_from_env", None),
        (dynamic, "_sample_q_token", "sample_q", None),
        (dynamic, "_sample_branch_q_token", "sample_branch_q", None),
        (dynamic, "_cactus_trunk_children", "trunk_children", None),
        (dynamic, "_validate_proposal_probs", "validate_q", None),
        (dynamic, "_set_target_tree_topology", "set_tree_topology", None),
        (fast, "residual_hit_tree_verify_fast", "fast_verify", None),
        (fast, "adaptive_trunk_relaxation_plan", "adaptive_plan_fast", None),
        (fast, "select_dynamic_candidates_fast", "select_fast", None),
        (fast, "batched_trunk_children", "batched_children", None),
        (fast, "resolve_batched_counts", "batched_counts", None),
        (fast, "_boost_known", "boost_row", None),
        (fast, "residual_distribution_nosync", "residual_row", None),
        (fast, "sample_categorical_nosync", "categorical_draw", None),
        (fast, "normalize_rows_nosync", "normalize_rows", None),
        (fixed, "_processed_target_probs", "target_probs", None),
        (fixed, "_append_round_audit", "round_audit", None),
        (rejection_module, "apply_sampling_constraints", "sampling_constraints", None),
    ]
    hooks = [hook for hook in hooks if hasattr(hook[0], hook[1])]
    originals = [(owner, name, component, extra, getattr(owner, name))
                 for owner,name,component,extra in hooks]
    prepare = GPUModelRunner._prepare_inputs
    sample = GPUModelRunner.sample_tokens

    def wrap(original, component, extra):
        @functools.wraps(original)
        def measured(*args, **kwargs):
            if recorder.frame is None:
                return original(*args, **kwargs)
            metadata = extra(args, kwargs) if extra else None
            with recorder.span(component, metadata):
                return original(*args, **kwargs)
        return measured

    for owner,name,component,extra,original in originals:
        setattr(owner, name, wrap(original, component, extra))

    @functools.wraps(prepare)
    def prepared(self, *args, **kwargs):
        result = prepare(self, *args, **kwargs)
        ids = self.input_batch.req_ids
        if len(ids) == 1:
            state = self.requests[ids[0]]
            recorder.begin(ids[0], len(state.output_token_ids),
                           int(self.query_start_loc.np[len(ids)]))
        return result

    @functools.wraps(sample)
    def sampled(self, *args, **kwargs):
        result = sample(self, *args, **kwargs)
        if recorder.frame is not None:
            state = self.requests.get(recorder.frame["request"])
            if state is None:
                raise RuntimeError("Observed request disappeared before profile flush")
            recorder.finish(len(state.output_token_ids))
        return result

    GPUModelRunner._prepare_inputs = prepared
    GPUModelRunner.sample_tokens = sampled
    _INSTALLED = True
    print(f"[ReMTP][Profile] bounded events enabled: {path} "
          f"max_rounds={recorder.max_rounds} per_request={recorder.per_request} "
          "skip_requests=1 automatic_topk_probe=off", flush=True)
