"""Bounded, opt-in copies of actual decision inputs; never draw random numbers."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict
import functools
import json
import os
from pathlib import Path

import torch

_ACTIVE = ContextVar("remtp_decision_trace", default=None)
_PROPOSAL = ContextVar("remtp_proposal_trace", default=None)
_INSTALLED = False
MAX_ROUNDS = 8


def active_trace():
    return _ACTIVE.get()


def cpu(value):
    return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value


def checkpoint_slot(history_length, checkpoints, next_index):
    """Take at most one frame per actual verification; consume crossed points."""
    if next_index >= len(checkpoints) or history_length < checkpoints[next_index]:
        return None
    end = next_index + 1
    while end < len(checkpoints) and checkpoints[end] <= history_length:
        end += 1
    return end


class SummaryRecorder:
    """Bounded metadata from the one real verifier result, with no new draws.

    Candidate IDs require one tiny device-to-host copy. P/Q/H, uniforms and
    plans come from diagnostics the verifier already computed. These observed
    runs must not be used for throughput comparisons.
    """
    def __init__(self, path, max_rounds=2048, max_requests=6):
        if not 1 <= max_rounds <= 2048 or not 1 <= max_requests <= 6:
            raise ValueError('Invalid decision summary budget')
        self.path = Path(path)
        self.max_rounds = max_rounds
        self.max_requests = max_requests
        self.counts = {}

    def save(self, runtime, topology, ids, result):
        request = str(getattr(runtime, 'diagnostic_request', ''))
        prompt = tuple(getattr(runtime, 'diagnostic_prompt_ids', ()))
        if not request or len(prompt) <= 64:
            return
        if request not in self.counts:
            if len(self.counts) >= self.max_requests:
                raise RuntimeError('Decision summary request budget exceeded')
            self.counts[request] = 0
        number = self.counts[request]
        if number >= self.max_rounds:
            raise RuntimeError('Decision summary round budget exceeded')
        candidates = cpu(ids).tolist()
        history = tuple(getattr(runtime, 'generated_history', ()))
        record = dict(schema=1, request=request, round=number,
            generated_length=len(history), history_suffix=history[-16:],
            topology=topology.choices, candidate_ids=candidates,
            result=asdict(result), performance_sample=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        self.counts[request] = number + 1


def install_decision_trace(directory):
    global _INSTALLED
    if _INSTALLED:
        return
    from remtp import fixed6_vllm as fixed, dynamic_tree_vllm as dynamic
    from vllm.v1.spec_decode.eagle import EagleProposer
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    counts = {"verify": {}, "proposal": {}}
    checkpoints = tuple(int(x) for x in os.getenv('REMTP_DECISION_TRACE_CHECKPOINTS', '').split(',') if x)
    if checkpoints and (len(checkpoints) > MAX_ROUNDS or checkpoints != tuple(sorted(set(checkpoints))) or checkpoints[0] < 0):
        raise ValueError('Decision trace checkpoints require <=8 sorted unique nonnegative positions')
    next_checkpoint = {}
    early_proposals = int(os.getenv('REMTP_DECISION_TRACE_EARLY_PROPOSALS', '0'))
    if not 0 <= early_proposals <= MAX_ROUNDS:
        raise ValueError('Early proposal budget requires 0..8 frames')
    summary_path = os.getenv('REMTP_DECISION_TRACE_SUMMARY_JSONL', '')
    summary = SummaryRecorder(summary_path,
        int(os.getenv('REMTP_DECISION_TRACE_SUMMARY_MAX_ROUNDS', '2048')),
        int(os.getenv('REMTP_DECISION_TRACE_SUMMARY_MAX_REQUESTS', '6'))) if summary_path else None

    def slot(kind):
        runtime = fixed._RUNTIME
        prompt = tuple(getattr(runtime, "diagnostic_prompt_ids", ()))
        request = str(getattr(runtime, "diagnostic_request", ""))
        # Ignore CUDA initialization and the short, separately counted warmup.
        if len(prompt) <= 64 or not request:
            return None
        number = counts[kind].get(request, 0)
        if checkpoints:
            if kind != 'verify':
                # Independent early proposal capture. Its pre-bookkeeping
                # history is not the complete prefix of the next verifier.
                if number >= early_proposals:
                    return None
            else:
                end = checkpoint_slot(len(getattr(runtime, 'generated_history', ())),
                                      checkpoints, next_checkpoint.get(request, 0))
                if end is None:
                    return None
                next_checkpoint[request] = end
        if number >= MAX_ROUNDS:
            return None
        counts[kind][request] = number + 1
        folder = root / request
        folder.mkdir(exist_ok=True)
        return folder, number, prompt, tuple(getattr(runtime, "generated_history", ()))

    verify = fixed.strict_tree_verify

    @functools.wraps(verify)
    def traced(topology, ids, draft, target, *args, **kwargs):
        location = slot("verify")
        if location is None:
            result = verify(topology, ids, draft, target, *args, **kwargs)
            if summary is not None:
                summary.save(fixed._RUNTIME, topology, ids, result)
            return result
        folder, number, prompt, history = location
        generator = args[0] if args else kwargs.get("generator")
        record = dict(schema=1, request=folder.name, round=number,
                      prompt_ids=prompt, history=history, topology=topology.choices,
                      ids=cpu(ids), draft=cpu(draft), target=cpu(target),
                      rng_device=str(generator.device) if generator is not None else None,
                      rng_before=generator.get_state().clone() if generator is not None else None)
        if checkpoints:
            record.update(capture_mode='checkpoint_verifier' if early_proposals else 'late_verifier_only', checkpoints=checkpoints,
                          early_proposal_limit=early_proposals,
                          observed_generated_length=len(history))
        token = _ACTIVE.set(record)
        try:
            result = verify(topology, ids, draft, target, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)
        record.update(result=asdict(result),
                      rng_after=generator.get_state().clone() if generator is not None else None)
        temporary = folder / f"verify_{number:02d}.tmp"
        torch.save(record, temporary)
        temporary.replace(folder / f"verify_{number:02d}.pt")
        if summary is not None:
            summary.save(fixed._RUNTIME, topology, ids, result)
        return result

    propose = EagleProposer.propose_tree

    @functools.wraps(propose)
    def proposed(self, *args, **kwargs):
        location = slot("proposal")
        if location is None:
            return propose(self, *args, **kwargs)
        folder, number, prompt, history = location
        names = ("batch_size", "logits", "positions", "hidden_states", "common_attn_metadata", "slot_mappings")
        values = {**dict(zip(names, args)), **kwargs}
        metadata = values["common_attn_metadata"]
        generator = (getattr(self._fixed6_sampling_metadata, "generators", None) or {}).get(0)
        # vLLM proposes before CPU bookkeeping appends this round's sampled
        # tokens. The next verification supplies the true committed prefix;
        # label this earlier history explicitly rather than pretending it is
        # the proposal's complete prefix.
        record = dict(schema=1, request=folder.name, round=number, prompt_ids=prompt,
                      history_before_current_sampling=history, root_hidden=cpu(values["hidden_states"]),
                      root_logits=cpu(values["logits"]), root_positions=cpu(values["positions"]),
                      seq_lens=cpu(metadata.seq_lens), branch_round=dynamic._BRANCH_PROPOSAL_ROUND,
                      rng_before=generator.get_state().clone() if generator is not None else None)
        record['tree_passes'] = []
        token = _PROPOSAL.set(record)
        try:
            result = propose(self, *args, **kwargs)
        finally:
            _PROPOSAL.reset(token)
        record.update(topology=fixed._RUNTIME.topology.choices,
                      ids=cpu(fixed._RUNTIME.draft_token_ids), q=cpu(fixed._RUNTIME.draft_probs),
                      rng_after=generator.get_state().clone() if generator is not None else None)
        temporary = folder / f"proposal_{number:02d}.tmp"
        torch.save(record, temporary)
        temporary.replace(folder / f"proposal_{number:02d}.pt")
        return result

    fixed.strict_tree_verify = traced
    EagleProposer.propose_tree = proposed
    tree_run = dynamic._run_dynamic_tree_nodes

    @functools.wraps(tree_run)
    def tree_observed(self, **kwargs):
        record = _PROPOSAL.get()
        if record is None:
            return tree_run(self, **kwargs)
        from vllm.model_executor.layers.attention import Attention
        node_ids = kwargs['node_ids']
        root_position = int(kwargs['root_positions'].reshape(-1)[0])
        meta = kwargs['base_common_metadata']
        detail = dict(node_ids=node_ids, root_position=root_position,
            tokens=cpu(torch.stack([kwargs['node_tokens'][i] for i in node_ids])),
            hidden_inputs=cpu(torch.stack([kwargs['node_hidden_inputs'][i] for i in node_ids])),
            seq_lens=cpu(meta.seq_lens), caches=[])
        # Bound the prefix copies; never copy the full allocated KV cache.
        for name, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            for cache in module.kv_cache:
                cache_info = dict(layer=name, shape=tuple(cache.shape), prefix_copied=False)
                if cache.ndim==5 and cache.shape[0]==2 and 0<=root_position<512:
                    positions=torch.arange(root_position+1,device=cache.device)
                    block_size=cache.shape[2]
                    table=meta.block_table_tensor[0]
                    blocks=table[positions//block_size].long()
                    values=cache[:,blocks,positions%block_size]
                    cache_info.update(prefix_copied=True, prefix_kv=cpu(values),
                        nonfinite_prefix=int((~torch.isfinite(values)).sum()))
                detail['caches'].append(cache_info)
        if (os.getenv('REMTP_MTP_OPERATOR_AUDIT', '0') == '1'
                and record['round'] == 0 and not record['tree_passes']):
            from remtp.mtp_operator_audit import audit_pass
            with audit_pass(self.model, detail):
                result = tree_run(self, **kwargs)
        else:
            result = tree_run(self, **kwargs)
        detail.update(last_hidden=cpu(result[0]), hidden=cpu(result[1]),
                      nonfinite_last_hidden=int((~torch.isfinite(result[0])).sum()))
        record['tree_passes'].append(detail)
        return result

    dynamic._run_dynamic_tree_nodes = tree_observed
    _INSTALLED = True
    print(f"[ReMTP][DecisionTrace] installed max_rounds={MAX_ROUNDS} checkpoints={checkpoints} early_proposals={early_proposals} summary={bool(summary)} directory={root}", flush=True)
