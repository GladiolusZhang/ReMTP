"""Opt-in head batching and request-prefix observation for Jetson experiments.

No forced EOS, string stop, logits penalty, or gold-answer access. Observation
uses the worker's committed output IDs, never speculative tree candidates.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import torch

_HEAD_INSTALLED = False
_OBSERVER_INSTALLED = False


def repeated_suffix(tokens, min_period=4, max_period=64, repetitions=3):
    """Three consecutive equal token blocks; bounded CPU work per round."""
    tail = tuple(tokens[-max_period * repetitions:])
    for period in range(min_period, min(max_period, len(tail) // repetitions) + 1):
        pattern = tail[-period:]
        if all(tail[-period * (i + 1):-period * i] == pattern
               for i in range(1, repetitions)):
            return True
    return False


def install_head_batching():
    global _HEAD_INSTALLED
    raw = os.getenv("REMTP_HEAD_BATCH_ROWS")
    if raw is None or _HEAD_INSTALLED:
        return
    rows = int(raw)
    if rows not in (1, 3):
        raise ValueError("REMTP_HEAD_BATCH_ROWS must be 1 or 3")
    from vllm.model_executor.models.mimo_mtp import MiMoMTP
    original = MiMoMTP.compute_logits
    probe = os.getenv("REMTP_PREFIX_DIAGNOSTIC_JSONL", "")
    probes = 0

    @functools.wraps(original)
    def batched(self, hidden):
        nonlocal probes
        result = original(self, hidden.expand(rows, -1).contiguous()) if rows > 1 and hidden.shape[0] == 1 else original(self, hidden)
        if result is not None and hidden.shape[0] == 1:
            result = result[:1]
            if probe and probes < 3:
                # Identical hidden input, real loaded BF16 head; do not draw RNG.
                one = original(self, hidden)
                three = original(self, hidden.expand(3, -1).contiguous())[:1]
                p, q = torch.softmax(one.float() / 0.6, -1), torch.softmax(three.float() / 0.6, -1)
                record = {"kind": "same_prefix_head", "max_abs_logit": float((one - three).abs().max()),
                          "probability_tv": float((p-q).abs().sum()/2),
                          "same_argmax": bool(torch.equal(p.argmax(-1), q.argmax(-1)))}
                Path(probe).parent.mkdir(parents=True, exist_ok=True)
                with open(probe, "a") as stream:
                    stream.write(json.dumps(record) + "\n")
                probes += 1
        return result

    MiMoMTP.compute_logits = batched
    _HEAD_INSTALLED = True
    print(f"[ReMTP][HeadBatch] input_rows={rows} (all applicable methods)", flush=True)


def install_generation_observer():
    global _OBSERVER_INSTALLED
    diagnostic = os.getenv("REMTP_PREFIX_DIAGNOSTIC_JSONL", "")
    state_directory = os.getenv("REMTP_STATE_DIAGNOSTIC_DIR", "")
    decision_directory = os.getenv("REMTP_DECISION_TRACE_DIR", "")
    local_rng = os.getenv("REMTP_REQUEST_LOCAL_BRANCH_RNG", "0") == "1"
    guard = os.getenv("REMTP_DYNAMIC_TREE_REPEAT_GUARD", "0") == "1"
    if _OBSERVER_INSTALLED or not (diagnostic or state_directory or decision_directory or local_rng or guard):
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from remtp import fixed6_vllm as fixed, dynamic_tree_vllm as dynamic
    prepare = GPUModelRunner._prepare_inputs
    last_request = None

    @functools.wraps(prepare)
    def observe(self, *args, **kwargs):
        nonlocal last_request
        req_ids = self.input_batch.req_ids
        if len(req_ids) == 1:
            req_id = req_ids[0]
            if local_rng and req_id != last_request:
                dynamic._BRANCH_PROPOSAL_ROUND = 0
            if req_id != last_request:
                fixed._RUNTIME.replay_saved = False
                fixed._RUNTIME.state_capture_calls = 0
                fixed._RUNTIME.state_proposal_calls = 0
            last_request = req_id
            state = self.requests[req_id]
            fixed._RUNTIME.generated_history = tuple(state.output_token_ids)
            fixed._RUNTIME.diagnostic_request = req_id
            fixed._RUNTIME.diagnostic_prompt_ids = tuple(state.prompt_token_ids or ())
        return prepare(self, *args, **kwargs)

    GPUModelRunner._prepare_inputs = observe
    if state_directory:
        install_state_diagnostics(state_directory)
    if decision_directory:
        from remtp.decision_trace import install_decision_trace
        install_decision_trace(decision_directory)
    if diagnostic:
        original = fixed.strict_tree_verify
        eos_ids = tuple(int(x) for x in os.getenv("REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS", "").split(",") if x)

        @functools.wraps(original)
        def diagnosed(topology, ids, draft, target, *args, **kwargs):
            result = original(topology, ids, draft, target, *args, **kwargs)
            history = getattr(fixed._RUNTIME, "generated_history", ())
            rows = []
            for detail in result.node_diagnostics:
                if detail.get("proposal_role") != "sampled_cactus_primary":
                    continue
                node = topology.nodes[detail["node"]]
                parent = 0 if node.parent is None else node.parent + 1
                p_eos = float(target[parent, list(eos_ids)].sum()) if eos_ids else 0.0
                py = float(detail["target_probability"])
                hy = float(detail["verification_probability"])
                scale = (1-hy)/max(1-py, 1e-30)
                h_eos = p_eos * scale if int(ids[node.index]) not in eos_ids else p_eos-py+hy
                rows.append({**detail, "token_id": int(ids[node.index]),
                             "target_eos_probability": p_eos, "verify_eos_probability": h_eos})
            record = {"kind": "verification", "request": getattr(fixed._RUNTIME, "diagnostic_request", ""),
                      "generated_length": len(history), "repeat_suffix": repeated_suffix(history),
                      "output_ids": result.output_token_ids, "terminal": result.terminal, "nodes": rows}
            # Save a few actual tree root rows for independent sequential replay.
            if len(history) in (0, 1) or (len(history) >= 128 and not getattr(fixed._RUNTIME, "replay_saved", False)):
                prompt = getattr(fixed._RUNTIME, "diagnostic_prompt_ids", ())
                if prompt:
                    top = target[0].topk(8)
                    record["sequential_replay"] = {"prefix_ids": list(prompt) + list(history),
                                                 "top_ids": top.indices.tolist(), "top_probs": top.values.tolist(),
                                                 "eos_ids": list(eos_ids), "eos_probs": target[0,list(eos_ids)].tolist()}
                    fixed._RUNTIME.replay_saved = len(history) >= 128
            Path(diagnostic).parent.mkdir(parents=True, exist_ok=True)
            with open(diagnostic, "a") as stream:
                stream.write(json.dumps(record) + "\n")
            return result

        fixed.strict_tree_verify = diagnosed
    _OBSERVER_INSTALLED = True


def install_state_diagnostics(directory):
    """Bounded, opt-in real tensors for the first 8 proposals of each request.

    All snapshots are CPU copies. No RNG draws, KV writes or hidden mutation.
    These probes add synchronization and must stay out of performance runs.
    """
    from remtp import fixed6_vllm as fixed, dynamic_tree_vllm as dynamic
    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm.model_executor.layers.attention.attention import Attention
    proposer=EagleProposer.propose_tree
    tree_run=dynamic._run_dynamic_tree_nodes

    def copy(value):return value.detach().cpu().clone()
    def folder():
        req=getattr(fixed._RUNTIME,'diagnostic_request','profile')
        path=Path(directory)/req;path.mkdir(parents=True,exist_ok=True);return path

    @functools.wraps(proposer)
    def proposed(self,*args,**kwargs):
        call=getattr(fixed._RUNTIME,'state_proposal_calls',0)
        fixed._RUNTIME.state_proposal_calls=call+1
        if call>=8:return proposer(self,*args,**kwargs)
        names=('batch_size','logits','positions','hidden_states','common_attn_metadata','slot_mappings')
        values={**dict(zip(names,args)),**kwargs}
        meta=values['common_attn_metadata']
        generator=(getattr(self._fixed6_sampling_metadata,'generators',None) or {}).get(0)
        record={'history':getattr(fixed._RUNTIME,'generated_history',()),
                'root_hidden':copy(values['hidden_states']),'root_logits':copy(values['logits']),
                'root_positions':copy(values['positions']),'seq_lens':copy(meta.seq_lens),
                'branch_round':dynamic._BRANCH_PROPOSAL_ROUND,
                'rng_before':generator.get_state().clone() if generator is not None else None}
        result=proposer(self,*args,**kwargs)
        record.update({'topology':fixed._RUNTIME.topology.choices,
                       'ids':copy(fixed._RUNTIME.draft_token_ids),
                       'q':copy(fixed._RUNTIME.draft_probs),
                       'rng_after':generator.get_state().clone() if generator is not None else None})
        torch.save(record,folder()/f'proposal_{call:02d}.pt')
        return result

    @functools.wraps(tree_run)
    def drafted(self,**kwargs):
        call=getattr(fixed._RUNTIME,'state_capture_calls',0)
        fixed._RUNTIME.state_capture_calls=call+1
        if getattr(fixed._RUNTIME,'state_proposal_calls',0)>8:return tree_run(self,**kwargs)
        ids=kwargs['node_ids'];meta=kwargs['base_common_metadata'];root=int(kwargs['root_positions'].reshape(-1)[0])
        record={'history':getattr(fixed._RUNTIME,'generated_history',()),
                'topology':fixed._RUNTIME.topology.choices,'node_ids':ids,
                'tokens':copy(torch.stack([kwargs['node_tokens'][i] for i in ids])),
                'hidden_inputs':copy(torch.stack([kwargs['node_hidden_inputs'][i] for i in ids])),
                'root_positions':copy(kwargs['root_positions']),'seq_lens':copy(meta.seq_lens)}
        for module in self.model.modules():
            if isinstance(module,Attention) and module.kv_cache:
                cache=module.kv_cache[0]
                if cache.ndim==5 and cache.shape[0]==2:
                    positions=torch.arange(root+1,device=cache.device)
                    block_size=cache.shape[2]
                    slots=meta.block_table_tensor[0][positions//block_size]*block_size+positions%block_size
                    record['prefix_kv']=copy(cache.flatten(1,2).index_select(1,slots.long()))
                    record['cache_layer']=getattr(module,'prefix','')
                    break
        output=tree_run(self,**kwargs)
        record.update({'last_hidden':copy(output[0]),'hidden':copy(output[1])})
        torch.save(record,folder()/f'tree_{call:02d}.pt')
        return output

    EagleProposer.propose_tree=proposed
    dynamic._run_dynamic_tree_nodes=drafted
    print(f'[ReMTP][StateProbe] bounded real hidden/Q/KV/RNG snapshots: {directory}',flush=True)
