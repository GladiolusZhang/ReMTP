"""Opt-in, first-pass MTP operator snapshots. No RNG draws or model writes."""
from contextlib import contextmanager

import torch


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return tuple(snapshot(v) for v in value)
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    return value


def nonfinite(value):
    if isinstance(value, torch.Tensor):
        return int((~torch.isfinite(value)).sum()) if value.is_floating_point() else 0
    if isinstance(value, (tuple, list)):
        return sum(nonfinite(v) for v in value)
    if isinstance(value, dict):
        return sum(nonfinite(v) for v in value.values())
    return 0


def attention_state(module):
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    meta = context.attn_metadata[module.layer_name]
    fields = ('num_actual_tokens', 'max_query_len', 'max_seq_len', 'num_prefills',
              'num_decodes', 'num_prefill_tokens', 'num_decode_tokens',
              'query_start_loc', 'seq_lens', 'block_table', 'slot_mapping', 'tree_attn_bias')
    state = dict(metadata={k: snapshot(getattr(meta, k, None)) for k in fields},
                 forward_slot_mapping=snapshot(context.slot_mapping),
                 cudagraph_mode=str(context.cudagraph_runtime_mode))
    cache = module.kv_cache[context.virtual_engine]
    length = int(meta.seq_lens[0])
    if cache.ndim != 5 or cache.shape[0] != 2 or not 0 < length <= 512:
        raise ValueError('MTP audit requires TREE_ATTN KV and at most 512 visible positions')
    positions = torch.arange(length, device=cache.device)
    size = cache.shape[2]
    blocks = meta.block_table[0, positions // size].long()
    state['visible_kv'] = snapshot(cache[:, blocks, positions % size])
    state['visible_slots'] = snapshot(blocks * size + positions % size)
    return state


@contextmanager
def audit_pass(model, detail):
    """Copy inputs in pre-hooks, before fused norms can overwrite them."""
    suffixes = ('embed_tokens', 'token_layernorm', 'hidden_layernorm', 'input_proj',
                'input_layernorm', 'qkv_proj', 'rotary_emb', 'attn', 'o_proj',
                'post_attention_layernorm', 'gate_up_proj', 'act_fn', 'down_proj',
                'final_layernorm')
    ops = []
    pending = {}
    handles = []

    def before(module, args, kwargs, name):
        record = dict(name=name, inputs=snapshot((args, kwargs)))
        if name.endswith('.attn'):
            record['attention_before'] = attention_state(module)
        pending[name] = record
        ops.append(record)

    def after(module, args, kwargs, output, name):
        record = pending.pop(name)
        record.update(output=snapshot(output))
        record['nonfinite_inputs'] = nonfinite(record['inputs'])
        record['nonfinite_output'] = nonfinite(record['output'])
        if name.endswith('.attn'):
            record['attention_after'] = attention_state(module)

    detail['operators'] = ops
    try:
        for name, module in model.named_modules():
            if name.split('.')[-1] in suffixes:
                handles.append(module.register_forward_pre_hook(
                    lambda m, a, kw, n=name: before(m, a, kw, n), with_kwargs=True))
                handles.append(module.register_forward_hook(
                    lambda m, a, kw, o, n=name: after(m, a, kw, o, n), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
