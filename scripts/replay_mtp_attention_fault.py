#!/usr/bin/env python3
"""Four bounded GPU attention calls per real frame; no model reload or RNG."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    files = sorted(args.trace_root.glob('operator_probe/*/decision_trace/*/proposal_00.pt'))
    if len(files) != 2:
        raise ValueError('Exactly two first-pass real frames required')
    import torch
    from vllm.v1.attention.ops import triton_unified_attention as unified
    from remtp.target_attention_fp32 import kernel_unified_attention_2d
    unified.kernel_unified_attention_2d = kernel_unified_attention_2d
    evidence = []
    for path in files:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        op = next(o for o in saved['tree_passes'][0]['operators'] if o['name'].endswith('.attn'))
        state = op['attention_after']; meta = state['metadata']
        q = op['inputs'][0][0].cuda().reshape(1, 32, 128)
        kv = state['visible_kv'].cuda()
        length = kv.shape[1]; size = 16; blocks = (length+size-1)//size
        cache = torch.zeros(2, blocks, size, 8, 128, dtype=kv.dtype, device='cuda')
        cache.flatten(1, 2)[:, :length] = kv
        table = torch.arange(blocks+4, device='cuda', dtype=torch.int32).reshape(1, -1)
        # Owned poisoned padding reproduces the invalid column access without
        # reading another allocation: declared logical shape is still 1x1.
        padded = torch.full((4, 2), float('inf'), dtype=torch.float32, device='cuda')
        padded[0, 0] = 0
        bad_bias = padded[:1, :1]
        outputs = {}
        for name, bias in [('recorded_stride', meta['tree_attn_bias'].cuda()),
                           ('poisoned_stride2', bad_bias),
                           ('compact_stride1', bad_bias.clone(memory_format=torch.contiguous_format)),
                           ('no_bias_reference', None)]:
            out = torch.empty_like(q)
            unified.unified_attention(q=q, k=cache[0], v=cache[1], out=out,
                cu_seqlens_q=meta['query_start_loc'].cuda(), max_seqlen_q=1,
                seqused_k=meta['seq_lens'].cuda(), max_seqlen_k=length,
                softmax_scale=128**-.5, causal=True, window_size=(-1, -1),
                block_table=table, softcap=0., q_descale=None,
                k_descale=torch.ones(1, 8, device='cuda'),
                v_descale=torch.ones(1, 8, device='cuda'), qq_bias=bias)
            outputs[name] = out.cpu().clone()
        checks = {name: dict(nonfinite=int((~torch.isfinite(out)).sum()),
                  same_reference=bool(torch.equal(out, outputs['no_bias_reference'])))
                  for name, out in outputs.items()}
        evidence.append(dict(file=str(path), recorded_bias_stride=list(meta['tree_attn_bias'].stride()),
                             checks=checks))
    args.output.write_text(json.dumps(dict(attention_calls=8, evidence=evidence), indent=2)+'\n')
    print(args.output.read_text())


if __name__ == '__main__':
    main()
