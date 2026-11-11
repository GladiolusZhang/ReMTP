#!/usr/bin/env python3
"""Isolate final norm versus unquantized LM-head at a saved real root."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from remtp.target_norm import fixed_rows_norm
from remtp.target_numerics import fixed_rows_linear
from scripts.validate_target_tree_consistency import snapshots,comparison
from scripts.run_adaptive_round4 import exact_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    import torch
    from safetensors import safe_open
    from vllm.model_executor.layers.layernorm import rms_norm
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    natural=next(r for r in snapshots(args.source/'natural/old_window') if len(r['output_ids'])==124)
    reference=exact_reference(snapshots(args.source/'references'),natural['prompt_ids']+natural['output_ids'],151645)
    if not torch.equal(natural['first_hidden'],reference['last_hidden']):raise ValueError('Hidden states differ before final norm')
    index=json.loads((ROOT/'models/FastMTP/model.safetensors.index.json').read_text())['weight_map']
    weights={}
    for key in ('model.norm.weight','lm_head.weight'):
        with safe_open(str(ROOT/'models/FastMTP'/index[key]),framework='pt',device='cpu') as f:
            weights[key]=f.get_tensor(key).to(device='cuda',dtype=torch.bfloat16)
    epsilon=json.loads((ROOT/'models/FastMTP/config.json').read_text())['rms_norm_eps']
    hidden=natural['first_hidden'].to('cuda')
    ids=reference['last_logits'].topk(8).indices.tolist();ids=[ids[0],*sorted(set(ids+natural['first_logits'].topk(8).indices.tolist()+natural['draft_ids'].tolist())-{ids[0]})]
    checks=[];small_norm=None;small_logits=None;fixed_references={}
    for rows in (1,10,13,64,255,256,280,1024,2048):
        x=hidden.expand(rows,-1).contiguous()
        normalized=fixed_rows_norm(rms_norm,x,weights['model.norm.weight'],epsilon)
        logits=torch.nn.functional.linear(normalized,weights['lm_head.weight'])
        norm=normalized[-1].cpu();actual=logits[-1].cpu();candidates={}
        for block in (16,32,64):
            fixed=fixed_rows_linear(normalized,weights['lm_head.weight'],block=block)
            candidate=fixed[-1].cpu();fixed_references.setdefault(block,candidate)
            candidates[str(block)]=dict(raw_identical=bool(torch.equal(candidate,fixed_references[block])),
                versus_natural=comparison(candidate,natural['first_logits'],ids,151645),
                versus_reference=comparison(candidate,reference['last_logits'],ids,151645))
            del fixed
        if small_norm is None:small_norm=norm;small_logits=actual
        checks.append(dict(rows=rows,norm_identical=bool(torch.equal(norm,small_norm)),
            versus_natural=comparison(actual,natural['first_logits'],ids,151645),
            versus_reference=comparison(actual,reference['last_logits'],ids,151645),
            fixed_candidates=candidates))
        del logits,normalized,x
    result=dict(checks=checks,real_hidden_identical=True,weight_shape=list(weights['lm_head.weight'].shape),
        passed=all(c['norm_identical'] and all(v['raw_identical'] for v in c['fixed_candidates'].values()) for c in checks),
        limitation='Stable GEMM routes can change original small-M rounding; full-model/natural validation remains required')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
