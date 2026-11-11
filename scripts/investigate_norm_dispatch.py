#!/usr/bin/env python3
"""Replay the saved first divergent norm with its real checkpoint weight."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from remtp.target_norm import fixed_rows_norm


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    import torch
    from safetensors import safe_open
    from vllm.model_executor.layers.layernorm import fused_add_rms_norm,rms_norm
    torch.set_num_threads(1)
    states={}
    for path in (args.source/'tree/target_probe').glob('*/state_*.pt'):
        state=torch.load(path,map_location='cpu',weights_only=True)
        states.setdefault(len(state['input_ids']),state)
    short=states[156];long=states[288]
    x,residual=short['ops']['6:input_norm']['inputs'][0]
    assert all(torch.equal(a,b) for a,b in zip((x,residual),long['ops']['6:input_norm']['inputs'][0]))
    key='model.layers.6.input_layernorm.weight'
    shard=json.loads((ROOT/'models/FastMTP/model.safetensors.index.json').read_text())['weight_map'][key]
    with safe_open(str(ROOT/'models/FastMTP'/shard),framework='pt',device='cpu') as f:
        weight=f.get_tensor(key).to(device='cuda',dtype=torch.bfloat16)
    eps=json.loads((ROOT/'models/FastMTP/config.json').read_text())['rms_norm_eps']
    def run(rows,fixed,fused=True):
        a=x.to('cuda').expand(rows,-1).contiguous().clone()
        b=residual.to('cuda').expand(rows,-1).contiguous().clone()
        if fixed:
            result=fixed_rows_norm(fused_add_rms_norm if fused else rms_norm,a,weight,eps,b if fused else None)
        else:
            result=fused_add_rms_norm(a,b,weight,eps) if fused else rms_norm(a,weight,eps)
        return tuple(v[0].cpu() for v in result) if fused else (result[0].cpu(),)
    old_small=run(156,False);old_long=run(288,False)
    expected=short['ops']['6:input_norm']['output'];expected_long=long['ops']['6:input_norm']['output']
    checks={}
    for fused in (False,True):
        reference=run(1,False,fused)
        checks['fused' if fused else 'plain']=[dict(rows=n,original_equal=all(torch.equal(a,b) for a,b in zip(reference,run(n,False,fused))),
             fixed_equal=all(torch.equal(a,b) for a,b in zip(reference,run(n,True,fused))))
             for n in (1,10,13,156,255,256,257,288,512,1024,2048)]
    result=dict(real_small_reproduced=all(torch.equal(a,b) for a,b in zip(old_small,expected)),
        real_large_reproduced=all(torch.equal(a,b) for a,b in zip(old_long,expected_long)),
        original_difference=float((old_small[0].float()-old_long[0].float()).abs().max()),
        fixed_large_matches_real_small=all(torch.equal(a,b) for a,b in zip(run(288,True),expected)),checks=checks,
        device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),epsilon=eps,
        remedy='Use the original CUDA operators in <=255-row views; no norm formula change')
    result['passed']=all(result[k] for k in ('real_small_reproduced','real_large_reproduced','fixed_large_matches_real_small')) and all(c['fixed_equal'] for v in checks.values() for c in v)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
