#!/usr/bin/env python3
"""Finite backend x prompt-length factorial at a real divergent prompt token."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_adaptive_round2 import Round2,settings,write_json
from scripts.investigate_target_consistency import score
from scripts.run_target_state_audit import state_comparison,reference_state_path
from scripts.validate_target_tree_consistency import snapshots


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    p.add_argument('--natural-source',required=True,type=Path);p.add_argument('--port',type=int,default=8000)
    p.add_argument('--rmsnorm-row-limit',type=int,choices=(0,255),default=0)
    p.add_argument('--lm-head-row-block',type=int,choices=(0,16),default=0)
    args=p.parse_args()
    args.prefix_cache=0;args.deep_diagnostic=False;args.target_probe=True;args.target_state_audit=True
    args.uniform_gemm=True;args.full_bf16_accum=True;args.bnb_row_block=64
    args.target_backend='TRITON_ATTN';args.triton_2d=True;args.attention_fp32=True;args.canonical_tree=True
    args.target_probe_min_generated=0;args.target_probe_max_generated=0;args.target_probe_max_frames=8
    args.target_state_reference_echo=False;args.target_state_row_position=129
    controller=Round2(args)
    import torch
    torch.set_num_threads(1)
    source=None
    for path in (args.natural_source/'natural/old_window/target_probe').glob('*/logits_*.pt'):
        record=torch.load(path,map_location='cpu',weights_only=True)
        if len(record['output_ids'])==131:source=record;break
    if source is None:raise ValueError('Original natural 131-token root absent')
    inputs={'short':list(source['prompt_ids']),
        'long':list(source['prompt_ids']+source['output_ids'])+[controller.eos_ids[0]]}
    write_json(controller.root/'input_manifest.json',dict(inputs=inputs,row_position=129))
    states={}
    try:
        for method in ('tree','target'):
            directory=controller.root/method;directory.mkdir(exist_ok=True)
            controller.jobs[method]=dict(phase='PREFILL_FACTORIAL',total=2)
            write_json(controller.root/'jobs.json',controller.jobs)
            controller.current=settings(head=1,method='dynamic_tree' if method=='tree' else 'target')
            controller.diagnostic=False;controller.status('PREFILL_FACTORIAL',method,0,2)
            controller.start_server(method,directory)
            for i,(label,ids) in enumerate(inputs.items()):
                payload=dict(model='TencentBAC/FastMTP',prompt=ids,max_tokens=0,echo=True,
                    prompt_logprobs=1,logprobs=1,temperature=.6,seed=29542)
                response=score(controller,payload)
                with (directory/'requests.jsonl').open('a') as f:
                    f.write(json.dumps(dict(payload=payload,response=response))+'\n');f.flush()
                controller.status('PREFILL_FACTORIAL',method,i+1,2)
            controller.stop_server()
            records=snapshots(directory)
            for label,ids in inputs.items():
                selected=[r for r in records if r['prompt_ids']==tuple(ids) and r['rows']==len(ids)-1]
                if len(selected)!=1:raise ValueError('Missing exact prefill raw row')
                states[f'{method}_{label}']=torch.load(reference_state_path(directory,selected[0]),map_location='cpu',weights_only=True)
        pairs=[('tree_short','tree_long'),('target_short','target_long'),
               ('tree_short','target_short'),('tree_long','target_long'),('tree_short','target_long')]
        comparisons={f'{a}_vs_{b}':state_comparison(states[a],states[b]) for a,b in pairs}
        write_json(controller.root/'prefill_checks.json',comparisons)
        controller.status('PREFILL_FACTORIAL_COMPLETE')
        lines=['# 第四轮：提示token129预填充来源定位','',
            '真实输入分别为原156token提示和原287前缀+EOS=288token；固定观测同一逻辑位置129。',
            '只改变后端/输入长度，公共数值与权重保持。四请求用于定位，不计质量或速度。','',
            '| 对照 | 首次算子差异 | 首次KV层/位置 |','| --- | --- | --- |']
        for name,c in comparisons.items():
            op=c['first_operator_difference'];kv=c['first_kv_difference']
            lines.append(f"| {name} | {op['operator'] if op else '无'} | {str(kv['layer'])+'/'+str(kv['first_changed_position']) if kv else '无'} |")
        (controller.root/'report.md').write_text('\n'.join(lines)+'\n')
        return 0
    finally:controller.stop_server()


if __name__=='__main__':raise SystemExit(main())
