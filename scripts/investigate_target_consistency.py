#!/usr/bin/env python3
"""Finite fixed-token Target diagnostics; no quality tuning or gold access."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_adaptive_round2 import Round2,settings,write_json


def manifest(source):
    rows=[json.loads(x) for x in (source/'diagnostics/anchor4/prefix_diagnostic.jsonl').read_text().splitlines()]
    wanted=[(109,1),(236,128),(157,1),(286,130)];out=[]
    for length,generated in wanted:
        matches=[r for r in rows if 'sequential_replay' in r and
                 len(r['sequential_replay']['prefix_ids'])==length and r['generated_length']==generated]
        if len(matches)!=1:raise ValueError(f'Ambiguous fixed prefix {length}/{generated}')
        r=matches[0];p=r['sequential_replay']
        out.append(dict(**p,request=r['request'],generated_length=generated,
                        prefix_sha256=hashlib.sha256(json.dumps(p['prefix_ids']).encode()).hexdigest(),
                        source=str(source/'diagnostics/anchor4/prefix_diagnostic.jsonl')))
    return out


def load_explicit_manifest(path):
    cases=json.loads(path.read_text());seen=set()
    if not cases:raise ValueError('Empty explicit prefix manifest')
    for case in cases:
        digest=hashlib.sha256(json.dumps(case['prefix_ids']).encode()).hexdigest()
        if digest!=case['prefix_sha256']:raise ValueError('Explicit prefix hash mismatch')
        if digest in seen:raise ValueError('Duplicate explicit prefix')
        seen.add(digest)
        if not case['prefix_ids'] or len(case['top_ids'])!=len(case['top_probs']):
            raise ValueError('Invalid explicit prefix probabilities')
    return cases


def score(controller,payload):
    req=urllib.request.Request(controller.base_url+'/v1/completions',
        data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=1800) as response:return json.load(response)


def summarize_native(directory,cases):
    import torch
    records=[]
    for path in (directory/'target_probe').glob('*/logits_*.pt'):
        r=torch.load(path,map_location='cpu',weights_only=True)
        records.append(r)
    checks=[]
    for case in cases:
        prefix=tuple(case['prefix_ids']);length=len(prefix);eos=case['eos_ids'][0]
        echo=[r for r in records if r['prompt_ids']==prefix+(eos,) and r['rows']==length]
        single=[r for r in records if r['prompt_ids']==prefix and r['rows']==1 and not r['output_ids']]
        decode=[r for r in records if r['prompt_ids']==prefix[:-1] and r['output_ids']==prefix[-1:] and r['rows']==1]
        if not echo or not single or not decode:
            raise ValueError(f'Missing actual Target rows at prefix {length}: {len(echo)}/{len(single)}/{len(decode)}')
        x=echo[0]['last_logits'].float();q=decode[0]['last_logits'].float()
        p=torch.softmax(x/.6,-1);s=torch.softmax(q/.6,-1)
        ids=case['top_ids'];ref=ids[0]
        old=torch.tensor(case['top_probs']).log()*.6
        err=float(((x[ids]-x[ref])-(old-old[0])).abs().max())
        shape_errors={str(n):float(((x[ids]-x[ref])-(v.float()[ids]-v.float()[ref])).abs().max())
                      for n,v in echo[0]['shape_probes'].items()}
        checks.append(dict(prefix_length=length,old_tree_vs_echo_top_log_odds=err,
                           echo_vs_decode_top_log_odds=float(((x[ids]-x[ref])-(q[ids]-q[ref])).abs().max()),
                           echo_vs_decode_tv=float((p-s).abs().sum()/2),
                           head_shape_top_log_odds=shape_errors,
                           hidden_echo_vs_decode=float((echo[0]['last_hidden']-decode[0]['last_hidden']).abs().max()),
                           echo_repeats=len(echo),single_repeats=len(single),decode_repeats=len(decode)))
    write_json(directory/'numerical_checks.json',checks)
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--source',required=True)
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--uniform-gemm',action='store_true')
    p.add_argument('--full-bf16-accum',action='store_true')
    p.add_argument('--bnb-row-block',type=int,choices=(0,16,64),default=0)
    p.add_argument('--target-backend',choices=('','FLASH_ATTN','TREE_ATTN','TRITON_ATTN'),default='')
    p.add_argument('--triton-2d',action='store_true')
    p.add_argument('--cache-modes',choices=('0','1','both'),default='both')
    p.add_argument('--attention-fp32',action='store_true')
    p.add_argument('--canonical-tree',action='store_true')
    p.add_argument('--manifest',type=Path);args=p.parse_args()
    args.prefix_cache=0;args.deep_diagnostic=False;args.target_probe=True
    controller=Round2(args)
    cases=load_explicit_manifest(args.manifest) if args.manifest else manifest(Path(args.source).resolve())
    write_json(controller.root/'prefix_manifest.json',cases)
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for cache in ((0,1) if args.cache_modes=='both' else (int(args.cache_modes),)):
            directory=controller.root/f'native_cache_{cache}';directory.mkdir(exist_ok=True)
            controller.current={**settings(method='target',head=1),'prefix_cache':cache}
            controller.jobs[directory.name]=dict(phase='TARGET_NUMERICAL_DIAG',total=24)
            write_json(controller.root/'jobs.json',controller.jobs)
            controller.diagnostic=False;controller.status('TARGET_NUMERICAL_DIAG',f'cache_{cache}',0,24)
            controller.start_server('target',directory)
            complete=0
            for case in cases:
                ids=case['prefix_ids'];eos=case['eos_ids'][0]
                for mode in ('echo','single','decode'):
                    for repeat in range(2):
                        payload=dict(model='TencentBAC/FastMTP',temperature=.6,seed=20261004,
                                     return_tokens_as_token_ids=True,return_token_ids=True)
                        if mode=='echo':payload.update(prompt=ids+[eos],echo=True,max_tokens=0,logprobs=20,prompt_logprobs=20)
                        elif mode=='single':payload.update(prompt=ids,max_tokens=1,logprobs=20,ignore_eos=True)
                        else:payload.update(prompt=ids[:-1],max_tokens=2,allowed_token_ids=[ids[-1]],ignore_eos=True)
                        response=score(controller,payload)
                        with (directory/'requests.jsonl').open('a') as f:
                            f.write(json.dumps(dict(payload=payload,response=response))+'\n');f.flush()
                        write_json(directory/f"{len(ids)}_{mode}_{repeat}.json",dict(prefix_sha256=case['prefix_sha256'],payload=payload,response=response))
                        complete+=1;controller.status('TARGET_NUMERICAL_DIAG',f'cache_{cache}',complete,24)
            controller.stop_server();summarize_native(directory,cases)
        controller.status('TARGET_NUMERICAL_DIAG_COMPLETE')
        return 0
    finally:controller.stop_server()


if __name__=='__main__':raise SystemExit(main())
