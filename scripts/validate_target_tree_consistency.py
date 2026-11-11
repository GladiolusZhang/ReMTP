#!/usr/bin/env python3
"""Fixed exact token prefixes: real tree root, ancestor rows and Target replay."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_adaptive_round2 import Round2,settings,write_json
from scripts.investigate_target_consistency import score


def snapshots(directory):
    import torch
    return [torch.load(p,map_location='cpu',weights_only=True)
            for p in (directory/'target_probe').glob('*/logits_*.pt')]


def comparison(a,b,ids,eos):
    import torch
    a=a.float();b=b.float();reference=ids[0]
    error=float(((a[ids]-a[reference])-(b[ids]-b[reference])).abs().max())
    eos_error=float(abs((a[eos]-a[reference])-(b[eos]-b[reference])))
    tv=float((torch.softmax(a/.6,-1)-torch.softmax(b/.6,-1)).abs().sum()/2)
    return dict(passed=error<=.15 and eos_error<=.15,max_log_odds_error=error,
                eos_log_odds_error=eos_error,tv=tv,tolerance=.15)


def validate_reuse(source, native, protocol, cases):
    """Accept saved roots only for the identical model, inputs and numerics."""
    keys=('model','packages','model_config_sha256','target_uniform_gemm',
          'target_full_bf16_accum','target_bnb_row_block','target_reference_backend',
          'target_triton_2d','target_attention_fp32','temperature','max_model_len','kv_cache',
          'no_think_template_sha256')
    for directory in (source,native):
        saved=json.loads((directory/'protocol.json').read_text())
        if directory==source and saved.get('canonical_tree_attention',False)!=protocol.get('canonical_tree_attention',False):
            raise ValueError('Reused tree ordering protocol mismatch')
        if any(saved.get(k)!=protocol.get(k) for k in keys):
            raise ValueError(f'Reused numerical protocol mismatch: {directory}')
        if json.loads((directory/'prefix_manifest.json').read_text())!=cases:
            raise ValueError(f'Reused prefix manifest mismatch: {directory}')
        saved_code=Path(saved['source_root'])
        critical=['remtp/target_numerics.py','remtp/target_diagnostics.py',
                         'remtp/fastmtp_verified_worker.py','remtp/fixed6_vllm.py',
                         'remtp/dynamic_tree_vllm.py','remtp/dynamic_mtp_tree.py',
                         'scripts/serve_fastmtp_verified.sh']
        if protocol.get('target_attention_fp32'):critical.append('remtp/target_attention_fp32.py')
        if protocol.get('canonical_tree_attention'):critical.append('remtp/canonical_tree_attention.py')
        for relative in critical:
            if hashlib.sha256((saved_code/relative).read_bytes()).digest()!=hashlib.sha256((ROOT/relative).read_bytes()).digest():
                raise ValueError(f'Reused execution source mismatch: {relative}')
    checks=json.loads((source/'root_gate.json').read_text())['checks']
    selected=[c for c in checks if c.get('cache')==0]
    if len(selected)!=24 or not all(c['passed'] for c in selected):
        raise ValueError('Reuse requires all 24 cache-off root checks to pass')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--native-dir',required=True)
    p.add_argument('--port',type=int,default=8000);p.add_argument('--uniform-gemm',action='store_true')
    p.add_argument('--full-bf16-accum',action='store_true')
    p.add_argument('--bnb-row-block',type=int,choices=(0,16,64),default=0)
    p.add_argument('--target-backend',choices=('','FLASH_ATTN','TREE_ATTN','TRITON_ATTN'),default='')
    p.add_argument('--triton-2d',action='store_true')
    p.add_argument('--attention-fp32',action='store_true')
    p.add_argument('--canonical-tree',action='store_true')
    p.add_argument('--cache-modes',choices=('0','1','both'),default='both')
    p.add_argument('--reuse-cache-off-from',type=Path);args=p.parse_args()
    args.prefix_cache=0;args.deep_diagnostic=False;args.target_probe=True
    controller=Round2(args);native=Path(args.native_dir).resolve()
    cases=json.loads((native/'prefix_manifest.json').read_text());write_json(controller.root/'prefix_manifest.json',cases)
    if args.reuse_cache_off_from:
        args.reuse_cache_off_from=args.reuse_cache_off_from.resolve()
        validate_reuse(args.reuse_cache_off_from,native,
                       json.loads((controller.root/'protocol.json').read_text()),cases)
    cache_modes=(0,1) if args.cache_modes=='both' else (int(args.cache_modes),)
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    all_checks=[];branches=[]
    def scheduled(phase,directory,total):
        controller.jobs[directory.name]=dict(phase=phase,total=total)
        write_json(controller.root/'jobs.json',controller.jobs)
    def saved(directory,payload,response):
        with (directory/'requests.jsonl').open('a') as f:
            f.write(json.dumps(dict(payload=payload,response=response))+'\n');f.flush()
    try:
        for cache in cache_modes:
            references=snapshots(native/f'native_cache_{cache}')
            for depth in (3,4):
                directory=controller.root/f'tree_d{depth}_cache{cache}';directory.mkdir(exist_ok=True)
                reused=args.reuse_cache_off_from if cache==0 else None
                if reused:
                    records=snapshots(reused/directory.name)
                    write_json(directory/'reused_from.json',dict(source=str(reused/directory.name),
                              request_count=8,read_only=True))
                else:
                    records=collect_roots(controller,directory,cache,depth,cases,scheduled,saved)
                for case in cases:
                    ids=tuple(case['prefix_ids']);eos=case['eos_ids'][0]
                    refs=[r for r in references if r['prompt_ids']==ids+(eos,) and r['rows']==len(ids)]
                    trees=[r for r in records if r['prompt_ids']==ids[:-1] and
                           r['output_ids']==ids[-1:] and 'logits_rows' in r]
                    if len(trees)!=2 or not refs:raise ValueError(f'Missing actual tree root: {len(ids)} {len(trees)}')
                    for r in trees:
                        if tuple(r['draft_ids'].tolist())!=r['actual_input_ids'][1:]:
                            raise ValueError('Captured draft IDs differ from actual Target inputs')
                        aligned=r['actual_input_ids'][0]==ids[-1] and r['actual_positions'][0]==len(ids)-1
                        check=comparison(r['first_logits'],refs[0]['last_logits'],case['top_ids'],eos)
                        check.update(prefix_length=len(ids),depth=depth,cache=cache,aligned=aligned)
                        check['passed'] &= aligned;all_checks.append(check)
                    all_checks.append(dict(prefix_length=len(ids),depth=depth,cache=cache,
                        repeated_root_identical=bool((trees[0]['first_logits']==trees[1]['first_logits']).all()),
                        passed=bool((trees[0]['first_logits']==trees[1]['first_logits']).all())))
                    if cache==0:
                        r=trees[0];paths=list(r['topology'])
                        primary=max((path for path in paths if all(rank==0 for rank in path)),key=len)
                        backup=next((path for path in paths if path[0]!=0),None)
                        if backup is None:
                            backup=next((path for path in paths if any(rank!=0 for rank in path)),None)
                        for path in (primary,backup):
                            if path is None:continue
                            ancestors=[paths.index(path[:n]) for n in range(1,len(path)+1)]
                            prefix=list(ids)+[int(r['draft_ids'][i]) for i in ancestors]
                            if r['actual_positions'][paths.index(path)+1]!=len(prefix)-1:
                                raise ValueError('Ancestor logical position mismatch')
                            branches.append(dict(prefix_ids=prefix,logits=r['logits_rows'][paths.index(path)+1],
                                                 eos=eos,depth=depth,path=path,base_prefix_sha256=case['prefix_sha256']))
                write_json(controller.root/'root_gate.json',dict(passed=all(r['passed'] for r in all_checks),checks=all_checks))
                if not all(r['passed'] for r in all_checks):
                    controller.status('TARGET_ROOT_FAILED');return 2
        # Verify branch rows against independent teacher-forced ancestor paths.
        directory=controller.root/'ancestor_target';directory.mkdir(exist_ok=True)
        scheduled('ANCESTOR_REPLAY',directory,len(branches))
        controller.current={**settings(method='target',head=1),'prefix_cache':0}
        controller.status('ANCESTOR_REPLAY','target',0,len(branches));controller.start_server('target',directory)
        for index,branch in enumerate(branches):
            payload=dict(model='TencentBAC/FastMTP',prompt=branch['prefix_ids']+[branch['eos']],max_tokens=0,
                         echo=True,prompt_logprobs=20,logprobs=20,temperature=.6,seed=20261004)
            response=score(controller,payload);write_json(directory/f'{index:02d}.json',dict(payload=payload,response=response))
            saved(directory,payload,response)
            controller.status('ANCESTOR_REPLAY','target',index+1,len(branches))
        controller.stop_server();records=snapshots(directory);checks=[]
        for branch in branches:
            prefix=tuple(branch['prefix_ids']);eos=branch['eos']
            refs=[r for r in records if r['prompt_ids']==prefix+(eos,) and r['rows']==len(prefix)]
            if not refs:raise ValueError('Missing ancestor reference')
            top=refs[0]['last_logits'].topk(8).indices.tolist()
            checks.append({**comparison(branch['logits'],refs[0]['last_logits'],top,eos),
                           'depth':branch['depth'],'path':branch['path'],'prefix_length':len(prefix),
                           'base_prefix_sha256':branch['base_prefix_sha256']})
        passed=all(x['passed'] for x in checks) and all(x['passed'] for x in all_checks)
        write_json(controller.root/'target_consistency_gate.json',dict(passed=passed,cache_modes=list(cache_modes),
                   reused_cache_off_from=str(args.reuse_cache_off_from) if args.reuse_cache_off_from else None,
                   root_checks=all_checks,branch_checks=checks))
        controller.status('TARGET_CONSISTENCY_PASSED' if passed else 'TARGET_CONSISTENCY_FAILED')
        return 0 if passed else 2
    finally:controller.stop_server()


def collect_roots(controller,directory,cache,depth,cases,scheduled,saved):
    scheduled('FIXED_TREE_ROOT',directory,8)
    controller.current={**settings(depth=depth,head=3,anchor=True,topk=True),'prefix_cache':cache}
    controller.diagnostic=False;controller.status('FIXED_TREE_ROOT',directory.name,0,8)
    controller.start_server('tree',directory)
    for index,case in enumerate(cases):
        for repeat in range(2):
            ids=case['prefix_ids']
            payload=dict(model='TencentBAC/FastMTP',prompt=ids[:-1],max_tokens=2,temperature=.6,
                         seed=20261004+repeat,allowed_token_ids=[ids[-1]],ignore_eos=True)
            response=score(controller,payload)
            write_json(directory/f'{len(ids)}_{repeat}.json',dict(payload=payload,response=response))
            saved(directory,payload,response)
            controller.status('FIXED_TREE_ROOT',directory.name,index*2+repeat+1,8)
    controller.stop_server()
    return snapshots(directory)


if __name__=='__main__':raise SystemExit(main())
