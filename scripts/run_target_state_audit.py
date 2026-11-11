#!/usr/bin/env python3
"""Finite natural Target-history audit, followed by exact-prefix references."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_adaptive_round2 import Round2,settings,write_json
from scripts.investigate_target_consistency import score
from scripts.validate_target_tree_consistency import comparison,snapshots


def difference(a,b):
    import torch
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        if a.shape!=b.shape:raise ValueError(f'State tensor shape mismatch: {a.shape}/{b.shape}')
        return float((a.float()-b.float()).abs().max()) if a.numel() else 0.
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        if len(a)!=len(b):raise ValueError('State tuple mismatch')
        return max((difference(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):raise ValueError('State keys mismatch')
        return max((difference(a[k],b[k]) for k in a),default=0.)
    if a!=b:raise ValueError('State scalar mismatch')
    return 0.


def reference_state_path(directory,record):
    # vLLM can call compute_logits for sampling and prompt logprobs after
    # one model forward. Match the exact saved call, not directory cardinality.
    path=directory/'target_probe'/record['request']/f"state_{record['calls']-1:02d}.pt"
    if not path.is_file():raise ValueError('Missing exact reference state/logit pair')
    return path


def state_comparison(a,b):
    import torch
    if a['prefix_ids']!=b['prefix_ids']:raise ValueError('State logical prefixes differ')
    if set(a['ops'])!=set(b['ops']) or set(a['kv'])!=set(b['kv']):raise ValueError('Missing Target layer/operator')
    ops=[];kv=[]
    order=('input_norm','qkv','rope','attention','o_proj','post_norm','gate_up','activation','down_proj','layer')
    for key in sorted(a['ops'],key=lambda k:(int(k.split(':')[0]),order.index(k.split(':')[1]))):
        x=a['ops'][key];y=b['ops'][key]
        # Position/input tensor shapes differ across full prefill and tree.
        # row_snapshot already selected the same logical query before ops.
        ops.append(dict(operator=key,input_error=difference(x['inputs'],y['inputs']),
                        output_error=difference(x['output'],y['output'])))
    for layer in a['kv']:
        delta=(a['kv'][layer].float()-b['kv'][layer].float()).abs()
        per_token=delta.amax(dim=(0,2,3));changed=(per_token>0).nonzero().reshape(-1)
        kv.append(dict(layer=layer,max_error=float(delta.max()),changed_tokens=len(changed),
            first_changed_position=int(changed[0]) if len(changed) else None,
            first_changed_positions=changed[:12].tolist()))
    return dict(operators=ops,kv=kv,
        first_operator_difference=next((r for r in ops if r['input_error'] or r['output_error']),None),
        first_kv_difference=next((r for r in kv if r['max_error']),None))


def report(controller,result=None):
    lines=['# Adaptive RAVEN 第四轮：长生成现场与逐层定位','',
        '本阶段为有快照开销的诊断，不作为质量或速度结果。原max_tokens=2048，未缩短原请求预算。','',
        f"阶段：{controller.live_phase}。逐请求状态见status.json。",'']
    if result:
        lines += [f"捕获并独立重算前缀：{len(result['checks'])}；原287前缀匹配：{result['historical_prefix_matched']}。",
            f"自然输出与第三轮相同：{result['natural_output_identical']}。",
            f"现场Target数值门槛：{'通过' if result['passed'] else '失败（保持正式优化停止）'}。",'',
            '| 已生成 | 前缀 | top相对差 | EOS相对差 | 完整TV | 首次算子差异 | 首次KV层/位置 |',
            '| ---: | ---: | ---: | ---: | ---: | --- | --- |']
        for r in result['checks']:
            op=r['state']['first_operator_difference'];kv=r['state']['first_kv_difference']
            lines.append(f"| {r['generated_length']} | {r['prefix_length']} | {r['max_log_odds_error']:.6g} | {r['eos_log_odds_error']:.6g} | {r['tv']:.6g} | {op['operator'] if op else '无'} | {str(kv['layer'])+'/'+str(kv['first_changed_position']) if kv else '无'} |")
        lines += ['',f"路径源/目的复制检查：{result['commit_copies']}项；失败{result['commit_failures']}项。",
            '详细实际输入/位置、逐层算子和KV差异见state_checks.json及target_probe CPU快照。',
            '诊断完成不等于修复；首次差异证据审查后才实施单因素修复与长状态验收。','']
    (controller.root/'report.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    p.add_argument('--original-run',required=True,type=Path);p.add_argument('--port',type=int,default=8000)
    p.add_argument('--reuse-natural',type=Path)
    p.add_argument('--reuse-references',type=Path)
    p.add_argument('--rmsnorm-row-limit',type=int,choices=(0,255),default=0)
    p.add_argument('--lm-head-row-block',type=int,choices=(0,16),default=0)
    p.add_argument('--allow-new-trajectory',action='store_true')
    p.add_argument('--min-generated',type=int,default=120);p.add_argument('--max-generated',type=int,default=136)
    args=p.parse_args()
    args.prefix_cache=0;args.deep_diagnostic=False;args.target_probe=True;args.target_state_audit=True
    args.uniform_gemm=True;args.full_bf16_accum=True;args.bnb_row_block=64
    args.target_backend='TRITON_ATTN';args.triton_2d=True;args.attention_fp32=True;args.canonical_tree=True
    args.target_probe_min_generated=args.min_generated;args.target_probe_max_generated=args.max_generated
    args.target_probe_max_frames=16;args.target_state_reference_echo=False
    controller=Round2(args)
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    controller.status('NATURAL_TARGET_WINDOW','old',0,1);report(controller)
    try:
        if args.reuse_natural:
            reused=args.reuse_natural.resolve()
            old_protocol=json.loads((reused/'protocol.json').read_text())
            protocol=json.loads((controller.root/'protocol.json').read_text())
            keys=('model','packages','model_config_sha256','target_uniform_gemm',
                'target_full_bf16_accum','target_bnb_row_block','target_reference_backend',
                'target_triton_2d','target_attention_fp32','canonical_tree_attention','prefix_caching',
                'temperature','quality_max_tokens','max_model_len','kv_cache','no_think_template_sha256')
            if any(protocol[k]!=old_protocol[k] for k in keys) or any(protocol.get(k,0)!=old_protocol.get(k,0) for k in ('target_rmsnorm_row_limit','target_lm_head_row_block')):raise ValueError('Natural reuse protocol mismatch')
            old_code=Path(old_protocol['source_root'])
            checked=[]
            for path in (old_code/'remtp').glob('*.py'):
                if path.name in ('target_diagnostics.py','target_state_audit.py'):continue
                relative=path.relative_to(old_code)
                if path.read_bytes()!=(ROOT/relative).read_bytes():raise ValueError(f'Natural reuse runtime mismatch: {relative}')
                checked.append(str(relative))
            job=json.loads((reused/'jobs.json').read_text())['natural/old_window']
            if job['settings']!=settings(head=1) or job['limit']!=2048 or job['pairs']!=[[2950,29542]]:
                raise ValueError('Natural reuse request mismatch')
            records=[json.loads(x) for x in (reused/'natural/old_window/requests.jsonl').read_text().splitlines()]
            if len(records)!=1:raise ValueError('Natural reuse requires one completed request')
            (controller.root/'natural').symlink_to(reused/'natural',target_is_directory=True)
            controller.jobs['natural/old_window']=job;write_json(controller.root/'jobs.json',controller.jobs)
            write_json(controller.root/'reused_natural.json',dict(source=str(reused),runtime_files_checked=checked,
                diagnostic_changes='layer metadata validation and TRITON KV extraction only; original snapshots read-only'))
        else:
            records=controller.requests('natural','old_window',settings(head=1),[(controller.by_id[2950],42)],
                limit=2048,diagnostic=True)
        import torch
        torch.set_num_threads(1)
        directory=controller.root/'natural/old_window/target_probe'
        source=sorted(directory.glob('*/state_*.pt'))
        if not source:raise ValueError('No real Target-window states captured')
        cases=[]
        for path in source:
            state=torch.load(path,map_location='cpu',weights_only=True)
            raw=torch.load(path.with_name(path.name.replace('state_','logits_')),map_location='cpu',weights_only=True)
            if not state['tree']:continue
            prefix=state['prefix_ids'];digest=hashlib.sha256(json.dumps(list(prefix)).encode()).hexdigest()
            if state['input_ids'][0]!=prefix[-1] or state['positions'][0]!=len(prefix)-1:
                raise ValueError('Captured actual Target root does not match logical prefix')
            cases.append(dict(prefix_ids=list(prefix),prefix_sha256=digest,
                state_file=str(path),logits_file=str(path.with_name(path.name.replace('state_','logits_'))),
                generated_length=len(raw['output_ids'])))
        write_json(controller.root/'window_manifest.json',cases)
        historic=json.loads((args.original_run/'diagnostics/sequential_target/prefix_manifest.json').read_text())
        expected=next(c for c in historic if len(c['prefix_ids'])==287)
        matched=any(c['prefix_sha256']==expected['prefix_sha256'] for c in cases)
        if not matched and not (args.allow_new_trajectory and args.rmsnorm_row_limit):
            raise ValueError('Original 287-token failure prefix not reproduced; investigate trajectory before repair')
        ref_dir=controller.root/'references';ref_dir.mkdir(exist_ok=True)
        controller.jobs['references']=dict(phase='TARGET_WINDOW_REFERENCES',total=len(cases))
        write_json(controller.root/'jobs.json',controller.jobs)
        args.target_probe_min_generated=0;args.target_probe_max_generated=0
        args.target_state_reference_echo=True
        write_json(ref_dir/'capture_protocol.json',dict(min_generated=0,max_generated=0,reference_echo=True))
        controller.current=settings(method='target',head=1);controller.diagnostic=False
        controller.status('TARGET_WINDOW_REFERENCES','target',0,len(cases));report(controller)
        if args.reuse_references:
            reused=args.reuse_references.resolve()
            old_protocol=json.loads((reused/'protocol.json').read_text())
            protocol=json.loads((controller.root/'protocol.json').read_text())
            keys=('model','packages','model_config_sha256','target_uniform_gemm',
                'target_full_bf16_accum','target_bnb_row_block','target_reference_backend',
                'target_triton_2d','target_attention_fp32','canonical_tree_attention','prefix_caching',
                'temperature','max_model_len','kv_cache','no_think_template_sha256')
            if any(protocol[k]!=old_protocol[k] for k in keys) or any(protocol.get(k,0)!=old_protocol.get(k,0) for k in ('target_rmsnorm_row_limit','target_lm_head_row_block')):raise ValueError('Reference reuse protocol mismatch')
            old_code=Path(old_protocol['source_root'])
            for path in (old_code/'remtp').glob('*.py'):
                if path.name in ('target_diagnostics.py','target_state_audit.py'):continue
                relative=path.relative_to(old_code)
                if path.read_bytes()!=(ROOT/relative).read_bytes():raise ValueError(f'Reference reuse runtime mismatch: {relative}')
            reused_dir=reused/'references'
            if json.loads((reused_dir/'capture_protocol.json').read_text())!=dict(min_generated=0,max_generated=0,reference_echo=True):
                raise ValueError('Reference capture protocol mismatch')
            saved=[json.loads(x) for x in (reused_dir/'requests.jsonl').read_text().splitlines()]
            if len(saved)!=len(cases):raise ValueError('Reference reuse requires complete request set')
            for c,r in zip(cases,saved):
                p=r['payload']
                if (p['prompt']!=c['prefix_ids']+[controller.eos_ids[0]] or p['max_tokens']!=0 or
                    not p['echo'] or p['temperature']!=.6):raise ValueError('Reference reuse prefix/score mismatch')
            (controller.root/'references/requests.jsonl').write_text((reused_dir/'requests.jsonl').read_text())
            ref_dir=reused_dir
            write_json(controller.root/'reused_references.json',dict(source=str(reused),requests=len(saved),read_only=True))
            controller.status('TARGET_WINDOW_REFERENCES','target',len(cases),len(cases))
        else:
            controller.start_server('target',ref_dir)
            for i,c in enumerate(cases):
                payload=dict(model='TencentBAC/FastMTP',prompt=c['prefix_ids']+[controller.eos_ids[0]],
                    echo=True,max_tokens=0,temperature=.6,prompt_logprobs=20,logprobs=20,seed=20261004)
                response=score(controller,payload)
                with (ref_dir/'requests.jsonl').open('a') as f:
                    f.write(json.dumps(dict(payload=payload,response=response))+'\n');f.flush()
                controller.status('TARGET_WINDOW_REFERENCES','target',i+1,len(cases));report(controller)
            controller.stop_server()
        references=snapshots(ref_dir);ref_paths=list((ref_dir/'target_probe').glob('*/state_*.pt'))
        checks=[]
        for c in cases:
            prefix=tuple(c['prefix_ids']);a=torch.load(c['state_file'],map_location='cpu',weights_only=True)
            raw=torch.load(c['logits_file'],map_location='cpu',weights_only=True)
            refs=[r for r in references if r['prompt_ids']==prefix+(controller.eos_ids[0],) and r['rows']==len(prefix)]
            if len(refs)!=1:raise ValueError('Missing/duplicate raw reference')
            ref_path=reference_state_path(ref_dir,refs[0])
            b=torch.load(ref_path,map_location='cpu',weights_only=True)
            reference_top=refs[0]['last_logits'].topk(8).indices.tolist()
            ids=[reference_top[0],*sorted(set(raw['first_logits'].topk(8).indices.tolist()+reference_top+raw['draft_ids'].tolist())-{reference_top[0]})]
            check=comparison(raw['first_logits'],refs[0]['last_logits'],ids,controller.eos_ids[0])
            check.update(prefix_length=len(prefix),prefix_sha256=c['prefix_sha256'],
                generated_length=c['generated_length'],state=state_comparison(a,b))
            checks.append(check)
        commits=[json.loads(line) for path in directory.glob('*/commit_audit.jsonl') for line in path.read_text().splitlines()]
        original=[json.loads(line) for line in (args.original_run/'diagnostics/old/requests.jsonl').read_text().splitlines()]
        old=next(r for r in original if r['question_id']==2950)
        result=dict(passed=all(c['passed'] for c in checks) and all(c['identical'] for c in commits),
            historical_prefix_matched=matched,natural_output_identical=records[0]['output']==old['output'],
            natural_output_tokens=records[0]['output_tokens'],checks=checks,commit_copies=len(commits),
            commit_failures=sum(not c['identical'] for c in commits),source_root=str(ROOT))
        write_json(controller.root/'state_checks.json',result)
        controller.status('TARGET_STATE_AUDIT_COMPLETE');report(controller,result)
        return 0
    except Exception as e:
        write_json(controller.root/'failure.json',dict(type=type(e).__name__,error=str(e)))
        controller.status('TARGET_STATE_AUDIT_FAILED');report(controller)
        raise
    finally:controller.stop_server()


if __name__=='__main__':raise SystemExit(main())
