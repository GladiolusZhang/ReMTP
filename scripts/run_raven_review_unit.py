#!/usr/bin/env python3
"""Frozen, finite evidence unit: profile -> Target checks -> train calibration."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_adaptive_round2 import Round2, settings, write_json
from scripts.run_adaptive_round4 import NUMERICAL_KEYS
from scripts.run_adaptive_round5 import output_signature
from scripts.run_adaptive_round6 import configs
from remtp.benchmark import chat_completion, fetch_metrics, metric_delta
from remtp.gsm8k_benchmark import PROMPT_TEMPLATE, benchmark_messages, extract_model_answer
from remtp.optimization_metrics import summarize

PROFILE_ORDER=('native','cactus','spec_cascade','B')
CALIBRATION_ORDER=tuple(reversed(PROFILE_ORDER))
CALIBRATION_IDS=(1532,2950,6009,3213,1231,5180,308,641)
LIMITS=dict(main_http=74,numeric=10,warmup=9,starts=9)
COMPONENTS={
    'target_math':['remtp/'+n+'.py' for n in ('target_numerics','target_norm','target_head',
        'canonical_tree_attention','target_attention_fp32')],
    'mtp_state':['remtp/'+n+'.py' for n in ('jetson_bf16_draft','mimo_mtp','probabilistic_mtp',
        'fixed6_vllm','fixed6_microtree','dynamic_tree_vllm')],
    'verification':['remtp/'+n+'.py' for n in ('dynamic_mtp_tree','cactus_mtp',
        'speculative_cascade','generation_guard')],
    'serving':['remtp/fastmtp_verified_worker.py','scripts/serve_fastmtp_verified.sh',
        'scripts/serve_fastmtp_jetson.sh','configs/fastmtp_no_think_chat_template.jinja'],
    'evaluation':['remtp/gsm8k_benchmark.py','remtp/benchmark.py','remtp/optimization_metrics.py'],
    'instrumentation':['remtp/runtime_profile.py','remtp/target_diagnostics.py'],
    'controller':['scripts/run_adaptive_round2.py','scripts/run_raven_review_unit.py'],
}


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())


class NumericRejected(RuntimeError):pass
class QualityRejected(RuntimeError):pass
class BudgetStop(RuntimeError):pass


def failure_kind(exc):
    if isinstance(exc,NumericRejected):return 'NUMERIC_REJECTED'
    if isinstance(exc,QualityRejected):return 'QUALITY_REJECTED'
    if isinstance(exc,BudgetStop):return 'BUDGET_STOPPED'
    if isinstance(exc,KeyboardInterrupt):return 'INTERRUPTED'
    return 'RUNTIME_ERROR'


class Ledger:
    def __init__(self,path,limits=LIMITS):
        self.path=Path(path);self.limits=dict(limits)
        self.entries=json.loads(self.path.read_text())['entries'] if self.path.exists() else {}
        if self.path.exists() and json.loads(self.path.read_text())['limits']!=self.limits:
            raise ValueError('Budget policy changed')
        self.save()

    def counts(self):
        return {kind:dict(attempted=sum(e['kind']==kind for e in self.entries.values()),
            completed=sum(e['kind']==kind and e['status']=='completed' for e in self.entries.values()),
            limit=limit) for kind,limit in self.limits.items()}

    def save(self):
        write_json(self.path,dict(limits=self.limits,counts=self.counts(),entries=self.entries,
            updated_utc=stamp(),policy='Reserve before invoking; an unsuccessful attempt consumes budget; no retries'))

    def reserve(self,kind,key,**metadata):
        identifier=kind+':'+key
        if identifier in self.entries:raise BudgetStop('Already attempted: '+identifier)
        if self.counts()[kind]['attempted']>=self.limits[kind]:raise BudgetStop('Budget exhausted: '+kind)
        self.entries[identifier]=dict(kind=kind,key=key,status='attempted',started_utc=stamp(),**metadata)
        self.save()

    def complete(self,kind,key,**metadata):
        entry=self.entries[kind+':'+key]
        if entry['status']!='attempted':raise ValueError('Duplicate completion')
        entry.update(status='completed',completed_utc=stamp(),**metadata);self.save()


def component_hashes(root):
    output={}
    for group,files in COMPONENTS.items():
        rows={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
        output[group]=dict(files=rows,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    return output


def saved_cases(reference):
    import torch
    paths=sorted((reference/'diagnostics/B_late/decision_trace').glob('*/verify_*.pt'))
    if len(paths)!=5:raise ValueError('Exactly five saved late frames required')
    cases=[]
    for index,path in enumerate(paths):
        frame=torch.load(path,map_location='cpu',weights_only=False)
        prefix=list(frame['prompt_ids'])+list(frame['history'])
        if not prefix or not torch.isfinite(frame['target']).all():
            raise ValueError('Incomplete/nonfinite saved Target frame')
        cases.append(dict(label=f'late_{index}',prefix_ids=prefix,
            generated_length=len(frame['history']),prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
            frame=str(path),frame_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if [c['generated_length'] for c in cases] != [129,514,1026,1792,2000]:
        raise ValueError('Late generated lengths changed')
    return cases


def profile_summary(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rounds=[r for r in rows if r['record_type']=='round']
    if not rounds or len(rounds)>32:raise ValueError('Profile missing or exceeds registered rounds')
    groups=defaultdict(list)
    for row in rows:
        if row['record_type']=='span':groups[row['domain']+'/'+row['component']].append(row)
    required=('target/target_forward','target/target_logits','draft/proposal','common/sampling','common/cache_update')
    if any(k not in groups for k in required):raise ValueError('Incomplete real profiling coverage')
    result=dict(rounds=len(rounds),requests=len({r['request'] for r in rounds}),components={},
        round_mean_wall=sum(r['wall_seconds'] for r in rounds)/len(rounds),
        round_mean_cuda_span=sum(r['cuda_span_seconds'] for r in rounds)/len(rounds),
        explicit_sync_seconds=sum(r['explicit_sync_seconds'] for r in rounds),
        timing_scope='Inclusive CPU walls and CUDA stream spans, not kernel sums or unprobed performance')
    for name,values in groups.items():
        result['components'][name]=dict(calls=len(values),
            mean_wall_seconds=sum(r['wall_seconds'] for r in values)/len(values),
            mean_cuda_span_seconds=sum(r['cuda_span_seconds'] for r in values)/len(values),
            cuda_span_per_round=sum(r['cuda_span_seconds'] for r in values)/len(rounds),
            wall_per_round=sum(r['wall_seconds'] for r in values)/len(rounds),
            top_level_calls=sum(r['parent_id']==0 for r in values))
    result['top_level_wall_per_round']=sum(r['wall_seconds'] for r in rows
        if r['record_type']=='span' and r['parent_id']==0)/len(rounds)
    return result


def compare_logits(tree,target,eos_ids):
    import torch
    a=tree.float();b=target.float()
    if a.shape!=b.shape or a.ndim!=1 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise NumericRejected('Incomplete/nonfinite raw logits')
    p=torch.softmax(a/.6,-1);q=torch.softmax(b/.6,-1)
    top=torch.unique(torch.cat((a.topk(min(8,a.numel())).indices,b.topk(min(8,b.numel())).indices,
                              torch.tensor(eos_ids,dtype=torch.long))))
    anchor=int(b.argmax())
    relative=(a-a[anchor])-(b-b[anchor])
    top_error=float(relative[top].abs().max())
    full_error=float(relative.abs().max())
    exact=bool(torch.equal(a,b))
    return dict(passed=exact,raw_equal=exact,max_abs_logit=float((a-b).abs().max()),
        top_eos_log_odds_error=top_error,full_relative_logit_error=full_error,
        probability_tv=float((p-q).abs().sum()/2),tree_eos=float(p[eos_ids].sum()),
        target_eos=float(q[eos_ids].sum()),legacy_tolerance=.15,
        legacy_tolerance_passed=top_error<=.15,
        policy='Any fresh raw difference stops for investigation; old tolerance is descriptive only')


class ReviewUnit(Round2):
    def __init__(self,args):
        super().__init__(args)
        self.protocol=json.loads((self.root/'protocol.json').read_text())
        reference=args.reference_dir.resolve()
        old=json.loads((reference/'protocol.json').read_text())
        for key in (*NUMERICAL_KEYS,'train_sha256','test_sha256','primary_prompt'):
            if self.protocol[key]!=old[key]:raise ValueError('Fixed condition mismatch: '+key)
        self.components=component_hashes(ROOT)
        runtime_reference=args.late_reference_dir.resolve()
        runtime_protocol=json.loads((runtime_reference/'protocol.json').read_text())
        for key in (*NUMERICAL_KEYS,'train_sha256','test_sha256','primary_prompt'):
            if self.protocol[key]!=runtime_protocol[key]:raise ValueError('Latest runtime condition mismatch: '+key)
        previous=component_hashes_for_reference(runtime_reference/'frozen_code')
        for group in previous:
            if self.components[group]!=previous[group]:raise ValueError('Unregistered math/runtime change: '+group)
        write_json(self.root/'component_hashes.json',dict(current=self.components,
            reference=previous,runtime_reference=str(runtime_reference),
            output_reference=str(reference),scope='Runtime identity uses latest cap-disabled B freeze; old repaired B only supplies output signatures; fresh numeric checks required'))
        self.cases=saved_cases(args.late_reference_dir.resolve())
        write_json(self.root/'numeric_cases.json',dict(cases=self.cases))
        selectors=[]
        for case in self.cases:
            prefix=case['prefix_ids']
            selectors.extend([dict(label=case['label']+'_tree',prompt_ids=prefix[:-1],output_ids=prefix[-1:]),
                              dict(label=case['label']+'_target',prompt_ids=prefix,output_ids=[])])
        write_json(self.root/'probe_selectors.json',dict(cases=selectors))
        self.variants={name:settings(method=name,head=1) for name in PROFILE_ORDER if name!='B'}
        self.variants['B']={**configs()['old_eos01'],'probability_ratio_cap':0.0}
        self.variants['target']=settings(method='target',head=1)
        self.ledger=Ledger(self.root/'ledger.json',getattr(args,'call_limits',LIMITS))
        self.state={};self.work={};self.halt=threading.Event();self.heartbeat=None
        self.started=time.monotonic()
        self.runtime_limit=getattr(args,'runtime_limit',86400)
        self.request_timeout=5400
        policy=dict(limits=self.ledger.limits,runtime_max_seconds=self.runtime_limit,retries=0,restart='no',
            automatic_adoption=False,heldout=False,formal=False,max_candidates_this_unit=0,
            profile_order=PROFILE_ORDER,calibration_order=CALIBRATION_ORDER,
            profile_ids=[308,641],profile_tokens=64,profile_max_rounds=32,profile_per_request=16,
            calibration_ids=CALIBRATION_IDS,seeds=[42,43],quality_tokens=2048,
            target_only_pairs=[[1532,42],[1532,43]],variants=self.variants,
            numeric_cases_sha256=hashlib.sha256((self.root/'numeric_cases.json').read_bytes()).hexdigest(),
            component_hashes={k:v['sha256'] for k,v in self.components.items()},
            runtime_reference=str(runtime_reference),output_reference=str(reference),
            generation_http_timeout_seconds=self.request_timeout,
            source_sha256=self.protocol['source_sha256'],
            profile_timing='CUDA stream spans/CPU inclusive walls; extra probes off; not unprobed throughput',
            order_note='Profile order and calibration order reversed; seeds interleaved per question; no formal order claim')
        policy.update(getattr(args,'registered_policy',{}))
        policy_path=self.root/'review_policy.json'
        if policy_path.exists() and json.loads(policy_path.read_text())!=json.loads(json.dumps(policy)):
            raise ValueError('Review policy changed; new directory required')
        write_json(policy_path,policy)
        self.ensure_budget([self.by_id[i] for i in CALIBRATION_IDS])

    def status(self,phase,variant='',completed=0,total=0):
        self.live_phase=phase
        counts=self.ledger.counts() if hasattr(self,'ledger') else {}
        self.state=dict(phase=phase,variant=variant,completed=completed,total=total,
            all_completed=counts.get('main_http',{}).get('completed',0),budget_counts=counts,
            current_work=dict(self.work),updated_utc=stamp(),controller_pid=os.getpid())
        write_json(self.root/'status.json',self.state)
        print(f'[{phase}] {variant} {completed}/{total}',flush=True)
        self.report()

    def pulse(self):
        while not self.halt.wait(15):
            device={}
            for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
                try:device[(p.parent/'type').read_text().strip()]=int(p.read_text())/1000
                except (OSError,ValueError):pass
            frequencies={}
            for p in Path('/sys/class/devfreq').glob('*/cur_freq'):
                try:frequencies[p.parent.name]=int(p.read_text())
                except (OSError,ValueError):pass
            server=self.server
            row=dict(updated_utc=stamp(),phase=self.state.get('phase'),work=dict(self.work),
                controller_pid=os.getpid(),server_pid=server.pid if server else None,
                server_alive=server is not None and server.poll() is None,
                thermal_celsius=device,frequencies_hz=frequencies,
                meminfo=Path('/proc/meminfo').read_text())
            write_json(self.root/'heartbeat.json',row)
            with (self.root/'telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def begin_work(self,phase,variant,**metadata):
        if time.monotonic()-self.started>self.runtime_limit:raise BudgetStop('Registered controller time limit reached')
        self.work=dict(phase=phase,variant=variant,started_utc=stamp(),**metadata)
        self.status(phase,variant)

    def warmup(self,directory):
        key=str(directory.relative_to(self.root))
        self.begin_work('WARMUP',key,max_tokens=8)
        self.ledger.reserve('warmup',key,max_tokens=8)
        began=time.perf_counter();response=super().warmup(directory)
        self.ledger.complete('warmup',key,seconds=time.perf_counter()-began,
            output_tokens=response['usage']['completion_tokens'])
        return response

    def start_server(self,variant,directory,profile=False):
        key=str(directory.relative_to(self.root))
        directory.mkdir(parents=True,exist_ok=True)
        self.begin_work('LOADING',variant,directory=key)
        self.ledger.reserve('starts',key,method=self.current['method'],profile=profile)
        super().start_server(variant,directory,profile)
        self.begin_work('EFFECTIVE_RUNTIME_CHECK',variant,directory=key)
        log=(directory/'server.log').read_text()
        method=self.current['method']
        if method=='target':
            expected='[ReMTP][FastMTPVerified] method=target speculative_decoding=off'
        else:
            expected=f'[ReMTP][FastMTPVerified] method={method} worker=active'
            if '[ReMTP][HeadBatch] input_rows=1' not in log:raise RuntimeError('Effective MTP head row mismatch')
        if expected not in log:raise RuntimeError('Effective worker missing: '+expected)
        if profile and 'automatic_topk_probe=off' not in log:raise RuntimeError('Bounded profile not installed')
        if method=='dynamic_tree' and ('relax_max_probability_ratio=0 ' not in log or 'lazy_residual=0' not in log):
            raise RuntimeError('B is not the registered unmodified configuration')
        self.ledger.complete('starts',key,health='passed',worker=method)

    def generate(self,phase,name,row,seed,limit):
        directory=self.root/phase/name;directory.mkdir(parents=True,exist_ok=True)
        seed+=row['question_id']*10
        key=f'{phase}/{name}/{row["question_id"]}/{seed}'
        self.begin_work(phase,name,question_id=row['question_id'],seed=seed,max_tokens=limit)
        before=fetch_metrics(self.base_url,30)
        self.ledger.reserve('main_http',key,question_id=row['question_id'],seed=seed,max_tokens=limit)
        began=time.perf_counter()
        response=chat_completion(self.base_url,'TencentBAC/FastMTP',
            benchmark_messages(PROMPT_TEMPLATE.format(question=row['question']),empty_system_prompt=True),.6,seed,limit,self.request_timeout)
        elapsed=time.perf_counter()-began;after=fetch_metrics(self.base_url,30)
        choice=response['choices'][0];text=choice['message']['content'] or ''
        pred,source=extract_model_answer(text)
        record=dict(question_id=row['question_id'],seed=seed,gold_answer=row['gold_answer'],
            predicted_answer=pred,answer_source=source,correct=pred==row['gold_answer'],
            finish_reason=choice['finish_reason'],output=text,
            output_tokens=response['usage']['completion_tokens'],client_seconds=elapsed,
            metrics=metric_delta(before,after),response_id=response['id'],max_tokens=limit)
        path=directory/'requests.jsonl'
        with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        self.ledger.complete('main_http',key,output_tokens=record['output_tokens'],seconds=elapsed)
        records=[json.loads(s) for s in path.read_text().splitlines()]
        write_json(directory/'summary.json',summarize(records))
        self.status(phase,name,len(records),2 if phase!='calibration' else 16)
        return record

    def numeric_call(self,case,role,directory):
        prefix=case['prefix_ids'];key=case['label']+'_'+role
        self.begin_work('TARGET_PREFIX',role,prefix_length=len(prefix),case=case['label'])
        payload=dict(model='TencentBAC/FastMTP',prompt=prefix[:-1] if role=='tree' else prefix,
            max_tokens=2 if role=='tree' else 1,temperature=.6,seed=20261007)
        if role=='tree':payload.update(allowed_token_ids=prefix[-1:],ignore_eos=True)
        # No echo/logprobs: only current/last rows, no prefix x vocabulary matrix.
        self.ledger.reserve('numeric',key,prefix_sha256=case['prefix_sha256'],role=role,max_tokens=payload['max_tokens'])
        request=urllib.request.Request(self.base_url+'/v1/completions',
            json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=1800) as response:result=json.load(response)
        write_json(self.root/'numeric'/f'{key}.json',dict(payload=payload,response=result))
        import torch
        files=list((directory/'target_probe').glob('*/logits_*.pt'))
        matches=[]
        for path in files:
            frame=torch.load(path,map_location='cpu',weights_only=True)
            if frame.get('case_label')==key:matches.append((path,frame))
        if len(matches)!=1:raise NumericRejected(f'Expected one actual raw frame for {key}, got {len(matches)}')
        path,frame=matches[0]
        if role=='tree':
            aligned=(frame['prompt_ids']==tuple(prefix[:-1]) and frame['output_ids']==tuple(prefix[-1:])
                and 'logits_rows' in frame and frame['actual_input_ids'][0]==prefix[-1]
                and frame['actual_positions'][0]==len(prefix)-1)
            logits=frame['first_logits']
        else:
            aligned=(frame['prompt_ids']==tuple(prefix) and not frame['output_ids']
                and frame['rows']==1 and frame['actual_input_ids'][-1]==prefix[-1]
                and frame['actual_positions'][-1]==len(prefix)-1)
            logits=frame['last_logits']
        if not aligned or frame['shape_probes']:raise NumericRejected('Raw Target frame/position/probe scope mismatch: '+key)
        self.ledger.complete('numeric',key,frame=str(path),aligned=True)
        return logits

    def run(self):
        with (self.root/'execution_attempt.json').open('x') as f:
            json.dump(dict(started_utc=stamp(),pid=os.getpid(),retries=0),f)
        self.heartbeat=threading.Thread(target=self.pulse,daemon=True);self.heartbeat.start()
        (self.root/'numeric').mkdir(exist_ok=True)
        self.args.target_probe=True
        self.args.target_probe_prefix_manifest=str(self.root/'probe_selectors.json')
        trees={}
        for name in PROFILE_ORDER:
            directory=self.root/'profile'/name;self.current=self.variants[name];self.diagnostic=False
            self.start_server(name,directory,profile=True)
            for qid in (308,641):self.generate('profile',name,self.by_id[qid],42,64)
            summary=profile_summary(directory/'profile.jsonl')
            if name=='B' and 'draft/tree_proposal' not in summary['components']:
                raise RuntimeError('Actual tree proposal was not measured')
            write_json(directory/'components.json',summary)
            self.report()
            if name=='B':
                for case in self.cases:trees[case['label']]=self.numeric_call(case,'tree',directory)
            self.stop_server()
        directory=self.root/'target_only'/'target';self.current=self.variants['target']
        self.start_server('target',directory)
        checks=[]
        import torch
        for case in self.cases:
            target=self.numeric_call(case,'target',directory)
            check=dict(case=case['label'],generated_length=case['generated_length'],
                       prefix_sha256=case['prefix_sha256'],**compare_logits(trees[case['label']],target,self.eos_ids))
            original=torch.load(case['frame'],map_location='cpu',weights_only=False)
            difference=torch.softmax(trees[case['label']].float()/.6,-1)-original['target'][0]
            check['fresh_tree_vs_saved_P_tv']=float(difference.abs().sum()/2)
            check['fresh_tree_vs_saved_P_max_abs']=float(difference.abs().max())
            check['saved_P_cpu_reduction_tolerance']=1e-6
            check['passed'] &= check['fresh_tree_vs_saved_P_tv']<=1e-6
            checks.append(check)
            write_json(self.root/'target_prefix_gate.json',dict(passed=all(c['passed'] for c in checks),
                complete=len(checks)==5,checks=checks,scope='Five actual saved histories, fresh teacher-forced tree/sequential raw rows; not all KV states'))
            if not check['passed']:raise NumericRejected('Fresh raw Target difference at '+case['label'])
        for seed in (42,43):self.generate('target_only','target',self.by_id[1532],seed,2048)
        self.stop_server();self.args.target_probe=False;self.args.target_probe_prefix_manifest=''
        for name in CALIBRATION_ORDER:
            directory=self.root/'calibration'/name;self.current=self.variants[name]
            self.start_server(name,directory)
            for index,qid in enumerate(CALIBRATION_IDS):
                for seed in ((42,43) if index%2==0 else (43,42)):
                    self.generate('calibration',name,self.by_id[qid],seed,2048)
            if name=='B':self.check_B_reproduction(directory)
            self.stop_server()
        self.make_analysis()
        self.status('COMPLETED_EVIDENCE_UNIT',completed=74,total=74)
        return 0

    def check_B_reproduction(self,directory):
        historical={}
        for p in (self.args.reference_dir/'regression').glob('B_seed*/requests.jsonl'):
            historical.update({(r['question_id'],r['seed']):r for r in (json.loads(s) for s in p.read_text().splitlines())})
        current=[json.loads(s) for s in (directory/'requests.jsonl').read_text().splitlines()]
        checks=[dict(question_id=r['question_id'],seed=r['seed'],
            passed=(r['question_id'],r['seed']) in historical and output_signature(r)==output_signature(historical[r['question_id'],r['seed']])) for r in current]
        write_json(self.root/'B_reproduction_gate.json',dict(passed=all(c['passed'] for c in checks),checks=checks,
            scope='16 complete B outputs; observation/controller changes, runtime math sources unchanged'))
        if not all(c['passed'] for c in checks):raise NumericRejected('B output changed despite unchanged math sources; inspect RNG/runtime')

    def make_analysis(self):
        data={name:json.loads((self.root/'calibration'/name/'summary.json').read_text()) for name in PROFILE_ORDER}
        profile={name:json.loads((self.root/'profile'/name/'components.json').read_text()) for name in PROFILE_ORDER}
        b=data['B'];comparisons={}
        for name in PROFILE_ORDER[:-1]:
            a=data[name]
            comparisons[name]=dict(correct_difference=b['correct']-a['correct'],
                stopped_correct_difference=b['stopped_correct']-a['stopped_correct'],
                truncated_difference=b['truncated']-a['truncated'],throughput_ratio=b['e2e_tok_s']/a['e2e_tok_s'],
                mean_time_ratio=b['mean_seconds']/a['mean_seconds'])
        write_json(self.root/'analysis.json',dict(calibration=data,comparisons=comparisons,profile=profile,
            candidate_selected=False,scope='Observed development diagnostics, no candidate adoption/independent formal claim'))
        self.report()

    def report(self):
        lines=['# 当前四方法成本与Target晚段诊断','',
            '有限证据单元：8条64-token profile、10次同前缀数值诊断、2条Target-only完整生成、64条2048训练校准。',
            '固定模型/公共数值/原提示与评分。预热和有探针请求排除出速度与质量结论。','']
        reuse=getattr(self.args,'registered_policy',{}).get('evidence_reuse')
        if reuse:
            lines+=['**观测修复后的有限续跑**：复用原单元8条profile与4个已通过前缀；本单元只新增1次末段数值补测、2条Target-only和64条校准，5次预热/启动。',
                    '原单元的失败、冻结、ledger及未完成门槛保留。两单元合计主HTTP上限74、数值尝试11、预热/启动10；总截止时间沿用原单元24小时。','']
        if hasattr(self,'state') and self.state:
            lines+=['当前阶段：'+self.state['phase']+'；当前组：'+self.state['variant']+'。','']
        if hasattr(self,'ledger'):
            lines+=['| 调用类别 | 已尝试 | 已完成 | 上限 |','| --- | ---: | ---: | ---: |']
            for kind,row in self.ledger.counts().items():lines.append(f"| {kind} | {row['attempted']} | {row['completed']} | {row['limit']} |")
            lines.append('')
        for path in sorted((self.root/'profile').glob('*/components.json')):
            data=json.loads(path.read_text());lines+=[f'## {path.parent.name} 分段计时','',
                f"观察{data['rounds']}轮；CPU平均轮跨度{data['round_mean_wall']:.4f}s。所有组件inclusive，嵌套不相加。CUDA为stream跨度，包含CPU发射间隙，不是纯kernel总和。",'',
                '| 组件 | 调用数 | CPU墙钟ms/轮 | CUDA跨度ms/轮 |','| --- | ---: | ---: | ---: |']
            for name,row in data['components'].items():
                lines.append(f"| {name} | {row['calls']} | {row['wall_per_round']*1000:.2f} | {row['cuda_span_per_round']*1000:.2f} |")
            lines.append('')
        lines+=['## 无探针训练校准（已观察开发题）','',
            '| 方法 | 请求 | 原判对 | 正常结束判对 | 截断 | E2E token/s | 平均秒 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for path in sorted((self.root/'calibration').glob('*/summary.json')):
            x=json.loads(path.read_text());lines.append(f"| {path.parent.name} | {x['requests']} | {x['correct']} | {x['stopped_correct']} | {x['truncated']} | {x['e2e_tok_s']:.4f} | {x['mean_seconds']:.1f} |")
        lines+=['','## Target与完整输出门槛','']
        for path in sorted(self.root.glob('*_gate.json')):
            gate=json.loads(path.read_text());lines.append(f"- {path.stem}：{'通过' if gate['passed'] else '未通过'}；范围见原gate。")
        if (self.root/'failure.json').exists():lines+=['','## 停止原因','',(self.root/'failure.json').read_text()]
        lines+=['','当前只补诊断/基准证据；新候选、独立留出及正式多种子比较尚未执行。',
            '关闭对话不影响systemd单元；0自动重试。失败/预算停止保留记录，不把质量拒绝称为运行崩溃。']
        temporary=self.root/'report.md.tmp';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(self.root/'report.md')

    def close(self):
        self.halt.set()
        if self.heartbeat:self.heartbeat.join(timeout=2)
        self.stop_server()


def component_hashes_for_reference(root):
    # The reference predates this controller. Only runtime domains are reused.
    result={}
    for group,files in COMPONENTS.items():
        if group in ('controller','instrumentation'):continue
        rows={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
        result[group]=dict(files=rows,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    return result


def arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--late-reference-dir',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();args.prefix_cache=0;args.deep_diagnostic=False;args.target_probe=False
    args.decision_trace=False;args.mtp_operator_audit=False;args.decision_trace_checkpoints=()
    args.uniform_gemm=True;args.full_bf16_accum=True;args.bnb_row_block=64
    args.rmsnorm_row_limit=255;args.target_backend='TRITON_ATTN';args.triton_2d=True
    args.lm_head_row_block=16;args.attention_fp32=True;args.canonical_tree=True
    args.target_probe_shape_rows=();args.target_probe_max_frames=1
    args.target_probe_prefix_manifest='';args.profile_max_rounds=32;args.profile_per_request=16
    return args


def main():
    args=arguments();controller=None;exit_code=1
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        controller=ReviewUnit(args)
        if args.check_only:exit_code=0;return 0
        exit_code=controller.run();return exit_code
    except BaseException as exc:
        kind=failure_kind(exc)
        if controller:
            write_json(controller.root/'failure.json',dict(kind=kind,type=type(exc).__name__,
                error=str(exc),traceback=traceback.format_exc(),failed_stage=controller.state.get('phase'),
                work=controller.work,updated_utc=stamp()))
            controller.status(kind)
        exit_code=3 if isinstance(exc,(NumericRejected,QualityRejected)) else 4 if isinstance(exc,BudgetStop) else 130 if isinstance(exc,KeyboardInterrupt) else 1
        print(f'{kind}: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return exit_code
    finally:
        if controller:
            controller.close()
            if not args.check_only:write_json(controller.root/'exit.json',dict(exit_code=exit_code,
                phase=controller.state.get('phase'),ended_utc=stamp(),server_cleaned=True))


if __name__=='__main__':raise SystemExit(main())
