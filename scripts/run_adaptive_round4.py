#!/usr/bin/env python3
"""Frozen finite pipeline: real-state gate, long/fresh KV, quality, speed.

No automatic retries or adoption. Diagnostics never count as timed quality.
The old state/failed candidates are only read, never overwritten or bypassed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_adaptive_round2 import Round2,settings,write_json
from scripts.investigate_target_consistency import score
from scripts.validate_target_tree_consistency import comparison,snapshots
from remtp.gsm8k_benchmark import load_gsm8k
from remtp.optimization_metrics import summarize,paired_gate,clustered_intervals

CHECKPOINTS=(0,1,31,32,63,64,127,128,255,256,511,512,1023,1024,1535,2040)
NUMERICAL_KEYS=('model','packages','model_config_sha256','target_uniform_gemm',
    'target_full_bf16_accum','target_bnb_row_block','target_rmsnorm_row_limit',
    'target_lm_head_row_block',
    'target_reference_backend','target_triton_2d','target_attention_fp32',
    'canonical_tree_attention','prefix_caching','temperature','max_model_len','kv_cache',
    'quality_max_tokens','no_think_template_sha256')


class GateRejected(Exception):
    """Expected rejection of a candidate, separate from a runtime failure."""


def numerical_source_check(directory,protocol):
    saved=json.loads((directory/'protocol.json').read_text())
    if any(saved.get(k)!=protocol.get(k) for k in NUMERICAL_KEYS):
        raise ValueError(f'Numerical source protocol mismatch: {directory}')
    old=Path(saved['source_root']);checked=[]
    for file in (ROOT/'remtp').glob('*.py'):
        if file.name in ('target_diagnostics.py','target_state_audit.py'):continue
        relative=file.relative_to(ROOT)
        if file.read_bytes()!=(old/relative).read_bytes():
            raise ValueError(f'Numerical execution source mismatch: {relative}')
        checked.append(str(relative))
    for relative in ('scripts/serve_fastmtp_jetson.sh','scripts/serve_fastmtp_verified.sh',
                     'configs/fastmtp_no_think_chat_template.jinja'):
        if (ROOT/relative).read_bytes()!=(old/relative).read_bytes():
            raise ValueError(f'Numerical serving source mismatch: {relative}')
        checked.append(relative)
    return checked


def exact_reference(records,prefix,eos):
    """Final prompt-logprob row, including a last chunk after 2048 tokens."""
    candidates=[]
    for record in records:
        positions=record['actual_positions']
        if (record['prompt_ids']==tuple(prefix) and record['rows']==1 and positions and
            positions[-1]==len(prefix)-1 and not record['output_ids']):
            candidates.append(record);continue
        last_chunk=(len(positions)>=2 and positions[-2]==len(prefix)-1 and record['rows']==len(positions)-1)
        # At a 2048 boundary, the root is the last input of an earlier full
        # chunk. EOS is a one-token final chunk with no prompt logits at all.
        earlier_full_chunk=(bool(positions) and positions[-1]==len(prefix)-1 and record['rows']==len(positions))
        if record['prompt_ids']==tuple(prefix)+(eos,) and (last_chunk or earlier_full_chunk):
            candidates.append(record)
    if not candidates:raise ValueError(f'Missing final Target prompt-logprob row: {len(prefix)}')
    # The sampling call precedes prompt logprobs. If the final chunk has two
    # inputs, both calls have M=1: only the latter is the requested root row.
    requests={r['request'] for r in candidates}
    if len(requests)!=1:raise ValueError('Duplicate Target prefix reference requests')
    return max(candidates,key=lambda r:r['calls'])


def compare_raw(actual,reference,draft,eos):
    ids=reference.float().topk(8).indices.tolist()
    union=set(ids+actual.float().topk(8).indices.tolist()+list(draft))
    result=comparison(actual,reference,[ids[0],*sorted(union-{ids[0]})],eos)
    import torch
    a=actual.float();b=reference.float()
    result.update(compared_ids=sorted(union),full_relative_logit_error=float(((a-a[ids[0]])-(b-b[ids[0]])).abs().max()),
                  raw_identical=bool(torch.equal(actual,reference)))
    return result


class Round4(Round2):
    def __init__(self,args):
        super().__init__(args)
        self.protocol=json.loads((self.root/'protocol.json').read_text())
        policy=dict(checkpoints=CHECKPOINTS,natural_quality_budget=2048,
            natural_configs=['d3h1_old','d3h3_topk','d4h1_topk','d4h3_topk'],
            tuning=['strict','d3_delta1','d3_delta05','d3_delta025','d4_delta025','anchor4_delta05'],
            baseline_methods=['native','cactus','spec_cascade'],baseline_depths=[3,4],
            reference_mode='full prefix recomputation, sample last-row raw logits, discard one generated token',
            quality_before_speed=True,formal_questions=100,seeds=[42,43,44],
            prefill_source=str(args.prefill_dir),state_source=str(args.state_audit_dir),
            old_run=str(args.original_run),failure_policy='archive and stop; no automatic retries')
        path=self.root/'round4_policy.json'
        if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(policy)):
            raise ValueError('Round4 policy changed; use a new directory')
        write_json(path,policy)

    def start_server(self,variant,directory,profile=False):
        options=dict(settings=self.current,numerical_protocol={k:self.protocol[k] for k in NUMERICAL_KEYS},
            target_probe=bool(self.args.target_probe),target_state_audit=bool(getattr(self.args,'target_state_audit',False)),
            checkpoints=list(getattr(self.args,'target_probe_checkpoints',())),
            max_frames=getattr(self.args,'target_probe_max_frames',32),prefix_diagnostic=self.diagnostic)
        path=directory/'runtime_options.json'
        if path.exists() and json.loads(path.read_text())!=options:
            raise ValueError('Effective stage runtime options changed; use a new directory')
        write_json(path,options)
        return super().start_server(variant,directory,profile)

    def gate(self,name,result):
        result.update(source_sha256=self.protocol['source_sha256'],source_root=str(ROOT),
            numerical_protocol={k:self.protocol[k] for k in NUMERICAL_KEYS})
        write_json(self.root/f'{name}_gate.json',result)
        if not result['passed']:
            self.status('STOPPED_'+name.upper()+'_GATE')
            raise GateRejected(name)

    def await_initial_gate(self):
        def prerequisite(directory,script,extra):
            if directory.exists():
                if not (directory/'protocol.json').exists():raise ValueError('Incomplete prerequisite directory; refusing unrecorded retry')
                return
            directory.mkdir();(directory/'v4_evidence').symlink_to(self.root/'v4_evidence',target_is_directory=True)
            command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts'/script),'--run-dir',str(directory),
                '--rmsnorm-row-limit','255','--lm-head-row-block','16','--port',str(self.args.port),*extra]
            with (directory/'controller.log').open('a') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        self.status('PREFILL_VALIDATION')
        prerequisite(self.args.prefill_dir,'investigate_prefill_history.py',
            ['--natural-source',str(ROOT/'results/target_state_round4_20261004_v2')])
        prefill=json.loads((self.args.prefill_dir/'prefill_checks.json').read_text())
        checked=numerical_source_check(self.args.prefill_dir,self.protocol)
        if len(prefill)!=5 or any(c['first_operator_difference'] or c['first_kv_difference'] for c in prefill.values()):
            self.gate('prefill',dict(passed=False,checks=prefill))
        self.gate('prefill',dict(passed=True,source=str(self.args.prefill_dir),files_checked=checked))
        self.status('REAL_STATE_VALIDATION')
        prerequisite(self.args.state_audit_dir,'run_target_state_audit.py',
            ['--original-run',str(self.args.original_run),'--allow-new-trajectory'])
        self.status('WAITING_REAL_STATE_GATE')
        directory=self.args.state_audit_dir;deadline=time.monotonic()+2700
        while not (directory/'state_checks.json').exists():
            if (directory/'failure.json').exists():
                raise ValueError('Real-state audit runtime failure: '+(directory/'failure.json').read_text())
            if time.monotonic()>deadline:raise TimeoutError('Real-state audit did not finish within 45 minutes')
            time.sleep(15)
        checked=numerical_source_check(directory,self.protocol)
        result=json.loads((directory/'state_checks.json').read_text())
        required=(result['passed'] and len(result['checks'])>=4 and
            result['commit_copies']>0 and result['commit_failures']==0)
        self.gate('real_state',dict(passed=bool(required),source=str(directory),files_checked=checked,
            checks=result['checks'],commit_copies=result['commit_copies'],
            natural_output_tokens=result['natural_output_tokens'],
            historical_prefix_matched=result['historical_prefix_matched'],natural_output_identical=result['natural_output_identical'],
            note='New fixed LMHead may change natural rounding/trajectory; all old exact failure IDs are independently rechecked in fresh_state'))
        # Result is written before the diagnostic service completes its cleanup.
        for _ in range(20):
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1',self.args.port))!=0:break
            time.sleep(1)
        else:raise ValueError('Previous diagnostic server still occupies the GPU port')

    def scoring(self,name,config,payloads,phase):
        directory=self.root/'state_validation'/name;directory.mkdir(parents=True,exist_ok=True)
        key=str(directory.relative_to(self.root))
        job=dict(phase=phase,total=len(payloads),settings=config,
            payload_sha256=hashlib.sha256(json.dumps(payloads,sort_keys=True).encode()).hexdigest())
        if key in self.jobs and self.jobs[key]!=job:raise ValueError('Scoring job protocol changed')
        self.jobs[key]=job;write_json(self.root/'jobs.json',self.jobs)
        path=directory/'requests.jsonl'
        if path.exists() and path.read_bytes() and not path.read_bytes().endswith(b'\n'):
            data=path.read_bytes();path.with_suffix('.partial').write_bytes(data[data.rfind(b'\n')+1:]);path.write_bytes(data[:data.rfind(b'\n')+1])
        done=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if [r['payload'] for r in done]!=payloads[:len(done)]:raise ValueError('Scoring checkpoint mismatch')
        self.status(phase,name,len(done),len(payloads))
        if len(done)<len(payloads):
            self.current=config;self.diagnostic=False;self.start_server(name,directory)
            for i,payload in enumerate(payloads[len(done):],len(done)):
                response=score(self,payload)
                with path.open('a') as f:
                    f.write(json.dumps(dict(payload=payload,response=response))+'\n');f.flush();os.fsync(f.fileno())
                self.status(phase,name,i+1,len(payloads))
            self.stop_server()
        return directory

    def reference_payload(self,prefix):
        # Only the final distribution is needed. Full-prefix prompt logprobs
        # materialize >2.4 GiB logits+logsoftmax at 2048 tokens on this device.
        # This preserves independent full prefix computation and captures the
        # complete raw final vocabulary before sampling, without that matrix.
        return dict(model='TencentBAC/FastMTP',prompt=list(prefix),max_tokens=1,
            logprobs=1,ignore_eos=True,temperature=.6,seed=20261004)

    def long_state_gate(self):
        self.args.target_probe=True;self.args.target_probe_checkpoints=CHECKPOINTS
        self.args.target_probe_max_frames=32
        configs={'d3h1_old':settings(head=1),
            'd3h3_topk':settings(head=3,topk=True),
            'd4h1_topk':settings(depth=4,head=1,topk=True),
            'd4h3_topk':settings(depth=4,head=3,topk=True)}
        for name,config in configs.items():
            pairs=self.tuning[:4] if name=='d3h1_old' else [self.tuning[1],self.tuning[3]]
            self.requests('state_natural',name,config,pairs,limit=2048,diagnostic=True)
        # A different request precedes q7311 in the original group; independently
        # restarting and running it first checks order and process reproduction.
        repeat=self.requests('state_natural','restart_order',configs['d3h1_old'],[self.tuning[3]],limit=2048,diagnostic=True)
        original=[json.loads(s) for s in (self.root/'state_natural/d3h1_old/requests.jsonl').read_text().splitlines()][3]
        self.gate('reproduction',dict(passed=all(original[k]==repeat[0][k] for k in ('output','output_tokens','finish_reason')),
            question_id=7311,seed=original['seed'],request_budget=2048))
        self.args.target_probe_checkpoints=()
        import torch
        torch.set_num_threads(1)
        expected_dir=self.root/'state_expected';expected_dir.mkdir(exist_ok=True)
        manifest=[];by_prefix={};longest=()
        for name in configs:
            for record in snapshots(self.root/'state_natural'/name):
                if 'logits_rows' not in record or not record['output_ids'] or len(record['prompt_ids'])<=64:continue
                prefix=record['prompt_ids']+record['output_ids']
                if len(prefix)>len(longest):longest=prefix
                if record['actual_input_ids'][0]!=prefix[-1] or record['actual_positions'][0]!=len(prefix)-1:
                    raise ValueError('Natural root actual position/token misaligned')
                choices=tuple(record['topology']);rows=[((),0,prefix)]
                if any(c>=127 for c in record.get('selected_checkpoints',())):
                    primary=max((p for p in choices if all(i==0 for i in p)),key=len)
                    backup=next((p for p in reversed(choices) if p[0]!=0),None)
                    if backup is None:backup=next((p for p in reversed(choices) if any(i!=0 for i in p)),None)
                    for path in (primary,backup):
                        if path is None:continue
                        indices=[choices.index(path[:n]) for n in range(1,len(path)+1)]
                        branch=prefix+tuple(int(record['draft_ids'][i]) for i in indices)
                        row=choices.index(path)+1
                        if record['actual_positions'][row]!=len(branch)-1:raise ValueError('Natural branch position misaligned')
                        rows.append((path,row,branch))
                for path,row,ids in rows:
                    digest=hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()
                    by_prefix.setdefault(ids,self.reference_payload(ids))
                    file=expected_dir/f'{len(manifest):04d}.pt'
                    torch.save(dict(logits=record['logits_rows'][row],draft=record['draft_ids']),file)
                    manifest.append(dict(config=name,prefix_ids=list(ids),prefix_sha256=digest,
                        generated_length=len(record['output_ids']),path=path,expected_file=str(file),
                        probe_checkpoints=record.get('selected_checkpoints',())))
        if not manifest or len(longest)<2049:
            raise ValueError('Natural coverage did not reach a real 2048-token prefix; no long-state gate bypass')
        write_json(self.root/'long_state_manifest.json',manifest)
        refs=self.scoring('long_target',settings(method='target',head=1),list(by_prefix.values()),'LONG_TARGET_REPLAY')
        records=snapshots(refs);checks=[]
        for case in manifest:
            expected=torch.load(case['expected_file'],map_location='cpu',weights_only=True)
            reference=exact_reference(records,case['prefix_ids'],self.eos_ids[0])
            checks.append(dict(**compare_raw(expected['logits'],reference['last_logits'],expected['draft'].tolist(),self.eos_ids[0]),
                config=case['config'],path=case['path'],prefix_length=len(case['prefix_ids']),
                prefix_sha256=case['prefix_sha256'],generated_length=case['generated_length']))
        self.gate('long_state',dict(passed=all(c['passed'] for c in checks),checks=checks,
            longest_actual_prefix=len(longest),checkpoint_policy=CHECKPOINTS,
            limitation='Reached checkpoints and two ancestor paths; coverage is finite, not a universal equality proof'))
        self.fresh_state_gate(longest)
        self.args.target_probe=False

    def fresh_state_gate(self,longest):
        import torch
        cases=[];seen=set()
        for run in (self.args.original_run,):
            path=run/'diagnostics/sequential_target/prefix_manifest.json'
            if path.exists():
                for case in json.loads(path.read_text()):
                    ids=tuple(case['prefix_ids'])
                    if ids not in seen:cases.append(ids);seen.add(ids)
        # Preserve actual old late-window and failed D4 ancestor prefixes, not
        # merely another prefix with the same token count.
        path=ROOT/'results/target_state_round4_20261004_v2/window_manifest.json'
        if not path.exists():raise ValueError('Missing original six late-window cases')
        for case in json.loads(path.read_text()):
            ids=tuple(case['prefix_ids'])
            if ids not in seen:cases.append(ids);seen.add(ids)
        for name in ('fp32_tree','cache_off_ancestor'):
            path=ROOT/f'results/target_probability_consistency_20261004/{name}/ancestor_target/requests.jsonl'
            if not path.exists():raise ValueError(f'Missing historical ancestor source: {name}')
            for line in path.read_text().splitlines():
                ids=tuple(json.loads(line)['payload']['prompt'][:-1])
                if ids not in seen:cases.append(ids);seen.add(ids)
        # Include canonical-tree's four earlier ancestor-replay cases as well.
        path=ROOT/'results/target_consistency_deep_audit_20261004/prefix_manifest.json'
        if path.exists():
            for case in json.loads(path.read_text()):
                ids=tuple(case['prefix_ids'])
                if ids not in seen:cases.append(ids);seen.add(ids)
        for length in (63,64,65,127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048,2049):
            ids=tuple(longest[:length])
            if len(ids)!=length:raise ValueError('Missing genuine long prefix for boundary replay')
            if ids not in seen:cases.append(ids);seen.add(ids)
        write_json(self.root/'fresh_boundary_manifest.json',[dict(prefix_ids=list(ids),prefix_sha256=hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()) for ids in cases])
        directory=self.scoring('boundary_target',settings(method='target',head=1),
            [self.reference_payload(ids) for ids in cases],'BOUNDARY_TARGET')
        references=snapshots(directory);checks=[];first={}
        for depth in (3,4):
            for head in (1,3):
                for repeat in (0,1):
                    ordered=cases if repeat==0 else list(reversed(cases))
                    payloads=[dict(model='TencentBAC/FastMTP',prompt=list(ids[:-1]),max_tokens=2,
                        temperature=.6,seed=20261004+repeat,allowed_token_ids=[ids[-1]],ignore_eos=True) for ids in ordered]
                    directory=self.scoring(f'boundary_d{depth}_h{head}_r{repeat}',settings(depth=depth,head=head,topk=True),payloads,'FRESH_BOUNDARY_TREE')
                    records=snapshots(directory)
                    for ids in cases:
                        trees=[r for r in records if r['prompt_ids']==ids[:-1] and r['output_ids']==ids[-1:] and 'logits_rows' in r]
                        if len(trees)!=1:raise ValueError('Missing/duplicate fresh root')
                        raw=trees[0];ref=exact_reference(references,ids,self.eos_ids[0]);key=(depth,head,ids)
                        if raw['actual_input_ids'][0]!=ids[-1] or raw['actual_positions'][0]!=len(ids)-1:raise ValueError('Fresh root position/token mismatch')
                        result=compare_raw(raw['first_logits'],ref['last_logits'],raw['draft_ids'].tolist(),self.eos_ids[0])
                        if repeat:
                            result['restart_order_identical']=bool(torch.equal(raw['first_logits'],first[key]))
                            result['passed'] &= result['restart_order_identical']
                        else:first[key]=raw['first_logits'].clone()
                        result.update(depth=depth,head=head,repeat=repeat,prefix_length=len(ids),
                            prefix_sha256=hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest())
                        checks.append(result)
                    self.gate('fresh_partial',dict(passed=all(c['passed'] for c in checks),checks=checks,
                        note='Partial result: never permits quality until fresh_state final gate exists'))
        self.gate('fresh_state',dict(passed=all(c['passed'] for c in checks),checks=checks,
            case_count=len(cases),cache_modes=[0],all_depth_head_combinations=True,
            limitation='Teacher-forced root checks are diagnostics, never quality scores'))

    def quality_speed(self):
        # Train-only selection. No online gold is provided to model workers.
        strict=settings(head=3,topk=True,delta=0.0,mix=0.0)
        self.requests('tuning','strict',strict,self.tuning)
        variants={'strict':strict,
            'd3_delta1':settings(head=3,topk=True),
            'd3_delta05':settings(head=3,topk=True,delta=.5,protect=True),
            'd3_delta025':settings(head=3,topk=True,delta=.25,mix=.25,protect=True),
            'd4_delta025':settings(depth=4,head=3,topk=True,delta=.25,mix=.25,protect=True),
            'anchor4_delta05':settings(depth=4,head=3,anchor=True,topk=True,delta=.5,protect=True)}
        old=self.requests('tuning','old',settings(head=1),self.tuning)
        tuned={n:self.requests('tuning',n,config,self.tuning) for n,config in variants.items()}
        selected={'old':settings(head=1)};baselines={}
        for method in ('native','cactus','spec_cascade'):
            options={}
            for depth in (3,4):
                config=settings(method=method,depth=depth)
                records=self.requests('baseline_selection',f'{method}_d{depth}',config,self.tuning)
                options[depth]=(config,records,summarize(records))
            depth=max(options,key=lambda d:(options[d][2]['stopped_correct'],options[d][2]['correct'],
                -options[d][2]['truncated'],-options[d][2]['mean_seconds']))
            selected[method]=options[depth][0];baselines[method]=options[depth][1]
        comparisons={name:{method:paired_gate(before,records,require_speed=False)
            for method,before in {'old':old,**baselines}.items()} for name,records in tuned.items()}
        eligible=[n for n in variants if all(c['passed'] for c in comparisons[n].values())]
        candidates=sorted(eligible,key=lambda n:(-summarize(tuned[n])['stopped_correct'],summarize(tuned[n])['mean_seconds']))[:2]
        self.gate('tuning_quality',dict(passed=bool(candidates),candidates=candidates,comparisons=comparisons,
            note='Correct/stopped-correct must not decrease, truncation must not increase; speed has no veto here'))
        reference=self.blocks('regression','old',selected['old'],self.regression,[42,43]);passed={};checks={}
        for n in candidates:
            records=self.blocks('regression',n,variants[n],self.regression,[42,43])
            checks[n]=paired_gate(reference,records,require_speed=False)
            if checks[n]['passed']:passed[n]=records
        self.gate('regression',dict(passed=bool(passed),comparisons=checks))
        chosen=min(passed,key=lambda n:(-summarize(passed[n])['stopped_correct'],summarize(passed[n])['mean_seconds']))
        selected['adaptive']=variants[chosen]
        write_json(self.root/'selection.json',dict(candidate_name=chosen,settings=selected,source='train tuning/regression only'))
        heldout={n:self.blocks('heldout',n,config,self.heldout,[42,43,44]) for n,config in selected.items()}
        checks={n:paired_gate(r,heldout['adaptive'],require_speed=False) for n,r in heldout.items() if n!='adaptive'}
        self.gate('heldout_quality',dict(passed=all(c['passed'] for c in checks.values()),comparisons=checks))
        # Test answers and IDs are accessed only after candidate freezing and the independent heldout gate.
        test=load_gsm8k(ROOT/'data/gsm8k/test.jsonl')
        old_requests=ROOT/'results/raven_jetson_gsm8k_n100_fp4_2048/native/requests.jsonl'
        excluded={json.loads(s)['question_id'] for s in old_requests.read_text().splitlines()}
        rows=sorted(random.Random(20261003).sample([r for r in test if r['question_id'] not in excluded],100),key=lambda r:r['question_id'])
        self.ensure_budget(rows)
        write_json(self.root/'formal_manifest.json',dict(question_ids=[r['question_id'] for r in rows],excluded_old_ids=sorted(excluded),settings=selected,seeds=[42,43,44]))
        names=list(selected);formal={}
        for i,seed in enumerate((42,43,44)):
            for n in names[i:]+names[:i]:
                formal.setdefault(n,[]).extend(self.requests('formal',f'{n}_seed{seed}',selected[n],[(row,seed) for row in rows]))
        comparisons={n:{**paired_gate(r,formal['adaptive'],require_speed=False),**clustered_intervals(r,formal['adaptive'])}
            for n,r in formal.items() if n!='adaptive'}
        quality=all(c['passed'] for c in comparisons.values())
        speed=all(summarize(formal['adaptive'])['mean_seconds']<summarize(formal[n])['mean_seconds'] and
            comparisons[n]['throughput_ratio_95ci'][0]>1 for n in ('native','cactus','spec_cascade'))
        # An honest negative result is saved and stops here; no test-set tuning.
        self.gate('formal',dict(passed=quality and speed,quality=quality,speed=speed,comparisons=comparisons))
        self.status('COMPLETE');return 0

    def run(self):
        self.await_initial_gate()
        self.long_state_gate()
        return self.quality_speed()

    def report(self):
        super().report()
        lines=['','## 第四轮执行与已验证原因','',
            '本轮先通过新冻结候选的数值门槛，再进行质量和速度评测；阶段失败会停止，结果不自动采用。',
            '已完成的定位证据：原第129位置在第6层融合RMSNorm因256行分派而分叉；'
            '固定≤255行调用原CUDA算子后，156/288×TREE/TRITON逐层/KV对照一致。',
            '随后证实BF16 LMHead在小M/大M存在舍入差异；固定16行的真实权重微测在M1～2048逐位稳定，'
            '并保持观测现场的概率。新完整模型/长状态与正式质量速度结果由本目录gate文件决定。','',
            '| 验收阶段 | 产物 |','| --- | --- |',
            '| 预填充与原自然现场 | prefill_validation/、real_state_validation/ |',
            '| 多配置长历史与祖先 | long_state_manifest.json、long_state_gate.json |',
            '| 全部旧故障、块边界、重启/逆序 | fresh_boundary_manifest.json、fresh_state_gate.json |',
            '| 截断、正确率与三个基线选参 | tuning_quality_gate.json |',
            '| 独立回归、留出与正式测试 | regression/、heldout/、formal/ |','',
            '主质量输出预算始终2048；固定种子只是复现控制，不等同于概率一致性。'
            '有探针的state_natural运行只作诊断；质量/速度组关闭探针并重新计时。','']
        summaries=sorted((self.root/'state_natural').glob('*/summary.json'))
        if summaries:
            lines+=['## 真实自然轨迹诊断（含观测开销，不作性能结论）','',
                '| 配置 | 已完成请求 | 正常结束 | 截断 |','| --- | ---: | ---: | ---: |']
            for p in summaries:
                s=json.loads(p.read_text());lines.append(f"| {p.parent.name} | {s['requests']} | {s['stopped']} | {s['truncated']} |")
        with (self.root/'report.md').open('a') as f:f.write('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    p.add_argument('--prefill-dir',required=True,type=Path)
    p.add_argument('--state-audit-dir',required=True,type=Path)
    p.add_argument('--original-run',required=True,type=Path)
    p.add_argument('--port',type=int,default=8000);p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    args.prefix_cache=0;args.deep_diagnostic=False;args.target_probe=False
    args.uniform_gemm=True;args.full_bf16_accum=True;args.bnb_row_block=64
    args.rmsnorm_row_limit=255;args.target_backend='TRITON_ATTN';args.triton_2d=True
    args.lm_head_row_block=16
    args.attention_fp32=True;args.canonical_tree=True
    controller=None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        controller=Round4(args)
        if args.check_only:
            controller.status('PREFLIGHT_COMPLETE');return 0
        return controller.run()
    except GateRejected as exc:
        write_json(controller.root/'failure.json',dict(type='GateRejected',category='candidate_gate',
            gate=str(exc),error='候选门槛未通过，按计划停止；不是服务崩溃，未采用候选',
            source_root=str(ROOT),timestamp_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))
        print(f'候选门槛未通过：{exc}；结果已保存，按计划停止。',flush=True)
        return 3
    except Exception as exc:
        if controller:
            write_json(controller.root/'failure.json',dict(type=type(exc).__name__,category='runtime_or_protocol',error=str(exc),source_root=str(ROOT),timestamp_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))
            controller.status('STOPPED_FAILURE')
        raise
    finally:
        if controller:controller.stop_server()


if __name__=='__main__':raise SystemExit(main())
