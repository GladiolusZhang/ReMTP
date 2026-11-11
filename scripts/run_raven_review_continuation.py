#!/usr/bin/env python3
"""One registered probe correction, followed by the unspent evidence work."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_raven_review_unit import (ReviewUnit,NumericRejected,BudgetStop,
    CALIBRATION_ORDER,CALIBRATION_IDS,PROFILE_ORDER,compare_logits,
    component_hashes_for_reference,profile_summary,saved_cases,stamp,write_json)

CONTINUATION_LIMITS=dict(main_http=66,numeric=1,warmup=5,starts=5)


def utc_epoch(value):
    return datetime.strptime(value,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc).timestamp()


def load_frame(origin,case,role):
    import torch
    directory=origin/('profile/B' if role=='tree' else 'target_only/target')/'target_probe'
    label=case['label']+'_'+role
    frames=[]
    for p in directory.glob('*/logits_*.pt'):
        frame=torch.load(p,map_location='cpu',weights_only=True)
        if frame.get('case_label')==label:frames.append((p,frame))
    if len(frames)!=1:raise ValueError('Ambiguous/missing reusable frame: '+label)
    return frames[0]


def aligned(frame,case,role):
    prefix=case['prefix_ids']
    if frame.get('case_label')!=case['label']+'_'+role or frame['shape_probes']:return False
    if role=='tree':
        return (frame['prompt_ids']==tuple(prefix[:-1]) and frame['output_ids']==tuple(prefix[-1:])
            and 'logits_rows' in frame and frame['actual_input_ids'][0]==prefix[-1]
            and frame['actual_positions'][0]==len(prefix)-1)
    return (frame['prompt_ids']==tuple(prefix) and not frame['output_ids'] and frame['rows']==1
        and frame['actual_input_ids'][-1]==prefix[-1] and frame['actual_positions'][-1]==len(prefix)-1)


def validate_origin(origin,cases,eos_ids,code_root=ROOT):
    """Reuse only this observed failure, after rechecking artifacts on CPU."""
    import torch
    failure=json.loads((origin/'failure.json').read_text())
    exit_info=json.loads((origin/'exit.json').read_text())
    if (failure['kind']!='NUMERIC_REJECTED' or
            failure['error']!='Raw Target frame/position/probe scope mismatch: late_4_target' or
            exit_info['exit_code']!=3 or not exit_info['server_cleaned']):
        raise ValueError('Continuation is restricted to the observed chunked-prefill probe failure')
    counts=json.loads((origin/'ledger.json').read_text())['counts']
    expected=dict(main_http=(8,8),numeric=(10,9),warmup=(5,5),starts=(5,5))
    if any((counts[k]['attempted'],counts[k]['completed'])!=v for k,v in expected.items()):
        raise ValueError('Origin budget/progress changed; do not duplicate work')
    if list((origin/'calibration').glob('*/requests.jsonl')) or list((origin/'target_only').glob('*/requests.jsonl')):
        raise ValueError('Long generation/calibration already attempted')
    if json.loads((origin/'numeric_cases.json').read_text())['cases']!=cases:
        raise ValueError('Actual prefix scope changed')
    current=component_hashes_for_reference(code_root)
    old=component_hashes_for_reference(origin/'frozen_code')
    if current!=old:raise ValueError('Runtime math/state changed; original evidence cannot be reused')
    recorded=json.loads((origin/'component_hashes.json').read_text())['current']
    if any(recorded[k]!=old[k] for k in old if k in recorded):raise ValueError('Origin math fingerprint mismatch')
    if set(old)-set(recorded) not in (set(),{'evaluation'}):
        raise ValueError('Unexpected missing runtime fingerprint')
    old_gate=json.loads((origin/'target_prefix_gate.json').read_text())
    if not old_gate['passed'] or old_gate['complete'] or len(old_gate['checks'])!=4:
        raise ValueError('Four previously exact prefix comparisons required')
    files=[origin/n for n in ('failure.json','exit.json','ledger.json','numeric_cases.json',
        'component_hashes.json','target_prefix_gate.json','execution_attempt.json')]
    for name in PROFILE_ORDER:
        directory=origin/'profile'/name
        summary=profile_summary(directory/'profile.jsonl')
        if summary!=json.loads((directory/'components.json').read_text()) or summary['requests']!=2:
            raise ValueError('Profile evidence changed: '+name)
        if name=='B' and 'draft/tree_proposal' not in summary['components']:
            raise ValueError('Tree proposal coverage missing')
        if len((directory/'requests.jsonl').read_text().splitlines())!=2:
            raise ValueError('Expected exactly two completed profile requests: '+name)
        files.extend(directory/n for n in ('profile.jsonl','components.json','requests.jsonl'))
    trees={};checks=[]
    for index,case in enumerate(cases):
        path,tree=load_frame(origin,case,'tree');files.append(path)
        if not aligned(tree,case,'tree'):raise ValueError('Saved tree frame is not aligned')
        trees[case['label']]=tree['first_logits']
        if not torch.isfinite(trees[case['label']]).all():raise ValueError('Nonfinite reusable tree logits')
        path,target=load_frame(origin,case,'target');files.append(path)
        if index==4:
            if (aligned(target,case,'target') or len(case['prefix_ids'])!=2223 or
                    target['actual_positions'][-1]!=2047 or len(target['actual_input_ids'])!=2048):
                raise ValueError('Expected the recorded incomplete prefill chunk')
            continue
        if not aligned(target,case,'target'):raise ValueError('Saved sequential frame is not aligned')
        check=dict(case=case['label'],generated_length=case['generated_length'],
            prefix_sha256=case['prefix_sha256'],**compare_logits(tree['first_logits'],target['last_logits'],eos_ids))
        original=torch.load(case['frame'],map_location='cpu',weights_only=False)
        check['fresh_tree_vs_saved_P_tv']=float((torch.softmax(tree['first_logits'].float()/.6,-1)-original['target'][0]).abs().sum()/2)
        check['passed'] &= check['fresh_tree_vs_saved_P_tv']<=1e-6
        if not check['passed']:raise ValueError('Reusable prefix failed unchanged gate')
        if any(old_gate['checks'][index][k]!=check[k] for k in ('case','raw_equal','max_abs_logit','fresh_tree_vs_saved_P_tv')):
            raise ValueError('Original prefix gate does not match its tensors')
        check['inherited_from']=str(origin);checks.append(check)
    hashes={str(p.relative_to(origin)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return dict(passed=True,origin=str(origin),artifact_hashes=hashes,
        runtime_hashes=old,inherited_profile_http=8,inherited_exact_prefixes=4,
        origin_numeric_attempted=10,origin_numeric_completed=9,
        correction_scope='Only late_4_target; previous provisional position2047 is excluded'),trees,checks


class Continuation(ReviewUnit):
    def __init__(self,args):
        origin=args.continue_from.resolve()
        cases=saved_cases(args.late_reference_dir.resolve())
        eos=[]
        for name in ('config.json','generation_config.json'):
            value=json.loads((ROOT/'models/FastMTP'/name).read_text()).get('eos_token_id',[])
            for token in ([value] if isinstance(value,int) else value or []):
                if token not in eos:eos.append(token)
        self.reuse,self.trees,self.inherited_checks=validate_origin(origin,cases,eos)
        root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True)
        registration=root/'continuation_registration.json'
        attempt=json.loads((origin/'execution_attempt.json').read_text())
        deadline=utc_epoch(attempt['started_utc'])+86400
        if registration.exists():record=json.loads(registration.read_text())
        else:
            record=dict(origin=str(origin),registered_utc=stamp(),deadline_epoch=deadline,
                absolute_deadline_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(deadline)),
                runtime_max_seconds=int(deadline-time.time()),limits=CONTINUATION_LIMITS,
                extra_numeric_attempts=1,extra_warmup_starts=1,automatic_retry=False)
            write_json(registration,record)
        if record['origin']!=str(origin) or record['deadline_epoch']!=deadline or record['limits']!=CONTINUATION_LIMITS:
            raise ValueError('Continuation registration changed')
        if deadline<=time.time():raise BudgetStop('Original 24-hour deadline reached')
        self.deadline_epoch=deadline
        args.call_limits=CONTINUATION_LIMITS;args.runtime_limit=record['runtime_max_seconds']
        args.registered_policy=dict(profile_order=[],profile_http_this_unit=0,
            target_prefix_cases_this_unit=['late_4_target'],evidence_reuse=self.reuse,
            absolute_deadline_utc=record['absolute_deadline_utc'],
            combined_attempt_limits=dict(main_http=74,numeric=11,warmup=10,starts=10))
        super().__init__(args)
        write_json(self.root/'evidence_reuse_gate.json',self.reuse)
        profile=self.root/'profile'
        if profile.exists():
            if not profile.is_symlink() or profile.resolve()!=origin/'profile':raise ValueError('Profile link changed')
        else:profile.symlink_to(origin/'profile',target_is_directory=True)

    def begin_work(self,*args,**kwargs):
        if time.time()>=self.deadline_epoch:raise BudgetStop('Original 24-hour deadline reached')
        return super().begin_work(*args,**kwargs)

    def run(self):
        with (self.root/'execution_attempt.json').open('x') as f:
            json.dump(dict(started_utc=stamp(),pid=__import__('os').getpid(),retries=0,origin=self.reuse['origin']),f)
        self.heartbeat=threading.Thread(target=self.pulse,daemon=True);self.heartbeat.start()
        (self.root/'numeric').mkdir(exist_ok=True)
        self.args.target_probe=True;self.args.target_probe_prefix_manifest=str(self.root/'probe_selectors.json')
        directory=self.root/'target_only/target';self.current=self.variants['target'];self.diagnostic=False
        self.start_server('target',directory)
        case=self.cases[-1];target=self.numeric_call(case,'target',directory)
        import torch
        check=dict(case=case['label'],generated_length=case['generated_length'],
            prefix_sha256=case['prefix_sha256'],**compare_logits(self.trees[case['label']],target,self.eos_ids))
        original=torch.load(case['frame'],map_location='cpu',weights_only=False)
        difference=torch.softmax(self.trees[case['label']].float()/.6,-1)-original['target'][0]
        check['fresh_tree_vs_saved_P_tv']=float(difference.abs().sum()/2)
        check['fresh_tree_vs_saved_P_max_abs']=float(difference.abs().max())
        check['saved_P_cpu_reduction_tolerance']=1e-6
        check['passed'] &= check['fresh_tree_vs_saved_P_tv']<=1e-6
        checks=[*self.inherited_checks,check]
        write_json(self.root/'target_prefix_gate.json',dict(passed=all(c['passed'] for c in checks),
            complete=True,checks=checks,reuse_gate='evidence_reuse_gate.json',
            scope='Four original exact comparisons plus one corrected final-chunk comparison; original negative exit retained; not all KV states'))
        if not check['passed']:raise NumericRejected('Corrected final-chunk raw Target difference: late_4')
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
        self.make_analysis();self.status('COMPLETED_EVIDENCE_UNIT',completed=66,total=66)
        return 0


def main():
    # Keep the common signal handling/failure/cleanup path, with this subclass.
    import scripts.run_raven_review_unit as common
    original_arguments=common.arguments
    def parse():
        if '--continue-from' not in sys.argv:raise ValueError('--continue-from is required')
        index=sys.argv.index('--continue-from');origin=Path(sys.argv[index+1])
        del sys.argv[index:index+2]
        args=original_arguments();args.continue_from=origin;return args
    common.arguments=parse;common.ReviewUnit=Continuation
    return common.main()


if __name__=='__main__':raise SystemExit(main())
