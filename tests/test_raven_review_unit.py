"""Safety gates for finite calls, real startup options and observational hooks."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from remtp.runtime_profile import ProfileRecorder
from scripts.run_raven_review_unit import (Ledger,BudgetStop,NumericRejected,QualityRejected,
    compare_logits,failure_kind,profile_summary,component_hashes,component_hashes_for_reference)
from scripts.run_adaptive_round2 import Round2,settings


class FakeCUDA:
    def __init__(self):self.syncs=0;self.events=0
    def is_initialized(self):return True
    def Event(self,enable_timing):
        assert enable_timing
        self.events+=1
        return SimpleNamespace(record=lambda:None,elapsed_time=lambda other:1.)
    def synchronize(self):self.syncs+=1


def test_profile_skips_warmup_prefill_extra_requests_and_caps_rounds(tmp_path):
    cuda=FakeCUDA();record=ProfileRecorder(tmp_path/'profile.jsonl',max_rounds=2,per_request=1,cuda=cuda)
    assert not record.begin('warmup',0,30)
    assert not record.begin('warmup',1,4)
    assert not record.begin('first',0,150)
    assert record.begin('first',1,4)
    with record.span('proposal'):
        with record.span('mtp_logits'):pass
    record.finish(4)
    assert not record.begin('first',4,4)
    assert record.begin('second',1,4)
    record.finish(5)
    assert not record.begin('third_numeric_call',1,10)
    assert cuda.syncs==2
    rows=[json.loads(s) for s in record.path.read_text().splitlines()]
    spans=[r for r in rows if r['record_type']=='span']
    assert spans[1]['parent_id']==spans[0]['span_id']
    assert all(r['domain']=='draft' for r in spans)
    assert [r['worker_committed_tokens'] for r in rows if r['record_type']=='round']==[3,4]


def test_profile_late_numeric_requests_skipped_even_if_round_cap_not_used(tmp_path):
    record=ProfileRecorder(tmp_path/'profile.jsonl',cuda=FakeCUDA())
    for name in ('warmup','first','second'):
        if record.begin(name,1,4):record.finish(2)
    assert record.rounds==2 and not record.begin('late_numeric',1,10)


def test_profile_cpu_observer_keeps_original_rng_and_calls_once(tmp_path):
    rng=torch.Generator().manual_seed(77);expected=torch.Generator().manual_seed(77)
    reference=torch.rand((),generator=expected)
    record=ProfileRecorder(tmp_path/'profile.jsonl',cuda=FakeCUDA())
    record.begin('warmup',1,4);assert record.begin('actual',1,4)
    calls=[]
    with record.span('proposal'):
        calls.append(1);value=torch.rand((),generator=rng)
    record.finish(2)
    assert calls==[1] and torch.equal(value,reference) and torch.equal(rng.get_state(),expected.get_state())


def test_ledger_failed_attempt_consumes_budget_and_duplicate_cannot_retry(tmp_path):
    ledger=Ledger(tmp_path/'ledger.json',{'main_http':2})
    ledger.reserve('main_http','a')
    with pytest.raises(BudgetStop,match='Already attempted'):ledger.reserve('main_http','a')
    ledger.reserve('main_http','b');ledger.complete('main_http','b')
    assert ledger.counts()['main_http']==dict(attempted=2,completed=1,limit=2)
    with pytest.raises(BudgetStop,match='exhausted'):ledger.reserve('main_http','c')
    reopened=Ledger(tmp_path/'ledger.json',{'main_http':2})
    assert reopened.counts()==ledger.counts()
    with pytest.raises(ValueError,match='policy changed'):Ledger(tmp_path/'ledger.json',{'main_http':3})


@pytest.mark.parametrize('error,kind',[(NumericRejected('P'),'NUMERIC_REJECTED'),
    (QualityRejected('accuracy'),'QUALITY_REJECTED'),(BudgetStop('limit'),'BUDGET_STOPPED'),
    (RuntimeError('server exited'),'RUNTIME_ERROR'),(KeyboardInterrupt(),'INTERRUPTED')])
def test_failure_kind_preserves_cause(error,kind):assert failure_kind(error)==kind


def test_fresh_target_difference_stops_even_within_legacy_tolerance():
    raw=torch.tensor([1.,2.,-1.,-10.]);assert compare_logits(raw,raw,[3])['passed']
    changed=raw.clone();changed[0]+=.01
    check=compare_logits(changed,raw,[3])
    assert check['legacy_tolerance_passed'] and not check['passed']
    with pytest.raises(NumericRejected):compare_logits(raw,torch.full_like(raw,float('nan')),[3])


def test_profile_summary_keeps_nested_cost_out_of_top_level_sum(tmp_path):
    recorder=ProfileRecorder(tmp_path/'profile.jsonl',cuda=FakeCUDA())
    recorder.begin('warmup',1,4);recorder.begin('actual',1,4)
    for name in ('target_forward','target_logits','proposal','sampling','cache_update'):
        with recorder.span(name):
            if name=='proposal':
                with recorder.span('mtp_forward'):pass
    recorder.finish(5)
    summary=profile_summary(recorder.path)
    assert summary['rounds']==1
    assert summary['components']['draft/mtp_forward']['top_level_calls']==0
    rows=[json.loads(s) for s in recorder.path.read_text().splitlines()]
    expected=sum(r['wall_seconds'] for r in rows if r['record_type']=='span' and r['parent_id']==0)
    assert summary['top_level_wall_per_round']==expected


def test_component_hashes_separate_observer_from_math(tmp_path):
    import scripts.run_raven_review_unit as module
    for files in module.COMPONENTS.values():
        for name in files:
            p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    current=component_hashes(tmp_path);old=component_hashes_for_reference(tmp_path)
    assert all(current[k]==old[k] for k in old)
    (tmp_path/'remtp/runtime_profile.py').write_text('new observation')
    changed=component_hashes(tmp_path)
    assert changed['instrumentation']!=current['instrumentation']
    assert changed['target_math']==current['target_math']


@pytest.mark.parametrize('enabled',[False,True])
def test_real_base_startup_passes_profile_and_probe_environment(tmp_path,monkeypatch,enabled):
    import scripts.run_adaptive_round2 as module
    controller=object.__new__(Round2)
    controller.current=settings(method='native',head=1);controller.diagnostic=False
    controller.args=SimpleNamespace(port=8000,target_probe=True,target_probe_shape_rows=(),
        target_probe_prefix_manifest=str(tmp_path/'selected.json'))
    controller.eos_ids=[19];controller.base_url='http://fake';controller.stop_server=lambda:None
    captured={};warmups=[]
    class Socket:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def connect_ex(self,*a):return 1
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*a):pass
    def popen(*args,**kwargs):
        captured.update(kwargs['env'])
        return SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(module.socket,'socket',Socket)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='15W'))
    monkeypatch.setattr(module.subprocess,'Popen',popen)
    monkeypatch.setattr(module.urllib.request,'urlopen',lambda *a,**k:Response())
    monkeypatch.setattr(module,'chat_completion',lambda *args:warmups.append(args))
    controller.start_server('native',tmp_path,profile=enabled)
    assert captured['REMTP_RUNTIME_PROFILE_JSONL']==str(tmp_path/'profile.jsonl') if enabled else captured['REMTP_RUNTIME_PROFILE_JSONL']==''
    assert captured['REMTP_TARGET_PROBE_SHAPE_ROWS']==''
    assert captured['REMTP_TARGET_PROBE_PREFIX_MANIFEST']==str(tmp_path/'selected.json')
    assert captured['REMTP_CACTUS_DELTA']=='1.0' and captured['REMTP_CASCADE_ALPHA']=='0.5'
    assert len(warmups)==1 and warmups[0][-2]==8


def test_target_probe_exact_selector_avoids_extra_projections(tmp_path,monkeypatch):
    from remtp import target_diagnostics as module
    import remtp
    calls=[]
    class Model:
        def compute_logits(self,hidden):calls.append(hidden.shape[0]);return torch.ones(hidden.shape[0],4)
    class Runner:
        def _prepare_inputs(self,*a):return 'original'
    monkeypatch.setitem(sys.modules,'vllm.model_executor.models.mimo',SimpleNamespace(MiMoForCausalLM=Model))
    monkeypatch.setitem(sys.modules,'vllm.v1.worker.gpu_model_runner',SimpleNamespace(GPUModelRunner=Runner))
    runtime=SimpleNamespace(target_tree_active=True,topology=SimpleNamespace(choices=((0,),)),
                            draft_token_ids=torch.tensor([7]))
    monkeypatch.setattr(remtp,'fixed6_vllm',SimpleNamespace(_RUNTIME=runtime),raising=False)
    manifest=tmp_path/'selectors.json'
    manifest.write_text(json.dumps(dict(cases=[dict(label='late_0_tree',prompt_ids=[1,2],output_ids=[9])])))
    monkeypatch.setenv('REMTP_TARGET_DIAGNOSTIC_DIR',str(tmp_path/'probe'))
    monkeypatch.setenv('REMTP_TARGET_PROBE_PREFIX_MANIFEST',str(manifest))
    monkeypatch.setenv('REMTP_TARGET_PROBE_SHAPE_ROWS','')
    monkeypatch.setenv('REMTP_TARGET_PROBE_MAX_FRAMES','1')
    monkeypatch.delenv('REMTP_TARGET_STATE_AUDIT',raising=False)
    module.install_target_diagnostics()
    runner=Runner();runner.input_batch=SimpleNamespace(req_ids=['warmup'])
    runner.requests={'warmup':SimpleNamespace(prompt_token_ids=[4],output_token_ids=[5]),
                     'actual':SimpleNamespace(prompt_token_ids=[1,2],output_token_ids=[9])}
    runner.query_start_loc=SimpleNamespace(np=[0,2]);runner.input_ids=SimpleNamespace(gpu=torch.tensor([9,7]))
    runner.positions=SimpleNamespace(gpu=torch.tensor([2,3]));model=Model();hidden=torch.ones(2,3)
    assert runner._prepare_inputs()=='original';model.compute_logits(hidden)
    assert not list((tmp_path/'probe').glob('*/*.pt'))
    runner.input_batch.req_ids=['actual'];runner._prepare_inputs();model.compute_logits(hidden)
    frames=list((tmp_path/'probe').glob('*/*.pt'));assert len(frames)==1
    frame=torch.load(frames[0],weights_only=True)
    assert calls==[2,2] and frame['shape_probes']=={} and frame['case_label']=='late_0_tree'
    assert frame['actual_input_ids']==(9,7) and frame['actual_positions']==(2,3)


def test_sequential_probe_waits_for_final_chunk_and_keeps_original_calls(tmp_path,monkeypatch):
    from remtp import target_diagnostics as module
    import remtp
    calls=[]
    class Model:
        def compute_logits(self,hidden):
            calls.append(hidden.shape[0]);return torch.full((hidden.shape[0],4),float(len(calls)))
    class Runner:
        def _prepare_inputs(self,*a):return 'original'
    monkeypatch.setitem(sys.modules,'vllm.model_executor.models.mimo',SimpleNamespace(MiMoForCausalLM=Model))
    monkeypatch.setitem(sys.modules,'vllm.v1.worker.gpu_model_runner',SimpleNamespace(GPUModelRunner=Runner))
    monkeypatch.setattr(remtp,'fixed6_vllm',SimpleNamespace(_RUNTIME=SimpleNamespace(target_tree_active=False)),raising=False)
    prefix=list(range(2223));manifest=tmp_path/'selectors.json'
    manifest.write_text(json.dumps(dict(cases=[dict(label='late_4_target',prompt_ids=prefix,output_ids=[])])))
    monkeypatch.setenv('REMTP_TARGET_DIAGNOSTIC_DIR',str(tmp_path/'probe'))
    monkeypatch.setenv('REMTP_TARGET_PROBE_PREFIX_MANIFEST',str(manifest))
    monkeypatch.setenv('REMTP_TARGET_PROBE_SHAPE_ROWS','')
    monkeypatch.setenv('REMTP_TARGET_PROBE_MAX_FRAMES','1')
    monkeypatch.delenv('REMTP_TARGET_STATE_AUDIT',raising=False)
    module.install_target_diagnostics()
    runner=Runner();runner.input_batch=SimpleNamespace(req_ids=['actual'])
    runner.requests={'actual':SimpleNamespace(prompt_token_ids=prefix,output_token_ids=[])}
    model=Model();hidden=torch.ones(1,3)
    for lo,hi in ((0,2048),(2048,2223)):
        runner.query_start_loc=SimpleNamespace(np=[0,hi-lo])
        runner.input_ids=SimpleNamespace(gpu=torch.tensor(prefix[lo:hi]))
        runner.positions=SimpleNamespace(gpu=torch.arange(lo,hi))
        assert runner._prepare_inputs()=='original'
        result=model.compute_logits(hidden)
        assert torch.equal(result,torch.full((1,4),float(len(calls))))
        if hi==2048:assert not list((tmp_path/'probe').glob('*/*.pt'))
    frames=list((tmp_path/'probe').glob('*/*.pt'));assert len(frames)==1
    frame=torch.load(frames[0],weights_only=True)
    assert calls==[1,1] and frame['calls']==2 and frame['saved']==1 and frame['shape_probes']=={}
    assert len(frame['actual_positions'])==175 and frame['actual_positions'][-1]==2222
    assert frame['actual_input_ids'][-1]==2222 and torch.equal(frame['last_logits'],result[-1])


def test_full_target_request_uses_long_transport_timeout_without_changing_generation(tmp_path,monkeypatch):
    import scripts.run_raven_review_unit as module
    controller=object.__new__(module.ReviewUnit)
    controller.root=tmp_path;controller.base_url='http://fake';controller.request_timeout=5400
    controller.ledger=Ledger(tmp_path/'ledger.json',{'main_http':1})
    controller.begin_work=lambda *a,**k:None;controller.status=lambda *a,**k:None
    captured=[]
    def completion(*args):
        captured.append(args)
        return dict(choices=[dict(message=dict(content='The answer is 42.'),finish_reason='stop')],
            usage=dict(completion_tokens=8),id='fake')
    monkeypatch.setattr(module,'chat_completion',completion)
    monkeypatch.setattr(module,'fetch_metrics',lambda *a:{})
    row=dict(question_id=1532,question='test',gold_answer='42')
    controller.generate('target_only','target',row,42,2048)
    assert len(captured)==1 and captured[0][-4:]==(.6,15362,2048,5400)
    assert controller.ledger.counts()['main_http']==dict(attempted=1,completed=1,limit=1)


@pytest.fixture
def continuation_origin(tmp_path,monkeypatch):
    import scripts.run_raven_review_unit as base
    from scripts import run_raven_review_continuation as module
    origin=tmp_path/'old';code=tmp_path/'new_code';origin.mkdir()
    for files in base.COMPONENTS.values():
        for name in files:
            for root in (code,origin/'frozen_code'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    def save(name,value):
        p=origin/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    save('failure.json',dict(kind='NUMERIC_REJECTED',error='Raw Target frame/position/probe scope mismatch: late_4_target'))
    save('exit.json',dict(exit_code=3,server_cleaned=True))
    save('ledger.json',dict(counts={k:dict(attempted=a,completed=c) for k,(a,c) in
        dict(main_http=(8,8),numeric=(10,9),warmup=(5,5),starts=(5,5)).items()}))
    save('component_hashes.json',dict(current=base.component_hashes(code)))
    save('execution_attempt.json',dict(started_utc='2026-10-07 13:08:49 UTC'))
    summary=dict(requests=2,components={'draft/tree_proposal':{}})
    monkeypatch.setattr(module,'profile_summary',lambda path:summary)
    for name in module.PROFILE_ORDER:
        save('profile/'+name+'/components.json',summary)
        (origin/'profile'/name/'profile.jsonl').write_text('original profile\n')
        (origin/'profile'/name/'requests.jsonl').write_text('{}\n{}\n')
    raw=torch.tensor([1.,2.,3.]);cases=[];checks=[]
    for i in range(5):
        prefix=[0]*(2222 if i==4 else i+1)+[2];path=origin/f'history_{i}.pt'
        torch.save(dict(target=torch.softmax(raw/.6,-1).unsqueeze(0)),path)
        case=dict(label=f'late_{i}',prefix_ids=prefix,generated_length=i,prefix_sha256=str(i),frame=str(path))
        cases.append(case)
        for role in ('tree','target'):
            frame=dict(case_label=case['label']+'_'+role,shape_probes={},rows=1,
                prompt_ids=tuple(prefix[:-1] if role=='tree' else prefix),
                output_ids=tuple(prefix[-1:] if role=='tree' else []),
                actual_input_ids=tuple(prefix[-1:] if role=='tree' else prefix),
                actual_positions=(len(prefix)-1,),first_logits=raw,last_logits=raw,logits_rows=raw.unsqueeze(0))
            if role=='target' and i==4:
                frame.update(actual_positions=tuple(range(2048)),actual_input_ids=(0,)*2048)
            directory=origin/('profile/B' if role=='tree' else 'target_only/target')/'target_probe'/f'case{i}'
            directory.mkdir(parents=True,exist_ok=True);torch.save(frame,directory/'logits_00.pt')
        if i<4:checks.append(dict(case=case['label'],raw_equal=True,max_abs_logit=0.,fresh_tree_vs_saved_P_tv=0.))
    save('numeric_cases.json',dict(cases=cases))
    save('target_prefix_gate.json',dict(passed=True,complete=False,checks=checks))
    return module,origin,code,cases


def test_continuation_reuses_only_completed_profiles_and_four_exact_cases(continuation_origin):
    module,origin,code,cases=continuation_origin
    reuse,trees,checks=module.validate_origin(origin,cases,[0],code)
    assert reuse['passed'] and len(trees)==5 and len(checks)==4
    assert module.CONTINUATION_LIMITS==dict(main_http=66,numeric=1,warmup=5,starts=5)
    assert all(c['raw_equal'] for c in checks) and reuse['origin_numeric_attempted']==10


def test_continuation_rejects_spent_work_or_math_change(continuation_origin):
    module,origin,code,cases=continuation_origin
    p=origin/'ledger.json';original=p.read_text();data=json.loads(original)
    data['counts']['main_http']['attempted']=9;p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='duplicate work'):module.validate_origin(origin,cases,[0],code)
    p.write_text(original);(code/'remtp/target_head.py').write_text('changed math')
    with pytest.raises(ValueError,match='math/state changed'):module.validate_origin(origin,cases,[0],code)


def test_continuation_keeps_original_absolute_deadline(monkeypatch):
    from scripts import run_raven_review_continuation as module
    controller=object.__new__(module.Continuation);controller.deadline_epoch=1.
    monkeypatch.setattr(module.time,'time',lambda:2.)
    with pytest.raises(BudgetStop,match='Original 24-hour'):controller.begin_work('test','target')
