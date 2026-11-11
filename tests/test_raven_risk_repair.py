import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch

from scripts.run_raven_risk_repair import (RiskRepairUnit, LIMITS, CALIBRATION_ORDER,
    CALIBRATION_IDS, registered_source_changes, completed_root_evidence, select_cuda_frames,
    check_risk_semantics)
from scripts.run_raven_review_unit import COMPONENTS, Ledger, NumericRejected, QualityRejected
from scripts.run_raven_root_cause_unit import ReplayRejected, EXECUTION_PAIRS
from scripts.run_adaptive_round6 import configs
from scripts.analyze_raven_root_causes import PAIRS
from scripts.wait_raven_risk_repair import dependency_state
from remtp.dynamic_mtp_tree import DynamicTreeConfig, DynamicTreeTopology, residual_hit_tree_verify


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def record(qid, seed):
    return dict(question_id=qid, seed=seed+qid*10, gold_answer=str(qid),
        predicted_answer=str(qid), answer_source='hash', correct=True, finish_reason='stop',
        output='#### '+str(qid), output_tokens=2, client_seconds=1., metrics={},
        prompt_token_ids=[1]*65, output_token_ids=[qid, seed])


def test_only_registered_runtime_files_can_change(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b'
    for group,names in COMPONENTS.items():
        for name in names:
            for root in (a,b):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    (a/'remtp/dynamic_mtp_tree.py').write_text('repair')
    assert set(registered_source_changes(a,b))=={'remtp/dynamic_mtp_tree.py'}
    (a/'remtp/target_norm.py').write_text('unregistered math')
    with pytest.raises(ValueError,match='target_norm'):
        registered_source_changes(a,b)


def complete_dependency(tmp_path):
    dump(tmp_path/'exit.json',dict(exit_code=0))
    dump(tmp_path/'status.json',dict(phase='COMPLETED_ROOT_CAUSE_UNIT'))
    dump(tmp_path/'B_reproduction_gate.json',dict(passed=True,requests=6))
    dump(tmp_path/'target_natural_prefix_gate.json',dict(passed=True,complete=True,checks=[{}]*12))
    dump(tmp_path/'natural_forensics.json',dict(passed=True,complete=True,
        requests=[dict(signature_equal=True) for _ in PAIRS]))
    p=tmp_path/'natural/B/requests.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(record(q,s))+'\n' for q,s in PAIRS))


@pytest.mark.parametrize('fault',['exit','failure','partial','phase','replay','coverage'])
def test_dependency_requires_complete_success_and_real_pair_coverage(tmp_path,fault):
    complete_dependency(tmp_path)
    assert len(completed_root_evidence(tmp_path)[0])==6
    if fault=='exit':dump(tmp_path/'exit.json',dict(exit_code=3))
    if fault=='failure':dump(tmp_path/'failure.json',dict(error='preserve failure'))
    if fault=='partial':dump(tmp_path/'target_natural_prefix_gate.json',
        dict(passed=True,complete=False,checks=[{}]*11))
    if fault=='phase':dump(tmp_path/'status.json',dict(phase='natural'))
    if fault=='replay':dump(tmp_path/'natural_forensics.json',dict(passed=True,complete=True,
        requests=[dict(signature_equal=False)]*6))
    if fault=='coverage':
        (tmp_path/'natural/B/requests.jsonl').write_text(
            ''.join(json.dumps(record(6009,42))+'\n' for _ in PAIRS))
    with pytest.raises(NumericRejected):
        completed_root_evidence(tmp_path)


@pytest.mark.parametrize('state,ended,expected',[
    ('active',False,False),('active',True,False),('deactivating',True,False),
    ('inactive',True,True),('inactive',False,'error'),('failed',False,'error')])
def test_waiter_waits_for_exit_and_cleanup_not_elapsed_time(tmp_path,monkeypatch,state,ended,expected):
    import scripts.wait_raven_risk_repair as wait
    monkeypatch.setattr(wait.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=state))
    if ended:
        dump(tmp_path/'exit.json',dict(exit_code=0))
        dump(tmp_path/'status.json',dict(phase='COMPLETED_ROOT_CAUSE_UNIT'))
    if expected=='error':
        with pytest.raises(RuntimeError):dependency_state(tmp_path,'unit')
    else:
        assert dependency_state(tmp_path,'unit') is expected


def test_failed_dependency_never_starts_candidate(tmp_path,monkeypatch):
    import scripts.wait_raven_risk_repair as wait
    monkeypatch.setattr(wait.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='inactive'))
    dump(tmp_path/'exit.json',dict(exit_code=3))
    with pytest.raises(RuntimeError,match='禁止启动'):
        dependency_state(tmp_path,'unit')


def test_cuda_selection_is_first_last_plus_earliest_remaining_and_checks_hashes(tmp_path):
    from scripts.run_raven_risk_repair import digest
    rows=[];cases=[]
    for i in range(6):
        coupled=[]
        for j in range(4):
            p=tmp_path/f'{i}_{j}.pt';p.write_bytes(f'frame{i}/{j}'.encode())
            raw=tmp_path/f'raw{i}_{j}.pt';raw.write_bytes(f'raw{i}/{j}'.encode())
            coupled.append(dict(frame=str(p),frame_sha256=digest(p),raw_frame=str(raw),
                                raw_frame_sha256=digest(raw),prefix_sha256=f'{i}/{j}'))
        rows.append(dict(coupled_frames=coupled))
        cases.extend((coupled[0],coupled[-1]))
    dump(tmp_path/'numeric_cases.json',dict(cases=cases))
    result=select_cuda_frames(tmp_path,dict(requests=rows))
    assert [r['path'] for r in result[:12]]==[r['frame'] for r in cases]
    assert [Path(r['path']).name for r in result[12:]]==['0_1.pt','0_2.pt','1_1.pt','1_2.pt']
    Path(result[0]['path']).write_bytes(b'changed')
    with pytest.raises(NumericRejected,match='changed'):
        select_cuda_frames(tmp_path,dict(requests=rows))


def bare(tmp_path):
    c=object.__new__(RiskRepairUnit)
    c.root=tmp_path;c.state={};c.work={};c.forensics=[];c.natural=[];c.cases=[]
    c.ledger=Ledger(tmp_path/'ledger.json',LIMITS)
    c.started=time.monotonic();c.runtime_limit=86400;c.request_timeout=5400
    c.server=None;c.diagnostic=False;c.base_url='http://fake';c.eos_ids=[2]
    c.args=SimpleNamespace(port=8000,decision_trace=False,decision_trace_only=False,
        target_probe=False,target_probe_shape_rows=(),decision_trace_checkpoints=(),
        target_probe_min_generated=0,decision_trace_early_proposals=0)
    c.current={**configs()['old_eos01'],'probability_ratio_cap':0.,'risk_policy':'target_odds'}
    return c


def test_actual_startup_mro_installs_risk_and_omits_all_calibration_probes(tmp_path,monkeypatch):
    import scripts.run_adaptive_round2 as base
    c=bare(tmp_path);c.stop_server=lambda:None;captured={}
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
        kwargs['stdout'].write('[ReMTP][FastMTPVerified] method=dynamic_tree worker=active\n'
            '[ReMTP][HeadBatch] input_rows=1\n'
            'relax_max_probability_ratio=0 lazy_residual=0 risk_policy=target_odds \n')
        kwargs['stdout'].flush()
        return SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(base.socket,'socket',Socket)
    monkeypatch.setattr(base.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='15W'))
    monkeypatch.setattr(base.subprocess,'Popen',popen)
    monkeypatch.setattr(base.urllib.request,'urlopen',lambda *a,**k:Response())
    monkeypatch.setattr(base,'chat_completion',lambda *a:dict(usage={'completion_tokens':8}))
    c.start_server('target_odds',tmp_path/'calibration/target_odds')
    assert captured['REMTP_DYNAMIC_TREE_RELAX_RISK_POLICY']=='target_odds'
    assert all(captured[k]=='' for k in ('REMTP_PREFIX_DIAGNOSTIC_JSONL','REMTP_DECISION_TRACE_DIR',
        'REMTP_DECISION_TRACE_SUMMARY_JSONL','REMTP_TARGET_DIAGNOSTIC_DIR','REMTP_RUNTIME_PROFILE_JSONL'))
    assert c.ledger.counts()['warmup']['completed']==c.ledger.counts()['starts']['completed']==1


def test_full_pipeline_exercises_actual_http_ids_budget_replays_and_reporting(tmp_path,monkeypatch):
    """Transport/model are stubbed; actual controller ordering/ledger/HTTP/report run."""
    import scripts.run_raven_risk_repair as repair
    import scripts.run_raven_root_cause_unit as root
    c=bare(tmp_path);c.reference=tmp_path/'reference';c.review_reference=tmp_path/'review'
    c.by_id={q:dict(question_id=q,question=f'Question {q}',gold_answer=str(q)) for q in CALIBRATION_IDS}
    c.historical={(q,s+q*10):record(q,s) for q in CALIBRATION_IDS for s in (42,43)}
    b={**configs()['old_eos01'],'probability_ratio_cap':0.,'risk_policy':'none'}
    c.variants={n:dict(b,risk_policy='target_odds' if n=='target_odds' else 'none')
                for n in (*CALIBRATION_ORDER,'target')}
    references=[record(q,s) for q,s in PAIRS]
    monkeypatch.setattr(repair,'completed_root_evidence',lambda *a:(references,{}))
    monkeypatch.setattr(repair,'select_cuda_frames',lambda *a:[])
    # Dependency file hashes require existing immutable inputs in the actual run.
    monkeypatch.setattr(repair,'digest',lambda *a:'mock-file-hash')
    def child(*a,**kw):
        ledger=Ledger(tmp_path/'ledger.json',LIMITS)
        for i in range(48):
            ledger.reserve('cuda_verify',str(i));ledger.complete('cuda_verify',str(i))
        dump(tmp_path/'cuda_gate.json',dict(passed=True,complete=True))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(repair.subprocess,'run',child)
    c.pulse=lambda:None
    starts=[]
    def start(name,directory):
        starts.append((name,c.args.decision_trace_only,c.args.target_probe))
        for kind in ('starts','warmup'):
            c.ledger.reserve(kind,str(directory));c.ledger.complete(kind,str(directory))
    c.start_server=start;c.stop_server=lambda:None
    c.inspect_natural=lambda r,d,require_historical_replay:None
    def target():
        starts.append(('target',False,True))
        for kind in ('starts','warmup'):
            c.ledger.reserve(kind,'target');c.ledger.complete(kind,'target')
        for i in range(12):
            c.ledger.reserve('numeric',str(i));c.ledger.complete('numeric',str(i))
        dump(tmp_path/'target_natural_prefix_gate.json',dict(passed=True,complete=True))
        return [],[]
    c.check_target_prefixes=target
    payloads=[]
    class Response:
        def __init__(self,value):self.value=value
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self):return json.dumps(self.value).encode()
    def http(req,timeout):
        payload=json.loads(req.data);payloads.append(payload)
        seed=42 if payload['seed']%10==2 else 43
        qid=(payload['seed']-seed)//10
        return Response(dict(id=f'chatcmpl-{qid}-{seed}',prompt_token_ids=[1]*65,
            usage=dict(completion_tokens=2),choices=[dict(finish_reason='stop',
            token_ids=[qid,seed],message=dict(content='#### '+str(qid)))]))
    monkeypatch.setattr(root.urllib.request,'urlopen',http)
    monkeypatch.setattr(root,'fetch_metrics',lambda *a:{})
    assert c.run()==0
    assert c.state['phase']=='COMPLETED_REPAIR_UNIT'
    assert len(payloads)==86 and all(p['return_token_ids'] and p['max_tokens']==2048
        and p['temperature']==.6 for p in payloads)
    assert starts[:2]==[('target_odds',True,True),('target',False,True)]
    assert starts[2:]==[(n,False,False) for n in CALIBRATION_ORDER]
    assert all(v['completed']==v['limit']==v['attempted'] for v in c.ledger.counts().values())
    assert all(r['observation_enabled'] and not r['performance_sample'] for r in c.natural)
    unobserved=repair.requests(tmp_path/'calibration/target_odds/requests.jsonl')
    assert all(not r['observation_enabled'] and r['performance_sample'] for r in unobserved)
    assert json.loads((tmp_path/'candidate6_observer_actual_ids_gate.json').read_text())['passed']
    assert '结果分析' in (tmp_path/'report.md').read_text()


def test_smoke_refuses_losing_a_benefit_control_before_calibration(tmp_path):
    c=bare(tmp_path);c.natural=[record(q,s) for q,s in PAIRS]
    reference=[record(q,s) for q,s in PAIRS]
    control=next(r for r in c.natural if r['question_id']==5180 and r['seed']==51843)
    control['correct']=False
    with pytest.raises(QualityRejected,match='smoke_quality'):
        c.check_smoke(reference)
    assert c.ledger.counts()['main_http']['attempted']==0


def test_cuda_child_timeout_keeps_its_attempted_budget_in_parent_failure_report(tmp_path,monkeypatch):
    import scripts.run_raven_risk_repair as repair
    c=bare(tmp_path);c.reference=tmp_path/'reference';c.pulse=lambda:None
    monkeypatch.setattr(repair,'completed_root_evidence',lambda *a:([],{}))
    monkeypatch.setattr(repair,'select_cuda_frames',lambda *a:[])
    monkeypatch.setattr(repair,'digest',lambda *a:'mock-hash')
    def child(*args,**kwargs):
        ledger=Ledger(tmp_path/'ledger.json',LIMITS)
        ledger.reserve('cuda_verify','attempted_before_timeout')
        raise repair.subprocess.TimeoutExpired('child',1800)
    monkeypatch.setattr(repair.subprocess,'run',child)
    with pytest.raises(repair.subprocess.TimeoutExpired):
        c.run()
    assert c.ledger.counts()['cuda_verify']['attempted']==1
    assert c.ledger.counts()['cuda_verify']['completed']==0
    c.status('RUNTIME_ERROR')
    assert 'cuda_verify | 1 | 0 | 48' in (tmp_path/'report.md').read_text()


def test_actual_id_replay_detects_a_serialization_or_observer_difference(tmp_path):
    c=bare(tmp_path);a=record(5180,43);b=record(5180,43);b['output_token_ids']=[99,43]
    c.replay_gate('signature',[a],[b],require_ids=False)
    with pytest.raises(ReplayRejected):
        c.replay_gate('ids',[a],[b],require_ids=True)


def test_full_H_semantic_checker_rejects_an_unrecorded_rng_or_different_residual():
    c=DynamicTreeConfig(max_nodes=1,max_depth=1,support_mode='residual_hit_anchor',
        adaptive_relaxation=True,relaxation_risk_policy='target_odds')
    p=torch.tensor([[.0001,.6,.3999],[.2,.3,.5]])
    q=torch.tensor([[.9,.04,.06]]);top=DynamicTreeTopology.from_paths(((0,),))
    rng=torch.Generator().manual_seed(2);before=rng.get_state();trace={}
    result=residual_hit_tree_verify(top,torch.tensor([0]),q,p,rng,config=c,decision_trace=trace)
    assert check_risk_semantics(result,trace,before,rng.get_state())['strict']==1
    wrong=before.clone();wrong[0]^=1
    with pytest.raises(NumericRejected,match='consumption'):
        check_risk_semantics(result,trace,wrong,rng.get_state())
    trace['events'][1]['probs']=torch.tensor([1.,0.,0.])
    with pytest.raises(NumericRejected,match='same full H'):
        check_risk_semantics(result,trace,before,rng.get_state())
