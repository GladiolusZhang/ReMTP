"""User objective, finite execution and effective startup regression coverage."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from remtp.optimization_metrics import dominance_gate, paired_gate
from scripts.run_raven_dominance_search import (DominanceSearch, variants, registered_changes,
    LIMITS, ORDER, CANDIDATES, BASELINES)
from scripts.run_raven_review_unit import Ledger, BudgetStop, NumericRejected, COMPONENTS


def record(qid=1,base=42,correct=True,seconds=10.,tokens=20,finish='stop'):
    if finish == 'length' and tokens == 20:
        tokens = 2048
    return dict(question_id=qid,seed=qid*10+base,correct=correct,client_seconds=seconds,
        output_tokens=tokens,finish_reason=finish,output='#### 1',metrics={},
        gold_answer='1',max_tokens=2048,prompt_token_ids=[1]*65,output_token_ids=[1]*tokens,
        performance_sample=True,observation_enabled=False)


def test_truncation_and_stopped_correct_do_not_veto_faster_correct_output():
    before=[record()];after=[record(seconds=5.,finish='length')]
    assert dominance_gate(before,after)['passed']
    assert not paired_gate(before,after)['passed']


def test_aggregate_improvement_can_include_pairwise_and_seed_losses():
    before=[record(qid=1),record(qid=2,base=43,correct=False),record(qid=3,base=43,correct=False)]
    after=[record(qid=1,correct=False,seconds=5.),record(qid=2,base=43,seconds=5.),record(qid=3,base=43,seconds=5.)]
    gate=dominance_gate(before,after)
    assert gate['passed'] and len(gate['accuracy_losses'])==1 and len(gate['accuracy_gains'])==2
    assert gate['per_seed']['42']['candidate']['correct']==0


@pytest.mark.parametrize('after,reason',[(record(correct=False,seconds=5.),'accuracy'),
    (record(seconds=15.,tokens=60),'latency'),(record(seconds=5.,tokens=5),'throughput')])
def test_accuracy_or_each_speed_metric_can_reject(after,reason):
    assert 'all:'+reason in dominance_gate([record()],[after])['failures']


@pytest.mark.parametrize('fault',['empty','duplicate','missing','prompt','budget','time','tokens','probe'])
def test_incomplete_unfair_or_observed_comparisons_rejected(fault):
    a=[record()];b=[record(seconds=5.)]
    if fault=='empty':a=[];b=[]
    if fault=='duplicate':b*=2
    if fault=='missing':b[0]['seed']+=1
    if fault=='prompt':b[0]['prompt_token_ids']=[9]*65
    if fault=='budget':b[0]['max_tokens']=64
    if fault=='time':b[0]['client_seconds']=float('nan')
    if fault=='tokens':b[0]['output_tokens']=0
    if fault=='probe':b[0]['observation_enabled']=True
    with pytest.raises(ValueError):dominance_gate(a,b)


def bare(tmp_path):
    c=object.__new__(DominanceSearch);c.root=tmp_path;c.state={};c.work={};c.analysis=None;c.datasets={}
    c.variants=variants();c.current=c.variants['B'];c.server=None;c.started=__import__('time').monotonic()
    c.runtime_limit=86400;c.ledger=Ledger(tmp_path/'ledger.json',LIMITS);c.request_timeout=10
    c.args=SimpleNamespace(port=8000,prefix_cache=0,uniform_gemm=True,full_bf16_accum=True,bnb_row_block=64,
        rmsnorm_row_limit=255,lm_head_row_block=16,target_backend='TRITON_ATTN',triton_2d=True,
        attention_fp32=True,canonical_tree=True,decision_trace=False,decision_trace_only=False,
        target_probe=False,target_probe_shape_rows=(),decision_trace_checkpoints=())
    c.eos_ids=[151645,151643];c.diagnostic=False;c.base_url='http://fake';c.heartbeat=None
    return c


def test_only_additive_metrics_function_is_permitted(tmp_path):
    current=tmp_path/'current';reference=tmp_path/'reference'
    for group,files in COMPONENTS.items():
        for file in files:
            for root in (current,reference):
                p=root/file;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('def summarize(): pass\ndef paired_gate(): pass\ndef clustered_intervals(): pass\ndef summarize_fast(): pass\n')
    p=current/'remtp/optimization_metrics.py';p.write_text(p.read_text()+'def dominance_gate(): pass\n')
    assert set(registered_changes(current,reference)[2])=={'remtp/optimization_metrics.py'}
    p.write_text(p.read_text().replace('def summarize(): pass','def summarize(): return 1'))
    with pytest.raises(ValueError,match='Historical'):registered_changes(current,reference)


def test_unregistered_target_change_rejected(tmp_path):
    current=tmp_path/'current';reference=tmp_path/'reference'
    for group,files in COMPONENTS.items():
        for file in files:
            for root in (current,reference):
                p=root/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('')
    (current/'remtp/target_norm.py').write_text('# changed')
    with pytest.raises(ValueError,match='Unregistered'):registered_changes(current,reference)


@pytest.mark.parametrize('name',CANDIDATES)
def test_actual_startup_mro_applies_candidate_and_disables_probes(tmp_path,monkeypatch,name):
    import scripts.run_adaptive_round2 as base
    c=bare(tmp_path);c.current=c.variants[name];c.stop_server=lambda:None;captured={}
    class Socket:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def connect_ex(self,*args):return 1
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
    cfg=c.current
    log=f"[ReMTP][FastMTPVerified] method=dynamic_tree worker=active [ReMTP][HeadBatch] input_rows=1 D={cfg['depth']} N_max={cfg['nodes']} adaptive_mix={cfg['mix']:g} cactus_delta={cfg['delta']:g} risk_policy=none relax_max_probability_ratio=0 lazy_residual=0"
    def popen(*args,**kwargs):
        captured.update(kwargs['env']);kwargs['stdout'].write(log);kwargs['stdout'].flush()
        return SimpleNamespace(pid=9,poll=lambda:None)
    monkeypatch.setattr(base.socket,'socket',Socket)
    monkeypatch.setattr(base.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='15W'))
    monkeypatch.setattr(base.subprocess,'Popen',popen)
    monkeypatch.setattr(base.urllib.request,'urlopen',lambda *a,**k:Response())
    c.warmup=lambda d:None
    c.start_server(name,tmp_path/'calibration'/name)
    assert captured['REMTP_DYNAMIC_TREE_MAX_DEPTH']==str(cfg['depth'])
    assert captured['REMTP_DYNAMIC_TREE_MAX_NODES']==str(cfg['nodes'])
    assert captured['REMTP_DYNAMIC_TREE_RELAX_RISK_POLICY']=='none'
    assert captured['REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MIX']=='.25' or float(captured['REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MIX'])==.25
    for key in ('REMTP_DECISION_TRACE_DIR','REMTP_TARGET_DIAGNOSTIC_DIR','REMTP_RUNTIME_PROFILE_JSONL','REMTP_PREFIX_DIAGNOSTIC_JSONL'):
        assert not captured[key]


def test_complete_unit_runs_all_pairs_despite_individual_losses(tmp_path,monkeypatch):
    c=bare(tmp_path);c.by_id={q:{'question_id':q} for q in range(8000)}
    c.halt=__import__('threading').Event();c.pulse=lambda:None;calls=[];starts=[]
    def start(name,directory):
        starts.append(name);directory.mkdir(parents=True,exist_ok=True)
    c.start_server=start;c.stop_server=lambda:None
    c.check_reference=lambda name,data:None;c.make_analysis=lambda:None
    def generate(phase,name,row,seed,limit):
        calls.append((name,row['question_id'],seed,limit))
        return record(qid=row['question_id'],base=seed,correct=False,finish='length')
    c.generate=generate
    assert c.run()==0
    assert len(calls)==112 and starts==list(ORDER)
    assert all(len(data)==16 for data in c.datasets.values())


def test_analysis_uses_three_baseline_objective_not_truncation_or_pairwise_gate(tmp_path,monkeypatch):
    import scripts.run_raven_dominance_search as module
    c=bare(tmp_path);c.policy={'objective':'accuracy_speed','scope':'test'}
    c.datasets={name:[record(qid=1,seconds=10.)] for name in ('B',*BASELINES)}
    c.datasets.update({name:[record(qid=1,seconds=5.,finish='length')] for name in CANDIDATES})
    monkeypatch.setattr(module,'clustered_intervals',lambda *a:dict(throughput_ratio_95ci=[2.,2.],accuracy_difference_95ci=[0.,0.]))
    c.make_analysis()
    assert c.analysis['development_point_winner'] in CANDIDATES
    assert all(row['point_passed'] for row in c.analysis['candidates'].values())


def test_ledger_does_not_allow_exceeding_new_unit_budget(tmp_path):
    ledger=Ledger(tmp_path/'ledger.json',{'main_http':1})
    ledger.reserve('main_http','first')
    with pytest.raises(BudgetStop):ledger.reserve('main_http','second')


def test_reference_reproduction_requires_actual_ids_not_just_matching_text(tmp_path):
    c=bare(tmp_path);c.reference=tmp_path/'previous'
    directory=c.reference/'calibration/B';directory.mkdir(parents=True)
    data=[record(qid=q) for q in range(1,17)]
    (directory/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
    assert c.check_reference('B',data) is None
    altered=deepcopy(data);altered[0]['output_token_ids'][0]=9
    with pytest.raises(NumericRejected,match='reproduction'):c.check_reference('B',altered)
