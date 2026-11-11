import pytest
import torch

from remtp.dynamic_mtp_tree import DynamicTreeConfig, DynamicTreeTopology, adaptive_trunk_relaxation_plan, residual_hit_tree_verify
from remtp.fixed6_microtree import residual_distribution
from remtp.decision_trace import checkpoint_slot
from scripts.run_adaptive_round4 import GateRejected
from scripts.run_raven_probability_cap import ProbabilityCap, trace_semantics


@pytest.mark.parametrize('value',[-1,.5,float('nan'),float('inf')])
def test_invalid_cap_rejected(value):
    with pytest.raises(ValueError,match='relaxation_max_probability_ratio'):
        DynamicTreeConfig(relaxation_max_probability_ratio=value).validate()


@pytest.mark.parametrize('adaptive',[False,True])
@pytest.mark.parametrize('cap',[1.,2.,4.])
def test_capped_full_h_controls_acceptance_and_rejected_distribution(adaptive,cap):
    topology=DynamicTreeTopology.from_paths(((0,),))
    p=torch.tensor([.001,.199,.8]); q=torch.tensor([.9,.05,.05])
    config=DynamicTreeConfig(max_depth=1,max_nodes=1,support_mode='residual_hit_anchor',
        adaptive_relaxation=adaptive,relaxation_max_probability_ratio=cap)
    trace={}
    result=residual_hit_tree_verify(topology,torch.tensor([0]),q[None],p[None].repeat(2,1),
        torch.Generator().manual_seed(0),config=config,decision_trace=trace)
    event=trace['events'][0]
    expected=p*((1-cap*.001)/(1-.001));expected[0]=cap*.001;expected/=expected.sum()
    assert torch.allclose(event['h'],expected,atol=1e-7)
    assert not event['keep']
    assert result.node_diagnostics[0]['accept_probability']==pytest.approx(float(event['h'][0]/q[0]))
    assert torch.equal(trace['events'][1]['probs'],residual_distribution(event['h'],event['q']))
    trace_semantics(trace,cap)


def test_adaptive_redistribution_cannot_escape_cap_or_eos_protection():
    topology=DynamicTreeTopology.from_paths(((0,),(0,0),(0,0,0)))
    ids=torch.tensor([0,0,0]);q=torch.tensor([[.9,.1]]).repeat(3,1)
    p=torch.tensor([[.001,.999],[.1,.9],[.3,.7],[.5,.5]])
    config=DynamicTreeConfig(max_depth=3,max_nodes=3,support_mode='residual_hit_anchor',
        adaptive_relaxation=True,adaptive_relaxation_mix=1.,relaxation_max_probability_ratio=2.)
    plan=adaptive_trunk_relaxation_plan(topology,ids,q,p,config=config,protected_node_indices=frozenset({1}))
    assert plan[1]['baseline_relaxation_tv']==plan[1]['adaptive_relaxation_tv']==0
    for index,row in plan.items():
        assert row['adaptive_relaxation_tv']<=float(p[index,0])+1e-8
    assert plan[0]['adaptive_block_tv']<=plan[0]['adaptive_block_baseline_tv']+1e-7


@pytest.mark.parametrize('p_value',[0.,1e-10,1.])
def test_probability_endpoints_finite_and_bounded(p_value):
    p=torch.tensor([p_value,1-p_value]);q=torch.tensor([.9,.1]);trace={}
    residual_hit_tree_verify(DynamicTreeTopology.from_paths(((0,),)),torch.tensor([0]),q[None],p[None].repeat(2,1),
        torch.Generator().manual_seed(0),config=DynamicTreeConfig(max_depth=1,max_nodes=1,
        support_mode='residual_hit_anchor',adaptive_relaxation=True,relaxation_max_probability_ratio=2),decision_trace=trace)
    trace_semantics(trace,2)


@pytest.mark.parametrize('adaptive',[False,True])
def test_cap_one_preserves_strict_h_for_all_proposed_tokens(adaptive):
    p=torch.tensor([.1,.2,.7]);q=torch.tensor([.7,.2,.1]);contribution=torch.zeros_like(p)
    topology=DynamicTreeTopology.from_paths(((0,),))
    for y in range(3):
        trace={}
        residual_hit_tree_verify(topology,torch.tensor([y]),q[None],p[None].repeat(2,1),
            torch.Generator().manual_seed(0),config=DynamicTreeConfig(max_depth=1,max_nodes=1,
            support_mode='residual_hit_anchor',adaptive_relaxation=adaptive,relaxation_max_probability_ratio=1),decision_trace=trace)
        assert torch.equal(trace['events'][0]['h'],p)
        alpha=min(1,float(p[y]/q[y]));contribution[y]+=q[y]*alpha
        contribution+=q[y]*(1-alpha)*residual_distribution(p,q)
    # Enumerate first-token rejection sampling rather than matching seeded text.
    assert torch.allclose(contribution,p,atol=1e-7)


def test_late_checkpoints_skip_proposal_and_take_each_threshold_once():
    points=(128,512,1024,1792,2000)
    assert checkpoint_slot(127,points,0) is None
    assert checkpoint_slot(130,points,0)==1
    assert checkpoint_slot(511,points,1) is None
    assert checkpoint_slot(513,points,1)==2
    assert checkpoint_slot(2003,points,2)==5
    assert checkpoint_slot(2048,points,5) is None


def test_controller_negative_gate_stops_without_training(tmp_path):
    c=object.__new__(ProbabilityCap);c.root=tmp_path;c.protocol={'source_sha256':'new'}
    statuses=[];c.status=lambda phase:statuses.append(phase);c.report=lambda:None
    with pytest.raises(GateRejected):c.gate('late_B_replay',{'passed':False})
    assert statuses==['STOPPED_LATE_B_REPLAY']


@pytest.mark.parametrize('method,cap,worker',[
    ('native',0,'[ReMTP][FastMTPVerified] method=native worker=active'),
    ('dynamic_tree',0,'relax_max_probability_ratio=0 '),
    ('dynamic_tree',2,'relax_max_probability_ratio=2 ')])
def test_start_server_delegates_to_correct_base_and_confirms_effective_worker(tmp_path,monkeypatch,method,cap,worker):
    from scripts.run_adaptive_round2 import Round2
    c=object.__new__(ProbabilityCap);c.current={'method':method,'probability_ratio_cap':cap}
    called=[]
    def start(self,*args):
        called.append(args);(tmp_path/'server.log').write_text('[ReMTP][HeadBatch] input_rows=1\n'+worker+'\n[ReMTP][ProbMTP] full proposal distributions enabled')
    monkeypatch.setattr(Round2,'start_server',start)
    c.start_server('test',tmp_path)
    assert called==[('test',tmp_path,False)]


def test_missing_effective_cap_log_refuses_training(tmp_path,monkeypatch):
    from scripts.run_adaptive_round2 import Round2
    c=object.__new__(ProbabilityCap);c.current={'method':'dynamic_tree','probability_ratio_cap':2}
    monkeypatch.setattr(Round2,'start_server',lambda *args:(tmp_path/'server.log').write_text('[ReMTP][HeadBatch] input_rows=1'))
    with pytest.raises(ValueError,match='Effective worker/cap'):c.start_server('test',tmp_path)


def test_report_discloses_scope_and_budget(tmp_path):
    c=object.__new__(ProbabilityCap);c.root=tmp_path;c.report()
    text=(tmp_path/'report.md').read_text()
    assert '32次' in text and '144条' in text and '不执行留出或正式' in text
