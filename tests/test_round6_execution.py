"""Semantic equivalence and real controller/hook paths before a GPU run."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from remtp.dynamic_mtp_tree import DynamicTreeConfig, DynamicTreeTopology, residual_hit_tree_verify
from remtp.optimization_metrics import summarize
from scripts.run_adaptive_round2 import Round2
from scripts.run_adaptive_round4 import GateRejected, NUMERICAL_KEYS
from scripts.run_adaptive_round6 import Round6, configs, equal_trace, proposal_boundary_matches, equal_saved_tensor


@pytest.mark.parametrize('mode',['sampled_primary_shadow','residual_hit_anchor','residual_hit_strict','residual_hit_cactus'])
@pytest.mark.parametrize('adaptive',[False,True])
@pytest.mark.parametrize('protected',[False,True])
def test_lazy_residual_preserves_decisions_full_events_rng_and_inputs(mode,adaptive,protected):
    topology=DynamicTreeTopology.from_paths([(0,),(1,),(2,),(0,0),(0,1),(1,0),(0,0,0),(1,0,0)])
    ids=torch.tensor([node.path[-1] for node in topology.nodes])
    config=DynamicTreeConfig(max_nodes=8,support_mode=mode,adaptive_relaxation=adaptive,
        protect_eos_relaxation=protected,eos_token_ids=(3,),eos_protection_threshold=.01)
    terminals=set()
    for seed in range(30):
        rng=torch.Generator().manual_seed(seed)
        q=torch.rand(8,4,generator=rng).pow(4);p=torch.rand(9,4,generator=rng).pow(4)
        if seed==0:q.zero_();p.zero_()  # real fallback normalization branch
        if seed==1:q[1:].fill_(float('nan'))  # observed legacy raw-Q fallback
        before=[x.clone() for x in (ids,q,p)]
        a=torch.Generator().manual_seed(seed+200);b=torch.Generator().manual_seed(seed+200)
        left={};right={}
        old=residual_hit_tree_verify(topology,ids,q,p,a,config=config,decision_trace=left)
        new=residual_hit_tree_verify(topology,ids,q,p,b,config=config,lazy_residual=True,decision_trace=right)
        assert asdict(old)==asdict(new)
        assert torch.equal(a.get_state(),b.get_state())
        assert all(equal_saved_tensor(x,y) for x,y in zip(before,(ids,q,p)))
        assert len(left['events'])==len(right['events'])
        assert all(equal_trace(x,y) for x,y in zip(left['events'],right['events']))
        terminals.add(old.terminal)
    assert len(terminals)>=2


def test_accepted_nodes_skip_residual_and_trace_does_not_consume_rng(monkeypatch):
    import remtp.dynamic_mtp_tree as module
    topology=DynamicTreeTopology.from_paths([(0,),(0,0),(0,0,0)])
    ids=torch.zeros(3,dtype=torch.long);q=torch.tensor([[.01,.99]]*3);p=torch.tensor([[1.,0.]]*4)
    config=DynamicTreeConfig(max_nodes=3,support_mode='residual_hit_anchor',adaptive_relaxation=True)
    calls=[];original=module.residual_distribution
    def residual(a,b):calls.append(True);return original(a,b)
    monkeypatch.setattr(module,'residual_distribution',residual)
    a=torch.Generator().manual_seed(99);b=torch.Generator().manual_seed(99)
    plain=residual_hit_tree_verify(topology,ids,q,p,a,config=config)
    assert len(calls)==3
    calls.clear();trace={}
    lazy=residual_hit_tree_verify(topology,ids,q,p,b,config=config,lazy_residual=True,decision_trace=trace)
    assert not calls and asdict(plain)==asdict(lazy)
    assert torch.equal(a.get_state(),b.get_state())
    assert torch.equal(trace['events'][0]['p'],trace['events'][0]['h'])


def test_real_start_super_chain_records_new_effective_options(tmp_path,monkeypatch):
    controller=object.__new__(Round6);controller.current=configs()['anchor4_eos01']
    controller.protocol={k:None for k in NUMERICAL_KEYS}
    controller.args=SimpleNamespace(target_probe=False,decision_trace=True)
    controller.diagnostic=True;calls=[]
    def base(self,*args):
        calls.append((self,args))
        (tmp_path/'server.log').write_text('relax_depth_weights=[1.0, 1.0, 1.0, 1.0] lazy_residual=0 [ReMTP][DecisionTrace] installed')
    monkeypatch.setattr(Round2,'start_server',base)
    controller.start_server('anchor4_eos01',tmp_path)
    assert calls==[(controller,('anchor4_eos01',tmp_path,False))]
    options=json.loads((tmp_path/'round6_effective_options.json').read_text())
    assert options['decision_trace'] and not options['lazy_residual']
    assert len(options['explicit_relax_weights'])==4


def test_gate_rejection_saves_new_source_and_stops(tmp_path):
    controller=object.__new__(Round6);controller.root=tmp_path
    controller.protocol={**{k:None for k in NUMERICAL_KEYS},'source_sha256':'NEW'}
    statuses=[];controller.status=lambda name:statuses.append(name)
    with pytest.raises(GateRejected):controller.gate('replay',dict(passed=False))
    assert statuses==['STOPPED_REPLAY_GATE']
    assert json.loads((tmp_path/'replay_gate.json').read_text())['source_sha256']=='NEW'


def test_report_uses_real_evaluation_schema(tmp_path):
    controller=object.__new__(Round6);controller.root=tmp_path
    folder=tmp_path/'diagnostics'/'B';folder.mkdir(parents=True)
    record=dict(question_id=1,seed=52,correct=True,output='#### 2',gold_answer='2',
        finish_reason='stop',output_tokens=3,client_seconds=1)
    (folder/'summary.json').write_text(json.dumps(summarize([record])))
    controller.report()
    assert '| B | 1 | 1 | 1 | 0 | 3.000 | 1.0 |' in (tmp_path/'report.md').read_text()


def test_bounded_hook_skips_warmup_and_keeps_real_proposal_rng_boundary(tmp_path,monkeypatch):
    from remtp import decision_trace as module
    runtime=SimpleNamespace(diagnostic_request='warmup',diagnostic_prompt_ids=(1,)*3,generated_history=())
    topology=DynamicTreeTopology.from_paths([(0,),(0,0),(0,0,0)])
    ids=torch.zeros(3,dtype=torch.long);q=torch.tensor([[.1,.9]]*3);p=torch.tensor([[.5,.5]]*4)
    config=DynamicTreeConfig(max_nodes=3,support_mode='residual_hit_anchor')
    runtime.topology=topology;runtime.draft_token_ids=ids;runtime.draft_probs=q
    def verify(t,ids,q,p,rng):return residual_hit_tree_verify(t,ids,q,p,rng,config=config,decision_trace=module.active_trace())
    fixed=SimpleNamespace(_RUNTIME=runtime,strict_tree_verify=verify)
    dynamic=SimpleNamespace(_BRANCH_PROPOSAL_ROUND=0,_run_dynamic_tree_nodes=lambda *args,**kwargs:None)
    class Proposer:
        def propose_tree(self,**kwargs):
            torch.rand((),generator=self._fixed6_sampling_metadata.generators[0]);return 'unchanged'
    import remtp
    monkeypatch.setattr(remtp,'fixed6_vllm',fixed,raising=False)
    monkeypatch.setattr(remtp,'dynamic_tree_vllm',dynamic,raising=False)
    monkeypatch.setitem(sys.modules,'vllm.v1.spec_decode.eagle',SimpleNamespace(EagleProposer=Proposer))
    monkeypatch.setattr(module,'_INSTALLED',False)
    module.install_decision_trace(tmp_path)
    rng=torch.Generator().manual_seed(1);proposer=Proposer()
    proposer._fixed6_sampling_metadata=SimpleNamespace(generators={0:rng})
    kwargs=dict(batch_size=1,logits=p[:1],positions=torch.tensor([90]),hidden_states=p[:1],
                common_attn_metadata=SimpleNamespace(seq_lens=torch.tensor([91])))
    proposer.propose_tree(**kwargs);fixed.strict_tree_verify(topology,ids,q,p,rng)
    assert not list(tmp_path.glob('*/*.pt'))
    runtime.diagnostic_request='actual';runtime.diagnostic_prompt_ids=(1,)*90
    for i in range(10):
        runtime.generated_history=(5,)*i
        assert proposer.propose_tree(**kwargs)=='unchanged'
        fixed.strict_tree_verify(topology,ids,q,p,rng)
    assert len(list((tmp_path/'actual').glob('proposal_*.pt')))==8
    assert len(list((tmp_path/'actual').glob('verify_*.pt')))==8
    proposal=torch.load(tmp_path/'actual/proposal_00.pt',weights_only=False)
    frame=torch.load(tmp_path/'actual/verify_00.pt',weights_only=False)
    assert torch.equal(proposal['rng_after'],frame['rng_before'])
    assert frame['events'] and frame['config']==config
    assert module.active_trace() is None


def test_actual_mtp_position_shift_and_same_nan_masks_are_explicit():
    proposal=dict(history_before_current_sampling=(),topology=((0,),),
        root_positions=torch.tensor([170]), ids=torch.tensor([0]),
        q=torch.tensor([[float('nan'),float('nan')]]), rng_after=torch.tensor([1],dtype=torch.uint8))
    frame=dict(prompt_ids=(1,)*171, history=(4207,),topology=proposal['topology'],
        ids=proposal['ids'].clone(),draft=proposal['q'].clone(),rng_before=proposal['rng_after'].clone())
    assert proposal_boundary_matches(proposal,frame)
    proposal['root_positions']+=1
    assert not proposal_boundary_matches(proposal,frame)
    assert not equal_saved_tensor(torch.tensor([float('nan'),1.]),torch.tensor([1.,float('nan')]))


def test_nonfinite_proposal_rejected_before_rng_instead_of_uniform_verifier_mismatch():
    from remtp.dynamic_tree_vllm import _sample_q_token
    from remtp.fixed6_microtree import normalize_distribution
    q=torch.full((11,),float('nan'))
    assert torch.allclose(normalize_distribution(q),torch.full((11,),1/11))
    for seed in range(20):
        meta=SimpleNamespace(temperature=torch.tensor([.6]),generators={0:torch.Generator().manual_seed(seed)})
        before=meta.generators[0].get_state().clone()
        with pytest.raises(ValueError, match='Invalid MTP proposal Q'):
            _sample_q_token(q,meta)
        assert torch.equal(before,meta.generators[0].get_state())
