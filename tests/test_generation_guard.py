from dataclasses import replace

import pytest
import torch

from remtp.dynamic_mtp_tree import DynamicTreeConfig, DynamicTreeTopology, adaptive_trunk_relaxation_plan, residual_hit_tree_verify
from remtp.generation_guard import repeated_suffix
from remtp.optimization_metrics import paired_gate, clustered_intervals


def test_repeat_detection_requires_consecutive_blocks_and_has_no_cross_request_state():
    assert repeated_suffix([1,2,3,4]*3)
    assert not repeated_suffix([1,2,3,4]*2)
    assert not repeated_suffix([1,2,3,4]*2+[1,2,3,5])
    assert not repeated_suffix([1,2,3,4])


@pytest.mark.parametrize("adaptive", [False, True])
def test_guarded_block_matches_strict_verification_outputs_and_rng(adaptive):
    topology=DynamicTreeTopology.from_paths([(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1)])
    ids=torch.tensor([0,1,0,1,0,1])
    config=DynamicTreeConfig(max_nodes=6,support_mode="residual_hit_anchor",adaptive_relaxation=adaptive)
    protected=frozenset(n.index for n in topology.nodes if n.path==(0,)*n.depth)
    for seed in range(40):
        rng=torch.Generator().manual_seed(seed)
        q=torch.rand(6,4,generator=rng);p=torch.rand(7,4,generator=rng)
        a=torch.Generator().manual_seed(seed+200);b=torch.Generator().manual_seed(seed+200)
        guarded=residual_hit_tree_verify(topology,ids,q,p,a,config=config,protected_node_indices=protected)
        strict=residual_hit_tree_verify(topology,ids,q,p,b,config=replace(config,cactus_delta=0))
        assert guarded.output_token_ids==strict.output_token_ids
        assert guarded.accepted_node_indices==strict.accepted_node_indices
        assert torch.equal(a.get_state(),b.get_state())


def test_guard_removes_budget_before_reallocation():
    t=DynamicTreeTopology.from_paths([(0,),(0,0),(0,0,0)])
    ids=torch.tensor([0,0,0]);q=torch.tensor([[.1,.9],[.9,.1],[.9,.1]])
    p=torch.tensor([[.6,.4],[.1,.9],[.1,.9],[.5,.5]])
    c=DynamicTreeConfig(support_mode="residual_hit_anchor",adaptive_relaxation=True,adaptive_relaxation_mix=1)
    plan=adaptive_trunk_relaxation_plan(t,ids,q,p,config=c,protected_node_indices=frozenset({0}))
    assert plan[0]["adaptive_relaxation_tv"]==0
    assert plan[0]["baseline_relaxation_tv"]==0
    assert plan[0]["adaptive_block_recovered_tv"]==pytest.approx(0,abs=1e-7)


def test_eos_guard_applies_to_anchor_and_keeps_exact_residual(monkeypatch):
    import remtp.dynamic_mtp_tree as module
    t=DynamicTreeTopology.from_paths([(0,)])
    p=torch.tensor([[.1,.1,.8],[.2,.1,.7]])
    q=torch.tensor([[.9,.05,.05]])
    captured=[]
    original=module.sample_categorical
    def sample(probs,generator):
        captured.append(probs.clone())
        return original(probs,generator)
    monkeypatch.setattr(module,"sample_categorical",sample)
    c=DynamicTreeConfig(max_depth=1,max_nodes=1,support_mode="residual_hit_anchor",adaptive_relaxation=True,
                        protect_eos_relaxation=True,eos_token_ids=(2,))
    r=residual_hit_tree_verify(t,torch.tensor([0]),q,p,torch.Generator().manual_seed(0),config=c)
    assert r.node_diagnostics[0]["relaxation_protected"]
    assert r.node_diagnostics[0]["relaxation_tv"]==pytest.approx(0)
    assert r.node_diagnostics[0]["verification_probability"]==pytest.approx(.1)
    assert torch.allclose(captured[0],torch.tensor([0.,.05,.75])/.8)
    assert not r.accepted_node_indices


def test_invalid_eos_is_rejected():
    with pytest.raises(ValueError,match="requires eos"):
        DynamicTreeConfig(protect_eos_relaxation=True).validate()


def test_moderate_eos_guard_uses_strict_sampling_instead_of_forcing_eos():
    # EOS is neither top-1 nor >=0.5: the old protection does not activate.
    topology=DynamicTreeTopology.from_paths([(0,)])
    ids=torch.tensor([0]);q=torch.tensor([[.95,.03,.02]])
    p=torch.tensor([[.6,.23,.17],[.6,.23,.17]])
    base=DynamicTreeConfig(max_depth=1,max_nodes=1,support_mode="residual_hit_anchor",
        adaptive_relaxation=True,protect_eos_relaxation=True,eos_token_ids=(2,))
    relaxed=residual_hit_tree_verify(topology,ids,q,p,torch.Generator().manual_seed(0),config=base)
    assert not relaxed.node_diagnostics[0]['relaxation_protected']
    assert relaxed.node_diagnostics[0]['relaxation_tv']>0
    strict=replace(base,cactus_delta=0)
    for seed in range(20):
        a=torch.Generator().manual_seed(seed);b=torch.Generator().manual_seed(seed)
        protected=residual_hit_tree_verify(topology,ids,q,p,a,config=replace(base,eos_protection_threshold=.01))
        reference=residual_hit_tree_verify(topology,ids,q,p,b,config=strict)
        assert protected.node_diagnostics[0]['relaxation_tv']==0
        assert protected.output_token_ids==reference.output_token_ids
        assert torch.equal(a.get_state(),b.get_state())


def row(qid,seed=42,correct=True,finish="stop",seconds=10):
    return dict(question_id=qid,seed=seed+qid*10,correct=correct,finish_reason=finish,output="#### 1",answer_source="hash",output_tokens=20,client_seconds=seconds)


def test_quality_gate_rejects_truncated_correct_and_per_seed_regression():
    old=[row(1,42),row(1,43,False)]
    new=[row(1,42,finish="length",seconds=8),row(1,43,True,seconds=8)]
    gate=paired_gate(old,new)
    assert not gate["passed"]
    assert "42:stopped_correct" in gate["failures"]
    assert "42:truncated" in gate["failures"]
    with pytest.raises(ValueError,match="matching"):
        paired_gate(old,[row(2)])


def test_bootstrap_retains_paired_questions_and_seeds():
    old=[row(q,s) for q in (1,2,3) for s in (42,43,44)]
    new=[{**r,"client_seconds":r["client_seconds"]/2} for r in old]
    ci=clustered_intervals(old,new,draws=100)
    assert ci["throughput_ratio_95ci"]==[2,2]
    assert ci["accuracy_difference_95ci"]==[0,0]
