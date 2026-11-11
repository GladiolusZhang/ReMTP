from dataclasses import replace

import pytest
import torch

from remtp.dynamic_mtp_tree import (DynamicTreeConfig, DynamicTreeTopology,
    adaptive_trunk_relaxation_plan, residual_hit_tree_verify)
from remtp.fixed6_microtree import residual_distribution
from scripts.analyze_raven_root_causes import budget_counterexample


def config(**kwargs):
    return DynamicTreeConfig(max_nodes=3, max_depth=3, support_mode='residual_hit_anchor',
        adaptive_relaxation=True, relaxation_risk_policy='target_odds', **kwargs)


def test_target_odds_policy_removes_the_demonstrated_upstream_amplification():
    old=budget_counterexample()
    new=budget_counterexample('target_odds')
    assert old['root_marginal_shift_ratio']==pytest.approx(1.5,rel=2e-6)
    assert new['root_marginal_shift_ratio']==pytest.approx(1.,rel=2e-6)
    assert new['all_blocks_spend_no_more_TV']
    assert new['adaptive_root_output']==pytest.approx(new['uniform_root_output'],abs=1e-7)


@pytest.mark.parametrize('prob,mode',[(.0001,'strict'),(.01,'no_reallocation'),(.2,'none')])
def test_risk_tiers_are_target_relative_and_fixed(prob,mode):
    topology=DynamicTreeTopology.from_paths(((0,),(0,0),(0,0,0)))
    p=torch.tensor([[1-prob,prob],[.5,.5],[.5,.5],[.5,.5]])
    q=torch.full((3,2),.5);ids=torch.tensor([1,0,0])
    plan=adaptive_trunk_relaxation_plan(topology,ids,q,p,config=config())
    row=plan[0]
    assert row['adaptive_risk_mode']==mode
    if mode=='strict':
        assert row['adaptive_relaxation_tv']==row['baseline_relaxation_tv']==0
    elif mode=='no_reallocation':
        assert row['adaptive_relaxation_tv']<=row['baseline_relaxation_tv']+1e-7
    assert row['adaptive_block_tv']<=row['adaptive_block_baseline_tv']+1e-7


def test_strict_risk_allowance_is_removed_before_recovered_pool():
    topology=DynamicTreeTopology.from_paths(((0,),(0,0),(0,0,0)))
    p=torch.tensor([[.0001,.9999],[.05,.95],[.7,.3],[.5,.5]])
    q=torch.tensor([[.9,.1],[.99,.01],[.5,.5]])
    plan=adaptive_trunk_relaxation_plan(topology,torch.tensor([0,0,0]),q,p,config=config())
    assert plan[0]['baseline_relaxation_tv']==plan[0]['adaptive_relaxation_tv']==0
    # Only the third position has recovered allowance; the first contributes 0.
    assert plan[0]['adaptive_block_recovered_tv']==pytest.approx(
        plan[2]['baseline_relaxation_tv'],abs=1e-7)
    assert plan[1]['adaptive_relaxation_tv']==pytest.approx(
        plan[1]['baseline_relaxation_tv'],abs=1e-7)
    assert plan[0]['adaptive_block_spent_tv']==pytest.approx(0.,abs=1e-7)


@pytest.mark.parametrize('draft_y',[.9,.00005])
def test_strict_risk_preserves_exact_P_residual_draws_and_rng(draft_y):
    topology=DynamicTreeTopology.from_paths(((0,),))
    p=torch.tensor([[.0001,.6,.3999],[.2,.3,.5]])
    q=torch.tensor([[draft_y,(1-draft_y)*.4,(1-draft_y)*.6]])
    candidate=replace(config(),max_nodes=1,max_depth=1)
    strict=replace(candidate,relaxation_risk_policy='none',cactus_delta=0.)
    for seed in range(8):
        a=torch.Generator().manual_seed(seed);b=torch.Generator().manual_seed(seed)
        trace={}
        actual=residual_hit_tree_verify(topology,torch.tensor([0]),q,p,a,
            config=candidate,decision_trace=trace)
        expected=residual_hit_tree_verify(topology,torch.tensor([0]),q,p,b,config=strict)
        event=trace['events'][0]
        assert torch.equal(event['h'],event['p'])
        assert actual.output_token_ids==expected.output_token_ids
        assert actual.accepted_node_indices==expected.accepted_node_indices
        assert torch.equal(a.get_state(),b.get_state())
        assert actual.node_diagnostics[0]['adaptive_risk_mode']=='strict'
        assert actual.node_diagnostics[0]['relaxation_protected']
        if not event['keep']:
            assert torch.equal(trace['events'][1]['probs'],residual_distribution(event['p'],event['q']))


def test_eos_protection_still_removes_allowance_before_risk_allocation():
    topology=DynamicTreeTopology.from_paths(((0,),(0,0),(0,0,0)))
    # Stay above the threshold: FP32 normalization can round .01 just below .01.
    p=torch.tensor([[.5,.48,.02],[.2,.78,.02],[.7,.28,.02],[.5,.48,.02]])
    q=torch.tensor([[.9,.09,.01],[.99,.005,.005],[.8,.19,.01]])
    trace={}
    c=replace(config(),protect_eos_relaxation=True,eos_token_ids=(2,),eos_protection_threshold=.01)
    residual_hit_tree_verify(topology,torch.tensor([0,0,0]),q,p,
        torch.Generator().manual_seed(3),config=c,decision_trace=trace)
    assert all(torch.equal(e['h'],e['p']) for e in trace['events'] if e['kind']=='accept_uniform')


def test_disabled_risk_policy_has_no_new_result_metadata():
    topology=DynamicTreeTopology.from_paths(((0,),))
    p=torch.tensor([[.2,.8],[.3,.7]]);q=torch.tensor([[.9,.1]])
    c=replace(config(),max_nodes=1,max_depth=1,relaxation_risk_policy='none')
    value=residual_hit_tree_verify(topology,torch.tensor([0]),q,p,
        torch.Generator().manual_seed(2),config=c)
    assert not any('adaptive_risk_policy' in row or 'eos_or_prefix_protected' in row
                   for row in value.node_diagnostics)


@pytest.mark.parametrize('kwargs',[
    {'relaxation_risk_policy':'unknown'},
    {'relaxation_risk_policy':'target_odds','adaptive_relaxation':False},
    {'relaxation_risk_policy':'target_odds','support_mode':'residual_hit_strict'}])
def test_risk_policy_fails_closed_in_unsupported_modes(kwargs):
    c=DynamicTreeConfig(**kwargs)
    with pytest.raises(ValueError):c.validate()
