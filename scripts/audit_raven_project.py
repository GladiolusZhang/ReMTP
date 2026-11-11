#!/usr/bin/env python3
"""CPU-only project review from frozen tensors and saved requests; no serving."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from remtp.cactus_mtp import cactus_target_distribution
from remtp.dynamic_mtp_tree import (DynamicTreeTopology, anchor_reusable_topology,
    level_expansion_order, residual_hit_tree_verify, select_dynamic_candidates)
from remtp.optimization_metrics import summarize, paired_gate


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def decision_events_equal(left,right):
    if len(left)!=len(right): return False
    for a,b in zip(left,right):
        # Physical row/node indices change under pruning. P/Q/H and draws may not.
        for key in set(a)|set(b):
            if key in ('node','parent_row'):continue
            x,y=a.get(key),b.get(key)
            if isinstance(x,torch.Tensor):
                if not isinstance(y,torch.Tensor) or not torch.equal(x,y):return False
            elif x!=y:return False
    return True


def main():
    torch.set_num_threads(1)
    repair=ROOT/'results/mtp_repaired_regression_20261006'
    cap=ROOT/'results/raven_probability_cap_20261007'
    files=sorted((repair/'repair_smoke/old_eos01/decision_trace').glob('*/verify_*.pt'))
    files+=sorted((repair/'diagnostics/old_eos01/decision_trace').glob('*/verify_*.pt'))
    files+=sorted((cap/'diagnostics/B_late/decision_trace').glob('*/verify_*.pt'))
    if len(files)!=29:raise ValueError('Expected exactly 8 short +16 natural +5 late B frames')
    prune=[];topk=[];passes=[]
    for index,path in enumerate(files):
        frame=torch.load(path,map_location='cpu',weights_only=False)
        assert torch.isfinite(frame['draft']).all() and torch.isfinite(frame['target']).all()
        topology=DynamicTreeTopology.from_paths(frame['topology'])
        reduced,keep=anchor_reusable_topology(topology)
        remap={old:new for new,old in enumerate(keep)}
        protected=frozenset(frame['protected_node_indices'])
        results=[];events=[];states=[]
        for pruned in (False,True):
            rng=torch.Generator().manual_seed(20261007+index);trace={}
            result=residual_hit_tree_verify(reduced if pruned else topology,
                frame['ids'][list(keep)] if pruned else frame['ids'],
                frame['draft'][list(keep)] if pruned else frame['draft'],
                frame['target'][[0]+[i+1 for i in keep]] if pruned else frame['target'],rng,
                config=replace(frame['config'],prune_anchor_descendants=pruned),
                protected_node_indices=frozenset(remap[i] for i in protected if i in remap) if pruned else protected,
                decision_trace=trace)
            results.append(result);events.append(trace['events']);states.append(rng.get_state())
        a,b=results
        passed=(a.output_token_ids==b.output_token_ids and a.terminal==b.terminal
            and [topology.nodes[i].path for i in a.accepted_node_indices]==[reduced.nodes[i].path for i in b.accepted_node_indices]
            and torch.equal(states[0],states[1]) and decision_events_equal(*events))
        assert passed,path
        prune.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          before=topology.node_count,after=reduced.node_count,passed=True))
        # Bounded exact selector checks on 16 real rows, including diagnostics.
        if len(topk)<16:
            q=frame['draft'][0]
            a=select_dynamic_candidates(q,depth=1,config=replace(frame['config'],exact_topk_candidates=False))
            b=select_dynamic_candidates(q,depth=1,config=replace(frame['config'],exact_topk_candidates=True))
            assert torch.equal(a.token_ids,b.token_ids)
            assert {k:v for k,v in a.__dict__.items() if k!='token_ids'}=={k:v for k,v in b.__dict__.items() if k!='token_ids'}
            topk.append(dict(file=str(path),passed=True))
        proposal=path.with_name(path.name.replace('verify_','proposal_'))
        if proposal.exists():
            p=torch.load(proposal,map_location='cpu',weights_only=False)
            sizes=[len(r['node_ids']) for r in p['tree_passes']]
            passes.append(dict(file=str(proposal),pass_node_rows=sizes,total_rows=sum(sizes),
                               final_nodes=topology.node_count))
    log=records(cap/'diagnostics/B_late/prefix_diagnostic.jsonl')
    rounds=[r for r in log if r['kind']=='verification']
    nodes=[n for r in rounds for n in r['nodes']]
    windows=[]
    for begin,end in ((0,128),(128,512),(512,1024),(1024,1792),(1792,2048)):
        rr=[r for r in rounds if begin<=r['generated_length']<end]
        nn=[n for r in rr for n in r['nodes']]
        windows.append(dict(begin=begin,end=end,rounds=len(rr),visited_primary=len(nn),
            max_target_eos=max(n['target_eos_probability'] for n in nn),
            mean_target_eos=sum(n['target_eos_probability'] for n in nn)/len(nn),
            mean_verification_eos=sum(n['verify_eos_probability'] for n in nn)/len(nn),
            protected=sum(n.get('relaxation_protected',False) for n in nn),
            accepted_only_under_relaxation=sum(n['normal_survival'] and n['accept_random']>
                min(1.,n['target_probability']/max(n['draft_probability'],1e-30)) for n in nn)))
    before=records(cap/'regression/B_seed42/requests.jsonl');after=records(cap/'regression/cap2_seed42/requests.jsonl')
    gate=paired_gate(before,after,require_speed=False)
    assert gate['failures']==json.loads((cap/'train_seed42_quality_gate.json').read_text())['failures']
    # Enumerate all proposed tokens: sampled-token-specific H is not a single P.
    p=torch.tensor([.55,.30,.15]);q=torch.tensor([.10,.80,.10]);laws=[]
    for delta in (0.,1.):
        law=torch.zeros_like(p)
        for y in range(3):
            h=cactus_target_distribution(p[None],torch.tensor([y]),delta)[0]
            alpha=min(1.,float(h[y]/q[y]));residual=(h-q).clamp_min(0)
            if float(residual.sum())>0:residual/=residual.sum()
            law[y]+=q[y]*alpha;law+=q[y]*(1-alpha)*residual
        assert abs(float(law.sum())-1)<1e-6
        if delta==0:assert torch.allclose(law,p,rtol=0,atol=1e-6)
        else:assert torch.allclose(law,q,rtol=0,atol=1e-6)
        laws.append(dict(delta=delta,p=p.tolist(),q=q.tolist(),output_law=law.tolist(),
                         total_variation_from_p=float((law-p).abs().sum()/2)))
    # A future strict continuation must account for selection after a Q draw.
    # Reserve the primary slot; two recovery parents compete for one slot.
    # Equal parent reach, Q_A=[.9,.1], Q_B=[.5,.5]. A is retained only on y=0.
    p_branch=torch.tensor([.5,.5]);q_branch=torch.tensor([.9,.1])
    branch_law=torch.zeros_like(p_branch);retention=[]
    branch_config=replace(frame['config'],allocation_mode='residual_coverage')
    for y in range(2):
        order=level_expansion_order(((float(q_branch[y]),(1,0)),(.5,(2,0))),
                                   config=branch_config)
        retained=order[0]==0;retention.append(retained)
        if retained:
            alpha=min(1.,float(p_branch[y]/q_branch[y]))
            residual=(p_branch-q_branch).clamp_min(0);residual/=residual.sum()
            conditional=(1-alpha)*residual;conditional[y]+=alpha
        else:
            conditional=p_branch
        branch_law+=q_branch[y]*conditional
    assert retention==[True,False]
    assert torch.allclose(branch_law,torch.tensor([.55,.45]),rtol=0,atol=1e-6)
    output=dict(pruning=dict(frames=prune,total_before=sum(r['before'] for r in prune),
        total_after=sum(r['after'] for r in prune),cpu_verifier_calls=58,
        scope='Copied real probabilities + fresh CPU RNG; no new MTP/Target/KV/GPU or runtime speed proof'),
        exact_topk=dict(cases=topk,cpu_selector_calls=32,scope='16 real root-Q rows; not GPU end-to-end proof'),
        draft_recomputation=dict(frames=passes,total_processed_rows=sum(r['total_rows'] for r in passes),
            total_final_nodes=sum(r['final_nodes'] for r in passes),scope='Observed node rows; not milliseconds or safe removal proof'),
        B1532_late=dict(rounds=len(rounds),exact_repeat_rounds=sum(r['repeat_suffix'] for r in rounds),
            max_target_eos=max(n['target_eos_probability'] for n in nodes),windows=windows,
            scope='One B trajectory, EOS on visited parent P; not Native same-prefix EOS or causal proof'),
        cap_seed42=gate,relaxed_marginal_example=laws,
        selected_branch_example=dict(p=p_branch.tolist(),q=q_branch.tolist(),
            retained_by_token=retention,output_law=branch_law.tolist(),
            total_variation_from_p=float((branch_law-p_branch).abs().sum()/2),
            scope='Toy using production allocation ordering; future strict-continuation proof obligation, not an active anchor failure'),
        budget=dict(original_planned_mixed_calls=2684,original_planned_main_http=2620,
                    original_conditional_cuda=64,natural_used=16,full_train_allocated=288,full_train_completed=192,
                    unused_train=96,scope='Original round6 allocation only; new repair/profile units must remain separately counted'),
        gpu_calls_this_audit=0,source_policy='Review adds an analysis script; algorithms and frozen sources unchanged')
    path=ROOT/'results/project_review_20261007.json';path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(prune_before=output['pruning']['total_before'],prune_after=output['pruning']['total_after'],
        frame_count=29,draft_processed_rows=output['draft_recomputation']['total_processed_rows'],
        draft_final_nodes=output['draft_recomputation']['total_final_nodes'],
        late_rounds=len(rounds),exact_repetition=sum(r['repeat_suffix'] for r in rounds),
        max_target_eos=output['B1532_late']['max_target_eos'],gpu_calls=0,artifact=str(path)),ensure_ascii=False))


if __name__=='__main__':main()
