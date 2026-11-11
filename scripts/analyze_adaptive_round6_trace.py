#!/usr/bin/env python3
"""Read complete conditional tensors; do not infer causality from two answers."""
from collections import defaultdict
from pathlib import Path


def analyze(files):
    import torch
    from remtp.dynamic_mtp_tree import DynamicTreeTopology
    roots=defaultdict(list); low=[]; samples=[]; frames=[]; invalid=[]; mtp_passes=[]
    for path in files:
        frame=torch.load(path,map_location='cpu',weights_only=False)
        group=Path(path).parents[2].name
        topology=DynamicTreeTopology.from_paths(frame['topology'])
        proposal_path=Path(path).with_name(Path(path).name.replace('verify_','proposal_'))
        proposal=torch.load(proposal_path,map_location='cpu',weights_only=False)
        for number, detail in enumerate(proposal.get('tree_passes',[])):
            mtp_passes.append(dict(file=str(proposal_path),group=group,generated=len(frame['history']),
                pass_index=number,node_ids=detail['node_ids'],root_position=detail['root_position'],
                nonfinite_hidden_inputs=int((~torch.isfinite(detail['hidden_inputs'])).sum()),
                nonfinite_last_hidden=detail['nonfinite_last_hidden'],
                prefix_cache_checks=[{k:v for k,v in c.items() if k!='prefix_kv'} for c in detail['caches']]))
        for field in ('draft','target'):
            for index, row in enumerate(frame[field]):
                count=int((~torch.isfinite(row)).sum())
                if count:
                    invalid.append(dict(file=str(path),group=group,generated=len(frame['history']),
                        field=field,row=index,nonfinite_values=count,vocab=row.numel(),
                        token_id=int(frame['ids'][index]) if field=='draft' else None,
                        issue='Raw proposal is sampled before verifier fallback normalization' if field=='draft' else 'Invalid raw Target P'))
        # Root is the real, committed common prefix. Candidate prefixes are
        # already in the tensor dump but are not used as causal common roots.
        prefix=tuple(frame['prompt_ids'])+tuple(frame['history'])
        roots[prefix].append((str(path),frame['target'][0]))
        output_offset=0
        for event in frame['events']:
            kind=event['kind']
            if kind=='accept_uniform':
                token=event['token_id'];p=float(event['p'][token]);q=float(event['q'][token]);h=float(event['h'][token])
                tv=float((event['p']-event['h']).abs().sum()/2)
                strict=min(1.,p/max(q,1e-30))
                if event['keep'] and event['uniform']>strict and p<.001:
                    low.append(dict(file=str(path),group=group,generated=len(frame['history']),
                        output_offset=output_offset,token_id=token,P=p,Q=q,H=h,u=event['uniform'],
                        full_tv=tv,H_over_P=h/max(p,1e-30),strict_acceptance=strict,
                        relaxed_acceptance=min(1.,h/max(q,1e-30)),
                        node_path=topology.nodes[event['node']].path))
                if event['keep']:output_offset+=1
            else:
                samples.append(dict(file=str(path),group=group,generated=len(frame['history']),
                    output_offset=output_offset,role=kind,token_id=event['token'],
                    input_probability=float(event['probs'][event['token']]),
                    full_distribution_saved=True,original_rng_before_and_after_saved=True))
                output_offset+=1
        frames.append(dict(file=str(path),group=group,generated=len(frame['history']),round=frame['round'],
            visited=sum(e['kind']=='accept_uniform' for e in frame['events']),
            accepted=sum(e['kind']=='accept_uniform' and e['keep'] for e in frame['events']),
            terminal=frame['result']['terminal']))
    comparisons=[]
    for prefix, rows in roots.items():
        for index,(left,a) in enumerate(rows):
            for right,b in rows[index+1:]:
                if Path(left).parents[2].name==Path(right).parents[2].name:continue
                comparisons.append(dict(prefix_tokens=len(prefix),left=left,right=right,
                    raw_probability_identical=bool(torch.equal(a,b)),
                    full_probability_tv=float((a-b).abs().sum()/2),
                    max_abs=float((a-b).abs().max())))
    return dict(frames=frames,actual_common_root_comparisons=comparisons,nonfinite_probability_rows=invalid,
        mtp_state_passes=mtp_passes,
        low_target_relaxation_only_accepts=low,correction_and_bonus_samples=samples,
        limitations=['Only the first eight rounds were captured; no full late-KV audit.',
            'Saved categorical inputs are normalized inside sample_categorical before multinomial.',
            'Full common-root probability equality is scoped evidence, not inherited old raw-logit gates.',
            'Strict disagreement at the same u establishes a local relaxation effect, not corrected answer causality.',
            'Original CUDA RNG is used for verifier replays; CPU-generated seeds are not substitutes.'])
