#!/usr/bin/env python3
"""CPU evidence for returning to B; never load a model or draw RNG."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--repair-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output path; preserve prior evidence')
    baseline, repair = args.baseline_dir.resolve(), args.repair_dir.resolve()
    # Use the completed run's actual config classes and original evaluator.
    sys.path.insert(0, str(repair/'frozen_code'))
    import torch
    from scripts.analyze_raven_root_causes import validate_summary, request_folder
    torch.set_num_threads(2)
    audit = read(repair/'offline_analysis_20261009.json')
    assert audit['cpu_audit_passed']
    for directory in (baseline, repair):
        assert read(directory/'exit.json')['exit_code'] == 0
    hashes = {}
    datasets, grouped = [], []
    for directory, method in ((baseline, 'B'), (repair, 'target_odds')):
        natural = directory/'natural'/method
        request_file = natural/'requests.jsonl'
        summary_file = natural/'decision_summary.jsonl'
        hashes[str(request_file)] = sha(request_file)
        hashes[str(summary_file)] = sha(summary_file)
        assert audit['artifact_hashes'][str(request_file)] == sha(request_file)
        requests = {(r['question_id'], r['seed']): r for r in rows(request_file)}
        groups = defaultdict(list)
        for row in rows(summary_file):
            groups[row['request']].append(row)
        summaries = {}
        for key, record in requests.items():
            matching = [v for k, v in groups.items() if k.startswith(record['response_id']+'-') or k == record['response_id']]
            assert len(matching) == 1
            validate_summary(matching[0], record)
            summaries[key] = matching[0]
        datasets.append(requests)
        grouped.append(summaries)
    assert datasets[0].keys() == datasets[1].keys()
    eos_ids = read(repair/'protocol.json')['eos_ids']
    aligned = []
    frame_counts = Counter()
    eos_summary = {}
    for key, old in datasets[0].items():
        new = datasets[1][key]
        shared = common_prefix(old['output_token_ids'], new['output_token_ids'])
        assert old['prompt_token_ids'] == new['prompt_token_ids']
        changes, first_output_change = [], None
        for left, right in zip(grouped[0][key], grouped[1][key]):
            n = left['generated_length']
            if (n != right['generated_length'] or left['topology'] != right['topology']
                    or left['candidate_ids'] != right['candidate_ids']):
                break
            assert n <= shared
            ln = {x['node']: x for x in left['result']['node_diagnostics'] if x['proposal_role'] == 'sampled_cactus_primary'}
            rn = {x['node']: x for x in right['result']['node_diagnostics'] if x['proposal_role'] == 'sampled_cactus_primary'}
            for node in sorted(ln.keys() & rn.keys()):
                a, b = ln[node], rn[node]
                # These are actual scalar inputs and draws, not inferred seeds.
                for field in ('target_probability', 'draft_probability', 'accept_random'):
                    assert a[field] == b[field], (key, n, node, field)
                if a['verification_probability'] == b['verification_probability']:
                    continue
                changes.append(dict(round=left['round'], generated_length=n, node=node,
                    token_id=left['candidate_ids'][node], depth=a['trunk_position'],
                    p=a['target_probability'], q=a['draft_probability'],
                    b_h=a['verification_probability'], candidate_h=b['verification_probability'],
                    b_alpha=a['accept_probability'], candidate_alpha=b['accept_probability'],
                    uniform=a['accept_random'], b_keep=a['normal_survival'], candidate_keep=b['normal_survival'],
                    mode=b['adaptive_risk_mode'], support=b['adaptive_target_support_ratio'],
                    b_recovered_tv=a['adaptive_block_recovered_tv'], candidate_recovered_tv=b['adaptive_block_recovered_tv']))
            if left['result']['output_token_ids'] != right['result']['output_token_ids']:
                first_output_change = dict(round=left['round'], generated_length=n,
                    b_result=left['result']['output_token_ids'], candidate_result=right['result']['output_token_ids'],
                    accept_flips=[x for x in changes if x['round'] == left['round'] and x['b_keep'] != x['candidate_keep']],
                    full_distribution_captured=False)
                break
        aligned.append(dict(question_id=key[0], base_seed=key[1]-key[0]*10,
            shared_api_prefix_tokens=shared, output_ids_identical=old['output_token_ids']==new['output_token_ids'],
            h_changes_before_first_output_change=changes,
            changed_modes=dict(Counter(x['mode'] for x in changes)),
            first_output_change=first_output_change))

        # Only aligned registered frames permit full P/Q/RNG comparisons.
        pair_frames = []
        for directory, method, record in ((baseline, 'B', old), (repair, 'target_odds', new)):
            folder = request_folder(directory/'natural'/method/'decision_trace', record['response_id'])
            frames = {}
            eos_rows = eos_summary.setdefault(method, [])
            for path in sorted(folder.glob('verify_*.pt')):
                hashes[str(path)] = sha(path)
                frame = torch.load(path, map_location='cpu', weights_only=False)
                n = len(frame['history'])
                assert list(frame['history']) == record['output_token_ids'][:n]
                frames[n] = frame
                frame_counts[method] += 1
                for event in frame['events']:
                    if event['kind'] != 'accept_uniform':
                        continue
                    p, h = event['p'].float(), event['h'].float()
                    pe, he = float(p[eos_ids].sum()), float(h[eos_ids].sum())
                    eos_rows.append(dict(question_id=key[0], base_seed=key[1]-key[0]*10,
                        generated_length=n, node=event['node'], eos_p=pe, eos_h=he,
                        h_eos_over_p=he/pe if pe else None,
                        accepted=event['keep'], residual_consumed=not event['keep'],
                        candidate_is_eos=event['token_id'] in eos_ids))
            pair_frames.append(frames)
        comparisons = []
        for n in sorted(pair_frames[0].keys() & pair_frames[1].keys()):
            if n > shared:
                continue
            a, b = pair_frames[0][n], pair_frames[1][n]
            eq = dict(generated_length=n, history_equal=a['history']==b['history'],
                topology_equal=a['topology']==b['topology'], ids_equal=torch.equal(a['ids'], b['ids']),
                full_p_equal=torch.equal(a['target'], b['target']), full_q_equal=torch.equal(a['draft'], b['draft']),
                rng_before_equal=torch.equal(a['rng_before'], b['rng_before']))
            assert all(v for k, v in eq.items() if k != 'generated_length'), (key, eq)
            comparisons.append(eq)
            if first_output_change and n == first_output_change['generated_length']:
                first_output_change['full_distribution_captured'] = True
        aligned[-1]['registered_shared_prefix_full_frames'] = comparisons

    eos_stats = {}
    for method, data in eos_summary.items():
        method_index = 0 if method == 'B' else 1
        all_nodes = [node for sequence in grouped[method_index].values() for row in sequence
            for node in row['result']['node_diagnostics'] if node['proposal_role'] == 'sampled_cactus_primary']
        measurable = [r for r in data if r['eos_p'] > 0 and not r['candidate_is_eos']]
        suppressed = [r for r in measurable if r['eos_h'] < r['eos_p']*(1-1e-6)]
        eos_stats[method] = dict(visited_events=len(data), non_eos_positive_p_events=len(measurable),
            suppressed_events=len(suppressed), max_observed_eos_p=max(r['eos_p'] for r in data),
            below_1e_7=sum(r['eos_p']<1e-7 for r in data),
            minimum_observed_h_eos_over_p=min(r['h_eos_over_p'] for r in measurable), rows=data,
            all_round_scalar_eos=dict(visited_nodes=len(all_nodes),
                max_target_eos=max(n['target_eos_probability'] for n in all_nodes),
                below_1e_7=sum(n['target_eos_probability']<1e-7 for n in all_nodes),
                eos_or_prefix_protected=sum(n.get('eos_or_prefix_protected', n['relaxation_protected']) for n in all_nodes),
                scope='All visited primary scalar summaries, including rounds without complete H'),
            scope='Actual registered local H, not the algorithm marginal or all termination states')

    # A complete finite counterexample: preserving EOS in each candidate H
    # does not preserve EOS in the overall candidate-conditioned algorithm.
    p = torch.tensor([.45, .45, .1], dtype=torch.float64)
    q = torch.tensor([.7, .25, .05], dtype=torch.float64)
    hs = (torch.tensor([.7, .2, .1], dtype=torch.float64), p, p)
    marginal = torch.zeros_like(p)
    for y, h in enumerate(hs):
        assert torch.isclose(h.sum(), torch.tensor(1., dtype=h.dtype)) and h[2] == p[2]
        alpha = min(1., float(h[y]/q[y]))
        marginal[y] += q[y]*alpha
        if alpha < 1.:
            correction = (h-q).clamp_min(0.)
            marginal += q[y]*(1-alpha)*correction/correction.sum()
    assert torch.allclose(marginal, q) and marginal[2] < p[2]
    # For a candidate-independent H, exhaustive accept/correct enumeration
    # does reproduce H, including its EOS mass. This is a local math check.
    q_bar = q.clone()
    q_bar[:2] *= (1-p[2])/(1-q[2])
    q_bar[2] = p[2]
    causal_mixture_checks = []
    for beta in (0., .5, 1.):
        h = (1-beta)*p+beta*q_bar
        alpha = torch.minimum(h/q, torch.ones_like(q))
        result = q*alpha
        rejection_mass = float((q*(1-alpha)).sum())
        if rejection_mass:
            correction = (h-q).clamp_min(0.)
            result += rejection_mass*correction/correction.sum()
        assert torch.allclose(result, h) and torch.isclose(result[2], p[2])
        causal_mixture_checks.append(dict(beta=beta, common_h=h.tolist(), algorithm_marginal=result.tolist(),
            kl_to_p=float((h*(h/p).log()).sum())))
    summaries = audit['summaries']
    required_speed = summaries['cactus']['e2e_tok_s']/summaries['B']['e2e_tok_s']
    results = dict(cpu_audit_passed=True, audited_utc=datetime.now(timezone.utc).isoformat(),
        baseline_source_sha256=read(baseline/'protocol.json')['source_sha256'],
        parent_audit_sha256=sha(repair/'offline_analysis_20261009.json'),
        aligned_natural_pairs=aligned, registered_frame_counts=dict(frame_counts), eos=eos_stats,
        eos_preserving_h_counterexample=dict(p=p.tolist(), q=q.tolist(), h_by_candidate=[h.tolist() for h in hs],
            algorithm_marginal=marginal.tolist(), eos_index=2, local_h_eos=float(p[2]), marginal_eos=float(marginal[2])),
        candidate_independent_eos_mixture=dict(p=p.tolist(), q=q.tolist(), eos_projected_q=q_bar.tolist(),
            checks=causal_mixture_checks, scope='One prefix, three candidates; no task-quality or production-pipeline proof'),
        minimum_tps_gain_to_match_cactus_pct=(required_speed-1)*100,
        round_cost_reduction_at_unchanged_mal_pct=(1-1/required_speed)*100,
        artifact_hashes=hashes,
        new_gpu_calls=0, new_model_calls=0, new_http_calls=0,
        scope='Shared-prefix real scalar decisions and registered full frames on 6 known training pairs; '
              'toy marginal proof is not GSM8K end-to-end causality or a new benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=True, pairs=len(aligned), frames=dict(frame_counts),
        first_output_accept_flips=[dict(qid=r['question_id'], seed=r['base_seed'],
            first=r['first_output_change']) for r in aligned],
        eos={k:{a:b for a,b in v.items() if a not in ('rows','scope')} for k,v in eos_stats.items()},
        required_tps_gain_pct=results['minimum_tps_gain_to_match_cactus_pct'], output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
