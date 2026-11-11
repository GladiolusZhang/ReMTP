#!/usr/bin/env python3
"""CPU forensics on observed training outputs and actual verifier evidence."""
from __future__ import annotations

from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remtp.gsm8k_benchmark import extract_model_answer
from scripts.run_adaptive_round2 import write_json

PAIRS = ((6009, 42), (2950, 43), (3213, 43), (5180, 42),
         (1231, 43), (5180, 43))
CHECKPOINTS = (0, 4, 32, 96, 128, 512, 1536, 2000)


def budget_counterexample(risk_policy="none"):
    """Exact enumeration: less block TV can still shift the root more.

    This uses the current default delta/mix/max-ratio/target-power, two tokens,
    D3, and all eight equally likely primary draft paths. Future P equals Q,
    leaving unused local allowance available for the planner to move upstream.
    It is a mathematical diagnostic, not a measured task-quality failure.
    """
    import torch
    from remtp.cactus_mtp import cactus_target_distribution
    from remtp.dynamic_mtp_tree import (DynamicTreeConfig, DynamicTreeTopology,
        adaptive_trunk_relaxation_plan, _candidate_boost_distribution)
    from remtp.fixed6_microtree import residual_distribution
    topology = DynamicTreeTopology.from_paths([(0,), (0,0), (0,0,0)])
    config = DynamicTreeConfig(max_nodes=3, max_depth=3, support_mode='residual_hit_anchor',
        adaptive_relaxation=True, cactus_delta=1., relaxation_depth_weights=(1.,1.,1.),
        relaxation_risk_policy=risk_policy)
    q = torch.full((3,2), .5)
    p = torch.tensor([[.99,.01],[.5,.5],[.5,.5],[.5,.5]])
    uniform_output = torch.zeros(2); adaptive_output = torch.zeros(2)
    blocks = []
    for path in product((0,1), repeat=3):
        ids = torch.tensor(path)
        plan = adaptive_trunk_relaxation_plan(topology, ids, q, p, config=config)
        y = path[0]
        uniform = cactus_target_distribution(p[:1], ids[:1], config.cactus_delta)[0]
        adaptive = _candidate_boost_distribution(p[0], y,
            float(p[0,y])+plan[0]['adaptive_relaxation_tv'])
        for h, output in ((uniform, uniform_output), (adaptive, adaptive_output)):
            alpha = min(1., float(h[y])/float(q[0,y]))
            onehot = torch.zeros(2); onehot[y] = 1.
            output += (alpha*onehot+(1-alpha)*residual_distribution(h,q[0]))/8
        blocks.append(dict(path=path,
            uniform_block_tv=plan[0]['adaptive_block_baseline_tv'],
            adaptive_block_tv=plan[0]['adaptive_block_tv'],
            root_effective_delta=plan[0]['adaptive_effective_delta']))
    uniform_tv = float((uniform_output-p[0]).abs().sum()/2)
    adaptive_tv = float((adaptive_output-p[0]).abs().sum()/2)
    return dict(target_root=p[0].tolist(), draft_root=q[0].tolist(),
        uniform_root_output=uniform_output.tolist(), adaptive_root_output=adaptive_output.tolist(),
        uniform_root_marginal_tv=uniform_tv, adaptive_root_marginal_tv=adaptive_tv,
        all_blocks_spend_no_more_TV=all(
            b['adaptive_block_tv'] <= b['uniform_block_tv']+1e-7 for b in blocks),
        root_marginal_shift_ratio=adaptive_tv/uniform_tv, blocks=blocks,
        scope='Eight CPU-enumerated two-token D3 paths using current default planner. '
              'Shows block local-TV sum is insufficient to preserve root divergence; '
              'does not prove the cause of a particular GSM8K output.')


def requests(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def output_forensics(reference):
    """Text annotation only; the original scorer remains the scoring rule."""
    methods = {}
    for method in ('native', 'cactus', 'spec_cascade', 'B'):
        rows = requests(reference / 'calibration' / method / 'requests.jsonl')
        if len(rows) != 16:
            raise ValueError('Complete 16-request calibration required')
        methods[method] = {(r['question_id'], r['seed']-r['question_id']*10): r for r in rows}
        if len(methods[method]) != len(rows):
            raise ValueError('Duplicate calibration pair')
        for r in rows:
            prediction, source = extract_model_answer(r['output'])
            if (prediction, source, prediction == r['gold_answer']) != (
                    r['predicted_answer'], r['answer_source'], r['correct']):
                raise ValueError('Original scorer mismatch')
    pairs = []
    for qid, seed in PAIRS:
        entries = {}
        b = methods['B'][qid, seed]['output']
        for method, rows in methods.items():
            r = rows[qid, seed]
            text = r['output']
            common = 0
            while common < min(len(text), len(b)) and text[common] == b[common]:
                common += 1
            marker = text.find('####')
            entries[method] = dict(correct=r['correct'], finish=r['finish_reason'],
                tokens=r['output_tokens'], seconds=r['client_seconds'],
                prediction=r['predicted_answer'], answer_source=r['answer_source'],
                first_hash_character=marker if marker >= 0 else None,
                contains_gold_string=r['gold_answer'] in text,
                common_prefix_characters_with_B=common,
                first_text_difference_with_B=text[max(0, common-50):common+180],
                first_hash_context=text[max(0, marker-80):marker+240] if marker >= 0 else None,
                tail=text[-400:])
        pairs.append(dict(question_id=qid, seed=seed,
            role='failure' if (qid, seed) in PAIRS[:4] else 'benefit_control',
            methods=entries))
    return dict(pairs=pairs, budget_counterexample=budget_counterexample(),
        scope='Observed development text only. Same seed does not imply '
        'the same RNG path across methods. A gold substring is not proof of a valid final answer.')


def request_folder(directory, response_id):
    paths = [p for p in Path(directory).iterdir() if p.is_dir()
             and (p.name == response_id or p.name.startswith(response_id + '-'))]
    if len(paths) != 1:
        raise ValueError(f'Expected one engine request folder for {response_id}: {paths}')
    return paths[0]


def validate_summary(rows, record):
    """Use returned live IDs, including final EOS/budget clipping."""
    output = record['output_token_ids']
    if len(output) != record['output_tokens'] or not rows:
        raise ValueError('Missing live token IDs or decision summaries')
    counts = Counter()
    metrics = []
    expected_request = rows[0]['request']
    for number, row in enumerate(rows):
        n = row['generated_length']
        if row['request'] != expected_request or row['round'] != number or not 1 <= n < len(output):
            raise ValueError('Summary request/round/generated position mismatch')
        if row['history_suffix'] != output[max(0, n-16):n]:
            raise ValueError('Actual committed prefix differs from API token IDs')
        emitted = row['result']['output_token_ids']
        visible = min(len(emitted), len(output)-n)
        if emitted[:visible] != output[n:n+visible]:
            raise ValueError('Verifier output differs from API token IDs')
        if number+1 < len(rows) and rows[number+1]['generated_length'] != n+len(emitted):
            raise ValueError('Unexpected intermediate commit gap')
        counts['rounds'] += 1
        counts[row['result']['terminal']] += 1
        counts['validated_nodes'] += row['result']['target_validation_nodes']
        for node in row['result']['node_diagnostics']:
            if node.get('proposal_role') != 'sampled_cactus_primary':
                continue
            p, q, h = (float(node[k]) for k in
                ('target_probability', 'draft_probability', 'verification_probability'))
            u = float(node['accept_random'])
            strict_alpha = min(1., p/max(q, 1e-30))
            protected = node.get('eos_or_prefix_protected', node.get('relaxation_protected', False))
            delta = 0. if protected else node['base_effective_relaxation_delta']
            uniform_h = min(1., p+math.sqrt(max(0., 2*delta*p*(1-p))))
            uniform_alpha = min(1., uniform_h/max(q, 1e-30))
            adaptive_alpha = float(node['accept_probability'])
            # Flag float boundary cases separately instead of silently treating
            # CPU formulas as bitwise CUDA counterfactual replays.
            ambiguous = min(abs(u-strict_alpha), abs(u-uniform_alpha)) < 1e-6
            keep = bool(node['normal_survival'])
            counts['visited_primary'] += 1
            counts['accepted_primary'] += keep
            counts['protected_primary'] += bool(protected)
            counts['counterfactual_boundary'] += ambiguous
            if not ambiguous:
                counts['extra_accept_over_strict'] += keep and u > strict_alpha
                counts['accept_differs_from_uniform'] += keep != (u <= uniform_alpha)
            metrics.append(dict(generated_length=n, node=node['node'],
                depth=node['trunk_position'], token_id=row['candidate_ids'][node['node']],
                p=p, q=q, h=h, h_over_p=h/max(p, 1e-30), local_tv=max(0., h-p),
                effective_delta=node['effective_relaxation_delta'],
                base_delta=node['base_effective_relaxation_delta'],
                uniform_h=uniform_h, strict_alpha=strict_alpha, uniform_alpha=uniform_alpha,
                adaptive_alpha=adaptive_alpha, uniform=u, accepted=keep,
                extra_over_strict=keep and u>strict_alpha and not ambiguous,
                differs_from_uniform=keep != (u<=uniform_alpha) and not ambiguous,
                boundary=ambiguous, protected=protected,
                target_eos=node.get('target_eos_probability', 0.)))
    last = rows[-1]
    if last['generated_length']+len(last['result']['output_token_ids']) < len(output):
        raise ValueError('Final decision does not cover complete API output')
    return dict(counts=dict(counts), positions=metrics,
        scope='Observed actual uniforms with CPU scalar counterfactual thresholds; '
              'not a new natural generation or an RNG-equivalent GPU replay.')


def frame_risk(frame, eos_ids):
    import torch
    from remtp.cactus_mtp import cactus_target_distribution
    from remtp.fixed6_microtree import residual_distribution
    from remtp.dynamic_mtp_tree import DynamicTreeTopology
    topology = DynamicTreeTopology.from_paths(frame['topology'])
    details = {r['node']: r for r in frame['result']['node_diagnostics']}
    rows = []
    for event in frame['events']:
        if event['kind'] != 'accept_uniform':
            continue
        p, q, h = (event[k].float() for k in ('p', 'q', 'h'))
        y = event['token_id']
        if any(not bool(torch.isfinite(t).all()) for t in (p, q, h)):
            raise ValueError('Nonfinite actual P/Q/H')
        detail = details[event['node']]
        protected = detail.get('eos_or_prefix_protected', detail.get('relaxation_protected', False))
        delta = 0. if protected else (
            frame['config'].relaxation_delta(topology.nodes[event['node']].depth))
        uniform = cactus_target_distribution(p[None], torch.tensor([y]), delta)[0]
        py, qy, hy = float(p[y]), float(q[y]), float(h[y])
        rz = residual_distribution(h, q)
        rz_uniform = residual_distribution(uniform, q)
        rows.append(dict(node=event['node'], token_id=y, p=py, q=qy, h=hy,
            p_rank=1+int((p>p[y]).sum()), p_over_top=py/max(float(p.max()), 1e-30),
            h_over_p=hy/max(py, 1e-30), tv=float((h-p).abs().sum()/2),
            eos_p=float(p[eos_ids].sum()), eos_h=float(h[eos_ids].sum()),
            uniform_h=float(uniform[y]),
            adaptive_vs_uniform_H_tv=float((h-uniform).abs().sum()/2),
            adaptive_vs_uniform_residual_tv=float((rz-rz_uniform).abs().sum()/2),
            observed_uniform=event['uniform'], accepted=event['keep'],
            strict_keep=event['uniform']<=min(1., py/max(qy, 1e-30)),
            uniform_keep=event['uniform']<=min(1., float(uniform[y])/max(qy, 1e-30))))
        rows[-1]['residual_consumed'] = not event['keep']
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite forensic evidence')
    write_json(args.output, output_forensics(args.reference_dir.resolve()))
    print('CPU output/scorer forensics saved: '+str(args.output))


if __name__ == '__main__':
    main()
