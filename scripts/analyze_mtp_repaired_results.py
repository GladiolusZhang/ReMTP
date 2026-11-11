#!/usr/bin/env python3
"""Read-only post-run audit; writes a separate analysis artifact, no GPU calls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remtp.gsm8k_benchmark import (
    NUMBER_PATTERN, HASH_ANSWER_PATTERN, BOXED_ANSWER_PATTERN,
    normalize_number, extract_model_answer,
)
from remtp.optimization_metrics import summarize, paired_gate, clustered_intervals
from scripts.run_adaptive_round5 import output_signature


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.run_dir.resolve()
    hashes = {}
    def read(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(line) for line in path.read_text().splitlines()]
    status = json.loads((root/'status.json').read_text())
    if status['phase'] != 'COMPLETE_REPAIR_AND_LAZY_REGRESSION' or (root/'failure.json').exists():
        raise ValueError('Expected a fully completed repair unit without failure')
    policy = json.loads((root/'repair_execution_protocol.json').read_text())
    perf = json.loads((root/'performance_summary.json').read_text())
    for path in [root/'status.json', root/'repair_execution_protocol.json', root/'performance_summary.json', *root.glob('*_gate.json')]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if sum(len(read(path)) for path in (root/'repair_smoke').glob('*/requests.jsonl')) != 2:
        raise ValueError('Expected exactly two short probes')
    groups = {}
    effective_options = []
    for variant in ('B', 'lazy_residual'):
        rows = []
        for seed in (42,43,44):
            folder = root/'regression'/f'{variant}_seed{seed}'
            block = read(folder/'requests.jsonl')
            log = (folder/'server.log').read_text()
            lazy = int(variant == 'lazy_residual')
            if f'lazy_residual={lazy} ' not in log or '[ReMTP][HeadBatch] input_rows=1' not in log:
                raise ValueError('Effective worker options differ from variant')
            hashes[str(folder/'server.log')] = hashlib.sha256((folder/'server.log').read_bytes()).hexdigest()
            device = json.loads((folder/'device_before.json').read_text())
            hashes[str(folder/'device_before.json')] = hashlib.sha256((folder/'device_before.json').read_bytes()).hexdigest()
            effective_options.append(dict(variant=variant, seed=seed, lazy_residual=lazy,
                head_rows=1, power=device['power'], temperatures_before=device['thermal_celsius']))
            if len(block) != 24 or [r['question_id'] for r in block] != policy['train_ids']:
                raise ValueError('Training count/order mismatch')
            for r in block:
                predicted, source = extract_model_answer(r['output'])
                if (r['seed'] != r['question_id']*10+seed or predicted != r['predicted_answer']
                        or source != r['answer_source'] or r['correct'] != (predicted == r['gold_answer'])):
                    raise ValueError('Saved scoring/seed mismatch')
            rows.extend(block)
        if summarize(rows) != perf['baseline' if variant == 'B' else 'candidate']:
            raise ValueError('Aggregate differs from raw requests')
        groups[variant] = rows
    baseline, candidate = groups['B'], groups['lazy_residual']
    bmap = {(r['question_id'],r['seed']):r for r in baseline}
    exact = [output_signature(r) == output_signature(bmap[r['question_id'],r['seed']]) for r in candidate]
    if not all(exact) or len(exact) != 72:
        raise ValueError('Complete output equivalence failed')
    if paired_gate(baseline,candidate,require_speed=True) != perf['speed_gate']:
        raise ValueError('Speed/quality gate differs from raw data')
    if clustered_intervals(baseline,candidate) != perf['intervals']:
        raise ValueError('Question-cluster bootstrap differs from saved result')
    truncations = []
    by_question = defaultdict(list)
    for r in baseline:
        if r['finish_reason'] != 'length':
            continue
        markers = list(HASH_ANSWER_PATTERN.finditer(r['output'])) + list(BOXED_ANSWER_PATTERN.finditer(r['output']))
        earlier_gold = [m.start() for m in NUMBER_PATTERN.finditer(r['output'])
                        if normalize_number(m.group()) == r['gold_answer']]
        item = {k:r[k] for k in ('question_id','seed','correct','predicted_answer','gold_answer',
                                 'answer_source','output_tokens','client_seconds')}
        item.update(base_seed=r['seed']-r['question_id']*10, markers=len(markers),
                    contains_gold_number=bool(earlier_gold),
                    repeated_sentence_heuristic=bool(summarize([r])['repetitive_requests']),
                    tail=r['output'][-300:])
        truncations.append(item)
        by_question[r['question_id']].append(item)

    import torch
    from transformers import AutoTokenizer
    from scripts.analyze_adaptive_round5_faults import reconstruct
    from scripts.run_adaptive_round6 import proposal_boundary_matches
    tok = AutoTokenizer.from_pretrained(ROOT/'models/FastMTP', trust_remote_code=True, local_files_only=True)
    natural = []; initial = {}; risky = []
    for name in ('old_eos01', 'anchor4_eos01'):
        folder = root/'diagnostics'/name
        records = read(folder/'requests.jsonl'); logs = read(folder/'prefix_diagnostic.jsonl')
        reqs = list(dict.fromkeys(r['request'] for r in logs if r.get('kind') == 'verification'
            and len(r.get('sequential_replay', {}).get('prefix_ids', [])) > 64))
        if len(records) != 2 or len(reqs) != 2:
            raise ValueError('Expected two complete natural requests per variant')
        old = read(Path(policy['historical_reference'])/'diagnostics'/name/'requests.jsonl')
        for req, record, previous in zip(reqs,records,old):
            rows = [r for r in logs if r.get('kind')=='verification' and r['request']==req]
            reconstruct(rows,record,lambda ids:tok.decode(ids,skip_special_tokens=True))
            natural.append(dict(group=name, **{k:record[k] for k in ('question_id','seed','correct',
                'predicted_answer','gold_answer','answer_source','finish_reason','output_tokens','client_seconds')},
                exact_token_reconstruction=True, historical_output_identical=output_signature(record)==output_signature(previous)))
            initial[name,record['question_id']] = torch.load(folder/'decision_trace'/req/'proposal_00.pt',
                map_location='cpu',weights_only=False)['tree_passes'][0]
            for row in rows:
                for n in row['nodes']:
                    py=n['target_probability'];hy=n['verification_probability'];qy=n['draft_probability']
                    if n['trunk_accepted'] and py>0 and hy/py>100 and row['generated_length']<128:
                        risky.append(dict(group=name, question_id=record['question_id'], seed=record['seed'],
                            generated=row['generated_length'], token_id=n['token_id'], text=tok.decode([n['token_id']]),
                            p=py,q=qy,h=hy,h_over_p=hy/py,uniform=n['accept_random'],
                            strict_alpha=min(1.,py/max(qy,1e-30)),relaxed_alpha=n['accept_probability']))
    frame_checks=[]
    for phase in ('repair_smoke','diagnostics'):
        for path in sorted((root/phase).glob('*/decision_trace/*/verify_*.pt')):
            f=torch.load(path,map_location='cpu',weights_only=False)
            q=torch.load(path.with_name(path.name.replace('verify_','proposal_')),map_location='cpu',weights_only=False)
            frame_checks.append(dict(file=str(path), boundary=proposal_boundary_matches(q,f),
                finite_q=bool(torch.isfinite(f['draft']).all()),finite_p=bool(torch.isfinite(f['target']).all()),
                finite_hidden=all(t['nonfinite_last_hidden']==0 for t in q['tree_passes'])))
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    if len(frame_checks)!=48 or not all(all(r[k] for k in ('boundary','finite_q','finite_p','finite_hidden')) for r in frame_checks):
        raise ValueError('Expected 48 valid captured frames')
    a,b=initial['old_eos01',1231],initial['anchor4_eos01',1231]
    first=dict(inputs_identical=bool(torch.equal(a['hidden_inputs'],b['hidden_inputs'])),
               tokens_identical=bool(torch.equal(a['tokens'],b['tokens'])),
               outputs_identical=bool(torch.equal(a['last_hidden'],b['last_hidden'])),
               B_nonfinite=a['nonfinite_last_hidden'],anchor_nonfinite=b['nonfinite_last_hidden'])
    cuda=json.loads((root/'new_cuda_equivalence_gate.json').read_text())
    if not cuda['passed'] or cuda['cuda_verifier_calls']!=32 or not cuda['complete']:
        raise ValueError('CUDA gate incomplete')
    result=dict(source=str(root),analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        finished_utc=status['updated_utc'],complete_requests=150, exact_output_pairs=72,
        baseline=summarize(baseline),candidate=summarize(candidate),effective_options=effective_options,
        quality_gate=paired_gate(baseline,candidate,require_speed=False),speed_gate=perf['speed_gate'],intervals=perf['intervals'],
        throughput_change_percent=(perf['candidate']['e2e_tok_s']/perf['baseline']['e2e_tok_s']-1)*100,
        natural_requests=natural,captured_frames=frame_checks,q1231_first_mtp=first,
        cuda_calls=32,residual_calls_avoided=sum(c['residual_calls_avoided'] for c in cuda['checks']),
        risky_relaxation_observations=risky,truncations=truncations,
        truncation_cost=dict(requests=len(truncations), correct=sum(r['correct'] for r in truncations),
            seconds=sum(r['client_seconds'] for r in truncations),
            time_fraction=sum(r['client_seconds'] for r in truncations)/perf['baseline']['client_seconds'],
            tokens=sum(r['output_tokens'] for r in truncations),
            token_fraction=sum(r['output_tokens'] for r in truncations)/perf['baseline']['output_tokens'],
            repeated_sentence_overlap=sum(r['repeated_sentence_heuristic'] for r in truncations)),
        per_question_truncations={str(k):[r['base_seed'] for r in v] for k,v in by_question.items()},
        source_hashes=hashes,limitations=['CPU/offline only; no new GPU calls or training.',
            '24 selected train questions; 72 seed requests are not 72 independent questions.',
            'Finite raw tensors cover captured rounds; Q guards do not prove all Target/KV/long contexts.',
            'A correct number appearing earlier does not prove correct completed reasoning.',
            'Relaxation-only acceptance is local evidence, not proof of final answer causality.',
            'Three paper baselines and heldout/test are not part of this completed unit.'])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('finished_utc','complete_requests','exact_output_pairs',
        'throughput_change_percent','q1231_first_mtp','truncation_cost','per_question_truncations',
        'risky_relaxation_observations')},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
