#!/usr/bin/env python3
"""CPU-only audit of the completed review evidence, using its frozen scorer."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir.resolve();code=root/'frozen_code'
    # Load the recorded implementation before importing any project modules.
    sys.path.insert(0,str(code))
    from remtp.gsm8k_benchmark import extract_model_answer,load_gsm8k
    from remtp.optimization_metrics import summarize,paired_gate,clustered_intervals
    from scripts.run_adaptive_round5 import output_signature
    from scripts.run_raven_review_unit import (CALIBRATION_IDS,PROFILE_ORDER,
        component_hashes_for_reference,compare_logits,profile_summary)
    from scripts.run_raven_review_continuation import load_frame,aligned
    import torch

    status=json.loads((root/'status.json').read_text())
    exit_info=json.loads((root/'exit.json').read_text())
    assert status['phase']=='COMPLETED_EVIDENCE_UNIT' and exit_info['exit_code']==0 and exit_info['server_cleaned']
    assert not (root/'failure.json').exists()
    policy=json.loads((root/'review_policy.json').read_text())
    ledger=json.loads((root/'ledger.json').read_text())
    assert all(r['attempted']==r['completed']==r['limit'] for r in ledger['counts'].values())
    reuse=json.loads((root/'evidence_reuse_gate.json').read_text());origin=Path(reuse['origin'])
    assert reuse['passed']
    hashes={}
    for name,sha in reuse['artifact_hashes'].items():
        path=origin/name;assert digest(path)==sha,('Changed original evidence',name);hashes[str(path)]=sha
    components=component_hashes_for_reference(code)
    assert components==reuse['runtime_hashes']
    saved=json.loads((root/'analysis.json').read_text())
    dataset={r['question_id']:r for r in load_gsm8k(code/'data/gsm8k/train.jsonl')}
    protocol=json.loads((root/'protocol.json').read_text())
    assert digest(code/'data/gsm8k/train.jsonl')==protocol['train_sha256']
    expected={(qid,qid*10+seed) for qid in CALIBRATION_IDS for seed in (42,43)}

    def validate(records,pairs):
        assert len(records)==len(pairs) and {(r['question_id'],r['seed']) for r in records}==pairs
        for row in records:
            pred,source=extract_model_answer(row['output'])
            assert row['gold_answer']==dataset[row['question_id']]['gold_answer']
            assert (pred,source)==(row['predicted_answer'],row['answer_source'])
            assert row['correct']==(pred==row['gold_answer'])
            assert row['max_tokens']==2048 and 0<row['output_tokens']<=2048
            assert row['client_seconds']>0 and row['finish_reason'] in ('length','stop')

    records={};summaries={};profiles={};effective={}
    common_keys=('REMTP_TARGET_BNB_UNIFORM_GEMM','REMTP_TARGET_BF16_FULL_ACCUM',
        'REMTP_TARGET_BNB_ROW_BLOCK','REMTP_TARGET_RMSNORM_ROW_LIMIT','REMTP_TARGET_LM_HEAD_ROW_BLOCK',
        'TARGET_ATTENTION_BACKEND','REMTP_TARGET_TRITON_2D','REMTP_TARGET_ATTENTION_FP32',
        'REMTP_CANONICAL_TREE_ATTENTION','ENABLE_PREFIX_CACHING','MTP_TOKENS','REMTP_HEAD_BATCH_ROWS',
        'MAX_MODEL_LEN','KV_CACHE_MEMORY_BYTES')
    probe_keys=('REMTP_RUNTIME_PROFILE_JSONL','REMTP_TARGET_DIAGNOSTIC_DIR',
        'REMTP_PREFIX_DIAGNOSTIC_JSONL','REMTP_DECISION_TRACE_DIR','REMTP_TREE_AUDIT_JSONL')
    for name in PROFILE_ORDER:
        directory=root/'calibration'/name
        path=directory/'requests.jsonl';rows=read(path);validate(rows,expected)
        records[name]=rows;summaries[name]=summarize(rows)
        assert summaries[name]==json.loads((directory/'summary.json').read_text())==saved['calibration'][name]
        env=json.loads((directory/'effective_env.json').read_text())
        assert all(env[k]=='' for k in probe_keys)
        assert env['REMTP_TARGET_STATE_AUDIT']=='0' and env['REMTP_MTP_OPERATOR_AUDIT']=='0'
        effective[name]={k:env[k] for k in common_keys}
        assert effective[name]==effective[PROFILE_ORDER[0]]
        profile_path=root/'profile'/name/'profile.jsonl'
        profiles[name]=profile_summary(profile_path)
        assert profiles[name]==saved['profile'][name]
        hashes[str(path)]=digest(path)
        hashes[str(directory/'effective_env.json')]=digest(directory/'effective_env.json')
    b=records['B'];b_index={(r['question_id'],r['seed']):r for r in b};comparisons={}
    for name in PROFILE_ORDER[:-1]:
        a=records[name];sa=summaries[name];sb=summaries['B']
        changes=[]
        for row in a:
            other=b_index[row['question_id'],row['seed']]
            if row['correct']!=other['correct'] or row['finish_reason']!=other['finish_reason']:
                keys=('correct','finish_reason','output_tokens','predicted_answer','answer_source','client_seconds')
                changes.append(dict(question_id=row['question_id'],base_seed=row['seed']-row['question_id']*10,
                    baseline={k:row[k] for k in keys},raven={k:other[k] for k in keys}))
        comparisons[name]=dict(quality_gate=paired_gate(a,b,require_speed=False),
            quality_and_speed_gate=paired_gate(a,b,require_speed=True),
            intervals=clustered_intervals(a,b),changes=changes,
            throughput_change_pct=(sb['e2e_tok_s']/sa['e2e_tok_s']-1)*100,
            mean_latency_change_pct=(sb['mean_seconds']/sa['mean_seconds']-1)*100,
            output_change_pct=(sb['output_tokens']/sa['output_tokens']-1)*100,
            acceptance_length_change_pct=(sb['mean_acceptance_length']/sa['mean_acceptance_length']-1)*100,
            round_cost_change_pct=(sb['decode_seconds_per_round']/sa['decode_seconds_per_round']-1)*100)
    target_path=root/'target_only/target/requests.jsonl';target=read(target_path)
    validate(target,{(1532,1532*10+seed) for seed in (42,43)})
    target_summary=summarize(target)
    assert target_summary==json.loads((root/'target_only/target/summary.json').read_text())
    hashes[str(target_path)]=digest(target_path)

    raw_gate=json.loads((root/'target_prefix_gate.json').read_text())
    cases=json.loads((root/'numeric_cases.json').read_text())['cases'];raw_checks=[]
    assert raw_gate['passed'] and raw_gate['complete'] and len(raw_gate['checks'])==len(cases)==5
    eos=protocol['eos_ids']
    for index,case in enumerate(cases):
        tree_path,tree=load_frame(origin,case,'tree')
        source=root if index==4 else origin
        target_frame_path,target_frame=load_frame(source,case,'target')
        assert aligned(tree,case,'tree') and aligned(target_frame,case,'target')
        check=compare_logits(tree['first_logits'],target_frame['last_logits'],eos)
        assert check['passed'] and check['max_abs_logit']==0
        historical=torch.load(case['frame'],map_location='cpu',weights_only=False)
        tv=float((torch.softmax(tree['first_logits'].float()/.6,-1)-historical['target'][0]).abs().sum()/2)
        assert tv<=1e-6 and tv==raw_gate['checks'][index]['fresh_tree_vs_saved_P_tv']
        raw_checks.append(dict(case=case['label'],generated_length=case['generated_length'],
            raw_equal=check['raw_equal'],max_abs_logit=check['max_abs_logit'],tv=check['probability_tv'],
            saved_P_tv=tv,target_eos=check['target_eos'],actual_final_position=target_frame['actual_positions'][-1]))
        for p in (tree_path,target_frame_path):hashes[str(p)]=digest(p)
    historical={}
    for p in Path(policy['output_reference']).glob('regression/B_seed*/requests.jsonl'):
        historical.update({(r['question_id'],r['seed']):r for r in read(p)});hashes[str(p)]=digest(p)
    signature_matches=sum(output_signature(r)==output_signature(historical[r['question_id'],r['seed']]) for r in b)
    reproduction=json.loads((root/'B_reproduction_gate.json').read_text())
    assert signature_matches==16 and reproduction['passed'] and len(reproduction['checks'])==16
    per_seed={name:{str(s):summarize([r for r in data if r['seed']-r['question_id']*10==s])
        for s in (42,43)} for name,data in records.items()}
    trunc=[r for r in b if r['finish_reason']=='length']
    details=[]
    for r in b:
        s=summarize([r]);details.append(dict(question_id=r['question_id'],base_seed=r['seed']-r['question_id']*10,
            correct=r['correct'],finish_reason=r['finish_reason'],output_tokens=r['output_tokens'],
            answer_source=r['answer_source'],predicted_answer=r['predicted_answer'],gold_answer=r['gold_answer'],
            repetitive=bool(s['repetitive_requests']),credible=bool(s['credible_stopped_correct']),
            seconds=r['client_seconds']))
    origin_ledger=json.loads((origin/'ledger.json').read_text())
    combined={k:dict(attempted=ledger['counts'][k]['attempted']+origin_ledger['counts'][k]['attempted'],
        completed=ledger['counts'][k]['completed']+origin_ledger['counts'][k]['completed'],
        limit=policy['combined_attempt_limits'][k]) for k in ledger['counts']}
    assert all(v['attempted']<=v['limit'] for v in combined.values())
    b_cost=dict(requests=len(trunc),correct=sum(r['correct'] for r in trunc),
        seconds=sum(r['client_seconds'] for r in trunc),
        time_fraction=sum(r['client_seconds'] for r in trunc)/summaries['B']['client_seconds'],
        output_fraction=sum(r['output_tokens'] for r in trunc)/summaries['B']['output_tokens'])
    for name in ('analysis.json','review_policy.json','exit.json','ledger.json','target_prefix_gate.json','B_reproduction_gate.json'):
        hashes[str(root/name)]=digest(root/name)
    result=dict(analyzed_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        analyzed_utc8=datetime.now(timezone(timedelta(hours=8))).isoformat(),source=str(root),
        analysis_script_sha256=digest(Path(__file__)),exit=exit_info,combined_counts=combined,
        summaries=summaries,per_seed=per_seed,comparisons=comparisons,profile=profiles,
        target_only_summary=target_summary,target_only=[{k:r[k] for k in
            ('question_id','seed','correct','finish_reason','output_tokens','client_seconds','predicted_answer','answer_source')} for r in target],
        numeric_checks=raw_checks,baseline_identical_output_signatures=signature_matches,
        signature_scope='Exact output text, output token count, finish reason; API records do not contain token IDs',
        B_requests=details,B_truncation_cost=b_cost,effective_common=effective,
        source_hashes=hashes,new_gpu_requests=0,candidate_selected=False,
        limitations=['Eight previously observed, problem-focused train questions; two seeds each, not sixteen independent questions.',
            'Four methods ran in separate services; question-cluster bootstrap does not model machine or ordering uncertainty.',
            'Five raw checks use saved question1532 histories; not full KV or all quality-failure histories.',
            'Target-only repeats on one question/two seeds establish reproduction, not unique cause of all truncations.',
            'No new optimization candidate, independent heldout/test, uniform ablation or HumanEval in this unit.'])
    if args.output.exists():raise ValueError('Use a new offline artifact; do not overwrite evidence')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(finished_utc=exit_info['ended_utc'],combined_counts=combined,summaries=summaries,
        target_only_summary=target_summary,baseline_identical_output_signatures=signature_matches,
        numeric_cases=len(raw_checks),quality_gates={n:v['quality_gate']['passed'] for n,v in comparisons.items()},
        B_truncation_cost=b_cost),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
