#!/usr/bin/env python3
"""Audit completed repair artifacts on CPU, using the run's frozen evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(data):
    return {(r['question_id'], r['seed']):r for r in data}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args();run=args.run_dir.resolve();code=run/'frozen_code'
    if args.output.exists():
        raise ValueError('Keep previous audits; use a new output path')
    sys.path.insert(0,str(code))
    from remtp.gsm8k_benchmark import extract_model_answer, load_gsm8k, PROMPT_TEMPLATE, benchmark_messages
    from remtp.optimization_metrics import summarize, paired_gate, clustered_intervals
    from scripts.run_adaptive_round5 import output_signature
    from scripts.run_raven_review_unit import component_hashes, compare_logits
    from scripts.run_raven_risk_repair import (CALIBRATION_IDS, CALIBRATION_ORDER,
        completed_root_evidence, registered_source_changes, select_cuda_frames)
    from scripts.run_raven_root_cause_unit import natural_frames, early_proposals
    from scripts.analyze_raven_root_causes import PAIRS, validate_summary
    import torch
    policy=read(run/'repair_policy.json');protocol=read(run/'protocol.json')
    root=Path(policy['dependence']);review=Path(policy['review_reference'])
    assert read(run/'exit.json')['exit_code']==0
    assert read(run/'exit.json')['server_cleaned'] and read(run/'status.json')['phase']=='COMPLETED_REPAIR_UNIT'
    assert not (run/'failure.json').exists()
    source=hashlib.sha256()
    for f in sorted([*code.glob('remtp/*.py'),*code.glob('scripts/*'),*code.glob('configs/*')]):
        if f.is_file():source.update(str(f.relative_to(code)).encode()+f.read_bytes())
    assert source.hexdigest()==protocol['source_sha256']
    components=read(run/'component_hashes.json')
    assert components['current']==component_hashes(code)
    assert components['actual_changes']==registered_source_changes(code,root/'frozen_code')
    assert digest(code/'data/gsm8k/train.jsonl')==protocol['train_sha256']
    assert digest(code/'data/gsm8k/test.jsonl')==protocol['test_sha256']
    data={r['question_id']:r for r in load_gsm8k(code/'data/gsm8k/train.jsonl')}
    hashes={}

    def validate(records,pairs,directory,observed):
        assert len(records)==len(pairs) and set(index(records))==pairs
        for r in records:
            pred,kind=extract_model_answer(r['output'])
            assert r['gold_answer']==data[r['question_id']]['gold_answer']
            assert (pred,kind)==(r['predicted_answer'],r['answer_source'])
            assert r['correct']==(pred==r['gold_answer'])
            assert r['max_tokens']==2048 and 0<r['output_tokens']<=2048 and r['client_seconds']>0
            assert len(r['output_token_ids'])==r['output_tokens'] and r['prompt_token_ids']
            assert r['observation_enabled']==observed and r['performance_sample']!=observed
            if r['finish_reason']=='length':assert r['output_tokens']==2048
            else:assert r['finish_reason']=='stop'
            response=read(directory/f"response_{r['question_id']}_{r['seed']}.json")
            payload=response['payload'];value=response['response'];choice=value['choices'][0]
            assert payload['temperature']==.6 and payload['max_tokens']==2048
            assert payload['seed']==r['seed'] and payload['return_token_ids']
            assert payload['chat_template_kwargs']=={'enable_thinking':False}
            assert payload['messages']==benchmark_messages(
                PROMPT_TEMPLATE.format(question=data[r['question_id']]['question']),empty_system_prompt=True)
            assert value['id']==r['response_id'] and value['prompt_token_ids']==r['prompt_token_ids']
            assert choice['token_ids']==r['output_token_ids']
            assert choice['message']['content']==r['output'] and choice['finish_reason']==r['finish_reason']
            assert value['usage']['completion_tokens']==r['output_tokens']
        hashes[str(directory/'requests.jsonl')]=digest(directory/'requests.jsonl')

    expected={(q,q*10+s) for q in CALIBRATION_IDS for s in (42,43)}
    saved=read(run/'analysis.json');datasets={};summaries={};envs={};historical_matches={}
    common=('MAX_MODEL_LEN','KV_CACHE_MEMORY_BYTES','QUANTIZATION','LOAD_FORMAT','ENABLE_PREFIX_CACHING',
        'REMTP_TARGET_BNB_UNIFORM_GEMM','REMTP_TARGET_BF16_FULL_ACCUM','REMTP_TARGET_BNB_ROW_BLOCK',
        'REMTP_TARGET_RMSNORM_ROW_LIMIT','REMTP_TARGET_LM_HEAD_ROW_BLOCK','TARGET_ATTENTION_BACKEND',
        'REMTP_TARGET_TRITON_2D','REMTP_TARGET_ATTENTION_FP32','REMTP_CANONICAL_TREE_ATTENTION',
        'REMTP_HEAD_BATCH_ROWS','MTP_TOKENS')
    for name in CALIBRATION_ORDER:
        directory=run/'calibration'/name
        records=rows(directory/'requests.jsonl');validate(records,expected,directory,False)
        datasets[name]=records;summaries[name]=summarize(records)
        assert summaries[name]==saved['summaries'][name]==read(directory/'summary.json')
        env=read(directory/'effective_env.json')
        for k in ('REMTP_DECISION_TRACE_DIR','REMTP_DECISION_TRACE_SUMMARY_JSONL',
                'REMTP_RUNTIME_PROFILE_JSONL','REMTP_TARGET_DIAGNOSTIC_DIR',
                'REMTP_PREFIX_DIAGNOSTIC_JSONL','REMTP_STATE_DIAGNOSTIC_DIR','REMTP_TREE_AUDIT_JSONL'):
            assert not env[k],('Unexpected calibration probe',name,k)
        assert env['REMTP_TARGET_STATE_AUDIT']==env['REMTP_MTP_OPERATOR_AUDIT']=='0'
        assert env['REMTP_DYNAMIC_TREE_RELAX_RISK_POLICY']==('target_odds' if name=='target_odds' else 'none')
        envs[name]={k:env[k] for k in common}
        assert envs[name]==envs['native']
        if name!='target_odds':
            historical=index(rows(review/'calibration'/name/'requests.jsonl'))
            historical_matches[name]=sum(output_signature(r)==output_signature(historical[k])
                                         for k,r in index(records).items())
            assert historical_matches[name]==16
        hashes[str(directory/'effective_env.json')]=digest(directory/'effective_env.json')

    comparisons={}
    for name in CALIBRATION_ORDER[:-1]:
        comparisons[name]=dict(quality=paired_gate(datasets[name],datasets['target_odds'],False),
            quality_speed=paired_gate(datasets[name],datasets['target_odds'],True),
            intervals=clustered_intervals(datasets[name],datasets['target_odds']))
        assert comparisons[name]==saved['comparisons'][name]

    root_rows,root_forensics=completed_root_evidence(root)
    candidate_rows=rows(run/'natural/target_odds/requests.jsonl')
    natural_pairs={(q,q*10+s) for q,s in PAIRS}
    validate(root_rows,natural_pairs,root/'natural/B',True)
    validate(candidate_rows,natural_pairs,run/'natural/target_odds',True)
    gate=paired_gate(root_rows,candidate_rows,False)
    smoke=read(run/'smoke_quality_gate.json')
    assert all(smoke[k]==v for k,v in gate.items())
    assert smoke['passed'] and all(c['passed'] for c in smoke['benefit_controls'])
    for n,observed in (('B',root_rows),('target_odds',candidate_rows)):
        current=index(datasets[n])
        assert all(output_signature(r)==output_signature(current[k]) and
            r['prompt_token_ids']==current[k]['prompt_token_ids'] and
            r['output_token_ids']==current[k]['output_token_ids'] for k,r in index(observed).items())

    numeric={};natural={}
    for unit,name,records in ((root,'B',root_rows),(run,'target_odds',candidate_rows)):
        directory=unit/'natural'/name;findings=read(unit/'natural_forensics.json')
        saved_by={(r['question_id'],r['seed']):r for r in findings['requests']}
        summary=rows(directory/'decision_summary.jsonl');couplings=[];risk_modes=Counter()
        for r in records:
            observed=[x for x in summary if x['request']==r['response_id'] or
                      x['request'].startswith(r['response_id']+'-')]
            item=saved_by[r['question_id'],r['seed']-r['question_id']*10]
            assert validate_summary(observed,r)==item['summary']
            coupled=natural_frames(directory,r)
            assert coupled==item['coupled_frames']
            assert early_proposals(directory,r,observed)==item['early_proposals']
            couplings.extend(coupled)
            for row in observed:
                for d in row['result']['node_diagnostics']:
                    if 'adaptive_risk_mode' in d:risk_modes[d['adaptive_risk_mode']]+=1
        natural[name]=dict(requests=6,coupled_frames=len(couplings),visited_risk_modes=dict(risk_modes))
        for c in couplings:
            for key,sha in (('frame','frame_sha256'),('raw_frame','raw_frame_sha256')):
                assert digest(Path(c[key]))==c[sha]
                hashes[c[key]]=c[sha]
        cases=read(unit/'numeric_cases.json');g=read(unit/'target_natural_prefix_gate.json')
        assert g['passed'] and g['complete'] and len(g['checks'])==len(cases['cases'])==12
        fresh={}
        for f in (unit/'numeric/target/target_probe').glob('*/logits_*.pt'):
            raw=torch.load(f,map_location='cpu',weights_only=True)
            assert raw['case_label'] not in fresh
            fresh[raw['case_label']]=(f,raw)
        checks=[]
        for c,old in zip(cases['cases'],g['checks']):
            label=cases['canonical_fresh_label'][c['label']]
            fresh_path,f=fresh[label+'_target'];prefix=c['prefix_ids']
            assert old['label']==c['label'] and old['fresh_source_label']==label
            assert tuple(f['prompt_ids'])==tuple(prefix) and not f['output_ids'] and f['rows']==1
            assert f['actual_positions'][-1]==len(prefix)-1 and f['actual_input_ids'][-1]==prefix[-1]
            assert not f['shape_probes']
            actual=torch.load(c['raw_frame'],map_location='cpu',weights_only=True)['first_logits']
            comparison=compare_logits(actual,f['last_logits'],protocol['eos_ids'])
            assert comparison==old['comparison'] and comparison['passed']
            checks.append(dict(label=c['label'],raw_equal=True,max_abs_logit=comparison['max_abs_logit'],
                generated_length=c['generated_length'],fresh_label=label))
            hashes[str(fresh_path)]=digest(fresh_path)
        assert len(fresh)==g['unique_fresh_calls_completed']==11
        ledger=read(unit/'ledger.json')
        assert all(c['attempted']==c['completed']<=c['limit'] for c in ledger['counts'].values())
        assert ledger['counts']['numeric']['completed']==len(fresh)
        numeric[name]=dict(cases=12,unique_fresh_calls=len(fresh),checks=checks)

    cuda=read(run/'cuda_gate.json')
    selected=select_cuda_frames(root,root_forensics)
    assert selected==read(run/'cuda_frames.json')['frames']
    assert cuda['passed'] and cuda['complete'] and cuda['verifier_calls']==48 and len(cuda['checks'])==16
    assert all(c['disabled_bitwise_equal'] and c['candidate_semantics'] for c in cuda['checks'])
    for a,c in zip(selected,cuda['checks']):
        assert all(c[k]==v for k,v in a.items())
        assert digest(Path(c['path']))==c['sha256']
    entries=read(run/'ledger.json')['entries']
    for c in selected:
        for mode in ('legacy','disabled','target_odds'):
            assert entries['cuda_verify:'+c['sha256']+'/'+mode]['status']=='completed'
    for f,sha in read(run/'dependency_gate.json')['files'].items():assert digest(root/f)==sha
    for n in ('B16_legacy_signature_gate.json','B6_root_actual_ids_gate.json',
              'candidate6_observer_actual_ids_gate.json'):assert read(run/n)['passed']
    assert not saved['development_point_gate_passed'] and not saved['throughput_interval_gate_passed']
    assert read(run/'development_result_gate.json')['complete'] and not read(run/'development_result_gate.json')['passed']

    b=index(datasets['B']);candidate=index(datasets['target_odds'])
    changes=[]
    for k,r in b.items():
        c=candidate[k];fields=('correct','finish_reason','output_tokens','client_seconds','predicted_answer','answer_source')
        n=next((i for i,(u,v) in enumerate(zip(r['output_token_ids'],c['output_token_ids'])) if u!=v),
               min(len(r['output_token_ids']),len(c['output_token_ids'])))
        changes.append(dict(question_id=k[0],base_seed=k[1]-k[0]*10,
            old={f:r[f] for f in fields},candidate={f:c[f] for f in fields},
            output_ids_equal=r['output_token_ids']==c['output_token_ids'],
            common_prefix_output_tokens=n))
    losses=[x for x in changes if x['old']['correct'] and not x['candidate']['correct']]
    gains=[x for x in changes if not x['old']['correct'] and x['candidate']['correct']]
    length_losses=[x for x in changes if x['old']['finish_reason']=='stop' and x['candidate']['finish_reason']=='length']
    length_gains=[x for x in changes if x['old']['finish_reason']=='length' and x['candidate']['finish_reason']=='stop']
    cost={}
    for name,records in datasets.items():
        trunc=[r for r in records if r['finish_reason']=='length']
        cost[name]=dict(truncated=len(trunc),correct=sum(r['correct'] for r in trunc),
            time_fraction=sum(r['client_seconds'] for r in trunc)/summaries[name]['client_seconds'],
            token_fraction=sum(r['output_tokens'] for r in trunc)/summaries[name]['output_tokens'])
    metrics={}
    for name in CALIBRATION_ORDER[:-1]:
        old,new=summaries[name],summaries['target_odds']
        metrics[name]={k:(new[k]/old[k]-1)*100 for k in
            ('e2e_tok_s','mean_seconds','output_tokens','mean_acceptance_length','decode_seconds_per_round')}
    end=datetime.strptime(read(run/'exit.json')['ended_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    value=dict(cpu_audit_passed=True,audited_utc=datetime.now(timezone.utc).isoformat(),
        finished_utc8=end.astimezone(timezone(timedelta(hours=8))).isoformat(),
        source_hash=protocol['source_sha256'],runtime_changes=components['actual_changes'],
        ledger=read(run/'ledger.json')['counts'],historical_signatures_equal=historical_matches,
        summaries=summaries,comparisons=comparisons,per_seed={n:{str(s):summarize(
            [r for r in records if r['seed']-r['question_id']*10==s]) for s in (42,43)}
            for n,records in datasets.items()},smoke_quality=smoke,natural=natural,numeric=numeric,
        cuda=dict(calls=48,frames=16,passed=True,scope=cuda['scope']),
        accuracy_losses=losses,accuracy_gains=gains,stop_to_length=length_losses,length_to_stop=length_gains,
        all_pair_changes=changes,metric_changes_pct=metrics,truncation_cost=cost,artifact_hashes=hashes,
        scope='CPU verification of completed artifacts; no new model/HTTP/CUDA calls. '
              '80 unobserved calibration samples = 8 observed training questions x2 seeds x5 methods. '
              'Numeric gates cover twelve natural cases/eleven different prefixes per unit; '
              'cross-unit overlap is not independent additional coverage. No universal state or quality guarantee.')
    with args.output.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(cpu_audit_passed=True,calibration_records=80,natural_records=12,
        accuracy_loss_pairs=[(r['question_id'],r['base_seed']) for r in losses],
        accuracy_gain_pairs=[(r['question_id'],r['base_seed']) for r in gains],
        output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':
    main()
