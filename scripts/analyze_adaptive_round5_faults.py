#!/usr/bin/env python3
"""Offline token/decision forensics. Never loads a GPU model or changes traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def reconstruct(rows, record, decode):
    """Use the next committed length, not every speculative return token."""
    if not rows or 'sequential_replay' not in rows[0]:
        raise ValueError('Missing initial exact prefix')
    first=rows[0];generated=first['generated_length']
    if generated<=0:raise ValueError('Missing first committed output token')
    prefix=first['sequential_replay']['prefix_ids']
    prompt=prefix[:-generated];output=prefix[-generated:]
    events=[None]*generated
    for i,row in enumerate(rows):
        if row['generated_length']!=len(output):raise ValueError('Committed token positions do not match')
        if 'sequential_replay' in row and row['sequential_replay']['prefix_ids']!=prompt+output:
            raise ValueError('Exact prefix checkpoint does not match reconstructed history')
        count=(rows[i+1]['generated_length']-len(output)) if i+1<len(rows) else record['output_tokens']-len(output)
        if not 0<count<=len(row['output_ids']):raise ValueError('Missing or inconsistent committed output slice')
        output.extend(row['output_ids'][:count]);events.extend((i,j) for j in range(count))
    if len(output)!=record['output_tokens'] or decode(output)!=record['output']:
        raise ValueError('Reconstructed exact token IDs disagree with API response')
    return dict(prompt_ids=prompt,output_ids=output,events=events)


def event_detail(trace, rows, index):
    origin=trace['events'][index]
    if origin is None:return dict(role='initial target token; no verification event')
    row_index,offset=origin;row=rows[row_index]
    kept=[];rejected=None
    for node in row['nodes']:
        if node['trunk_accepted']:kept.append(node)
        else:rejected=node;break
    detail=dict(round_index=row_index,generated_length=row['generated_length'],offset=offset,
        terminal=row['terminal'],returned_ids=row['output_ids'],visited_primary=row['nodes'])
    if offset<len(kept):
        if kept[offset]['token_id']!=trace['output_ids'][index]:raise ValueError('Primary token/return mismatch')
        detail.update(role='accepted primary',primary=kept[offset])
    elif rejected is not None and offset==len(kept):
        detail.update(role='sampled residual correction',rejected_primary=rejected,
            missing='Full H/Q and correction-token probability are not in the scalar log')
    elif ((row['terminal']=='residual-hit-target-anchor' and rejected is not None and offset==len(kept)+1)
          or (row['terminal']=='sampled-primary-full-cactus-bonus' and rejected is None and offset==len(kept))):
        detail.update(role='unmodified Target bonus',missing='Actual bonus row full P and sampling RNG are not in scalar log')
    else:raise ValueError('Unsupported event role; refusing inference')
    return detail


def compare(left,right,left_rows,right_rows,decode):
    if left['prompt_ids']!=right['prompt_ids']:raise ValueError('Different exact prompt IDs')
    a,b=left['output_ids'],right['output_ids']
    index=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    result=dict(common_output_tokens=index,same_output=a==b,
        common_prefix_ids=left['prompt_ids']+a[:index],
        common_tail=decode(a[max(0,index-24):index]),
        limitation='A different token at equal seed is not evidence of incorrect distributions or a causal quality regression')
    for name,trace,rows in (('left',left,left_rows),('right',right,right_rows)):
        if index<len(trace['output_ids']):
            result[name]=dict(token_id=trace['output_ids'][index],
                token_text=decode([trace['output_ids'][index]]),
                following_text=decode(trace['output_ids'][index:index+20]),
                decision=event_detail(trace,rows,index))
        else:result[name]=dict(role='end of committed response')
    return result


def statistics(rows,trace,decode):
    visited=[(r,n) for r in rows for n in r['nodes']]
    accepted=[(r,n) for r,n in visited if n['trunk_accepted']]
    extra=[(r,n) for r,n in accepted if n['accept_random']>min(1,n['target_probability']/max(n['draft_probability'],1e-30))]
    low=[]
    for r,n in extra:
        if n['target_probability']>=.001 or r['generated_length']>128:continue
        low.append(dict(generated_length=r['generated_length'],trunk_position=n['trunk_position'],
            token_id=n['token_id'],token_text=decode([n['token_id']]),P=n['target_probability'],
            Q=n['draft_probability'],H=n['verification_probability'],u=n['accept_random'],
            strict_acceptance=n['target_probability']/max(n['draft_probability'],1e-30),
            relaxed_acceptance=n['accept_probability'],local_tv=n['relaxation_tv'],
            committed_tail=decode(trace['output_ids'][max(0,r['generated_length']-24):r['generated_length']]),
            returned_text=decode(r['output_ids'])))
    return dict(rounds=len(rows),visited_primary=len(visited),accepted_primary=len(accepted),
        accepted_only_due_to_relaxation=len(extra),early_low_target_accepted=low,
        max_accepted_local_tv=max(n['relaxation_tv'] for r,n in accepted),
        exact_repeat_rounds=sum(r['repeat_suffix'] for r in rows),
        protected_visited_primary=sum(n.get('relaxation_protected',False) for r,n in visited),
        residual_computations_for_accepted_primary=len(accepted),
        limitation='Counts cover visited primary nodes; hypothetical strict rejection does not prove corrected full answers')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path);args=p.parse_args()
    source=args.run_dir.resolve();out=args.output_dir.resolve()
    if out.exists():raise ValueError('Use a new offline evidence directory')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/FastMTP',local_files_only=True,trust_remote_code=True)
    decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True)
    proof={};groups={};cases={}
    for name in ('old','old_eos01','anchor4_eos01'):
        directory=source/'diagnostics'/name;path=directory/'prefix_diagnostic.jsonl'
        lines=[json.loads(s) for s in path.read_text().splitlines()]
        request_ids=list(dict.fromkeys(r['request'] for r in lines
            if len(r.get('sequential_replay',{}).get('prefix_ids',[]))>64))
        records=[json.loads(s) for s in (directory/'requests.jsonl').read_text().splitlines()]
        if len(request_ids)!=len(records) or [(r['question_id'],r['seed']) for r in records]!=[(1231,12353),(5180,51843)]:
            raise ValueError('Unexpected diagnostic request pairing')
        for file in (path,directory/'requests.jsonl',directory/'runtime_options.json'):
            proof[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        for request,record in zip(request_ids,records):
            rows=[r for r in lines if r.get('request')==request and r.get('kind')=='verification']
            trace=reconstruct(rows,record,decode);key=str(record['question_id'])
            groups[name,key]=(rows,trace)
            cases.setdefault(key,dict(base_seed=43,actual_seed=record['seed'],groups={}))['groups'][name]=dict(
                request=request,correct=record['correct'],finish_reason=record['finish_reason'],
                output_tokens=record['output_tokens'],stats=statistics(rows,trace,decode),
                first_accept_random=rows[0]['nodes'][0]['accept_random'],
                first_root=rows[0]['sequential_replay'])
    for key,case in cases.items():
        case['comparisons']={}
        for left,right in (('old','anchor4_eos01'),('old','old_eos01')):
            lr,lt=groups[left,key];rr,rt=groups[right,key]
            case['comparisons'][left+'__'+right]=compare(lt,rt,lr,rr,decode)
        a=case['groups']['old']['first_root'];b=case['groups']['anchor4_eos01']['first_root']
        case['first_root_top8_and_eos_equal']=all(a[k]==b[k] for k in ('prefix_ids','top_ids','top_probs','eos_ids','eos_probs'))
    out.mkdir(parents=True)
    (out/'fault_localization.json').write_text(json.dumps(dict(source=str(source),source_hashes=proof,cases=cases,
        protocol='Offline exact token reconstruction, next committed length, EOS/API decode checks; no fresh GPU test'),
        ensure_ascii=False,indent=2)+'\n')
    for key,case in cases.items():
        fork=case['comparisons']['old__anchor4_eos01']
        print(f"题{key}/seed43：同前缀{fork['common_output_tokens']}输出token；首个分叉角色 "
              f"{fork['left']['decision']['role']} → {fork['right']['decision']['role']}")
    print('证据：'+str(out/'fault_localization.json'))


if __name__=='__main__':main()
