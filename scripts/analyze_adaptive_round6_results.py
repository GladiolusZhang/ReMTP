#!/usr/bin/env python3
"""Offline post-run audit of completed Round6 outputs and fork distributions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from transformers import AutoTokenizer
    from scripts.analyze_adaptive_round5_faults import reconstruct, compare
    from remtp.dynamic_mtp_tree import DynamicTreeTopology
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True,type=Path)
    args=p.parse_args();root=args.run_dir.resolve()
    tok=AutoTokenizer.from_pretrained(ROOT/'models/FastMTP',local_files_only=True,trust_remote_code=True)
    decode=lambda ids:tok.decode(ids,skip_special_tokens=True)
    groups={};hashes={};requests=[];initial_proposals={}
    for name in ('old_eos01','anchor4_eos01'):
        folder=root/'diagnostics'/name
        logs=[json.loads(s) for s in (folder/'prefix_diagnostic.jsonl').read_text().splitlines()]
        records=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
        request_ids=list(dict.fromkeys(r['request'] for r in logs if r.get('kind')=='verification'
            and len(r.get('sequential_replay',{}).get('prefix_ids',[]))>64))
        if len(records)!=2 or len(request_ids)!=2:
            raise ValueError('Expected two completed natural requests per group')
        for file in (folder/'prefix_diagnostic.jsonl',folder/'requests.jsonl'):
            hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        for req,record in zip(request_ids,records):
            rows=[r for r in logs if r.get('kind')=='verification' and r.get('request')==req]
            trace=reconstruct(rows,record,decode)
            frames=[];target_rows={};sampling_events={}
            files=sorted((folder/'decision_trace'/req).glob('verify_*.pt'))
            if len(files)!=8:raise ValueError('Expected first eight captured rounds')
            for file in files:
                frame=torch.load(file,map_location='cpu',weights_only=False)
                hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
                if tuple(trace['output_ids'][:len(frame['history'])])!=frame['history']:
                    raise ValueError('Frame history does not match complete reconstructed output')
                topology=DynamicTreeTopology.from_paths(frame['topology'])
                path_index={n.path:n.index for n in topology.nodes}
                base=tuple(frame['prompt_ids'])+tuple(frame['history'])
                target_rows.setdefault(base,[]).append((str(file),0,frame['target'][0]))
                for node in topology.nodes:
                    ancestry=tuple(int(frame['ids'][path_index[node.path[:i]]]) for i in range(1,node.depth+1))
                    target_rows.setdefault(base+ancestry,[]).append((str(file),node.index+1,frame['target'][node.index+1]))
                prefix=base
                for event in frame['events']:
                    if event['kind']=='accept_uniform':
                        if event['keep']:prefix+=(event['token_id'],)
                    else:
                        sampling_events.setdefault(prefix,[]).append((str(file),event))
                        prefix+=(event['token'],)
                frames.append(frame)
            key=record['question_id']
            initial_proposals[name,key]=torch.load(files[0].with_name('proposal_00.pt'),map_location='cpu',weights_only=False)
            groups[name,key]=(rows,trace,target_rows,sampling_events)
            requests.append(dict(group=name,**{k:record[k] for k in ('question_id','seed','correct',
                'predicted_answer','gold_answer','answer_source','finish_reason','output_tokens','client_seconds')},
                exact_token_reconstruction=True,captured_rounds=len(frames)))
    forks={}
    for question in (1231,5180):
        a,at,ap,ae=groups['old_eos01',question];b,bt,bp,be=groups['anchor4_eos01',question]
        fork=compare(at,bt,a,b,decode);prefix=tuple(fork['common_prefix_ids'])
        pa=ap.get(prefix,[]);pb=bp.get(prefix,[])
        shared=[]
        for fa,ia,ra in pa:
            for fb,ib,rb in pb:
                shared.append(dict(left_file=fa,left_row=ia,right_file=fb,right_row=ib,
                    raw_processed_P_identical=bool(torch.equal(ra,rb)),
                    full_TV=float((ra-rb).abs().sum()/2),max_abs=float((ra-rb).abs().max())))
        sa=ae.get(prefix,[]);sb=be.get(prefix,[]);sampling=[]
        for fa,ea in sa:
            for fb,eb in sb:
                aa=ea['probs']/ea['probs'].sum();bb=eb['probs']/eb['probs'].sum()
                sampling.append(dict(left_file=fa,right_file=fb,left_role=ea['kind'],right_role=eb['kind'],
                    left_token=ea['token'],right_token=eb['token'],left_text=decode([ea['token']]),
                    right_text=decode([eb['token']]),input_distributions_identical=bool(torch.equal(ea['probs'],eb['probs'])),
                    normalized_input_TV_cpu=float((aa-bb).abs().sum()/2),
                    original_cuda_rng_before_identical=bool(torch.equal(ea['rng_before'],eb['rng_before']))))
        forks[str(question)]=dict(first_different_token_1based=fork['common_output_tokens']+1,
            common_tail=fork['common_tail'],event_roles=[fork['left']['decision']['role'],fork['right']['decision']['role']],
            exact_fork_prefix_target_rows=shared,actual_fork_sampling=sampling)
    gate_files=sorted(root.glob('*_gate.json'))
    gates={file.name:json.loads(file.read_text()) for file in gate_files}
    for file in gate_files:hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    cuda=gates['cuda_equivalence_gate.json']
    if not cuda['complete'] or cuda['cuda_verifier_calls']!=64 or len(cuda['checks'])!=32:
        raise ValueError('Incomplete conditional CUDA budget')
    left=initial_proposals['old_eos01',1231];right=initial_proposals['anchor4_eos01',1231]
    a=left['tree_passes'][0];b=right['tree_passes'][0]
    comparison=dict(root_hidden_identical=bool(torch.equal(left['root_hidden'],right['root_hidden'])),
        tree_input_hidden_identical=bool(torch.equal(a['hidden_inputs'],b['hidden_inputs'])),
        tree_input_hidden_maxabs=float((a['hidden_inputs'].float()-b['hidden_inputs'].float()).abs().max()),
        tree_tokens_identical=bool(torch.equal(a['tokens'],b['tokens'])),
        root_positions_identical=bool(torch.equal(left['root_positions'],right['root_positions'])),
        proposal_seq_lens_identical=bool(torch.equal(left['seq_lens'],right['seq_lens'])),
        prefix_KV_identical=bool(torch.equal(a['caches'][0]['prefix_kv'],b['caches'][0]['prefix_kv'])),
        prefix_KV_maxabs=float((a['caches'][0]['prefix_kv'].float()-b['caches'][0]['prefix_kv'].float()).abs().max()),
        B_nonfinite_output=a['nonfinite_last_hidden'],anchor_nonfinite_output=b['nonfinite_last_hidden'],
        scope='First MTP pass q1231: logical prefix and hidden only; current-node KV/full metadata/operator inputs not captured')
    result=dict(run_dir=str(root),requests=requests,gates={k:v['passed'] for k,v in gates.items()},forks=forks,
        q1231_first_mtp_input_comparison=comparison,analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cuda=dict(frames=32,verifier_calls=64,passed=cuda['passed'],
            residual_calls_avoided=sum(c['residual_calls_avoided'] for c in cuda['checks'])),
        source_hashes=hashes,limitations=['Offline audit; no new GPU calls.',
            'Fork P equality is for captured processed probability rows, not a universal raw-logit/KV proof.',
            'CPU-normalized categorical input TV is descriptive; actual CUDA RNG states are compared as bytes.',
            'Sampled seed changes and distribution differences do not alone establish final answer causality.'])
    (root/'postrun_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('requests','gates','forks','cuda')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
