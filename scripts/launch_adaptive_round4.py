#!/usr/bin/env python3
"""Freeze the current authorized candidate and launch one finite user unit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def copy_completed_natural(source,destination):
    """Copy completed requests only; never extend the old result directory."""
    import sys
    sys.path.insert(0,str(ROOT))
    from scripts.run_adaptive_round4 import numerical_source_check,NUMERICAL_KEYS
    protocol=json.loads((destination/'protocol.json').read_text())
    checked=numerical_source_check(source,protocol)
    old=json.loads((source/'protocol.json').read_text())
    for key in ('train_sha256','primary_prompt','no_think_template_sha256'):
        if old[key]!=protocol[key]:raise ValueError('Natural checkpoint dataset/prompt mismatch')
    from transformers import AutoTokenizer
    from remtp.gsm8k_benchmark import load_gsm8k,PROMPT_TEMPLATE,benchmark_messages
    import torch
    torch.set_num_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/FastMTP',trust_remote_code=True,local_files_only=True)
    template=(ROOT/'configs/fastmtp_no_think_chat_template.jinja').read_text()
    dataset={r['question_id']:r for r in load_gsm8k(ROOT/'data/gsm8k/train.jsonl')}
    jobs=json.loads((source/'jobs.json').read_text());copied={}
    for key,job in jobs.items():
        if not key.startswith('state_natural/'):continue
        old_dir=source/key;path=old_dir/'requests.jsonl'
        if not path.exists():continue
        records=[json.loads(line) for line in path.read_text().splitlines()]
        if not records:continue
        if [[r['question_id'],r['seed']] for r in records]!=job['pairs'][:len(records)]:
            raise ValueError('Completed natural checkpoint order mismatch')
        allowed={tuple(tokenizer.apply_chat_template(benchmark_messages(PROMPT_TEMPLATE.format(question=dataset[r['question_id']]['question']),empty_system_prompt=True),
            tokenize=True,add_generation_prompt=True,chat_template=template,enable_thinking=False)) for r in records}
        target=destination/key;target.mkdir(parents=True);(target/'target_probe').mkdir()
        requests=set()
        for directory in (old_dir/'target_probe').iterdir():
            files=sorted(directory.glob('logits_*.pt'))
            if not files:continue
            record=torch.load(files[0],map_location='cpu',weights_only=True)
            if record['prompt_ids'] in allowed:
                shutil.copytree(directory,target/'target_probe'/directory.name);requests.add(directory.name)
        if len(requests)!=len(records):raise ValueError('Missing/duplicate completed natural probe requests')
        shutil.copyfile(path,target/'requests.jsonl')
        shutil.copyfile(old_dir/'runtime_options.json',target/'runtime_options.json')
        lines=[line for line in (old_dir/'prefix_diagnostic.jsonl').read_text().splitlines() if json.loads(line).get('request') in requests]
        (target/'prefix_diagnostic.jsonl').write_text('\n'.join(lines)+'\n')
        copied[key]=dict(completed=len(records),captured_requests=sorted(requests))
    if not copied:raise ValueError('No complete natural request to checkpoint')
    (destination/'reused_natural.json').write_text(json.dumps(dict(source=str(source),read_only=True,
        files_checked=checked,groups=copied,numerical_protocol={k:protocol[k] for k in NUMERICAL_KEYS}),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--unit',default='remtp-adaptive-round4')
    p.add_argument('--reuse-prerequisites',type=Path)
    p.add_argument('--reuse-completed-natural',type=Path);args=p.parse_args()
    root=args.run_dir.resolve()
    if root.parent!=ROOT/'results':raise ValueError('Result directory must be directly under project/results')
    if root.exists():raise ValueError('New candidate requires a new result directory')
    root.mkdir(parents=True);code=root/'frozen_code';code.mkdir()
    if args.reuse_prerequisites:
        old=args.reuse_prerequisites.resolve()
        checks=json.loads((old/'real_state_validation/state_checks.json').read_text())
        prefill=json.loads((old/'prefill_validation/prefill_checks.json').read_text())
        if not checks['passed'] or any(c['first_operator_difference'] or c['first_kv_difference'] for c in prefill.values()):
            raise ValueError('Only complete passed prerequisites may be reused')
        for name in ('prefill_validation','real_state_validation'):
            (root/name).symlink_to((old/name).resolve(),target_is_directory=True)
        (root/'reused_prerequisites.json').write_text(json.dumps(dict(source=str(old),read_only=True,
            reason='Diagnostic final-chunk pairing correction only; numerical files/protocol checked again by the controller'),indent=2)+'\n')
    for name in ('remtp','scripts','configs','tests'):
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('.venv','data','models','results'):
        (code/name).symlink_to(ROOT/name,target_is_directory=True)
    (root/'v4_evidence').symlink_to(ROOT/'results/target_prefill_round4_20261004/v4_evidence',target_is_directory=True)
    command=[str(ROOT/'.venv/bin/python'),str(code/'scripts/run_adaptive_round4.py'),
        '--run-dir',str(root),'--prefill-dir',str(root/'prefill_validation'),
        '--state-audit-dir',str(root/'real_state_validation'),
        '--original-run',str(ROOT/'results/adaptive_optimization_20261004_target_canonical')]
    # Pure CPU protocol/budget check: no model service or GPU benchmark.
    subprocess.run([*command,'--check-only'],cwd=ROOT,check=True)
    if args.reuse_completed_natural:copy_completed_natural(args.reuse_completed_natural.resolve(),root)
    launch=dict(unit=args.unit+'.service',run_dir=str(root),command=command,
        restart='no',runtime_max_seconds=1209600,source_root=str(code))
    (root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    subprocess.run(['systemd-run','--user','--unit='+args.unit,'--property=Restart=no',
        '--property=SuccessExitStatus=3',
        '--property=RuntimeMaxSec=1209600','--working-directory='+str(ROOT),*command],check=True)
    latest=ROOT/'results/adaptive_round4_latest.json'
    temporary=latest.with_suffix('.tmp');temporary.write_text(json.dumps(launch,indent=2)+'\n');temporary.replace(latest)
    link=ROOT/'doc/RESULT-ADAPTIVE-RAVEN-ROUND4.md'
    if link.is_symlink():link.unlink()
    elif link.exists():raise ValueError('Round4 report entry is a real document; refusing to overwrite')
    link.symlink_to(Path('../results')/root.name/'report.md')
    print('进度：bash scripts/status_adaptive_round4.sh')
    print('报告：'+str(link))


if __name__=='__main__':main()
