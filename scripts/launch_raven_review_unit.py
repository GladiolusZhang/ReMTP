#!/usr/bin/env python3
"""Freeze and launch one finite review evidence unit, without automatic retries."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import socket
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--late-reference-dir',type=Path,required=True)
    p.add_argument('--unit',default='remtp-raven-review-20261007')
    p.add_argument('--continue-from',type=Path)
    args=p.parse_args();run=args.run_dir.resolve()
    if run.parent!=ROOT/'results' or run.exists():raise ValueError('Use a new project/results directory')
    if subprocess.run(['systemctl','--user','is-active','--quiet',args.unit]).returncode==0:
        raise ValueError('Unit is already active')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8000))==0:raise ValueError('Port 8000 occupied')
    linger=subprocess.run(['loginctl','show-user',str(__import__('os').getuid()),'-p','Linger'],
                          capture_output=True,text=True,check=True).stdout.strip()
    if linger!='Linger=yes':raise ValueError('Persistent user systemd session requires Linger=yes')
    link=ROOT/'doc/RESULT-RAVEN-REVIEW-LIVE.md'
    if link.exists() and not link.is_symlink():raise ValueError('Refusing to overwrite a real report')
    run.mkdir();code=run/'frozen_code';code.mkdir()
    for name in ('remtp','scripts','configs','tests'):
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('.venv','data','models','results'):
        (code/name).symlink_to(ROOT/name,target_is_directory=True)
    for name in ('README.md','AGENTS.md'):
        shutil.copy2(ROOT/name,code/name)
    (run/'v4_evidence').symlink_to(ROOT/'results/target_prefill_round4_20261004/v4_evidence',target_is_directory=True)
    (run/'plan.md').write_bytes((ROOT/'plan/adaptive-raven-project-review-optimization-2026-10-07.md').read_bytes())
    script='run_raven_review_continuation.py' if args.continue_from else 'run_raven_review_unit.py'
    command=[str(ROOT/'.venv/bin/python'),str(code/'scripts'/script),
        '--run-dir',str(run),'--reference-dir',str(args.reference_dir.resolve()),
        '--late-reference-dir',str(args.late_reference_dir.resolve())]
    if args.continue_from:command.extend(['--continue-from',str(args.continue_from.resolve())])
    subprocess.run([*command,'--check-only'],cwd=ROOT,check=True)
    policy=json.loads((run/'review_policy.json').read_text())
    runtime=policy['runtime_max_seconds']
    if policy.get('absolute_deadline_utc'):
        deadline=datetime.strptime(policy['absolute_deadline_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
        runtime=int((deadline-datetime.now(timezone.utc)).total_seconds())
        if runtime<=0:raise ValueError('Registered 24-hour deadline reached')
    manifest=dict(unit=args.unit+'.service',run_dir=str(run),command=command,
        main_http=policy['limits']['main_http'],numeric_diagnostics=policy['limits']['numeric'],
        warmups=policy['limits']['warmup'],max_starts=policy['limits']['starts'],retries=0,restart='no',
        runtime_max_seconds=runtime,heldout=False,formal=False,automatic_adoption=False,
        evidence_reuse=policy.get('evidence_reuse'),combined_attempt_limits=policy.get('combined_attempt_limits'))
    (run/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['systemd-run','--user','--unit='+args.unit,'--property=Restart=no',
        '--property=RuntimeMaxSec='+str(runtime),'--property=TimeoutStopSec=90',
        '--working-directory='+str(ROOT),*command],check=True)
    (ROOT/'results/raven_review_latest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if link.is_symlink():link.unlink()
    link.symlink_to(Path('../results')/run.name/'report.md')
    print('进度：bash scripts/status_raven_review_unit.sh')


if __name__=='__main__':main()
