#!/usr/bin/env python3
"""Freeze and detach one registered accuracy/speed search, without retries."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--unit',default='remtp-raven-dominance-search-20261009')
    a=p.parse_args();run=a.run_dir.resolve()
    if run.parent!=ROOT/'results' or run.exists():
        raise ValueError('Use a new project/results directory')
    if subprocess.run(['systemctl','--user','is-active','--quiet',a.unit]).returncode==0:
        raise ValueError('Unit already active')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8000))==0:
            raise ValueError('Port 8000 occupied')
    linger=subprocess.run(['loginctl','show-user',str(os.getuid()),'-p','Linger'],
        capture_output=True,text=True,check=True).stdout.strip()
    if linger!='Linger=yes':
        raise ValueError('Detached experiment requires Linger=yes')
    link=ROOT/'doc/RESULT-RAVEN-DOMINANCE-LIVE.md'
    if link.exists() and not link.is_symlink():
        raise ValueError('Refusing to replace a real report')
    run.mkdir();code=run/'frozen_code';code.mkdir()
    for name in ('remtp','scripts','configs','tests'):
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('.venv','data','models','results'):
        (code/name).symlink_to(ROOT/name,target_is_directory=True)
    for name in ('README.md','AGENTS.md'):
        shutil.copy2(ROOT/name,code/name)
    (run/'v4_evidence').symlink_to(ROOT/'results/target_prefill_round4_20261004/v4_evidence',target_is_directory=True)
    (run/'plan.md').write_bytes((ROOT/'plan/adaptive-raven-dominance-2026-10-09.md').read_bytes())
    command=[str(ROOT/'.venv/bin/python'),str(code/'scripts/run_raven_dominance_search.py'),
        '--run-dir',str(run),'--reference-dir',str(a.reference_dir.resolve())]
    subprocess.run([*command,'--check-only'],cwd=ROOT,check=True)
    policy=json.loads((run/'dominance_policy.json').read_text())
    manifest=dict(unit=a.unit+'.service',run_dir=str(run),command=command,limits=policy['limits'],
        candidate_count=policy['candidate_count'],runtime_max_seconds=policy['runtime_max_seconds'],
        objective=policy['objective'],retries=0,restart='no',formal=False,heldout=False,automatic_adoption=False)
    (run/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['systemd-run','--user','--unit='+a.unit,'--property=Restart=no',
        '--property=RuntimeMaxSec='+str(policy['runtime_max_seconds']),'--property=TimeoutStopSec=90',
        '--working-directory='+str(ROOT),*command],check=True)
    (ROOT/'results/raven_dominance_latest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if link.is_symlink():link.unlink()
    link.symlink_to(Path('../results')/run.name/'report.md')
    print('进度：bash scripts/status_raven_dominance.sh')


if __name__=='__main__':main()
