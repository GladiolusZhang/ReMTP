#!/usr/bin/env python3
"""Freeze a single repair and detach its stdlib dependency waiter."""
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
    p.add_argument('--review-reference-dir',type=Path,required=True)
    p.add_argument('--unit',default='remtp-raven-target-odds-repair-20261008')
    a=p.parse_args();run=a.run_dir.resolve();reference=a.reference_dir.resolve()
    if run.parent!=ROOT/'results' or run.exists():
        raise ValueError('Use a new project/results directory')
    if subprocess.run(['systemctl','--user','is-active','--quiet',a.unit]).returncode==0:
        raise ValueError('Repair unit already active')
    dependency=json.loads((reference/'launch.json').read_text())
    if Path(dependency['run_dir']).resolve()!=reference:
        raise ValueError('Dependency launch directory differs')
    dependency_unit=dependency['unit']
    dependency_active=subprocess.run(['systemctl','--user','is-active','--quiet',
        dependency_unit]).returncode==0
    if not dependency_active and not (reference/'exit.json').exists():
        raise ValueError('Dependency has neither active unit nor exit evidence')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8000))==0 and not dependency_active:
            raise ValueError('Port occupied without registered active dependency')
    linger=subprocess.run(['loginctl','show-user',str(os.getuid()),'-p','Linger'],
        capture_output=True,text=True,check=True).stdout.strip()
    if linger!='Linger=yes':
        raise ValueError('Persistent user systemd requires Linger=yes')
    link=ROOT/'doc/RESULT-RAVEN-TARGET-ODDS-REPAIR-LIVE.md'
    if link.exists() and not link.is_symlink():
        raise ValueError('Refusing to overwrite a real report')
    run.mkdir();code=run/'frozen_code';code.mkdir()
    for name in ('remtp','scripts','configs','tests'):
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('.venv','data','models','results'):
        (code/name).symlink_to(ROOT/name,target_is_directory=True)
    for name in ('README.md','AGENTS.md'):
        shutil.copy2(ROOT/name,code/name)
    (run/'v4_evidence').symlink_to(
        ROOT/'results/target_prefill_round4_20261004/v4_evidence',target_is_directory=True)
    (run/'plan.md').write_bytes((ROOT/'plan/adaptive-raven-target-odds-repair-2026-10-08.md').read_bytes())
    command=[str(ROOT/'.venv/bin/python'),str(code/'scripts/run_raven_risk_repair.py'),
        '--run-dir',str(run),'--reference-dir',str(reference),
        '--review-reference-dir',str(a.review_reference_dir.resolve())]
    subprocess.run([*command,'--check-only'],cwd=ROOT,check=True)
    policy=json.loads((run/'repair_policy.json').read_text())
    wait_command=['/usr/bin/python3',str(code/'scripts/wait_raven_risk_repair.py'),
        '--run-dir',str(run),'--reference-dir',str(reference),'--dependency-unit',dependency_unit]
    manifest=dict(unit=a.unit+'.service',run_dir=str(run),command=wait_command,
        controller_command=command,dependency_unit=dependency_unit,reference_dir=str(reference),
        limits=policy['limits'],runtime_max_seconds=policy['runtime_max_seconds'],
        restart='no',retries=0,max_candidates=1,heldout=False,formal=False,
        automatic_adoption=False,waiting_loads_torch=False)
    (run/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['systemd-run','--user','--unit='+a.unit,'--property=Restart=no',
        '--property=RuntimeMaxSec='+str(policy['runtime_max_seconds']),
        '--property=TimeoutStopSec=90','--working-directory='+str(ROOT),*wait_command],check=True)
    (ROOT/'results/raven_risk_repair_latest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if link.is_symlink():
        link.unlink()
    link.symlink_to(Path('../results')/run.name/'report.md')
    print('进度：bash scripts/status_raven_risk_repair.sh')


if __name__=='__main__':
    main()
