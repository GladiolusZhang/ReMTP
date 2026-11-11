#!/usr/bin/env python3
"""Freeze one new bounded unit; no restart or source edits during execution."""
import argparse
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
    p.add_argument('--unit',default='remtp-raven-probability-cap-20261007')
    args=p.parse_args();run=args.run_dir.resolve()
    if run.parent != ROOT/'results' or run.exists(): raise ValueError('Use new project/results directory')
    if subprocess.run(['systemctl','--user','is-active','--quiet',args.unit]).returncode==0:
        raise ValueError('Unit already active')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8000))==0: raise ValueError('Port 8000 occupied')
    link=ROOT/'doc/RESULT-RAVEN-PROBABILITY-CAP-LIVE.md'
    if link.exists() and not link.is_symlink(): raise ValueError('Refusing to overwrite real report')
    run.mkdir();code=run/'frozen_code';code.mkdir()
    for name in ('remtp','scripts','configs','tests'):
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('.venv','data','models','results'):
        (code/name).symlink_to(ROOT/name,target_is_directory=True)
    (run/'v4_evidence').symlink_to(ROOT/'results/target_prefill_round4_20261004/v4_evidence',target_is_directory=True)
    command=[str(ROOT/'.venv/bin/python'),str(code/'scripts/run_raven_probability_cap.py'),
             '--run-dir',str(run),'--reference-dir',str(args.reference_dir.resolve())]
    subprocess.run([*command,'--check-only'],cwd=ROOT,check=True)
    manifest=dict(unit=args.unit+'.service',run_dir=str(run),command=command,
                  natural_calls=7,conditional_cuda_calls=32,train_requests=144,
                  restart='no',retries=0,runtime_max_seconds=259200,heldout=False,formal=False)
    (run/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['systemd-run','--user','--unit='+args.unit,'--property=Restart=no',
        '--property=SuccessExitStatus=3','--property=RuntimeMaxSec=259200',
        '--property=TimeoutStopSec=90','--working-directory='+str(ROOT),*command],check=True)
    (ROOT/'results/raven_probability_cap_latest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if link.is_symlink(): link.unlink()
    link.symlink_to(Path('../results')/run.name/'report.md')
    print('进度：bash scripts/status_raven_probability_cap.sh')

if __name__=='__main__':main()
