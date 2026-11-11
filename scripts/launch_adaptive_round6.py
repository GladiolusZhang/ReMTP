#!/usr/bin/env python3
"""Freeze, preflight, then launch one bounded background unit."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--reference-dir',required=True,type=Path)
    p.add_argument('--unit',default='remtp-adaptive-round6')
    p.add_argument('--prior-aborted-dir',type=Path)
    args=p.parse_args();root=args.run_dir.resolve()
    link=ROOT/'doc/RESULT-ADAPTIVE-RAVEN-ROUND6.md'
    if root.parent!=ROOT/'results' or root.exists():raise ValueError('Use a new project/results directory')
    if link.exists() and not link.is_symlink():raise ValueError('Refusing to overwrite a real report')
    if subprocess.run(['systemctl','--user','is-active','--quiet',args.unit]).returncode==0:
        raise ValueError('Unit already active')
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',8000))==0:raise ValueError('Port 8000 occupied')
    root.mkdir();code=root/'frozen_code';code.mkdir()
    for name in ('remtp','scripts','configs','tests'):
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('.venv','data','models','results'):
        (code/name).symlink_to(ROOT/name,target_is_directory=True)
    (root/'v4_evidence').symlink_to(ROOT/'results/target_prefill_round4_20261004/v4_evidence',target_is_directory=True)
    command=[str(ROOT/'.venv/bin/python'),str(code/'scripts/run_adaptive_round6.py'),
        '--run-dir',str(root),'--reference-dir',str(args.reference_dir.resolve())]
    if args.prior_aborted_dir:command+=['--prior-aborted-dir',str(args.prior_aborted_dir.resolve())]
    subprocess.run([*command,'--check-only'],cwd=ROOT,check=True)
    launch=dict(unit=args.unit+'.service',run_dir=str(root),command=command,source_root=str(code),
        restart='no',runtime_max_seconds=259200,scope='4 initial diagnostics + 64 CUDA verifier calls; unresolved raw Q/P stops expansion',
        prior_aborted_requests=int(args.prior_aborted_dir is not None),
        natural_request_limit=15 if args.prior_aborted_dir else 16,
        warmups='8 generated tokens per fresh server, excluded from quality/speed; recorded separately in server logs')
    (root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    subprocess.run(['systemd-run','--user','--unit='+args.unit,'--property=Restart=no',
        '--property=SuccessExitStatus=3','--property=RuntimeMaxSec=259200',
        '--working-directory='+str(ROOT),*command],check=True)
    latest=ROOT/'results/adaptive_round6_latest.json';tmp=latest.with_suffix('.tmp')
    tmp.write_text(json.dumps(launch,indent=2)+'\n');tmp.replace(latest)
    if link.is_symlink():link.unlink()
    link.symlink_to(Path('../results')/root.name/'report.md')
    print('进度：bash scripts/status_adaptive_round6.sh')


if __name__=='__main__':main()
