#!/usr/bin/env python3
"""Freeze and launch a bounded train-only experiment without a terminal."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--reference-dir',required=True,type=Path)
    p.add_argument('--unit',default='remtp-adaptive-round5-eos')
    args=p.parse_args()
    root=args.run_dir.resolve()
    if root.parent != ROOT/'results' or root.exists():
        raise ValueError('Use a new directory directly under project/results')
    root.mkdir();code=root/'frozen_code';code.mkdir()
    for name in ('remtp','scripts','configs','tests'):
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('.venv','data','models','results'):
        (code/name).symlink_to(ROOT/name,target_is_directory=True)
    (root/'v4_evidence').symlink_to(ROOT/'results/target_prefill_round4_20261004/v4_evidence',target_is_directory=True)
    command=[str(ROOT/'.venv/bin/python'),str(code/'scripts/run_adaptive_round5.py'),
        '--run-dir',str(root),'--reference-dir',str(args.reference_dir.resolve())]
    subprocess.run([*command,'--check-only'],cwd=ROOT,check=True)
    launch=dict(unit=args.unit+'.service',run_dir=str(root),command=command,source_root=str(code),
        restart='no',runtime_max_seconds=259200,scope='104 new requests maximum, train ablation only')
    (root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    subprocess.run(['systemd-run','--user','--unit='+args.unit,'--property=Restart=no',
        '--property=SuccessExitStatus=3','--property=RuntimeMaxSec=259200',
        '--working-directory='+str(ROOT),*command],check=True)
    latest=ROOT/'results/adaptive_round5_latest.json'
    temp=latest.with_suffix('.tmp');temp.write_text(json.dumps(launch,indent=2)+'\n');temp.replace(latest)
    link=ROOT/'doc/RESULT-ADAPTIVE-RAVEN-ROUND5.md'
    if link.is_symlink():link.unlink()
    elif link.exists():raise ValueError('Refusing to overwrite an existing real report')
    link.symlink_to(Path('../results')/root.name/'report.md')
    print('进度：bash scripts/status_adaptive_round5.sh')


if __name__=='__main__':main()
