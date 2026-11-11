#!/usr/bin/env python3
"""Freeze the 100+100 reproduction, check it, and launch outside the conversation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, default=ROOT / 'results/raven_dominance_search_20261009')
    parser.add_argument('--unit', default='remtp-raven-full-100x100-20261010')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != ROOT / 'results':
        raise ValueError('Use a project/results directory')
    if not args.resume and run.exists():
        raise ValueError('Use a new result directory or explicitly resume its frozen code')
    if args.resume and not (run / 'launch.json').is_file():
        raise ValueError('No registered frozen run to resume')
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', args.unit]).returncode == 0:
        raise ValueError('Experiment unit is already active')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 8000)) == 0:
            raise ValueError('Port 8000 is occupied; unrelated services will not be stopped')
    linger = subprocess.run(['loginctl', 'show-user', str(os.getuid()), '-p', 'Linger'],
                            capture_output=True, text=True, check=True).stdout.strip()
    if linger != 'Linger=yes':
        raise ValueError('Detached execution requires user lingering')
    link = ROOT / 'doc/RESULT-RAVEN-100X100-LIVE.md'
    if link.exists() and not link.is_symlink():
        raise ValueError('Refusing to replace a real document')
    if not args.resume:
        run.mkdir()
        code = run / 'frozen_code'
        code.mkdir()
        for name in ('remtp', 'scripts', 'configs', 'tests'):
            shutil.copytree(ROOT / name, code / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('.venv', 'models', 'results'):
            (code / name).symlink_to(ROOT / name, target_is_directory=True)
        for name in ('README.md', 'AGENTS.md'):
            shutil.copy2(ROOT / name, code / name)
        for name in ('gsm8k/test.jsonl', 'humaneval/HumanEval.jsonl.gz'):
            target = run / 'frozen_data' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'data' / name, target)
        (run / 'plan.md').write_bytes((ROOT / 'plan/raven-full-100x100-2026-10-10.md').read_bytes())
        command = [str(ROOT / '.venv/bin/python'), str(code / 'scripts/run_raven_full_benchmark.py'),
                   '--run-dir', str(run), '--reference-dir', str(args.reference_dir.resolve())]
        subprocess.run([*command, '--check-only', '--verify-evaluator'], cwd=ROOT, check=True)
        policy = json.loads((run / 'protocol.json').read_text())
        manifest = dict(unit=args.unit + '.service', run_dir=str(run), command=command,
            generation_requests=1000, evaluation_requests=500, resume=True, restart='no',
            runtime_max_seconds=policy['runtime_max_seconds'], source_sha256=policy['source_sha256'])
        (run / 'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = json.loads((run / 'launch.json').read_text())
        if manifest['unit'] != args.unit + '.service':
            raise ValueError('Resume must use the registered unit')
        command = manifest['command']
        subprocess.run([*command, '--check-only'], cwd=ROOT, check=True)
        command = [*command, '--resume']
        subprocess.run(['systemctl', '--user', 'reset-failed', args.unit], check=False)
    subprocess.run(['systemd-run', '--user', '--unit=' + args.unit, '--property=Restart=no',
        '--property=RuntimeMaxSec=' + str(manifest['runtime_max_seconds']), '--property=TimeoutStopSec=90',
        '--property=MemoryAccounting=yes', '--property=OOMPolicy=stop',
        '--working-directory=' + str(ROOT), *command], check=True)
    (ROOT / 'results/raven_full_latest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if link.is_symlink():
        link.unlink()
    link.symlink_to(Path('../results') / run.name / 'report.md')
    print('后台已启动；进度：bash scripts/status_raven_full_benchmark.sh')


if __name__ == '__main__':
    main()
