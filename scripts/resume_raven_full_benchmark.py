#!/usr/bin/env python3
"""Resume the immutable 100+100 generation with a registered Docker OOM classification fix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit_existing(run):
    protocol = json.loads((run / 'protocol.json').read_text())
    counts, assets = [], {}
    for method in protocol['order']:
        entry = dict(method=method, generated={}, evaluated=0)
        for dataset in protocol['datasets']:
            directory = run / dataset / method
            manifest = json.loads((directory / 'sample_manifest.json').read_text())
            path = directory / 'requests.jsonl'
            raw = path.read_bytes() if path.exists() else b''
            complete = raw[:raw.rfind(b'\n') + 1] if raw and not raw.endswith(b'\n') else raw
            records = [json.loads(line) for line in complete.splitlines() if line.strip()]
            expected = [(row['sample_id'], row['seed']) for row in manifest]
            if [(row['sample_id'], row['seed']) for row in records] != expected[:len(records)]:
                raise ValueError('Saved generation order/seed mismatch: ' + str(path))
            entry['generated'][dataset] = len(records)
            if path.exists():
                assets[str(path.relative_to(run))] = dict(sha256=sha(path), prefix_bytes=len(complete))
            for record in records:
                if len(record['output_token_ids']) != record['output_tokens']:
                    raise ValueError('Incomplete saved token IDs')
                response = run / record['raw_response']
                if json.loads(response.read_text())['record'] != record:
                    raise ValueError('Raw response and checkpoint disagree')
                assets[str(response.relative_to(run))] = dict(sha256=sha(response), prefix_bytes=response.stat().st_size)
            if dataset == 'humaneval':
                path = directory / 'evaluation.jsonl'
                raw = path.read_bytes() if path.exists() else b''
                complete = raw[:raw.rfind(b'\n') + 1] if raw and not raw.endswith(b'\n') else raw
                evaluations = [json.loads(line) for line in complete.splitlines() if line.strip()]
                if [(row['sample_id'], row['seed']) for row in evaluations] != expected[:len(evaluations)] or len(evaluations) > len(records):
                    raise ValueError('Saved official evaluation identity mismatch')
                for result, record in zip(evaluations, records):
                    if result['output_sha256'] != hashlib.sha256(record['output'].encode()).hexdigest():
                        raise ValueError('Evaluation source hash mismatch')
                entry['evaluated'] = len(evaluations)
                if path.exists():
                    assets[str(path.relative_to(run))] = dict(sha256=sha(path), prefix_bytes=len(complete))
        counts.append(entry)
    for name, expected in protocol['source_files'].items():
        if sha(run / 'frozen_code' / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    for dataset, expected in protocol['dataset_sha256'].items():
        filename = 'gsm8k/test.jsonl' if dataset == 'gsm8k' else 'humaneval/HumanEval.jsonl.gz'
        if sha(run / 'frozen_data' / filename) != expected:
            raise ValueError('Frozen dataset changed')
    return dict(counts=counts, generation_completed=sum(sum(row['generated'].values()) for row in counts),
                evaluation_completed=sum(row['evaluated'] for row in counts), preserved_assets=assets,
                frozen_source_sha256=protocol['source_sha256'])


def worker(args):
    run = args.run_dir.resolve()
    recovery = args.recovery_dir.resolve()
    registration = json.loads((recovery / 'addendum.json').read_text())
    for name, expected in registration['adapter_source_files'].items():
        if sha(recovery / 'frozen_scripts' / name) != expected:
            raise ValueError('Registered recovery adapter changed')
    if sha(run / 'protocol.json') != registration['original_protocol_sha256']:
        raise ValueError('Original experiment protocol changed')
    sys.path.insert(0, str(run / 'frozen_code'))
    original = load_module('_raven_original_full', run / 'frozen_code/scripts/run_raven_full_benchmark.py')
    adapter = load_module('_raven_humaneval_recovery', Path(__file__).with_name('humaneval_container_recovery.py'))
    import remtp.humaneval_evaluator as evaluator
    def execute(source, **kwargs):
        return adapter.evaluate_with_state(source, command_factory=evaluator._docker_command,
            runner_path=Path(evaluator.__file__).with_name('humaneval_container_runner.py'),
            evidence_dir=recovery / 'container_evidence', **kwargs)
    original.evaluate_source_docker = execute
    manifest = json.loads((run / 'launch.json').read_text())
    sys.argv = [*manifest['command'][1:], '--resume']
    if args.check_only:
        sys.argv.append('--check-only')
    return original.main()


def launch(args):
    run = args.run_dir.resolve()
    project = run.parent.parent
    manifest = json.loads((run / 'launch.json').read_text())
    unit = manifest['unit']
    if run.parent != project / 'results':
        raise ValueError('Use the registered project results directory')
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0:
        raise ValueError('The experiment is already active')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 8000)) == 0:
            raise ValueError('Port 8000 occupied')
    if subprocess.check_output(['loginctl', 'show-user', str(os.getuid()), '-p', 'Linger'], text=True).strip() != 'Linger=yes':
        raise ValueError('Detached execution requires Linger=yes')
    subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], check=True, stdout=subprocess.DEVNULL)
    preflight = json.loads((run / 'recovery_preflight_20261012/result.json').read_text())
    if (not preflight['passed'] or preflight['adapter_sha256'] != sha(Path(__file__).with_name('humaneval_container_recovery.py'))):
        raise ValueError('Recovery evaluator requires the matching real Docker preflight')
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    recovery = run / ('recovery_' + timestamp)
    recovery.mkdir()
    backup = recovery / 'original_state'
    backup.mkdir()
    for name in ('status.json', 'heartbeat.json', 'failure.json', 'exit.json', 'report.md', 'summary.json',
                 'summary.csv', 'protocol.json', 'launch.json'):
        if (run / name).is_file():
            shutil.copy2(run / name, backup / name)
    audit = audit_existing(run)
    write_json(recovery / 'checkpoint_audit.json', audit)
    sources = recovery / 'frozen_scripts'
    sources.mkdir()
    for name in ('resume_raven_full_benchmark.py', 'humaneval_container_recovery.py'):
        shutil.copy2(Path(__file__).with_name(name), sources / name)
    policy = json.loads((run / 'protocol.json').read_text())
    registration = dict(original_source_sha256=policy['source_sha256'], original_protocol_sha256=sha(run / 'protocol.json'),
        adapter_source_files={name: sha(sources / name) for name in ('resume_raven_full_benchmark.py', 'humaneval_container_recovery.py')},
        change='Retain Docker State until inspection; confirmed OOMKilled=true/137 is candidate runtime_error, all other failures retain original classification',
        unchanged=dict(model=True, generation_code=True, checkpoint_order=True, dataset=True, prompt=True, seed=True,
                       timeout_seconds=8, memory='512m', cpus='1.0', runner=True, image_id=policy['evaluation']['image_id']),
        before_generation=audit['generation_completed'], before_evaluation=audit['evaluation_completed'],
        preflight=str(run / 'recovery_preflight_20261012/result.json'),
        authorization='User requested resume; necessary evaluator classification repair', created_utc=datetime.now(timezone.utc).isoformat())
    write_json(recovery / 'addendum.json', registration)
    command = [manifest['command'][0], str(sources / 'resume_raven_full_benchmark.py'),
               '--run-dir', str(run), '--recovery-dir', str(recovery), '--worker']
    subprocess.run([*command, '--check-only'], cwd=project, check=True)
    subprocess.run(['systemctl', '--user', 'reset-failed', unit], check=False, capture_output=True)
    subprocess.run(['systemd-run', '--user', '--unit=' + unit.removesuffix('.service'), '--property=Restart=no',
        '--property=RuntimeMaxSec=' + str(manifest['runtime_max_seconds']), '--property=TimeoutStopSec=90',
        '--property=MemoryAccounting=yes', '--property=OOMPolicy=stop', '--working-directory=' + str(project),
        *command], check=True)
    resumed = dict(manifest, command=command, original_command=manifest['command'], recovery_dir=str(recovery),
                   evaluator_addendum=str(recovery / 'addendum.json'), resume_started=datetime.now(timezone.utc).isoformat())
    write_json(recovery / 'launch.json', resumed)
    write_json(project / 'results/raven_full_latest.json', resumed)
    print(f"已续跑：保留{audit['generation_completed']}/1000生成、{audit['evaluation_completed']}/500判题；修复登记：{recovery}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--recovery-dir', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    return worker(args) if args.worker else launch(args)


if __name__ == '__main__':
    raise SystemExit(main())
