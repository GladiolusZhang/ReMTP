"""Preserve Docker exit evidence and classify candidate OOM without changing limits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid


def classify_exit(returncode, stdout, stderr, state, *, timed_out, timeout, memory):
    # SIGKILL/137 alone is insufficient evidence of a candidate memory failure.
    if timed_out:
        return {'status': 'timeout', 'detail': f'> {timeout:g}s'}
    if state and state.get('OOMKilled') is True and state.get('ExitCode') == 137 and returncode == 137:
        return {'status': 'runtime_error',
                'detail': f'Candidate exceeded container memory limit {memory}; Docker OOMKilled=true, exit=137',
                'oom_killed': True, 'container_exit_code': 137}
    lines = [line for line in stdout.splitlines() if line.strip()]
    if returncode != 0 or not lines:
        detail = (stderr or stdout or f'exit code {returncode}').strip()
        return {'status': 'container_error', 'detail': detail[-1000:]}
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {'status': 'container_error', 'detail': (stderr + '\n' + stdout)[-1000:].strip()}
    return {'status': str(result.get('status', 'runner_error')), 'detail': str(result.get('detail', ''))[:1000]}


def evaluate_with_state(source, *, image, timeout, memory, cpus, command_factory, runner_path, evidence_dir):
    """Use the frozen command/runner, retain state until inspect, then remove only our container."""
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    name = 'remtp-humaneval-recovery-' + identifier[:16]
    state = None
    timed_out = False
    with tempfile.TemporaryDirectory(prefix='remtp-humaneval-') as directory:
        job = Path(directory) / 'job.json'
        job.write_text(json.dumps({'source': source}, ensure_ascii=False), encoding='utf-8')
        command = command_factory(name=name, image=image, runner_path=Path(runner_path),
                                  job_path=job, memory=memory, cpus=cpus)
        command.remove('--rm')  # Inspect immediately after exit, before explicit removal.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(['docker', 'kill', name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False, timeout=15)
                stdout, stderr = process.communicate(timeout=15)
            inspection = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name],
                                        capture_output=True, text=True, check=False, timeout=15)
            if inspection.returncode == 0:
                state = json.loads(inspection.stdout)
            result = classify_exit(process.returncode, stdout, stderr, state, timed_out=timed_out,
                                   timeout=timeout, memory=memory)
            evidence = dict(container_name=name, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                image=image, timeout=timeout, memory=memory, cpus=cpus, returncode=process.returncode,
                timed_out=timed_out, state=state, result=result,
                inspection_error=inspection.stderr[-1000:] if inspection.returncode else '')
            evidence_path = evidence_dir / (identifier + '.json')
            with evidence_path.open('x', encoding='utf-8') as handle:
                json.dump(evidence, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
                handle.flush()
                import os
                os.fsync(handle.fileno())
            return dict(result, container_evidence=str(evidence_path))
        finally:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, timeout=15)

