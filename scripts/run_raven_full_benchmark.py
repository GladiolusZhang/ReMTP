#!/usr/bin/env python3
"""Frozen, resumable five-method reproduction of the upstream 100+100 suite."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remtp.benchmark import aggregate_records, fetch_metrics, metric_delta
from remtp.gsm8k_benchmark import (
    PROMPT_TEMPLATE as GSM_PROMPT, benchmark_messages, extract_model_answer,
    load_gsm8k, sample_gsm8k,
)
from remtp.humaneval_benchmark import (
    PROMPT_TEMPLATE as HUMAN_PROMPT, load_humaneval, sample_humaneval,
    sanitize_generation,
)
from remtp.humaneval_evaluator import (
    build_candidate_source, docker_available, docker_image_id, evaluate_source_docker,
)
from remtp.optimization_metrics import dominance_gate, clustered_intervals
from scripts.run_adaptive_round2 import Round2, settings
from scripts.run_raven_dominance_search import variants as development_variants
from scripts.run_raven_review_unit import component_hashes, stamp

ORDER = ('native', 'cactus', 'spec_cascade', 'adaptive_original', 'deep4_conservative')
DATASETS = ('gsm8k', 'humaneval')
LABELS = dict(native='Native', cactus='Cactus', spec_cascade='SpecCascade',
              adaptive_original='优化前 Adaptive RAVEN', deep4_conservative='deep4_conservative')
SAMPLE_SEEDS = dict(gsm8k=20260730, humaneval=20260802)
SAMPLES = 100
BASE_SEED = 42
MAX_TOKENS = 2048
RUNTIME_SECONDS = 14 * 86400
INFRASTRUCTURE_STATUSES = {'container_error', 'runner_error'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_record(path, record):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def read_checkpoint(path, expected):
    """Preserve a torn final append; reject corruption or changed completed IDs."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        boundary = raw.rfind(b'\n') + 1
        damaged = path.with_name(path.name + f'.partial.{time.time_ns()}')
        damaged.write_bytes(raw[boundary:])
        with path.open('r+b') as handle:
            handle.truncate(boundary)
            handle.flush()
            os.fsync(handle.fileno())
        raw = raw[:boundary]
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    actual = [(row['sample_id'], row['seed']) for row in records]
    if len(records) > len(expected) or actual != expected[:len(records)]:
        raise ValueError('Checkpoint is not the registered sample/seed prefix: ' + str(path))
    return records


def variants():
    original = {**settings(head=1), 'eos_threshold': .5, 'repeat_guard': False,
                'relax_weights': [1.] * 3, 'lazy_residual': False,
                'probability_ratio_cap': 0., 'risk_policy': 'none'}
    # Keep the chosen candidate byte-for-byte equivalent to its completed screen.
    result = {name: settings(method=name, head=1) for name in ORDER[:3]}
    result['adaptive_original'] = original
    result['deep4_conservative'] = development_variants()['deep4_conservative']
    return result


def sample_id(dataset, row):
    return row['question_id'] if dataset == 'gsm8k' else row['task_id']


def question_number(dataset, row):
    return int(row['question_id']) if dataset == 'gsm8k' else int(row['task_id'].rsplit('/', 1)[1])


def quality_result(result):
    if result.get('status') in INFRASTRUCTURE_STATUSES:
        raise RuntimeError('HumanEval infrastructure failure: ' + json.dumps(result, ensure_ascii=False))
    if result.get('status') not in {'passed', 'failed', 'syntax_error', 'runtime_error', 'timeout'}:
        raise RuntimeError('Unknown HumanEval evaluation status: ' + repr(result))
    return result['status'] == 'passed'


class FullBenchmark(Round2):
    def __init__(self, args):
        # Reuse the tested server environment and cleanup without inheriting train-set gates.
        self.args = args
        self.root = args.run_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.server = None
        self.current = settings(head=1)
        self.diagnostic = False
        self.base_url = f'http://127.0.0.1:{args.port}'
        self.state = {}
        self.work = {}
        self.halt = threading.Event()
        self.thread = None
        self.variants = variants()
        self.reference = args.reference_dir.resolve()
        self.components = component_hashes(ROOT)
        reference_components = component_hashes(self.reference / 'frozen_code')
        if self.components != reference_components:
            raise ValueError('Public runtime/verification/scoring changed since the completed screen')
        if json.loads((self.reference / 'exit.json').read_text())['exit_code'] != 0:
            raise ValueError('Development reference did not finish')
        self.eos_ids = []
        for name in ('config.json', 'generation_config.json'):
            token = json.loads((ROOT / 'models/FastMTP' / name).read_text()).get('eos_token_id', [])
            for value in ([token] if isinstance(token, int) else token or []):
                if value not in self.eos_ids:
                    self.eos_ids.append(value)
        self.paths = dict(gsm8k=self.root / 'frozen_data/gsm8k/test.jsonl',
                          humaneval=self.root / 'frozen_data/humaneval/HumanEval.jsonl.gz')
        self.rows = dict(gsm8k=sample_gsm8k(load_gsm8k(self.paths['gsm8k']), SAMPLES, SAMPLE_SEEDS['gsm8k']),
                         humaneval=sample_humaneval(load_humaneval(self.paths['humaneval']), SAMPLES, SAMPLE_SEEDS['humaneval']))
        self.expected = {dataset: [(sample_id(dataset, row), BASE_SEED + question_number(dataset, row) * 10)
                                   for row in self.rows[dataset]] for dataset in DATASETS}
        existing_protocol = self.root / 'protocol.json'
        pinned = (json.loads(existing_protocol.read_text())['evaluation']['image_id']
                  if existing_protocol.exists() else args.image)
        self.image = docker_image_id(pinned)
        if not docker_available() or not self.image:
            raise RuntimeError('Docker access and the python:3-slim ARM64 image are required')
        copied_files = sorted(path for folder in ('remtp', 'scripts', 'configs')
                              for path in (ROOT / folder).rglob('*')
                              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc')
        code_files = {str(path.relative_to(ROOT)): sha(path) for path in copied_files}
        prompt_checks = self.check_context()
        self.protocol = dict(upstream_commit='3afafa1', upstream_branch='research/residual-aligned-tree-mtp',
            order=list(ORDER), datasets=list(DATASETS), samples_per_dataset=SAMPLES,
            sample_seeds=SAMPLE_SEEDS, generation_seed=BASE_SEED, seed_formula='task_number*10+42',
            temperature=.6, max_tokens=MAX_TOKENS, max_model_len=4096, upstream_max_model_len=8192,
            model='TencentBAC/FastMTP', target='bitsandbytes FP4', draft='BF16 physical MTP head 0 recursive',
            kv_cache='1024M', head_input_rows=1, prefix_caching=0, variants=self.variants,
            main_generation_requests=1000, humaneval_evaluations=500, runtime_max_seconds=RUNTIME_SECONDS,
            observers=False, truncation_gate=False, quality_early_stopping=False, automatic_optimization=False,
            automatic_retry=False, resume='Completed generation and evaluation records are skipped',
            dataset_sha256={name: sha(path) for name, path in self.paths.items()},
            prompt_templates=dict(gsm8k=GSM_PROMPT, humaneval=HUMAN_PROMPT), prompt_checks=prompt_checks,
            model_config_sha256=sha(ROOT / 'models/FastMTP/config.json'),
            no_think_template_sha256=sha(ROOT / 'configs/fastmtp_no_think_chat_template.jinja'),
            packages={name: importlib.metadata.version(name) for name in ('torch', 'vllm', 'bitsandbytes', 'transformers')},
            source_files=code_files, source_sha256=hashlib.sha256(json.dumps(code_files, sort_keys=True).encode()).hexdigest(),
            reference_dir=str(self.reference), public_components=self.components,
            evaluation=dict(image_tag=args.image, image_id=self.image, timeout_seconds=8., memory='512m', cpus='1.0',
                            execution='Original restricted Docker runner and official tests; excluded from generation times'),
            scope='Original fixed 100+100 single-seed reproduction; previously observed GSM8K subset, not a new holdout')
        protocol_path = self.root / 'protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != self.protocol:
            raise ValueError('Frozen protocol changed; refusing to mix experiments')
        atomic_json(protocol_path, self.protocol)
        self.records = {}
        self.evaluations = {}
        for method in ORDER:
            self.records[method] = {}
            for dataset in DATASETS:
                directory = self.root / dataset / method
                directory.mkdir(parents=True, exist_ok=True)
                manifest = self.make_manifest(dataset)
                path = directory / 'sample_manifest.json'
                if path.exists() and json.loads(path.read_text()) != manifest:
                    raise ValueError('Sample manifest changed: ' + str(path))
                atomic_json(path, manifest)
                self.records[method][dataset] = read_checkpoint(directory / 'requests.jsonl', self.expected[dataset])
                for record, row in zip(self.records[method][dataset], self.rows[dataset]):
                    self.validate_record(record, sample_id(dataset, row), BASE_SEED + question_number(dataset, row) * 10)
                    if record['method'] != method or record['task'] != dataset:
                        raise ValueError('Saved generation belongs to another method or dataset')
            self.evaluations[method] = read_checkpoint(self.root / 'humaneval' / method / 'evaluation.jsonl', self.expected['humaneval'])
            if len(self.evaluations[method]) > len(self.records[method]['humaneval']):
                raise ValueError('Evaluation has no matching generation')
        self.status('READY')

    def check_context(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/FastMTP', trust_remote_code=True, local_files_only=True)
        template = (ROOT / 'configs/fastmtp_no_think_chat_template.jinja').read_text()
        result = {}
        for dataset in DATASETS:
            lengths = [len(tokenizer.apply_chat_template(self.messages(dataset, row), tokenize=True,
                add_generation_prompt=True, chat_template=template, enable_thinking=False)) for row in self.rows[dataset]]
            maximum = max(lengths)
            if maximum + MAX_TOKENS + 12 > 4096:
                raise ValueError(f'{dataset}: prompt/output/draft budget exceeds context')
            result[dataset] = dict(max_prompt_tokens=maximum, output_budget=MAX_TOKENS, reserve_tokens=12,
                                   all_fit=True)
        return result

    def make_manifest(self, dataset):
        output = []
        for row in self.rows[dataset]:
            entry = dict(sample_id=sample_id(dataset, row), question_id=question_number(dataset, row),
                         seed=BASE_SEED + question_number(dataset, row) * 10,
                         prompt_sha256=hashlib.sha256(json.dumps(self.messages(dataset, row), sort_keys=True).encode()).hexdigest())
            if dataset == 'gsm8k':
                entry.update(question_sha256=hashlib.sha256(row['question'].encode()).hexdigest(), gold_answer=row['gold_answer'])
            else:
                entry.update(task_id=row['task_id'], test_sha256=hashlib.sha256(row['test'].encode()).hexdigest())
            output.append(entry)
        return output

    @staticmethod
    def messages(dataset, row):
        content = GSM_PROMPT.format(question=row['question']) if dataset == 'gsm8k' else HUMAN_PROMPT.format(prompt=row['prompt'])
        return benchmark_messages(content, empty_system_prompt=True)

    def preflight_evaluator(self):
        """Check all selected official solutions, and a deliberately wrong solution."""
        results = []
        def check(row):
            source = build_candidate_source(row, dict(candidate=row['canonical_solution'], candidate_mode='completion'))
            outcome = evaluate_source_docker(source, image=self.image, timeout=8., memory='512m', cpus='1.0')
            return dict(task_id=row['task_id'], **outcome)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(check, self.rows['humaneval']):
                results.append(result)
                if len(results) % 20 == 0:
                    print(f"HumanEval reference preflight: {len(results)}/100", flush=True)
        wrong = evaluate_source_docker('assert False\n', image=self.image, timeout=8., memory='512m', cpus='1.0')
        passed = all(row['status'] == 'passed' for row in results) and wrong['status'] == 'failed'
        atomic_json(self.root / 'evaluator_preflight.json', dict(passed=passed, image_id=self.image,
            canonical_solutions=results, known_wrong=wrong, checked_utc=stamp()))
        if not passed:
            raise RuntimeError('Official HumanEval reference preflight failed; inspect evaluator_preflight.json')
        print('HumanEval reference preflight passed: 100/100; wrong control rejected', flush=True)

    def status(self, phase, variant='', completed=0, total=0):
        self.state = dict(phase=phase, variant=variant, completed=completed, total=total,
            all_completed=sum(len(rows) for datasets in getattr(self, 'records', {}).values() for rows in datasets.values()),
            all_total=1000, evaluation_completed=sum(map(len, getattr(self, 'evaluations', {}).values())), evaluation_total=500,
            current_work=dict(self.work), updated_utc=stamp(), controller_pid=os.getpid())
        atomic_json(self.root / 'status.json', self.state)
        print(f"[{phase}] {variant} {completed}/{total}; total={self.state['all_completed']}/1000", flush=True)
        self.report()

    def pulse(self):
        while not self.halt.wait(15):
            thermal = {}
            for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
                try:
                    thermal[(path.parent / 'type').read_text().strip()] = int(path.read_text()) / 1000
                except (OSError, ValueError):
                    pass
            server = self.server
            row = dict(updated_utc=stamp(), phase=self.state.get('phase'), work=dict(self.work),
                controller_pid=os.getpid(), server_pid=server.pid if server else None,
                server_alive=server is not None and server.poll() is None,
                meminfo=Path('/proc/meminfo').read_text(), thermal_celsius=thermal)
            atomic_json(self.root / 'heartbeat.json', row)
            append_record(self.root / 'telemetry.jsonl', row)

    def warmup(self, directory):
        self.work = dict(method=self.active_method, task='warmup', started_utc=stamp())
        self.status('WARMUP', self.active_method)
        value = super().warmup(directory)
        atomic_json(directory / f'warmup_attempt_{self.attempt}.json', value)
        return value

    def start_server(self, method, directory, profile=False):
        directory.mkdir(parents=True, exist_ok=True)
        self.work = dict(method=method, task='model_load', started_utc=stamp())
        self.status('LOADING', method)
        if component_hashes(ROOT) != self.components:
            raise ValueError('Frozen public source changed during execution')
        super().start_server(method, directory, profile=False)
        log = (directory / 'server.log').read_text()
        config = self.current
        expected = [f"[ReMTP][FastMTPVerified] method={config['method']} worker=active",
                    '[ReMTP][HeadBatch] input_rows=1',
                    '[ReMTP][TargetNumerics] sm87 FP4: uniform dequant + BF16 linear (all methods)',
                    '[ReMTP][TargetNumerics] unified 2D probability/value product FP32 (all methods)']
        if config['method'] == 'dynamic_tree':
            expected += [f"D={config['depth']} ", f"N_max={config['nodes']} ",
                         f"adaptive_mix={config['mix']:g} ", f"cactus_delta={config['delta']:g} ",
                         'adaptive_relaxation=1 ', 'lazy_residual=0 ', 'risk_policy=none ']
        if any(marker not in log for marker in expected):
            raise RuntimeError('Effective Worker or public numerical configuration missing from startup log')
        env = json.loads((directory / 'effective_env.json').read_text())
        for name in ('REMTP_DECISION_TRACE_DIR', 'REMTP_DECISION_TRACE_SUMMARY_JSONL',
                     'REMTP_TARGET_DIAGNOSTIC_DIR', 'REMTP_RUNTIME_PROFILE_JSONL', 'REMTP_PREFIX_DIAGNOSTIC_JSONL'):
            if env.get(name):
                raise RuntimeError('Unexpected performance observer: ' + name)
        atomic_json(directory / f'startup_check_attempt_{self.attempt}.json', dict(passed=True, expected_markers=expected,
            method=method, effective_env_sha256=sha(directory / 'effective_env.json'), checked_utc=stamp()))

    def generate(self, dataset, method, row):
        directory = self.root / dataset / method
        identifier = sample_id(dataset, row)
        number = question_number(dataset, row)
        seed = BASE_SEED + number * 10
        payload = dict(model='TencentBAC/FastMTP', messages=self.messages(dataset, row),
                       chat_template_kwargs={'enable_thinking': False}, temperature=.6, seed=seed,
                       max_tokens=MAX_TOKENS, return_token_ids=True)
        self.work = dict(method=method, dataset=dataset, sample_id=identifier, seed=seed, started_utc=stamp())
        self.status('GENERATING', method, len(self.records[method][dataset]), SAMPLES)
        # A crash between saving a complete raw response and journal append requires no regeneration.
        for saved in sorted(directory.glob(f'response_{number}_{seed}_attempt_*.json'), reverse=True):
            envelope = json.loads(saved.read_text())
            if envelope.get('payload') == payload and envelope.get('record'):
                record = envelope['record']
                self.validate_record(record, identifier, seed)
                append_record(directory / 'requests.jsonl', record)
                return record
        before = fetch_metrics(self.base_url, 30)
        started = time.perf_counter()
        request = urllib.request.Request(self.base_url + '/v1/chat/completions', json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5400) as response:
            value = json.load(response)
        elapsed = time.perf_counter() - started
        response_path = directory / f'response_{number}_{seed}_attempt_{self.attempt}.json'
        envelope = dict(payload=payload, response=value, client_seconds=elapsed, metrics_before=before)
        atomic_json(response_path, envelope)
        after = fetch_metrics(self.base_url, 30)
        choice = value['choices'][0]
        text = choice['message']['content'] or ''
        record = dict(task=dataset, method=method, sample_id=identifier, question_id=number, seed=seed,
            output=text, output_tokens=int(value['usage']['completion_tokens']), client_seconds=elapsed,
            metrics=metric_delta(before, after), response_id=value['id'], finish_reason=choice['finish_reason'],
            prompt_token_ids=value.get('prompt_token_ids'), output_token_ids=choice.get('token_ids'),
            max_tokens=MAX_TOKENS, observation_enabled=False, performance_sample=True,
            raw_response=str(response_path.relative_to(self.root)), execution_attempt=self.attempt)
        if dataset == 'gsm8k':
            predicted, source = extract_model_answer(text)
            record.update(gold_answer=row['gold_answer'], predicted_answer=predicted, answer_source=source,
                          correct=predicted == row['gold_answer'])
        else:
            candidate, mode = sanitize_generation(text, row['entry_point'])
            record.update(task_id=row['task_id'], candidate=candidate, candidate_mode=mode, correct=None)
        self.validate_record(record, identifier, seed)
        envelope.update(record=record, metrics_after=after)
        atomic_json(response_path, envelope)
        append_record(directory / 'requests.jsonl', record)
        return record

    @staticmethod
    def validate_record(record, identifier, seed):
        if record['sample_id'] != identifier or record['seed'] != seed:
            raise ValueError('Response identity differs from frozen selection')
        if (not record['prompt_token_ids'] or not record['output_token_ids']
                or len(record['output_token_ids']) != record['output_tokens']
                or not 0 < record['output_tokens'] <= MAX_TOKENS
                or not math.isfinite(record['client_seconds']) or record['client_seconds'] <= 0
                or record['finish_reason'] not in {'stop', 'length'}
                or record['max_tokens'] != MAX_TOKENS):
            raise ValueError('Incomplete live IDs, output budget, finish reason, or request timing')

    def evaluate(self, method):
        self.stop_server()  # Docker execution never overlaps measured GPU generation.
        records = self.records[method]['humaneval']
        evaluations = self.evaluations[method]
        for record, task in zip(records[len(evaluations):], self.rows['humaneval'][len(evaluations):]):
            self.work = dict(method=method, dataset='humaneval', task='official_test', sample_id=record['sample_id'], started_utc=stamp())
            self.status('EVALUATING', method, len(evaluations), SAMPLES)
            result = evaluate_source_docker(build_candidate_source(task, record), image=self.image,
                                             timeout=8., memory='512m', cpus='1.0')
            correct = quality_result(result)
            outcome = dict(sample_id=record['sample_id'], task_id=record['task_id'], seed=record['seed'], correct=correct,
                output_sha256=hashlib.sha256(record['output'].encode()).hexdigest(), image_id=self.image, **result)
            append_record(self.root / 'humaneval' / method / 'evaluation.jsonl', outcome)
            evaluations.append(outcome)
            self.write_summaries(method, 'humaneval')
        self.status('EVALUATED', method, len(evaluations), SAMPLES)

    def scored_records(self, method, dataset):
        records = self.records[method][dataset]
        if dataset == 'gsm8k':
            return records
        evaluations = self.evaluations[method]
        output = []
        for index, row in enumerate(records):
            record = dict(row)
            if index < len(evaluations):
                evaluation = evaluations[index]
                if evaluation['output_sha256'] != hashlib.sha256(row['output'].encode()).hexdigest():
                    raise ValueError('HumanEval checkpoint no longer matches generated source')
                record.update(correct=evaluation['correct'], evaluation_status=evaluation['status'])
            output.append(record)
        return output

    def summary(self, method, dataset):
        records = self.scored_records(method, dataset)
        if not records:
            return None
        summary = aggregate_records(records, dataset, SAMPLES)
        known = [row for row in records if row['correct'] is not None]
        summary.update(method=method, correct=sum(row['correct'] for row in known),
            evaluated=len(known), accuracy=sum(row['correct'] for row in known) / len(known) if known else None,
            mean_seconds=sum(row['client_seconds'] for row in records) / len(records),
            truncated_outputs=sum(row['finish_reason'] == 'length' for row in records),
            generation_complete=len(records) == SAMPLES, quality_complete=len(known) == SAMPLES)
        if dataset == 'humaneval':
            summary.update(pass_at_1=summary['accuracy'], evaluation_status='complete' if len(known) == SAMPLES else 'pending_docker_evaluation')
        return summary

    def write_summaries(self, method, dataset):
        summary = self.summary(method, dataset)
        if summary:
            atomic_json(self.root / dataset / method / 'summary.json', dict(config=dict(
                run_name=method, samples=SAMPLES, sample_seed=SAMPLE_SEEDS[dataset], generation_seed=BASE_SEED,
                temperature=.6, max_tokens=MAX_TOKENS, mtp_tokens=self.variants[method]['depth'],
                data_sha256=self.protocol['dataset_sha256'][dataset], evaluation_image_id=self.image), results=[summary]))

    def report(self):
        if not hasattr(self, 'records'):
            return
        lines = ['# RAVEN 五方法：原协议 100+100 正式复现实验', '',
            f"状态：{self.state.get('phase', '准备')}；生成 {self.state.get('all_completed', 0)}/1000；",
            f"HumanEval 判题 {self.state.get('evaluation_completed', 0)}/500。更新时间：{stamp()}。", '',
            'GSM8K test 抽样种子20260730；HumanEval抽样种子20260802；基础生成种子42，逐题为题号×10+42。',
            '温度0.6、输出2048、空system/no-thinking。FastMTP FP4 Target/BF16 MTP，五方法共用修复后的数值实现。',
            '优化前原方法关闭后加EOS保护；deep4_conservative保留筛选时的EOS0.01配置。截断仅作诊断，不提前淘汰方法。', '',
            '未完成组的数值仅表示当前进度；HumanEval判题不计入模型生成时间。']
        rows = []
        for dataset in DATASETS:
            lines += ['', '## ' + ('GSM8K' if dataset == 'gsm8k' else 'HumanEval'), '',
                '| 方法 | 生成 | 判对/已评分 | 正确率/pass@1 | E2E tok/s | 平均整题秒 | MAL | 截断 |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for method in ORDER:
                summary = self.summary(method, dataset)
                if summary:
                    rows.append(summary)
                    quality = f"{summary['accuracy']:.2%}" if summary['accuracy'] is not None else '待判题'
                    mal = f"{summary['mean_acceptance_length']:.4f}" if summary['mean_acceptance_length'] is not None else '—'
                    lines.append(f"| {LABELS[method]} | {summary['requests']}/100 | {summary['correct']}/{summary['evaluated']} | "
                        f"{quality} | {summary['e2e_output_tok_s']:.4f} | {summary['mean_seconds']:.2f} | {mal} | {summary['truncated_outputs']} |")
                else:
                    lines.append(f'| {LABELS[method]} | 0/100 | — | — | — | — | — | — |')
        analysis = self.root / 'analysis.json'
        if analysis.exists():
            lines += ['', '## 完整结果比较', '',
                '| 数据集 | deep4_conservative 对比 | 正确率差/百分点 | 吞吐变化 | 整题耗时变化 | 质量和速度点门槛 |',
                '| --- | --- | ---: | ---: | ---: | --- |']
            comparisons = json.loads(analysis.read_text())['comparisons']
            for dataset, methods in comparisons.items():
                for baseline, comparison in methods.items():
                    point = comparison['point']
                    old, new = point['baseline'], point['candidate']
                    lines.append(f"| {dataset} | {LABELS[baseline]} | {100 * point['accuracy_difference']:+.2f} | "
                        f"{100 * (point['throughput_ratio'] - 1):+.2f}% | "
                        f"{100 * (new['mean_seconds'] / old['mean_seconds'] - 1):+.2f}% | {'通过' if point['passed'] else '未通过'} |")
            lines += ['', '点门槛要求正确率不低于比较方法、E2E吞吐更高且平均整题时间更低；95%配对区间保存在analysis.json。',
                '这是固定100+100、单基础种子的复现；GSM8K100题已被历史实验观察，不能称为新独立留出集。']
        failure = self.root / 'failure.json'
        if failure.exists() and self.state.get('phase') in {'FAILED', 'INTERRUPTED'}:
            lines += ['', '## 停止原因', '', json.loads(failure.read_text())['error']]
        lines += ['', 'MAL=1+总接受草稿token/总验证轮数；同时保存accepted_draft_tokens_per_round。',
            '每题原始payload/response、token IDs、计时、指标与评分检查点均保留。运行不会自动调参或采用候选。']
        temporary = self.root / 'report.md.tmp'
        temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        temporary.replace(self.root / 'report.md')
        atomic_json(self.root / 'summary.json', dict(protocol=str(self.root / 'protocol.json'), results=rows, complete=self.state.get('phase') == 'COMPLETED'))
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            temporary = self.root / 'summary.csv.tmp'
            with temporary.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(self.root / 'summary.csv')

    def run(self):
        attempts = self.root / 'execution_attempts'
        attempts.mkdir(exist_ok=True)
        self.attempt = 1 + max((int(path.stem) for path in attempts.glob('*.json') if path.stem.isdecimal()), default=0)
        if self.attempt > 1 and not self.args.resume:
            raise ValueError('Run already attempted; use the frozen launcher --resume explicitly')
        atomic_json(attempts / f'{self.attempt}.json', dict(started_utc=stamp(), automatic_retry=False))
        preflight = json.loads((self.root / 'evaluator_preflight.json').read_text())
        if not preflight['passed'] or preflight['image_id'] != self.image:
            raise RuntimeError('Pinned evaluator reference preflight is not valid')
        self.thread = threading.Thread(target=self.pulse, daemon=True)
        self.thread.start()
        for method in ORDER:
            self.active_method = method
            self.current = self.variants[method]
            incomplete = any(len(self.records[method][dataset]) < SAMPLES for dataset in DATASETS)
            if incomplete:
                self.start_server(method, self.root / 'servers' / method / f'attempt_{self.attempt}')
                for dataset in DATASETS:
                    records = self.records[method][dataset]
                    for row in self.rows[dataset][len(records):]:
                        records.append(self.generate(dataset, method, row))
                        self.write_summaries(method, dataset)
                        self.status('GENERATING', method, len(records), SAMPLES)
                self.stop_server()
            self.evaluate(method)
        self.work = dict(task='final_analysis', started_utc=stamp())
        self.status('ANALYZING')
        comparisons = {}
        for dataset in DATASETS:
            candidate = self.scored_records('deep4_conservative', dataset)
            comparisons[dataset] = {method: dict(point=dominance_gate(self.scored_records(method, dataset), candidate),
                intervals=clustered_intervals(self.scored_records(method, dataset), candidate)) for method in ORDER[:-1]}
        all_three = all(comparisons[dataset][method]['point']['passed'] for dataset in DATASETS for method in ORDER[:3])
        atomic_json(self.root / 'analysis.json', dict(comparisons=comparisons, exceeds_all_three_point=all_three,
            automatic_adoption=False, scope=self.protocol['scope'], analyzed_utc=stamp()))
        self.work = {}
        self.status('COMPLETED')
        return 0

    def close(self):
        self.stop_server()
        self.halt.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.lock.close()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--image', default='python:3-slim')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--verify-evaluator', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.prefix_cache = 0
    args.uniform_gemm = True
    args.full_bf16_accum = True
    args.bnb_row_block = 64
    args.rmsnorm_row_limit = 255
    args.lm_head_row_block = 16
    args.target_backend = 'TRITON_ATTN'
    args.triton_2d = True
    args.attention_fp32 = True
    args.canonical_tree = True
    return args


def main():
    args = arguments()
    controller = None
    code = 1
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        controller = FullBenchmark(args)
        if args.verify_evaluator:
            controller.preflight_evaluator()
        code = 0 if args.check_only else controller.run()
        return code
    except BaseException as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        if controller:
            atomic_json(controller.root / 'failure.json', dict(error=str(error), type=type(error).__name__,
                traceback=traceback.format_exc(), work=controller.work, updated_utc=stamp()))
            controller.status('INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED')
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        return code
    finally:
        if controller:
            controller.close()
            if not args.check_only:
                result = dict(exit_code=code, phase=controller.state.get('phase'), ended_utc=stamp(), server_cleaned=True)
                atomic_json(controller.root / 'exit.json', result)
                if hasattr(controller, 'attempt'):
                    atomic_json(controller.root / 'execution_attempts' / f'{controller.attempt}_exit.json', result)


if __name__ == '__main__':
    raise SystemExit(main())
