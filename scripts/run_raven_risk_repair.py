#!/usr/bin/env python3
"""One fixed Target-support repair, with fresh CUDA and natural Target gates."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_adaptive_round2 import Round2, settings, write_json
from scripts.run_adaptive_round4 import NUMERICAL_KEYS
from scripts.run_adaptive_round5 import output_signature
from scripts.run_adaptive_round6 import configs, equal_trace
from scripts.run_raven_review_unit import (Ledger, NumericRejected, QualityRejected,
    BudgetStop, COMPONENTS, CALIBRATION_IDS, component_hashes, stamp, failure_kind)
from scripts.run_raven_root_cause_unit import RootCauseUnit, ReplayRejected, EXECUTION_PAIRS
from scripts.run_raven_probability_cap import load_legacy
from scripts.analyze_raven_root_causes import PAIRS, CHECKPOINTS, requests
from remtp.optimization_metrics import summarize, paired_gate, clustered_intervals

LIMITS = dict(main_http=86, numeric=12, cuda_verify=48, warmup=7, starts=7)
RUNTIME_SECONDS = 86400
CALIBRATION_ORDER = ('native', 'cactus', 'spec_cascade', 'B', 'target_odds')
ALLOWED_CHANGES = frozenset(('remtp/dynamic_mtp_tree.py', 'remtp/dynamic_tree_vllm.py',
                              'scripts/run_adaptive_round2.py'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered_source_changes(current, previous):
    changes = {}
    for group, names in COMPONENTS.items():
        for name in names:
            old, new = digest(previous/name), digest(current/name)
            if old != new:
                if name not in ALLOWED_CHANGES:
                    raise ValueError('Unregistered runtime/evaluation change: '+name)
                changes[name] = dict(previous=old, current=new, group=group)
    return changes


def completed_root_evidence(reference):
    """Reject partial, failed or relabeled root-cause evidence."""
    def read(name):
        return json.loads((reference/name).read_text())
    end, status = read('exit.json'), read('status.json')
    replay, target, natural = (read(n) for n in
        ('B_reproduction_gate.json', 'target_natural_prefix_gate.json', 'natural_forensics.json'))
    if (end['exit_code'] != 0 or status['phase'] != 'COMPLETED_ROOT_CAUSE_UNIT'
            or (reference/'failure.json').exists() or not replay['passed']
            or replay['requests'] != 6 or not target['passed'] or not target['complete']
            or len(target['checks']) != 12 or not natural['passed'] or not natural['complete']):
        raise NumericRejected('Dependency is not a complete passing root-cause unit')
    data = requests(reference/'natural/B/requests.jsonl')
    if {(r['question_id'], r['seed']-r['question_id']*10) for r in data} != set(PAIRS) or len(data) != 6:
        raise NumericRejected('Root natural coverage differs from registered pairs')
    if not all(r['signature_equal'] for r in natural['requests']):
        raise ReplayRejected('Root B observation replay failed')
    return data, natural


def select_cuda_frames(reference, natural):
    """Twelve first/last frames plus four earliest remaining frames; no gold selection."""
    cases = json.loads((reference/'numeric_cases.json').read_text())['cases']
    selected = []
    by_path = {c['frame']: c for r in natural['requests'] for c in r['coupled_frames']}
    for c in cases:
        if c['frame'] not in selected:
            selected.append(c['frame'])
    for r in natural['requests']:
        for c in r['coupled_frames']:
            if len(selected) == 16:
                break
            if c['frame'] not in selected:
                selected.append(c['frame'])
    if len(selected) != 16 or len(cases) != 12:
        raise NumericRejected('Exactly sixteen registered distinct real frames required')
    result = []
    for path in selected:
        c = by_path[path]
        if digest(path) != c['frame_sha256'] or digest(c['raw_frame']) != c['raw_frame_sha256']:
            raise NumericRejected('Root frame artifact changed: '+path)
        result.append(dict(path=path, sha256=c['frame_sha256'],
                           prefix_sha256=c['prefix_sha256']))
    return result


def check_risk_semantics(result, trace, before, after):
    """Inspect the consumed H, residual and RNG event chain, not only alpha."""
    import torch
    from remtp.fixed6_microtree import residual_distribution
    details = {d['node']: d for d in result.node_diagnostics if 'node' in d}
    pending = None
    state = before
    modes = dict(strict=0, no_reallocation=0, none=0)
    for event in trace['events']:
        if not torch.equal(state, event['rng_before']):
            raise NumericRejected('Random consumption outside recorded verifier events')
        state = event['rng_after']
        if event['kind'] == 'accept_uniform':
            p, q, h = (event[k] for k in ('p', 'q', 'h'))
            y = event['token_id']
            detail = details[event['node']]
            if not bool(torch.isfinite(h).all()) or not bool((h >= 0).all()) or abs(float(h.sum())-1) > 2e-6:
                raise NumericRejected('Candidate H is not a finite probability distribution')
            mode = detail.get('adaptive_risk_mode', 'none')
            modes[mode] += 1
            if (mode == 'strict' or detail.get('eos_or_prefix_protected', False)) and not torch.equal(h, p):
                raise NumericRejected('Protected candidate did not use exactly P')
            if mode == 'no_reallocation':
                tv = float((h-p).abs().sum()/2)
                if tv > detail['baseline_relaxation_tv']+2e-6:
                    raise NumericRejected('Risk position receives extra relaxation')
            if detail.get('adaptive_relaxation_enabled') and (
                    detail['adaptive_block_tv'] > detail['adaptive_block_baseline_tv']+2e-6):
                raise NumericRejected('Candidate exceeds registered block allowance')
            alpha = min(1., float(h[y])/max(float(q[y]), 1e-30))
            if event['keep'] != (event['uniform'] <= alpha):
                raise NumericRejected('Acceptance does not use the recorded H/Q')
            pending = None if event['keep'] else residual_distribution(h, q)
        elif event['kind'] in ('residual_correction', 'continuation_correction'):
            if pending is None or not torch.allclose(pending, event['probs'], rtol=3e-6, atol=1e-8):
                raise NumericRejected('Correction does not use the same full H')
            pending = None
    if pending is not None or not torch.equal(state, after):
        raise NumericRejected('Missing correction or unrecorded final RNG consumption')
    return modes


def cuda_gate(run, reference):
    """Exactly 16 x 3 CUDA verifier calls in a process that exits before serving."""
    import torch
    from remtp.dynamic_mtp_tree import DynamicTreeTopology, DynamicTreeConfig, residual_hit_tree_verify
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise NumericRejected('CUDA unavailable')
    frames = json.loads((run/'cuda_frames.json').read_text())['frames']
    if len(frames) != 16:
        raise NumericRejected('Registered CUDA frame count differs')
    ledger = Ledger(run/'ledger.json', LIMITS)
    legacy = load_legacy(reference)
    checks = []
    for item in frames:
        path = Path(item['path'])
        if digest(path) != item['sha256']:
            raise NumericRejected('Registered CUDA frame changed')
        f = torch.load(path, map_location='cpu', weights_only=False)
        if f['rng_before'] is None or not f['rng_device'].startswith('cuda'):
            raise NumericRejected('Original CUDA generator missing')
        config_values = asdict(f['config'])
        off = DynamicTreeConfig(**config_values)
        old = legacy.DynamicTreeConfig(**{k: v for k, v in config_values.items()
                                          if k in {a.name for a in fields(legacy.DynamicTreeConfig)}})
        new = replace(off, relaxation_risk_policy='target_odds')
        values = [f[k].cuda() for k in ('ids', 'draft', 'target')]
        results, traces, states = [], [], []
        semantics = {}
        for name, config, module in (('legacy', old, legacy),
                ('disabled', off, sys.modules['remtp.dynamic_mtp_tree']),
                ('target_odds', new, sys.modules['remtp.dynamic_mtp_tree'])):
            key = item['sha256']+'/'+name
            ledger.reserve('cuda_verify', key, frame=str(path), mode=name)
            rng = torch.Generator(device=f['rng_device']); rng.set_state(f['rng_before'])
            trace = {}
            result = module.residual_hit_tree_verify(
                module.DynamicTreeTopology.from_paths(f['topology']), *values, rng,
                config=config, protected_node_indices=f['protected_node_indices'],
                lazy_residual=False, decision_trace=trace)
            results.append(asdict(result)); traces.append(trace); states.append(rng.get_state())
            if name == 'target_odds':
                semantics = check_risk_semantics(result, trace, f['rng_before'], rng.get_state())
            ledger.complete('cuda_verify', key)
        same = (results[0] == results[1] == f['result']
            and torch.equal(states[0], states[1]) and torch.equal(states[1], f['rng_after'])
            and len(traces[0]['events']) == len(traces[1]['events']) == len(f['events'])
            and all(equal_trace(a, b) and equal_trace(b, c) for a, b, c in
                    zip(traces[0]['events'], traces[1]['events'], f['events'])))
        checks.append(dict(**item, disabled_bitwise_equal=bool(same), candidate_semantics=True,
                           visited_risk_modes=semantics))
        write_json(run/'cuda_gate.json', dict(passed=all(c['disabled_bitwise_equal'] for c in checks),
            complete=len(checks)==16, checks=checks, verifier_calls=3*len(checks),
            scope='16 real recorded CUDA frames; legacy/off P/Q/H/result/RNG equality and '
                  'candidate full H/residual/RNG chain; not full Target/KV or quality proof'))
        if not same:
            raise ReplayRejected('Disabled candidate differs from legacy or saved actual frame')


class RiskRepairUnit(RootCauseUnit):
    def __init__(self, args):
        # Preserve ReviewUnit -> Round2 startup; RootCauseUnit's math-identity
        # constructor is deliberately replaced by an explicit repair whitelist.
        Round2.__init__(self, args)
        self.reference = args.reference_dir.resolve()
        self.review_reference = args.review_reference_dir.resolve()
        self.protocol = json.loads((self.root/'protocol.json').read_text())
        old = json.loads((self.reference/'protocol.json').read_text())
        for key in (*NUMERICAL_KEYS, 'train_sha256', 'test_sha256', 'primary_prompt'):
            if self.protocol[key] != old[key]:
                raise ValueError('Fixed protocol mismatch: '+key)
        changes = registered_source_changes(ROOT, self.reference/'frozen_code')
        write_json(self.root/'component_hashes.json', dict(current=component_hashes(ROOT),
            allowed_changes=sorted(ALLOWED_CHANGES), actual_changes=changes,
            scope='Explicit repair source changes; all Target math, proposal/cache math, '
                  'serving, original scorer and statistical calculations otherwise identical'))
        self.historical = {(r['question_id'], r['seed']): r for r in
            requests(self.review_reference/'calibration/B/requests.jsonl')}
        if len(self.historical) != 16:
            raise ValueError('Complete original B16 required')
        baseline = {**configs()['old_eos01'], 'probability_ratio_cap': 0., 'risk_policy': 'none'}
        self.variants = dict(B=baseline, target_odds={**baseline, 'risk_policy': 'target_odds'},
            target=settings(method='target', head=1), native=settings(method='native', head=1),
            cactus=settings(method='cactus', head=1),
            spec_cascade=settings(method='spec_cascade', head=1))
        self.ledger = Ledger(self.root/'ledger.json', LIMITS)
        self.state = {}; self.work = {}; self.forensics = []
        self.natural = []; self.cases = []
        self.halt = threading.Event(); self.heartbeat = None
        self.started = time.monotonic(); self.runtime_limit = RUNTIME_SECONDS
        self.request_timeout = 5400
        policy = dict(limits=LIMITS, runtime_max_seconds=RUNTIME_SECONDS, retries=0, restart='no',
            candidate='target_odds', max_candidates=1, heldout=False, formal=False,
            automatic_adoption=False, dependence=str(self.reference), review_reference=str(self.review_reference),
            thresholds=dict(strict_target_relative_below=.001, no_extra_relaxation_below=.1),
            cuda_selection='12 first/last recorded natural frames plus 4 earliest remaining distinct frames',
            cuda_verifier_calls=48, natural_pairs=PAIRS, execution_order=EXECUTION_PAIRS,
            natural_checkpoints=CHECKPOINTS, early_proposals_per_request=3,
            full_verifier_frames_per_request=8, natural_target_frames_per_request=8,
            summary_max_rounds_per_request=2048, natural_disk_max_bytes=2*1024**3,
            fresh_target_gate='12 actual natural cases, <=12 unique fresh same-ID sequential-prefill calls; '
                'raw bitwise equality, no additional shape projections, duplicate prefixes explicitly shared',
            calibration_ids=CALIBRATION_IDS, calibration_seeds=(42, 43),
            calibration_order=CALIBRATION_ORDER, quality_tokens=2048,
            smoke_quality='same six B cases: correct/stopped_correct nondecrease and truncation nonincrease '
                'in aggregate/each seed; both benefit controls remain correct and stop',
            development_gate='aggregate/each seed quality and truncation nonregression versus all three '
                'baselines and new B; fewer truncations than B, higher throughput AND lower full-answer '
                'mean latency versus all; question-cluster intervals reported, no holdout/test claim',
            observation_gate='candidate six observed outputs equal six unobserved actual-ID outputs; '
                'new B16 signatures match historical and B6 actual IDs match root unit',
            calibration_observers=False, generation_http_timeout_seconds=5400,
            source_sha256=self.protocol['source_sha256'], variants=self.variants,
            allowed_changes=sorted(ALLOWED_CHANGES))
        path = self.root/'repair_policy.json'
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(policy)):
            raise ValueError('Repair policy changed; use a new directory')
        write_json(path, policy)
        self.ensure_budget([self.by_id[q] for q in CALIBRATION_IDS])

    def start_server(self, variant, directory, profile=False):
        super().start_server(variant, directory, profile)
        if self.current['method'] == 'dynamic_tree':
            expected = 'risk_policy='+self.current.get('risk_policy', 'none')+' '
            if expected not in (directory/'server.log').read_text():
                raise NumericRejected('Registered risk policy not installed: '+expected)
        if not self.args.target_probe and not self.args.decision_trace_only:
            env = json.loads((directory/'effective_env.json').read_text())
            for key in ('REMTP_PREFIX_DIAGNOSTIC_JSONL', 'REMTP_DECISION_TRACE_DIR',
                    'REMTP_DECISION_TRACE_SUMMARY_JSONL', 'REMTP_TARGET_DIAGNOSTIC_DIR',
                    'REMTP_RUNTIME_PROFILE_JSONL', 'REMTP_STATE_DIAGNOSTIC_DIR'):
                if env.get(key):
                    raise NumericRejected('Unobserved calibration has a probe: '+key)

    def generate(self, phase, name, row, seed, limit):
        record = super().generate(phase, name, row, seed, limit)
        records = requests(self.root/phase/name/'requests.jsonl')
        write_json(self.root/phase/name/'summary.json', summarize(records))
        self.status(phase, name, len(records), 16 if phase=='calibration' else 6)
        return record

    def configure_observers(self, enabled):
        self.diagnostic = False
        self.args.decision_trace = enabled; self.args.decision_trace_only = enabled
        self.args.decision_trace_checkpoints = CHECKPOINTS if enabled else ()
        self.args.decision_trace_early_proposals = 3 if enabled else 0
        self.args.target_probe = enabled; self.args.target_probe_min_generated = 1 if enabled else 0
        self.args.target_probe_min_prompt_tokens = 65 if enabled else 0
        self.args.target_probe_checkpoints = CHECKPOINTS if enabled else ()
        self.args.target_probe_max_frames = 8 if enabled else 1
        self.args.target_probe_prefix_manifest = ''

    def gate(self, name, value):
        write_json(self.root/(name+'_gate.json'), value)
        self.report()
        if not value['passed']:
            raise QualityRejected(name)

    def check_smoke(self, reference):
        value = paired_gate(reference, self.natural, require_speed=False)
        controls = {(1231, 43), (5180, 43)}
        value['benefit_controls'] = [dict(question_id=r['question_id'], seed=r['seed']-r['question_id']*10,
            passed=bool(r['correct'] and r['finish_reason']=='stop')) for r in self.natural
            if (r['question_id'], r['seed']-r['question_id']*10) in controls]
        value['passed'] = value['passed'] and len(value['benefit_controls'])==2 and all(
            c['passed'] for c in value['benefit_controls'])
        value['scope'] = '6 observed development cases; diagnostic timing excluded from speed'
        self.gate('smoke_quality', value)

    def replay_gate(self, name, current, reference, require_ids=False):
        keyed = {(r['question_id'], r['seed']): r for r in current}
        checks = []
        for old in reference:
            key = old['question_id'], old['seed']
            new = keyed.get(key)
            same = new is not None and output_signature(new)==output_signature(old)
            if require_ids:
                same = same and new['prompt_token_ids']==old['prompt_token_ids'] and (
                                new['output_token_ids']==old['output_token_ids'])
            checks.append(dict(question_id=key[0], seed=key[1], passed=bool(same)))
        value = dict(passed=all(c['passed'] for c in checks), checks=checks, actual_id_equality=require_ids)
        write_json(self.root/(name+'_gate.json'), value)
        if not value['passed']:
            raise ReplayRejected(name)

    def make_analysis(self, datasets):
        candidate = datasets['target_odds']
        comparisons = {name: dict(quality=paired_gate(data, candidate, require_speed=False),
            quality_speed=paired_gate(data, candidate, require_speed=True),
            intervals=clustered_intervals(data, candidate))
            for name, data in datasets.items() if name!='target_odds'}
        fewer_truncations = summarize(candidate)['truncated'] < summarize(datasets['B'])['truncated']
        point_passed = fewer_truncations and all(c['quality_speed']['passed'] for c in comparisons.values())
        robust_speed = all(c['intervals']['throughput_ratio_95ci'][0]>1 for c in comparisons.values())
        value = dict(summaries={n:summarize(data) for n,data in datasets.items()},
            comparisons=comparisons, fewer_truncations_than_B=fewer_truncations,
            development_point_gate_passed=point_passed, throughput_interval_gate_passed=robust_speed,
            formal_goal_achieved=False, automatically_adopted=False,
            scope='Observed 8 hard training questions x 2 seeds; 8 question clusters, no holdout/test. '
                  'Same model/numerics/prompt/scorer/output budget, all calibration probes off. '
                  'Quality and speed evaluated separately; failed quality is not a runtime fault.')
        write_json(self.root/'analysis.json', value)
        # A full unsuccessful experiment still completes; adoption remains gated.
        write_json(self.root/'development_result_gate.json', dict(passed=point_passed and robust_speed,
            complete=True, point_gate=point_passed, throughput_interval_gate=robust_speed,
            scope=value['scope']))
        return value

    def run(self):
        with (self.root/'execution_attempt.json').open('x') as f:
            json.dump(dict(started_utc=stamp(), retries=0), f)
        self.heartbeat = threading.Thread(target=self.pulse, daemon=True); self.heartbeat.start()
        reference, natural = completed_root_evidence(self.reference)
        selected = select_cuda_frames(self.reference, natural)
        write_json(self.root/'cuda_frames.json', dict(frames=selected, selection='12 first/last + 4 earliest remaining'))
        write_json(self.root/'dependency_gate.json', dict(passed=True, reference=str(self.reference),
            files={n:digest(self.reference/n) for n in ('exit.json','status.json','B_reproduction_gate.json',
                'target_natural_prefix_gate.json','natural_forensics.json','numeric_cases.json',
                'natural/B/requests.jsonl')}, scope='Completed natural B and twelve actual-prefix evidence cases'))
        self.begin_work('CUDA_COMPATIBILITY', 'target_odds', verifier_calls=48)
        try:
            process = subprocess.run([sys.executable, str(ROOT/'scripts/run_raven_risk_repair.py'),
                '--cuda-run-dir', str(self.root), '--cuda-reference-dir', str(self.reference)],
                cwd=ROOT, timeout=1800)
        finally:
            # Even a failed/timed-out child owns real reservations. Reload before
            # reporting failure; never overwrite or hide its consumed allowance.
            self.ledger = Ledger(self.root/'ledger.json', LIMITS)
        if process.returncode or not (self.root/'cuda_gate.json').exists():
            raise NumericRejected('CUDA compatibility/semantics subprocess failed')
        gate = json.loads((self.root/'cuda_gate.json').read_text())
        if not gate['passed'] or not gate['complete']:
            raise NumericRejected('CUDA compatibility/semantics gate incomplete')
        self.configure_observers(True); self.current=self.variants['target_odds']
        directory=self.root/'natural/target_odds'
        self.start_server('target_odds', directory)
        for qid, seed in EXECUTION_PAIRS:
            record = self.generate('natural', 'target_odds', self.by_id[qid], seed, 2048)
            self.natural.append(record)
            self.inspect_natural(record, directory, require_historical_replay=False)
        self.stop_server()
        self.check_target_prefixes()
        self.check_smoke(reference)
        self.configure_observers(False)
        datasets = {}
        for name in CALIBRATION_ORDER:
            self.current = self.variants[name]
            self.start_server(name, self.root/'calibration'/name)
            data = []
            for seed in (42, 43):
                for qid in CALIBRATION_IDS:
                    data.append(self.generate('calibration', name, self.by_id[qid], seed, 2048))
            self.stop_server(); datasets[name] = data
            if name == 'B':
                self.replay_gate('B16_legacy_signature', data, list(self.historical.values()))
                self.replay_gate('B6_root_actual_ids', data, reference, require_ids=True)
            if name == 'target_odds':
                self.replay_gate('candidate6_observer_actual_ids', data, self.natural, require_ids=True)
        self.make_analysis(datasets)
        self.work = {}; self.status('COMPLETED_REPAIR_UNIT', completed=86, total=86)
        return 0

    def report(self):
        if not hasattr(self, 'ledger'):
            return
        lines = ['# Adaptive RAVEN Target相对支持修复实验', '',
            '最终目标：原正确率与正常结束判对不低于Native/Cactus/SpecCascade，截断减少，吞吐更高且整题更快。',
            '本轮只测试一个预注册候选；8道已观察训练难题/两seed，尚非独立留出或正式论文结论。', '',
            '阶段：'+self.state.get('phase', 'PREPARED'), '',
            '当前任务：'+json.dumps(self.work, ensure_ascii=False), '',
            '| 类别 | 尝试 | 完成 | 上限 |', '| --- | ---: | ---: | ---: |']
        for k,c in self.ledger.counts().items():
            lines.append(f"| {k} | {c['attempted']} | {c['completed']} | {c['limit']} |")
        for p in sorted(self.root.glob('*_gate.json')):
            g=json.loads(p.read_text())
            lines += ['', p.stem+'：'+('通过' if g['passed'] else '未通过')+
                      '；完整='+str(g.get('complete', True))+'。']
        analysis=self.root/'analysis.json'
        if analysis.exists():
            x=json.loads(analysis.read_text())
            lines += ['', '## 同协议无探针校准', '',
                '| 方法 | 原判对/16 | 正常结束判对 | 截断 | token/s | 平均整题秒 |',
                '| --- | ---: | ---: | ---: | ---: | ---: |']
            for name,s in x['summaries'].items():
                lines.append(f"| {name} | {s['correct']} | {s['stopped_correct']} | {s['truncated']} | "
                             f"{s['e2e_tok_s']:.4f} | {s['mean_seconds']:.2f} |")
            lines += ['', '## 结果分析', '']
            for name,c in x['comparisons'].items():
                old=c['quality']['baseline'];new=c['quality']['candidate']
                lines.append(f"- 对{name}：原判对变化{new['correct']-old['correct']:+d}，正常结束判对"
                    f"{new['stopped_correct']-old['stopped_correct']:+d}，截断"
                    f"{new['truncated']-old['truncated']:+d}；吞吐"
                    f"{(new['e2e_tok_s']/old['e2e_tok_s']-1)*100:+.2f}%，平均整题耗时"
                    f"{(new['mean_seconds']/old['mean_seconds']-1)*100:+.2f}%。"
                    f"质量门槛{'通过' if c['quality']['passed'] else '未通过'}，"
                    f"质量/速度联合门槛{'通过' if c['quality_speed']['passed'] else '未通过'}；"
                    f"吞吐比题聚类95%区间{c['intervals']['throughput_ratio_95ci']}。")
            lines += ['', '发展样本联合点门槛：'+('通过' if x['development_point_gate_passed'] else '未通过')+
                '；吞吐区间门槛：'+('通过' if x['throughput_interval_gate_passed'] else '未通过')+'。',
                '该样本针对已知故障，不代表总体正确率；候选不自动采用，完整目标仍需独立留出/test验证。']
        failure=self.root/'failure.json'
        if failure.exists():
            lines += ['', '## 停止原因', '', failure.read_text()]
        lines += ['', '数值/实机输出复现失败与质量拒绝分开记录。0自动重试，24小时含依赖等待；'
            '后台systemd可在对话关闭后运行。原评分、模型、模板与2048输出预算保持。']
        tmp=self.root/'report.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(self.root/'report.md')


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--reference-dir', type=Path)
    p.add_argument('--review-reference-dir', type=Path)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--cuda-run-dir', type=Path)
    p.add_argument('--cuda-reference-dir', type=Path)
    a=p.parse_args()
    a.prefix_cache=0; a.deep_diagnostic=False; a.target_probe=False
    a.decision_trace=False; a.decision_trace_only=False; a.mtp_operator_audit=False
    a.decision_trace_checkpoints=(); a.decision_trace_early_proposals=0
    a.uniform_gemm=True; a.full_bf16_accum=True; a.bnb_row_block=64
    a.rmsnorm_row_limit=255; a.target_backend='TRITON_ATTN'; a.triton_2d=True
    a.lm_head_row_block=16; a.attention_fp32=True; a.canonical_tree=True
    a.target_probe_shape_rows=(); a.target_probe_prefix_manifest=''
    return a


def main():
    a=arguments()
    if a.cuda_run_dir:
        if not a.cuda_reference_dir:
            raise ValueError('CUDA reference required')
        cuda_gate(a.cuda_run_dir.resolve(), a.cuda_reference_dir.resolve())
        return 0
    if not a.run_dir or not a.reference_dir or not a.review_reference_dir:
        raise ValueError('Run and two reference directories required')
    controller=None;code=1
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        controller=RiskRepairUnit(a)
        if a.check_only:
            controller.report();code=0;return 0
        code=controller.run();return code
    except BaseException as exc:
        kind='REPLAY_REJECTED' if isinstance(exc, ReplayRejected) else failure_kind(exc)
        if controller:
            write_json(controller.root/'failure.json', dict(kind=kind,error=str(exc),
                type=type(exc).__name__,traceback=traceback.format_exc(),
                failed_stage=controller.state.get('phase'),work=controller.work,updated_utc=stamp()))
            controller.status(kind)
        code=3 if isinstance(exc,(NumericRejected,QualityRejected)) else 4 if isinstance(
            exc,BudgetStop) else 130 if isinstance(exc,KeyboardInterrupt) else 1
        print(kind+': '+str(exc),file=sys.stderr,flush=True)
        return code
    finally:
        if controller:
            controller.close()
            if not a.check_only:
                write_json(controller.root/'exit.json', dict(exit_code=code,phase=controller.state.get('phase'),
                    ended_utc=stamp(),server_cleaned=True))


if __name__=='__main__':
    raise SystemExit(main())
