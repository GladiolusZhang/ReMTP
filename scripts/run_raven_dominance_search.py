#!/usr/bin/env python3
"""Finite full-output candidate screen under the user's accuracy/speed objective."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_adaptive_round2 import Round2, settings, write_json
from scripts.run_adaptive_round4 import NUMERICAL_KEYS
from scripts.run_adaptive_round5 import output_signature
from scripts.run_adaptive_round6 import configs
from scripts.run_raven_review_unit import (ReviewUnit, Ledger, NumericRejected, BudgetStop,
    component_hashes, component_hashes_for_reference, stamp, failure_kind, CALIBRATION_IDS)
from scripts.run_raven_root_cause_unit import RootCauseUnit, arguments, requests
from remtp.optimization_metrics import summarize, dominance_gate, clustered_intervals

ORDER = ('B', 'balanced3', 'deep4_balanced', 'deep4_conservative', 'spec_cascade', 'cactus', 'native')
CANDIDATES = ('balanced3', 'deep4_balanced', 'deep4_conservative')
BASELINES = ('native', 'cactus', 'spec_cascade')
LIMITS = dict(main_http=112, starts=7, warmup=7, numeric=0)
RUNTIME_SECONDS = 86400


def variants():
    base = {**configs()['old_eos01'], 'probability_ratio_cap': 0., 'risk_policy': 'none'}
    result = {'B': base}
    for name, depth, delta in (('balanced3', 3, .75), ('deep4_balanced', 4, .75), ('deep4_conservative', 4, .5)):
        result[name] = {**base, 'depth': depth, 'nodes': depth*3, 'head': 1, 'delta': delta,
            'mix': .25, 'relax_weights': [1.]*depth, 'anchor': False, 'repeat_guard': False}
    result.update({name: settings(method=name, head=1) for name in BASELINES})
    return result


def registered_changes(current, reference):
    new, old = component_hashes(current), component_hashes_for_reference(reference)
    changes = {}
    for group, previous in old.items():
        for file, value in previous['files'].items():
            if new[group]['files'][file] != value:
                changes[file] = group
    if set(changes) - {'remtp/optimization_metrics.py'}:
        raise ValueError('Unregistered runtime change: '+str(changes))
    # New objective may be added; historical scoring/statistical definitions
    # must not be edited to reclassify prior experiments.
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    a, b = functions(current/'remtp/optimization_metrics.py'), functions(reference/'remtp/optimization_metrics.py')
    for name in ('summarize', 'paired_gate', 'clustered_intervals', 'summarize_fast'):
        if a.get(name) != b.get(name):
            raise ValueError('Historical metric function changed: '+name)
    return new, old, changes


class DominanceSearch(RootCauseUnit):
    def __init__(self, args):
        Round2.__init__(self, args)
        self.reference = args.reference_dir.resolve()
        self.protocol = json.loads((self.root/'protocol.json').read_text())
        previous = json.loads((self.reference/'protocol.json').read_text())
        if not json.loads((self.reference/'exit.json').read_text())['exit_code'] == 0:
            raise NumericRejected('Reference run did not complete')
        for key in (*NUMERICAL_KEYS, 'train_sha256', 'test_sha256', 'primary_prompt'):
            if self.protocol[key] != previous[key]:
                raise NumericRejected('Fixed protocol mismatch: '+key)
        self.components, old, changes = registered_changes(ROOT, self.reference/'frozen_code')
        write_json(self.root/'component_hashes.json', dict(current=self.components, reference=old,
            actual_changes=changes, scope='Runtime/scorer unchanged; only an additive objective function'))
        self.variants = variants()
        self.ledger = Ledger(self.root/'ledger.json', LIMITS)
        self.state = {}; self.work = {}; self.halt = threading.Event(); self.heartbeat = None
        self.started = time.monotonic(); self.runtime_limit = RUNTIME_SECONDS; self.request_timeout = 5400
        self.datasets = {}; self.analysis = None
        self.policy = dict(objective='accuracy_nondecrease_and_faster_than_all_three',
            limits=LIMITS, runtime_max_seconds=RUNTIME_SECONDS, order=ORDER, variants=self.variants,
            question_ids=CALIBRATION_IDS, seeds=[42,43], max_tokens=2048,
            candidate_count=3, retries=0, restart='no', observers=False, numeric_calls=0,
            stopped_correct_gate=False, truncation_gate=False, pairwise_loss_gate=False, per_seed_gate=False,
            scoring='original unchanged scorer', statistical_unit='question cluster, all registered seeds',
            heldout=False, formal=False, automatic_adoption=False,
            early_stopping='runtime/numeric/budget only; complete quality pairs despite individual losses',
            scope='Known training development set; no claim of independent benchmark dominance')
        self.policy.update(source_sha256=self.protocol['source_sha256'], reference_dir=str(self.reference),
            seed_formula='question_id*10+base_seed', bootstrap_draws=2000,
            common_protocol_note='protocol.json is the inherited common runtime template; '
                'actual variants, 2 seeds, 3 candidates and no heldout execution are registered here')
        path = self.root/'dominance_policy.json'
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(self.policy)):
            raise ValueError('Policy changed; new unit required')
        write_json(path, self.policy)
        self.ensure_budget([self.by_id[q] for q in CALIBRATION_IDS])
        self.report()

    def start_server(self, name, directory, profile=False):
        super().start_server(name, directory, profile)
        env = json.loads((directory/'effective_env.json').read_text())
        for key in ('REMTP_DECISION_TRACE_DIR', 'REMTP_DECISION_TRACE_SUMMARY_JSONL',
            'REMTP_TARGET_DIAGNOSTIC_DIR', 'REMTP_RUNTIME_PROFILE_JSONL', 'REMTP_PREFIX_DIAGNOSTIC_JSONL'):
            if env.get(key):
                raise NumericRejected('Unexpected observational probe: '+key)
        if name in CANDIDATES or name == 'B':
            cfg = self.variants[name]
            log = (directory/'server.log').read_text()
            for expected in (f"D={cfg['depth']} ", f"N_max={cfg['nodes']} ",
                f"adaptive_mix={cfg['mix']:g}", f"cactus_delta={cfg['delta']:g}", 'risk_policy=none'):
                if expected not in log:
                    raise NumericRejected('Effective configuration missing: '+expected)

    def generate(self, phase, name, row, seed, limit):
        record = super().generate(phase, name, row, seed, limit)
        if (len(record['output_token_ids']) != record['output_tokens'] or record['observation_enabled']
                or not record['performance_sample'] or record['max_tokens'] != 2048):
            raise NumericRejected('Incomplete or observed natural output')
        count = len(requests(self.root/phase/name/'requests.jsonl'))
        self.status(phase, name, count, 16)
        return record

    def check_reference(self, name, data):
        old = {(r['question_id'],r['seed']):r for r in requests(self.reference/'calibration'/name/'requests.jsonl')}
        checks = [dict(question_id=r['question_id'], seed=r['seed'],
            passed=output_signature(r)==output_signature(old[r['question_id'],r['seed']])
                and r['output_token_ids']==old[r['question_id'],r['seed']]['output_token_ids']) for r in data]
        passed = len(checks)==16 and all(c['passed'] for c in checks)
        write_json(self.root/(name+'_reproduction_gate.json'),dict(passed=passed,checks=checks))
        if not passed:
            raise NumericRejected('Reference output reproduction failed: '+name)

    def make_analysis(self):
        candidates = {}
        for name in CANDIDATES:
            data = self.datasets[name]
            comparisons = {baseline: dict(point=dominance_gate(before,data),
                intervals=clustered_intervals(before,data))
                for baseline,before in self.datasets.items() if baseline in (*BASELINES,'B')}
            point = all(comparisons[b]['point']['passed'] for b in BASELINES)
            intervals = all(comparisons[b]['intervals']['throughput_ratio_95ci'][0]>1
                and comparisons[b]['intervals']['accuracy_difference_95ci'][0]>=0 for b in BASELINES)
            candidates[name] = dict(comparisons=comparisons, point_passed=point,
                statistical_screen_passed=point and intervals)
        feasible = [name for name in CANDIDATES if candidates[name]['point_passed']]
        selected = min(feasible,key=lambda n:summarize(self.datasets[n])['mean_seconds']) if feasible else None
        self.analysis = dict(summaries={k:summarize(v) for k,v in self.datasets.items()}, candidates=candidates,
            development_point_winner=selected, automatically_adopted=False, formal_goal_achieved=False,
            objective=self.policy['objective'],scope=self.policy['scope'])
        write_json(self.root/'analysis.json',self.analysis)
        write_json(self.root/'development_result_gate.json',dict(passed=selected is not None,
            complete=True, point_winner=selected, independent_validation_required=True))
        self.report()

    def run(self):
        with (self.root/'execution_attempt.json').open('x') as f:
            json.dump(dict(started_utc=stamp(), retries=0),f)
        self.heartbeat=threading.Thread(target=self.pulse,daemon=True); self.heartbeat.start()
        for name in ORDER:
            self.current=self.variants[name]; directory=self.root/'calibration'/name
            self.start_server(name,directory)
            records=[]
            for qid in CALIBRATION_IDS:
                for seed in (42,43):
                    records.append(self.generate('calibration',name,self.by_id[qid],seed,2048))
            self.datasets[name]=records
            write_json(directory/'summary.json',summarize(records))
            if name in (*BASELINES,'B'):
                self.check_reference(name,records)
            self.stop_server()
        self.make_analysis(); self.status('COMPLETED_DOMINANCE_SCREEN')
        return 0

    def report(self):
        lines=['# Adaptive RAVEN：三基准优势搜索进度','',
            '目标：原评分正确率不低于Native/Cactus/SpecCascade，token吞吐更高、整题耗时更低。',
            '截断、正常结束判对、逐题及逐seed差异仅报告；112完整请求，3候选，0自动重试。','']
        if self.state:
            lines+=['阶段：'+self.state['phase'],'当前任务：'+json.dumps(self.work,ensure_ascii=False),'']
        lines+=['| 类别 | 已尝试 | 已完成 | 上限 |','| --- | ---: | ---: | ---: |']
        for kind,c in self.ledger.counts().items():
            lines.append(f"| {kind} | {c['attempted']} | {c['completed']} | {c['limit']} |")
        lines+=['','| 方法 | 完成 | 原判对 | 截断（诊断） | token/s | 平均整题秒 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name in ORDER:
            p=self.root/'calibration'/name/'summary.json'
            if p.exists():
                s=json.loads(p.read_text())
                lines.append(f"| {name} | {s['requests']}/16 | {s['correct']} | {s['truncated']} | {s['e2e_tok_s']:.4f} | {s['mean_seconds']:.2f} |")
        if self.analysis:
            lines+=['','开发集点门槛候选：'+str(self.analysis['development_point_winner']),
                '完整配对、题聚类区间与各基准逐项比较见analysis.json；未自动采用或展开留出/test。']
        if (self.root/'failure.json').exists():
            lines+=['','## 停止现场','',(self.root/'failure.json').read_text()]
        lines+=['','固定模型/数值/原prompt/评分与2048输出预算；探针关闭。24小时绝对上限。',
            '这8道已观察题用于开发选择，不能当作论文正式结论。']
        p=self.root/'report.md.tmp'; p.write_text('\n'.join(lines)+'\n'); p.replace(self.root/'report.md')


def main():
    args=arguments(); controller=None; code=1
    def interrupt(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:
        controller=DominanceSearch(args)
        if args.check_only:
            code=0; return code
        code=controller.run(); return code
    except BaseException as exc:
        kind=failure_kind(exc)
        if controller:
            write_json(controller.root/'failure.json',dict(kind=kind,error=str(exc),type=type(exc).__name__,
                traceback=traceback.format_exc(),work=controller.work,updated_utc=stamp()))
            controller.status(kind)
        code=3 if isinstance(exc,NumericRejected) else 4 if isinstance(exc,BudgetStop) else 130 if isinstance(exc,KeyboardInterrupt) else 1
        print(kind+': '+str(exc),file=sys.stderr,flush=True); return code
    finally:
        if controller:
            controller.close()
            if not args.check_only:
                write_json(controller.root/'exit.json',dict(exit_code=code,phase=controller.state.get('phase'),
                    ended_utc=stamp(),server_cleaned=True))


if __name__=='__main__': raise SystemExit(main())
