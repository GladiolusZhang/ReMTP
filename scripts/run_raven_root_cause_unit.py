#!/usr/bin/env python3
"""One finite natural-trace unit, followed by actual-state Target comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_adaptive_round2 import Round2, settings, write_json
from scripts.run_adaptive_round4 import NUMERICAL_KEYS
from scripts.run_adaptive_round5 import output_signature
from scripts.run_adaptive_round6 import configs
from scripts.run_raven_review_unit import (ReviewUnit, Ledger, NumericRejected,
    BudgetStop, component_hashes, component_hashes_for_reference, stamp,
    failure_kind, compare_logits)
from scripts.analyze_raven_root_causes import (PAIRS, CHECKPOINTS, requests,
    output_forensics, request_folder, validate_summary, frame_risk)
from remtp.benchmark import fetch_metrics, metric_delta
from remtp.gsm8k_benchmark import PROMPT_TEMPLATE, benchmark_messages, extract_model_answer
from remtp.optimization_metrics import summarize

LIMITS = dict(main_http=6, numeric=12, warmup=2, starts=2)
RUNTIME_SECONDS = 14400
EXECUTION_PAIRS = (PAIRS[-1], *PAIRS[:-1])


class ReplayRejected(NumericRejected):
    pass


def unique_prefix_cases(cases):
    """Same actual IDs share one fresh Target call, avoiding ambiguous selectors."""
    unique = []
    labels = {}
    mapping = {}
    for case in cases:
        prefix = tuple(case['prefix_ids'])
        if prefix not in labels:
            labels[prefix] = case['label']
            unique.append(case)
        mapping[case['label']] = labels[prefix]
    return unique, mapping


def natural_frames(directory, record):
    """Align live IDs and each real natural Target row before any fresh replay."""
    import torch
    trace = request_folder(directory/'decision_trace', record['response_id'])
    probe = request_folder(directory/'target_probe', record['response_id'])
    frames = sorted(trace.glob('verify_*.pt'))
    raw_files = sorted(probe.glob('logits_*.pt'))
    if not 2 <= len(frames) <= 8 or len(raw_files) != len(frames):
        raise NumericRejected('Missing/extra coupled natural verification/raw frames')
    raws = {}
    for path in raw_files:
        f = torch.load(path, map_location='cpu', weights_only=True)
        key = tuple(f['output_ids'])
        if key in raws:
            raise NumericRejected('Duplicate actual raw prefix')
        raws[key] = (path, f)
    coupled = []
    for path in frames:
        f = torch.load(path, map_location='cpu', weights_only=False)
        history = tuple(f['history'])
        prompt = tuple(f['prompt_ids'])
        if (prompt != tuple(record['prompt_token_ids']) or not history or
                history != tuple(record['output_token_ids'][:len(history)])):
            raise NumericRejected('Natural committed prefix differs from API IDs')
        if history not in raws:
            raise NumericRejected('Natural raw and verifier prefixes do not align')
        raw_path, raw = raws[history]
        if (tuple(raw['prompt_ids']) != prompt or raw['shape_probes'] or
                'logits_rows' not in raw or raw['topology'] != f['topology'] or
                not torch.equal(raw['draft_ids'], f['ids']) or
                raw['actual_input_ids'][0] != history[-1] or
                raw['actual_positions'][0] != len(prompt)+len(history)-1):
            raise NumericRejected('Natural tree input/position/topology mismatch')
        if not bool(torch.isfinite(f['draft']).all() and torch.isfinite(f['target']).all()):
            raise NumericRejected('Nonfinite natural P/Q')
        p = torch.softmax(raw['first_logits'].float()/.6, -1)
        actual = f['target'][0].float()
        tv = float((p-actual).abs().sum()/2)
        if tv > 1e-6:
            raise NumericRejected('Actual P is not the recorded natural raw Target distribution')
        coupled.append(dict(frame=str(path), raw_frame=str(raw_path),
            generated_length=len(history), prefix_ids=list(prompt+history),
            prefix_sha256=hashlib.sha256(json.dumps(list(prompt+history)).encode()).hexdigest(),
            actual_raw_to_P_cpu_tv=tv, frame_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            raw_frame_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest()))
    return coupled


def early_proposals(directory, record, summaries):
    """Record bookkeeping alignment; copies are not independent MTP/KV replays."""
    import torch
    trace = request_folder(directory/'decision_trace', record['response_id'])
    paths = sorted(trace.glob('proposal_*.pt'))
    if len(paths) != 3:
        raise NumericRejected('Exactly three registered early proposal frames required')
    checks = []
    for number, path in enumerate(paths):
        f = torch.load(path, map_location='cpu', weights_only=False)
        row = summaries[number]
        if (f['branch_round'] != number or f['ids'].tolist() != row['candidate_ids'] or
                list(map(list, f['topology'])) != row['topology'] or
                not bool(torch.isfinite(f['q']).all())):
            raise NumericRejected('Early proposal/actual verification bookkeeping mismatch')
        root = int(f['root_positions'].reshape(-1)[0])
        checks.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            branch_round=number, verification_generated_length=row['generated_length'],
            history_before_current_sampling_length=len(f['history_before_current_sampling']),
            root_position=root,
            observed_position_offset_from_committed_last=(
                len(record['prompt_token_ids'])+row['generated_length']-1-root),
            actual_ids_topology_equal=True, q_finite=True,
            scope='Early observed proposal/summary alignment; no full MTP/KV equivalence claim'))
    return checks


class RootCauseUnit(ReviewUnit):
    def __init__(self, args):
        # Retain the tested ReviewUnit -> Round2 MRO for startup/warmup/cleanup.
        # This diagnostic has its own policy, cases and ledger.
        Round2.__init__(self, args)
        self.reference = args.reference_dir.resolve()
        self.protocol = json.loads((self.root/'protocol.json').read_text())
        old = json.loads((self.reference/'protocol.json').read_text())
        for key in (*NUMERICAL_KEYS, 'train_sha256', 'test_sha256', 'primary_prompt'):
            if self.protocol[key] != old[key]:
                raise ValueError('Fixed protocol mismatch: '+key)
        previous = component_hashes_for_reference(self.reference/'frozen_code')
        self.components = component_hashes(ROOT)
        for group in previous:
            if self.components[group] != previous[group]:
                raise ValueError('Unregistered mathematical/runtime change: '+group)
        self.components['root_cause_observer'] = dict(files={
            n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in (
                'remtp/decision_trace.py', 'remtp/target_diagnostics.py',
                'scripts/run_adaptive_round2.py', 'scripts/run_raven_root_cause_unit.py',
                'scripts/analyze_raven_root_causes.py')})
        write_json(self.root/'component_hashes.json', dict(current=self.components,
            fixed_reference=previous, scope='Public math/state/verifier/serving/scorer source identity; '
                'new instrumentation separately hashed and must pass natural replay checks'))
        self.historical = {(r['question_id'], r['seed']): r for r in
            requests(self.reference/'calibration/B/requests.jsonl')}
        if len(self.historical) != 16:
            raise ValueError('Complete historical B calibration required')
        self.variants = {'B': {**configs()['old_eos01'], 'probability_ratio_cap': 0.0},
                         'target': settings(method='target', head=1)}
        self.ledger = Ledger(self.root/'ledger.json', LIMITS)
        self.state = {}; self.work = {}
        self.halt = threading.Event(); self.heartbeat = None
        self.started = time.monotonic(); self.runtime_limit = RUNTIME_SECONDS
        self.request_timeout = 5400
        self.natural = []; self.cases = []; self.forensics = []
        policy = dict(limits=LIMITS, runtime_max_seconds=RUNTIME_SECONDS, retries=0,
            restart='no', heldout=False, formal=False, automatic_adoption=False,
            max_candidates_this_unit=0, pairs=PAIRS, execution_order=EXECUTION_PAIRS,
            checkpoints=CHECKPOINTS,
            quality_tokens=2048, early_proposals_per_request=3,
            verifier_frames_per_request=8, natural_target_frames_per_request=8,
            summary_max_rounds_per_request=2048, summary_max_requests=6,
            shape_probe_calls=0, cuda_counterfactual_replays=0,
            diagnostics_disk_max_bytes=2*1024**3,
            fresh_target_selection='first and last captured actual prefix per request; '
                '12 natural case comparisons, <=12 unique sequential-prefill HTTP calls. '
                'Exactly identical full ID prefixes share one fresh raw result; '
                'one actual final-row capture per unique call',
            natural_comparison='returned live prompt/output IDs and original text/token-count/finish '
                'signature versus historical B; diagnostic timings excluded from speed claims',
            probability_gate='natural retained-KV raw versus fresh sequential raw must be bitwise equal; '
                'raw-to-saved-P CPU softmax TV <=1e-6; no relaxed fresh raw tolerance',
            variants=self.variants, generation_http_timeout_seconds=self.request_timeout,
            source_sha256=self.protocol['source_sha256'],
            original_reference=str(self.reference), observation_only=True)
        p = self.root/'root_cause_policy.json'
        if p.exists() and json.loads(p.read_text()) != json.loads(json.dumps(policy)):
            raise ValueError('Policy changed; use a new directory')
        write_json(p, policy)
        write_json(self.root/'output_forensics.json', output_forensics(self.reference))
        self.ensure_budget([self.by_id[q] for q, _ in PAIRS])

    def generate(self, phase, name, row, seed, limit):
        """The original sampling payload, plus vLLM's serialization-only ID flag."""
        directory = self.root/phase/name
        directory.mkdir(parents=True, exist_ok=True)
        seed += row['question_id']*10
        key = f'{phase}/{name}/{row["question_id"]}/{seed}'
        self.begin_work(phase, name, question_id=row['question_id'], seed=seed, max_tokens=limit)
        before = fetch_metrics(self.base_url, 30)
        payload = dict(model='TencentBAC/FastMTP', messages=benchmark_messages(
            PROMPT_TEMPLATE.format(question=row['question']), empty_system_prompt=True),
            chat_template_kwargs={'enable_thinking': False}, temperature=.6,
            seed=seed, max_tokens=limit, return_token_ids=True)
        self.ledger.reserve('main_http', key, question_id=row['question_id'], seed=seed, max_tokens=limit)
        began = time.perf_counter()
        req = urllib.request.Request(self.base_url+'/v1/chat/completions',
            json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
            value = json.load(response)
        elapsed = time.perf_counter()-began
        write_json(directory/f'response_{row["question_id"]}_{seed}.json',
                   dict(payload=payload, response=value))
        after = fetch_metrics(self.base_url, 30)
        choice = value['choices'][0]
        text = choice['message']['content'] or ''
        pred, source = extract_model_answer(text)
        record = dict(question_id=row['question_id'], seed=seed, gold_answer=row['gold_answer'],
            predicted_answer=pred, answer_source=source, correct=pred==row['gold_answer'],
            finish_reason=choice['finish_reason'], output=text,
            output_tokens=value['usage']['completion_tokens'], client_seconds=elapsed,
            metrics=metric_delta(before, after), response_id=value['id'], max_tokens=limit,
            prompt_token_ids=value.get('prompt_token_ids'), output_token_ids=choice.get('token_ids'),
            observation_enabled=bool(self.diagnostic or getattr(self.args, 'decision_trace_only', False)
                                     or getattr(self.args, 'target_probe', False)),
            performance_sample=not bool(self.diagnostic or getattr(self.args, 'decision_trace_only', False)
                                        or getattr(self.args, 'target_probe', False)))
        with (directory/'requests.jsonl').open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
        self.ledger.complete('main_http', key, seconds=elapsed, output_tokens=record['output_tokens'])
        if not record['prompt_token_ids'] or not record['output_token_ids']:
            raise NumericRejected('Installed API did not return live prompt/output IDs')
        self.status(phase, name, len(requests(directory/'requests.jsonl')), len(PAIRS))
        return record

    def inspect_natural(self, record, directory, require_historical_replay=True):
        import torch
        ref = self.historical[record['question_id'], record['seed']]
        signature_equal = output_signature(record) == output_signature(ref)
        if require_historical_replay and not signature_equal:
            raise ReplayRejected('Observed B output differs from historical unobserved B')
        summary = [r for r in requests(directory/'decision_summary.jsonl')
            if r['request'] == record['response_id'] or r['request'].startswith(record['response_id']+'-')]
        findings = validate_summary(summary, record)
        coupled = natural_frames(directory, record)
        proposal = early_proposals(directory, record, summary)
        risks = []
        for c in coupled:
            f = torch.load(c['frame'], map_location='cpu', weights_only=False)
            risks.append(dict(generated_length=c['generated_length'],
                              events=frame_risk(f, self.eos_ids)))
        qid = record['question_id']; base = record['seed']-qid*10
        self.forensics.append(dict(question_id=qid, seed=base, signature_equal=signature_equal,
            summary=findings, captured_risk=risks, early_proposals=proposal, coupled_frames=coupled))
        for label, c in (('first', coupled[0]), ('last', coupled[-1])):
            self.cases.append({**c, 'label': f'q{qid}_s{base}_{label}',
                               'question_id': qid, 'seed': base})
        write_json(self.root/'natural_forensics.json', dict(requests=self.forensics,
            passed=True, complete=len(self.forensics)==len(PAIRS), scope='Observed training '
            'metadata/coupling; scalar counterfactuals do not establish natural causal outcome'))
        size = sum(p.stat().st_size for n in ('decision_trace', 'target_probe')
                   for p in (directory/n).rglob('*') if p.is_file())
        if size > 2*1024**3:
            raise BudgetStop('Registered diagnostic disk allowance exceeded')

    def run(self):
        attempt = self.root/'execution_attempt.json'
        with attempt.open('x') as f:
            json.dump(dict(started_utc=stamp(), retries=0), f)
        self.heartbeat = threading.Thread(target=self.pulse, daemon=True)
        self.heartbeat.start()
        directory = self.root/'natural/B'
        self.current = self.variants['B']; self.diagnostic = False
        self.args.decision_trace = True; self.args.decision_trace_only = True
        self.args.decision_trace_checkpoints = CHECKPOINTS
        self.args.decision_trace_early_proposals = 3
        self.args.target_probe = True; self.args.target_probe_min_generated = 1
        self.args.target_probe_min_prompt_tokens = 65
        self.args.target_probe_checkpoints = CHECKPOINTS; self.args.target_probe_max_frames = 8
        self.start_server('B', directory)
        for qid, seed in EXECUTION_PAIRS:
            record = self.generate('natural', 'B', self.by_id[qid], seed, 2048)
            self.natural.append(record)
            self.inspect_natural(record, directory)
        self.stop_server()
        write_json(self.root/'B_reproduction_gate.json', dict(passed=True, requests=len(self.natural),
            scope='Live IDs internally consistent; historical text/count/finish signatures equal. '
                'Historical API did not save IDs, so no cross-run token-ID equality claim.'))
        checks, unique_cases = self.check_target_prefixes()
        write_json(self.root/'analysis.json', dict(observation_only=True, candidate_selected=False,
            natural_requests=len(self.natural), actual_prefix_checks=len(checks),
            unique_fresh_prefixes=len(unique_cases),
            target_gate_passed=True, findings=self.forensics, quality_scope='Observed development cases '
                'only; no new quality candidate and no formal throughput claim'))
        self.work = {}
        self.status('COMPLETED_ROOT_CAUSE_UNIT', completed=6, total=6)
        return 0

    def check_target_prefixes(self):
        """Check twelve actual natural cases using at most twelve fresh calls."""
        if len(self.cases) != 12:
            raise NumericRejected('Exactly twelve registered actual-prefix cases required')
        unique_cases, prefix_labels = unique_prefix_cases(self.cases)
        write_json(self.root/'numeric_cases.json', dict(cases=self.cases,
            unique_fresh_cases=unique_cases, canonical_fresh_label=prefix_labels))
        selectors = [dict(label=c['label']+'_target', prompt_ids=c['prefix_ids'], output_ids=[])
                     for c in unique_cases]
        write_json(self.root/'probe_selectors.json', dict(cases=selectors))
        self.args.decision_trace = False; self.args.decision_trace_only = False
        self.args.decision_trace_checkpoints = (); self.args.decision_trace_early_proposals = 0
        self.args.target_probe_checkpoints = (); self.args.target_probe_min_generated = 0
        self.args.target_probe_max_frames = 1
        self.args.target_probe_prefix_manifest = str(self.root/'probe_selectors.json')
        self.current = self.variants['target']
        directory = self.root/'numeric/target'; directory.mkdir(parents=True, exist_ok=True)
        self.start_server('target', directory)
        import torch
        checks = []; fresh_by_label = {}
        for case in self.cases:
            canonical = prefix_labels[case['label']]
            if canonical not in fresh_by_label:
                fresh_by_label[canonical] = self.numeric_call(case, 'target', directory)
            fresh = fresh_by_label[canonical]
            actual = torch.load(case['raw_frame'], map_location='cpu', weights_only=True)['first_logits']
            check = dict(label=case['label'], question_id=case['question_id'], seed=case['seed'],
                generated_length=case['generated_length'], prefix_sha256=case['prefix_sha256'],
                fresh_source_label=canonical, fresh_reused=canonical!=case['label'],
                comparison=compare_logits(actual, fresh, self.eos_ids))
            checks.append(check)
            passed = all(c['comparison']['passed'] for c in checks)
            write_json(self.root/'target_natural_prefix_gate.json', dict(passed=passed,
                complete=len(checks)==12, checks=checks,
                unique_fresh_calls_completed=len(fresh_by_label),
                scope='12 actual retained-KV natural cases versus fresh sequential same-ID '
                    'prefill. Exactly identical full prefixes share one fresh call; '
                    'not 12 necessarily distinct prefixes or universal state proof'))
            if not passed:
                raise NumericRejected('Natural Target versus sequential raw mismatch: '+case['label'])
        self.stop_server()
        return checks, unique_cases

    def report(self):
        lines = ['# Adaptive RAVEN 根因定位进度', '',
            '目标：原评分正确率及正常结束判对不低于Native/Cactus/SpecCascade，同时吞吐更高、整题更快。',
            '本单元只定位，6条完整自然轨迹＋12个真实缓存前缀的独立Target检查；没有新算法候选。', '']
        if self.state:
            lines += ['阶段：'+self.state['phase']+'；当前任务：'+json.dumps(self.work, ensure_ascii=False), '']
        lines += ['| 调用类别 | 已尝试 | 已完成 | 上限 |', '| --- | ---: | ---: | ---: |']
        for key, c in self.ledger.counts().items():
            lines.append(f"| {key} | {c['attempted']} | {c['completed']} | {c['limit']} |")
        lines += ['', '## 已采自然轨迹', '',
            '| 题/seed | 原输出复现 | 轮数 | 相对严格P/Q额外接受 | 与同树uniform接受不同 |', '| --- | --- | ---: | ---: | ---: |']
        for row in self.forensics:
            c = row['summary']['counts']
            lines.append(f"| {row['question_id']}/{row['seed']} | 是 | {c['rounds']} | "
                f"{c.get('extra_accept_over_strict',0)} | {c.get('accept_differs_from_uniform',0)} |")
        lines += ['', '上述是同一现场uniform的局部反事实，不能解释为换算法后整条输出的因果效果；所有带探针耗时排除出速度比较。']
        gate = self.root/'target_natural_prefix_gate.json'
        if gate.exists():
            g = json.loads(gate.read_text())
            lines += ['', f"真实缓存前缀数值检查：{len(g['checks'])}/12；当前门槛{'通过' if g['passed'] else '未通过'}。"]
        failure = self.root/'failure.json'
        if failure.exists():
            lines += ['', '## 停止原因', '', failure.read_text()]
        lines += ['', '4小时绝对运行上限、无自动重试；关闭对话后由systemd继续。原模型/数值/模板/评分/输出预算保持。']
        tmp = self.root/'report.md.tmp'; tmp.write_text('\n'.join(lines)+'\n'); tmp.replace(self.root/'report.md')


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    a.prefix_cache=0; a.deep_diagnostic=False; a.target_probe=False
    a.decision_trace=False; a.decision_trace_only=False; a.mtp_operator_audit=False
    a.decision_trace_checkpoints=(); a.decision_trace_early_proposals=0
    a.uniform_gemm=True; a.full_bf16_accum=True; a.bnb_row_block=64
    a.rmsnorm_row_limit=255; a.target_backend='TRITON_ATTN'; a.triton_2d=True
    a.lm_head_row_block=16; a.attention_fp32=True; a.canonical_tree=True
    a.target_probe_shape_rows=(); a.target_probe_prefix_manifest=''
    return a


def main():
    a = arguments(); controller = None; code = 1
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        controller = RootCauseUnit(a)
        if a.check_only:
            code = 0
            return code
        code = controller.run()
        return code
    except BaseException as exc:
        kind = 'REPLAY_REJECTED' if isinstance(exc, ReplayRejected) else failure_kind(exc)
        if controller:
            write_json(controller.root/'failure.json', dict(kind=kind, error=str(exc),
                type=type(exc).__name__, traceback=traceback.format_exc(),
                failed_stage=controller.state.get('phase'), work=controller.work, updated_utc=stamp()))
            controller.status(kind)
        code = 3 if isinstance(exc, NumericRejected) else 4 if isinstance(exc, BudgetStop) else (
            130 if isinstance(exc, KeyboardInterrupt) else 1)
        print(kind+': '+str(exc), file=sys.stderr, flush=True)
        return code
    finally:
        if controller:
            controller.close()
            if not a.check_only:
                write_json(controller.root/'exit.json', dict(exit_code=code,
                    phase=controller.state.get('phase'), ended_utc=stamp(), server_cleaned=True))


if __name__ == '__main__':
    raise SystemExit(main())
