#!/usr/bin/env python3
"""Finite diagnosis and decision-preserving optimization; no test-set tuning."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_adaptive_round2 import Round2, settings, write_json
from scripts.run_adaptive_round4 import Round4, GateRejected, NUMERICAL_KEYS
from scripts.run_adaptive_round5 import output_signature
from remtp.optimization_metrics import paired_gate, summarize


def configs():
    old = settings(head=1)
    anchor = settings(depth=4, head=3, anchor=True, topk=True, delta=.5, protect=True)
    return {
        'old_eos01': {**old, 'protect': True, 'eos_threshold': .01,
                      'repeat_guard': False, 'relax_weights': [1.0]*3, 'lazy_residual': False},
        'anchor4_eos01': {**anchor, 'eos_threshold': .01,
                          'repeat_guard': True, 'relax_weights': [1.0]*4, 'lazy_residual': False},
    }


def equal_trace(left, right):
    import torch
    if left.keys() != right.keys():
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
                return False
        elif a != b:
            return False
    return True


def equal_saved_tensor(left, right):
    """Raw diagnostics may contain the same NaNs; report them separately."""
    import torch
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.is_floating_point():
        same_nan = torch.isnan(left) & torch.isnan(right)
        return bool(torch.all((left == right) | same_nan))
    return bool(torch.equal(left, right))


def proposal_boundary_matches(proposal, frame):
    import torch
    earlier=proposal['history_before_current_sampling']
    # EAGLE/MTP shifts input tokens while keeping the positions of the
    # corresponding previous Target hidden states (see set_inputs_first_pass).
    expected=len(frame['prompt_ids'])+len(frame['history'])-2
    return (tuple(frame['history'][:len(earlier)])==tuple(earlier)
            and int(proposal['root_positions'].reshape(-1)[0])==expected
            and frame['topology']==proposal['topology']
            and equal_saved_tensor(frame['ids'],proposal['ids'])
            and equal_saved_tensor(frame['draft'],proposal['q'])
            and torch.equal(frame['rng_before'],proposal['rng_after']))


def cuda_equivalence(files, output):
    """Exactly 2 calls/frame, <=64 calls. Restore the original CUDA state."""
    import torch
    from remtp.dynamic_mtp_tree import DynamicTreeTopology, residual_hit_tree_verify
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or not 1 <= len(files) <= 32:
        raise ValueError('CUDA and 1..32 saved frames required')
    checks = []
    for path in files:
        frame = torch.load(path, map_location='cpu', weights_only=False)
        if frame['rng_before'] is None or not frame['rng_device'].startswith('cuda'):
            raise ValueError('Original CUDA generator is missing')
        topology = DynamicTreeTopology.from_paths(frame['topology'])
        inputs = [frame[key].cuda() for key in ('ids', 'draft', 'target')]
        results = []
        events = []
        states = []
        for lazy in (False, True):
            rng = torch.Generator(device=frame['rng_device'])
            rng.set_state(frame['rng_before'])
            trace = {}
            result = residual_hit_tree_verify(topology, *inputs, rng,
                config=frame['config'], protected_node_indices=frame['protected_node_indices'],
                lazy_residual=lazy, decision_trace=trace)
            results.append(asdict(result)); events.append(trace['events']); states.append(rng.get_state())
        passed = (results[0] == results[1] == frame['result']
                  and torch.equal(states[0], states[1]) and torch.equal(states[0], frame['rng_after'])
                  and len(events[0]) == len(events[1]) == len(frame['events'])
                  and all(equal_trace(a, b) and equal_trace(a, c)
                          for a, b, c in zip(events[0], events[1], frame['events'])))
        checks.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           passed=bool(passed), terminal=frame['result']['terminal'],
                           accepted=len(frame['result']['accepted_node_indices']),
                           residual_calls_avoided=sum(e['kind']=='accept_uniform' and e['keep'] for e in frame['events'])))
        write_json(output, dict(passed=all(c['passed'] for c in checks), checks=checks,
            cuda_verifier_calls=len(checks)*2, complete=len(checks)==len(files),
            scope='Actual captured tensors/events/output/RNG equality; not full Target/KV proof or speed benchmark'))
        if not passed:
            raise ValueError(f'CUDA conditional replay differs: {path}')


class Round6(Round4):
    def __init__(self, args):
        Round2.__init__(self, args)
        self.protocol = json.loads((self.root/'protocol.json').read_text())
        self.source = args.reference_dir.resolve()
        self.controls = configs()
        self.policy = dict(reference_dir=str(self.source), controls=self.controls,
            candidate={**self.controls['old_eos01'], 'lazy_residual': True},
            first_pairs=[[1231,43],[5180,43]], additional_ids=[1019,7300,6530,7311],
            diagnostic_request_limit=16, conditional_cuda_call_limit=64,
            train_ids=[r['question_id'] for r in self.regression], train_seeds=[42,43,44],
            train_request_limit=144, quality_max_tokens=2048,
            train_seed_block_order={'42':['B','lazy_residual'],'43':['lazy_residual','B'],'44':['B','lazy_residual']},
            observation_round_limit=8, retries=0, formal_test=False, adoption=False,
            stages=['first_four_replay','conditional_cuda_equivalence','additional_and_restart_diagnosis',
                    'fresh_B_and_lazy_train_regression'],
            remaining_plan='Quality/truncation candidate requires conditional diagnosis; heldout/formal stay gated',
            source_policy='No old numerical gate inheritance after source changes; new proofs are scope limited')
        if getattr(args,'prior_aborted_dir',None):
            prior=args.prior_aborted_dir.resolve()
            jobs=json.loads((prior/'jobs.json').read_text())
            saved=sum(len(p.read_text().splitlines()) for p in prior.glob('*/*/requests.jsonl'))
            if saved or set(jobs)!={'diagnostics/old_eos01'} or not (prior/'failure.json').exists():
                raise ValueError('Expected exactly one interrupted initial request, zero completed requests')
            self.policy.update(prior_aborted_dir=str(prior),prior_aborted_requests=1,
                diagnostic_request_limit=15, deferred_restart_pair=['old_eos01',1231,43],
                budget_note='Prior aborted request consumes one of the 16 natural diagnostic slots; old_eos01/1231 full restart deferred')
        path = self.root/'round6_policy.json'
        if path.exists() and json.loads(path.read_text()) != self.policy:
            raise ValueError('Round6 policy changed; use a new directory')
        write_json(path, self.policy)

    def preflight(self):
        old = json.loads((self.source/'protocol.json').read_text())
        for key in (*NUMERICAL_KEYS, 'train_sha256', 'test_sha256', 'primary_prompt'):
            if old[key] != self.protocol[key]:
                raise ValueError('Reference protocol mismatch: '+key)
        jobs = json.loads((self.source/'jobs.json').read_text())
        references = {}; provenance = {}
        for name, config in self.controls.items():
            key = 'diagnostics/'+name
            path = self.source/key/'requests.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            original_config = {k:v for k,v in config.items() if k not in ('relax_weights','lazy_residual')}
            # The old anchor used protect=True as its implicit repeat setting.
            if name == 'anchor4_eos01':
                original_config.pop('repeat_guard')
            expected = [[q,s+q*10] for q,s in self.policy['first_pairs']]
            job = jobs[key]
            if (job['settings'] != original_config or job['pairs'] != expected
                    or job['limit'] != 2048 or not job['diagnostic']
                    or [[r['question_id'],r['seed']] for r in records] != expected):
                raise ValueError('Historical replay reference incomplete/incompatible: '+name)
            for record in records:
                if record['gold_answer'] != self.by_id[record['question_id']]['gold_answer']:
                    raise ValueError('Reference gold mismatch')
            references[name] = {(r['question_id'],r['seed']):r for r in records}
            provenance[name] = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        write_json(self.root/'reference_manifest.json', dict(files=provenance,
            historical_source_sha256=old['source_sha256'], current_source_sha256=self.protocol['source_sha256'],
            inherited_numerical_gate=False, scope='Historical outputs used for replay only; old speed not reused'))
        self.reference_records=references
        self.status('PREFLIGHT_COMPLETE')
        return references

    def status(self, phase, variant='', completed=0, total=0):
        Round2.status(self,phase,variant,completed,total)
        # Validate the first complete request before allowing its group to
        # continue. A failed startup/semantic replay is not a tuning result.
        if phase=='diagnostics' and variant in getattr(self,'reference_records',{}) and completed:
            path=self.root/phase/variant/'requests.jsonl'
            record=json.loads(path.read_text().splitlines()[completed-1])
            original=self.reference_records[variant][record['question_id'],record['seed']]
            passed=output_signature(record)==output_signature(original)
            if not passed or (variant=='old_eos01' and completed==1):
                self.gate('first_request_replay',dict(passed=passed,question_id=record['question_id'],
                    seed=record['seed'],scope='First full quality-budget GPU startup and historical output check'))

    def start_server(self, variant, directory, profile=False):
        # Execute Round4's real super() chain, then annotate the actual options.
        Round4.start_server(self, variant, directory, profile)
        options = json.loads((directory/'runtime_options.json').read_text())
        options.update(decision_trace=bool(self.diagnostic and self.args.decision_trace),
                       trace_rounds=8 if self.diagnostic else 0,
                       lazy_residual=bool(self.current.get('lazy_residual',False)),
                       explicit_relax_weights=self.current['relax_weights'])
        write_json(directory/'round6_effective_options.json', options)
        log = (directory/'server.log').read_text()
        if 'relax_depth_weights='+str(self.current['relax_weights']) not in log:
            raise ValueError('Worker did not report the explicit relaxation weights')
        if 'lazy_residual='+str(int(self.current.get('lazy_residual',False))) not in log:
            raise ValueError('Worker did not report the residual switch')
        if self.diagnostic and '[ReMTP][DecisionTrace] installed' not in log:
            raise ValueError('Worker did not install full decision capture')

    def trace_index(self, names):
        import torch
        files = []
        index = []
        for name in names:
            directory = self.root/'diagnostics'/name
            records = [json.loads(line) for line in (directory/'requests.jsonl').read_text().splitlines()]
            folders = sorted((directory/'decision_trace').iterdir())
            if len(folders) != len(records):
                raise ValueError('Trace/request count mismatch: '+name)
            for folder in folders:
                frames = sorted(folder.glob('verify_*.pt'))
                proposals = sorted(folder.glob('proposal_*.pt'))
                if len(frames) != 8 or len(proposals) != 8:
                    raise ValueError('Missing first eight trace frames: '+str(folder))
                for number, path in enumerate(frames):
                    frame = torch.load(path, map_location='cpu', weights_only=False)
                    proposal = torch.load(proposals[number], map_location='cpu', weights_only=False)
                    if (frame['round'] != number or not proposal_boundary_matches(proposal,frame)
                            or frame['config'].max_depth != self.controls[name.split('_restart')[0].split('_additional')[0]]['depth']):
                        raise ValueError('Proposal/verification state boundary mismatch: '+str(path))
                    index.append(dict(file=str(path), group=name, round=number,
                        generated=len(frame['history']), vocab=frame['target'].shape[-1],
                        events=len(frame['events']), rng_proposal_to_verify_identical=True))
                files.extend(frames)
        write_json(self.root/'trace_index.json',dict(frames=index,
            observation_only=True, scope='First 8 rounds; excludes later KV/position failure claims'))
        return files

    def run(self):
        import subprocess
        refs = self.preflight()
        pairs = [(self.by_id[q],s) for q,s in self.policy['first_pairs']]
        replay = []; captured = {}
        for name, config in self.controls.items():
            records = self.requests('diagnostics',name,config,pairs,diagnostic=True)
            captured[name] = records
            for record in records:
                original = refs[name][record['question_id'],record['seed']]
                replay.append(dict(group=name, question_id=record['question_id'],seed=record['seed'],
                                   passed=output_signature(record)==output_signature(original)))
        self.gate('natural_replay',dict(passed=all(r['passed'] for r in replay),checks=replay,
            scope='New-source actual 2048-token outputs; no inherited numerical gate'))
        files = self.trace_index(list(self.controls))
        self.status('CUDA_CONDITIONAL_EQUIVALENCE')
        index = self.root/'cuda_frames.json'; write_json(index,[str(path) for path in files])
        # A child process releases the CUDA context before the next model load.
        subprocess.run([str(ROOT/'.venv/bin/python'), str(Path(__file__).resolve()),
            '--cuda-frames',str(index),'--cuda-output',str(self.root/'cuda_equivalence.json')],check=True)
        equality = json.loads((self.root/'cuda_equivalence.json').read_text())
        self.gate('cuda_equivalence',equality)
        from scripts.analyze_adaptive_round6_trace import analyze
        diagnosis=analyze(files)
        write_json(self.root/'conditional_diagnosis.json',diagnosis)
        common=diagnosis['actual_common_root_comparisons']
        self.gate('observed_common_prefix_probability',dict(passed=len(common)>=2
            and all(c['full_probability_tv']<=.15 for c in common),checks=common,
            scope='Full processed-P for actual shared initial roots only; raw-logit/KV coverage not inherited'))
        # Nonfinite raw Q is silently made uniform by verification, whereas
        # the exponential-race proposer samples it without that normalization.
        # Equality to historical outputs cannot validate this sampling rule.
        invalid=diagnosis['nonfinite_probability_rows']
        self.gate('raw_proposal_probability',dict(passed=not invalid,checks=invalid,
            scope='Do not expand performance/quality runs with an unresolved raw Q/P fault',
            issue='Nonfinite Q proposal versus uniform verifier fallback requires MTP-state localization'))
        extra = [(self.by_id[q],43) for q in self.policy['additional_ids']]
        for name, config in self.controls.items():
            restart_pairs=pairs[1:] if name=='old_eos01' and self.policy.get('prior_aborted_requests') else pairs
            restart = self.requests('diagnostics',name+'_restart',config,restart_pairs,diagnostic=True)
            previous={(r['question_id'],r['seed']):r for r in captured[name]}
            self.gate(name+'_restart',dict(passed=all(output_signature(previous[r['question_id'],r['seed']])==output_signature(r)
                for r in restart),scope='Independent restart for listed pairs, same source/config/output budget'))
            self.requests('diagnostics',name+'_additional',config,extra,diagnostic=True)
        # Do not touch heldout/test. These fresh train runs isolate pure compute changes.
        baseline = []; candidate = []
        # Preregistered alternating seed-block order reduces a simple
        # all-baseline-first thermal/time confound. Formal CI is separate.
        for seed in (42,43,44):
            order = self.policy['train_seed_block_order'][str(seed)]
            for name in order:
                config = self.controls['old_eos01'] if name=='B' else self.policy['candidate']
                records = self.blocks('regression',name,config,self.regression,[seed])
                (baseline if name=='B' else candidate).extend(records)
        exact = [dict(question_id=a['question_id'],seed=a['seed'],
            passed=(a['question_id'],a['seed'])==(b['question_id'],b['seed'])
                   and output_signature(a)==output_signature(b)) for a,b in zip(baseline,candidate)]
        self.gate('train_output_equivalence',dict(passed=len(exact)==72 and all(r['passed'] for r in exact),checks=exact))
        quality = paired_gate(baseline,candidate,require_speed=False)
        self.gate('train_quality',quality)
        write_json(self.root/'performance_summary.json',dict(baseline=summarize(baseline),candidate=summarize(candidate),
            quality=quality, scope='Train performance only; unchanged outputs imply unchanged truncation; no three-baseline claim'))
        self.status('COMPLETE_DIAGNOSIS_AND_EQUIVALENT_SPEED_TRAIN')
        return 0

    def report(self):
        lines = ['# 第六轮：完整分叉定位与懒残差优化','',
            '本报告由后台控制器持续更新；未完成阶段不构成成功结论。','',
            '保持模型、量化、Target数值路径、提示、temperature=0.6及2048输出预算。',
            '先完成4条故障复现和64次同张量CUDA验证；原始Q/P含非有限值时停止性能/质量扩展，转入MTP故障定位。',
            '诊断仅前8轮复制完整P/Q/H、纠正/bonus概率、主proposal位置/隐藏状态及CUDA RNG；不拿诊断耗时评速度。','']
        status = self.root/'status.json'
        if status.exists():
            x=json.loads(status.read_text())
            lines += [f"阶段：`{x['phase']}`；当前组 `{x['variant']}`：{x['completed']}/{x['total']}。",
                      f"累计保存 {x['all_completed']} 条主请求；更新时间 {x['updated_utc']}。",'']
        if getattr(self,'policy',None):
            lines += [f"本单元上限：{self.policy['diagnostic_request_limit']}条自然诊断+144条训练；先前中断请求{self.policy.get('prior_aborted_requests',0)}条单列，合计自然诊断额度不超过16。",'']
        lines += ['## 门槛','']
        for path in sorted(self.root.glob('*_gate.json')):
            gate=json.loads(path.read_text())
            lines.append(f"- {path.stem}：{'通过' if gate['passed'] else '未通过'}。")
        for phase in ('diagnostics','regression'):
            paths=sorted((self.root/phase).glob('*/summary.json'))
            if not paths:continue
            lines += ['',f'## {"诊断（不计正式速度）" if phase=="diagnostics" else "训练对照"}','',
                      '| 组 | 请求 | 判对 | 正常结束且判对 | 截断 | E2E token/s | 平均秒 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            for path in paths:
                x=json.loads(path.read_text())
                # Keep this renderer tied to the shared evaluator's real schema.
                lines.append(f"| {path.parent.name} | {x['requests']} | {x['correct']} | {x['stopped_correct']} | {x['truncated']} | {x['e2e_tok_s']:.3f} | {x['mean_seconds']:.1f} |")
        diagnosis=self.root/'conditional_diagnosis.json'
        if diagnosis.exists():
            d=json.loads(diagnosis.read_text())
            lines += ['','## 完整条件证据','',
                '完整P/Q/H、原CUDA RNG与重放结果见 `conditional_diagnosis.json`、`cuda_equivalence_gate.json` 和 `trace_index.json`。',
                f"发现原始概率含非有限值的行：{len(d['nonfinite_probability_rows'])}。原始NaN不会被当作正常概率相等证明；其位置/mask独立记录。",
                '局部严格/松弛决策不同是风险证据，不能单独证明最终错答的因果。']
        lines += ['','## 结论边界','',
            '懒残差默认关闭，只跳过已接受节点的纯残差计算；不能改变答案或减少截断。',
            '源码已变化：不继承旧数值gate。当前新gate只证明记录的自然重放/条件验收范围。',
            '本控制器执行下一步的定位和等价性能阶段。质量/截断候选、完整数值重验、独立留出及三基准正式比较仍按总计划门槛推进；不会自动采用候选。']
        failure=self.root/'failure.json'
        if failure.exists():lines += ['','## 停止原因','',failure.read_text()]
        path=self.root/'report.md'; tmp=path.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir');p.add_argument('--reference-dir',type=Path)
    p.add_argument('--port',type=int,default=8000);p.add_argument('--check-only',action='store_true')
    p.add_argument('--cuda-frames',type=Path);p.add_argument('--cuda-output',type=Path)
    p.add_argument('--prior-aborted-dir',type=Path)
    args=p.parse_args()
    if args.cuda_frames:
        cuda_equivalence([Path(s) for s in json.loads(args.cuda_frames.read_text())],args.cuda_output);return 0
    if not args.run_dir or not args.reference_dir:p.error('run-dir and reference-dir required')
    args.prefix_cache=0;args.deep_diagnostic=False;args.target_probe=False;args.decision_trace=True
    args.uniform_gemm=True;args.full_bf16_accum=True;args.bnb_row_block=64
    args.rmsnorm_row_limit=255;args.target_backend='TRITON_ATTN';args.triton_2d=True
    args.lm_head_row_block=16;args.attention_fp32=True;args.canonical_tree=True
    controller=None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        controller=Round6(args)
        if args.check_only:controller.preflight();return 0
        return controller.run()
    except GateRejected as exc:
        if controller:
            probability_fault=str(exc)=='raw_proposal_probability'
            write_json(controller.root/'failure.json',dict(type='GateRejected',
                category='implementation_fault' if probability_fault else 'evidence_gate',gate=str(exc),
                error='原始Q/P存在非有限值；必须先定位MTP状态与修复提议/验收一致性' if probability_fault else '复现/等价/质量门槛未通过；停止扩展，旧记录保留',
                timestamp_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))
            controller.report()
        return 3
    except BaseException as exc:
        if controller:
            write_json(controller.root/'failure.json',dict(type=type(exc).__name__,category='interrupted' if isinstance(exc,KeyboardInterrupt) else 'runtime_or_protocol',
                error=str(exc),timestamp_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))
            controller.status('STOPPED_FAILURE')
        raise
    finally:
        if controller:controller.stop_server()


if __name__=='__main__':raise SystemExit(main())
