#!/usr/bin/env python3
"""New-source probability gates then decision-preserving speed regression."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_adaptive_round2 import Round2, write_json
from scripts.run_adaptive_round4 import NUMERICAL_KEYS, GateRejected
from scripts.run_adaptive_round5 import output_signature
from scripts.run_adaptive_round6 import configs, proposal_boundary_matches
from remtp.optimization_metrics import paired_gate, summarize, clustered_intervals


def trace_checks(directory):
    import torch
    files = sorted(directory.glob('decision_trace/*/verify_*.pt'))
    checks = []
    for path in files:
        frame = torch.load(path, map_location='cpu', weights_only=False)
        proposal = torch.load(path.with_name(path.name.replace('verify_', 'proposal_')),
                              map_location='cpu', weights_only=False)
        checks.append(dict(file=str(path), boundary=proposal_boundary_matches(proposal, frame),
            finite_q=bool(torch.isfinite(frame['draft']).all()),
            finite_p=bool(torch.isfinite(frame['target']).all()),
            first_hidden_finite=all(t['nonfinite_last_hidden'] == 0 for t in proposal['tree_passes'])))
    return files, checks


class Repaired(Round2):
    def __init__(self, args):
        super().__init__(args)
        self.protocol = json.loads((self.root/'protocol.json').read_text())
        self.reference = args.reference_dir.resolve()
        old = json.loads((self.reference/'protocol.json').read_text())
        for key in (*NUMERICAL_KEYS, 'train_sha256', 'test_sha256', 'primary_prompt'):
            if old[key] != self.protocol[key]:
                raise ValueError('Unregistered numerical/model/data setting change: '+key)
        if len(self.regression) != 24:
            raise ValueError('Expected fixed original 24 regression questions')
        write_json(self.root/'repair_execution_protocol.json', dict(
            historical_reference=str(self.reference), inherited_numerical_gate=False,
            source_sha256=self.protocol['source_sha256'], repairs=['dense draft bias', 'valid query bias reads', 'invalid Q fail before RNG'],
            short_calls=2, short_max_tokens=32, conditional_cuda_calls=32,
            main_diagnostic_calls=4, main_max_tokens=2048,
            original_remaining_natural_after_unit=7, train_requests=144,
            train_ids=[r['question_id'] for r in self.regression], train_seeds=[42,43,44],
            train_order={'42':['B','lazy_residual'],'43':['lazy_residual','B'],'44':['B','lazy_residual']},
            restart='no', automatic_retries=0, heldout=False, formal=False,
            warmup='8 tokens per fresh server; excluded',
            scope='new affected-path proof + train compute comparison; no full old numerical-gate inheritance'))

    def gate(self, name, value):
        write_json(self.root/(name+'_gate.json'), {**value, 'source_sha256':self.protocol['source_sha256']})
        self.report()
        if not value['passed']:
            self.status('STOPPED_'+name.upper())
            raise GateRejected(name)

    def run(self):
        import torch
        controls = configs()
        cuda_files = []
        for name, config in controls.items():
            self.requests('repair_smoke', name, config, [(self.by_id[1231],43)], limit=32, diagnostic=True)
            files, checks = trace_checks(self.root/'repair_smoke'/name)
            self.gate(name+'_short_probability', dict(passed=len(files)==8 and
                all(all(r[k] for k in ('boundary','finite_q','finite_p','first_hidden_finite')) for r in checks), checks=checks))
            cuda_files.extend(files)
        # 16 actual frames x 2 implementations; old 64-call allotment untouched.
        write_json(self.root/'cuda_frames.json', [str(f) for f in cuda_files])
        self.status('NEW_CUDA_CONDITIONAL_GATE')
        subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run_adaptive_round6.py'),
            '--cuda-frames', str(self.root/'cuda_frames.json'), '--cuda-output', str(self.root/'cuda_equivalence.json')],
            cwd=ROOT, check=True)
        self.gate('new_cuda_equivalence', json.loads((self.root/'cuda_equivalence.json').read_text()))
        # Actual first Target root has the same committed prefix across sources.
        root_checks = []
        for name, config in controls.items():
            records = self.requests('diagnostics', name, config,
                [(self.by_id[1231],43),(self.by_id[5180],43)], diagnostic=True)
            files, checks = trace_checks(self.root/'diagnostics'/name)
            self.gate(name+'_full_probability', dict(passed=len(files)==16 and
                all(all(r[k] for k in ('boundary','finite_q','finite_p','first_hidden_finite')) for r in checks), checks=checks,
                scope='first 8 observed rounds/request; proposal Q validation also active throughout request'))
            old_frames = [torch.load(p, map_location='cpu', weights_only=False)
                for p in (self.reference/'diagnostics'/name/'decision_trace').glob('*/verify_00.pt')]
            for path in files:
                if path.name != 'verify_00.pt':
                    continue
                frame = torch.load(path, map_location='cpu', weights_only=False)
                matches = [o for o in old_frames if o['prompt_ids']==frame['prompt_ids'] and o['history']==frame['history']]
                root_checks.append(dict(file=str(path), matched_reference=len(matches)==1,
                    identical_p=len(matches)==1 and torch.equal(frame['target'][0], matches[0]['target'][0])))
            write_json(self.root/(name+'_repair_quality_observation.json'), dict(records=records,
                summary=summarize(records), scope='two preselected fault questions; not population accuracy or adoption gate'))
        self.gate('new_initial_target_root', dict(passed=len(root_checks)==4 and
            all(r['matched_reference'] and r['identical_p'] for r in root_checks), checks=root_checks,
            scope='four actual first processed-P rows bitwise identical; later Target/logit/KV gates not inherited'))
        baseline = []; candidate = []
        for seed in (42,43,44):
            order = ('B','lazy_residual') if seed != 43 else ('lazy_residual','B')
            block = {}
            for name in order:
                config = {**controls['old_eos01'], 'lazy_residual':name=='lazy_residual'}
                block[name] = self.blocks('regression', name, config, self.regression, [seed])
                (baseline if name=='B' else candidate).extend(block[name])
            old_by_key = {(r['question_id'],r['seed']):r for r in block['B']}
            exact = all(output_signature(r)==output_signature(old_by_key[r['question_id'],r['seed']]) for r in block['lazy_residual'])
            self.gate(f'train_seed{seed}_output', dict(passed=exact, pairs=len(block['B'])))
            self.gate(f'train_seed{seed}_quality', paired_gate(block['B'], block['lazy_residual'], require_speed=False))
        self.gate('train_quality', paired_gate(baseline, candidate, require_speed=False))
        write_json(self.root/'performance_summary.json', dict(baseline=summarize(baseline), candidate=summarize(candidate),
            speed_gate=paired_gate(baseline,candidate,require_speed=True),
            intervals=clustered_intervals(baseline,candidate),
            scope='24 train questions x 3 seeds per variant; exact-output lazy compute change cannot reduce truncation; no formal three-baseline claim'))
        self.status('COMPLETE_REPAIR_AND_LAZY_REGRESSION')
        return 0

    def report(self):
        # Base renderer contains quality tables; add visible current-unit scope.
        super().report()
        path = self.root/'report.md'
        text = path.read_text()
        text = text.replace('# Adaptive RAVEN 优化执行报告', '# MTP故障修复与懒残差训练对照')
        text = text.replace('本轮显式独立 head 开关和请求局部 branch RNG；旧配置按新协议重跑，旧日志只作诊断。',
            '先2条32token修复探针+32次新CUDA条件调用，再4条完整2048诊断；门槛通过才做144条B/懒残差训练。原始Q故障、旧冻结与负门槛保留。')
        text = text.replace('短测只有两条提示、128 token，仅筛选性能。调参和回归题不是独立正式证据。',
            '本单元短探针为每配置1条32token，只检查数值。训练题不是独立正式证据。')
        text += '\n## 本轮结论边界\n\n旧输出可因修复非法Q而改变；不强制复现含故障输出。短探针和诊断耗时不计性能。\n未采用质量候选，未运行独立留出或三基准正式测试。截断改善需要单独的算法质量候选，懒残差只优化计算。\n'
        if (self.root/'failure.json').exists():
            text += '\n## 停止原因\n\n```json\n'+(self.root/'failure.json').read_text()+'\n```\n'
        path.write_text(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    args.prefix_cache=0; args.deep_diagnostic=False; args.target_probe=False
    args.decision_trace=True; args.mtp_operator_audit=True
    args.uniform_gemm=True; args.full_bf16_accum=True; args.bnb_row_block=64
    args.rmsnorm_row_limit=255; args.target_backend='TRITON_ATTN'; args.triton_2d=True
    args.lm_head_row_block=16; args.attention_fp32=True; args.canonical_tree=True
    controller = None
    def interrupted(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        controller = Repaired(args)
        if args.check_only: return 0
        return controller.run()
    except BaseException as exc:
        if controller:
            write_json(controller.root/'failure.json',dict(type=type(exc).__name__,error=str(exc)))
            controller.status('STOPPED_FAILURE')
        if isinstance(exc, GateRejected): return 3
        raise
    finally:
        if controller: controller.stop_server()


if __name__ == '__main__': raise SystemExit(main())
