#!/usr/bin/env python3
"""Bounded first-operator diagnosis. Short probes are not quality measurements."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_adaptive_round2 import Round2, write_json
from scripts.run_adaptive_round6 import configs


class Repair(Round2):
    def run(self):
        for name, config in configs().items():
            self.requests('operator_probe', name, config,
                          [(self.by_id[1231], 43)], limit=32, diagnostic=True)
        import torch
        evidence = []
        for path in sorted(self.root.glob('operator_probe/*/decision_trace/*/proposal_00.pt')):
            saved = torch.load(path, map_location='cpu', weights_only=False)
            ops = saved['tree_passes'][0]['operators']
            rows = [dict(name=o['name'], nonfinite_inputs=o['nonfinite_inputs'],
                         nonfinite_output=o['nonfinite_output']) for o in ops]
            evidence.append(dict(file=str(path), variant=path.parents[2].name,
                operators=rows, first_bad=next((r for r in rows if r['nonfinite_output']), None)))
        if len(evidence) != 2:
            raise ValueError('Expected one first-pass operator trace per configuration')
        write_json(self.root/'operator_findings.json', dict(evidence=evidence))
        self.status('OPERATOR_DIAGNOSIS_COMPLETE')
        return 0

    def report(self):
        state = json.loads((self.root/'status.json').read_text()) if (self.root/'status.json').exists() else {}
        lines = ['# MTP概率故障：算子定位', '',
                 '本单元只做2条32token算子探针，另有每服务8token预热；不作为正确率、截断或速度证据。',
                 '旧第六轮结果和冻结代码保留；不沿用旧数值门槛。', '',
                 f"当前阶段：{state.get('phase', '准备')}；已完成探针：{state.get('all_completed', 0)}/2。"]
        for name in ('operator_findings.json', 'failure.json'):
            if (self.root/name).exists():
                lines += ['', f'## {name}', '', '```json', (self.root/name).read_text(), '```']
        (self.root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    args.prefix_cache=0; args.deep_diagnostic=False; args.target_probe=False
    args.decision_trace=True; args.mtp_operator_audit=True
    args.uniform_gemm=True; args.full_bf16_accum=True; args.bnb_row_block=64
    args.rmsnorm_row_limit=255; args.target_backend='TRITON_ATTN'; args.triton_2d=True
    args.lm_head_row_block=16; args.attention_fp32=True; args.canonical_tree=True
    controller = None
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        controller = Repair(args)
        write_json(controller.root/'repair_protocol.json', dict(
            registered_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            purpose='first MTP nonfinite operator localization',
            short_probe_calls=2, max_output_tokens=32, warmup_tokens_per_server=8,
            servers=2, question_id=1231, base_seed=43, actual_seed=12353,
            operator_capture='first tree pass of first real proposal only',
            scope='separate bounded repair diagnosis; excluded from original natural quality budget',
            original_round6_remaining_natural_requests=11,
            gpu_verifier_replay_calls=0, automatic_retries=0))
        if args.check_only:
            return 0
        return controller.run()
    except BaseException as exc:
        if controller:
            write_json(controller.root/'failure.json', dict(type=type(exc).__name__, error=str(exc)))
            controller.status('STOPPED_FAILURE')
        raise
    finally:
        if controller:
            controller.stop_server()


if __name__ == '__main__':
    raise SystemExit(main())
