#!/usr/bin/env python3
"""Finite EOS protection ablation; no automatic adoption or test-set tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_adaptive_round2 import Round2, settings, write_json
from scripts.run_adaptive_round4 import Round4, GateRejected, NUMERICAL_KEYS, numerical_source_check
from remtp.gsm8k_benchmark import extract_model_answer
from remtp.optimization_metrics import paired_gate, summarize


def output_signature(record):
    return record['output'], record['output_tokens'], record['finish_reason']


def validate_reference(records, job, config, pairs, by_id):
    expected = [[row['question_id'], seed + row['question_id'] * 10] for row, seed in pairs]
    if (job['settings'] != config or job['pairs'] != expected or job['limit'] != 2048
            or job['diagnostic'] or len(records) != len(expected)
            or [[r['question_id'], r['seed']] for r in records] != expected):
        raise ValueError('Reference must contain every ordered question/seed under the exact quality protocol')
    for record in records:
        gold = by_id[record['question_id']]['gold_answer']
        prediction, source = extract_model_answer(record['output'])
        if (record['gold_answer'] != gold or record['predicted_answer'] != prediction
                or record['answer_source'] != source or record['correct'] != (prediction == gold)):
            raise ValueError('Reference dataset/evaluation mismatch')


class Round5(Round4):
    start_server = Round4.start_server
    gate = Round4.gate

    def __init__(self, args):
        # Round4.start_server uses zero-argument super(), so this controller
        # must be its subclass. Initialize the shared protocol without the
        # Round4-specific prefill/state policy and its required arguments.
        Round2.__init__(self, args)
        self.protocol = json.loads((self.root / 'protocol.json').read_text())
        self.source = args.reference_dir.resolve()
        old = settings(head=1)
        anchor = settings(depth=4, head=3, anchor=True, topk=True, delta=.5, protect=True)
        self.controls = {'old': old, 'anchor4_delta05': anchor}
        # Exactly one intervention per matching structural control: stronger
        # existing EOS protection. Keep the old control's repeat guard OFF.
        self.candidates = {
            'old_eos01': {**old, 'protect': True, 'eos_threshold': .01, 'repeat_guard': False},
            'anchor4_eos01': {**anchor, 'eos_threshold': .01},
        }
        policy = dict(reference_dir=str(self.source), candidates=self.candidates,
            regression_ids=[r['question_id'] for r in self.regression], seeds=[42,43],
            diagnostic_pairs=[[1231,43],[5180,43]], quality_max_tokens=2048,
            quality_gate='aggregate AND each seed: correct/stopped_correct do not decrease; truncation does not increase',
            adoption=False, formal_test=False, retries=0,
            limitation='Train ablation only; threshold is a hypothesis, not a proven remedy')
        path = self.root / 'round5_policy.json'
        if path.exists() and json.loads(path.read_text()) != policy:
            raise ValueError('Round5 policy changed; use a new directory')
        write_json(path, policy)

    def references(self):
        checked = numerical_source_check(self.source, self.protocol)
        previous = json.loads((self.source / 'protocol.json').read_text())
        for key in ('train_sha256','primary_prompt','no_think_template_sha256'):
            if previous[key] != self.protocol[key]:
                raise ValueError('Reference dataset/prompt changed')
        # This explicitly permits only the two default-preserving environment
        # overrides. Any other controller execution change forbids reuse.
        before = (Path(previous['source_root']) / 'scripts/run_adaptive_round2.py').read_text()
        expected = before.replace('str(int(x["protect"])),\n                   REMTP_DYNAMIC_TREE_EOS_THRESHOLD="0.5"',
            'str(int(x.get("repeat_guard", x["protect"]))),\n                   REMTP_DYNAMIC_TREE_EOS_THRESHOLD=str(x.get("eos_threshold", 0.5))')
        if expected != (ROOT / 'scripts/run_adaptive_round2.py').read_text():
            raise ValueError('Unreviewed controller change; reference reuse forbidden')
        proof = {}
        for name, count in (('prefill',None),('real_state',None),('reproduction',None),('long_state',152),('fresh_state',344)):
            path = self.source / f'{name}_gate.json'
            gate = json.loads(path.read_text())
            if (not gate['passed'] or any(gate['numerical_protocol'].get(k) != self.protocol[k] for k in NUMERICAL_KEYS)
                    or gate['source_sha256'] != previous['source_sha256']
                    or (count is not None and len(gate.get('checks',[])) != count)):
                raise ValueError(f'Incomplete or incompatible numerical proof: {name}')
            proof[name] = dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        jobs = json.loads((self.source / 'jobs.json').read_text())
        references = {}; provenance = {}
        for name, config in self.controls.items():
            records = []
            for seed in (42,43):
                key = f'regression/{name}_seed{seed}'
                directory = self.source / key
                path = directory / 'requests.jsonl'
                block = [json.loads(line) for line in path.read_text().splitlines()]
                validate_reference(block, jobs[key], config, [(r,seed) for r in self.regression], self.by_id)
                options = json.loads((directory/'runtime_options.json').read_text())
                if (options['settings'] != config or options['target_probe'] or options['target_state_audit']
                        or options['prefix_diagnostic']
                        or any(options['numerical_protocol'].get(k) != self.protocol[k] for k in NUMERICAL_KEYS)):
                    raise ValueError('Reference effective runtime mismatch')
                records.extend(block)
                provenance[key] = dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),requests=len(block))
            references[name] = records
        write_json(self.root/'reference_manifest.json',dict(read_only=True,source=str(self.source),
            files_checked=checked,numerical_proof=proof,quality_references=provenance,
            limitation='Historical TRAIN reference timing is descriptive only; no formal speed conclusion'))
        self.gate('inherited_numerical',dict(passed=True,proof=proof,files_checked=checked,
            note='Execution and numerical protocol unchanged; reuse observed coverage, not a universal proof'))
        return references

    def diagnostic_summary(self):
        result = {}
        for path in sorted((self.root/'diagnostics').glob('*/prefix_diagnostic.jsonl')):
            rows=[]
            for line in path.read_text().splitlines():
                record=json.loads(line)
                rows.extend(n for n in record.get('nodes',[]) if 'verify_eos_probability' in n)
            suppressed=[n for n in rows if n['target_eos_probability'] >= .01
                and n['verify_eos_probability'] < n['target_eos_probability']-1e-7]
            result[path.parent.name]=dict(visited_primary_rows=len(rows),eos_ge01_suppressed=len(suppressed),
                max_suppression=max((n['target_eos_probability']-n['verify_eos_probability'] for n in suppressed),default=0))
        write_json(self.root/'eos_diagnosis.json',dict(groups=result,
            limitation='Visited prefixes differ between runs; conditional suppression is not a causal quality proof'))

    def run(self):
        refs=self.references()
        pairs=[(self.by_id[q],s) for q,s in ((1231,43),(5180,43))]
        replay=[]
        for name in ('old','anchor4_delta05'):
            records=self.requests('diagnostics',name,self.controls[name],pairs,diagnostic=True)
            lookup={(r['question_id'],r['seed']):r for r in refs[name]}
            for record in records:
                original=lookup[record['question_id'],record['seed']]
                replay.append(dict(question_id=record['question_id'],seed=record['seed'],control=name,
                    passed=output_signature(record)==output_signature(original)))
        self.gate('replay',dict(passed=all(r['passed'] for r in replay),checks=replay))
        for name, config in self.candidates.items():
            self.requests('diagnostics',name,config,pairs,diagnostic=True)
        self.diagnostic_summary()
        checks={}
        for name, config in self.candidates.items():
            records=self.blocks('regression',name,config,self.regression,[42,43])
            checks[name]=paired_gate(refs['old'],records,require_speed=False)
            write_json(self.root/'quality_progress.json',checks)
        self.gate('regression',dict(passed=any(c['passed'] for c in checks.values()),comparisons=checks,
            note='Train ablation; a pass does not adopt a method or prove gains over three baselines'))
        self.status('COMPLETE_TRAIN_ABLATION')
        return 0

    def report(self):
        Round2.report(self)
        path=self.root/'report.md'
        text=path.read_text().replace('# Adaptive RAVEN 优化执行报告','# 第五轮：EOS保护单因素消融')
        text=text.replace('短测只有两条提示、128 token，仅筛选性能。调参和回归题不是独立正式证据。',
            '本轮诊断与回归均为2048输出预算；训练消融不是独立正式证据。')
        text += '\n\n## 本轮范围\n\n仅训练消融与机制诊断。旧结果只读保留；诊断组不作速度依据。'
        text += '\nEOS阈值0.01只关闭已有松弛，仍由原P/Q验收与残差采样决定token，不强制结束。'
        text += '\n本轮不会自动进入正式测试或采用候选；通过后仍需独立留出及Native/Cactus/SpecCascade同题重测。\n'
        if (self.root/'eos_diagnosis.json').exists():
            text += '\n机制统计见 `eos_diagnosis.json`。\n'
        path.write_text(text)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--reference-dir',required=True,type=Path)
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    args.prefix_cache=0;args.deep_diagnostic=False;args.target_probe=False
    args.uniform_gemm=True;args.full_bf16_accum=True;args.bnb_row_block=64
    args.rmsnorm_row_limit=255;args.target_backend='TRITON_ATTN';args.triton_2d=True
    args.lm_head_row_block=16;args.attention_fp32=True;args.canonical_tree=True
    controller=None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        controller=Round5(args)
        if args.check_only:
            controller.references();controller.status('PREFLIGHT_COMPLETE');return 0
        return controller.run()
    except GateRejected as exc:
        write_json(controller.root/'failure.json',dict(type='GateRejected',category='candidate_gate',gate=str(exc),
            error='质量或复现门槛未通过；结果保留，未采用候选',timestamp_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))
        print(f'候选门槛未通过：{exc}',flush=True);return 3
    except Exception as exc:
        if controller:
            write_json(controller.root/'failure.json',dict(type=type(exc).__name__,category='runtime_or_protocol',
                error=str(exc),timestamp_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))
            controller.status('STOPPED_FAILURE')
        raise
    finally:
        if controller:controller.stop_server()


if __name__=='__main__':raise SystemExit(main())
