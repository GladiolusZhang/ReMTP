#!/usr/bin/env python3
"""Bounded Native diagnosis, probability-cap proof, then gated train regression."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_adaptive_round2 import Round2, settings, write_json
from scripts.run_adaptive_round4 import NUMERICAL_KEYS, GateRejected
from scripts.run_adaptive_round5 import output_signature
from scripts.run_adaptive_round6 import configs, equal_trace
from remtp.optimization_metrics import paired_gate, summarize, clustered_intervals

NATIVE_PAIRS = ((1532,42),(1532,43),(1532,44),(2950,43),(6009,42),(3213,43))
CHECKPOINTS = (128,512,1024,1792,2000)
CAP = 2.0


def load_legacy(reference):
    path = reference/'frozen_code/remtp/dynamic_mtp_tree.py'
    spec = importlib.util.spec_from_file_location('remtp._probability_cap_legacy', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trace_semantics(trace, cap):
    """Check the actual H used by both acceptance and rejected-token draws."""
    import torch
    from remtp.fixed6_microtree import residual_distribution
    pending = None
    for event in trace['events']:
        if event['kind'] == 'accept_uniform':
            p, q, h = (event[k] for k in ('p','q','h'))
            y = event['token_id']
            assert bool(torch.isfinite(h).all()) and bool((h >= 0).all())
            assert abs(float(h.sum())-1) <= 2e-6
            bound = cap*float(p[y])
            assert float(h[y]) <= bound + max(1e-12, bound*3e-6)
            alpha = min(1.0, float(h[y])/max(float(q[y]),1e-30))
            assert event['keep'] == (event['uniform'] <= alpha)
            pending = None if event['keep'] else residual_distribution(h,q)
        elif event['kind'] in ('residual_correction','continuation_correction'):
            # CPU reductions of copied GPU rows can round differently.
            assert pending is not None and torch.allclose(pending,event['probs'],rtol=3e-6,atol=1e-8)
            pending = None


def cuda_gate(reference, output):
    import torch
    from remtp.dynamic_mtp_tree import DynamicTreeTopology, residual_hit_tree_verify
    torch.set_num_threads(1)
    frames = sorted((reference/'repair_smoke/old_eos01/decision_trace').glob('*/verify_*.pt'))
    if len(frames) != 8 or not torch.cuda.is_available():
        raise ValueError('Eight real repaired B frames and CUDA required')
    attempt=output.with_suffix('.attempt.json')
    if attempt.exists() or output.exists():
        raise ValueError('CUDA budget already attempted; no automatic replay')
    write_json(attempt,dict(reserved_calls=32,automatic_retries=0,reference=str(reference)))
    legacy = load_legacy(reference)
    checks = []
    for path in frames:
        frame = torch.load(path,map_location='cpu',weights_only=False)
        topology = DynamicTreeTopology.from_paths(frame['topology'])
        old_topology = legacy.DynamicTreeTopology.from_paths(frame['topology'])
        values = [frame[k].cuda() for k in ('ids','draft','target')]
        old_config = legacy.DynamicTreeConfig(**{k:v for k,v in asdict(frame['config']).items()
                                                if k != 'relaxation_max_probability_ratio'})
        outputs = []; traces = []; states = []
        for mode in ('legacy','disabled','cap','cap_lazy'):
            rng = torch.Generator(device=frame['rng_device']); rng.set_state(frame['rng_before'])
            trace = {}
            func = legacy.residual_hit_tree_verify if mode == 'legacy' else residual_hit_tree_verify
            config = old_config if mode == 'legacy' else replace(frame['config'],
                        relaxation_max_probability_ratio=CAP if mode.startswith('cap') else 0.0)
            result = func(old_topology if mode == 'legacy' else topology, *values, rng,
                config=config, protected_node_indices=frame['protected_node_indices'],
                lazy_residual=mode=='cap_lazy', decision_trace=trace)
            outputs.append(asdict(result)); traces.append(trace); states.append(rng.get_state())
            if mode.startswith('cap'): trace_semantics(trace,CAP)
        def same(i,j):
            return (outputs[i] == outputs[j] and torch.equal(states[i],states[j])
                and len(traces[i]['events']) == len(traces[j]['events'])
                and all(equal_trace(a,b) for a,b in zip(traces[i]['events'],traces[j]['events'])))
        passed = (same(0,1) and outputs[1] == frame['result']
                  and torch.equal(states[1],frame['rng_after']) and same(2,3))
        checks.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           passed=bool(passed),cap_semantics=True))
        write_json(output,dict(passed=all(c['passed'] for c in checks),complete=len(checks)==8,
            cuda_verifier_calls=4*len(checks),checks=checks,
            scope='8 saved real B frames, old/off equality and cap H/residual/RNG; not Target/KV/long-context proof'))
        if not passed: raise ValueError('New CUDA conditional gate failed: '+str(path))


class ProbabilityCap(Round2):
    def __init__(self,args):
        super().__init__(args)
        self.reference = args.reference_dir.resolve()
        self.protocol = json.loads((self.root/'protocol.json').read_text())
        old = json.loads((self.reference/'protocol.json').read_text())
        for key in (*NUMERICAL_KEYS,'train_sha256','test_sha256','primary_prompt'):
            if self.protocol[key] != old[key]: raise ValueError('Setting mismatch: '+key)
        historical = []
        for path in sorted((self.reference/'regression').glob('B_seed*/requests.jsonl')):
            historical.extend(json.loads(s) for s in path.read_text().splitlines())
        if len(historical) != 72: raise ValueError('Complete repaired B reference required')
        self.historical = {(r['question_id'],r['seed']):r for r in historical}
        self.baseline = {**configs()['old_eos01'],'probability_ratio_cap':0.0}
        self.candidate = {**self.baseline,'probability_ratio_cap':CAP}
        write_json(self.root/'cap_execution_protocol.json',dict(
            source_sha256=self.protocol['source_sha256'],historical_reference=str(self.reference),
            candidate=self.candidate,native_pairs=NATIVE_PAIRS,B_late_pair=[1532,43],
            late_checkpoints=CHECKPOINTS,late_max_frames=5,late_scope='verifier only, no paired proposal/KV claim',
            natural_calls=7,remaining_old_natural_after_unit=0,new_conditional_cuda_calls=32,
            train_requests=144,train_ids=[r['question_id'] for r in self.regression],train_seeds=[42,43,44],
            order={'42':['B','cap2'],'43':['cap2','B'],'44':['B','cap2']},
            warmup='8 tokens per server, <=8 services; diagnostic B <=3 same-prefix head probes x2 LMHead calls',
            retries=0,restart='no',heldout=False,formal=False,automatic_adoption=False,
            local_cap='H(y)<=2P(y); acceptance and residual both use rebuilt H',
            quality_reference=dict(correct=67,stopped_correct=61,truncated=11),
            truncation_improvement='<=10 AND lower than fresh B; no per-seed quality decline/truncation increase',
            source_policy='old numerical gates not inherited; actual unchanged numeric sources checked and scope limited'))
        # New controller/observer are allowed; actual Target/MTP numeric sources must match.
        unchanged = ['fixed6_vllm.py','fixed6_microtree.py','target_attention_fp32.py','canonical_tree_attention.py',
                     'target_numerics.py','target_norm.py','target_head.py','jetson_bf16_draft.py',
                     'probabilistic_mtp.py','fastmtp_verified_worker.py','generation_guard.py']
        rows = []
        for name in unchanged:
            a = ROOT/'remtp'/name; b = self.reference/'frozen_code/remtp'/name
            rows.append(dict(file=name,identical=a.read_bytes()==b.read_bytes(),sha256=hashlib.sha256(a.read_bytes()).hexdigest()))
        if not all(r['identical'] for r in rows): raise ValueError('Unregistered numeric runtime source change')
        write_json(self.root/'unchanged_numeric_sources.json',dict(checks=rows,
                    scope='Source identity only, not a numerical gate'))

    def start_server(self,variant,directory,profile=False):
        super().start_server(variant,directory,profile)
        log = (directory/'server.log').read_text()
        if '[ReMTP][HeadBatch] input_rows=1' not in log:
            raise ValueError('Effective head row setting not confirmed')
        if self.current['method'] == 'native':
            expected = '[ReMTP][FastMTPVerified] method=native worker=active'
        else:
            expected = 'relax_max_probability_ratio='+format(self.current['probability_ratio_cap'],'g')+' '
        if expected not in log: raise ValueError('Effective worker/cap not confirmed: '+expected)
        if self.current['method']=='native' and '[ReMTP][ProbMTP] full proposal distributions enabled' not in log:
            raise ValueError('Native full-Q adapter not confirmed')

    def gate(self,name,value):
        write_json(self.root/(name+'_gate.json'),{**value,'source_sha256':self.protocol['source_sha256']})
        self.report()
        if not value['passed']:
            self.status('STOPPED_'+name.upper()); raise GateRejected(name)

    def run(self):
        import torch
        native = self.requests('diagnostics','native',settings(method='native',head=1),
                               [(self.by_id[q],s) for q,s in NATIVE_PAIRS])
        write_json(self.root/'native_pairs.json',dict(pairs=[dict(
            question_id=r['question_id'],seed=r['seed'],native=r,
            historical_B=self.historical[r['question_id'],r['seed']]) for r in native],
            scope='Same question/seed/prompt/budget, different legitimate RNG path; no formal speed/causality claim'))
        self.args.decision_trace=True; self.args.decision_trace_checkpoints=CHECKPOINTS
        records = self.requests('diagnostics','B_late',self.baseline,[(self.by_id[1532],43)],diagnostic=True)
        self.args.decision_trace=False; self.args.decision_trace_checkpoints=()
        frames = sorted((self.root/'diagnostics/B_late/decision_trace').glob('*/verify_*.pt'))
        finite = []; lengths = []
        for path in frames:
            f = torch.load(path,map_location='cpu',weights_only=False)
            finite.append(bool(torch.isfinite(f['target']).all() and torch.isfinite(f['draft']).all()
                and all(bool(torch.isfinite(e['h']).all()) for e in f['events'] if e['kind']=='accept_uniform')))
            lengths.append(len(f['history']))
        reference = self.historical[1532,43+1532*10]
        self.gate('late_B_replay',dict(passed=output_signature(records[0])==output_signature(reference)
                and len(frames)==5 and all(finite) and all(a<=b<a+12 for a,b in zip(CHECKPOINTS,lengths)),
                exact_output=output_signature(records[0])==output_signature(reference),frames=len(frames),
                finite=finite,generated_lengths=lengths,scope='One complete B output + five late raw P/Q frames'))
        self.status('NEW_CAP_CUDA_GATE')
        subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_raven_probability_cap.py'),
            '--cuda-reference',str(self.reference),'--cuda-output',str(self.root/'cap_cuda.json')],cwd=ROOT,check=True)
        self.gate('new_cap_conditional',json.loads((self.root/'cap_cuda.json').read_text()))
        baseline=[]; candidate=[]
        for seed in (42,43,44):
            order=('B','cap2') if seed != 43 else ('cap2','B')
            block={}
            for name in order:
                block[name]=self.blocks('regression',name,self.baseline if name=='B' else self.candidate,self.regression,[seed])
                (baseline if name=='B' else candidate).extend(block[name])
                if name=='B':
                    previous=[self.historical[r['question_id'],r['seed']] for r in block['B']]
                    self.gate(f'train_seed{seed}_B_replay',dict(passed=all(output_signature(a)==output_signature(b)
                            for a,b in zip(previous,block['B'])),pairs=len(previous)))
            self.gate(f'train_seed{seed}_quality',paired_gate(block['B'],block['cap2'],require_speed=False))
        self.gate('train_quality',paired_gate(baseline,candidate,require_speed=False))
        a,b=summarize(baseline),summarize(candidate)
        intervals=clustered_intervals(baseline,candidate)
        speed=paired_gate(baseline,candidate,require_speed=True)
        if intervals['throughput_ratio_95ci'][0]<=1:
            speed['failures'].append('all:throughput_ci')
            speed['passed']=False
        write_json(self.root/'performance_summary.json',dict(baseline=a,candidate=b,speed_gate=speed,
            intervals=intervals,scope='24 train questions x3 seeds, no independent heldout or three-baseline claim'))
        self.gate('train_truncation',dict(passed=b['truncated']<=10 and b['truncated']<a['truncated'],
                                        baseline=a['truncated'],candidate=b['truncated']))
        self.status('COMPLETE_CAP_TRAIN_REGRESSION')
        return 0

    def report(self):
        lines=['# Native截断诊断与Adaptive RAVEN概率上限验证','',
               '固定H(y)/P(y)上限2；默认关闭，保持原提示、评分、2048预算及模型/公共数值环境。',
               '6条Native诊断+1条B后段观测→32次新CUDA条件验收→最多144条同源B/cap2训练对照。',
               '0自动重试，负数值/质量门槛停止扩展；不自动采用，不执行留出或正式三基准比较。','']
        if (self.root/'status.json').exists():
            state=json.loads((self.root/'status.json').read_text())
            lines.extend([f"阶段：{state['phase']}；当前组：{state['variant']} {state['completed']}/{state['total']}；已保存{state['all_completed']}条。",''])
        lines.extend(['| 阶段/配置 | 请求 | 判对 | 正常结束且判对 | 截断 | 吞吐token/s | 平均秒 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for path in sorted(self.root.glob('*/*/requests.jsonl')):
            rows=[json.loads(s) for s in path.read_text().splitlines()]; x=summarize(rows)
            lines.append(f"| {path.parent.parent.name}/{path.parent.name} | {len(rows)} | {x['correct']} | {x['stopped_correct']} | {x['truncated']} | {x['e2e_tok_s']:.4f} | {x['mean_seconds']:.1f} |")
        lines+=['','诊断含观测耗时，排除出速度结论。Native为仓库MTP链基线，合法随机路径不同，不要求与RAVEN逐字一致。','']
        for path in sorted(self.root.glob('*_gate.json')):
            gate=json.loads(path.read_text());lines.append(f"- {path.stem}：{'通过' if gate['passed'] else '未通过'}")
        if (self.root/'performance_summary.json').exists():
            x=json.loads((self.root/'performance_summary.json').read_text())
            lines+=['',f"训练速度门槛：{'通过' if x['speed_gate']['passed'] else '未通过'}；吞吐比95%区间：{x['intervals']['throughput_ratio_95ci']}。"]
        if (self.root/'failure.json').exists(): lines+=['','## 停止原因','',(self.root/'failure.json').read_text()]
        lines+=['','当前有限条件验收不覆盖完整Target/KV/长上下文；训练通过也不等于论文三基准目标达成。']
        path=self.root/'report.md';temporary=path.with_suffix('.md.tmp')
        temporary.write_text('\n'.join(lines)+'\n');temporary.replace(path)


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir');p.add_argument('--reference-dir',type=Path)
    p.add_argument('--port',type=int,default=8000);p.add_argument('--check-only',action='store_true')
    p.add_argument('--cuda-reference',type=Path);p.add_argument('--cuda-output',type=Path)
    args=p.parse_args()
    args.prefix_cache=0;args.deep_diagnostic=False;args.target_probe=False
    args.decision_trace=False;args.mtp_operator_audit=False;args.decision_trace_checkpoints=()
    args.uniform_gemm=True;args.full_bf16_accum=True;args.bnb_row_block=64
    args.rmsnorm_row_limit=255;args.target_backend='TRITON_ATTN';args.triton_2d=True
    args.lm_head_row_block=16;args.attention_fp32=True;args.canonical_tree=True
    return args


def main():
    args=arguments()
    if args.cuda_reference:
        cuda_gate(args.cuda_reference,args.cuda_output);return 0
    if not args.run_dir or not args.reference_dir: raise ValueError('Run and reference directories required')
    controller=None
    def interrupted(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        controller=ProbabilityCap(args)
        if args.check_only: return 0
        return controller.run()
    except BaseException as exc:
        if controller:
            write_json(controller.root/'failure.json',dict(type=type(exc).__name__,error=str(exc)))
            controller.status('STOPPED_FAILURE')
        if isinstance(exc,GateRejected): return 3
        raise
    finally:
        if controller: controller.stop_server()


if __name__=='__main__': raise SystemExit(main())
