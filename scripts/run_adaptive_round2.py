#!/usr/bin/env python3
"""Finite autonomous optimization: diagnose, ablate, tune, gate, heldout, test.

Run only from a frozen source copy. Quality failures stop without adopting a
candidate. Every HTTP request is saved before advancing; no online gold access
is provided to the serving worker. All primary quality requests use 2048 tokens.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remtp.benchmark import chat_completion, fetch_metrics, metric_delta
from remtp.gsm8k_benchmark import PROMPT_TEMPLATE, benchmark_messages, extract_model_answer, load_gsm8k
from remtp.optimization_metrics import summarize, paired_gate, clustered_intervals
from scripts.run_adaptive_optimization import Controller, write_json


def settings(method="dynamic_tree", depth=3, head=3, anchor=False, topk=False,
             delta=1.0, protect=False, mix=0.5, adaptive=True):
    return dict(method=method, depth=depth, nodes=depth*3, head=head, anchor=anchor,
                topk=topk, delta=delta, protect=protect, mix=mix, adaptive=adaptive)


class Round2(Controller):
    def __init__(self, args):
        self.args = args
        if getattr(args,'bnb_row_block',0) and not (getattr(args,'uniform_gemm',False) and getattr(args,'full_bf16_accum',False)):
            raise ValueError('Fixed BNB row blocks require uniform GEMM and full BF16 accumulation')
        if getattr(args,'triton_2d',False) and getattr(args,'target_backend','')!='TRITON_ATTN':
            raise ValueError('Unified 2D reference requires TRITON_ATTN')
        self.root = Path(args.run_dir).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root/".lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.server = None; self.base_url = f"http://127.0.0.1:{args.port}"
        self.current = settings(); self.diagnostic = False
        self.jobs = json.loads((self.root/"jobs.json").read_text()) if (self.root/"jobs.json").exists() else {}
        self.train = load_gsm8k(ROOT/"data/gsm8k/train.jsonl")
        self.eos_ids=[]
        for config_name in ('config.json','generation_config.json'):
            value=json.loads((ROOT/'models/FastMTP'/config_name).read_text()).get('eos_token_id',[])
            for token in ([value] if isinstance(value,int) else value or []):
                if token not in self.eos_ids:self.eos_ids.append(token)
        self.by_id = {r["question_id"]:r for r in self.train}
        old = json.loads((self.root/"v4_evidence/protocol.json").read_text())
        self.regression = [self.by_id[i] for i in old["calibration_ids"]]
        excluded = set(old["calibration_ids"]+[1,2])
        self.heldout = sorted(random.Random(20261002).sample([r for r in self.train if r["question_id"] not in excluded],24),key=lambda r:r["question_id"])
        # Fixed before any candidate result: four regressions plus two ordinary controls.
        self.tuning = [(self.by_id[i],s) for i,s in [(2629,42),(2950,42),(7300,42),(7311,43),(308,42),(641,43)]]
        self.variants = {
            "old":settings(head=1), "topk":settings(head=1,topk=True),
            "head":settings(), "head_topk":settings(topk=True),
            "anchor3":settings(anchor=True,topk=True),
            "anchor4":settings(depth=4,anchor=True,topk=True),
        }
        files = sorted([*ROOT.glob("remtp/*.py"), *ROOT.glob("scripts/*"), *ROOT.glob("configs/*")])
        digest=hashlib.sha256()
        for path in files:
            if path.is_file(): digest.update(str(path.relative_to(ROOT)).encode()+path.read_bytes())
        protocol = dict(source_root=str(ROOT), source_sha256=digest.hexdigest(), variants=self.variants,
                        model="TencentBAC/FastMTP", target="bitsandbytes FP4", draft="BF16",
                        temperature=0.6, max_model_len=4096, quality_max_tokens=2048, kv_cache="1024M",
                        request_local_branch_rng=True, regression_ids=old["calibration_ids"],
                        prefix_caching=getattr(args,'prefix_cache',0),
                        deep_diagnostic=getattr(args,'deep_diagnostic',False),
                        target_probe=dict(enabled=getattr(args,'target_probe',False),
                            state_audit=getattr(args,'target_state_audit',False),
                            min_generated=getattr(args,'target_probe_min_generated',0),
                            max_generated=getattr(args,'target_probe_max_generated',2147483647),
                            max_frames=getattr(args,'target_probe_max_frames',32),
                            checkpoints=getattr(args,'target_probe_checkpoints',()),
                            reference_echo=getattr(args,'target_state_reference_echo',False),
                            row_position=getattr(args,'target_state_row_position',-1)),
                        target_uniform_gemm=getattr(args,'uniform_gemm',False),
                        target_full_bf16_accum=getattr(args,'full_bf16_accum',False),
                        target_bnb_row_block=getattr(args,'bnb_row_block',0),
                        target_rmsnorm_row_limit=getattr(args,'rmsnorm_row_limit',0),
                        target_lm_head_row_block=getattr(args,'lm_head_row_block',0),
                        target_reference_backend=getattr(args,'target_backend',''),
                        baseline_attention_backend='TRITON_ATTN' if getattr(args,'target_backend','')=='TRITON_ATTN' else 'default',
                        target_triton_2d=getattr(args,'triton_2d',False),
                        target_attention_fp32=getattr(args,'attention_fp32',False),
                        canonical_tree_attention=getattr(args,'canonical_tree',False),
                        heldout_ids=[r["question_id"] for r in self.heldout], seeds=[42,43,44],
                        tune_pairs=[(r["question_id"],s) for r,s in self.tuning],
                        repeat_guard=dict(min_period=4,max_period=64,repetitions=3), eos_threshold=0.5,eos_ids=self.eos_ids,
                        packages={n:importlib.metadata.version(n) for n in ("torch","vllm","bitsandbytes","transformers")},
                        train_sha256=hashlib.sha256((ROOT/"data/gsm8k/train.jsonl").read_bytes()).hexdigest(),
                        test_sha256=hashlib.sha256((ROOT/"data/gsm8k/test.jsonl").read_bytes()).hexdigest(),
                        model_config_sha256=hashlib.sha256((ROOT/"models/FastMTP/config.json").read_bytes()).hexdigest(),
                        primary_prompt=PROMPT_TEMPLATE, no_think_template_sha256=hashlib.sha256((ROOT/"configs/fastmtp_no_think_chat_template.jinja").read_bytes()).hexdigest(),
                        max_final_candidates=2, bootstrap_draws=2000)
        existing=self.root/"protocol.json"
        if existing.exists() and json.loads(existing.read_text())!=json.loads(json.dumps(protocol)):
            raise ValueError("Frozen code/protocol mismatch; use a new directory")
        write_json(existing,protocol)
        self.ensure_budget(self.regression+self.heldout+self.train[:2])

    def ensure_budget(self,rows):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(ROOT/"models/FastMTP",trust_remote_code=True,local_files_only=True)
        template=(ROOT/"configs/fastmtp_no_think_chat_template.jinja").read_text()
        length=max(len(tok.apply_chat_template(benchmark_messages(PROMPT_TEMPLATE.format(question=r["question"]),empty_system_prompt=True),
                          tokenize=True,add_generation_prompt=True,chat_template=template,enable_thinking=False)) for r in rows)
        if length+2048+12>4096: raise ValueError("Prompt + output + draft exceeds context")

    def status(self,phase,variant="",completed=0,total=0):
        self.live_phase=phase
        phase_jobs=[v for v in self.jobs.values() if v["phase"]==phase]
        counts={}
        for k,v in self.jobs.items():
            p=self.root/k/"requests.jsonl"
            counts[k]=len(p.read_text().splitlines()) if p.exists() else 0
        write_json(self.root/"status.json",dict(phase=phase,variant=variant,completed=completed,total=total,
                  phase_completed=sum(counts[k] for k,v in self.jobs.items() if v["phase"]==phase),
                  phase_total=sum(v["total"] for v in phase_jobs),all_completed=sum(counts.values()),
                  scheduled_total=sum(v["total"] for v in self.jobs.values()),
                  updated_utc=time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime()),controller_pid=os.getpid(),
                  note="后续阶段仅在门槛通过后安排；scheduled_total 是已安排请求数，不是最终总数"))
        print(f"[{phase}] {variant} {completed}/{total}",flush=True)
        self.report()

    def start_server(self,variant,directory,profile=False):
        self.stop_server()
        # TCP check also catches an unrelated service without a /health route.
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1",self.args.port))==0: raise ValueError("Port occupied; refusing to stop unrelated process")
        x=self.current
        memory=Path('/proc/meminfo').read_text()
        power=subprocess.run(['nvpmodel','-q'],capture_output=True,text=True).stdout.strip()
        temperatures={}
        for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
            try:temperatures[(p.parent/'type').read_text().strip()]=int(p.read_text())/1000
            except (OSError,ValueError):pass
        write_json(directory/'device_before.json',dict(power=power,thermal_celsius=temperatures,meminfo=memory,
                        timestamp_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))
        env={k:v for k,v in os.environ.items() if not k.startswith(("REMTP_","VLLM_"))}
        env.update(dict(METHOD=x["method"],PORT=str(self.args.port),MAX_MODEL_LEN="4096",KV_CACHE_MEMORY_BYTES="1024M",
                   ENABLE_PREFIX_CACHING=str(x.get('prefix_cache',getattr(self.args,'prefix_cache',0))),
                   QUANTIZATION="bitsandbytes",LOAD_FORMAT="bitsandbytes",MTP_TOKENS=str(x["depth"]),
                   REMTP_HEAD_BATCH_ROWS=str(x["head"]),REMTP_TREE_AUDIT_JSONL="",REMTP_TREE_TRACE="0",
                   REMTP_DYNAMIC_TREE_ADAPTIVE_RELAXATION=str(int(x["adaptive"])),
                   REMTP_DYNAMIC_TREE_PRUNE_ANCHOR_DESCENDANTS="0",REMTP_DYNAMIC_TREE_EXACT_TOPK=str(int(x["topk"])),
                   REMTP_DYNAMIC_TREE_ANCHOR_ONLY_PROPOSAL=str(int(x["anchor"])),
                   REMTP_DYNAMIC_TREE_MAX_DEPTH=str(x["depth"]),REMTP_DYNAMIC_TREE_MAX_NODES=str(x["nodes"]),
                   REMTP_DYNAMIC_TREE_RELAX_DEPTH_WEIGHTS=','.join(map(str,x.get('relax_weights',[1.0]*x['depth']))),
                   REMTP_DYNAMIC_TREE_LAZY_RESIDUAL=str(int(x.get('lazy_residual',False))),
                   REMTP_DYNAMIC_TREE_RELAX_MAX_PROBABILITY_RATIO=str(x.get('probability_ratio_cap',0)),
                   REMTP_DYNAMIC_TREE_RELAX_RISK_POLICY=str(x.get('risk_policy','none')),
                   REMTP_DYNAMIC_TREE_CACTUS_DELTA=str(x["delta"]),REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MIX=str(x["mix"]),
                   REMTP_DYNAMIC_TREE_PROTECT_EOS=str(int(x["protect"])),REMTP_DYNAMIC_TREE_REPEAT_GUARD=str(int(x.get("repeat_guard", x["protect"]))),
                   REMTP_DYNAMIC_TREE_EOS_THRESHOLD=str(x.get("eos_threshold", 0.5)),REMTP_REQUEST_LOCAL_BRANCH_RNG="1" if x["method"]=="dynamic_tree" else "0",
                   REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS=','.join(map(str,self.eos_ids)),
                   REMTP_PREFIX_DIAGNOSTIC_JSONL=str(directory/"prefix_diagnostic.jsonl") if self.diagnostic else "",
                   REMTP_STATE_DIAGNOSTIC_DIR=str(directory/'state_diagnostic') if getattr(self.args,'deep_diagnostic',False) else '',
                   REMTP_DECISION_TRACE_DIR=str(directory/'decision_trace') if (self.diagnostic or getattr(self.args,'decision_trace_only',False)) and getattr(self.args,'decision_trace',False) else '',
                   REMTP_DECISION_TRACE_CHECKPOINTS=','.join(map(str,getattr(self.args,'decision_trace_checkpoints',()))),
                   REMTP_DECISION_TRACE_EARLY_PROPOSALS=str(getattr(self.args,'decision_trace_early_proposals',0)),
                   REMTP_DECISION_TRACE_SUMMARY_JSONL=str(directory/'decision_summary.jsonl') if getattr(self.args,'decision_trace_only',False) and getattr(self.args,'decision_trace',False) else '',
                   REMTP_DECISION_TRACE_SUMMARY_MAX_ROUNDS=str(getattr(self.args,'decision_summary_max_rounds',2048)),
                   REMTP_DECISION_TRACE_SUMMARY_MAX_REQUESTS=str(getattr(self.args,'decision_summary_max_requests',6)),
                   REMTP_MTP_OPERATOR_AUDIT=str(int(getattr(self.args,'mtp_operator_audit',False))),
                   REMTP_TARGET_DIAGNOSTIC_DIR=str(directory/'target_probe') if getattr(self.args,'target_probe',False) else '',
                   REMTP_TARGET_STATE_AUDIT=str(int(getattr(self.args,'target_state_audit',False))),
                   REMTP_TARGET_PROBE_MIN_GENERATED=str(getattr(self.args,'target_probe_min_generated',0)),
                   REMTP_TARGET_PROBE_MIN_PROMPT_TOKENS=str(getattr(self.args,'target_probe_min_prompt_tokens',0)),
                   REMTP_TARGET_PROBE_MAX_GENERATED=str(getattr(self.args,'target_probe_max_generated',2147483647)),
                   REMTP_TARGET_PROBE_MAX_FRAMES=str(getattr(self.args,'target_probe_max_frames',32)),
                   REMTP_TARGET_PROBE_CHECKPOINTS=','.join(map(str,getattr(self.args,'target_probe_checkpoints',()))),
                   REMTP_TARGET_PROBE_PREFIX_MANIFEST=str(getattr(self.args,'target_probe_prefix_manifest','')),
                   REMTP_TARGET_PROBE_SHAPE_ROWS=','.join(map(str,getattr(self.args,'target_probe_shape_rows',(1,10,13)))),
                   REMTP_TARGET_STATE_REFERENCE_ECHO=str(int(getattr(self.args,'target_state_reference_echo',False))),
                   REMTP_TARGET_STATE_ROW_POSITION=str(getattr(self.args,'target_state_row_position',-1)),
                   REMTP_TARGET_BNB_UNIFORM_GEMM=str(int(getattr(self.args,'uniform_gemm',False))),
                   REMTP_TARGET_BF16_FULL_ACCUM=str(int(getattr(self.args,'full_bf16_accum',False))),
                   REMTP_TARGET_BNB_ROW_BLOCK=str(getattr(self.args,'bnb_row_block',0)),
                   REMTP_TARGET_RMSNORM_ROW_LIMIT=str(getattr(self.args,'rmsnorm_row_limit',0)),
                   REMTP_TARGET_LM_HEAD_ROW_BLOCK=str(getattr(self.args,'lm_head_row_block',0)),
                   TARGET_ATTENTION_BACKEND=getattr(self.args,'target_backend',''),
                   REMTP_TARGET_TRITON_2D=str(int(getattr(self.args,'triton_2d',False))),
                   REMTP_TARGET_ATTENTION_FP32=str(int(getattr(self.args,'attention_fp32',False))),
                   REMTP_CANONICAL_TREE_ATTENTION=str(int(getattr(self.args,'canonical_tree',False))),
                   REMTP_RUNTIME_PROFILE_JSONL=str(directory/'profile.jsonl') if profile else '',
                   REMTP_PROFILE_MAX_ROUNDS=str(getattr(self.args,'profile_max_rounds',32)),
                   REMTP_PROFILE_MAX_ROUNDS_PER_REQUEST=str(getattr(self.args,'profile_per_request',16)),
                   REMTP_PROFILE_SKIP_REQUESTS='1',
                   REMTP_CACTUS_DELTA=str(x.get('cactus_delta',1.0)),
                   REMTP_CASCADE_RULE=x.get('cascade_rule','token_v3'),
                   REMTP_CASCADE_ALPHA=str(x.get('cascade_alpha',0.5)),
                   PYTHONPATH=str(ROOT),PYTHONUNBUFFERED="1"))
        write_json(directory/'effective_env.json',{k:v for k,v in env.items()
            if k.startswith(('REMTP_','VLLM_')) or k in ('METHOD','PORT','MAX_MODEL_LEN',
                'KV_CACHE_MEMORY_BYTES','QUANTIZATION','LOAD_FORMAT','MTP_TOKENS',
                'ENABLE_PREFIX_CACHING','TARGET_ATTENTION_BACKEND','PYTHONPATH')})
        with (directory/"server.log").open("a") as log:
            self.server=subprocess.Popen([str(ROOT/"scripts/serve_fastmtp_jetson.sh")],cwd=ROOT,env=env,
                                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(240):
            if self.server.poll() is not None: raise RuntimeError(f"Server exited; inspect {directory/'server.log'}")
            try:
                with urllib.request.urlopen(self.base_url+"/health",timeout=2) as response:
                    if response.status==200: break
            except Exception: time.sleep(2)
        else: raise RuntimeError("Server health timed out")
        self.warmup(directory)

    def warmup(self,directory):
        return chat_completion(self.base_url,"TencentBAC/FastMTP",benchmark_messages("Calculate 2+2.",empty_system_prompt=True),0.6,1234,8,1800)

    def requests(self,phase,name,config,pairs,limit=2048,diagnostic=False):
        directory=self.root/phase/name;directory.mkdir(parents=True,exist_ok=True)
        key=f"{phase}/{name}"
        job=dict(phase=phase,total=len(pairs),settings=config,limit=limit,diagnostic=diagnostic,
                 pairs=[(r["question_id"],s+r["question_id"]*10) for r,s in pairs])
        job=json.loads(json.dumps(job))
        if key in self.jobs and self.jobs[key]!=job: raise ValueError("Job protocol changed")
        self.jobs[key]=job;write_json(self.root/"jobs.json",self.jobs)
        path=directory/"requests.jsonl"
        # Repair only an incomplete last line, never an invalid completed record.
        if path.exists() and path.read_bytes() and not path.read_bytes().endswith(b"\n"):
            data=path.read_bytes();path.with_suffix(".partial").write_bytes(data[data.rfind(b"\n")+1:])
            path.write_bytes(data[:data.rfind(b"\n")+1])
        records=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        expected=job["pairs"]
        if [[r["question_id"],r["seed"]] for r in records]!=expected[:len(records)]: raise ValueError("Checkpoint order mismatch")
        self.status(phase,name,len(records),len(pairs))
        if len(records)<len(pairs):
            self.current=config;self.diagnostic=diagnostic;self.start_server(name,directory)
            for row,seed in pairs[len(records):]:
                seed+=row["question_id"]*10;before=fetch_metrics(self.base_url,30);started=time.perf_counter()
                response=chat_completion(self.base_url,"TencentBAC/FastMTP",benchmark_messages(PROMPT_TEMPLATE.format(question=row["question"]),empty_system_prompt=True),0.6,seed,limit,1800)
                elapsed=time.perf_counter()-started;after=fetch_metrics(self.base_url,30);choice=response["choices"][0]
                text=choice["message"]["content"] or "";pred,source=extract_model_answer(text)
                record=dict(question_id=row["question_id"],seed=seed,gold_answer=row["gold_answer"],predicted_answer=pred,
                            answer_source=source,correct=pred==row["gold_answer"],finish_reason=choice["finish_reason"],
                            output=text,output_tokens=response["usage"]["completion_tokens"],client_seconds=elapsed,
                            metrics=metric_delta(before,after))
                with path.open("a") as f:
                    f.write(json.dumps(record,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
                records.append(record);self.status(phase,name,len(records),len(pairs))
            self.stop_server()
        write_json(directory/"summary.json",summarize(records));self.report()
        return records

    def report(self):
        lines=["# Adaptive RAVEN 优化执行报告","","状态与结果由后台控制器更新；未完成阶段不构成成功结论。",
               "所有主质量请求使用 2048 token、temperature=0.6；模型/环境保持不变。",
               "本轮显式独立 head 开关和请求局部 branch RNG；旧配置按新协议重跑，旧日志只作诊断。",""]
        lines += ["公共数值配置（全部适用方法相同）："
                  f"uniform FP4 GEMM={getattr(self.args,'uniform_gemm',False)}；"
                  f"BF16完整归约={getattr(self.args,'full_bf16_accum',False)}；"
                  f"固定行块={getattr(self.args,'bnb_row_block',0)}；"
                  f"RMSNorm单次行上限={getattr(self.args,'rmsnorm_row_limit',0)}；"
                  f"BF16 LMHead行块={getattr(self.args,'lm_head_row_block',0)}；"
                  f"独立Target/链基线后端={getattr(self.args,'target_backend','') or '默认'}；"
                  f"统一2D={getattr(self.args,'triton_2d',False)}；"
                  f"P×V FP32={getattr(self.args,'attention_fp32',False)}；"
                  f"树祖先读取顺序统一={getattr(self.args,'canonical_tree',False)}；"
                  f"跨请求前缀缓存={getattr(self.args,'prefix_cache',0)}。",""]
        if (self.root/"status.json").exists():
            x=json.loads((self.root/"status.json").read_text())
            lines += [f"当前阶段：`{x['phase']}`；当前组 `{x['variant']}`：{x['completed']}/{x['total']}；已完成请求 {x['all_completed']}。",""]
        for phase in ("diagnostics","ablation","tuning","baseline_selection","regression","heldout","formal","supplement"):
            summaries=sorted((self.root/phase).glob("*/summary.json")) if (self.root/phase).exists() else []
            if not summaries: continue
            title=dict(diagnostics='诊断',ablation='消融',tuning='训练集选参',baseline_selection='基线配置选择',
                       regression='回归',heldout='独立留出',formal='正式测试',supplement='补充消融')[phase]
            lines += [f"## {title}","","| 配置 | 请求 | 原判对 | 正常结束且判对 | 截断 | E2E tok/s | 平均秒 | p95 秒 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for p in summaries:
                s=json.loads(p.read_text());lines.append(f"| {p.parent.name} | {s['requests']} | {s['correct']} | {s['stopped_correct']} | {s['truncated']} | {s['e2e_tok_s']:.3f} | {s['mean_seconds']:.1f} | {s['p95_seconds']:.1f} |")
            lines.append("")
        for p in sorted(self.root.glob("*_gate.json")):
            gate=json.loads(p.read_text());lines += [f"## {p.stem}","",f"结论：{'通过' if gate.get('passed') else '未通过'}；详情见 `{p.name}`。",""]
        if (self.root/"diagnosis.json").exists():
            d=json.loads((self.root/"diagnosis.json").read_text())
            lines += ["## 诊断", "", f"真实 head 同前缀探针：{d['head_probes']} 条，最大概率 TV {d['max_head_tv']:.6g}。",
                      f"已访问主路径中，松弛降低 EOS 质量的记录：{d['eos_suppressed_rows']}；重复前缀轮次：{d['repeated_rounds']}。",
                      "以上说明机制/数值影响，不单独证明截断的因果原因。", ""]
        lines += ["## 结果解释","","短测只有两条提示、128 token，仅筛选性能。调参和回归题不是独立正式证据。",
                  "只有留出和独立 test 全部完成才评价最终候选；点估计不下降不能证明真实正确率绝对相同。",
                  "公共 head 加速与树深度分别报告；正式区间按题聚类、包含全部种子。",""]
        temporary=self.root/"report.md.tmp";temporary.write_text("\n".join(lines));temporary.replace(self.root/"report.md")

    def diagnose(self):
        self.requests("diagnostics","old",self.variants["old"],self.tuning[:4],diagnostic=True)
        # Replay the same request after different preceding requests. Local RNG
        # must make its output invariant to server restart/order.
        repeat=self.requests("diagnostics","old_repeat",self.variants["old"],[self.tuning[3]],diagnostic=True)
        first=json.loads((self.root/"diagnostics/old/requests.jsonl").read_text().splitlines()[3])
        same=all(repeat[0][key]==first[key] for key in ('output','output_tokens','finish_reason'))
        write_json(self.root/'reproduction_gate.json',dict(passed=same,question_id=first['question_id'],
                   seed=first['seed'],max_tokens=2048,prefix_caching=getattr(self.args,'prefix_cache',1),
                   output_tokens=[first['output_tokens'],repeat[0]['output_tokens']],
                   finish_reason=[first['finish_reason'],repeat[0]['finish_reason']],
                   output_sha256=[hashlib.sha256(r['output'].encode()).hexdigest() for r in (first,repeat[0])],
                   note='同配置、同题、同种子、完整2048输出预算，独立服务重启；不是质量提升证据'))
        if not same: raise ValueError("Same configuration/seed replay differs; investigate before tuning")
        self.requests("diagnostics","anchor4",self.variants["anchor4"],self.tuning[:4],diagnostic=True)
        diagnostics=[]
        # Stable group/line order; later diagnostic directories cannot change
        # the selected replay cases on resume.
        for name in ('old','old_repeat','anchor4'):
            p=self.root/'diagnostics'/name/'prefix_diagnostic.jsonl'
            diagnostics.extend({**json.loads(s),'source_file':str(p)} for s in p.read_text().splitlines())
        heads=[r for r in diagnostics if r["kind"]=="same_prefix_head"]
        verify=[r for r in diagnostics if r["kind"]=="verification"]
        diagnosis=dict(head_probes=len(heads),max_head_tv=max((r["probability_tv"] for r in heads),default=0),
                       eos_suppressed_rows=sum(n["verify_eos_probability"]<n["target_eos_probability"]-1e-10 for r in verify for n in r["nodes"]),
                       repeated_rounds=sum(r["repeat_suffix"] for r in verify),request_replay_identical=True,
                       limitation="EOS probability suppression is a mechanism, not proof of causation; sequential root replay is a separate diagnostic.")
        write_json(self.root/"diagnosis.json",diagnosis)
        self.requests('diagnostics','strict_delta0',{**self.variants['anchor4'],'delta':0.0},[self.tuning[4]],128,diagnostic=True)
        self.sequential_replay(diagnostics)

    def sequential_replay(self,diagnostics):
        """Compare tree root vs independently prefilling exactly the same IDs.

        Prompt logprobs are raw (T=1). Compare log odds after multiplying tree
        log odds by T=.6, avoiding an unknown full-vocabulary normalizer.
        Appended EOS is input for scoring, never a forced generated output.
        """
        replays=[{**r['sequential_replay'],'source_file':r.get('source_file',''),
                  'request':r['request'],'generated_length':r['generated_length'],
                  'prefix_sha256':hashlib.sha256(json.dumps(r['sequential_replay']['prefix_ids']).encode()).hexdigest()}
                 for r in diagnostics if 'sequential_replay' in r
                 and len(r['sequential_replay']['prefix_ids'])>64]
        chosen=[];seen=set()
        for r in replays:
            key=tuple(r['prefix_ids'])
            if key not in seen:chosen.append(r);seen.add(key)
            if len(chosen)==4:break
        if not chosen:raise ValueError("No actual request prefixes captured for sequential replay")
        directory=self.root/'diagnostics/sequential_target';directory.mkdir(parents=True,exist_ok=True)
        frozen_cases=directory/'prefix_manifest.json'
        if frozen_cases.exists() and json.loads(frozen_cases.read_text())!=chosen:
            raise ValueError('Sequential replay prefix manifest changed')
        write_json(frozen_cases,chosen)
        result=directory/'checks.json'
        if result.exists():
            if not json.loads(result.read_text())['passed']:
                raise ValueError('Saved tree/sequential check failed; refusing to bypass it')
            return
        self.current=settings(method='target',head=1);self.diagnostic=False
        self.status('SEQUENTIAL_REPLAY','target',0,len(chosen))
        self.start_server('target',directory)
        checks=[]
        try:
            for r in chosen:
                eos=r['eos_ids'][0]
                payload=dict(model='TencentBAC/FastMTP',prompt=r['prefix_ids']+[eos],echo=True,
                             max_tokens=0,temperature=.6,logprobs=20,prompt_logprobs=20,
                             return_tokens_as_token_ids=True,return_token_ids=True,seed=20261002)
                req=urllib.request.Request(self.base_url+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=1800) as response:data=json.load(response)
                write_json(directory/f"response_{r['prefix_sha256']}.json",dict(payload=payload,response=data))
                raw=data['choices'][0]['prompt_logprobs'][-1]
                ref=r['top_ids'][0];ref_value=raw[str(ref)]['logprob']
                errors=[]
                for token,p in zip(r['top_ids'],r['top_probs']):
                    if str(token) in raw and p>0:
                        expected=.6*math.log(p/r['top_probs'][0]);actual=raw[str(token)]['logprob']-ref_value
                        errors.append(abs(expected-actual))
                eos_error=None
                if r['eos_probs'][0]>0 and str(eos) in raw:
                    eos_error=abs(.6*math.log(r['eos_probs'][0]/r['top_probs'][0])-(raw[str(eos)]['logprob']-ref_value))
                checks.append(dict(prefix_length=len(r['prefix_ids']),prefix_sha256=r['prefix_sha256'],
                                   source_file=r['source_file'],request=r['request'],generated_length=r['generated_length'],
                                   matched_top_tokens=len(errors),max_abs_log_odds_error=max(errors),eos_log_odds_error=eos_error))
            write_json(result,dict(checks=checks,temperature=.6,tolerance=.15,
                                  passed=all(x['matched_top_tokens']>=4 and x['max_abs_log_odds_error']<=.15 and
                                             (x['eos_log_odds_error'] is None or x['eos_log_odds_error']<=.15) for x in checks),
                                  limitation='Root distributions only; BF16/FP4 numerical tolerance, not proof of every branch cache'))
        finally:self.stop_server()
        if not json.loads(result.read_text())['passed']:raise ValueError('Tree/sequential root mismatch; investigate before tuning')

    def blocks(self,phase,name,config,rows,seeds):
        return [r for seed in seeds for r in self.requests(phase,f"{name}_seed{seed}",config,[(row,seed) for row in rows])]

    def run(self):
        self.diagnose()
        perf={}
        for name,x in self.variants.items():
            perf[name]=self.requests("ablation",name,x,[(r,42) for r in self.train[:2]],128)
        repeat=self.requests("ablation","old_repeat",self.variants["old"],[(r,42) for r in self.train[:2]],128)
        old_rate=summarize(perf["old"]+repeat)["e2e_tok_s"]
        eligible=sorted([n for n in self.variants if n!="old" and summarize(perf[n])["e2e_tok_s"]>old_rate*1.01],
                        key=lambda n:summarize(perf[n])["e2e_tok_s"],reverse=True)
        write_json(self.root/"performance_gate.json",dict(passed=bool(eligible),baseline_tok_s=old_rate,eligible=eligible))
        if not eligible: self.status("STOPPED_PERFORMANCE_GATE");return 3
        structure=eligible[0];base=self.variants[structure]
        tuning_variants={"minimal":base,
                         "guard_delta1":{**base,"protect":True},
                         "guard_delta05":{**base,"protect":True,"delta":0.5},
                         "guard_delta025":{**base,"protect":True,"delta":0.25},
                         "guard_mix0":{**base,"protect":True,"mix":0.0}}
        alternate=next((n for n in eligible if self.variants[n]['depth']!=base['depth']),None)
        if alternate:
            tuning_variants['alternate_depth']=self.variants[alternate]
            tuning_variants['alternate_guard']={**self.variants[alternate],'protect':True}
        tuned={}
        control=self.requests("tuning","old",self.variants["old"],self.tuning)
        for name,x in tuning_variants.items(): tuned[name]=self.requests("tuning",name,x,self.tuning)
        candidates=sorted([n for n in tuned if paired_gate(control,tuned[n])["passed"]],
                          key=lambda n:(-summarize(tuned[n])["stopped_correct"],summarize(tuned[n])["mean_seconds"]))[:2]
        write_json(self.root/"tuning_gate.json",dict(passed=bool(candidates),structure=structure,candidates=candidates,
                      comparisons={n:paired_gate(control,r) for n,r in tuned.items()}))
        if not candidates: self.status("STOPPED_TUNING_GATE");return 3
        reference=self.blocks("regression","old",self.variants["old"],self.regression,[42,43])
        passed={};checks={}
        for name in candidates:
            records=self.blocks("regression",name,tuning_variants[name],self.regression,[42,43])
            checks[name]=paired_gate(reference,records)
            if checks[name]["passed"]: passed[name]=records
        write_json(self.root/"regression_gate.json",dict(passed=bool(passed),comparisons=checks))
        if not passed: self.status("STOPPED_REGRESSION_GATE");return 3
        chosen='minimal' if 'minimal' in passed else min(passed,key=lambda n:summarize(passed[n])["mean_seconds"])
        selected={"old":self.variants["old"],"adaptive":tuning_variants[chosen]}
        # All baseline settings selected only on tuning train, before heldout.
        for method in ("native","cactus","spec_cascade"):
            options={}
            for depth in (3,4):
                x=settings(method=method,depth=depth)
                records=self.requests("baseline_selection",f"{method}_d{depth}",x,self.tuning)
                s=summarize(records);options[depth]=(x,s)
            depth=max(options,key=lambda d:(options[d][1]["stopped_correct"],options[d][1]["correct"],-options[d][1]["truncated"],-options[d][1]["mean_seconds"]))
            selected[method]=options[depth][0]
        write_json(self.root/"selection.json",dict(candidate_name=chosen,settings=selected,source="tuning/regression train only"))
        heldout={n:self.blocks("heldout",n,x,self.heldout,[42,43,44]) for n,x in selected.items()}
        comparisons={n:paired_gate(r,heldout["adaptive"]) for n,r in heldout.items() if n!="adaptive"}
        write_json(self.root/"heldout_gate.json",dict(passed=all(x["passed"] for x in comparisons.values()),comparisons=comparisons))
        if not all(x["passed"] for x in comparisons.values()):self.status("STOPPED_HELDOUT_GATE");return 3
        # Final IDs and answers are only loaded after selection/heldout pass.
        test=load_gsm8k(ROOT/"data/gsm8k/test.jsonl")
        old_requests=ROOT/"results/raven_jetson_gsm8k_n100_fp4_2048/native/requests.jsonl"
        excluded={json.loads(s)["question_id"] for s in old_requests.read_text().splitlines()}
        rows=sorted(random.Random(20261003).sample([r for r in test if r["question_id"] not in excluded],100),key=lambda r:r["question_id"])
        self.ensure_budget(rows)
        manifest=dict(question_ids=[r["question_id"] for r in rows],excluded_old_ids=sorted(excluded),settings=selected,seeds=[42,43,44])
        write_json(self.root/"formal_manifest.json",manifest)
        formal={}
        # Rotate method order by seed to reduce time/temperature confounding.
        names=list(selected)
        for i,seed in enumerate((42,43,44)):
            for n in names[i:]+names[:i]:
                formal.setdefault(n,[]).extend(self.requests("formal",f"{n}_seed{seed}",selected[n],[(row,seed) for row in rows]))
        comparisons={n:{**paired_gate(r,formal["adaptive"]),**clustered_intervals(r,formal["adaptive"])} for n,r in formal.items() if n!="adaptive"}
        quality_speed=all(x["passed"] for x in comparisons.values())
        stable_speed=all(x["throughput_ratio_95ci"][0]>1 for n,x in comparisons.items() if n in ("native","cactus","spec_cascade"))
        write_json(self.root/"formal_gate.json",dict(passed=quality_speed and stable_speed,quality_speed=quality_speed,stable_speed=stable_speed,comparisons=comparisons))
        # Independent supplementary ablations do not influence selection.
        for name,x in [("uniform",{**selected['adaptive'],"adaptive":False}),
                       ("guard_off",{**selected['adaptive'],"protect":False})]:
            self.requests("supplement",name,x,[(row,42) for row in rows])
        self.status("COMPLETE" if quality_speed and stable_speed else "STOPPED_FORMAL_GATE")
        return 0 if quality_speed and stable_speed else 3


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);p.add_argument("--port",type=int,default=8000)
    p.add_argument("--check-only",action="store_true")
    p.add_argument('--prefix-cache',type=int,choices=(0,1),default=1)
    p.add_argument('--deep-diagnostic',action='store_true')
    p.add_argument('--uniform-gemm',action='store_true')
    p.add_argument('--full-bf16-accum',action='store_true')
    p.add_argument('--bnb-row-block',type=int,choices=(0,16,64),default=0)
    p.add_argument('--target-backend',choices=('','FLASH_ATTN','TREE_ATTN','TRITON_ATTN'),default='')
    p.add_argument('--triton-2d',action='store_true')
    p.add_argument('--attention-fp32',action='store_true')
    p.add_argument('--canonical-tree',action='store_true')
    p.add_argument('--rmsnorm-row-limit',type=int,choices=(0,255),default=0)
    p.add_argument('--lm-head-row-block',type=int,choices=(0,16),default=0)
    args=p.parse_args()
    controller=None
    def interrupted(*args):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        controller=Round2(args)
        if args.check_only:
            controller.status("PREFLIGHT_COMPLETE");return 0
        return controller.run()
    except (ValueError,FileNotFoundError,BlockingIOError) as exc:
        if controller:controller.status("STOPPED_PROTOCOL_ERROR")
        print(str(exc),file=sys.stderr,flush=True);return 2
    except Exception:
        if controller:controller.status("FAILED_RUNTIME")
        raise
    finally:
        if controller:controller.stop_server()


if __name__=="__main__":raise SystemExit(main())
