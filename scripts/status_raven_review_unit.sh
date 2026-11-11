#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$PROJECT_DIR" <<'PY'
import json,subprocess,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
project=Path(sys.argv[1]);manifest_path=project/'results/raven_review_latest.json'
if not manifest_path.exists():raise SystemExit('新证据单元尚未启动。')
manifest=json.loads(manifest_path.read_text());root=Path(manifest['run_dir'])
if manifest.get('evidence_reuse'):
    print('本单元为观测修复续跑：复用8条profile和4个精确前缀，旧失败保留。')
    print('两单元合计尝试上限：'+json.dumps(manifest['combined_attempt_limits'],ensure_ascii=False))
subprocess.run(['systemctl','--user','show',manifest['unit'],'-p','ActiveState','-p','SubState',
    '-p','Result','-p','NRestarts','-p','ExecMainStatus'])
def read(name):
    path=root/name
    return json.loads(path.read_text()) if path.exists() else {}
state=read('status.json');heartbeat=read('heartbeat.json');ledger=read('ledger.json')
finished=state.get('phase') in {'COMPLETED_EVIDENCE_UNIT','NUMERIC_REJECTED','QUALITY_REJECTED',
    'BUDGET_STOPPED','INTERRUPTED','RUNTIME_ERROR'}
print('阶段：'+state.get('phase','准备')+'；当前组：'+state.get('variant',''))
work=heartbeat.get('work') or state.get('current_work',{})
if work:print(('最后执行任务：' if finished else '当前任务：')+json.dumps(work,ensure_ascii=False))
if heartbeat:
    moment=datetime.strptime(heartbeat['updated_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    age=(datetime.now(timezone.utc)-moment).total_seconds()
    local=moment.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8')
    if finished:
        print(f"最后心跳：{local}；当时服务存活：{heartbeat.get('server_alive')}（历史记录，实验已结束）")
    else:
        print(f"心跳：{local}，距今{age:.0f}秒；服务存活：{heartbeat.get('server_alive')}")
for name,row in ledger.get('counts',{}).items():
    print(f"{name}：已完成{row['completed']}/{row['limit']}，已尝试{row['attempted']}（失败也占预算）")
for name in ('target_prefix_gate.json','B_reproduction_gate.json'):
    gate=read(name)
    if gate:print(f"{name}：{'通过' if gate['passed'] else '未通过'}；完整={gate.get('complete','见checks')}")
for name in ('failure.json','exit.json'):
    record=read(name)
    if record:print(name+'：'+json.dumps(record,ensure_ascii=False))
print('实时中文报告：'+str(project/'doc/RESULT-RAVEN-REVIEW-LIVE.md'))
print('本单元只补证据，不自动采用候选或启动独立正式测试。')
PY
