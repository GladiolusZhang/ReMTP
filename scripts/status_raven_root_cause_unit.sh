#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$PROJECT_DIR" <<'PY'
from datetime import datetime,timezone,timedelta
import json,subprocess,sys
from pathlib import Path
project=Path(sys.argv[1]); manifest_path=project/'results/raven_root_cause_latest.json'
if not manifest_path.exists():raise SystemExit('根因定位单元尚未启动。')
manifest=json.loads(manifest_path.read_text());root=Path(manifest['run_dir'])
subprocess.run(['systemctl','--user','show',manifest['unit'],'-p','ActiveState','-p','SubState',
    '-p','Result','-p','NRestarts','-p','ExecMainStatus'])
def read(name):
    p=root/name
    return json.loads(p.read_text()) if p.exists() else {}
state=read('status.json'); heartbeat=read('heartbeat.json')
finished=bool(read('exit.json')) or state.get('phase') in {
    'COMPLETED_ROOT_CAUSE_UNIT','NUMERIC_REJECTED','REPLAY_REJECTED','BUDGET_STOPPED','INTERRUPTED','RUNTIME_ERROR'}
print('阶段：'+state.get('phase','准备'))
work=state.get('current_work') or heartbeat.get('work',{})
if work:print(('最后任务：' if finished else '当前任务：')+json.dumps(work,ensure_ascii=False))
if heartbeat:
    dt=datetime.strptime(heartbeat['updated_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    local=dt.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8')
    print(('最后心跳：' if finished else '心跳：')+local)
    print(('当时服务存活（历史记录）：' if finished else '服务存活：')+str(heartbeat.get('server_alive')))
for kind,c in read('ledger.json').get('counts',{}).items():
    print(f"{kind}：完成{c['completed']}/{c['limit']}，尝试{c['attempted']}（失败占预算）")
for name in ('B_reproduction_gate.json','target_natural_prefix_gate.json'):
    g=read(name)
    if g:print(name+'：'+('通过' if g['passed'] else '未通过')+'；完整='+str(g.get('complete',True)))
failure=read('failure.json')
if failure:print('停止原因：'+failure.get('kind','')+' '+failure.get('error',''))
print('中文报告：'+str(root/'report.md'))
print('诊断带探针，不计正式速度；没有自动采用候选或展开留出/test。')
PY
