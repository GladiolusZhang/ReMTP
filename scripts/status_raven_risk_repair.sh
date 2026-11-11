#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$PROJECT_DIR" <<'PY'
from datetime import datetime,timezone,timedelta
import json,subprocess,sys
from pathlib import Path
project=Path(sys.argv[1]);p=project/'results/raven_risk_repair_latest.json'
if not p.exists():raise SystemExit('Target支持修复单元尚未启动。')
manifest=json.loads(p.read_text());root=Path(manifest['run_dir'])
subprocess.run(['systemctl','--user','show',manifest['unit'],'-p','ActiveState','-p','SubState',
    '-p','Result','-p','NRestarts','-p','ExecMainStatus'])
def read(name):
    p=root/name
    return json.loads(p.read_text()) if p.exists() else {}
state=read('status.json');heartbeat=read('heartbeat.json');end=read('exit.json')
finished=bool(end)
print('阶段：'+state.get('phase','准备'))
work=state.get('current_work') or heartbeat.get('work',{})
if work:print(('最后任务：' if finished else '当前任务：')+json.dumps(work,ensure_ascii=False))
if state.get('phase')=='WAITING_ROOT_DIAGNOSTICS':
    print('轻量等待中：未加载Torch或第二个GPU服务。')
if heartbeat:
    dt=datetime.strptime(heartbeat['updated_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    print(('最后心跳：' if finished else '心跳：')+
        dt.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8'))
    print(('当时服务存活（历史记录）：' if finished else '服务存活：')+str(heartbeat.get('server_alive')))
for kind,c in read('ledger.json').get('counts',{}).items():
    print(f"{kind}：完成{c['completed']}/{c['limit']}，尝试{c['attempted']}（失败占预算）")
for path in sorted(root.glob('*_gate.json')):
    g=json.loads(path.read_text())
    print(path.name+'：'+('通过' if g['passed'] else '未通过')+'；完整='+str(g.get('complete',True)))
failure=read('failure.json')
if failure:print('停止原因：'+failure.get('kind','')+' '+failure.get('error',''))
analysis=read('analysis.json')
if analysis:
    print('发展样本点门槛：'+str(analysis['development_point_gate_passed'])+
        '；尚非独立留出/正式结论。')
print('中文报告：'+str(root/'report.md'))
print('单候选、0自动重试；数值/质量/速度分开，未自动采用或展开留出/test。')
PY
