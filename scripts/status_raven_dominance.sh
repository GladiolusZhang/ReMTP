#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$PROJECT_DIR" <<'PY'
from datetime import datetime,timezone,timedelta
import json,subprocess,sys
from pathlib import Path
project=Path(sys.argv[1]);p=project/'results/raven_dominance_latest.json'
if not p.exists():raise SystemExit('三基准优势搜索尚未启动。')
manifest=json.loads(p.read_text());root=Path(manifest['run_dir'])
subprocess.run(['systemctl','--user','show',manifest['unit'],'-p','ActiveState','-p','SubState',
    '-p','Result','-p','NRestarts','-p','ExecMainStatus'])
def read(name):
    p=root/name
    return json.loads(p.read_text()) if p.exists() else {}
state=read('status.json');heartbeat=read('heartbeat.json');end=read('exit.json')
print('阶段：'+state.get('phase','准备'))
work=state.get('current_work') or heartbeat.get('work',{})
if work:print(('最后任务：' if end else '当前任务：')+json.dumps(work,ensure_ascii=False))
if heartbeat:
    dt=datetime.strptime(heartbeat['updated_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    print(('最后心跳：' if end else '心跳：')+dt.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8'))
for kind,c in read('ledger.json').get('counts',{}).items():
    print(f"{kind}：完成{c['completed']}/{c['limit']}，尝试{c['attempted']}（失败占预算）")
failure=read('failure.json')
if failure:print('停止原因：'+failure.get('kind','')+' '+failure.get('error',''))
analysis=read('analysis.json')
if analysis:print('开发集点门槛候选：'+str(analysis['development_point_winner']))
print('截断/正常结束/逐题损失仅分析；正确率、吞吐与整题耗时决定门槛。')
print('中文报告：'+str(root/'report.md'))
PY
