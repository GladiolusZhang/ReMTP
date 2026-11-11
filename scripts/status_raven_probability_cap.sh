#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$PROJECT_DIR" <<'PY'
import json,subprocess,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
project=Path(sys.argv[1]); manifest=json.loads((project/'results/raven_probability_cap_latest.json').read_text())
root=Path(manifest['run_dir'])
subprocess.run(['systemctl','--user','show',manifest['unit'],'-p','ActiveState','-p','SubState','-p','Result','-p','NRestarts'])
state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
stamp=state.get('updated_utc','')
if stamp: stamp=datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8')
print(f"阶段：{state.get('phase','准备')}；当前组：{state.get('variant','')} {state.get('completed',0)}/{state.get('total',0)}；更新时间：{stamp}")
for phase,label,limit in [('diagnostics','2048完整诊断',7),('regression','2048训练对照',144)]:
    count=sum(len(p.read_text().splitlines()) for p in (root/phase).glob('*/requests.jsonl'))
    print(f'{label}已保存：{count}/{limit}条（后续训练须门槛放行）')
for path in sorted(root.glob('*_gate.json')):
    gate=json.loads(path.read_text());print(f"{path.stem}：{'通过' if gate['passed'] else '未通过'}")
if (root/'failure.json').exists():print((root/'failure.json').read_text())
print(f'实时中文报告：{project}/doc/RESULT-RAVEN-PROBABILITY-CAP-LIVE.md')
PY
