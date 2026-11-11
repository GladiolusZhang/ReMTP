#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$PROJECT_DIR" <<'PY'
from datetime import datetime, timezone, timedelta
import json, subprocess, sys
from pathlib import Path
project = Path(sys.argv[1])
manifest_path = project / 'results/raven_full_latest.json'
if not manifest_path.exists():
    raise SystemExit('100+100 正式实验尚未启动。')
manifest = json.loads(manifest_path.read_text())
root = Path(manifest['run_dir'])
subprocess.run(['systemctl', '--user', 'show', manifest['unit'], '-p', 'ActiveState', '-p', 'SubState',
                '-p', 'Result', '-p', 'NRestarts', '-p', 'ExecMainStatus'])
def read(name):
    path = root / name
    return json.loads(path.read_text()) if path.exists() else {}
state = read('status.json')
print('阶段：' + state.get('phase', '准备'))
print(f"生成进度：{state.get('all_completed', 0)}/1000；HumanEval判题：{state.get('evaluation_completed', 0)}/500")
if state.get('current_work'):
    print('任务：' + json.dumps(state['current_work'], ensure_ascii=False))
heartbeat = read('heartbeat.json')
if heartbeat:
    timestamp = datetime.strptime(heartbeat['updated_utc'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    print('心跳：' + timestamp.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8'))
    print('模型服务存活：' + str(heartbeat['server_alive']))
failure = read('failure.json')
if failure and state.get('phase') in {'FAILED', 'INTERRUPTED'}:
    print('停止原因：' + failure['error'])
print('中文实时报告：' + str(project / 'doc/RESULT-RAVEN-100X100-LIVE.md'))
print('原始结果目录：' + str(root))
PY
