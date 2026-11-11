#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/.venv/bin/python" - "$PROJECT_DIR" <<'PY'
import json,subprocess,sys
from pathlib import Path
project=Path(sys.argv[1]);launch=json.loads((project/'results/adaptive_round6_latest.json').read_text())
root=Path(launch['run_dir'])
subprocess.run(['systemctl','--user','show',launch['unit'],'-p','ActiveState','-p','SubState','-p','MainPID','-p','NRestarts','-p','Result'])
state=json.loads((root/'status.json').read_text())
print(f"阶段：{state['phase']}；当前组：{state['variant']} {state['completed']}/{state['total']}")
print(f"本单元已保存 {state['all_completed']} 条主请求；先前中断 {launch.get('prior_aborted_requests',0)} 条单列；更新时间：{state['updated_utc']}")
probe=root/state['phase']/state['variant']/'prefix_diagnostic.jsonl'
if probe.exists():
    for line in reversed(probe.read_text().splitlines()):
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r.get('kind')=='verification':
            print(f"本组最近生成记录：{r['generated_length']} token（该请求可能已完成）");break
for path in sorted(root.glob('*_gate.json')):
    gate=json.loads(path.read_text());print(f"{path.name}：{'通过' if gate['passed'] else '未通过'}")
if (root/'failure.json').exists():print((root/'failure.json').read_text())
print(f"中文报告：{project}/doc/RESULT-ADAPTIVE-RAVEN-ROUND6.md")
PY
