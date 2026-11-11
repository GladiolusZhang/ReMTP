#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/.venv/bin/python" - "$PROJECT_DIR" <<'PY'
import json,subprocess,sys
from pathlib import Path
project=Path(sys.argv[1]);launch=json.loads((project/'results/adaptive_round5_latest.json').read_text())
root=Path(launch['run_dir'])
subprocess.run(['systemctl','--user','show',launch['unit'],'-p','ActiveState','-p','SubState','-p','MainPID','-p','NRestarts'])
state=json.loads((root/'status.json').read_text())
print(f"阶段：{state['phase']}；当前组：{state['variant']} {state['completed']}/{state['total']}")
print(f"本轮已保存 {state['all_completed']}/104 条新请求；更新时间：{state['updated_utc']}")
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
print(f"中文报告：{root}/report.md")
PY
