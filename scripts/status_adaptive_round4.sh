#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/.venv/bin/python" - "$PROJECT_DIR" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
project=Path(sys.argv[1]);latest=project/'results/adaptive_round4_latest.json'
state={}
if latest.exists():
    launch=json.loads(latest.read_text());root=Path(launch['run_dir']);unit=launch['unit']
else:
    root=project/'results/target_state_norm_round4_20261004';unit='remtp-target-state-norm-round4.service'
subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus','-p','NRestarts'])
path=root/'status.json'
if path.exists():
    state=json.loads(path.read_text())
    print(f"阶段：{state['phase']}；当前组：{state['variant']} {state['completed']}/{state['total']}")
    print(f"主流水线已保存请求：{state['all_completed']}；主阶段更新时间：{state['updated_utc']}")
    for child in ('prefill_validation','real_state_validation'):
        path=root/child/'status.json'
        if path.exists():
            child_state=json.loads(path.read_text())
            print(f"子阶段 {child}：{child_state['phase']} 当前组 {child_state['completed']}/{child_state['total']}；"
                f"累计请求 {child_state['all_completed']}；更新 {child_state['updated_utc']}")
if path.exists() and state.get('phase')=='state_natural':
    probes=[root/'state_natural'/state['variant']/'prefix_diagnostic.jsonl']
else:
    probes=[root/'real_state_validation/natural/old_window/prefix_diagnostic.jsonl'] if state.get('phase') in ('REAL_STATE_VALIDATION','WAITING_REAL_STATE_GATE') else []
existing=[p for p in probes if p.exists()]
if existing:
    probe=max(existing,key=lambda p:p.stat().st_mtime)
    rows=probe.read_text().splitlines()
    if rows:
        try:
            parsed=[json.loads(line) for line in rows]
            requests={r['request'] for r in parsed if len(r.get('sequential_replay',{}).get('prefix_ids',()))>64}
            actual=[r for r in parsed if r.get('request') in requests and 'generated_length' in r]
            if actual:print(f"本组最近生成记录：{actual[-1]['generated_length']} token（该请求可能已完成）；来源：{probe.parent.name}")
        except json.JSONDecodeError:pass
elif state.get('phase')=='state_natural':
    print('当前组正在加载/预热服务；尚无正式请求的生成记录。')
print(f"实际结果目录：{root}")
print(f"Target现场快照：{len(list(root.glob('real_state_validation/natural/*/target_probe/*/state_*.pt')))}")
gates=sorted(root.glob('*_gate.json'),key=lambda p:p.stat().st_mtime)
if gates:
    gate=json.loads(gates[-1].read_text());print(f"最新门槛：{gates[-1].name} {'通过' if gate['passed'] else '失败'}")
print(f"中文报告：{root}/report.md")
if (root/'failure.json').exists():print('失败详情：'+(root/'failure.json').read_text())
PY
