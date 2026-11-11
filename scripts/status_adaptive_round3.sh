#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
systemctl --user show remtp-adaptive-round3.service \
  -p ActiveState -p SubState -p MainPID -p ExecMainStatus -p NRestarts || true
"$PROJECT_DIR/.venv/bin/python" - "$PROJECT_DIR/results/adaptive_optimization_20261004_target_canonical" <<'PY'
import json
import sys
from pathlib import Path
root=Path(sys.argv[1]);status=root/'status.json'
if not status.exists():
    print('尚未建立本轮优化状态；先查看 Target 一致性验收。')
else:
    x=json.loads(status.read_text())
    print(f"阶段：{x['phase']}；当前组：{x['variant']} {x['completed']}/{x['total']}")
    print(f"已保存请求：{x['all_completed']}；已安排请求：{x['scheduled_total']}")
    print(f"最后更新时间：{x['updated_utc']}")
    print(x['note'])
print(f'中文报告：{root}/report.md')
PY
journalctl --user -u remtp-adaptive-round3.service -n 8 --no-pager
