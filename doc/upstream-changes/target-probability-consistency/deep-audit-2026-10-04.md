# 详细失败复查：诊断与计划改动

日期：2026-10-04。正式算法、数值开关和模型权重未修改。

| 文件 | 本次改动 | 实际验证 |
| --- | --- | --- |
| `scripts/investigate_target_consistency.py` | 新增显式 `--manifest`，校验token hash、重复前缀和必要概率字段；支持本次真实109/238/157/287，不默认重测旧236/286；显式canonical开关 | 24条原生请求实际完成；关键9执行文件保持第三轮字节一致 |
| `tests/test_target_probability_diagnostics.py` | 增加真实manifest正常读取、token被修改后拒绝的针对性检查 | 相关7 passed（3.12秒），无软件安装/升级 |
| `doc/ANALYSIS-TARGET-CONSISTENCY-DEEP-AUDIT-2026-10-04.md` | 中文失败证据、验收缺口、官方版本适用范围、本机60请求及待验证假设 | 原始JSON/快照/单位退出状态核对 |
| `plan/target-consistency-deep-repair-2026-10-04.md` | 首次长状态分叉定位、单因素修复、长状态gate、截断及三基线后续顺序 | 步骤1诊断完成；步骤2以后的实际长状态修复未执行 |
| 原计划/修复文档、本目录README、总CHANGELOG和研究日志 | 增加最新入口与明确历史状态 | 不覆盖冻结来源/失败gate或旧结果 |

## 有限实机结果

`results/target_consistency_deep_audit_20261004/`：原生24、新鲜树16、独立祖先16、原old/head1新鲜root4，
共60条对照请求。原生检查top/hidden差0；树24root项、16祖先及head1的4项top/EOS检查均通过。
287新鲜head1 hidden/top/EOS差0，而原长生成记录相对新echo仍差0.250001。
因此排查重点为真实长历史/输入位置/KV提交/算术，未证明具体算子或拷贝错误。

本机CPU隔离检查还确认BF16 flag=False保持split-K=True，(False,False)才能同时关闭；
仅记录线索，未改变模型进程。vLLM0.18文档及源码不支持把全局不变模式直接作为sm87/TREE修复。
两个有限user unit已exit0；正式round3仍failed/exit2/NRestarts0，未恢复。

初次测试调用缺pytest/导入路径，使用已有系统pytest并设置项目/.venv依赖路径后7项通过；
没有安装pytest或升级环境。完整379/2skip是此前结果，不冒充本次测试或长状态成功。

下一步：[长状态定位计划](../../../plan/target-consistency-deep-repair-2026-10-04.md)。
