# 局部概率上限与Native诊断改动记录

> 完成状态更新（2026-10-07，UTC+8）：17:39:27在seed42原判对22→21的质量门槛停止；7诊断、32 CUDA及48训练完成，剩余96训练未执行。下方启动期状态为历史记录；固定cap2未采用。

日期：2026-10-07（UTC+8）。状态：新冻结已启动，Native健康检查/预热通过；首条2048完整诊断生成中。

| 文件 | 改动与原因 |
| --- | --- |
| remtp/dynamic_mtp_tree.py | 新增默认0的局部H(y)/P(y)上限；分配与最终完整H同时约束，接受与残差一致，避免只截接受概率 |
| remtp/dynamic_tree_vllm.py | 解析环境开关并打印实际值，控制器需核验worker日志 |
| remtp/decision_trace.py | 默认first8保留，新增最多8个checkpoint的verifier-only帧；本轮5个后段点，无额外模型/RNG，不声称proposal配对 |
| scripts/run_adaptive_round2.py | 显式传递cap和checkpoint开关；未指定时0/空保持原行为 |
| scripts/run_raven_probability_cap.py | 6 Native+1 B重放→32新CUDA→最多144训练，负gate停、保留实际输出/质量/性能结果 |
| scripts/launch_raven_probability_cap.py | 新目录冻结、check-only、systemd无自动重试启动；保留旧运行入口/结果 |
| scripts/status_raven_probability_cap.sh | 中文进度与UTC+8时间，显示诊断/训练预算及失败现场 |
| tests/test_probability_ratio_cap.py | 测接受/残差共用H、严格R=1分布、端点/保护/再分配、启动路径与失败停止 |

## 已执行验证

- 首轮相关CPU：125项通过，11.87秒。后续有效worker/控制器回归60项通过，6.84秒。
- `py_compile`、`bash -n`、`git diff --check`通过（后续最终变更需再检查）。
- 16真实帧的旧实现/禁用cap CPU等价实际通过，32次CPU调用，输出/事件/RNG均一致；这是新CPU RNG，不替代原CUDA重放。
- 新冻结check-only及manifest/policy预算、有效候选标志检查通过；11个数值/运行核心源文件与修复冻结相同（不是数值gate）。
- systemd用户单元active/running，0重启、Linger=yes，Native健康检查/8token预算预热实际通过，有效Native worker/head1/full-Q adapter均确认，首条1532/42完整诊断生成中；后段B、32 CUDA和训练尚未完成，不预写成功。

## 限制与论文说明

R=2是算法变更，不是无损实现优化。它只限制被采样候选概率，保留近似松弛性质。
可能降低接受长度；是否改善正确率/截断及整题耗时需要实测。Native诊断6个配对来自开发故障集，不能代替三基准正式比较。
Target完整长上下文/KV数值证明、唯一候选冻结后的独立留出仍未完成。
