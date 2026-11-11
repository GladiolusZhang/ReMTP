# 目标更新：只以超过三个基准评价效果

> 新增比较条件（2026-10-14）：用户明确D4/N12的Deep对D3/N9原Adaptive不公平，Deep不予采用。后续候选与原Adaptive固定D3/N9，三个链式基线沿用原D3；完整同题同seed、公共环境/生成协议保持一致，并报告实际草稿与验证成本。相同预算是方法收益的比较前提；截断仍只诊断，正确率与两种速度的目标保留。[当前优化计划](../plan/adaptive-raven-equal-budget-2026-10-14.md)。下文扩大深度/节点数的路线与旧门槛为历史。

> 首单元完结（2026-10-10）：112/112生成/7启动预热全部完成、exit0/0重试。两个D4对原B/Native通过开发点门槛，仍未通过全部三基线门槛。[中文结果与分析](RESULT-RAVEN-DOMINANCE-2026-10-10.md)。目标不变，截断仍仅诊断；当前没有新后台，下文启动/首题状态属于历史。

日期：2026-10-09（UTC+8）。用户明确：不在乎是否截断，目标为超过Native、Cactus、SpecCascade，允许使用全部可行优化路线。

> 首后台搜索11:21:13 UTC+8已启动，冻结check-only和source hash通过，active/running、0重试；B加载中，尚无新质量/速度结果。[实时中文报告](RESULT-RAVEN-DOMINANCE-LIVE.md)，查询bash scripts/status_raven_dominance.sh。下文“实现阶段”为启动前状态。

11:24核验：B服务健康、预热HTTP200及有效配置检查已通过，1532/42首条完整生成中；主生成尝试1/112、完成0，启动/预热各1/7、0重试。尚未产生新完整质量/速度样本。

## 当前判定

保持原评分正确率不低于三个基准，E2E token吞吐更高、平均整题耗时更低。截断、正常结束判对、重复、逐题/逐seed差异保留为解释指标；这些变化不再单独拒绝候选。

整体质量按完整预登记样本比较，允许单题有损有益。原B为参考，方法优化可以改变输出/RNG/近似分布；宣称等价计算优化时才要求相应等价证明。比较使用同一模型、输入、原评分和公共生成协议。

## 已执行的修改与复核

- 新dominance_gate加入remtp/optimization_metrics.py；历史summarize/paired_gate/clustered_intervals函数保留。旧冻结与已保存gate不覆盖。
- 按新目标CPU重评上一轮target_odds，仍对三个基准失败：质量10/16低于Native13/Cactus14/SpecCascade14，平均耗时539.85秒比三者都慢。去掉截断门槛没有把该候选变成胜出者。
- [新目标CPU重评原始产物](../results/raven_objective_revision_20261009.json)保存，不覆盖旧gate；22项CPU检查通过。
- 新有限后台控制器准备比较B、三个Adaptive参数/深度候选与三个基准，各16完整生成；三候选D3/delta.75/mix.25、D4/delta.75/mix.25、D4/delta.5/mix.25。当前实现阶段，启动及CPU验证实际状态见总日志，未宣称已获新GPU结果。

优化范围扩展到松弛/深度/宽度、自适应树、MTP蒸馏/预测、完整H/残差策略、KV/算子/图执行和必要的公共模型/协议调整。每次对比条件同步，开发筛选后做独立验证。[硬件感知构树研究](https://arxiv.org/abs/2402.12374)、[自适应草稿长度研究](https://arxiv.org/abs/2405.19715)。

详细步骤、预期效果及首单元额度：[新执行计划](../plan/adaptive-raven-dominance-2026-10-09.md)。专门变更：[记录](upstream-changes/dominance-objective/2026-10-09.md)。
