# 根因定位观测与有限控制器

> 完结更新（2026-10-09核验）：原B根因10月8日10:32:30 UTC+8正常结束，6自然请求、11fresh、2启动/预热，exit0、0重试；12/12现场raw精确通过。独立CPU核验43完整耦合帧和实际提交ID通过；没有把probe的时间当速度。随后候选执行完成但质量/速度拒绝，见[中文结果](../../RESULT-RAVEN-TARGET-ODDS-2026-10-09.md)。以下未启动/运行状态为历史，旧冻结和门槛保留。

日期：2026-10-08（UTC+8）。状态：实现完成，54项相关CPU测试与真实check-only通过，GPU尚未启动。

> 实际启动更新：本文件上句为启动前历史状态。09:26:01独立冻结启动remtp-raven-root-cause-20261008.service，10:11完成4/6自然轨迹并复现历史输出，内部实际ID/自然raw-P耦合通过；fresh Target门槛待完成。工作区新风险候选在[独立改动文件](target-odds-repair-2026-10-08.md)登记，未应用到运行中的原B冻结。

## 原版改动

- remtp/decision_trace.py：可选全轮摘要，从实际验证结果取得P/Q/H、uniform、提交结果；有限请求/轮数。检查点模式可单独采3个早段proposal，标签保留pre-bookkeeping历史；不把早段副本误认为晚段完整prefix证明。原默认检查点模式仍为verifier-only。
- remtp/target_diagnostics.py：新增最小prompt长度选择，跳过本单元短预热；只复制实际Target结果。沿用已修复的最终prefill分段位置检查。
- scripts/run_adaptive_round2.py：透传新的可选观测参数，支持只开decision trace而不开旧prefix diagnostic，避免旧head首3次的额外投影。默认行为保持。
- 新CPU分析脚本、根因控制器、独立冻结启动与中文状态查询、针对性测试、计划及报告。

## 范围

固定现有Target/MTP数学、真实Q采样、主验收/残差、KV提交与原评测；所有观测放在边界包装层。摘要每轮只新增一次小规模ID传回和文件写入；没有新的模型前向/投影/抽样。摘要和完整帧仍可能同步CPU/GPU，实机必须验证输出复现；有观测耗时不能用于速度比较。

完整帧/真实raw/API提交ID对齐，12个独立fresh前缀与自然保留KV root对比。源组件一致不是数值证明；新的raw门槛与输出签名需要实际跑过。早段proposal副本只证明观测/bookkeeping一致，不独立证明完整MTP状态正确。

预算：6主HTTP/12数值/2预热/2启动，4小时，无自动重试；具体范围见[计划](../../../plan/adaptive-raven-root-cause-2026-10-08.md)。不采用候选、不重写旧gate/freeze，不自动展开长期搜索。

## 已实际验证

- 53项CPU测试，4.10秒，覆盖原验证器仅执行一次、Torch RNG不多消费、有限摘要、真实ID与最终budget裁剪、prefix/位置/原始P错配拒绝、短预热/prefill跳过、启动MRO与实际env、旧数值探针和EOS边界。
- 独立CPU原评分/输出检查已保存results/raven_root_cause_offline_20261008.json；现代码八路径完整枚举证明总局部TV下降不保证首token边缘分布偏移下降。具体概率与推理见[中文根因记录](../../ROOT-CAUSE-ADAPTIVE-RAVEN-2026-10-08.md)。
- py_compile、bash语法、git diff --check通过。CPU测试不证明GPU长期状态正确；六条自然输出/12个raw门槛仍待实机。

启动前追加：完整prefix去重边界测试通过，避免两seed首prefix相同造成manifest二义性；最终CPU54项/3.96秒，真实workspace check-only通过（只核验组件/协议、加载CPU tokenizer，不加载GPU）。README/优化报告入口同步根因阶段。用户授权定位后直接制定/实施修复并重跑；仍以原数值/质量门槛决定扩展。
