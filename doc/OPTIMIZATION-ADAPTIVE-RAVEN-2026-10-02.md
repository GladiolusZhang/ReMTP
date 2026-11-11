# Adaptive RAVEN 第二轮优化实施记录

> 2026-10-04核查：修复后服务已在Target树/顺序概率门槛失败时停止；完整输出重放通过，
> 四道回归题两配置均正常答对、无截断，正式对照尚未开展。
> 下方运行状态均为历史快照，当前以[最新分析](STATUS-ADAPTIVE-RAVEN-2026-10-04.md)为准。

> 最新状态：MTP草稿有效前缀长度已修复，7请求GPU短回放全部通过；新后台
> `remtp-adaptive-round2-fixed.service` 已开始完整长回放，见[修复说明](FIX-ADAPTIVE-RAVEN-REPRODUCIBILITY-2026-10-02.md)。
> 以下是历史实施记录：旧后台于北京时间 2026-10-02 10:11:54 因同配置/同种子重放不一致主动停止，
> 只完成 9 次诊断请求。下文“正在运行”为启动时快照。
> 完整分析见[第二轮停止与结果分析](ANALYSIS-ADAPTIVE-RAVEN-ROUND2-2026-10-02.md)。

日期：2026-10-02（北京时间）。状态快照：实现及 CPU 回归完成，后台诊断正在运行；没有新的完整质量结论。
执行依据：[下一轮计划](../plan/adaptive-raven-next-optimization-2026-10-02.md)。

## 本次实现

| 项目 | 作用与范围 |
| --- | --- |
| 独立公共 head 开关 | `REMTP_HEAD_BATCH_ROWS=1/3`，复制单行输入，返回原行数；用于 Adaptive 及三个适用基线，避免只给我们的方法加速。未设置时保持上一轮入口行为。 |
| EOS 松弛保护 | `REMTP_DYNAMIC_TREE_PROTECT_EOS=1`；Target EOS 总质量达到 0.5、top1 是 EOS 或草稿为 EOS 时，该位置严格验证。没有强制输出 EOS。 |
| 重复保护 | `REMTP_DYNAMIC_TREE_REPEAT_GUARD=1`；仅使用已生成 token，4–64 token 的后缀块连续重复 3 次时，当前块取消松弛。没有根据答案或题号停止。 |
| TV 预算约束 | 保护位置在 TV 回收池计算前置零，额度不转移；接受概率和 residual correction 用同一个验证分布。 |
| 请求局部备选 RNG | `REMTP_REQUEST_LOCAL_BRANCH_RNG=1`，每个新请求重置备选轮数，避免断点/顺序改变后续请求的备选采样；主请求 RNG 不额外消耗。旧配置在此新协议下重跑。 |
| 前缀诊断 | `REMTP_PREFIX_DIAGNOSTIC_JSONL`，真实加载 head 的单行/三行同输入概率对照、主路径 P/Q/EOS 与松弛后 EOS；随后独立 Target 服务重放相同 token 前缀，对照 tree root 的 log odds。诊断不计入测速。 |

EOS/重复保护是新方法候选，均默认关闭；不能把局部严格验证解释为整段分布无损。
顺序回放只检查 root 分布，不能代替全部分支缓存证明。其数值容差与实测误差保存在结果中。

## 有限后台流程

`scripts/run_adaptive_round2.py` 按如下顺序运行，失败时保存结果并停止，成功才进入下一阶段：

1. 原版/v4 退化题诊断、改变请求顺序的复测、delta=0 诊断与独立顺序 Target 回放。
2. 两条固定短提示，逐项测试 topk、head、构树及 D3/D4；重复原版检查漂移。
3. 固定 6 个 train 请求选参：最小配置、EOS/重复保护、delta=1/0.5/0.25、mix=0，并保留另一个树深候选（若通过速度筛选）。
4. 最多两个候选进入原 24 题 × 2 seed 完整回归；原版按相同新协议重跑。
5. 三个基线各自用 train 比较 D3/D4，公共 head 加速一致；冻结候选与基线设置。
6. 独立 24 train 题 × 3 seed × 5 方法留出校准，各种子及总体质量、截断、速度门槛通过才放行。
7. 新 100 test 题 × 3 seed × 5 方法正式测试，排除旧 100 题；按种子轮换方法次序，报告按题聚类区间。uniform/guard-off 各补做同一正式题的 seed42 消融。

主质量上限始终 2048；128 token 只用于性能/严格诊断。记录代码、包、数据、模板哈希，
验证 prompt + 输出 + 草稿不超过 4096。逐请求 flush/fsync 保存，可修复末尾未完成行；
拒绝代码/协议变动和占用端口，有限重试，正常门槛失败不重启。
若使用留出/正式题继续选参，必须更换新的独立验证题；控制器不会自动按这些答案再调参。

## 验收与交付

主指标为正常结束且判对，原判分器保留；截断判对不当作完整答对。
同时报告最终答案标记检查、重复、最后数字回退、平均/中位数/p95 耗时、解码吞吐、MAL 和每小时正常答对数。
对三个主基线分别验收质量和截断不退化、吞吐与平均耗时改善；正式速度区间下界必须大于 1。
5% 吞吐改善仍是工程目标，不将未达 5% 的收益伪称已满足该目标。

- 证据和运行目录：`results/adaptive_optimization_20261002/`；旧 v4 另存 `v4_evidence/`，原目录不改。
- 服务：`remtp-adaptive-round2.service`，user lingering 允许对话/SSH 关闭后继续；当前不配置开机启动。
- 实时进度：`status.json`；`phase_total` 是当前阶段已安排请求，后续门槛通过后才增加任务。
- 自动中文报告：[第二轮结果](RESULT-ADAPTIVE-RAVEN-ROUND2.md)。
- 查看日志：`journalctl --user -u remtp-adaptive-round2.service -f`。

本次实际 CPU 回归为 364 passed、2 skipped；Python 编译、diff 与 user service 校验通过。
冻结源代码在 `results/adaptive_optimization_20261002/frozen_code/`，协议预检通过；
服务已启动，health=200、NRestarts=0。模型 warmup 的三次真实 head 同输入探针
均保持 argmax，logits 最大绝对差为 0.0625、概率 TV 最大约 1.79e-9；
此小样本只覆盖 warmup 输入，不证明全部前缀数值等价，也不代表质量验收通过。
所有更晚的状态以研究日志和自动报告为准；完整退化回放、顺序 Target 比对、正确率和正式收益尚待完成。
