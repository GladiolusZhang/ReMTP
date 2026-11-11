# MTP概率故障修复与继续优化

> 完成更新（2026-10-07，UTC+8）：本单元于11:03:28正常完成150条请求和32次CUDA条件验收。质量/输出等价通过，速度gate未通过，截断仍11/72。下方“正在验证/进入诊断”属于2026-10-06启动记录；最终分析见[完成结果与中文分析](RESULT-MTP-REPAIRED-REGRESSION-2026-10-07.md)。

日期：2026-10-06；时间统一使用UTC。

## 已经确认的故障

真实GPU算子探针显示，B/1231首轮进入Attention之前的embedding、norm、input projection、QKV、RoPE均有限；当前节点KV写入之后也有限。
Attention首次输出4096个NaN，随后o_proj、MLP和最终隐藏受到污染。B/Anchor该次Q/K/V和写入后的可见KV逐位相同。

原因路径是**单元素树掩码的错误步长使内核越过逻辑掩码范围读取**：

| 环节 | 实际情况 |
| --- | --- |
| vLLM draft builder | 从2×2矩阵取1×1切片，调用`.contiguous()` |
| PyTorch | 1×1视图已经被视为连续，仍保留父矩阵row stride=2 |
| Triton QQ bias读取 | 把row stride同时当有效列数，允许读取逻辑1×1之外的列 |
| 污染对照 | 两份真实Q/KV各加自有Inf padding，均输出4096NaN |
| 紧密掩码对照 | row stride=1；均有限，并与无bias参考输出逐位相等 |

这解释为什么相同有效输入可能因相邻分配内容不同而输出不同。
旧现场相邻字节没有保存，因此不宣称那里实际一定为Inf；已实测逻辑范围外读取路径及其NaN后果。
这也不能直接解释Anchor之前两道错答：此前捕获NaN的是B，Anchor的过强松弛仍需独立质量调查。

## 已实施的修复

1. 在实际MTP draft metadata边界，对不符合“row stride=逻辑宽度”的bias做紧密clone。已经紧密的矩阵不增加分配。
2. QQ bias内核读取增加有效query行/头mask，保护query tile padding。
3. primary/branch/fallback抽样前验证Q有限、非负、正总量。非法Q直接停止且不消费RNG，避免旧提议固定token0、验收却假定均匀Q的分布不一致。
4. 合法Q仍使用原指数竞赛采样；CPU验证输出与随机末状态不变。

没有换模型、软件包或量化方式，没有强制EOS或调整答案提取规则。

## 实际验证与当前后台单元

- 2条32token真实算子探针完成，独立服务各8token预热。
- 修复前后各8次GPU注意力对照完成，原始文件分开保留；只补query mask不能修复错误列宽，紧密metadata必须同时存在。
- 120项相关CPU测试通过（11.60秒），22项新控制器/概率/metadata/audit测试通过（2.83秒）；py_compile、bash -n、diff检查通过。
- 新冻结目录：`results/mtp_repaired_regression_20261006/frozen_code`；check-only通过，后台单元`remtp-mtp-repaired-regression-20261006.service`已启动，Restart=no、上限72小时。
- 新单元先2条32token修复smoke、32次新CUDA条件验收、4条完整2048诊断。Q/隐藏有限、观测边界、验收等价、4实际Target首根P逐位一致全部通过后，自动运行原未使用144条训练对照。
- 训练用固定24题×3种子×B/懒残差；每种子块立即核对完整输出、正确率和截断；不启用trace，不自动采用，不运行独立留出/正式测试。

**B与Anchor两条修复后smoke、16帧×2=32次新CUDA条件验收均实际通过；现进入完整2048诊断，随后依门槛继续训练，以[实时中文报告](RESULT-MTP-PROBABILITY-REPAIR-LIVE.md)为准。**

B首轮真实Attention与MTP隐藏均0个非有限值，实际bias stride=(1,1)，原始Q为9×151680且全部有限；前8轮Q/P、每个tree pass隐藏和proposal→verify边界均通过。

进度：

```bash
cd /home/apex/ReMTP-residual
bash scripts/status_mtp_repaired_regression.sh
```

关闭对话后systemd用户服务继续运行；任一新门槛失败会停止，不自动重复试错。

## 正确率、截断与速度的结论边界

当前结果证明并修复一条具体MTP数值故障路径，还没有证明总体正确率或相对三个基准的速度改善。
Q合法性检查增加同步，实际代价随训练结果报告。懒残差只省计算，完整输出相等时正确率与截断也相等，不能把它当截断修复。

下一项质量优化应在有限Q下约束对低Target概率token的过强松弛：现有`adaptive_max_ratio=1.5`限制的是分配TV相对原TV，**不是H/P的上限**，仍可能出现H/P数百倍。
下一候选应明确限制局部提升与深层松弛，并检查EOS质量及后段重复；通过独立质量/截断门槛后才追求更高MAL。
正式比较之前仍需补新源码完整Target/logits/KV/长上下文验证与独立留出；本次首根P gate不替代它们。

证据目录：`results/mtp_operator_20261006/`（算子与条件对照）和`results/mtp_repaired_regression_20261006/`（新后台单元）。
改动说明：[专门记录](upstream-changes/target-probability-consistency/mtp-operator-repair-2026-10-06.md)；计划：[执行计划](../plan/mtp-probability-repair-2026-10-06.md)。
