# 第三轮停止原因与结果核查

核查日期：2026-10-04（北京时间）。**当前后台已停止，优化未完成。**

## 直接原因

`remtp-adaptive-round3.service` 于 **13:43:53** 主动退出，exit=2、NRestarts=0、MainPID=0。
控制器完成10条生成诊断和4条独立 Target 评分后报：
`Tree/sequential root mismatch; investigate before tuning`。

| 实际生成前缀长度 | 已生成 token | 最大 top log-odds 误差 | EOS 误差 | 0.15 门槛 |
| --- | ---: | ---: | ---: | --- |
| 109 | 1 | 4.61e-7 | 4.19e-7 | 通过 |
| 238 | 130 | 8.66e-7 | 6.86e-7 | 通过 |
| 157 | 1 | 8.86e-7 | 1.46e-8 | 通过 |
| **287** | **131** | **0.250001** | **0.046876** | **失败** |

失败记录来自 old 配置、训练题2950、seed29542。前缀 SHA256：
`fedd79885fca4a768be8cf7baba88ef7a28cc0c0a47711b76189c159763b33aa`。
控制器保留原门槛，停止在独立概率回放阶段，没有进入消融、选参或正式基线对照。

本轮各服务日志未检出 OOM、CUDA error、traceback 或 fatal；systemd记录内存峰值11.6G。
停止由概率检查明确触发，尚不能据此定位剩余差异具体来自哪一个算子或KV状态。

## 为什么固定验收通过后仍会失败

此前24项root检查和16条祖先评分通过，结论限于固定前缀及其主/备选路径。
完整生成会产生新历史；本轮287-token前缀不同于此前固定286-token前缀。
它经历了真实连续生成和多轮接受/提交，固定前缀独立构造的检查没有证明覆盖这些状态。
**现有修复仍不完整，不能宣称所有 Target 条件概率已一致。**

同配置同seed的完整重放另已通过：题7311两次均1250 token、正常结束、输出hash相同。
这证明本次该请求可重复，不能代替 Target 条件概率正确性。

自动控制器负责执行测试、筛选和断点续跑；概率/协议错误退出2后按设计不自动重启。
反复启动同一冻结协议会再次读到失败gate，不会解决新的不一致。

## 已完成的四题质量诊断

| 指标 | old | anchor4 |
| --- | ---: | ---: |
| 原答案解析判对 | 4/4 | 4/4 |
| 正常结束且判对 | 2/4 | 3/4 |
| 2048-token截断 | 2/4 | 1/4 |
| 输出token总数 | 5829 | 3445 |
| 吞吐 token/s | 2.569 | 2.932 |
| 平均单题秒 | 567.3 | 293.8 |

四题吞吐+14.1%、平均时延-48.2%；这只是训练诊断，两者采用本轮相同公共数值配置。
不能作为 Native/Cactus/SpecCascade 的正式加速证据。
题2950由old的483-token正常结束变为anchor4的2048-token截断，截断问题仍未解决。
128-token strict短诊断不并入上述质量统计。

## 下一定位点

1. 固定本次四个实际前缀和来源，保留新287-token失败记录及原始响应。
2. 用完全相同的287-token条件，比较原生完整prefill、末token decode、独立构造树root。
3. 若新鲜对照通过而真实长生成仍失败，检查接受/提交后的真实输入、位置、有效KV及逐层hidden；
   若新鲜对照也失败，继续定位公共数值/attention执行形状。
4. 修复后同时复核旧固定案例、新失败前缀和真实长生成，均通过才恢复优化；不提高容差。

以上是进一步定位方向，**本次核查没有新GPU执行或新算法修复，也没有重启后台。**

证据：`results/adaptive_optimization_20261004_target_canonical/` 中的status、reproduction_gate、
diagnostics各组requests/summary、sequential_target的prefix_manifest/checks/response。
此前验收与来源仍完整保留在 `results/target_probability_consistency_20261004/canonical_tree/`。
