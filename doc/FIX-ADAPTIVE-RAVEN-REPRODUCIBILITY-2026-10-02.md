# Adaptive RAVEN 草稿复现问题：原因与修复

> 2026-10-04核查：下面“长回放已启动”为历史快照。完整同seed重放已经通过，
> 后台随后于10月2日15:17:27北京时间因Target树/顺序概率对照失败停止；
> 当前没有继续运行，见[最新实验分析](STATUS-ADAPTIVE-RAVEN-2026-10-04.md)。

日期：2026-10-02。状态：已定位并修复，7请求真实 GPU 短回放全部通过；后台长回放已启动。

## 原因

旧流程停止于同题、同种子重放失败。第一次实验的7311题输出592个token，重启后为1700个，
两次都答对；但不能把这种差异直接解释为方法质量或速度变化。

本次增加真实张量诊断，复现了缓存开启时的请求顺序依赖：

- 第一轮 root hidden、logits、完整 Q、RNG 与实际 MTP prefix KV 完全一致。
- 第二轮 root 位置114，真实有效前缀只有115个token；Target传入的序列长度仍是123。
- 草稿沿用了 `123 + 当前树节点数`，额外把8个槽当作普通上下文：其中既可能有刚写入的
  本轮树节点，也可能有前一轮没有接受的节点或旧内容；树mask的作用位置随之错位。
- 两次真实前缀KV和草稿输入hidden仍完全相同，首次三节点草稿输出hidden最大差值却为4.0，
  整个Q矩阵最大概率差为0.0373194。差异先出现在草稿，随后改变接受路径和随机数消耗。

前缀缓存/分配历史使错误可见槽中的内容不同；缓存关闭时本次短回放恰好一致，
但正确实现应该只读取有效前缀，不能依赖旧槽碰巧相同。

## 修复

`remtp/fixed6_vllm.py` 的累计草稿 attention 长度改为：

```text
seq_len = root_position + 1 + 当前累计树节点数
有效上下文长度 = seq_len - query_len = root_position + 1
```

同步最大序列长度、CPU query offsets，清除旧长度影子；物理BFS写槽与逻辑RoPE位置规则不变。
没有改Target模型、量化、接受公式、松弛预算或停止条件。

另增加显式前缀缓存开关、默认关闭的有界hidden/Q/KV/RNG诊断，以及独立回放控制器。
控制器恢复时也会拒绝跳过已保存的失败Target顺序概率检查。

## 实际验证

| 检查 | 结果 |
| --- | --- |
| 旧代码、缓存开启、同进程题7311→2629→7311，各128token | 输出不同；前8轮草稿在第5次提案前已分叉 |
| 旧代码、缓存关闭、同进程重复 | 输出相同；前8次提案Q的最大TV=0 |
| 旧代码、缓存关闭、独立服务重启 | 输出相同；前8次提案Q的最大TV=0 |
| 修复后完整CPU测试 | 369 passed、2 skipped；随后新增缓存污染回归，相关6项通过 |
| Python编译、shell语法、diff空白检查 | 通过 |
| 修复后真实GPU、缓存开启、同进程顺序重放 | 通过；128token输出相同、8次提案Q最大TV=0；24次树hidden/KV完全一致、RNG一致 |
| 修复后真实GPU、缓存关闭、同进程顺序重放 | 通过；128token输出相同、8次提案Q最大TV=0、root hidden最大差=0 |
| 修复后真实GPU、缓存关闭、独立重启 | 通过；128token输出相同、8次提案Q最大TV=0、root hidden最大差=0 |

修复检查于北京时间2026-10-02 14:33:33完成，`remtp-repro-fixed.service`
自动退出、ExecMainStatus=0；三个对照全部通过 `--require-match` 门槛。

旧证据：`results/adaptive_repro_fix_20261002/`。
修复冻结代码与新证据：该目录的 `fixed/`；服务 `remtp-repro-fixed.service`。

## 后续与限制

短回放验证的是状态复现，不是正确率或性能。通过后才恢复2048-token长回放、
独立Target概率检查、选参与质量门槛。新正式记录使用新目录，保留旧失败记录。

下一轮目录为 `results/adaptive_optimization_20261002_fixed/`，代码已冻结、上下文预检通过，
独立服务 `remtp-adaptive-round2-fixed.service` 已启动，MainPID=442893、NRestarts=0。前缀缓存保持显式开启，
深度张量快照关闭；完整长回放失败仍直接停止，不能进入选参。
自动中文报告见[修复后第二轮报告](RESULT-ADAPTIVE-RAVEN-ROUND2-FIXED.md)。
后台顺序：旧配置4题→7311独立重启长回放→anchor4诊断→严格/Target概率检查→选参、质量门槛和正式对照。
关闭对话后服务继续；不会用旧失败日志充当新代码的正式结果。

该问题可能影响截断与测速，但尚未证明解释所有截断；重复句段和松弛降低EOS质量
仍需要长输出对照。修复后不能沿用旧统计证明加速或准确率提升。

查看当前修复检查：

```bash
systemctl --user status remtp-repro-fixed.service --no-pager
cat /home/apex/ReMTP-residual/results/adaptive_repro_fix_20261002/fixed/reproduction_checks.json
systemctl --user status remtp-adaptive-round2-fixed.service --no-pager
cat /home/apex/ReMTP-residual/results/adaptive_optimization_20261002_fixed/status.json
```
