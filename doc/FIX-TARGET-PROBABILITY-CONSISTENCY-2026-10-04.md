# Target 概率一致性修复与优化恢复

> 深入复查已完成60条新鲜对照，但持续生成问题尚未修复。
> 当前结论与后续动作见 [详细调查](ANALYSIS-TARGET-CONSISTENCY-DEEP-AUDIT-2026-10-04.md)。

> **最新状态：第三轮已于2026-10-04 13:43:53（北京时间）停止。** 真实长生成的新287-token前缀
> top误差0.25超过0.15门槛；下文“验收通过/已启动”是固定案例及启动时的历史结论。
> 当前修复尚未覆盖全部生成状态，详见[第三轮停止核查](STATUS-ADAPTIVE-RAVEN-ROUND3-2026-10-04.md)。

日期：2026-10-04（北京时间）。**状态：固定前缀及祖先概率验收通过；第三轮无人值守优化已启动。**

## 当前结果

- 12:28:14 完成最终 GPU 验收：24 项 D3/D4 root 检查 + 16 条主/备选祖先评分全部通过。
- root 与祖先的最大 top / EOS log-odds 差均为 **0**，保留原 0.15 门槛。
- 完整词表 TV：root 最大 8.31e-12，祖先最大 1.88e-10。
- 完整 CPU 回归：**379 通过、2 跳过**（46.88 秒）；随后仅复用校验的小修复重跑相关 6 测试通过。
- `remtp-adaptive-round3.service` 已启用并启动，关闭对话后继续运行；协议/质量失败会停止。
- 启动收尾：模型实际 health=200，warmup 完成，首条完整 2048-budget 质量请求正在生成，NRestarts=0。

这说明当前固定条件前缀检查已修复；**仍未证明正确率不下降、截断全面改善或快于三个基线。**
这些结论由正在运行的新优化和正式对照给出。

## 为什么前一轮停止

前一轮树模式与独立 Target 回放的四个 root 前缀误差为
0.125 / 0.1875 / 0.25 / 0.375，后三项超过原 0.15 门槛。
同 seed 输出可重复，并不能证明条件概率相同，因此不能直接继续选参。

本次核对了真实 token、位置、hidden、原始 logits、温度语义和完整词表 TV，
没有提高误差容差，也没有用强制结束改善质量指标。

## 定位与已经完成的验证

| 对照 | 实际结果 | 结论 |
| --- | --- | --- |
| 默认原生 Target，48 次请求 | prefill/decode log-odds 差最高 0.625，TV 最高约 0.05144 | 原生有限精度计算自身也存在明显执行形状差异 |
| CUDA FP4 GEMM 隔离 | 单改 BNB 路由或 BF16 归约均差 1.0；两者一起 M1/M13 差 0 | 两项都需要，但不足以保证全模型一致 |
| 两项组合，全模型 48 次请求 | 部分前缀仍失败 | 未采用为最终修复 |
| 再加固定 64 行，48 次原生请求 | cache-off 四例 hidden / top 差 0；cache-on 末例 top 差 0.125 | 固定线性层计算形状有效 |
| 固定行，但树与 FLASH_ATTN 参考比较 | D3/D4 实际 16 条 root 请求，仍有 top 0.375 / EOS 0.328125 | attention 后端不同不能作为严格参考 |
| 统一 TRITON 2D attention，48 次原生请求 | cache-off 四例 hidden / top 差 0 | 避免原生单 token 的 3D 分段 softmax 路径 |
| 新鲜 D3/D4 树 root，缓存关 | 实际 16 条请求、24 项检查全部通过，top / EOS 差 0，改变备选 seed root 不变 | 通过当前四个固定 root 前缀检查 |
| D3 树 root，缓存开 | 实际 8 条请求，286-token 前缀 top 差 0.25 | 原门槛失败；本轮全部方法采用缓存关 |
| 独立祖先路径回放，缓存关 | 实际 12 条路径；D3 主路径 160-token 前缀 top 差 0.1875，TV 6.90e-6 | root 一致仍不足以通过祖先检查；未恢复优化 |
| FP32 attention 候选，原生缓存关 | 实际 24 条请求，四例 prefill/decode hidden / top 差 0 | 最新公共数值候选原生对照通过 |
| FP32 attention 候选，树/祖先 | 实际 16 root 请求全部通过；16 祖先中 D3 全通过，D4 的 161-token 前缀 top 差 0.25 / EOS 0.15234375 / TV 0.06682 | FP32 乘积仍不足以修复所有路径，整体 gate 失败 |
| 再统一祖先 KV 读取顺序 | 实际 16 root 请求、24 项检查 + 16 祖先评分全部通过，top / EOS 差均为 0 | 最终固定条件前缀验收通过，恢复后续优化 |

这里的“缓存关”只关闭**跨请求前缀复用**，普通解码 KV 缓存仍保留。
所有失败候选、冻结代码、原始输出和失败 gate 都保存在独立目录中。

## 当前候选修复

1. FP4 权重仍以 FP4 存储；在 sm87 上统一逐层反量化后执行 BF16 linear。
2. BF16 GEMM 关闭低精度归约，使用固定 64 行块；不改变实际 attention token 数。
3. 独立 Target 和三个链基线使用 TRITON_ATTN；树继续 TREE_ATTN，统一 2D attention。
4. 关闭跨请求前缀缓存，所有方法一致设置。
5. 本地复制 vLLM 0.18.0 的 2D kernel，**只将 softmax P 与 V 的乘积保留 FP32 / IEEE**。
   保留原 KV 索引、mask、位置、返回 dtype 和 launch 参数；不编辑安装包源码。
6. 树 Target 验证通过 token-level block table 将每个节点的**已提交前缀 + 实际祖先 + 自己**
   按连续逻辑顺序读取。一个节点映射为一个虚拟单 token query，物理 KV 仅作共享 view，
   没有整份缓存复制或原地改写；由索引保证兄弟/后代不可见。
   索引布局只在一次 forward 的各层复用，新 prepare 清空，避免跨请求残留。

第 5 项改善 D3，但 D4 仍失败，因此新增第 6 项消除 BFS 与连续祖先的归约顺序差异。
上表缓存开启的失败发生在新增第 5 项之前，
最新候选尚未重新验证缓存开启，所以继续采用已验证的关闭配置。
BFS 物理槽与连续祖先槽跨越不同 attention tile，
原 kernel 每 tile 将 P 转为 BF16，在线 softmax 的分块会改变量化时机。
FP32 乘积真实回放仍失败，不能单独当作完整修复；第 6 项的全模型回放全部通过。
CUDA 随机隔离用例原版与候选均相同，只证明候选能够编译运行，不能证明修复有效。

## 最终验收与后台安排

原生 cache-off 24 条请求已完成；第 6 项只作用于树验证，顺序参考算术不变。
新冻结代码下 D3/D4 共 16 条 root 请求 → 16 条主/备选祖先评分已全部完成并通过。
选择规则提前固定，不按是否通过挑样本。
无 root 备选时取内部第一条备选路径，并核对逻辑位置。

已启动 `remtp-adaptive-round3.service`，依次进行：

1. 四题完整 **2048-token** 预算诊断和跨服务同 seed 重放。
2. 生成中的真实前缀与独立 Target 再次比较，保留 0.15 门槛。
3. 实现/树深消融、质量保护选参、24 题回归和独立留出。
4. 对 Native、Cactus、SpecCascade 使用相同公共数值配置重新测试。
5. 新抽取正式测试题、多种子比较和中文自动报告。

质量评价同时检查正确率、正常结束且判对、截断数；速度同时检查 token/s 与整题时延。
协议/质量失败会停止并保存证据。服务由用户级 systemd 管理，可以关闭对话继续运行。

## 证据和限制

- 诊断证据：`results/target_probability_consistency_20261004/`。
- 顺序参考：`fp32_reference/`；最终树候选：`canonical_tree/`，内部 `probe_code/` 冻结源码。
- `fp32_tree/` 保留整体失败；`adaptive_optimization_20261004_target_fixed/` 仅预检，未启动。
- 后续优化采用新目录 `results/adaptive_optimization_20261004_target_canonical/`。
- 专门改动记录：[target-probability-consistency/README.md](upstream-changes/target-probability-consistency/README.md)。

固定前缀通过不是所有上下文的数学证明，也不是正确率/加速目标已完成。
固定行计算和 FP32 attention 可能增加成本；最终性能必须由公共配置下的新实验给出，
不能沿用旧基线速度。当前没有发生 OOM，但仍按单 GPU、一次一个服务执行。

## 查看优化进度

```bash
cd /home/apex/ReMTP-residual
bash scripts/status_adaptive_round3.sh
```

实时日志：`journalctl --user -u remtp-adaptive-round3.service -f`。
中文自动报告：[第三轮优化报告](RESULT-ADAPTIVE-RAVEN-ROUND3.md)。
脚本中的已安排总数随通过的门槛增加，不是最终全部请求数量。
