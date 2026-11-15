# RAVEN v4：按目标模型决定松弛位置（目标支持度门槛），W4A16 + MAXN 初步结果（2026-11-13）

## 1. 出发点与设计

- Cactus 这类方案是围绕草稿给出的 token 做松弛：无论目标模型怎么看，草稿提出哪个 token 就给哪个 token 加成。它的加成 √(2δp(1−p)) 在 p 很小时远大于 p，例如目标概率 6e-5 的 token，只要草稿给它的概率低于 1.1%，就必定被接受。
- MTP 头作为草稿并不可靠。日志显示，Cactus 接受目标概率 <1e-3 的 token 的频率是精确采样的约 35 倍（`doc/RESULT-W4A16-MAXN-HUMANEVAL-SUITE-2026-11-13.md` 第 5 节）。
- RAVEN v3 的自适应规划会把 Cactus 在饱和位置用不满的名义额度回收后重新分配，按实际接受计算比 Cactus 放宽得更多（每位置多接受的质量多 7%~9%，低支持度接受多约 20%）。这偏离了"按目标模型松弛"的初衷。
- **v4：由目标模型决定在哪里松弛、松弛多少。**
  - 目标支持度门槛：只有当目标给草稿 token 的概率 p(y) ≥ 1% 时才允许松弛，否则 H = P，按精确规则验证，修正分布也是精确残差。门槛只看目标分布，与草稿概率 q 无关；单元测试检查了"换一个草稿分布，被门槛挡住的位置完全不变"。
  - 每个位置的额度是 Cactus 的 √(2δp(1−p))，只取决于目标概率 p。
  - **v4-safe**：自适应上限取 1 倍名义额度，每个位置的接受概率都不超过 Cactus 在同一 token 上的接受概率；门槛以下完全精确。
  - **v4-gated**：Cactus 在门槛以下位置实际会用掉的额度，只允许挪给目标支持的位置（单个位置最多 1.5 倍），整轮实际松弛总量不超过 Cactus。
- 代码：
  - `remtp/dynamic_mtp_tree.py`：
    - `DynamicTreeConfig` 新增三个字段：`relaxation_min_target_probability`、`relaxation_min_target_ratio`、`adaptive_relaxation_pool`；
    - `adaptive_trunk_relaxation_plan` 内实现门槛和 gated 额度池；
    - 修正了 docstring 中"实际松弛总量不超过静态控制"的错误说法（原实现只对名义额度成立）。
  - 环境变量：`REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_PROB`、`REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_RATIO`、`REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_POOL`。
  - 默认值保持旧行为。快速验证路径直接使用规划结果，不需要回退到参考实现。
- 测试 `tests/test_target_support_gate.py`（15 项）覆盖：
  - 参数校验；
  - 默认值不改变旧规划；
  - 门槛只由目标决定；
  - safe 变体每个节点的接受概率 ≤ Cactus，门槛以下等于 min(1,p/q)；
  - gated 变体实际松弛总量 ≤ Cactus，且额度确实挪到了被支持的位置；
  - 两个变体的快速验证与参考实现逐位相同（含随机数状态）。
- 已有的 133 项树验证、快速路径、概率上限、target_odds 测试全部通过。

## 2. 离线选门槛（用 2026-11-12 套件的 RAVEN 日志，按记录的随机数重放主干判定）

| 门槛 | MAL 估计（v3 实测 3.373） | p<1e-2 接受/千 token（v3 为 13.1） |
| --- | ---: | ---: |
| p ≥ 1e-3 | 3.354 | 8.3 |
| **p ≥ 1e-2** | **3.325** | **0.88**（精确采样约 0.9） |
| p ≥ 3e-2 | 3.297 | 0.87 |
| p/p_max ≥ 1e-2 | 3.330 | 2.2 |
| p/p_max ≥ 3e-2 | 3.301 | 0.87 |

选择绝对门槛 p ≥ 1%：在同样的 MAL 代价下，它去掉低支持度接受的效果最好。

## 3. 小规模 A/B（HumanEval 前 30 题，冻结 prompt 与种子，输出上限 512，W4A16 RTN，MAXN，`results/raven_v4_ab_20261113/`）

五组用同一个模型、同一批题目和种子、同样的预算，运行顺序交错，全部开启逐轮日志。

| 方法 | MAL | 每轮秒 | decode tok/s | 每位置多接受的质量 | p<1e-3 接受/千 token | p<1e-2 | p<0.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native（精确） | 2.608 | 0.1498 | 17.43 | 0.000 | 0.00 | 0.80 | 5.6 |
| Cactus | 2.909 | 0.1504 | 19.33 | 0.058 | 3.33 | 8.85 | 27.0 |
| RAVEN v3 | 3.214 | 0.1814 | 17.71 | 0.072 | 5.19 | 11.47 | 30.6 |
| **RAVEN v4-safe** | **3.128** | 0.1823 | 17.15 | **0.049** | **0.09** | **0.71** | 19.1 |
| RAVEN v4-gated | 3.131 | 0.1824 | 17.16 | 0.049 | 0.09 | 0.61 | 19.0 |

（对照：同一批草稿 token 按精确规则，p<1e-2 的期望接受约 0.5~0.9 个/千 token。）

- **质量（分布层面）**：
  - v4 接受目标概率低于 1e-3 和 1e-2 的 token 的频率，已经降到精确采样的水平：Cactus 是它的约 37 倍和约 13 倍。
  - 每个位置多接受的质量比 Cactus 少 15%。
  - 也就是说，v4 对目标分布的偏离严格小于 Cactus，而且完全不在目标认为不合理的 token 上松弛。
- **MAL**：
  - v4-safe 比 Cactus 高 7.5%，比 v3 低 2.7%。
  - 用本次 v3 的日志重放 v4-safe 规则，估计 3.167（3.149~3.186），实测 3.128，在 30 题的采样波动范围内一致。
  - gated 与 safe 几乎相同，因为门槛以下可挪用的额度很少。建议采用规则更简单、保证更强的 v4-safe。
- **pass@1**：Native 12/30、Cactus 13/30、v3 11/30、v4-safe 14/30、v4-gated 14/30。30 题、512 token 上限，没有统计意义，只作记录。
- **速度**：RAVEN 每轮比 Cactus 多 32 ms，v4-safe 的 decode 比 Cactus 低 11%。要在速度上超过 Cactus，每轮需要降到 0.150 × 3.128 / 2.909 ≈ 0.161 s 以下，也就是至少砍掉 21 ms 的额外开销。

## 4. W4A16 + MAXN 分段计时（GSM8K 17/353/856，各 32 轮，`results/w4a16_profile_20261113/`）

| 每轮 CUDA 时间 | Cactus | RAVEN v3 |
| --- | ---: | ---: |
| 目标前向（Marlin） | 73.1 ms | 63.5 ms |
| 目标 LM head（BF16） | 13.7 ms | 13.7 ms |
| 草稿 MTP 层（3 步） | 16.4 ms | 20.4 ms |
| 草稿 LM head（3 步） | 40.8 ms | 40.8 ms |
| 树提议中的非模型部分 | — | 约 23 ms（墙钟约 37 ms） |
| 验证（墙钟） | 3.1 ms | 23.4 ms |

- BF16 的 LM head 每次计算 13.6 ms（约 91 GB/s，已到带宽上限），每轮 4 次，约占一轮的 36%。它是所有推测方法共同的瓶颈，与验证方法无关。
- RAVEN 多出的时间基本都是主机端开销：验证约 +20 ms，树提议约 +15~18 ms。分段计时已补充更细的钩子（`remtp/runtime_profile.py`），用于下一步定位。

## 5. 主机端开销：批量候选选择与并列修复（输出逐 token 不变）

- 细分计时显示：
  - RAVEN 每轮的模型计算约 139 ms：目标前向 61、目标 LM head 13.6、草稿 MTP 层 23、草稿 LM head 40.8，反而略少于 Cactus 的约 144 ms；
  - 但每轮墙钟 177~180 ms，Cactus 是 150 ms。
  - RAVEN 多出的约 38 ms 是 GPU 空等 CPU 的时间：树提议每层都要读回 GPU 结果，由 CPU 决定候选再发射下一步；验证约 9 ms 的 Python 逻辑也在关键路径上。
- **批量候选选择**（`REMTP_DYNAMIC_TREE_BATCH_SELECT=1`，`raven_fast_path.batched_trunk_children`）：
  - 做法：同一层所有父节点照原顺序采样并发射运算，一次读回；只用于诊断的计数在提议结束时一次读回。
  - 测试：11 项等价测试（含并列、无合格 token、one-hot、阈值比较的 float32 语义）。
  - 实机：3 题输出逐 token 相同，每轮 0.1801 → 0.1781 s（−1.1%）。
- **并列修复**（`raven_fast_path._relevant_tie`）：
  - 现象：原快速选择在 top-k 里出现任何并列都会回退到参考实现，回退要对整个词表做稳定排序，再加多次同步。T=0.6 下草稿分布很尖，尾部常常同时下溢成 0，于是每轮平均回退 0.94 次。
  - 依据：低于 `min_draft_prob` 的名次永远不会被选中，只在可选的值之间检查并列，决策和诊断都与参考实现相同（新测试覆盖，原"并列必回退"测试已改为"可选值并列才回退"）。
  - 实机：输出逐 token 相同，每轮 0.1781 → 0.1771 s。
- 合计每轮只省约 3 ms（−1.7%）。原因是之前计时里那些墙钟大多是在等前面排队的 GPU 计算，同步次数本身不是主要成本。剩下约 35 ms 分散在每层的 Python 决策和验证逻辑里；要再大幅下降，需要结构性改动，例如在 GPU 上用固定形状构造树、使用 CUDA graph（目前所有方法都是 eager 模式）。

## 6. 校准的 GPTQ 模型（`models/FastMTP-W4A16-GPTQ`，`scripts/quantize_fastmtp_w4a16_gptq.py`）

- 方法：在本机逐层流式量化，每层按 q/k/v → o → gate/up → down 的顺序（true-sequential），组内缩放用 MSE 网格选截断系数，阻尼 1%，组大小 128，对称 4 位，GPTQ v1 格式（vLLM `gptq_marlin`）。
- 校准数据：128×512 个 token，取自 venv 里的 Python 源码和包说明文档，按路径哈希与留出集分开；一半套用聊天模板。不含任何评测数据。
- 耗时：38 分钟（每层约 62 秒）。只量化 36 层主干的线性层；词嵌入、LM head、norm、bias 和 MTP 层保留 BF16，草稿与之前完全相同。
- 正确性：
  - 层前向与 transformers Qwen2DecoderLayer 的最大差 1.4e-4；
  - 打包往返逐位一致；
  - 从导出的 checkpoint 反量化后的评测结果，与量化脚本内的结果逐位一致；
  - vLLM 启动日志确认使用 `MarlinLinearKernel`。
- 留出文本（16×512 token，与校准集文件不重叠）上相对 BF16 的误差：

| 模型 | KL（T=1） | KL（T=0.6） | top-1 一致率 | NLL（BF16 为 1.968） |
| --- | ---: | ---: | ---: | ---: |
| **GPTQ（新）** | **0.123** | **0.194** | **88.6%** | **1.998** |
| bnb NF4（vLLM 实际部署配置，HumanEval 约 62%） | 0.136 | 0.214 | 88.1% | 2.015 |
| RTN（11-12 套件所用，HumanEval 约 40%） | 0.260 | 0.404 | 84.4% | 2.067 |
| bnb FP4（仿真，仅作参照） | 0.245 | 0.381 | 84.0% | 2.061 |

- 更正：vLLM 对未量化 checkpoint 的 bitsandbytes 现场量化使用 NF4（`quant_type="nf4", compress_statistics=True`），不是 FP4，之前文档里写的"bnb FP4"有误。
- 结论：GPTQ 的输出误差略小于原来部署的 NF4；RTN 的误差是它们的两倍，这解释了 RTN 在 HumanEval 上掉的约 20 个百分点。今后统一使用 GPTQ 模型。

## 7. GPTQ 模型上的公平对比（HumanEval 前 30 题，冻结 prompt 与种子，输出上限 512，MAXN，`results/gptq_ab_20261113/`）

四组用同一个 GPTQ 模型、同一批题目和种子、同样的预算，全部开启逐轮日志。RAVEN 为 v4-safe，开启批量选择和并列修复。

| 方法 | pass@1 | 截断 | 平均输出 token | MAL | 每轮秒 | decode tok/s | 每位置多接受的质量 | p<1e-3 接受/千 token | p<1e-2 | p<0.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native（精确） | 24/30 | 2 | 167 | 2.883 | 0.1501 | 19.24 | 0.000 | 0.00 | 0.20 | 3.4 |
| Cactus | 24/30 | 2 | 162 | 3.131 | 0.1502 | 20.87 | 0.032 | 2.46 | 5.34 | 15.8 |
| SpecCascade | 24/30 | 1 | 156 | 3.001 | 0.1494 | 20.10 | 0.007 | 0.00 | 0.00 | 0.4 |
| **RAVEN v4-safe** | 23/30 | 0 | 153 | **3.320** | 0.1758 | 18.88 | 0.023 | **0.00** | 0.65 | 8.7 |

（同一批草稿按精确规则，p<1e-2 的期望接受为 0.34~0.40 个/千 token。）

- **模型质量恢复了。** 同样 30 题、同样 512 上限：
  - Native pass@1 从 RTN 的 12/30 升到 24/30；
  - 截断从 11 题降到 2 题；
  - 平均输出从 376 token 降到 167 token，不再出现长篇反复自我检查。
- **分布层面：**
  - Cactus 仍然接受目标概率 <1e-3 的 token，每千 token 2.5 个，精确采样为 0；
  - v4-safe 是 0，p<1e-2 的接受也在精确水平附近；
  - v4 每位置多接受的质量比 Cactus 少 28%。
- **MAL：** v4-safe 比 Cactus 高 6.0%，比 Native 高 15%。
- **pass@1：** 四组都在 23~24/30，30 题无法区分；GPTQ 模型输出短（平均约 160 token），每组只有约 4.8k 个 token，分布指标的计数也较少。
- **速度：** v4-safe 每轮比 Cactus 多 26 ms，decode 低 9.5%。

## 8. 下一步

1. 结构性降低 RAVEN 主机端开销（剩余约 35 ms/轮），包括 GPU 端固定形状的树构造、验证逻辑向量化、CUDA graph。目标是每轮 ≤ 0.150 × 3.32 / 3.13 ≈ 0.159 s。
2. 草稿 LM head 压缩（量化副本或裁剪词表）对所有方法同样适用，比较时各方法都要用。
3. 质量结论：在 GPTQ 模型上，用 2048 预算和更多题目或种子做 v4-safe 对 Cactus 的配对比较；大样本放在 4090 上。论文中加入低支持度接受指标。

## 9. 工程优化：只改实现、不改算法（2026-11-13 晚）

原则：算法、草稿分布、提议规则和接受规则都不动；每项优化都要求同种子输出逐 token 相同。

- **已完成的主机端优化**（输出逐 token 相同，已用旧代码 worktree 做干净对比）：
  - 批量候选选择；
  - 并列回退只看可选值；
  - 候选选择每个父节点的设备运算从约 72 个减到约 22 个（GPU 上 799 个父节点逐位一致）；
  - `normalize_rows_nosync` 少发射运算，均匀分布改为标量；
  - `_boost_known` 去掉多余的 clone；
  - 自适应规划改为读回后在 CPU 张量上用同一运算序列计算（GPU 上 300 次与参考实现逐位一致，含随机数状态）。
- **干净的端到端测量**（GSM8K 3 题，416 轮，不开任何计时器，`results/engopt_ab_20261113/`）：

  | 代码 | 每轮 |
  | --- | ---: |
  | 旧代码（HEAD 44dec73） | 0.1813 s |
  | 新代码 | 0.1802 s |
  | 新代码 + 批量选择 | 0.1792 s（−1.1%） |

  三组输出逐 token 相同。
- 微基准（空闲 GPU）显示候选选择每层 9.5 → 4.2 ms、验证 7.3 → 5.1 ms，但端到端只省约 2 ms。原因是候选选择的运算发射时 GPU 还在算前面排队的 LM head，CPU 发射时间本来就被掩盖了。
- 更正：此前几次 `AB_PROFILE=1` 的对比里，每次都新增了计时钩子（包括每轮调用十几次的小函数），计时开销越来越大，不能用来比较优化前后。以后衡量速度一律不开计时器。
- **CPU 时间线**（torch.profiler，`results/torch_trace_20261113/`）。这台 Jetson 的驱动设置是 `RmProfilingAdminOnly=1`，没有 root 抓不到 GPU kernel，只有 CPU 事件。以下数字是开着 profiler 时的每轮 CPU 时间（偏大）：

  | | Cactus | RAVEN v4 |
  | --- | ---: | ---: |
  | 目标前向的发射（内部没有任何同步） | 100.7 ms | 85.9 ms |
  | 提议 | 70.3 ms | 86.2 ms |
  | 验证 | 4.3 ms | 9.4 ms |

  - 目标前向每层发射约 30 个运算：4 次 Marlin、注意力（Triton，Python 启动开销大）、KV 更新、norm、rotary 等。运算本身约 57 ms，运算之间的 Python（vLLM 模块调用、Triton JIT 启动、张量分配）约 43 ms，没有单一热点。
  - 不开 profiler 时目标前向的发射约 72 ms，而按带宽估算 GPU 实际计算约 44 ms。**所有方法（包括 AR）都卡在 eager 模式的 CPU 发射上。**
- **结论**：
  1. 最大的工程杠杆是 CUDA graph（目前所有方法都设了 `ENFORCE_EAGER=1`）。目标前向每轮预计可省 20~30 ms，对所有方法（包括 AR）都有效，需要测试与现有钩子是否兼容。
  2. RAVEN 相对 Cactus 每轮仍多约 29 ms，来自树提议里的 Python（`_run_dynamic_tree_nodes` 每层的元数据准备和发射、候选决策之后的 CPU 段）和验证及其前后处理。CUDA graph 之后 Cactus 的每轮会变短，这部分固定开销的占比会更大，所以两条线都要做。

## 10. CUDA graph、CPU 调频与去同步（2026-11-14 凌晨）

原则不变：只改实现，算法、草稿分布、提议规则和接受规则都不动；每项改动都要求同种子输出逐 token 相同。下面所有对比都是 GSM8K 3 题（17/353/856）、GPTQ 模型、MAXN、不开任何计时器，数据在 `results/cudagraph_ab_20261113/`。

### 10.1 CUDA graph

- 链式方法（Cactus/Native/AR）：`ENFORCE_EAGER=0` 加 `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}`（AR 用 `[1]`）。不经过 torch.compile，重放的就是 eager 下的同一批内核，prefill 仍走 eager。服务脚本新增 `REMTP_COMPILATION_CONFIG` 透传。
- RAVEN：vLLM 0.18 的 TREE_ATTN 不支持 CUDA graph，新增 `remtp/tree_cudagraph.py`：
  - `REMTP_TREE_HOST_METADATA=1`：vLLM 在树注意力第一层里用两次 `.item()` 惰性计算子元数据（每次目标和草稿前向各两次同步）。单请求时这两个最大值在主机上已有，改为在 `build()` 里直接填好；内核拿到的张量是同一批视图。`REMTP_TREE_HOST_METADATA_CHECK=1` 会逐次与设备值比对（验证用，跑完 410 轮无差异）。
  - `REMTP_TREE_CUDAGRAPH=1`：目标树掩码放进固定的 (N+1)×(N+1) 缓冲区，每轮把 n×n 掩码拷进左上角。Triton 2D 内核只按行步长寻址掩码，块外的键本来就被因果掩码挡掉，所以 logits 逐位不变。TREE_ATTN 声明完整图支持（`max_num_seqs=1` 时每批只有一个请求）；用 FULL 模式，捕获尺寸 1..10。
  - `tests/test_tree_cudagraph.py`：10 项，含用 vLLM 原版和我们的 FP32 版 Triton 内核在 GPU 上逐位比对（上下文 5/37/200，树大小 2..10）。
- 结果：Cactus 每轮 0.1487 → 0.1321 s（−11%），RAVEN 却几乎没变（去同步后 0.1697 → 0.1738 s）。

### 10.2 原因：CPU 调频

- GPU 调频器 `nvhost_podgov`，解码期间一直在 918 MHz（每个请求都记录了 `trans_stat`），不是问题。
- CPU 调频器是 `schedutil`（729–1984 MHz），没有 root 无法修改。图模式下引擎线程发射完整个目标前向只要 0.2 ms，随后要等 GPU 约 50 ms。CUDA 默认让等待的线程睡眠，`schedutil` 随即把 CPU 集群降到约 740 MHz（最低档）；醒来后的验证、建树、草稿发射都在低频下运行，同一段 Python 的耗时约为 eager 时的 2 倍（CPU 时间线：验证 14.2 vs 6.9 ms，草稿前向发射 7.5 vs 3.5 ms）。
- 新增 `remtp/cuda_spin_wait.py`（`REMTP_CUDA_SPIN_WAIT=1`）：在 vLLM 初始化前给主上下文设 `CU_CTX_SCHED_SPIN`，等待时自旋而不是睡眠，CPU 集群保持在约 1984 MHz。只改等待方式，不改任何计算。
- 补充：即使在 eager 下，CPU 集群平均也只有约 1000–1300 MHz。此前 Jetson 上的所有计时都是在 CPU 未满频时测的，对所有方法一样。

### 10.3 去掉 RAVEN 的不必要同步

- 用 `torch.cuda.set_sync_debug_mode` 在服务里统计每轮的同步位置（`remtp/host_timeline.py`，`REMTP_HOST_TIMELINE_SYNC=1`）：Cactus 每轮 4 次，RAVEN 每轮 30 多次。其中几次发生在 GPU 还有排队工作时，会挡住 CPU 继续排后面的运算：
  - 采样器里 `torch.tensor([input_count], device=...)`（可分页内存的 H2D 拷贝会同步）；
  - `_is_greedy` 每次 `float()` 读温度；
  - 只供审计日志用的 `target_top1_ids`（基准运行时审计关闭）；
  - 验证里 `parent_rows` 的 H2D 拷贝；
  - 选择里 `p32[trunk]`：用 0 维 CUDA 张量做下标，PyTorch 会先在内部 `.item()`，每轮约 3.8 次；
  - 另有给 CUDA 张量元素赋 Python 标量、按层取节点用 Python 列表做下标等。
- `REMTP_DYNAMIC_TREE_ASYNC_HOST=1`：固定内存加 `non_blocking` 拷贝、`fill_`、单元素下标张量、连续切片、缓存的主机温度、审计关闭时不算 top1。设备上的数值、内核、顺序和随机数消耗都不变。`tests/test_async_host.py` 10 项（CPU 和 GPU 上开关前后逐位一致，含随机数状态）。

### 10.4 结果（每轮秒数，3 题；所有组输出与各自 eager 基线逐 token 相同）

| | eager | 图模式 | 图模式 + 自旋 | + 异步主机读 |
| --- | ---: | ---: | ---: | ---: |
| Cactus | 0.1487 | 0.1321 | 0.1228 | — |
| RAVEN v4-safe | 0.1762（去同步 0.1697） | 0.1738 | 0.1428 | 0.1396 |

- 所有方法的解码吞吐（tok/s，3 题；各方法输出不同，题数很少，只看量级）：

  | 方法 | eager | 图模式 + 自旋（RAVEN 另加异步主机读） | 提升 | 每轮 token | 相对 AR |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | AR | 14.76 | 16.72 | +13% | 1 | 1.00× |
  | Native | 19.35 | 23.73 | +23% | 2.919 | 1.42× |
  | Cactus | 22.09 | 26.75 | +21% | 3.285 | 1.60× |
  | RAVEN v4-safe | 19.45 | 24.55 | +26% | 3.427 | 1.47× |

- RAVEN 每轮 token 比 Cactus 多 4.3%，追平 Cactus 吞吐需要每轮 ≤ 0.128 s，还差约 11 ms；目前吞吐比 Cactus 低 8%（优化前低 12%）。

### 10.5 剩余差距在哪里

- GPU 事件计时（`remtp/gpu_event_timing.py`，`AB_GPU_EVENTS=1`，不加同步）：
  - 图模式下目标前向 GPU 时间约 47 ms（10 行时约 49 ms）；
  - 每次 BF16 LM head 13.5 ms，每轮 4 次，共约 54 ms，占 Cactus 一轮的 44%；
  - 每次草稿前向 5.4 ms；
  - Cactus 每轮 123 ms 里 GPU 忙约 117 ms，已经是 GPU 瓶颈。
- 主机时间线（`remtp/host_timeline.py` + `scripts/analyze_host_timeline.py`，只记 `perf_counter`）：RAVEN 每轮在 GPU 空闲时的主机段合计约 10 ms：
  - 验证首次读回后的规划和逐节点判定约 2.4 ms（逐节点的随机数读回是保持逐 token 一致所必需的）；
  - 两次层间元数据准备约 2.6 ms；
  - 树结构整理约 2.3 ms；
  - 选择后处理 0.7 ms；
  - KV 提交 0.9 ms；
  - 修正采样和并列回退约 1.5 ms。
- 这是"每层由主机决定树结构"的结构性开销：每轮 3 个决策点加一次验证，每个点读回后都有 2–3 ms 的 Python。在输出逐 token 相同的约束下，继续做小改预计只能再省 2–4 ms。

### 10.6 小规模公平对比（HumanEval 前 30 题，冻结 prompt 与种子，输出上限 512，GPTQ，MAXN，`results/engopt_suite_20261114/`）

四种方法用同一套工程配置（`scripts/run_w4a16_maxn_suite.py` 的 `SUITE_ENGINE_OPT=1`：完整 CUDA graph + 自旋等待，RAVEN 另加可捕获的树注意力和异步主机读），约 25 分钟跑完。

| 方法 | pass@1 | 截断 | 平均输出 token | MAL | 每轮秒 | decode tok/s | E2E tok/s | 相对 AR（decode） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AR | 22/30 | 1 | 158 | — | — | 16.23 | 15.74 | 1.00× |
| Native | 24/30 | 2 | 167 | 2.883 | 0.1236 | 23.35 | 22.20 | 1.44× |
| Cactus | 24/30 | 2 | 162 | 3.131 | 0.1230 | 25.49 | 24.09 | 1.57× |
| RAVEN v4-safe | 23/30 | 0 | 153 | 3.320 | 0.1445 | 22.97 | 21.76 | 1.42× |

- Native、Cactus、RAVEN 三组的 30 题输出与 2026-11-13 的 eager 运行（第 7 节）**全部逐 token 相同**，所以 MAL、输出长度和判分都与第 7 节一致；变的只有速度。
- RAVEN 有两题（HumanEval/25、/42）首次判分为 Docker 超时（> 8 s），同一份代码重判 3 次都在 0.8 s 内通过，已在 `evaluation.jsonl` 追加 `adjudicated: rejudged` 记录。
- 速度：Cactus decode 20.87 → 25.49 tok/s（+22%），RAVEN 18.88 → 22.97 tok/s（+22%）。RAVEN 的 MAL 比 Cactus 高 6.0%，但每轮多 21.5 ms（+17%），decode 吞吐低 9.9%（优化前低 9.5%）。HumanEval 上 RAVEN 每轮平均节点更多，差距比 GSM8K 3 题（−8%）略大。
- 30 题的 pass@1 区分不了方法（RAVEN 对 Cactus：仅前者对 2 题、仅后者对 3 题，McNemar p=1.0）。

## 11. 建树搬到 GPU（2026-11-15 凌晨，用户同意通宵试做）

目标：RAVEN 每轮有 3 个由主机决定树结构的点（根节点、第 2 层、第 3 层的候选选择与保留），每个点都要读回 GPU 结果、在 Python 里做决定、再准备下一层输入，期间 GPU 空闲。把这些决定搬到 GPU 上，提议阶段只在最后读回一次。提议规则不变，验证仍在主机上（不改随机数流）。

### 11.1 实现（`remtp/gpu_tree.py`，开关 `REMTP_DYNAMIC_TREE_GPU_BUILD=1`）

- **固定形状**：只支持 v4-safe 用的 D3/N9/每父 3 子、`residual_coverage` 分配。第 1 层最多 3 个节点、第 2 层最多 4 个、总共 9 个。第 2 层的 MTP 前向固定 3 行，第 3 层固定 7 行，父节点 LM head 固定 3/4 行。空位不写草稿 KV（槽位 −1），也被注意力掩码挡掉；真实节点的行号、位置和 KV 槽位与主机路径完全相同。
- **候选选择**：逐个运算复现 `batched_trunk_children`（每行 1-D 归约，阈值先转 float32 再比较）。概率并列时按参考实现的稳定顺序（概率降序、编号升序）：先取 top-32，再做两次稳定排序。
- **保留**：在 GPU 上用两两比较复现主机的排序键（类别、主干父节点的子节点按秩、路径值降序、路径），预算公式与主机相同。
- **分支采样**：分支父节点用"请求种子 + 轮次 + 路径"决定的独立生成器。路径是结构性的，所以每轮预先为所有可能的分支路径生成噪声（第 2 层 2 条、第 3 层 8 条），按实际路径取用。主干仍用请求生成器，顺序不变。
- **一次读回**：提议结束时读回节点数和路径编码，主机据此建拓扑、交给 vLLM。
- **CUDA graph**：最初的 eager 版本反而慢了 9.5%：每轮要发射几百个小运算，每个约 50 µs，主机时间约 70 ms，超过这段 GPU 计算本身（约 57 ms），变成 CPU 瓶颈。把每层的"选择 + 保留"捕获成 CUDA graph（`_GraphedLevels`，每层一次重放）后，每步主机时间从 94 ms 降到约 32 ms。
- **校验模式**：`REMTP_DYNAMIC_TREE_GPU_BUILD_CHECK=<jsonl>` 每层都用主机参考实现在相同张量上重算选择、分支采样和保留，记录任何差异。
- 测试 `tests/test_gpu_tree.py` 11 项：选择与 `batched_trunk_children` 逐位一致（含 73 个并列回退的情形）、分支噪声、预算、保留与主机循环一致、图重放与 eager 逐位一致。

### 11.2 数值一致性

- 在相同输入下，GPU 的决策与主机完全相同：校验模式跑了 GSM8K 3 题（372 轮），0 处差异。
- 但输出不保证逐 token 相同：主机路径在只有 1 个节点时跑的是 1 行 GEMM，cuBLAS 对 1 行和 ≥2 行选的内核不同，最后一位不同，偶尔会改变一次采样，之后的轨迹分开。另外阈值和路径值用了 GPU 的 float64 exp/log，与 glibc 可能差最后一位（只有当某个值离判断边界不到一个 ulp 时才会改变决策）。
- 实测：GSM8K 3 题中 2 题逐 token 相同（q856 在第 91 个 token 分开）；HumanEval 前 30 题中 29 题相同（HumanEval/33 在第 322 个 token 分开）。分布层面没有任何变化。

### 11.3 速度

| | GSM8K 3 题 每轮秒 | HumanEval-30 每轮秒 | HumanEval-30 decode tok/s | MAL | pass@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cactus | — | 0.1239 | 25.30 | 3.131 | 24/30 |
| RAVEN 主机建树 | 0.1397 | 0.1414 | 23.48 | 3.320 | 23/30 |
| RAVEN GPU 建树 | 0.1358 | 0.1367 | 24.26 | 3.317 | 23/30 |

（HumanEval-30 三组在同一时段运行，`results/gpu_tree_suite_20261115/`；GSM8K 在 `results/gpu_tree_20261115/ab3g/`。）

- GPU 建树每轮省约 4 ms（−2.8% ~ −3.3%），与事前估计（3–5 ms）一致。
- RAVEN 吞吐相对 Cactus 从 −7.2% 缩到 −4.1%（HumanEval-30）。两种 RAVEN 判对的题完全相同。

### 11.4 HumanEval 全部 100 题（输出上限 512，三组同一时段，`results/gpu_tree_suite100_20261115/`）

| 方法 | pass@1 | MAL | 每轮秒 | decode tok/s | 相对 Cactus |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cactus | 55/100 | 3.090 | 0.1253 | 24.68 | — |
| RAVEN 主机建树 | 57/100 | 3.246 | 0.1422 | 22.82 | −7.5% |
| RAVEN GPU 建树 | 57/100 | 3.252 | 0.1379 | 23.59 | −4.4% |

- GPU 建树每轮比主机建树快 3.0%；MAL 和 pass@1 不变。
- 两种 RAVEN 有 82/100 题输出逐 token 相同。其余 18 题都在较后面才分开（第 52–459 个 token），原因同 11.2。
- GPU 建树那组 HumanEval/48 首次判分为 Docker 超时，同一份输出（与主机建树那组逐 token 相同）重判 3 次都在 0.8 s 内通过，已追加 `adjudicated: rejudged` 记录。

### 11.5 验证预发射（`REMTP_DYNAMIC_TREE_VERIFY_PRELAUNCH=1`，试过，不采用）

- 做法：在目标前向还没算完时，把 3 个主干节点的加成分布、验证概率和请求随机数（按原顺序抽）都排进 GPU 队列，与 p/q 一起读回，主机一次判断完。某个节点被拒时，在抽修正 token 之前把生成器偏移设回"逐节点路径抽了 k+1 个随机数"的位置（`get_offset`/`set_offset`），后面的抽样和原来完全一样。
- 正确性：`tests/test_async_host.py` 6 项（3 种模式 × 2 种门槛），结果与随机数生成器状态逐位一致，预发射每次都被用上、没有回退；GSM8K 3 题输出逐 token 相同。
- 速度没有改善：GPU 建树 0.1367 → 0.1373 s/轮，主机建树 0.1397 → 0.1420 s/轮。这些逐节点读回发生时 GPU 本来就空闲，省下的往返很少，而预发射多算了 3 行加成分布和 3 个随机数。保持关闭。

### 11.6 剩余差距（q856，GPU 事件计时，`results/gpu_tree_20261115/events/`）

| 每轮 | 墙钟 | 模型 GPU 计算 | 其余 |
| --- | ---: | ---: | ---: |
| Cactus | 123.3 ms | 116.7 ms | 6.6 ms |
| RAVEN 主机建树 | 138.0 ms | 118.7 ms | 19.3 ms |
| RAVEN GPU 建树 | 134.9 ms | 117.4 ms | 17.5 ms |

- 模型部分（目标前向、目标 LM head、3 次草稿前向、3 次草稿 LM head）的 GPU 时间已与 Cactus 基本相同。
- 剩下约 11 ms 全在"其余"：GPU 空闲（验证读回后的逐节点主机判断、每轮结束到下一轮目标前向之间的主机处理）加上建树和验证的小运算。
- 要继续缩小，只能把验证的主机判断也搬到 GPU（需要按结局预算所有修正抽样，并做随机数偏移记账）。按 11.5 的经验，节点判断时 GPU 本来就空闲，收益预计只有 1–2 ms；在 Jetson 上，RAVEN 吞吐大致停在比 Cactus 低 4% 左右。
