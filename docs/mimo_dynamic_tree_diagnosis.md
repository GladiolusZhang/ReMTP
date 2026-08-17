# MiMo 动态 MTP 树低接受长度诊断

## 结论

当前瓶颈不是 `tau_relax=0.7` 太严格，而是第 2、3 个物理 MTP head
提出的候选与目标分布几乎不重合。动态构树又主要在这些不可靠的深层
head 上增加宽度，导致更多 target 验证节点没有换来更长的可接受路径。

此外，vLLM 0.18 的 Eagle proposer 不会像 Xiaomi 旧版 vLLM fork 那样，
为每个物理 MTP layer 预填真实前缀 KV。修复该状态缺失后结果只小幅改善，
说明状态缺失是实现问题之一，但附加 MTP head 的概率错配才是剩余主因。

## 同协议 HumanEval-50

共同设置：`MiMo-7B-Base-MTP3`、MTP 深度 3、temperature 0.7、seed 42、
max_tokens 512、eager、同步调度、同一任务清单。

| 方法 | pass@1 | MAL | 相对 Native | decode tok/s | e2e tok/s | draft acceptance / useful-node | truncation | 平均树节点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native probabilistic MTP | 32.0% | 2.068 | +0.000 | 83.817 | 83.025 | 35.6% | 42.0% | - |
| Cactus + MTP | 8.0% | 2.158 | +0.090 | 87.377 | 86.412 | 38.6% | 32.0% | - |
| Dynamic MTP Tree + target relaxation | 26.0% | 2.075 | +0.007 | 71.796 | 71.236 | 19.5% | 52.0% | 5.520 |

这里 `MAL = 接受的草稿深度 + 1 个 target correction/bonus`。因此树的
`MAL=2.075` 实际只表示每轮平均接受 `1.075` 个草稿 token。

树行的 19.5% 是 `接受节点数 / target 验证节点数`，与链式逐位置 draft
acceptance 不是完全相同的统计量。

50 条上的 pass@1 置信区间很宽。配对结果中，Native 对 Cactus 为
`native-only=14, cactus-only=2`，exact McNemar `p=0.0042`；Native 对树为
`native-only=11, tree-only=8, p=0.648`。因此本次 seed 下 Cactus 的质量下降
值得警惕，而 Native 与树的质量差异不足以下定论。

### 为什么同一 MTP 下 Cactus 的 MAL 仍然更高

两边只共享 MTP 权重和物理 head 路由，并不共享同一种 proposal/verification：

- Cactus 从完整 `Q` 中逐位置采样一条链；对于采中的 `y`，先把 `p(y)` 提升为
  `h(y)=p(y)+sqrt(2*delta*p(y)*(1-p(y)))`，再以
  `min(1,h(y)/q(y))` 接受。
- 动态树不会从完整 `Q` 采样。它只保留超过动态阈值的高 `Q` token；验证时
  不使用 `h/q`，而要求
  `S=coverage^alpha*(p(y)/p_max)^(1-alpha) >= 0.7`。

因此“树节点更多”不等于“Cactus 候选的超集”：Cactus 有可能从 `Q` 的尾部
采到 target 支持的 token，而树会在构树阶段把它截掉；另一方面，即使树保留
了某个 token，只要相对 target top-1 支持或 sibling coverage 不够，仍会被硬
阈值删除。Cactus 的 `delta=1` 对中等 `p(y)` 的提升则很激进。

树还要求候选形成一条连续存活的前缀。祖先失败后，后代再多也不能提交。
本次树的根分支数只有 1.012，新增节点主要位于第 2/3 层；第 2/3 层实际接受率
只有 5.9%/5.1%。最后，路径采样还主动损失约 0.028 MAL。

数值上差距并没有表面看起来大：Cactus 每轮接受 1.158 个 draft token，树为
1.075，只相差 0.083。Cactus 用更强的分布偏移换到了这部分接受长度，同时本次
配对 HumanEval pass@1 从 Native 的 32% 降到 8%。

原始结果：

- `results/mimo_chain_baselines_prefillfix_20260807/comparison.md`
- `results/humaneval_dynamic_tree_prefillfix_20260807/comparison.md`
- `results/humaneval_dynamic_tree_prefillfix_20260807/tree/tree_summary.md`

## 问题一：物理 head 的前缀 KV 曾缺失

原实现的首次 proposer forward 只执行物理 layer 0，随后递归调用 layer 1/2。
这样 layer 1/2 没有处理已经提交的真实前缀，其独立 attention KV cache 不完整。

`remtp/mimo_mtp.py` 现在在每轮第一次 layer-0 proposal 时，将相同的真实前缀
token 和 target hidden rows 送入 layer 1/2，只保留 layer 0 的输出，其余调用
仅用于 KV side effect。这与 Xiaomi vLLM fork 中逐 `spec_step_idx` 的 proposer
prefill 不变量一致。

修复前后：

| 方法 | 修复前 MAL | 修复后 MAL | 修复前 e2e | 修复后 e2e |
|---|---:|---:|---:|---:|
| Native | 2.040 | 2.068 | 85.514 | 83.025 |
| Cactus | 2.158 | 2.158 | 90.097 | 86.412 |
| Dynamic Tree | 2.068 | 2.075 | 73.236 | 71.236 |

结论：修复使 Native/Tree 的 MAL 分别增加 0.028/0.007，但额外 prefill 降低了
吞吐；它没有解决深层 head 的主要概率错配。

## 问题二：树在错误的位置变宽

修复后的 8350 个 tree round：

| 深度 | 每父节点平均保留候选 | target coverage 中位数 | 相对 target top-1 支持中位数 |
|---|---:|---:|---:|
| 1 | 1.012 | 1.000 | 1.000 |
| 2 | 1.529 | `6.72e-18` | `3.78e-20` |
| 3 | 1.915 | `2.09e-15` | `1.05e-17` |

根层几乎从不分支；真正的宽度出现在第 2、3 层。但这些位置的候选对 target
而言通常接近零概率。因此平均 5.52 个 target 节点中，大部分是无效验证。

接受深度分布：

| 接受草稿深度 | 轮数 | 比例 |
|---:|---:|---:|
| 0 | 194 | 2.3% |
| 1 | 7419 | 88.9% |
| 2 | 650 | 7.8% |
| 3 | 87 | 1.0% |

## 问题三：降低验证阈值无法正常救回 MAL

在已经构造的候选树上离线重放，改为选择最长通过路径：

| `tau_relax` | oracle MAL |
|---:|---:|
| 0.9 | 2.086 |
| 0.7 | 2.104 |
| 0.3 | 2.111 |
| 0.1 | 2.117 |
| 0.01 | 2.123 |
| `1e-6` | 2.137 |
| `1e-12` | 2.248 |

要获得明显增益，阈值必须降到 `1e-12` 量级，即接受 target 几乎完全反对的
token。这不是合理的松弛区间，也说明候选质量而非 verifier 门槛是主瓶颈。

## 问题四：路径采样主动放弃一部分可用深度

当前 verifier 将所有存活节点（包括仍有存活后代的内部节点）都作为可采样
endpoint。2.7% 的 round 最终路径短于当前最长存活路径，平均损失约 0.028
draft token/round。若目标优先是 MAL，应只在最大存活路径之间采样，或显著
提高长度奖励。不过该项远小于深层 Q/P 错配，单独修复不足以改变结论。

## 问题五：计算成本大于覆盖收益

链式 MTP 深度 3 每轮验证 3 个 draft 节点；动态树平均验证 5.52 个节点，此外
还有多层 MTP 前缀 prefill、逐深度构树和最终 materialization。最终树仅比
Native 增加 0.007 MAL，却使 E2E 从 83.025 降到 71.236 tok/s（-14.2%）。

## 实现位置

| 文件 | 作用 |
|---|---|
| `remtp/mimo_mtp.py` | 物理 head 路由与每层前缀 KV prefill |
| `remtp/dynamic_tree_vllm.py` | 动态构树、物理 head 选择、vLLM TreeAttention 接口 |
| `remtp/dynamic_mtp_tree.py` | 候选阈值、target 松弛可信度、路径采样 |
| `scripts/serve_mimo_dynamic_tree.sh` | 动态树启动参数 |
| `scripts/run_humaneval_mimo_dynamic_tree.sh` | HumanEval 树实验 |
| `scripts/run_humaneval_mimo_chain_baselines.sh` | Native/Cactus 同协议基线 |
| `tests/test_mimo_mtp.py` | 物理层顺序与每层 KV prefill 测试 |

## 判断

当前版本不值得通过继续降低 `tau_relax` 来扩大松弛。下一步若继续研究，应先：

1. 用 teacher-forced reference 独立验证 layer 1/2 的 token 对齐、position 和 KV；
2. 统计 target token 命中 MTP top-2/top-3 的真实比例，确认是否存在可利用的
   多候选覆盖；
3. 若存在覆盖，再让树在早期有效分叉，并删除最终冗余 materialization；
4. 路径选择只比较最大存活路径；
5. 在 reference 对齐通过前，不把当前三物理 head 结果视为正式 MTP3 基线。

若 teacher-forced top-k 覆盖仍低，问题来自附加 MTP 权重本身，树和松弛验证
都无法在不显著损害质量的情况下提高接受长度。
