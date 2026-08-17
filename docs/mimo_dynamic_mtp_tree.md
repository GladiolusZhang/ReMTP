# MiMo dynamic MTP tree with target-dominant relaxation

> **状态路由修正（2026-08-12）：** 多分支路径提交后，下一轮 MTP 必须读取实际
> 选中 BFS 叶节点 `leaf_index + 1` 对应的 target hidden；原 vLLM 链式逻辑按
> “已接受 token 数”选行，会在路径如 `[0,2,5]` 时错误读取 row 3 而非 row 6。
> 修复前的所有多分支跨轮 MAL/质量结果均由错误状态污染，只能作为历史负结果。
> 当前推荐实验入口已改为
> `scripts/run_fastmtp_selected_residual_anchor.sh`，验证语义见
> [exact_residual_hit_tree.md](exact_residual_hit_tree.md)。

> **历史/替代通知（2026-08-12）：** 本文描述的 deterministic target-relative
> 选路、`prefix_reopen_rescue`、margin-adaptive rescue 和
> `sampled_primary_reopen` 均属于已完成的近似探索。最新 20 条结果中，最后一种
> 方法虽然把 GSM8K/HumanEval MAL 做到 `3.316/3.098`，但质量只有
> `70%/70%`；其中救援接受率约 97%，说明救援在采样 residual token 前就改变了
> 输出，过于激进。当前后续实验改用
> [exact residual-hit tree](exact_residual_hit_tree.md)：先从精确 Cactus residual
> 采样 correction，只有树候选恰好命中该 token 时才复用分支。本文保留作历史
> 设计记录，不再代表当前推荐验证语义。

> **随机采样结果有效性通知（2026-08-11）：** 本文在此通知之前记录的所有
> `temperature > 0` tree benchmark 使用了错误的 target 概率处理：vLLM 已在
> `apply_sampling_constraints` 中应用 temperature，tree sampler 又除了一次。
> 因此旧 Tree/Direct Cactus+Tree/Cactus-guided 数值仅保留为历史探索，不能再与
> Native、Chain Cactus 或 SpecCascade 作正式公平比较。贪心测试、树状态隔离测试
> 和定性工程结论不受这个随机温度错误直接影响。修复后的结果必须重新生成。

> **当前实验方法（2026-08-11）：** `cactus_trunk_rescue` 已保留为历史对照，
> 当前待测方法改为 `target_path_rescue`：整棵动态树逐节点读取各自父前缀下的
> target 分布并执行普通松弛验证。普通阶段先选最长存活路径，同长度再按原始
> target 支持选最可信路径；若该路径尚未达到最大深度，救援阶段只检查它的下一层
> 前沿，并最多追加一个 token。该方法仍是近似松弛解码，正式质量与 MAL 尚待
> benchmark。

## Algorithm

The method has a strict information boundary:

1. the proposal tree is built only from MiMo MTP probabilities, normalized
   entropy, depth and a global node budget;
2. target probabilities are read only after the complete tree has been sent
   through one TreeAttention forward;
3. target support prunes the tree; the verifier supports either a strict
   longest-depth selector or a confidence--length selector over all surviving
   prefixes.

For an MTP distribution `q` at draft depth `d`, the implementation computes

```text
H = -sum(q log q) / log(|V|)
U = H + mu * d / D
tau = max(tau_min, max(q) * exp(-kappa * U))
C = {x: q(x) >= tau}
```

To prevent the threshold from collapsing every uncertain position back to a
single chain, guarded top-k backups up to `max_children` may also be retained
when each has at least `tau_min` absolute mass and at least
`min_sibling_ratio` of top-1. Each parent and the whole tree retain hard caps.
This is a probability-guarded widening policy, not forced budget filling.
Level-wise allocation
always reserves one node for every remaining depth, so added siblings cannot
consume the budget before the strongest path reaches depth `D`.

When all eligible children do not fit into `N_max`, candidate paths are ranked
by

```text
E(path) = geometric_mean(path MTP probabilities) * exp(eta * depth / D)
```

and the highest-value nodes are retained. Paths and parent IDs are stored in
breadth-first order; siblings are never flattened into a causal token chain.
If the root has no token above `tau_min`, one top-1 placeholder is transported
through vLLM because the runtime cannot schedule a zero-node tree; the verifier
marks it pruned and emits an unmodified target fallback, so the below-floor
MTP token is never committed.

After one target forward, every sibling set `C_v` receives

```text
A_v = sum(target probability of retained siblings)
R_v(x) = p_v(x) / max(p_v)
S_v(x) = A_v**alpha * R_v(x)**(1-alpha)
```

Nodes with `S < tau_relax` and all of their descendants are pruned. Only a
surviving node without a surviving child is counted as a path endpoint. In
legacy `path_selection=longest` mode, the verifier first restricts selection
to the longest surviving endpoints. A path of length `L` has

```text
log Score(path) = mean(log S_i) + beta * log(L)
```

The equal-length endpoint is selected by `softmax(log Score / T_path)`;
`T_path=0` is argmax. The newer `path_selection=balanced` mode evaluates this
same score over every surviving prefix. It can therefore stop before a weak
tail when the extra `L**beta` reward is smaller than the loss in geometric
target support. The selected path is committed, followed by one token sampled
from the unmodified target row at the selected leaf. If no path survives, one
token is sampled from the unmodified target root distribution.

The FastMTP balanced/wider experiment uses `beta=0.75`, `T_path=0`, an
8-node whole-tree cap and at most two children per parent. Increasing the cap
from 6 to 8 gives more strong parent paths a chance to reach depth 3; retaining
the binary child cap prevents indiscriminate vocabulary-wide branching.

### Bounded failed-frontier rescue

The FastMTP quality/MAL configuration keeps the normal rule above unchanged
and adds one narrowly scoped rescue. If a currently surviving parent has no
normally surviving child, the verifier may try exactly one child with the
largest target-relative support. It is eligible only when it is not blocked by
EOS protection and satisfies both

```text
p(y) / max(P) >= rescue_min_relative
p(y) >= rescue_min_target_prob
```

For this candidate only, the verifier computes the scalar Cactus-style boost

```text
h(y) = min(1, p(y) + sqrt(2 * rescue_delta * p(y) * (1 - p(y))))
a_rescue = min(1, h(y) / q(y))
```

and accepts when `u <= a_rescue`. A rescue flag is propagated with the path,
so no committed path can use more than one rescue. Normal children are always
preferred in equal-depth path ties. This is deliberately different from
globally lowering `tau_relax`: risk is introduced only at the first dead
frontier and cannot compound at every depth. The default FastMTP comparison
uses `rescue_delta=0.5`, relative support floor `0.1` and absolute target
probability floor `0.001`.

This rule does not make tree decoding distribution-preserving. Tree path
selection was already approximate, and a top-Q tree candidate is not itself a
categorical sample from the full `Q`. The rescue must therefore be evaluated
by task quality and MAL together.

This final target token is an anchor and makes tree MAL comparable to the
repository's speculative-decoding reports:

```text
tree MAL = selected draft-path length + one target anchor
```

## Important statistical boundary

This is relaxed decoding, not exact speculative sampling. Path filtering and
path-level sampling generally do not reproduce the original target
distribution. Therefore every speed result must be accompanied by HumanEval
pass@1 or another task-quality metric. `tau_relax` is the principal
quality--speed control.

## vLLM implementation

The server starts vLLM with a maximum-width placeholder tree. The placeholder
is used only for startup validation and buffer allocation. Each round:

1. `remtp.dynamic_tree_vllm` produces a real variable-width topology;
2. proposer output contains only the real nodes, not padding tokens;
3. draft and target `TreeAttentionMetadataBuilder.tree_attn_bias` tensors are
   replaced with the real parent/ancestor mask before their forwards;
4. the scheduler receives the real draft length (`--no-async-scheduling` is
   required in vLLM 0.18);
5. target logits are mapped by explicit parent rows;
6. only the selected path's target and MTP KV slots are compacted into the
   canonical chain.

The target still runs once per speculative round. Actual nodes, target calls,
target latency, branching, coverage, survival and selected path are written to
JSONL. Local target signals are computed as one GPU batch and transferred to
the prefix scanner once per round, rather than synchronizing once per node.
The HumanEval runner also enables lightweight audit mode and disables readable
per-round traces so observability overhead is excluded from reported speed.

MiMo publishes three physical MTP modules. The supported primary setting is
`D=3`, mapped to layers 0/1/2. Setting `D>3` reuses physical layer 2 and is an
explicit extrapolation ablation, not a pretrained multi-head guarantee.

## Files

| file | role |
|---|---|
| `remtp/dynamic_mtp_tree.py` | topology, entropy threshold and target-dominant path verifier |
| `remtp/dynamic_tree_vllm.py` | dynamic proposer, per-round TreeAttention metadata and variable-width scheduler adapter |
| `remtp/fixed6_vllm.py` | shared target input, parent logits, KV path compaction and JSONL audit |
| `remtp/dynamic_tree_humaneval_report.py` | combine tree-native MAL with HumanEval throughput |
| `scripts/serve_mimo_dynamic_tree.sh` | standalone dynamic-tree server |
| `scripts/run_humaneval_mimo_dynamic_tree.sh` | end-to-end 50-task experiment |
| `scripts/run_fastmtp_mal_quality_50.sh` | live Native/Cactus/SpecCascade/tree comparison without the slow target-only row |

## Start a server manually

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

REMTP_DYNAMIC_TREE_MAX_DEPTH=3 \
REMTP_DYNAMIC_TREE_MAX_NODES=6 \
REMTP_DYNAMIC_TREE_MAX_CHILDREN=2 \
REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO=0.25 \
REMTP_DYNAMIC_TREE_TAU_MIN=0.02 \
REMTP_DYNAMIC_TREE_KAPPA=1.0 \
REMTP_DYNAMIC_TREE_MU=0.5 \
REMTP_DYNAMIC_TREE_ETA=0.25 \
REMTP_DYNAMIC_TREE_ALPHA=0.5 \
REMTP_DYNAMIC_TREE_TAU_RELAX=0.7 \
REMTP_DYNAMIC_TREE_BETA=0.5 \
REMTP_DYNAMIC_TREE_PATH_TEMPERATURE=0.3 \
REMTP_DYNAMIC_TREE_PATH_SELECTION=longest \
REMTP_DYNAMIC_TREE_FRONTIER_RESCUE=1 \
REMTP_DYNAMIC_TREE_RESCUE_DELTA=0.5 \
REMTP_DYNAMIC_TREE_RESCUE_MIN_RELATIVE=0.1 \
REMTP_DYNAMIC_TREE_RESCUE_MIN_TARGET_PROB=0.001 \
./scripts/serve_mimo_dynamic_tree.sh
```

In another terminal:

```bash
./scripts/request_mimo.sh
```

## HumanEval 50

Download the official dataset once if necessary:

```bash
./scripts/download_humaneval.sh
```

Run generation and the requested performance report:

```bash
SAMPLES=50 \
TEMPERATURE=0.7 \
SEED=42 \
MAX_TOKENS=512 \
./scripts/run_humaneval_mimo_dynamic_tree.sh
```

The final `comparison.md` contains:

- tree-native mean acceptance length;
- decode tokens/s;
- end-to-end output tokens/s;
- average target validation nodes and target calls/round.

To also execute official HumanEval tests in restricted Docker containers:

```bash
RUN_QUALITY_EVAL=1 SAMPLES=50 \
./scripts/run_humaneval_mimo_dynamic_tree.sh
```

For a short readable decision trace instead of a formal speed run, start the
server manually with `REMTP_TREE_TRACE=1 REMTP_TREE_AUDIT_DETAIL=1`. Detailed
tracing decodes and records every candidate and therefore should not be used
for throughput reporting.

All reports remain local under `results/humaneval_dynamic_tree_*`; server logs
remain under `logs/humaneval_dynamic_tree_*`.

## First parameter sweep

Use staged sweeps. Do not run the full Cartesian product.

1. construction: `tau_min`, `kappa`, `mu`, then `N_max`/`D`/`eta`;
2. verification: `alpha`, `tau_relax`;
3. path selection: `beta`, `T_path`.

Suggested primary values match the launch defaults. Lowering `tau_relax` is
the most direct way to increase relaxation, but must be judged against pass@1.
For the current FastMTP method, prefer the bounded frontier rescue over a
global threshold reduction, because the observed rejected-support values are
not concentrated immediately below `tau_relax=0.50`.

## Balanced wider-tree N=100 rerun

The frozen Native/Cactus/SpecCascade rows in `results/fastmtp_live_n100` do not
need to be regenerated. Run only the new dynamic-tree configuration and build
one comparison table with symbolic links to those immutable baselines:

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

SAMPLES=100 \
BASELINE_RUN_ROOT=results/fastmtp_live_n100 \
RUN_TAG=fastmtp_balanced_wide_n100 \
  ./scripts/run_fastmtp_balanced_wide_n100.sh
```

This configuration is intentionally one controlled change set:

```text
N_max=8, max_children=2, sibling_ratio=0.20
path_selection=balanced, beta=0.75, T_path=0
```

It should be judged first by GSM8K truncation/accuracy and then by MAL. The
8-node cap is a coverage experiment, not a claim that more nodes always help.

## D=3 versus D=4 guarded top-k experiment

The next full experiment loosens proposal guards without forcing every round
to use its complete node budget. Both depths use the same maximum of 10 target
validation nodes, at most three children per parent, `q >= 0.002`, and
`q/q_top1 >= 0.02`. D=4 repeats FastMTP's one physical MTP layer for one more
logical proposal step; it is not a separately trained fourth head.

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

RUN_TAG=fastmtp_depth34_relaxed_full \
  ./scripts/run_fastmtp_depth34_relaxed_tree.sh
```

Defaults are GSM8K 500 samples and all 164 HumanEval tasks. Native, Cactus and
SpecCascade are generated once in the D=3 phase and reused by the D=4 phase.
The combined report is written to
`results/$RUN_TAG/depth34_comparison.md`. Override `GSM8K_SAMPLES` or
`HUMANEVAL_SAMPLES` only for a deliberate shorter pilot.
# Reach-first D=3（2026-08-10，已由 soft-reach 取代）

完整 D3/D4 实验表明，D4 的表面 MAL 增益主要来自多一个可接受位置；将 D4
接受深度截断到 3 后，前三层 MAL 反而低于 D3。旧的几何平均优先级会让早期
低 Q 备选路径在后续高置信 token 后重新获得较高分数，并占用高可达主路径的
continuation 节点。

新增 `reach_first` 分配模式：

1. 同一深度先分配所有候选父路径的 rank-0 continuation；
2. 剩余预算再分配 rank-1，最后才分配 rank-2；
3. 每个 rank 档位内使用整条前缀的 Q 概率乘积排序，不再使用几何平均；
4. 节点上限仍是上限，不会用未通过 Q 守卫的 token 强制填满；
5. 正式主比较固定 `D=3`，D4 仅作为深度消融，不能用其天然第四层收益与
   三层 baseline 直接宣称算法优势。

对应入口为 `scripts/run_fastmtp_reach_first_tree.sh`。默认只运行新的 D3 树方法，
并复用已经完成、协议一致的 Native/Cactus/SpecCascade 结果。

## Soft-reach 修订

硬性的“所有 rank-0 都先于 rank-1”仍会隐含父路径配额。当前入口已经改为
`soft_reach`：

```text
proposal_utility(path)
  = product(Q along path) / (1 + lambda_rank * local_rank)
```

所有满足 Q 概率守卫的候选使用这个连续效用竞争当前全局剩余预算。rank 只降低
效用，不建立强制档位；一个高 reach 的 rank-1 候选可以超过低 reach 的 rank-0
候选。每层节点数、每个父节点的孩子数都没有下限，候选不足时预算保持未使用。
为未来深度预留容量也只限制当前层上限，不会强行创建未来节点。

验证后，`balanced` 路径选择继续对每个存活前缀计算：

```text
C(path) = exp(mean(log target_relative_support_i))
Score(path) = C(path) * length**beta
```

因此最终路径同时考虑目标置信度和长度：较长路径只有在长度收益足以覆盖置信度
下降时才会胜出，高置信短路径可以主动停止在弱尾部之前。

## Rejected-token audit 与目标确认式放宽

对 soft-reach D3 的完整日志分析表明，`coverage<0.05` 的拒绝几乎都是目标概率
接近 0 的噪声，不应放宽。真正可能有价值的是 coverage 已通过、但
`P(y)/P(top1)` 位于中间区域的候选。

无条件把阈值从 `0.35` 降到 `0.10` 会改变真实前缀，并可能令下一轮 MTP 更难
对齐。因此推荐的 `confirmed` 模式保留原始主条件，同时开放一个更宽但带证据的
区域：

```text
normal_accept = relative_target_support >= 0.35

extra_accept = relative_target_support >= 0.05
               and (
                 P(candidate) / Q(candidate) >= 1
                 or retained_child == target_next_top1
               )
```

coverage 与 EOS 保护完全不变。`retained_child == target_next_top1` 使用同一次树状
target forward 已经计算的下一位置分布，不增加 target 调用。路径提交仍由
`confidence(path) * length**beta` 决定；目标确认只让路径进入候选集，不保证它
最终被提交。
## 2026-08-10：N=100 target-confirmed 结果与经验阈值压力实验

最新 `target-confirmed` 版本并未形成有效松弛：

| 数据集 | Native 质量 / MAL | Cactus 质量 / MAL | Confirmed tree 质量 / MAL | nodes/round |
|---|---:|---:|---:|---:|
| GSM8K-100 | 90.0% / 3.040 | 90.0% / 3.434 | 88.0% / 3.184 | 6.904 |
| HumanEval-100 | 61.0% / 3.065 | 69.0% / 3.342 | 68.0% / 3.100 | 7.096 |

逐轮日志显示，`confirmed` 额外保留的节点为每轮 `0.082/0.070`，但真正进入
最终路径的只有每轮 `0.017/0.013`。因此它在验证层面看似增加了存活节点，实际
几乎没有改变提交前缀。

被拒节点的目标相对支持分布也呈明显断层：

| relative 区间 | GSM8K nodes/round | HumanEval nodes/round | 目标概率中位数 |
|---|---:|---:|---:|
| 0.05–0.35 | 0.040 | 0.039 | 约 0.06–0.20 |
| 0.01–0.05 | 0.122 | 0.106 | 约 0.011–0.030 |
| < 0.01 | 1.639 | 1.775 | `7.6e-14 / 1.2e-16` |

这说明日志里确实存在少量“目标稍微愿意接受”的候选，但数量不足以单独填平与
Cactus 的 MAL 差距。大量额外树节点并不是边缘可接受项，而是目标模型强烈反对
的分支。若短期目标改为先观察 MAL 天花板，就必须把实验明确标为压力测试，而不
能继续把它解释成保守松弛。

新增 `scripts/run_fastmtp_empirical_relaxation.sh`，提供三档只运行新树方法的配置：

- `plausible`：coverage/relative 均为 `0.01`，`beta=3`；
- `mal_first`：coverage/relative 均为 `0.001`，`beta=5`；
- `ceiling`：除 EOS 外不设 target mass 下限，仅用于测候选树的 MAL 上限。

路径仍按 `geometric target confidence * length^beta` 联合评分，并非强制最长；
`mal_first` 只是把长度权重提高。基于已记录节点的离线重放是保守下界，因为被新
救活父节点的后代在旧日志中没有验证信号。该下界预测 `mal_first` 的 MAL 为：

- GSM8K：约 `3.273`；
- HumanEval：约 `3.179`。

若真实在线结果仍明显低于 Cactus，原因就不再是 0.35 阈值过紧，而是 FastMTP
树分支缺少目标支持；继续放宽只能接受目标概率接近 0 的 token。

## 2026-08-10：用 Cactus 有效接受概率校准树松弛强度

> 历史/消融说明：本节的 `support_mode=cactus` 仍保留用于对齐链式 Cactus
> 松弛强度，但不再作为动态树主方案。它会在每个节点直接执行 Cactus 存活采样；
> 后续主方案改为下节的 `cactus_guided`，只在目标规则形成死前沿时使用 Cactus。

链式 Cactus 与旧树规则不能用同一个 relative 阈值直接比较。Cactus 对候选使用：

```text
h(y) = min(1, p(y) + sqrt(2 * delta * p(y) * (1-p(y))))
A_cactus(y) = min(1, h(y) / q(y))
```

默认 `delta=1` 的松弛很强。例如 `p(y)=0.01` 时，`h(y)` 已约为 `0.151`；
若 `q(y)=0.19`，候选仍有约 80% 的接受概率。旧 dynamic tree 对同一候选只看
`p(y)/p(top1)`，在阈值 0.35 下会直接剪枝。

在 confirmed N=100 日志中，把全部已拒节点换算为 Cactus `A(y)`，每轮额外
期望接受质量为 `0.218/0.189`。关键区间如下：

| relative 区间 | GSM8K mean A | HumanEval mean A |
|---|---:|---:|
| 0.05–0.10 | 0.743 | 0.775 |
| 0.02–0.05 | 0.709 | 0.698 |
| 0.01–0.02 | 0.609 | 0.596 |
| 0.001–0.005 | 0.430 | 0.433 |

这解释了 Cactus 只有单链三个候选仍能达到更高 MAL：它不是拥有更多候选，而是
在每个位置对低 `p` 候选进行概率提升，并利用 `q` 校准接受概率。

新增 `support_mode=cactus`：

1. 在 GPU 上并行计算每个树节点的 `h(y)` 与 `A_cactus(y)`；
2. 每个节点以 `A_cactus(y)` 进行局部存活采样；
3. 多个存活路径仍通过置信度—长度联合分数选择；
4. 路径置信度默认使用
   `relative^0.25 * A_cactus^0.75`，防止极低 `q` 让目标几乎不支持的 token
   获得满分；
5. EOS 保护和每轮一个 target tree forward 保持不变。

多兄弟节点分别使用候选特定 Cactus 分布，因此整棵树仍是近似输出分布，不应
宣称等价于严格 Cactus sampling。其用途是让树的松弛强度与链式 Cactus 可比，
再通过任务质量评估多候选覆盖是否有增益。

真实 FastMTP/vLLM/TREE_ATTN 短请求冒烟完成：26 rounds、64 output tokens，
审计记录 125 个已评估节点、51 个 Cactus 存活节点和 39 个最终选中节点；每个
节点均记录 `h(y)`、`A(y)`、随机数、存活事件和 path confidence。单提示 MAL
为 2.50，仅用于验证执行链路，不能当作 benchmark 结论。

### N=100 结果与失败归因

| 数据集 | Chain Cactus 质量 / MAL | Direct Cactus + Tree 质量 / MAL |
|---|---:|---:|
| GSM8K | 90.0% / 3.434 | 89.0% / 3.251 |
| HumanEval | 69.0% / 3.342 | 67.0% / 3.173 |

“树有更多候选”没有保证 MAL 超过链式 Cactus，原因是当前实现并不是在链式
Cactus 主干上单调增加备选，而是重新构造 top-Q 树、分别采样节点存活，再从
所有存活路径中重新选择输出。日志拆分如下：

| 指标 | GSM8K | HumanEval |
|---|---:|---:|
| 实际 tree MAL | 3.251 | 3.173 |
| 仅 all-zero top-Q 主路径 MAL | 2.835 | 2.761 |
| 树分支相对主路径增益 | +0.415 | +0.412 |
| 若总取最长存活路径的 MAL | 3.259 | 3.181 |
| 路径评分导致的 MAL 损失 | 0.008 | 0.008 |
| drafted nodes/round | 6.776 | 6.975 |
| parent 存活后实际可评估 nodes/round | 4.408 | 4.354 |
| 祖先失败导致无效 nodes/round | 2.369 | 2.621 |

因此树分支本身确实有效，每轮增加约 0.41 个接受草稿 token；真正的问题是作为
起点的 deterministic top-Q 主路径远弱于标准链式 Cactus 的 sampled-Q 主干，且
约 35%–38% 的树节点因祖先被拒而无法成为可提交路径。路径置信度—长度评分只在
约 0.8% 的轮次选择较短前缀，不是主要瓶颈。

额外分支在约 12% 的轮次被选中，但其下一轮对齐明显较弱：

| 上一轮路径 | GSM8K 下一轮 root 存活 | HumanEval 下一轮 root 存活 |
|---|---:|---:|
| 主分支 | 85.7% | 83.8% |
| 替代分支 | 71.3% | 65.8% |
| target fallback | 96.8% | 97.2% |

该统计存在上下文难度混杂，不能直接证明替代分支导致下降；但它说明额外分支
主要出现在困难状态，并不能被当成无成本的当前轮 MAL 增益。

下一版不应继续扩大当前独立分支树，而应采用 **Cactus trunk + rejection-only
tree rescue**：先保留与链式 Cactus 完全相同的 sampled-Q 三 token 主干、共享
随机数和逐位置验证；只有在主干第一次将被拒绝时，才允许同一父节点的侧分支
接管。这样树候选是链式基线的增量救援，而不是替换其主干和输出轨迹。该方向尚未
实现，当前 Direct Cactus + Tree 结果保留为负结果和设计诊断。

> 历史状态更新（2026-08-11）：上段“尚未实现”只描述 2026-08-10 当时状态。
> sampled-Q 主干与首次拒绝救援现已实现，见本文末尾的新章节；尚未完成正式
> N=100 质量/MAL 实验，不能提前宣称超过 Chain Cactus。

## 2026-08-10：目标主导的 Cactus-guided 死前沿校准

直接 `support_mode=cactus` 虽然让局部松弛强度与链式 Cactus 对齐，但它把每个树
节点的存活语义都替换成了 Cactus。尤其当 `q(y)` 很小时，`A_cactus(y)` 很容易
达到 1；若路径评分又主要依赖 `A_cactus`，长但目标支持弱的路径可能被高估。

新增主方案 `support_mode=cactus_guided`，明确拆分两个职责：

1. 动态树继续完全由 MTP 的 `Q`、熵、深度和 soft-reach 全局预算构造；
2. 正常节点仍按原目标相对支持 `relative >= tau_relax` 存活；
3. 只有已存活父节点没有任何正常后继时，才进入 Cactus-guided 死前沿校准；
4. 可救候选必须通过 coverage、EOS 和目标概率下限，并至少满足以下一个树证据：
   `P/Q >= proposal_ratio`、保留子节点命中下一步 target top-1，或当前 relative
   已进入主阈值一半以上的中等支持区；
5. 在合格兄弟中按
   `relative^w * A_cactus^(1-w)` 选择一个候选，再以 `A_cactus` 随机决定是否救活；
6. 每条路径最多经过一次 guided rescue，但救活后可以继续接受后面的正常节点；
7. 最终仍对所有存活前缀计算几何平均置信度与 `length^beta` 的联合分数，不默认
   选择最长路径。

推荐第一版配置：D=3、soft-reach、`N_max=10`、`tau_relax=0.35`、coverage
下限 0.05、`delta=1`、目标权重 `w=0.65`、`beta=1.5`。这里节点数仍是上限，
没有每层或每父节点硬配额。入口为：

```bash
RUN_TAG=fastmtp_cactus_guided_tree_n100_v2 \
  ./scripts/run_fastmtp_cactus_guided_tree.sh
```

该模式仍然是近似松弛解码，不具有严格分布保持保证。其目标不是复现 Cactus，
而是让 Cactus 只修复动态树的局部断链，同时保持构树、证据门控和路径决策均属于
目标主导动态树。

### 统一双方案实验入口

为避免把“逐节点 Cactus + 树”和“我们的 Cactus-guided 树”混为同一个方法，
新增统一入口：

```bash
RUN_TAG=fastmtp_tree_relaxation_n100 \
DIRECT_RUN_TAG=fastmtp_cactus_calibrated_tree_n100_v1 \
  ./scripts/run_fastmtp_tree_relaxation_ablation.sh
```

该命令按顺序执行或恢复：

1. `Direct Cactus + Dynamic Tree`：每个树节点直接使用 Cactus 存活概率，仅作为
   Cactus+Tree 消融；
2. `Cactus-guided Dynamic Tree (ours)`：正常验证仍由目标 relative 规则决定，
   Cactus 只校准死前沿。

两组实验固定 D=3、soft-reach、最多 10 个节点、每轮一次 target forward，并复用
相同 Native/Cactus/SpecCascade 基线。脚本逐样本打印进度，结束后将五个方法写入
同一个 `results/$RUN_TAG/comparison.md`，同时检查两棵树的样本清单与生成协议完全
一致。脚本还会从各自的 vLLM server log 核验实际加载的 `support_mode`，并写入
`tree_variant.json`，防止把旧 direct 结果误标成 guided 结果。已有的 direct run
可以通过 `DIRECT_RUN_TAG` 复用，不会重新生成。

## 2026-08-10：更宽的双救援 Cactus-guided 档位（v3）

> 实验状态：实现完成，尚未形成 N=100 结果。上一节 v2 仍是保守单救援基线；
> 本节 v3 取代它作为下一轮“追赶 Chain Cactus MAL”的推荐实验档位，但不改变
> 任何已经完成实验的含义。

v3 保留原有 Q 驱动、D=3、soft-reach、最多 10 节点的树结构，也保留“正常目标
验证优先、前沿完全死亡后才允许 Cactus 救援”的算法边界。它只扩大以下范围：

1. 正常存活阈值从 `relative >= 0.35` 降为 `0.25`；
2. 父节点候选集合的目标覆盖下限从 `0.05` 降为 `0.01`；
3. guided 候选最低 relative 从 `0.05` 降为 `0.01`，最低目标概率仍为 `0.001`；
4. `P(y)/Q(y)` 证据阈值从 `1.0` 降为 `0.25`；下一步 target top-1 命中和
   中等 relative 支持仍可作为替代证据；
5. Cactus `delta=1` 不变，guided 排序由
   `relative^0.5 * A_cactus^0.5` 决定；
6. 每条路径最多救援两次，而不是一次。第二次救援仍必须发生在后续另一个完全
   死亡的前沿，不能绕过尚有正常存活兄弟的节点；
7. 最终路径继续联合考虑置信度与长度，`beta=2.0`，不强制选择最长路径。

该档位的目标是缩小与 Chain Cactus 的 MAL 差距，不是保证达到某个数值。它仍是
近似松弛解码：多分支、逐节点随机救援和最终路径选择会改变输出分布，因此必须与
准确率一起评估。运行入口：

```bash
RUN_TAG=fastmtp_cactus_guided_wide_n100_v3 \
  ./scripts/run_fastmtp_cactus_guided_wide_tree.sh
```

脚本复用相同 Native/Cactus/SpecCascade N=100 基线，只重新运行本方法，并将两个
数据集的对比写入 `results/$RUN_TAG/comparison.md`。运行时审计新增
`guided_rescue_count_before`，用于区分第一次和第二次救援。

### v3 N=100 结果：阈值与第二次救援均进入无效区

> 结论：该档位没有追平 Chain Cactus，且不再推荐继续降低普通 relative 阈值或
> 增加每路径救援次数。保留结果作为负结果，下一步应改变主干语义而非继续调阈值。

| 数据集 | v3 质量 / MAL | Direct Cactus+Tree | Chain Cactus |
|---|---:|---:|---:|
| GSM8K-100 | 88.0% / 3.172 | 89.0% / 3.251 | 90.0% / 3.434 |
| HumanEval-100 | 64.0% / 3.133 | 67.0% / 3.173 | 69.0% / 3.342 |

逐轮日志说明最近参数调整没有形成 MAL 增益的原因：

1. 正常 relative 支持高度两极化。GSM8K 的 18,055 个 round 中，仅 1 个正常
   存活节点落在新开放的 `[0.25, 0.35)` 区间，且未进入最终路径；HumanEval 的
   19,262 个 round 中该区间为 0。因此 `tau_relax: 0.35 -> 0.25` 实际没有改变
   输出路径。
2. 第二次救援几乎不可达。GSM8K 只有 10 次、HumanEval 只有 1 次成功救援的
   `guided_rescue_count_before=1`，分别仅占 round 的 `0.055%/0.005%`。
3. 最终选路已不是瓶颈。两个数据集均没有出现“存在更长存活路径但评分器选择较短
   路径”的 round；继续提高 `beta` 不会增加 MAL。
4. 救援覆盖面太小。最终选择 guided rescue 的 round 仅占 `3.37%/2.53%`；即使
   将这些路径相对救援前缀的全部后续 token 都算作救援贡献，其 MAL 上界也只有
   `0.060/0.045`，远小于相对 Chain Cactus 的 `0.262/0.209` 差距。
5. 更多树节点不等于更多可提交 token。v3 每轮构造 `7.012/7.015` 个节点，但只有
   约 `59%` 位于存活祖先下可继续验证；accepted-draft/nodes 仅为约 `31%/30%`。
   候选 coverage 仍约为 `0.826/0.824`，与旧版本基本相同，说明新增节点主要是
   相关或低目标概率候选，而不是新的高价值路径。

Direct Cactus+Tree 已经对所有树节点使用 Cactus 存活概率，MAL 仍只有
`3.251/3.173`。这进一步证明剩余差距来自主干构造和采样语义：当前 deterministic
top-Q 树不是 Chain Cactus sampled-Q 三 token 主干的超集，侧分支又会随祖先失败
整体失效。后续若继续，应实现 `sampled-Q Cactus trunk + rejection-only tree
rescue`，先逐轮复用链式 Cactus 的候选、随机数和接受结果，再让侧分支只接管主干
第一次拒绝；而不是继续放低当前树的 relative 阈值。

## 2026-08-11：sampled-Q Cactus 主干 + 首次拒绝树救援

### 为什么这版与前几版不同

此前 Direct/Guided 动态树先确定性选择 top-Q 主路径，再让多条路径共同竞争。
它不是 Chain Cactus sampled-Q 链的超集，且大量节点会因祖先失败而不可达。因此
放低 relative 阈值或增加第二次救援几乎没有改变最终提交路径。

新模式 `support_mode=cactus_trunk_rescue` 把树改为“毛毛虫”结构：

```text
depth 1: sampled trunk y1 + up to two backups
depth 2: only expand y1 -> sampled trunk y2 + up to two backups
depth 3: only expand y1,y2 -> sampled trunk y3 + up to two backups
```

主干每层都从完整 FastMTP 分布 Q 通过与链式 probabilistic MTP 相同的
exponential-race primitive 采样。备选节点只按高 Q 确定性保留，不消耗请求采样
随机数；后续只递归展开主干，不再为侧分支建立大批注定不可达的后代。默认 D=3、
每层最多两个备选，因此最多 9 个 target tree nodes；节点数是上限，不会强制填满。

### 验证语义

对主干第 i 个 token，先构造与 Chain Cactus 相同的候选特定分布：

```text
h_i(y_i) = min(P_i(y_i)
                 + sqrt(2 delta P_i(y_i)(1-P_i(y_i))), 1)
A_i = min(1, h_i(y_i) / Q_i(y_i))
```

按顺序验证主干。若主干通过，当前层备选标为 `BACKUP_UNUSED`，不参与路径竞争。
第一次主干拒绝时才计算 Cactus correction residual：

```text
r_i(v) ∝ max(h_i(v) - Q_i(v), 0)
```

随后只在同父节点的备选中选择一个 residual 支持较高的候选。候选需通过最低
residual probability、relative residual support 和 EOS 保护，再以 residual 为
目标分布构造第二个 Cactus boost。救援成功后提交“此前已接受主干 + 备选 token +
该备选节点后的原始 target bonus”；救援失败或无合格备选时，从原 Cactus
residual 采样 correction。若三层主干全部接受，则照常提交原始 target bonus。

因此主干部分保持 Chain Cactus 的 Q、h、逐位置接受概率和 correction 定义；树
只在首次拒绝位置增加一次替代机会。侧候选是从 Q 确定性保留而不是从严格联合
multi-proposal 分布采样，所以救援部分仍是近似松弛解码，必须同时报告质量和 MAL，
不能宣称整体分布等价于 Cactus。

### 实现与运行

关键实现：

- `remtp/dynamic_tree_vllm.py`：sampled-Q 主干、备选选择、毛毛虫 TreeAttention
  拓扑和 FastMTP branch-local KV 物化；
- `remtp/dynamic_mtp_tree.py`：主干 Cactus 验证、residual rescue、target
  correction/bonus 与节点诊断；
- `scripts/run_fastmtp_cactus_trunk_rescue_tree.sh`：复用冻结
  Native/Cactus/SpecCascade 基线，只生成新方法结果。

先跑 20 条试验：

```bash
GSM8K_SAMPLES=20 HUMANEVAL_SAMPLES=20 \
RUN_TAG=fastmtp_cactus_trunk_rescue_pilot \
  ./scripts/run_fastmtp_cactus_trunk_rescue_tree.sh
```

通过后再跑 100 条：

```bash
GSM8K_SAMPLES=100 HUMANEVAL_SAMPLES=100 \
RUN_TAG=fastmtp_cactus_trunk_rescue_n100 \
  ./scripts/run_fastmtp_cactus_trunk_rescue_tree.sh
```

结果写入 `results/$RUN_TAG/comparison.md`，逐轮轻量审计位于对应 dynamic-tree
结果目录。若 `BASELINE_ROOT` 的样本数与本次请求一致，脚本复用三组冻结基线；
若不一致（例如先跑 20 条 pilot），脚本会重新生成 Native/Cactus/SpecCascade，
避免把不同样本规模强行写进同一张表。正式判断重点是：新方法 MAL 是否至少不低于 Chain Cactus，同时
GSM8K accuracy 与 HumanEval pass@1 是否接近 Native/Chain Cactus。

### 当前验证状态

CPU 单测覆盖：主干全接受与 target bonus、root 拒绝后备选救援、无救援时回到
Cactus residual correction。相关动态树/Cactus/cache-audit 测试共 46 项通过。

真实 FastMTP/vLLM/TREE_ATTN 短请求冒烟成功：48 output tokens、24 rounds，
每轮 target forward 固定为 1，实际节点数 5–9；3 轮主干全部接受，16 轮回到
Cactus correction，5 轮由树备选成功救援。单提示 MAL 为 2.083，仅证明运行链路、
缓存压缩和审计可用，不是 benchmark 效果结论。正式 N=100 尚未运行。

## 2026-08-11：20 条 trunk-rescue pilot 暴露双重温度错误

修复前 pilot 的原始结果如下。它用于发现实现错误，不再作为算法结论：

| 数据集 | Native 质量/MAL | Chain Cactus | SpecCascade | Trunk-rescue tree |
|---|---:|---:|---:|---:|
| GSM8K-20 | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 90.0% / 2.638 |
| HumanEval-20 | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 75.0% / 2.252 |

逐轮树日志显示，主干 root 接受率仅为 GSM8K `59.9%`、HumanEval `47.6%`，而
同协议 Chain Cactus 服务日志的 position-1 接受率通常约 `92%–95%`。树主干的
目标候选概率也异常尖锐：GSM8K/HumanEval 分别有 `24.2%/36.5%` 小于 `1e-6`，
同时有 `52.5%/51.0%` 大于 `0.999`。

根因位于 `remtp.fixed6_vllm._strict_tree_sampler`：

```text
raw target logits
  -> apply_sampling_constraints       # 已除以 T
  -> _temperature_probs               # 错误地再次除以 T
```

当实验设置 `T=0.6` 时，tree target 实际等价于约 `T=0.36`，但 draft Q 和所有链式
基线仍使用 `T=0.6`。这会放大 P/Q 不匹配，使很多 Cactus 主干 token 的 P(y) 被
压到近零。救援本身并非完全没有工作：GSM8K/HumanEval 分别有 `1096/735` 个 round
救援成功，且成功率接近 1；但它每轮最多补一个 token，无法抵消错误 target
分布造成的主干大幅回退。

修复后，processed target logits 在随机模式下只做 softmax；temperature=0 仍转成
one-hot，保持贪心语义。新增回归测试明确验证 `softmax(raw/T)`，并拒绝旧的
`softmax(raw/T/T)` 行为。

修复后真实 GPU smoke 已完成：tree 与 Chain Cactus 分别在相同通用提示、三个
seed、每个 128 output tokens 下运行。Tree 共 183 rounds，MAL `2.082`，Chain
Cactus 的服务窗口 MAL 为约 `2.11–2.15`；这一提示本身对 FastMTP 较难，因此两者
都低，但已不再出现 pilot 中 `2.25 vs 3.36` 的巨大结构性差距。Tree 每轮 target
forward 为 1，无 TreeAttention/cache 错误。该 smoke 只说明温度修复已加载且运行
链路正常；正式结论仍需重新运行同一 20 条 GSM8K/HumanEval pilot。

## 2026-08-11：逐节点目标支持、最长路径与 continuation rescue

> **历史实现通知：** 本节描述的“救援节点先进入整棵树、再参加最终路径竞争”是
> `target_path_rescue v1`，已被本文末尾的 v2 单-token extension 语义取代。v1 的
> 20 条结果和诊断仍保留，用来说明为何“获得救援资格”不等于实际增加 MAL。

### 为什么替换 sampled trunk 语义

修复温度后，20 条 `fastmtp_cactus_trunk_rescue_tempfix_pilot` 得到：

| 数据集 | Native | Chain Cactus | SpecCascade | sampled-trunk tree |
|---|---:|---:|---:|---:|
| GSM8K-20 质量/MAL | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 95.0% / 3.044 |
| HumanEval-20 质量/MAL | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 80.0% / 3.070 |

温度错误消失后，树与链的巨大异常差距消失，但 sampled-trunk 方法仍未利用完整
树：主干接受时同层旁支被标成 `BACKUP_UNUSED`；主干第一次拒绝后，旁支最多只
替代一个 correction token，且不会沿该旁支继续递归。因此多出的 target nodes
没有形成多条可比较的完整路径。

### 新验证规则

新模式为 `support_mode=target_path_rescue`。构树仍只读取 MTP 的 Q、熵、深度和
节点预算；target 信息只在整树一次 TreeAttention forward 后使用。对节点 `x`，
其父前缀对应 target 分布为 `P_v`：

```text
relative(x) = P_v(x) / max_z P_v(z)
normal_survival(x) = relative(x) >= tau_normal
```

所有满足普通条件的孩子同时存活；只有存活父节点的孩子会进入下一深度扫描。
候选集合 coverage 仅记录为诊断，不参与正向打分，避免通过机械加宽树来提高每个
兄弟的接受机会。

若一个孩子没有通过普通条件、当前还没到最深层，并且该路径尚未使用过救援，则
计算候选特定 Cactus 接受潜力：

```text
h(x) = min(P_v(x) + sqrt(2 delta P_v(x)(1-P_v(x))), 1)
A_cactus(x) = min(1, h(x) / Q_v(x))
rescue_score(x) = relative(x)^w * A_cactus(x)^(1-w)
```

候选还必须满足最低 relative、最低 target probability、EOS 保护和
`rescue_score >= tau_rescue`。每个父节点最多额外保留一个分数最高的失败孩子，
每条路径默认最多一次救援。该决策是确定性的 threshold rescue，不消耗额外随机
数；救援仅提供“继续查看已验证后继”的资格，并不保证最终提交。

最终只比较连续存活的 endpoint：

1. 先保留深度最大的路径；
2. 同长度路径比较 `geometric_mean(relative_i)`；
3. 选择 target 可信度最高者；
4. 提交所选 draft path，再提交该叶节点后的原始 target bonus。

Cactus score 只影响救援资格，不进入最终路径可信度，因此不能把一个 target 支持
较弱的救援节点包装成高可信路径。最深层不使用救援，因为它无法解锁任何后续
draft node。

### 当前默认档位与运行

默认 pilot 使用 D=3、节点上限 9、每父节点最多 3 个孩子：

```text
tau_normal = 0.20
tau_rescue = 0.18
rescue_min_relative = 0.02
rescue_min_target_prob = 1e-4
delta = 1.0
w = 0.65
max_rescues_per_path = 1
```

这些是实验起点，不是已经确认的最优参数。运行：

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate
./scripts/run_fastmtp_target_path_rescue_tree.sh
```

结果进入 `results/$RUN_TAG/comparison.md`，逐轮节点、target 支持、普通存活、救援
分数和最终路径位于对应 `tree_rounds.jsonl`。

### 验证状态与限制

CPU 测试覆盖最长路径优先、同长度 target 可信度决胜、普通兄弟仍存活时救援另一
分支并继续、以及最深层不救援。真实 FastMTP/TREE_ATTN 冒烟执行三个 96-token
请求，共 89 rounds；每轮 target forward 固定为 1，实际节点数 3--9，MAL 3.247。
6 个节点获得 continuation rescue，其中没有一个最终进入最长路径，因为其后继
未继续通过 target 支持。这说明阈值逻辑已经生效且没有在该提示上强制提交救援，
但单提示 MAL 不是正式 benchmark，不能据此判断准确率或相对 Cactus 的收益。

## 2026-08-11：救援改为基础路径后的单-token extension（v2）

> **历史结果通知：** v2 已完成 20 条实验，其结果和语义继续保留用于对照；后续
> `prefix_reopen_rescue` 不再把救援限制为选路后的单 token，而是在路径首次断裂处
> 重新开放已验证后继。新规则仍处于 pilot 阶段，尚未取代 v2 成为正式结论。

### v1 的 20 条结果与失效原因

用户完成的 `fastmtp_target_path_rescue_pilot` 得到：

| 数据集 | Native | Chain Cactus | SpecCascade | target-path rescue v1 |
|---|---:|---:|---:|---:|
| GSM8K-20 质量/MAL | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 95.0% / 3.173 |
| HumanEval-20 质量/MAL | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 75.0% / 3.132 |

GSM8K/HumanEval 分别有 `284/183` 个候选通过 v1 救援条件，但最终只有 `76/65`
个救援 token 被选择，对 MAL 的直接贡献仅 `+0.024/+0.020`。因此报告中的低
`rescue rounds` 不是阈值完全未触发，而是 v1 将救援定义成“恢复路径竞争资格”：
候选即使获救，也可能输给另一条同样长但 target 可信度更高的路径；若其后继未
普通存活，救援本身也未必让最终路径更长。

### v2 的两阶段语义

v2 明确分离两个阶段：

1. **普通松弛验证。** 每个节点在自己的父前缀下计算
   `relative=P(y)/max(P)`，`relative >= tau_normal` 即普通存活。扫描完整棵树后，
   先选普通存活深度最长的基础路径；只在同长度路径之间比较 target relative 的
   几何平均。
2. **一次救援延长。** 仅当基础路径深度小于 `D` 时，检查该路径末端的直接孩子；
   若基础路径为空，则检查 root。只从普通验证失败的孩子中计算
   `relative^w * A_cactus^(1-w)`，选出达到阈值的最高分候选并直接追加到基础路径。
   救援固定贡献一个 draft token，不重新参加整树路径竞争，不继续解锁第二个
   token，并继续受到 target probability floor 和 EOS 保护。

最终输出仍在选中路径后附加一次原始 target bonus，所以报告把接受长度拆成：

```text
ordinary relaxed accepted depth
+ one-token rescue extension contribution
+ target bonus
```

### 阈值与验证

对旧 20 条逐轮 P/Q 日志做 v2 反事实 replay，`tau_rescue=0.18` 预计仅带来
GSM8K/HumanEval `+0.022/+0.019` MAL。将它降到 `0.08` 后预计为
`+0.050/+0.034`；对应获救候选的 target-relative 中位数约为 `0.082`，因此新
pilot 默认使用 `0.08`。这是离线预测，不是新的质量结果，必须通过同一任务清单
实测准确率与 pass@1。

CPU 回归覆盖：最长基础路径、同长度 target tie-break、只沿选中前沿延长一个
token、以及仅剩一个深度槽时仍恰好延长一个 token。相关 compile/test 验证为
`52 passed`。真实 FastMTP/vLLM/TREE_ATTN 默认阈值 smoke 完成 89 rounds、每轮
target forward `1.000`，该单提示没有触发 extension；随后仅用于覆盖运行分支的
强制阈值 smoke 完成 109 rounds，其中 85 rounds 发生 extension，所有 extension
节点均为 `SELECTED`、terminal 为 `target-path-extension-bonus`，每轮 target
forward 仍为 `1.000`。强制阈值 smoke 不代表可用质量配置。

重新运行 20 条：

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate
RUN_TAG=fastmtp_target_path_extension_v2_pilot \
  ./scripts/run_fastmtp_target_path_rescue_tree.sh
```

新的 `tree_metrics.md` 会单独给出普通松弛深度、extension token 数和 extension
对 MAL 的直接贡献，避免再次把“救援资格”误当成“实际延长”。

### v2 实测结果

用户完成的 `fastmtp_target_path_extension_v2_pilot` 为：

| 数据集 | Native | Chain Cactus | SpecCascade | v2 tree |
|---|---:|---:|---:|---:|
| GSM8K-20 质量/MAL | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 85.0% / 3.181 |
| HumanEval-20 质量/MAL | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 80.0% / 3.123 |

v2 的 extension 只贡献 `+0.043/+0.027` MAL。更关键的是，root selection 只有
`85.3%/83.8%`：根节点一旦失败，v2 只能提交一个救援 token，无法利用同一次
target tree forward 已经验证过的后继。它因此没有处理日志中最主要的前缀截断
位置。

## 2026-08-11：前部路径重开与普通后继延伸（prefix-reopen）

### 为什么不再继续做概率连乘

vLLM 日志中的 per-position acceptance 是累计 reach rate，不是当前位置的条件
接受率；MAL 应按 `1 + r1 + r2 + r3` 计算，不能把这些日志值再相乘一次。算法内部
也不使用整条 target 概率乘积决定某个节点是否存活：节点只读取其自己父前缀下的
局部 `relative=P(y)/max(P)`。最终同长度路径用 relative 的几何平均比较，避免
长路径仅因多一个乘数而被系统性压低。

MTP 的累计 Q 乘积仍保留为**构树预算的 reach 代理**，因为一条 proposal 前缀
实际被走到需要其所有局部选择发生；它不参与 target 放行阈值。直接恢复 Q 的
几何平均会重新抬高早期低 Q 分支，并复现旧 allocator 挤掉高 reach 主路径的问题。

### 新规则

`support_mode=prefix_reopen_rescue` 保持普通规则 `relative>=0.20`，但改变首次失败
的因果位置：

1. 每个当前可达父路径先保留所有普通通过的孩子；
2. 若该路径尚未救援，所有同时满足 target probability、relative、EOS 和
   `relative^w * A_cactus^(1-w)` 阈值的普通失败孩子都可重开；
3. 重开的孩子记录一次 rescue debt，其已在同一次 target forward 中验证的后代
   可以继续按普通 relative 规则存活；
4. 后代不能再次救援，因此一条路径最多跨过一个低支持节点；
5. 阈值随深度收紧，默认 D=3 时为 `0.08/0.12/0.16`，把预算集中在能够解锁最多
   后继的前部位置；
6. 最终先选最大连续深度，同长度再按 target-relative 几何平均选可信路径。

这与 v2 的本质区别是：v2 的一次救援最多 `+1` draft token；prefix-reopen 的
一次救援如果后继重新与 target 对齐，可以同时解锁救援节点和若干普通后继。报告
字段 `rescue_unlocked_mal_gain` 专门统计这部分实际进入所选路径的贡献。

### 实现与运行

核心入口：

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate
RUN_TAG=fastmtp_prefix_reopen_rescue_pilot \
  GSM8K_SAMPLES=20 HUMANEVAL_SAMPLES=20 \
  ./scripts/run_fastmtp_prefix_reopen_rescue_tree.sh
```

它复用 `fastmtp_cactus_trunk_rescue_tempfix_pilot` 中相同任务和协议的 Native、
Cactus、SpecCascade 基线，只运行新树方法。默认仍为同等逻辑深度 D=3、最多 9
个 target tree nodes、每轮一次 target forward。实验结果写入
`results/$RUN_TAG/comparison.md`；逐轮诊断位于各数据集的 `tree_rounds.jsonl`。

### 当前验证状态

CPU 侧已覆盖：救援 root 后普通后继能够延伸、同一父节点的多个合格兄弟同时
存活、单路径不能二次救援，以及报告对“被救援节点解锁的实际 token 数”的统计。
真实 TencentBAC/FastMTP、vLLM 0.18、TREE_ATTN 冒烟在 3 个相同提示请求上完成
122 rounds：target forward/round 为 `1.000`，MAL 为 `2.361`，其中 rescue 解锁
贡献 `+0.205`；相同提示的 v2 为 124 rounds、MAL `2.315`，extension 贡献
`+0.097`。该匹配提示仅验证机制能够产生多 token 延伸，不能代替 GSM8K/HumanEval
质量实验，也不能据此宣称正式 MAL 提升。

## 2026-08-11：HumanEval 诊断与 target-margin 自适应救援

### Prefix-reopen 20 条结果

| 数据集 | Native | Cactus | SpecCascade | prefix-reopen tree |
|---|---:|---:|---:|---:|
| GSM8K 质量/MAL | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 95.0% / 3.262 |
| HumanEval 质量/MAL | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 75.0% / 3.135 |

HumanEval 的问题不是本轮 pass@1 低于基线，而是 MAL 只比 Native 增加 `0.025`，
尚未超过 SpecCascade。新救援对 GSM8K/HumanEval 分别贡献 `+0.105/+0.090`
MAL，说明重开机制有效，但 HumanEval 的普通接受深度从 v2 的 `2.096` 降为
`2.045`，额外救援改变轨迹后，下一轮 MTP 对齐损失抵消了部分收益。

逐轮相邻事件分析（请求边界未显式记录，因此只作机制诊断）显示：

| 前一轮事件 | GSM8K 下一轮 MAL | HumanEval 下一轮 MAL |
|---|---:|---:|
| ordinary path | 3.222 | 3.058 |
| rescue path | 2.854 | 2.627 |
| target fallback | 3.672 | 3.649 |

HumanEval 选择根部 branch 0/1/2 后，下一轮 MAL 分别为
`3.100/2.796/2.281`。这些是条件相关性，不是严格反事实，但一致说明：低 Q 备选
路径和低 target 支持救援容易把 FastMTP 带到后续更难对齐的状态。继续统一降低
阈值会同时放大 MAL 和结构错误风险。

### 自适应规则

新增父前缀 target top-1/top-2 log-margin：

```text
m_v = log P_v(top1) - log P_v(top2)
r_v = min(m_v / margin_reference, 1)
margin_multiplier = 1 + margin_penalty * r_v

effective_rescue_threshold
  = base_threshold
    * (1 + depth_penalty * (depth - 1))
    * margin_multiplier
```

默认参数为：

```text
base_threshold = 0.06
depth_penalty = 0.35
margin_reference = 1.5
margin_penalty = 1.0
minimum relative = 0.015
minimum target probability = 1e-4
```

因此目标不确定时，深度 1/2/3 的基础阈值约为 `0.06/0.081/0.102`，比旧版
`0.08/0.12/0.16` 更松；当 target log-margin 达到 1.5 时，阈值最多翻倍。
规则只读取目标分布的局部决断度，不识别代码、数学或具体 benchmark。

运行：

```bash
RUN_TAG=fastmtp_margin_adaptive_rescue_pilot \
  ./scripts/run_fastmtp_margin_adaptive_rescue_tree.sh
```

报告额外输出所选救援的平均 target log-margin 和实际阈值倍率，便于检查预算是否
仍集中到 target 强烈反对的位置。

### 验证状态

CPU 回归覆盖“相同候选在 target 犹豫时获救、在 target 明确时被拒绝”。真实
FastMTP/TREE_ATTN 冒烟使用两个代码提示和一个通用提示完成 121 rounds，每轮
target forward 为 `1.000`，MAL `2.380`，rescue 解锁贡献 `+0.116`，无
KV/cache/TreeAttention 错误。该 smoke 不评测 HumanEval pass@1，也不证明新默认
优于旧版；需要运行相同 20 条 pilot 后再决定是否扩大样本。

## 2026-08-11：margin 规则负结果与 sampled-primary 树恢复

> **历史/替代通知：** `prefix_reopen_rescue` 和其 target-margin 自适应版本保留为
> 失败机理对照，不再继续调阈值。新的 `sampled_primary_reopen` 尚处于 pilot，
> 没有取代冻结链式 ReMTP，也没有正式质量结论。

### 为什么 margin 自适应没有达到 HumanEval MAL 3.160

用户完成的 20 条结果为：

| 数据集 | Native | Chain Cactus | SpecCascade | margin-adaptive tree |
|---|---:|---:|---:|---:|
| GSM8K 质量/MAL | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 95.0% / 3.263 |
| HumanEval 质量/MAL | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 70.0% / 3.101 |

新规则在 GSM8K/HumanEval 仅贡献 `+0.071/+0.074` rescue MAL；所选救援的平均
target top-1/top-2 log-margin 仍为 `1.735/1.798`，对应阈值倍率为
`1.869x/1.892x`。也就是说，margin 同时放大阈值后没有把救援集中到 target
真正犹豫的位置，只减少了事件数量。

更重要的是跨轮相关性：HumanEval 在普通路径后的下一轮 MAL 约为 `3.015`，救援
路径后只有 `2.639`；GSM8K 分别为 `3.233/2.805`。这些数字混有请求边界和上下文
难度，只能作为诊断，但与 prefix-reopen 的旧日志方向一致。根本问题因此不是单个
threshold，而是旧动态树会用确定性 top-Q 候选和“最长路径优先”替换正常 proposal
轨迹；一轮多接收的收益会被下一轮更差的 MTP 对齐抵消。

### 新语义：Cactus 主干，树仅在拒绝时接管

`sampled_primary_reopen` 明确拆开主轨迹与备选树：

1. 每个深度的 local rank 0 从完整 MTP 分布 `Q` 采样，与 Native/Cactus 的
   proposal 语义一致；其他节点只是确定性高 Q backup。
2. sampled primary trunk 逐 token 使用与链式 Cactus 相同的 `H(y)/Q(y)` 接受
   概率。主干 token 一旦接受，所有同父 backup 标记为 `BACKUP_UNUSED`，不能仅因
   路径更长而替换它。
3. 只有主干第一次拒绝时，才构造相同 Cactus 残差 `(H-Q)+`。残差支持的 sibling
   可以执行一次近似树恢复；若失败，直接从原 Cactus 残差采 correction token。
4. 恢复 sibling 后，其已由同一次 target tree forward 验证过的后代可继续，但
   必须逐位置满足原始 target-relative 阈值，且不能再次 rescue。
5. 从恢复子树中先取最长连续路径，仅在长度相同时比较 target confidence；最终
   仍追加原始 target bonus。
6. 九节点预算按剩余深度均摊，并用 `reach_first` 排序：默认形成 sampled trunk
   加两个 root backup，尽量让三条 root 路径各自拥有后继，避免所有预算耗在浅层
   sibling 上。

这不是严格多 proposal speculative sampling：树恢复会改变输出分布。它的目标是
让未触发恢复的绝大多数轮保持链式 Cactus 轨迹，只把树的分布偏移限制在“Cactus
本来已经拒绝并准备输出 correction”的因果位置。

运行 20 条同协议 pilot：

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate
RUN_TAG=fastmtp_sampled_primary_reopen_pilot \
  ./scripts/run_fastmtp_sampled_primary_reopen_tree.sh
```

脚本复用匹配的 Native/Cactus/SpecCascade 结果，仅运行新树方法；若基线样本数不
匹配，则自动重跑四种方法。正式扩大样本前，优先判断 HumanEval 是否同时达到
`pass@1 >= 75%` 和 `MAL > 3.160`。若 pilot 仍失败，应先比较 sampled primary
自身的每位置 acceptance 与 Chain Cactus，而不是继续降低恢复阈值。

### 当前验证状态

CPU 回归新增完整恢复子树用例：sampled root 被拒绝、sibling 由 Cactus residual
接受、两个后代通过 target-relative 验证并与 target bonus 一起提交。动态树、
报告与相关测试共 `44 passed`，两个 shell 入口通过 `bash -n`。

真实 RTX 4090、TencentBAC/FastMTP、vLLM 0.18、TREE_ATTN 冒烟完成 3 个提示、
83 rounds：每轮 target forward 为 `1.000`，平均 target nodes 为 `8.494`；运行时
同时观察到 `cactus-trunk-bonus`、`cactus-trunk-correction` 和
`sampled-primary-reopen-bonus`。其中 15 轮触发恢复，恢复共解锁 22 个 draft
token，证明“拒绝后恢复并继续后代”真实执行。该非 benchmark smoke 的 MAL 为
`2.313`，不能用于判断 GSM8K/HumanEval 效果；正式 20 条结果仍待用户运行。
