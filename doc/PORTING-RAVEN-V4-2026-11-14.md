# RAVEN v4-safe 移植说明（2026-11-14，给 4090 参考）

本说明对应 `jetson-r39-residual-adapt` 分支上 2026-11-14 的两个提交，基线是 `44dec73`（v3 + 开销优化，移植说明见 `doc/PORTING-RAVEN-OVERHEAD-OPTIMIZATIONS-2026-11-11.md`）：

- `451e3c2`：**只含算法改动**（v4-safe 目标支持度门槛及其测试）。`git show 451e3c2` 即可看到完整的算法差异；只想对齐算法时只合并这一个提交即可，它单独通过测试。
- 紧随其后的提交：工程优化（输出逐 token 不变）、分析工具、文档。
- 2026-11-15 的提交：GPU 建树（`remtp/gpu_tree.py`）和在 4090 上直接运行本分支的脚本。**直接在 4090 上跑本分支请看 `doc/RUN-ON-4090-2026-11-15.md`。**

改动分两类，移植时请分开对待：

- **算法改动**（第 1 节）：只有一处，改变输出。4090 若要与 Jetson 跑同一个方法，必须移植。
- **工程改动**（第 2 节）：只改实现，同种子输出逐 token 不变。按需移植，其中有些只对 Jetson 有意义。

设计依据和实验数据见 `doc/RESULT-RAVEN-V4-TARGET-GATE-2026-11-13.md`（第 1–2 节 v4 设计，第 7 节 GPTQ 公平对比，第 9–10 节工程优化）。

---

## 1. 算法改动：目标支持度门槛（v4-safe）

### 1.1 出发点

- Cactus 围绕草稿给出的 token 做松弛：草稿提出哪个 token，就给哪个 token 加 √(2δp(1−p)) 的接受额度。p 很小时这个额度远大于 p，目标几乎不支持的 token 也会被接受。
- MTP 头作为草稿并不可靠。日志显示 Cactus 接受目标概率 <1e-3 的 token 的频率约为精确采样的 35 倍。
- RAVEN v3 的自适应规划会把 Cactus 在饱和位置用不满的名义额度回收后重新分配，实际放宽得比 Cactus 还多。
- **v4 的原则：由目标模型决定在哪里松弛、松弛多少，不信任草稿。**

### 1.2 规则

对主干上每个被验证的草稿 token y（p = 目标在父行上给 y 的概率，q = 草稿给 y 的概率）：

1. **门槛**：若 p(y) < 0.01，该位置不松弛，按精确规则验证（接受概率 min(1, p/q)，拒绝后的修正分布也是精确残差）。门槛只看目标分布，与 q 无关。
2. **额度**：其余位置的名义额度仍是 Cactus 的 √(2δp(1−p))（只取决于 p），自适应规划在名义额度内分配。
3. **上限 1.0 倍**：自适应规划给任一位置的额度不超过名义额度，因此每个位置的接受概率都 ≤ Cactus 在同一 token 上的接受概率。
4. 被门槛挡住的位置，其额度不再被重新分配（默认的 `nominal` 额度池下，回收量在门槛之后计算）。

提议、树结构（D3/N9）、修正采样、目标锚点（`residual_hit_anchor`）都不变。

### 1.3 代码位置

| 文件 | 改动 |
| --- | --- |
| `remtp/dynamic_mtp_tree.py` | `DynamicTreeConfig` 新增字段 `relaxation_min_target_probability`、`relaxation_min_target_ratio`、`adaptive_relaxation_pool`；`validate()` 的取值检查；属性 `target_support_gate` |
| `remtp/dynamic_mtp_tree.py` | 规划器拆成 `_plan_vectors`（额度向量，设备和主机共用同一运算序列）和 `_plan_from_values`（Python 分配逻辑）；**门槛逻辑在 `_plan_from_values` 里**，位于计算回收额度池之前（搜索 "Target-support gate"）；新增 `adaptive_trunk_relaxation_plan_host`（快速验证路径用，结果与设备版逐位相同） |
| `remtp/dynamic_tree_vllm.py` | `dynamic_tree_config_from_env()` 读取三个新环境变量；启动横幅打印 `target_gate_min_p=`、`target_gate_min_ratio=`、`adaptive_pool=` |
| `remtp/raven_fast_path.py` | 快速验证改为一次读回后调用 `adaptive_trunk_relaxation_plan_host`（不改变结果） |

默认值保持旧行为：门槛为 0、额度池为 `nominal` 时，规划结果与 v3 完全相同。

### 1.4 环境变量

在 v3 的配置上增加或修改：

```bash
REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_PROB=0.01      # 新增：目标支持度门槛
REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MAX_RATIO=1.0    # 由 1.5 改为 1.0
# 以下保持不变（门槛要求自适应规划开启）
REMTP_DYNAMIC_TREE_ADAPTIVE_RELAXATION=1
REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MIX=0.5
REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_POOL=nominal     # 默认值，可不写
```

Jetson 上 v4-safe 的完整算法配置（其余为 `scripts/serve_fastmtp_jetson.sh` 的默认值和冻结的 `adaptive_original/attempt_2` 环境）：

```bash
REMTP_DYNAMIC_TREE_SUPPORT_MODE=residual_hit_anchor
REMTP_DYNAMIC_TREE_MAX_DEPTH=3
REMTP_DYNAMIC_TREE_MAX_NODES=9
REMTP_DYNAMIC_TREE_MAX_CHILDREN=3
REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO=0.01
REMTP_DYNAMIC_TREE_TAU_MIN=0.001
REMTP_DYNAMIC_TREE_KAPPA=1.5
REMTP_DYNAMIC_TREE_MU=0.5
REMTP_DYNAMIC_TREE_ETA=0.0
REMTP_DYNAMIC_TREE_ALLOCATION=residual_coverage
REMTP_DYNAMIC_TREE_CACTUS_DELTA=1.0
REMTP_DYNAMIC_TREE_RELAX_DEPTH_WEIGHTS=1.0,1.0,1.0
REMTP_DYNAMIC_TREE_ADAPTIVE_RELAXATION=1
REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MIX=0.5
REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MAX_RATIO=1.0
REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_TARGET_POWER=0.25
REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_STEPS=16
REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_PROB=0.01
REMTP_DYNAMIC_TREE_RELAX_MAX_PROBABILITY_RATIO=0
REMTP_DYNAMIC_TREE_RELAX_RISK_POLICY=none
REMTP_DYNAMIC_TREE_FRONTIER_RESCUE=0
REMTP_DYNAMIC_TREE_PATH_SELECTION=balanced
REMTP_DYNAMIC_TREE_PATH_TEMPERATURE=0
REMTP_DYNAMIC_TREE_PROTECT_EOS=0
REMTP_DYNAMIC_TREE_REPEAT_GUARD=0
REMTP_REQUEST_LOCAL_BRANCH_RNG=1
```

### 1.5 验证

```bash
python -m pytest -q tests/test_target_support_gate.py   # 15 项
```

覆盖：参数校验；默认值不改变旧规划；门槛只由目标决定（换一个草稿分布，被挡住的位置完全不变）；v4-safe 每个节点的接受概率 ≤ Cactus，门槛以下等于 min(1, p/q)；快速验证与参考实现逐位相同（含随机数状态）。

启动后检查横幅里有 `target_gate_min_p=0.01` 和 `adaptive_max_ratio=1`。

### 1.6 Jetson 上的效果（供对照，不要求 4090 复现绝对值）

GPTQ 模型，HumanEval 前 30 题，输出上限 512（`results/gptq_ab_20261113/`）：

| 方法 | MAL | p<1e-3 接受/千 token | 每位置多接受的质量 |
| --- | ---: | ---: | ---: |
| Native（精确） | 2.883 | 0.00 | 0.000 |
| Cactus | 3.131 | 2.46 | 0.032 |
| RAVEN v4-safe | 3.320 | 0.00 | 0.023 |

v4-safe 的 MAL 比 Cactus 高 6%，同时去掉了目标概率 <1e-3 的接受。30 题的 pass@1 区分不了方法。

### 1.7 另一个变体（未采用）

v4-gated：`ADAPTIVE_RELAX_POOL=gated`、`ADAPTIVE_RELAX_MIX=1.0`、`ADAPTIVE_RELAX_MAX_RATIO=1.5`。Cactus 在门槛以下位置会用掉的额度，可以挪给目标支持的位置，整轮实际松弛总量不超过 Cactus。实测与 v4-safe 几乎相同，最终采用更简单的 v4-safe。

---

## 2. 工程改动（输出逐 token 不变）

所有开关默认关闭。每一项在 Jetson 上都用同种子 A/B 验证过输出逐 token 相同（`scripts/ab_canonical_tree_attention.py`）。

| 开关 | 文件 | 作用 | 4090 是否需要 |
| --- | --- | --- | --- |
| `REMTP_DYNAMIC_TREE_BATCH_SELECT=1`（需 `FAST_SELECT=1`） | `remtp/raven_fast_path.py`（`batched_trunk_children`）、`remtp/dynamic_tree_vllm.py` | 每层候选选择只读回一次；每个父节点的设备运算从约 72 个减到约 22 个 | 建议移植 |
| 并列回退只看可选值（无开关） | `remtp/raven_fast_path.py`（`_relevant_tie`） | 低于 `min_draft_prob` 的并列不再回退到全词表排序；结果与参考实现相同 | 随 `raven_fast_path.py` 一起 |
| `REMTP_TREE_HOST_METADATA=1` | `remtp/tree_cudagraph.py` | vLLM 0.18 的树注意力在第一层用两次 `.item()` 惰性计算子元数据；改为在 `build()` 里用主机标量构建。`REMTP_TREE_HOST_METADATA_CHECK=1` 可逐次与设备值核对 | 建议移植（vLLM 0.18 专用） |
| `REMTP_TREE_CUDAGRAPH=1` + `ENFORCE_EAGER=0` + `REMTP_COMPILATION_CONFIG` | `remtp/tree_cudagraph.py`、`scripts/serve_fastmtp_verified.sh` | 树注意力支持完整 CUDA graph：目标掩码放进固定缓冲区，TREE_ATTN 声明完整图支持（要求 `max_num_seqs=1`）。RAVEN 用 `{"mode":0,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,...,10]}` | 可选；若开，所有方法都要开 |
| 链式方法 CUDA graph | 同上 | Cactus/Native 用 `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4]}`，AR 用 `[1]`；不经过 torch.compile，内核与 eager 相同 | 与上一行一起 |
| `REMTP_DYNAMIC_TREE_ASYNC_HOST=1` | `remtp/raven_fast_path.py`、`remtp/fixed6_vllm.py`、`remtp/dynamic_tree_vllm.py` | 去掉每轮约 10 次不必要的同步：可分页内存 H2D 拷贝改为固定内存 `non_blocking`、0 维 CUDA 张量下标改为单元素下标、Python 标量赋值改 `fill_`、缓存主机温度、审计关闭时不算 top1 | 建议移植 |
| `REMTP_DYNAMIC_TREE_GPU_BUILD=1`（2026-11-15） | `remtp/gpu_tree.py`、`remtp/dynamic_tree_vllm.py` | 候选选择、保留和下一层输入都在 GPU 上完成，每层"选择 + 保留"是一个 CUDA graph，提议阶段只读回一次。决策与主机路径相同（校验模式 `REMTP_DYNAMIC_TREE_GPU_BUILD_CHECK=<jsonl>`），但固定行数的 GEMM 与 1 行 GEMM 最后一位不同，约 18% 的题会在较后的位置分开。Jetson 上每轮快 3% | 可选；只支持 D3/N9/每父 3 子、`residual_coverage` |
| `REMTP_CUDA_SPIN_WAIT=1` | `remtp/cuda_spin_wait.py` | 主上下文设 `CU_CTX_SCHED_SPIN`，等待 GPU 时自旋而不是睡眠 | **Jetson 专用**：解决 `schedutil` 在线程睡眠时把 CPU 降到最低档的问题。4090 主机先查 `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`，若是 `performance` 就不需要 |

Jetson 上的效果（GSM8K 3 题，解码 tok/s，eager → 图模式 + 自旋，RAVEN 另加异步主机读）：AR 14.8 → 16.7，Native 19.4 → 23.7，Cactus 22.1 → 26.8，RAVEN v4-safe 19.5 → 24.5。4090 的 CPU 发射比 Jetson 快得多，收益会小很多，需要实测。

公平性：若开 CUDA graph，所有方法（AR、Native、Cactus、RAVEN）都要用同一套配置；`scripts/run_w4a16_maxn_suite.py` 里 `SUITE_ENGINE_OPT=1` 就是按这个规则统一设置的，可参考。

---

## 3. 不需要移植的部分

- Jetson 目标数值补丁（`REMTP_TARGET_*`、bitsandbytes 行块等）和 W4A16/GPTQ 量化脚本（`scripts/quantize_fastmtp_w4a16_*.py`）：Jetson 内存不够跑 BF16 目标才需要。
- 分析工具（都是只在设置环境变量时生效的观测代码）：`remtp/round_logger.py`（逐轮日志）、`remtp/gpu_event_timing.py`（CUDA event 计时）、`remtp/host_timeline.py` + `scripts/analyze_host_timeline.py`（主机时间线和同步位置）、`remtp/torch_trace.py`、`remtp/runtime_profile.py` 的新钩子。4090 上做性能分析时可以直接用。
- `scripts/test_mtp_row_split_bitwise.py`：检查"一次算 M 行"和"拆成 1 行 + M−1 行"的 MTP 层 GEMM 是否逐位相同（"主干先行、分支后补"方案的前提）。Jetson 上只有 M=1 走了不同内核；4090 若要评估该方案，可以直接跑：`python scripts/test_mtp_row_split_bitwise.py <模型目录> 20`。

## 4. 建议的移植顺序

1. 拷贝第 1.3 节的三个文件（或直接合并本次提交），跑 `tests/test_target_support_gate.py`。
2. 加上第 1.4 节的两个环境变量，启动 RAVEN，检查横幅。
3. 在同一批题目和种子上对比 v3 与 v4-safe：MAL 预计下降约 1.5–3%（Jetson：按日志离线估计 −1.4%，RTN 模型 HumanEval-30 实测 3.214 → 3.128，−2.7%），目标概率 <1e-2 的接受应降到精确采样附近。
4. 需要提速时再按第 2 节逐项打开工程开关，每打开一项都用同种子 A/B 确认输出逐 token 相同。

相关测试：`tests/test_target_support_gate.py`、`tests/test_batched_trunk_children.py`、`tests/test_tree_cudagraph.py`、`tests/test_async_host.py`、`tests/test_raven_fast_path.py`、`tests/test_dynamic_mtp_tree.py`、`tests/test_round_logger.py`。
