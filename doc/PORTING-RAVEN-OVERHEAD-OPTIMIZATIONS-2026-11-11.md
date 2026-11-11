# RAVEN 每轮开销优化：全部改动汇总与 4090 移植指南（2026-11-11）

本文把 2026-11-10/11 两天做的三层优化合成一份可照着改的清单。所有数字来自 Jetson Orin NX（15W、eager、FastMTP FP4、D3/N9、冻结的 100+100 服务环境）；4090 上的收益没有测过，需要移植后自己量。

所有开关默认关闭。除第 5 节的可选项外，全部不改变接受规则、树预算、提议分布和 RNG 消费顺序：同一 prompt、同一 seed 下，开关前后输出逐 token 相同（第 1 项除外，见下）。

## 0. 一览

| 层 | 开关 | 省掉的东西 | Jetson 实测（每轮） | 输出是否逐 token 不变 |
| --- | --- | --- | ---: | --- |
| 1 | `REMTP_CANONICAL_TREE_ATTENTION=0` | 树注意力把前缀 KV 按节点数重复扫描 | 约 −25~30 ms（400 token 上下文），2k 上下文约 −150 ms | 否：1 个 bf16 ulp 的差异，约 3% 的调用出现，采样轨迹随后分叉 |
| 2 | `REMTP_DYNAMIC_TREE_FAST_SELECT=1` `REMTP_DYNAMIC_TREE_FAST_VERIFY=1` | 候选选择与验收里约 100 次 GPU→CPU 同步 | −20.6 ms | 是（3 题 513 轮） |
| 3a | `REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV=1` | 建树后的第 4 次 MTP forward，以及草稿 KV 的路径提交 | 3a+3b+3c 合计 −23.0 ms（相对 `FAST_COMMIT=1`），相对第 2 层 −25.9 ms | 是（3 题 513 轮） |
| 3b | `REMTP_DYNAMIC_TREE_FAST_METADATA=1` | 每轮 4~5 次在 GPU 上逐元素构造树掩码（每次 2.7 ms） | 同上 | 是 |
| 3c | `REMTP_DYNAMIC_TREE_FAST_COMMIT=2` | 路径提交里的恒等拷贝和暂存缓冲 | 同上 | 是 |
| 可选 | `REMTP_DYNAMIC_TREE_PATH_ORDERED_FIRST_PASS=1` | 草稿前缀被未接受的兄弟节点污染 | 待全天实验验证 | 否：改变提议分布 Q |

三层合计（同样三道 GSM8K 题）：每轮 1.3057 s → 1.2155 s。同上下文的 Cactus 约 1.143 s，差距从约 160 ms 降到约 72 ms。GSM8K 上按当前 MAL 增益算的收支平衡点约 69 ms。

## 1. 第一层：树注意力布局

- 现象：RAVEN 每轮耗时随上下文的斜率是链式方法的 8 倍（82 对 10.4 ms/千 token）。
- 原因：`remtp/canonical_tree_attention.py` 把 10 个树节点当成 10 条独立序列，每层把前缀 KV 扫 10 遍。vLLM 原生 `TREE_ATTN`（一条序列加 `qq_bias` 掩码）只扫一遍。
- 改法：环境变量 `REMTP_CANONICAL_TREE_ATTENTION=0`，不需要改代码。
- 4090 上先看启动日志。只有出现 `canonical ancestor KV reads, zero KV copies` 这一行，说明 canonical 布局在生效，这一项才相关。这个模块是 10-04 为 Jetson 数值逐位一致加的，上游提交 `3afafa1` 里没有。
- 已核实的事实：探针 552 次调用里 `qq_bias` 全部等于拓扑祖先掩码；537 次输出逐位相同，15 次有 1~2 行差 1 个 bf16 ulp。不是语义差异，但不能逐 token 复现旧记录。

## 2. 第二层：合并主机/设备同步

- 现象：固定开销里有约 100 次 `float(tensor)`、`.item()`、`torch.tensor(list, device=cuda)`。每次都让 GPU 排空队列。
- 新文件 `remtp/raven_fast_path.py`（整文件拷过去即可，无 Jetson 依赖）：
  - `select_dynamic_candidates_fast`：每个父节点 2 次传输。top-k 截点有并列时退回参考实现的全词表稳定排序。
  - `residual_hit_tree_verify_fast`：p/q 一次 gather，每个被验证节点一次传输，残差只在拒绝时计算。遇到 `protect_eos`、风险策略、概率上限这三种配置返回 `None`，由调用方走参考实现。
  - 采样助手 `sample_q_token_fast`、`sample_branch_q_token_fast`、`cactus_trunk_children_fast`、`temperature_probs_fast`：RNG 消费顺序与参考实现相同。
- 钩子（`remtp/dynamic_tree_vllm.py`）：
  - `_install_fast_select(config)`：把四个模块级函数换成快速版本。只支持 sampled-primary 系列模式且非 anchor-only，否则启动时报错。
  - `_dynamic_verify_dispatch`：`FAST_VERIFY=1` 且没有决策追踪时先试快速验证，返回 `None` 就走原路径。
  - `_dynamic_propose_tree` 里两处 `float(q[int(token)])` 循环改成一次 `q[ids].tolist()`。无开关，数值相同。
- 测试：`tests/test_raven_fast_path.py` 30 项，随机输入逐位对照 token、接受集合、终止原因、诊断字典、RNG 状态。

## 3. 第三层

### 3a. 跳过死的草稿 KV（收益最大的一项）

事实链，全部读代码确认，并由实机逐 token 相同验证：

1. `_dynamic_propose_tree` 末尾对全部节点再跑一次 `_run_dynamic_tree_nodes`，返回值被丢弃，唯一的副作用是把树节点的草稿层 KV 写进缓存。
2. 每次累积式 MTP 树 forward 都先重写自己全部节点的 KV 再做注意力，所以没有任何一次 forward 读别的 forward 留下的树节点 KV。
3. 下一轮 vLLM 的首次 MTP forward（`EagleProposer.propose` 的 first pass）处理上一步 target 的全部输入行，把草稿 KV 写到 target 的槽位 `A..A+n`，覆盖了上一轮全部草稿树槽位 `A..A+n-1`，也覆盖了 `_compact_selected_path` 刚搬好的草稿目标槽位 `A..A+a-1`。
4. 所以这次 forward 写的 KV 和草稿 KV 的路径提交都不会被读到。

改法（`_dynamic_propose_tree`）：开关打开且没有 anchor 剪枝时，不跑最后这次 forward，并把 `fixed._RUNTIME.draft_slots = None`，让路径提交跳过草稿部分。开着 `REMTP_DYNAMIC_TREE_PRUNE_ANCHOR_DESCENDANTS` 时启动报错，那个组合没有验证过。

移植前请在 4090 代码里确认第 3 条仍然成立：`propose_draft_token_ids` 走的是 `prepare_inputs_padded` 分支（没有设 `disable_padded_drafter_batch`），首遍 forward 的行数等于上一步 target 的输入行数。

### 3b. 树元数据缓存

- `_set_builder_topology` 每轮被调 4~5 次（三次草稿 forward 各一次，target 每个注意力组一次）。vLLM 的 `_prepare_tree_attn_bias` 在 GPU 上逐元素写 0/-inf，实测每次 2.67 ms。
- 改法：按 `(choices, dtype, device)` 缓存。首次在 CPU 上用同一个函数构造再一次拷到 GPU，之后命中缓存 0.0025 ms。上限 4096 项。掩码只读，vLLM 不会原地改它。
- `fixed6_vllm._offset_tensor`：`_tree_positions` 和 `_prepare_target_tree_inputs` 里的位置偏移小张量同样缓存。

### 3c. 原地路径提交

- 参考实现每层 gather、clone、scatter 各一次。`FAST_COMMIT=1` 是两段式多张量拷贝。
- `FAST_COMMIT=2`（`remtp/raven_fast_commit.py` 的 `copy_cache_slots_inplace`）：丢掉源等于目的的恒等对（主干第一个节点总是恒等）；其余每个被搬的节点一次 `torch._foreach_copy_`，按路径顺序执行，不用暂存缓冲。BFS 槽位下根到叶路径的顺序拷贝等价于缓冲式搬移，`inplace_order_is_safe` 逐次检查，不满足就退回两段式。
- 微基准（36 层，路径 0→3→6）：7.4 ms → 3.9 ms。剩下的大头是 Python 里建 144 个切片视图，不是拷贝本身。

### 第三层测试

`tests/test_raven_third_layer.py` 14 项：9 个节点里任取 1~3 个的全部 129 条递增路径 × 3 个基槽位，对照参考实现逐位相同；不安全顺序的回退；掩码缓存对照直接构造；偏移缓存；路径重排的行序与输入。另在 GPU 上核对了缓存掩码与 vLLM 原函数逐位相同。

## 4. 实测（GSM8K 第 17、353、856 题，同 prompt/seed/2048 预算）

| 配置 | 轮数 | 每轮耗时 | decode tok/s | 输出 |
| --- | ---: | ---: | ---: | --- |
| 基准配置（canonical=1） | 835 | 1.3057 s | 2.669 | 与 10 月记录逐 token 相同 |
| + 第一层 | 513 | 1.2620 s | 2.661 | 轨迹分叉（文本不同） |
| + 第二层 | 513 | 1.2414 s | 2.706 | 与上一行逐 token 相同 |
| + `FAST_COMMIT=1` | 513 | 1.2385 s | 2.712 | 相同 |
| + 第三层（3a+3b+3c） | 513 | 1.2155 s | 2.763 | 相同 |

## 5. 可选项：首遍 MTP 按接受路径重排（改变 Q，未定论）

读代码发现的一个草稿状态问题，不是性能问题：

- vLLM 的首遍 forward 假设 target 的输入行是一条链：第 `i` 行的草稿 KV 写到第 `i` 个槽位，被采样的行注意第 `0..i` 行。链式方法里这些行正好是已接受的前缀。
- 树的行是 BFS 顺序。接受路径是节点 0→3→6 时，叶子在第 7 行，它会注意到第 2、3、5、6 行，也就是被拒绝的兄弟节点。之后提交的草稿前缀槽位 `A+1..A+a` 里留下的也是 BFS 第 1~a 行而不是路径节点，而且永久留在草稿 KV 里。
- 只有主干没有兄弟、或只接受 0~1 个主干节点时，才和链式一致。

改法（`fixed6_vllm._path_ordered_first_pass`）：在 `EagleProposer.propose` 之前把 `target_token_ids`、`target_positions`、`target_hidden_states` 重排成“锚点、路径节点、其余行”，采样行号改成接受数 `a`。槽位映射不动。这样槽位 `A..A+a` 存的是（路径节点 j 的 target 隐状态，路径节点 j+1 的 token），与链式方法的不变量一致。验收规则完全不变，所以分布保证不受影响。

状态：三道题上每轮 token 3.452 对 3.359（343 轮对 513 轮，文本不同），差异在噪声范围内，不能当结论。全天实验里有一个 Spec-Bench 臂专门对照它。

## 6. 移植步骤

代码在 GitHub 分支 `jetson-r39-residual-adapt` 上，分成两个提交：先是 10 月的 Jetson 研究代码（三个钩子文件为 100+100 基准冻结时的版本），再是本文的全部优化。第二个提交的 diff 就是要移植的全部内容：

```bash
git fetch origin jetson-r39-residual-adapt
git log --oneline origin/jetson-r39-residual-adapt -3
C=$(git log --format=%h --grep='RAVEN per-round overhead' -1 origin/jetson-r39-residual-adapt)
git show --stat $C            # 改了哪些文件
git show $C -- remtp/          # 运行时代码的完整改动
```

如果 4090 的代码与 Jetson 分支差别不大，可以直接 `git cherry-pick $C`；差别大时按下面的步骤手工合并。

1. 拷贝新文件：`remtp/raven_fast_path.py`、`remtp/raven_fast_commit.py`、`tests/test_raven_fast_path.py`、`tests/test_raven_fast_commit.py`、`tests/test_raven_third_layer.py`。可选：`remtp/tree_attention_probe.py`、`scripts/ab_canonical_tree_attention.py`。
2. 打钩子补丁：`deploy/raven_overhead_port_20261111/raven_overhead_hooks.patch`。基线是 Jetson 上 10 月冻结的代码。

   ```bash
   cd <4090 仓库根目录>
   patch -p1 --dry-run < raven_overhead_hooks.patch   # 先看哪些 hunk 能打上
   patch -p1 < raven_overhead_hooks.patch
   ```

   对上游提交 `3afafa1` 试打的结果：`fixed6_vllm.py` 全部成功；`dynamic_tree_vllm.py` 7 个 hunk 里 6 个成功，失败的是 `_dynamic_verify_dispatch` 那一个（Jetson 侧多了决策追踪和重复保护参数）；`target_numerics.py` 上游没有，可以跳过，它只挂探针。
3. 失败的 hunk 手工改：在 `_dynamic_verify_dispatch` 里调用 `residual_hit_tree_verify(...)` 之前加

   ```python
   if os.getenv("REMTP_DYNAMIC_TREE_FAST_VERIFY", "0") == "1":
       from remtp.raven_fast_path import residual_hit_tree_verify_fast
       fast_result = residual_hit_tree_verify_fast(*args, **kwargs, config=config)
       if fast_result is not None:
           return fast_result
   ```

   如果 4090 的 `residual_hit_tree_verify` 签名与 Jetson 不同（例如没有 `protected_node_indices`、`decision_trace`），以 4090 的为准，并同步删掉 `residual_hit_tree_verify_fast` 里对应的参数。
4. 启动环境变量：

   ```bash
   export REMTP_CANONICAL_TREE_ATTENTION=0        # 仅当 4090 上启用了 canonical 布局
   export REMTP_DYNAMIC_TREE_FAST_SELECT=1
   export REMTP_DYNAMIC_TREE_FAST_VERIFY=1
   export REMTP_DYNAMIC_TREE_FAST_COMMIT=2
   export REMTP_DYNAMIC_TREE_FAST_METADATA=1
   export REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV=1
   ```

5. 核对启动日志里这五行都在：

   ```text
   [ReMTP][DynamicTree] fast candidate selection: two device transfers per parent
   [ReMTP][DynamicTree] fast residual-hit verification: batched transfers
   [ReMTP][DynamicTree] in-place ordered KV path commit, identity moves skipped
   [ReMTP][DynamicTree] cached host-built tree masks and position offsets
   [ReMTP][DynamicTree] dead draft KV skipped: no final tree rematerialization, no draft KV commit
   ```

6. 跑测试：

   ```bash
   python -m pytest -q tests/test_raven_fast_path.py tests/test_raven_fast_commit.py tests/test_raven_third_layer.py
   ```

7. 在 4090 上做等价性验证，这一步不能省：固定 2~3 个 prompt 和 seed，先全关开关跑一遍，再全开跑一遍，比较输出 token id。第 2、3 层必须逐 token 相同、轮数相同。不同就说明 4090 的代码路径和 Jetson 有差别（最可能是 3a 的前提不成立），先关掉 `SKIP_DEAD_DRAFT_KV` 再逐个排查。

## 7. 对 4090 的预期

没有测过，下面只是机制上的判断。

- 这些优化省的是 kernel 启动次数和同步次数，不是 FLOPs。4090 上每次同步的绝对时间比 Orin 小一个量级，但每轮总时间也小一个量级，所以占比不一定小。
- 3a 省的是一整次 MTP forward 加一次掩码构造，在任何硬件上都是纯收益。
- 第一层在 4090 上是否相关取决于那边有没有启用 canonical 布局。
- Jetson 上 FP4 线性层按 64 行补齐，target forward 对行数不敏感。4090 上如果不是这样，树的 10 行对 4 行会有真实的计算差，那一部分这几项优化管不到。
