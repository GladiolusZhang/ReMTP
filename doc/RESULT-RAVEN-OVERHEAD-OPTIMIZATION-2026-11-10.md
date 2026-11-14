# Adaptive RAVEN 每轮开销的来源与两层优化（2026-11-10）

状态：机制分析完成；注意力布局 A/B 和同步合并的实机验证见下文"实测"；通宵 GSM8K-100 与 Spec-Bench 结果见第 9 节；第三层优化见第 10 节。全部在 Jetson Orin NX、15W、eager、与 100+100 正式基准完全相同的冻结服务环境（`results/raven_full_100x100_20261010/servers/adaptive_original/attempt_2/effective_env.json`）下进行。本文不改变任何接受规则、树预算（D3/N9）或输出分布；目标只是降低每轮的时间成本。

## 1. 问题：MAL 升了，端到端没有快

100+100 基准（GSM8K）里，RAVEN 对 Cactus：每轮提交 token 3.471 对 3.278（+5.9%），每轮耗时 1.319 s 对 1.147 s（+15.0%），因此 decode 吞吐 −7.9%。按逐请求回归（R²≈0.99）：

| 方法 | 每轮耗时 = 固定 + 斜率 × 上下文(千 token) |
| --- | --- |
| Native / Cactus / SpecCascade | 1.139 s + 10.4 ms/k |
| Adaptive RAVEN（D3/N9） | 1.255 s + 82 ms/k |
| deep4（D4/N12，历史配置） | 1.32 s + 115 ms/k |

多出的约 172 ms/轮 = 固定约 115 ms + 随上下文增长的约 57–66 ms（平均上下文约 800 token）。目标模型 forward 本身（约 1.01 s）几乎不随行数变化：FP4 线性层按 64 行补齐，10 行和 4 行同价。**树在 FLOPs 上几乎免费；开销全在树周围的实现上。**

## 2. 随上下文增长的部分：canonical 树注意力

`remtp/canonical_tree_attention.py`（2026-10-04 为数值逐位一致而加入）把 10 个树节点当成 10 条独立的单 token 序列，每一层都把前缀 KV 扫 10 遍。原生 vLLM `TREE_ATTN` 路径（一条序列 + 10×10 `qq_bias`）只扫一遍。

注意力核微基准（36 层合计，同一 FP32 P×V 核）：

| 上下文 | 链式 4 行 | 原生 qq_bias 10 行 | canonical 10 行 |
| ---: | ---: | ---: | ---: |
| 256 | 17.1 ms | 31.1 ms | 63.5 ms |
| 512 | 21.1 | 48.0 | 78.4 |
| 1024 | 31.1 | 42.8 | 124.1 |
| 2048 | 38.4 | 81.3 | 228.4 |

两种树布局在合成输入上输出逐位相同。实机探针（`remtp/tree_attention_probe.py`，`REMTP_TREE_ATTENTION_PROBE_JSONL`）在真实生成的 552 次目标模型树注意力调用中确认：原生 `qq_bias` 掩码全部等于拓扑祖先掩码；537 次输出逐位相同，15 次有 1–2 行差 1 个 bf16 ulp（最大 0.0039，输出均值 0.116）。这不是语义差异，但足以让采样轨迹在几十到几百个 token 后分叉，所以原生路径不能逐 token 复现 10 月的记录；10-04 第四轮定位到的两个真实数值根因（RMSNorm 行数分派、BF16 LM head 舍入）与注意力布局无关，各有独立开关。

## 3. 固定的约 115 ms：主机/设备同步，不是计算

10-07 的分段计时（32 轮，GPU stream 时间）：RAVEN 对 Native，proposal +66 ms（候选选择 25 ms、`normalize_distribution` 每轮 23 次约 20 ms、多一次 MTP forward 约 17 ms）、验证 +29 ms、KV 路径提交 +17 ms。读代码：每次 `normalize_distribution` 有一次 `float(total)`、每个候选有 `float(q[token])`、每个节点有 `float(p_row[token])/float(q_row[token])/float(verify_row[token])/rand().item()`、每次 `sample_categorical` 两次同步……一轮约 100 次 GPU→CPU 同步，每次让 GPU 排空流水线。在 612 MHz 的 Orin 上这是十几到几十毫秒；在 4090 上是几毫秒。

## 4. 优化实现

全部默认关闭、由环境变量开启，数学与参考实现相同；`tests/test_raven_fast_path.py` 用随机输入逐位对照参考实现（输出 token、接受集合、终止原因、诊断字典、RNG 状态）。

| 开关 | 内容 |
| --- | --- |
| `REMTP_CANONICAL_TREE_ATTENTION=0` | 改回原生 `qq_bias` 树注意力（已有开关，基准中为 1） |
| `REMTP_DYNAMIC_TREE_FAST_SELECT=1` | `select_dynamic_candidates`/`_cactus_trunk_children`/采样助手：每个父节点 2 次传输；top-k 截点有并列时自动退回参考的全词表稳定排序；温度值每请求读一次 |
| `REMTP_DYNAMIC_TREE_FAST_VERIFY=1` | `residual_hit_tree_verify`：按行批量归一化（保留逐行求和以逐位一致）、所有节点的 p/q 一次 gather、每个被验证节点一次传输、残差只在拒绝时计算 |
| 无开关 | `_dynamic_propose_tree` 里两处 `float(q[int(token)])` 循环改为一次 gather（等价，始终生效） |

文件：`remtp/raven_fast_path.py`、`remtp/dynamic_tree_vllm.py`（分派与安装）、`remtp/tree_attention_probe.py`、`remtp/target_numerics.py`（探针挂载）、`scripts/ab_canonical_tree_attention.py`、`tests/test_raven_fast_path.py`。

## 5. 实测（GSM8K 第 17、353、856 题，与基准相同的 prompt/seed/2048 预算）

| 臂 | 配置 | 输出 token | 轮数 | 每轮耗时 | 每轮 token | decode tok/s | 与 10 月记录 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | canonical=1（基准配置） | 2910 | 835 | 1.3057 s | 3.485 | 2.669 | 3 条逐 token 相同，每轮耗时相同 |
| B | canonical=0 | 1723 | 513 | 1.2620 s | 3.359 | 2.661 | 第 254/152/41 个 token 分叉 |
| C | canonical=0 + FAST_SELECT + FAST_VERIFY | 1723 | 513 | 1.2414 s | 3.359 | 2.706 | 与臂 B 3 条逐 token 相同 |

臂 A 证明测量链路可信。臂 B 的输出比 A 短（分叉后走了不同轨迹，平均上下文约 414 对 643 token），所以 −44 ms 里有一部分来自上下文更短；按回归斜率折算，相同上下文下 canonical→qq_bias 约省 25–30 ms/轮，与微基准（约 400–500 token 时 30 ms）一致，在 2k 上下文时约 150 ms。B 的每轮 token 数 3.359 对 3.485 是不同文本上的数字，不能解释为 MAL 变化。

| D | C + FAST_COMMIT（36 层 KV 提交两次 `_foreach_copy_` 融合） | 1723 | 513 | 1.2385 s | 3.359 | 2.712 | 与臂 B/C 3 条逐 token 相同 |

臂 C 与臂 B 处理的是完全相同的 513 轮（输出逐 token 相同），所以 −20.6 ms/轮（−1.6%）、decode +1.7% 是纯粹的实现开销下降，不含任何轨迹差异。臂 D 的融合提交（`remtp/raven_fast_commit.py`，`REMTP_DYNAMIC_TREE_FAST_COMMIT=1`，6 项 CPU 等价测试）只再省 2.9 ms/轮（三题分别 −1.0/−4.0/−3.2 ms）：分段计时里 `cache_update` 的 17.9 ms 大部分不是这 110 次小 kernel 本身，这一项的预计收益下调。两层合计，在约 400 token 上下文下 RAVEN 每轮从约 1.289 s 降到 1.241 s（约 −48 ms）；对同上下文的 Cactus（约 1.143 s）仍多约 98 ms，还没有到按当前 MAL 算的收支平衡点（约 70 ms）。

逐题每轮耗时（B → C）：q17 1.2638→1.2429，q353 1.2628→1.2424，q856 1.2589→1.2385。

## 6. 还剩什么（第三层候选，按预计收益）

臂 C 配置下用仓库的分段计时器再采 32 轮（`AB_PROFILE=1`，短上下文，GPU stream 时间，ms/轮），与 10-07 的 Native、B 对照：

| 顶层组件 | Native | B（10-07） | C（fast） | C 对 Native |
| --- | ---: | ---: | ---: | ---: |
| target_forward | 1011.0 | 1027.7 | 1016.1 | +5（树注意力 8.0 对链式 3.2） |
| proposal | 103.1 | 169.1 | 161.9 | +59（含第 4 次 MTP forward 约 12、树元数据约 14、其余编排约 30） |
| sampling / verification | 1.3 | 30.5 | 18.5 | +17 |
| target_logits | 21.3 | 21.4 | 21.4 | 0 |
| cache_update（KV 路径提交） | 0.0 | 17.2 | 17.9 | +18 |
| 整轮 | 1139 | 1271 | 1241 | +102 |

按这份分布，下面是每轮还能收回的部分：

1. 建树结束后对 9 个节点再跑一次完整 MTP forward（`_run_dynamic_tree_nodes` 的最终重物化，约 17 ms）。深度 ≤2 节点的草稿 KV 在构树时已经按稳定的 BFS 槽位写好，这一步只为深度 3 节点补 KV；可以推迟到下一轮的第一次 MTP forward 里顺带算（把被接受的深度 3 节点作为额外一行），每轮省一次 forward。
2. 路径提交的逐层 gather/scatter 已用 `torch._foreach_copy_` 两段式融合（臂 D），实测只省约 3 ms；`cache_update` 剩下的约 15 ms 需要再分段计时才能归因（候选：`_update_states_after_model_execute` 原逻辑、GDN 状态分支判断、`.tolist()` 同步）。

## 8. 通宵实验（2026-11-10 夜，`results/overnight_20261110/`）

`scripts/run_overnight_20261110.py`（已冒烟测试三个阶段）依次运行：① 优化后 RAVEN（臂 D 配置）在 100+100 基准的 100 道 GSM8K 题上按原协议（同 prompt/seed/2048 预算）完整重跑并判分，报告与 10 月记录中 Native/Cactus/SpecCascade/原 RAVEN 相同题目的数字并列；② Spec-Bench 6 任务各 10 条、512 token、temperature 0.6，优化后 RAVEN；③ 同样抽样的 Cactus；④ 时间允许时 Native。进度看 `status.json`，汇总看 `overnight_report.md`，每条请求即时落盘可续跑（重复同一命令会跳过已完成的题）。
3. 其余 Python 编排里的小 kernel 与主机→设备拷贝：每轮 3–4 次用 Python 双重循环逐元素写 `tree_attn_bias`（每次约 26 个 1 元素 kernel）、多处 `torch.tensor(list, device=cuda)`（分页内存拷贝即同步点）、`adaptive_trunk_relaxation_plan` 内 4 次小张量构造。估计 10–15 ms；做法是按拓扑缓存这些张量、在 CPU 上构造掩码后一次拷贝。
4. 原生 qq_bias 树注意力比链式仍多 12–43 ms（10 行要 3 个 query block，各扫一遍前缀）。更深的改法是让 10 个 query 共用一次前缀扫描再合并尾部（flash-decoding 式两段 softmax），属于 kernel 级工作。

按 1–3 合计约 40–50 ms，加上已完成的两层，RAVEN 每轮开销可降到约 50 ms，低于当前 MAL 的收支平衡点；这时每轮多提交的 0.19 个 token 才会体现为端到端更快。

## 9. 通宵实验结果（2026-11-10 23:04 至 2026-11-11 10:53，全部阶段完成）

三个阶段全部跑完（Native 的 Spec-Bench 因超过 10 小时截止被跳过）。原始数据 `results/overnight_20261110/`。

GSM8K 100 题（与 100+100 基准相同题目/种子/2048 预算；`raven_opt` 为本次实跑，其余为 10 月记录）：

| 方法 | 正确率 | E2E tok/s | MAL | 每轮秒 | decode tok/s | 平均输出 token | 平均整题秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 84% | 2.5099 | 2.897 | 1.1482 | 2.5261 | 668.4 | 266.3 |
| cactus | 87% | 2.8333 | 3.274 | 1.1471 | 2.8579 | 561.1 | 198.0 |
| spec_cascade | 85% | 2.6615 | 3.076 | 1.1484 | 2.6818 | 599.2 | 225.1 |
| adaptive_original（10 月 RAVEN） | 88% | 2.6116 | 3.468 | 1.3192 | 2.6309 | 639.2 | 244.7 |
| raven_opt（臂 D 配置） | 89% | 2.7733 | 3.471 | 1.2435 | 2.7935 | 680.1 | 245.3 |

- MAL 保持（3.471 对 3.468），每轮耗时 −75.7 ms（−5.7%），E2E 吞吐 +6.2%（对原 RAVEN）、+10.5%（对 Native），对 Cactus 仍 −2.1%。
- 逐请求回归（上下文取输出 token 的一半，不含 prompt）：原 RAVEN 每轮 1.265 s + 81.8 ms/千 token；优化后 1.226 s + 25.4 ms/千 token；链式方法 1.140 s + 10.6 ms/千 token。斜率降到原来的 31%，固定项降约 39 ms。逐题配对：对原 RAVEN 平均 −56.0 ms/轮，对 Cactus 平均 +91.4 ms/轮。
- 100 题中 17 题输出与 10 月 RAVEN 逐 token 相同，其余在 1 ulp 数值差后分叉（见第 2 节），所以正确率 89 对 88、平均输出长度 680 对 639 属于不同采样轨迹，不能解释为质量变化。平均整题秒没有下降是因为这次输出更长。

Spec-Bench（6 任务各 10 条，multi_turn 按轮计 20 次请求，512 token，temperature 0.6，同一抽样；两方法均为本次实跑）：

| 任务 | RAVEN-opt decode tok/s | Cactus decode tok/s | RAVEN-opt MAL | Cactus MAL |
| --- | ---: | ---: | ---: | ---: |
| translation | 2.252 | 2.188 | 2.788 | 2.490 |
| summarization | 2.100 | 2.148 | 2.653 | 2.463 |
| math_reasoning | 2.761 | 2.798 | 3.411 | 3.195 |
| rag | 2.246 | 2.185 | 2.839 | 2.507 |
| qa | 2.176 | 2.084 | 2.712 | 2.379 |
| multi_turn | 2.447 | 2.415 | 3.046 | 2.763 |
| overall（70 次请求） | 2.396 | 2.361 | 2.990 | 2.701 |

- overall：MAL +10.7%，decode 吞吐 +1.5%，E2E 吞吐 2.336 对 2.306（+1.3%）。这是优化后 RAVEN 第一次在端到端上不低于 Cactus；但差距只有 1.5%，单次运行、70 次请求、两方法生成的文本不同，只能说"持平略高"，不能当作显著结论。
- 4 个任务领先、2 个落后。summarization（长上下文，树注意力斜率仍是链式的 2.4 倍）和 math_reasoning（Cactus 自身接受率已很高，MAL 增益只有 6.8%）落后，与第 6 节的剩余开销图一致。
- 折算每轮耗时：RAVEN-opt 约 1.248 s，Cactus 约 1.144 s，仍多约 104 ms/轮。按 Spec-Bench 的 MAL 增益（+10.7%），收支平衡点约 122 ms/轮，已越过；按 GSM8K 的 MAL 增益（+6.0%），平衡点约 69 ms/轮，还差约 22 ms。第三层（第 6 节 1–3 项，合计约 40–50 ms）做完后两个数据集都应转正，待实测。

运行环境备注：03:22 起校园网出网中断（WiFi 重新关联、DNS 降级、NetworkManager 进入 CONNECTED_SITE，Tailscale 连不上 DERP 与协调服务器，持续到 13:08），机器本身没有宕机，实验是本机回环请求，未受影响；13:08 机器被硬复位（复位原因 SYS_RESET_N），此时全部阶段已在 10:53 完成。

## 10. 第三层优化（2026-11-11 下午，已实机验证）

第 6 节的候选重新读代码后落成三项，全部默认关闭、输出逐 token 不变。移植清单见 `doc/PORTING-RAVEN-OVERHEAD-OPTIMIZATIONS-2026-11-11.md`。

| 开关 | 内容 |
| --- | --- |
| `REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV=1` | 不再跑建树后的第 4 次 MTP forward，也不再提交草稿 KV。原先设想的"推迟重物化"不需要：这次 forward 写的草稿树 KV 从来不会被读到。每次累积式树 forward 先重写自己全部节点的 KV 再做注意力；下一轮 vLLM 的首遍 MTP forward 按 target 的全部输入行把草稿槽位 `A..A+n` 重写一遍，覆盖了上一轮的全部树槽位和刚提交的路径槽位 |
| `REMTP_DYNAMIC_TREE_FAST_METADATA=1` | `_set_builder_topology` 的树掩码按拓扑缓存（原先每轮 4~5 次在 GPU 上逐元素构造，每次 2.67 ms；命中缓存 0.0025 ms）；位置偏移小张量同样缓存 |
| `REMTP_DYNAMIC_TREE_FAST_COMMIT=2` | 路径提交丢掉恒等拷贝，按路径顺序原地搬移，不用暂存缓冲；顺序不安全时退回两段式 |

实测（同样的 GSM8K 17/353/856，`results/third_layer_ab_20261111/`）：

| 臂 | 配置 | 输出 token | 轮数 | 每轮耗时 | decode tok/s | 对照 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| D | 第一、二层 + `FAST_COMMIT=1` | 1723 | 513 | 1.2385 s | 2.712 | |
| E | D + 第三层三项 | 1723 | 513 | 1.2155 s | 2.763 | 与 D 3 条逐 token 相同；−23.0 ms/轮（三题 −24.5/−21.6/−23.7） |
| F | E + `PATH_ORDERED_FIRST_PASS=1` | 1184 | 343 | 1.2115 s | 2.849 | 文本不同；每轮 token 3.452 对 3.359 |

臂 E 与臂 D 处理的是完全相同的 513 轮，所以 −23.0 ms 是纯实现开销，同时从实验上证实了被跳过的 forward 和草稿 KV 提交确实不影响任何后续计算。

分段计时（32 轮，GPU stream 时间，ms/轮；`results/third_layer_ab_20261111_profile/`）：

| 顶层组件 | Native | C（第二层后） | E（第三层后） | E 对 Native |
| --- | ---: | ---: | ---: | ---: |
| target_forward | 1011.0 | 1016.1 | 1016.1 | +5 |
| proposal | 103.1 | 161.9 | 144.7 | +42 |
| sampling / verification | 1.3 | 18.6 | 18.4 | +17 |
| target_logits | 21.3 | 21.4 | 21.5 | 0 |
| cache_update | 0.0 | 17.9 | 2.9 | +3 |
| 整轮 | 1139 | 1241 | 1209 | +70 |

proposal 内部：MTP forward 从 4 次降到 3 次（47.6 → 36.6 ms），与链式方法的 3 次相同；LM head 3 次 63.8 ms，也与链式相同。剩下约 36 ms 是每个父节点的候选选择、采样和拓扑构造。

三层合计：每轮 1.3057 s → 1.2155 s。对同上下文的 Cactus（约 1.143 s）还多约 72 ms；GSM8K 的收支平衡点约 69 ms，Spec-Bench 约 122 ms。

臂 F 是读代码时发现的另一件事，不是开销问题：vLLM 的首遍 MTP forward 假设 target 输入行是一条链，树的 BFS 行序让被采样的叶子行注意到被拒绝的兄弟节点，提交后的草稿前缀槽位里留下的也是 BFS 行而不是路径节点。`_path_ordered_first_pass` 把首遍输入重排成"锚点、路径、其余"。它改变提议分布 Q，不改变验收规则。三道题上的每轮 token 差异（+2.8%，343 对 513 轮，文本不同）在噪声范围内，不作结论；全天实验用一个 Spec-Bench 臂对照。

下一步候选（未做）：把一层内所有父节点的候选选择合成一次 top-k 和一次传输（估计 15~20 ms）；验收的 17 ms。

## 11. 全天实验（2026-11-11 启动，`results/day_20261111/`）

`scripts/run_day_20261111.py`，24 小时截止，可续跑，每完成一个阶段就重写 `day_report.md`：① HumanEval-100（冻结协议同题/同种子/2048 预算，Docker 判分），第三层配置的 RAVEN，与 10 月记录的四种方法并列；② Spec-Bench 新抽样（种子 20261111，6 任务各 15 条，512 token），第三层 RAVEN；③ 同抽样 Cactus；④ 同抽样 RAVEN + 首遍路径重排；⑤ 时间允许时 Native。

### 11.1 结果（2026-11-11 14:33 至 2026-11-12 13:10，全部阶段完成，22.6 小时）

实际顺序：Spec-Bench raven_v3 → cactus → raven_v3_pathfirst → HumanEval-100 raven_v3 → Spec-Bench native。完整表格见 `results/day_20261111/day_report.md`。

Spec-Bench（105 次请求/方法，同题同种子，各方法单次运行，文本各不相同）：

| 方法 | MAL | 每轮秒 | decode tok/s | E2E tok/s | E2E 对 Cactus（按题 bootstrap 95% 区间） |
| --- | ---: | ---: | ---: | ---: | --- |
| native | 2.441 | 1.1431 | 2.137 | 2.083 | −9.4% |
| cactus | 2.696 | 1.1431 | 2.360 | 2.300 | — |
| raven_v3 | 2.972 | 1.2237 | 2.430 | 2.368 | +3.0%（+1.5% ~ +4.6%） |
| raven_v3_pathfirst | 3.015 | 1.2229 | 2.467 | 2.401 | +4.4%（+2.7% ~ +6.2%） |

- raven_v3 六个任务 E2E 全部高于 Cactus（translation +10.1%、summarization +1.6%、math_reasoning +2.4%、rag +1.9%、qa +2.2%、multi_turn +2.8%）；通宵实验里落后的 summarization、math_reasoning 这次转为领先。每轮比 Cactus 多 80.6 ms。
- bootstrap 只重抽题号，不包含同一方法重复运行的采样波动；两方法输出长度不同，上下文长度也不同。
- pathfirst 对 raven_v3：MAL +1.4%、E2E +1.4%，每轮耗时相同。单次运行，不能据此说明它改善了提议分布。

HumanEval-100（冻结协议，raven_v3 实跑并 Docker 判分；其余为 10 月同题记录）：

| 方法 | pass@1 | E2E tok/s | MAL | 每轮秒 | 截断 |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 62% | 2.4701 | 2.855 | 1.1483 | 18 |
| cactus | 65% | 2.7573 | 3.185 | 1.1472 | 20 |
| 原 RAVEN（adaptive_original） | 59% | 2.5487 | 3.402 | 1.3260 | 22 |
| raven_v3 | 54% | 2.7336 | 3.377 | 1.2266 | 19 |

- 速度：对原 RAVEN E2E +7.3%、每轮 −99 ms，MAL 基本不变；对 Cactus E2E −0.9%。
- 质量：判分状态 passed 54、failed 21、syntax_error 22、runtime_error 3，未出现判题环境错误。精确 McNemar 配对检验：对原 RAVEN 仅原对 12、仅新对 7，p=0.36；对 Cactus 仅 Cactus 对 22、仅新对 11，p=0.08；对 Native p=0.19。raven_v3 与原 RAVEN 接受规则相同，差异来自 1 ulp 注意力数值差后的不同采样轨迹，100 题无法区分；但 RAVEN 两次 HumanEval（59%、54%）都低于 Cactus（65%），需要多种子重复才能判断是否有系统性质量差。

## 7. 复现

```bash
cd ~/ReMTP-residual && source .venv/bin/activate && source scripts/cuda_env.sh
# 三臂依次运行；每臂起一次服务，重放同样的三道题
python scripts/ab_canonical_tree_attention.py results/<dir> 17,353,856 1,0,0+fast
# 探针：在任意服务环境里加 REMTP_TREE_ATTENTION_PROBE_JSONL=<file>
# 等价性测试
PYTHONPATH=$PWD:$PWD/.venv/lib/python3.12/site-packages /usr/bin/python3 -m pytest -q tests/test_raven_fast_path.py
```

结果目录：`results/canonical_attention_ab_20261110/`（臂 A、B）、`results/canonical_attention_ab_20261110_fast/`（臂 C）；注意力微基准与探针原始记录在会话临时目录，本文已摘录全部数字。
