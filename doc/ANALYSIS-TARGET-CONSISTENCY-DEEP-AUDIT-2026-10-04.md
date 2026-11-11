# Target 概率一致性：第三轮失败复查与解决路径

日期：2026-10-04。本文区分实测结论、源码线索和待验证假设。
正式优化保持停止；本次只运行有限诊断，不修改权重、接受算法或失败阈值。

## 1. 已确定的失败原因

第三轮在北京时间 13:43:53 主动退出，`STOPPED_PROTOCOL_ERROR`、exit=2、无自动重启。
直接触发条件是：同一 token 前缀的实际树 root 概率与独立 Target 重算不一致，超过原门槛 0.15。
已完成 10 条生成诊断、4 条独立评分，后续消融、选参、留出和正式基线比较均未执行。
日志未检出 OOM/CUDA error；现有证据不支持把这次停止归因于内存不足。

| 实际前缀长度 | 已生成 token | 最大相对 logit 差 | EOS 相对差 | 门槛 |
| ---: | ---: | ---: | ---: | --- |
| 109 | 1 | 约 4.61e-7 | 约 4.19e-7 | 通过 |
| 238 | 130 | 约 8.66e-7 | 约 6.86e-7 | 通过 |
| 157 | 1 | 约 8.86e-7 | 约 1.46e-8 | 通过 |
| **287** | **131** | **0.25000074** | **0.04687600** | **失败** |

失败来自 old（D3/head=1）配置、train 2950、seed=29542，request：
`chatcmpl-a32163daa5770da8-b3b0f560`。
前缀 SHA256：`fedd79885fca4a768be8cf7baba88ef7a28cc0c0a47711b76189c159763b33aa`。
这里的 0.25 是乘回 temperature=0.6 后的相对 logit 差，不是正确率下降 25%。
对应某些被检查 token 相对概率比可相差约 `exp(0.25/0.6)=1.52` 倍；
原日志只保存 top8/EOS，无法据此恢复完整词表 TV 或证明质量已下降。

原证据：[停止核查](STATUS-ADAPTIVE-RAVEN-ROUND3-2026-10-04.md)，
`results/adaptive_optimization_20261004_target_canonical/diagnostics/sequential_target/`。

## 2. 为什么前几次“固定案例修好”仍然失败

前几次修复确实解决了已测案例，但我安排的验收覆盖不足，恢复优化过早。

1. **重算相同文本不能复现所有运行状态。** 新鲜 prefill 把整个历史一起计算；
   长生成保留多轮树验证及路径提交形成的 KV。两者 token 相同，计算历史可能不同。
2. **未定位长轨迹首次差异。** 原状态探针只看每请求前 8 次提案，Target raw 探针默认只看前 32 次调用；
   正式控制器没有启用后者。新的失败发生在 generated_length=131，缺少当时的逐层 hidden、
   Target KV 和压紧前后快照，不能直接证明是哪一层或哪次提交出错。
3. **评分样本不足。** 控制器取首次出现的 4 个唯一前缀，第三轮只覆盖前两道 old 题，
   未涵盖另两题、anchor4 轨迹或 512/1024/2048 的状态。记录点仅为 0、1、首次达到 128。
4. **独立树验证配置不完全相同。** 默认固定树检查为 head=3、anchor/topk；失败配置是 old/head=1。
   本次追加 head=1 对照，不能把 head=3 的结果直接解释成 old 长生成已经修复。
5. **比较器还有盲区。** HTTP top20 可能缺失 EOS；当前代码允许 EOS 无值，并只要求部分 top token 匹配。
   新方案要求原始完整 logits、显式 EOS/实际草稿 token 和严格 token/position 对齐。
6. **同 seed 输出重复不等于跨执行方式概率一致。** 7311 的两次 1250-token 重放完全一致，
   只能说明该配置和请求可重复，不能覆盖树与顺序 Target 的条件概率一致性。

改进原则：后续必须先获得长生成中的首次差异证据，再选择修复；不能用几个新鲜前缀通过
或增加容差代替长状态验收。历史失败和冻结代码继续保留。

## 3. 本次新增的实机对照

采用第三轮**真实** manifest（109/238/157/287），逐 token hash 校验；不沿用旧的 236/286 前缀。
关键 9 个执行文件与第三轮冻结代码逐字节一致。公共配置保持 FP4/row64、BF16 原 flag、
TRITON 2D、FP32 P×V、canonical 祖先读序、跨请求 prefix caching=0。
关闭跨请求缓存仍保留每次生成所需的普通 KV 缓存。

### 原生 Target：24 条请求，已完成

四个前缀分别做完整 echo 评分、新鲜单行生成、强制最后一 token 后真实 decode，各两次。
强制 token 仅构造诊断前缀，raw logits 在采样约束前采集，不计入质量或性能实验。

| 前缀 | echo/decode hidden 最大差 | 检查 top token 相对差 | 完整词表 TV | 原长生成树/新 echo 相对差 |
| ---: | ---: | ---: | ---: | ---: |
| 109 | 0 | 0 | 约 1.13e-12 | 约 1.91e-6 |
| 238 | 0 | 0 | 约 7.37e-13 | 约 7.15e-7 |
| 157 | 0 | 0 | 约 9.74e-11 | 约 4.77e-7 |
| **287** | **0** | **0** | **约 1.40e-13** | **约 0.250001** |

同 hidden 的输出投影 M=1/10/13，检查 top token 的相对差均为 0。
每种请求两次重复的 raw logits 均逐位相同；single/decode 与 echo 的 hidden 差均为 0。
完整词表中少数 logits 仍有 0.03125～0.0625 的差，完整 TV 如表；不能宣称所有 logits 逐位相同。

### 新鲜 D3/D4 树：16 条请求与 16 条祖先评分，已完成

24 项 root 检查全部通过，top/EOS 相对差均 0，完整 TV 最大 `1.133e-12`。
16 条主/备选祖先路径全部通过，top/EOS 相对差均 0，完整 TV 最大 `7.985e-12`。
核对实际 root token、逻辑 position、实际输入 draft IDs；改变备选 seed 未改变 root logits。
这些是新鲜 head=3/anchor/topk 计算的结果，**不是持续生成 131 token 后的缓存验收**。

### 原失败 old/head=1：追加 4 条请求，已完成

157/287 两个精确前缀，各用 seed=29542（原失败 seed）和 29543 构造新鲜树。
head=1、D3、非anchor、非topk、原松弛/自适应参数均与 old 配置一致。
4 项 raw root 检查全通过，实际 root token/position 对齐，hidden/top/EOS 差均 0；
287 完整 TV 为 `1.394e-13`。证据：`fresh_old/fresh_old_gate.json`。

截至北京时间 17:57:51，本次 **60 条对照请求**已全部结束，两项有限 user unit 均 exit=0。
这些结果进一步支持把排查重点转向实际持续生成的状态/计算历史及其记录对齐。
仍然没有复现原 131-token 累积 KV、原时刻完整树拓扑和 RNG 状态；不能据此断言具体 KV 复制 bug。

证据根目录：`results/target_consistency_deep_audit_20261004/`。
原生结果：`native_replay/native_cache_0/numerical_checks.json`、`native_all_modes_audit.json`；
新鲜树结果：`fresh_tree/target_consistency_gate.json`。启动 warmup 不计入上述对照请求数。

## 4. 资料调研与本机适用范围

### 4.1 重复性和批次/切分不变性不同

PyTorch 2.10 明确不保证数学等价的整批计算与逐片计算逐位相同；
控制随机 seed 无法消除由执行形状引起的归约差异。
[PyTorch 2.10 数值精度说明](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)。

Thinking Machines 的原始研究将 RMSNorm、矩阵乘法、attention 作为需要逐一保持归约顺序的算子；
attention 还需对 prefill、decode 和 KV 分块方式保持一致。统一一个后端名称不够，
必须测其实际 launch、tile、读序与切分行为。这支持下一步做算子和历史状态对照，
并不证明本项目已找到具体根因。
[原始研究](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)。

### 4.2 不能直接开启 vLLM 全局不变模式作为本机修复

本机为 vLLM 0.18.0、Orin sm87。**该版本**官方文档要求 compute capability≥9.0，
已验证模型列表不包含当前 MiMo/FastMTP；不能用较新文档的硬件支持范围替代 0.18 的验证。
[vLLM 0.18.0 Batch Invariance](https://docs.vllm.ai/en/v0.18.0/features/batch_invariance/)。

该版本源码的支持后端列表不包含 TREE_ATTN，开启全局模式会遇到后端校验。
源码在 torch≥2.10 使用 `(False, False)` 关闭 BF16 低精度归约及 split-K，并调整其他算子。
本机隔离进程实际读取结果如下，未修改正在运行的 worker：

| 设置 | reduced precision | split-K |
| --- | --- | --- |
| 当前 `False` | False | **True** |
| 候选 `(False, False)` | False | False |

这是**已确认的配置差异**，但新鲜失败前缀当前已经通过，尚无证据证明关闭 split-K 能修复持续生成。
应对首次差异层的真实输入做单因素 A/B，再决定是否采用；不绕过 TREE 后端校验宣布官方支持。
[vLLM 0.18.0 原始实现](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/model_executor/layers/batch_invariant.py)。

### 4.3 数值一致性与方法近似分别验收

精确 speculative decoding 的分布保证要求接受与残差使用一致的 Target 条件概率。
Adaptive RAVEN 的松弛本身允许近似，不能直接套用精确采样的“质量无损”结论；
公共 Target 的数值一致性应先修复，方法质量再用独立题集检验。
[原始论文](https://arxiv.org/abs/2211.17192)。

## 5. 待验证原因及对应实验

| 假设 | 当前证据 | 下一项能区分它的检查 |
| --- | --- | --- |
| 多轮生成形成的 KV 与完整重算数值不同 | 新鲜原生/D3/D4一致，历史 287 仍差 0.25 | 原失败 seed 自然重放到 131，保存全部 Target 层逻辑 KV 与 root hidden；与同 token 完整重算比较 |
| Target 路径压紧选错槽、有效长度或兄弟污染 | 有多轮路径提交，但没有失败时的 Target 提交快照 | 每次接受前后核对 selected path、源/目的槽、36 层复制字节和有效长度；拒绝槽扰动须不影响可见结果 |
| 记录的前缀/位置与实际 root 输入错配 | manifest hash 已验；长生成当时没有 raw input/position 快照 | 同时保存 request history、实际 input IDs/positions、scheduler/block/slot；任何错位先修采集/状态 |
| 某个归约算子随 shape/切分变化 | 已确认 split-K 未关闭；prefill 多 query 与 canonical 单 query 仍不同 | 定位首次 hidden 差所在层；依次比较 norm、QKV/RoPE、attention、output/MLP、LM head 的真实输入/输出 |
| 草稿配置/拓扑改变暴露 Target 行布局问题 | 默认新鲜树与 old/head1 不同 | 本次追加 old/head1 新鲜 root；之后保存失败时完整拓扑和实际节点输入重放 |
| 阈值/概率记录误判 | 0.25 多次重算一致，普通归一化常数已通过相对差消除 | 对同一现场同时保存 raw logits 与约束后 P，显式检查温度/top-p/EOS，不依赖 HTTP top20 缺项 |

不能仅凭独立重算 KV 不同就断言“拷贝错误”：不同算术可能已经生成不同 KV。
须先验证提交时目的槽与其选中的源槽逐位一致，再追溯源 KV 的生成计算。
最优先目标是得到“首次哪个 token、哪层、提交前或后发生差异”的证据。

## 6. 质量与速度结果的正确解释

第三轮 old/anchor4 各四题，原解析判对均为 4/4，但正常结束且判对为 2/4→3/4，截断为 2→1。
吞吐 2.569→2.932 tok/s（+14.1%），平均时延 567.3→293.8 秒（-48.2%）。
这些只是未通过 Target gate 的训练诊断，不能作为 Adaptive 对 Native/Cactus/SpecCascade 的论文收益。
2950 还从 483 token 正常结束退化到 2048 token 截断，问题未解决。

下一轮顺序：**现场定位 → 最小修复 → 累积状态验收 → 截断/质量校准 → 三基线公平测速**。
具体步骤及每步预期产物见
[下一步审慎优化计划](../plan/target-consistency-deep-repair-2026-10-04.md)。

## 7. 本次修改与限制

- 仅增加诊断 CLI 的显式 manifest/hash 校验及针对性测试，不改变正式算法。
- 本次相关 CPU 测试实际 7 passed（3.12 秒），运行使用现有系统 pytest 与项目依赖路径。
  `.venv` 本身未安装 pytest；两次初始调用因测试入口/导入路径失败，修正调用方式后通过，未安装或升级软件。
- 正式 round3 不重启；有限诊断由独立用户级 systemd unit 执行并收尾。
- 历史 379 passed/2 skipped 属于上一版完整回归，不能当作本次长生成已通过的证明。
- 尚未完成实际长历史的逐层 Target KV/提交定位，也未运行三个正式基线的新比较。
