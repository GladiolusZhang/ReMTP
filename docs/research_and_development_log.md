# ReMTP 研究与开发总日志

最后更新：2026-08-17
当前分支：`research/residual-aligned-tree-mtp`

本文档是仓库的唯一持续维护总索引，记录已经做过的实现、实验、负结果、协议
变更和当前有效入口。算法细节仍可保存在各专题文档中，但任何后续修改都必须
同步追加到本文档。

## 1. 维护规则

状态含义：

- `已验证`：代码测试和相应实验均已完成；
- `代码已验证`：单元测试或静态检查通过，但尚未完成真实 GPU 实验；
- `实验性`：可以运行，但算法近似、统计结论或系统正确性仍需验证；
- `负结果`：实现或实验完成，但没有达到预设门槛；
- `已冻结`：保留现有定义和结果，不再继续调参；
- `已废弃`：存在已知错误，不得进入正式表格；
- `历史结果`：只用于追踪研究过程，不能自动与当前协议混合。

后续每条记录至少写明：日期、状态、动机、修改、文件、验证、结果和限制。

实验原始数据和报告默认只保存在本地 `results/`、`logs/`、`checkpoints/`，不因
代码推送自动上传。总日志可以记录关键指标和本地结果路径。

## 2. 当前有效结论

1. Qwen3.5-4B 原生 MTP、完整概率提议、Cactus、SpecCascade、Exact-TV、
   ReMTP、HumanEval/GSM8K 评测链路均已实现。
2. Qwen3.5 链式 ReMTP 已冻结；继续统一放大 TV 或 margin 的边际收益很低，
   且容易沿 Cactus 的质量下降方向移动。
3. Qwen3.5 Fixed-6 树是负结果：分支树 MAL 下降，正确性参考 target forward
   延迟相对链增加 168.4%，因此停止 `tree + ReMTP`。
4. MiMo-7B 的额外 MTP layer 1/2 为 pretrained-only，深层 P/Q 重叠很低；
   这不是继续降低树验证阈值能够解决的问题。
5. FastMTP 包含一个训练好的物理 MTP head，并通过共享权重递归生成三个草稿
   位置；它不是三个独立训练 head。
6. 2026-08-09 前由旧 FastMTP 四方法脚本产生的 Cactus/SpecCascade 结果无效，
   因为 Worker 被覆盖成 Native。
7. 当前 FastMTP 有效入口是
   `scripts/run_fastmtp_verified_comparison.sh`；动态树仍是改变输出分布的近似
   方法，不能标成 lossless speculative decoding。

## 3. 研究阶段与操作记录

### 3.1 最小 Qwen3.5 MTP 与逐轮可观察机制

状态：`已验证`

完成内容：

- 固定 vLLM 0.18.0、Qwen3.5-4B、本地 checkpoint 和 RTX 4090 单卡配置；
- 实现 `MTP draft / TARGET verify / VERIFY / COMMIT` 四阶段日志；
- 支持 greedy 和 temperature sampling；
- 解释并记录 draft token、target token、bonus、recover、接受随机数和最长前缀；
- 增加本地 checkpoint 完整性检查、模型下载和请求脚本；
- 为 Qwen3.5 GDN/CUDA Graph 启动错误提供 eager/no-CUDAGraph 路径。

主要文件：

- `remtp/trace.py`、`remtp/worker.py`；
- `scripts/serve.sh`、`scripts/request.sh`、`scripts/download_model.sh`；
- `README.md`。

关键认识：MAL 定义为 `1 + accepted_draft_tokens / draft_rounds`，其中 1 是每轮
目标模型产生的 correction 或 bonus。因此 MTP=2 时 MAL 上限是 3，出现 2.3
并不代表接受了超过两个草稿 token。

### 3.2 完整 MTP 分布与标准概率推测解码

状态：`已验证`

完成内容：

- 将 vLLM 默认 MTP argmax 路径改为从完整 `Q` 采样；
- proposal、接受概率和残差纠正统一使用同一份 `Q`；
- 保留 greedy 为 one-hot proposal；
- 增加 P/Q、随机数、接受概率、recover/bonus 的诊断；
- 明确随机种子只用于复现实验采样，不是推测解码额外引入的理论参数。

标准随机验证：

```text
A(y) = min(1, P(y) / Q(y))
residual(x) proportional to max(P(x) - Q(x), 0)
```

主要文件：`remtp/probabilistic_mtp.py`、概率 Worker 和相应 benchmark/trace 脚本。

### 3.3 Spec-Bench 基准设施

状态：`已验证`

完成内容：

- 支持 translation、summarization、math_reasoning、rag 固定抽样；
- 区分 decode tok/s、E2E tok/s 和 MAL；
- 将 trace 与正式吞吐测量分离；
- 处理 Qwen3.5 GDN 与 CUDA Graph capture 冲突。

早期 translation-20 示例结果：decode 151.186 tok/s、E2E 121.096 tok/s、
MAL 2.357，MTP=2。

### 3.4 论文复现：SpecCascade 与 Cactus

状态：`Qwen 路径已验证；FastMTP 旧入口已废弃并重建`

完成内容：

- 复现 *Faster Cascades via Speculative Decoding* 的 TokenV3；
- 复现 Cactus candidate-specific 临时验证分布；
- 两者均使用完整概率 MTP 作为 proposal；
- 实现统一 residual sampling、诊断和报告；
- 之后所有主要 benchmark 均保留 Native、Cactus、SpecCascade 作为基线。

Cactus 当前实现：

```text
H(y) = min(P(y) + sqrt(2 delta P(y)(1-P(y))), 1)
A(y) = min(1, H(y)/Q(y))
recovery proportional to max(H-Q, 0)
```

同一前缀、候选和随机数下 `H(y)>=P(y)`，所以 Cactus 局部接受概率不会低于
Native；完整生成轨迹和任务质量仍会因临时分布改变而不同。

专题文档：`docs/cactus_mtp.md`、`docs/speculative_cascade_mtp.md`。

### 3.5 GSM8K 与 HumanEval 质量评测

状态：`已验证`

完成内容：

- GSM8K 固定抽样、答案解析、准确率和格式合规率；
- HumanEval 固定抽样、代码提取、受限 Docker 执行、pass@1、超时和语法错误；
- 同一 prompt manifest、temperature、seed、max_tokens 和 system prompt 协议检查；
- 支持历史基线复用，协议不匹配时拒绝合表；
- 后续比较脚本默认同时报告任务质量、MAL、接受率、decode 和 E2E。

主要文件：

- `remtp/gsm8k_benchmark.py`；
- `remtp/humaneval_benchmark.py`、`remtp/humaneval_evaluator.py`；
- `docs/humaneval.md`。

### 3.6 Gated-depth-KL、Exact-TV 与块级预算探索

状态：`旧 gated-depth-KL 已废弃；Exact-TV 为重要历史方向`

依次尝试：

1. rank、目标概率、margin、深度衰减和 KL 二分的 gated-depth-KL；
2. Cactus 概率提升上限 `H(y)<=Q(y)`；
3. Cactus 总 TV 的饱和感知回收；
4. 按前缀价值、目标支持和静态 head reliability 重新分配；
5. future target support 作为 veto；
6. P/Q JS、top-k overlap、hidden cosine 等块特征消融；
7. Exact-TV + head calibration；
8. Exact-TV + head/hidden + future veto。

MTP=6、GSM8K-200 的代表结果：

| 方法 | accuracy | E2E tok/s | MAL | draft acceptance |
|---|---:|---:|---:|---:|
| Cactus + MTP | 80.0% | 180.418 | 4.890 | 64.8% |
| SpecCascade TokenV3 | 90.0% | 146.331 | 3.999 | 50.0% |
| Cactus + `H(y)<=Q(y)` | 79.5% | 177.800 | 4.826 | 63.8% |
| Exact-TV + head calibration | 74.5% | 193.414 | 5.281 | 71.3% |
| Exact-TV + head/hidden + future veto | 73.0% | 194.662 | 5.322 | 72.0% |

结论：Exact-TV 显著提高 MAL/吞吐，但任务质量下降。Future JS/hidden 一致并不
等价于语义正确；“给定错误前缀后继续一致”不能作为前部 token 的正向证据。

### 3.7 遗憾负反馈探索

状态：`跨块 hidden steering 为负结果；当前块风险控制保留为研究记录`

尝试过的定义和作用位置：

- 只记录 strict reject、relaxed accept 的 causal relaxed acceptance；
- 使用 `P->H` 被转移的目标概率质量构造 regret direction；
- 在下一块 MTP Module 1 hidden 副本注入方向；
- posterior scale、top-1 bias 等低成本弱反馈；
- 将遗憾从事后方向 steering 改为当前块的 verification debt；
- debt 约束位置预算、累计预算、future relaxation 和 target recovery。

代表结果：

| 方法 | accuracy | E2E tok/s | MAL | draft acceptance |
|---|---:|---:|---:|---:|
| Exact-TV + head/hidden + future veto | 73.0% | 194.662 | 5.322 | 72.0% |
| Exact-TV + regret feedback | 75.0% | 182.832 | 5.293 | 71.5% |

Regret audit：strict-acceptable ratio 85.13%，causal relaxed acceptances 4224，
TV/accepted draft 0.074986，feedback injections 2067。

反思：遗憾方向来自旧位置的替代 token 偏好，不能直接搬到下一位置；反馈发生在
风险 token 已提交之后，因果位置过晚；额外 hidden/logit 处理降低吞吐。

分支保留：`research/future-only-regret-exploration`。

### 3.8 当前块 verification debt 与 target-anchored 控制

状态：`负结果/未通过联合 Pareto 条件`

方法使用：

```text
debt_i = A_relaxed_i - A_strict_i
```

在提交前限制高 debt、极端 target gap 和累计块风险，并尝试 balanced debt、
Cactus-dominant target surplus、risk swap、target recovery 和 block shield。

GSM8K-100 代表结果：

| 方法 | accuracy | E2E tok/s | MAL | draft acceptance |
|---|---:|---:|---:|---:|
| Cactus | 75.0% | 182.470 | 4.957 | 66.0% |
| Native | 82.0% | 141.004 | 3.781 | 46.3% |
| Current-block debt | 86.0% | 153.987 | 4.186 | 53.1% |
| Cactus-dominant target surplus | 75.0% | 182.919 | 4.965 | 66.1% |

没有方法同时满足 accuracy>Cactus、MAL>Cactus、E2E>=Cactus。之后不再使用
针对单数据集的自动 fallback 作为正式方法定义。

专题文档：`docs/current_block_verification_debt.md`。

### 3.9 Expected-Regret Router

状态：`训练与实验完成，负结果`

完成内容：

- 构建 UltraChat、No Robots、Magicoder、OASST1 混合无标签 prompt 语料；
- benchmark 去污染和按 request 切分；
- 收集 P/Q/H、debt、target margin/entropy、hidden/direction 等块 trace；
- 训练冻结 target/MTP 之外的小型低秩 Router；
- Router 输出 head 引导强度、proposal scale 和 Exact-TV budget scale；
- 重新设计训练目标、early stopping、审计集和 identity fallback。

Pilot 收集 3987 个请求。第一版 10 epoch validation loss 基本保持 1.584，
budget scale 饱和到 1；V2 在 epoch 7 early stop，best epoch 1，
validation model gain 0.000968、audit gain 0.001024、sign accuracy 0.6102。

HumanEval-164 代表结果：

| 方法 | pass@1 | E2E tok/s | MAL |
|---|---:|---:|---:|
| Native | 79.9% | 209.741 | 5.622 |
| Cactus | 65.9% | 222.051 | 6.022 |
| Exact-TV target-only future veto | 60.4% | 225.114 | 6.121 |
| Learned expected-regret Router | 59.8% | 215.688 | 6.132 |

结论：Router 没有带来可靠质量改善，同时增加推理开销。专题文档：
`docs/exact_tv_regret_router.md`、`docs/regret_router_data.md`。

### 3.10 方案一、方案二、方案三及融合

状态：`已完成比较；最终收敛到 margin-calibrated prefix relaxation`

- 方案一：累计风险预算的边际—熵联合验证；
- 方案二：同轮前视哨兵、target anchor 和跨轮 risk debt；
- 方案三：熵感知动态链长，作为树方向的低成本替代；
- 方案一+二：联合风险与前视控制；
- 后续增加 stronger/ultra/high-relaxation 和 joint risk-credit 版本。

HumanEval-164 代表结果：

| 方法 | pass@1 | E2E tok/s | MAL |
|---|---:|---:|---:|
| Native | 79.9% | 210.753 | 5.622 |
| Cactus | 65.9% | 222.471 | 6.022 |
| SpecCascade | 80.5% | 214.770 | 5.827 |
| Scheme 1 | 81.7% | 201.382 | 5.675 |
| Scheme 2 | 80.5% | 207.702 | 5.647 |
| Scheme 3 | 80.5% | 208.052 | 5.494 |
| Scheme 1+2 | 80.5% | 207.021 | 5.648 |
| Scheme 2 high-relaxation | 78.7% | 211.943 | 5.805 |
| Joint Scheme 1+2 risk-credit | 78.0% | 209.182 | 5.824 |
| 冻结 ReMTP margin-calibrated prefix | 81.1% | 210.581 | 5.751 |

冻结 ReMTP 在该协议下保持较高质量，但只释放了 Native→Cactus MAL 空间的约
三分之一，继续统一放宽会明显损失质量。

### 3.11 Target-band、prefix credit 与决策 trace

状态：`完成探索；未超过冻结 ReMTP`

Target-band 只在 MTP token 位于目标 top-1 附近且 margin 条件允许时提升候选，
并用 block TV 控制总偏移。随后实现 Prefix-Credit，直接优化整个有序前缀的生存
概率，并将后部位置的收益按前缀可达率计权。

HumanEval Prefix-Credit 结果约为 pass@1 76.2%、MAL 5.726–5.735、E2E
207.4–208.6 tok/s，没有超过 target-top1 基线或冻结 ReMTP。

增加了小样本 decision trace，可查看每题候选、P/Q、TV、strict/relaxed-only、
拒绝位置和最终输出。专题文档：`docs/decision_trace_and_tuning.md`、
`docs/prefix_credit_mtp.md`。

### 3.12 Proposal-Calibrated MTP

状态：`已验证；严格分布一致，但整体收益有限`

方法只变换 proposal：

```text
Q_tilde_i = softmax(logits_i / (T_request * T_head_i))
```

候选采样、接受概率和残差纠正统一使用 `Q_tilde`，target P 不变，因此仍严格
恢复目标分布。离线目标为：

```text
E[MAL] = 1 + a1 + a1*a2 + ... + a1*...*aK
```

正式测试 MTP=2/4/6/8，比较 Native、Cactus、SpecCascade 和 proposal
calibration，共 32 行完成。代表结果：

- GSM8K MTP=6：Native MAL 3.811；Cactus 4.894；SpecCascade 4.010；
  Proposal-Calibrated 3.888。
- HumanEval MTP=6：Native pass@1 79.9%、MAL 5.622；SpecCascade 80.5%、
  MAL 5.827；Proposal-Calibrated 76.8%、MAL 5.830。
- GSM8K MTP=8 的 Proposal-Calibrated accuracy 90.6%、MAL 4.104，但跨任务
  和深度并不稳定。

本地完整表：`results/proposal_calibration_depths_20260805_143940/comparison.md`。
专题文档：`docs/proposal_calibrated_mtp.md`。

### 3.13 Fixed-6 Micro-Tree（Qwen3.5）

状态：`负结果，已停止`

完成内容：

- 实现 `6-chain`、`4+2`、`3+2+1` 三个固定六节点拓扑；
- 正确维护 BFS parent、position、TreeAttention mask、attention KV、Qwen GDN
  conv/SSM 和 MTP 递归状态；
- 实现严格概率树验证、分支隔离测试、selected-path compaction 和 GPU audit；
- 先做 unit/reference/GPU smoke，再做 8 条 GSM8K pilot；
- 未实现 tree+ReMTP，未跑正式 GSM8K/HumanEval。

结果：

| 拓扑 | MAL | E2E tok/s | MAL vs chain |
|---|---:|---:|---:|
| 6-chain | 3.764 | 102.094 | 0 |
| 4+2 | 2.725 | 35.915 | -1.039 |
| 3+2+1 | 2.494 | 32.916 | -1.270 |

4+2 correctness reference target forward 为 39.392 ms，6-chain 为 14.675 ms，
增加 168.4%。树覆盖和系统开销均未过门槛。专题文档：
`docs/fixed6_microtree.md`。

### 3.14 MiMo-7B、MTP3 overlay 与动态树

状态：`基础设施已实现；当前模型结果较差`

完成内容：

- 下载 MiMo-7B-Base、MiMo MTP add-on 和 MiMo-7B-RL-0530；
- 构造本地 MTP3 overlay 和逻辑 tree view；
- 实现 physical 0-1-2 与 repeated 0-0-0 路由；
- 修复额外物理 MTP layer 的真实前缀 KV prefill；
- 实现动态熵阈值构树、TreeAttention、target-dominant relaxed verification、
  EOS protection、路径日志和 tree-native MAL；
- 在 GSM8K/HumanEval 比较 Native、Cactus、SpecCascade 和动态树。

MiMo correctness-100：

| 数据集 | Native quality/MAL | Cactus quality/MAL | Tree quality/MAL |
|---|---|---|---|
| GSM8K | 43.0% / 2.007 | 35.0% / 2.253 | 39.0% / 2.057 |
| HumanEval | 29.0% / 2.060 | 8.0% / 2.151 | 23.0% / 2.071 |

诊断显示根层平均分支约 1.01，深度 2/3 的 target coverage 接近零； lowering
`tau_relax` 只有到 `1e-12` 量级才显著提高 oracle MAL。根因是 layer 1/2
pretrained-only 的 P/Q 错配，而不是单纯验证过严。

MiMo-RL-0530 还修正了空 system、temperature=0.6、max_tokens=2048 和 EOS
保护，并比较 physical 012 与 repeated 000。专题文档：
`docs/mimo_mtp_tree.md`、`docs/mimo_dynamic_mtp_tree.md`、
`docs/mimo_dynamic_tree_diagnosis.md`。

### 3.15 FastMTP 初次集成（Claude Code）

日期：2026-08-09
状态：`部分代码保留，旧实验入口已废弃`

初次增加：

- `scripts/download_fastmtp.sh`；
- Native/Cactus/SpecCascade/Dynamic Tree 服务脚本；
- `remtp/fastmtp_worker.py`、`remtp/fastmtp_dynamic_tree_worker.py`；
- 四方法 runner/report；
- 多层 MTP 数据收集和训练方案文档；
- FastMTP 使用、架构和训练计划文档。

Claude Code 的交接摘要还声称新增了
`scripts/collect_mtp_training_data.py`、`scripts/train_multi_layer_mtp.py` 和
`scripts/prepare_and_train_6layer_mtp.sh`，但 2026-08-09 工作树核查时这三个
文件均不存在。因此“6-layer MTP 训练代码已准备”属于无文件支撑的旧声明，
当前状态只能记为方案，不能运行或引用为已实现。

官方 checkpoint 审计确认：

- architecture 为 `MiMoForCausalLM`；
- `num_nextn_predict_layers=1`；
- `num_speculative_steps=3`；
- 权重只包含 `model.mtp_layers.0`；
- 正确语义是一个训练好的物理 head 重复三次；
- 官方 2.03x 包含 SGLang/EAGLE 和 language-aware vocabulary compression，
  不能直接等同于本仓库 full-vocab vLLM 的绝对 tok/s。

发现的错误：

1. `serve_fastmtp_native.sh` 无条件重写 `WORKER_CLS`；
2. Cactus 和 SpecCascade wrapper 没有成功激活各自 Worker；
3. SpecCascade runner 使用错误的环境变量前缀；
4. Dynamic Tree 没有 `TREE_ATTN`、逻辑 node view 或正确 tree config；
5. Dynamic Tree Worker 安装了不兼容的 probabilistic chain adapter；
6. 报告读取不存在的 summary/tree metric key；
7. 旧实验 temperature=0.7 且未使用显式空 system，与 MiMo-RL 可信协议不同；
8. 文档将未完成 GPU 验证的训练和树代码描述为“可直接运行”，结论过度。

旧运行 `results/fastmtp_four_way_50_20260809` 中 Native、Cactus、SpecCascade
的 GSM8K/HumanEval 数值完全一致；日志均显示
`FastMTPProbabilisticWorker`。这些 Cactus/SpecCascade 行不得引用。

### 3.16 FastMTP 已校验重构

日期：2026-08-09
状态：`代码已验证，待端口/GPU 空闲后跑 pilot`

完成修改：

- 新增 FastMTP checkpoint 校验和单物理 head/多逻辑节点 view；
- 新增独立 verified Worker，防止旧脚本覆盖；
- Native、Cactus、SpecCascade 统一先安装 repeated-layer0 和完整 Q；
- runner 强制检查 Worker marker 和真实 adapter diagnostic marker；
- 增加 Target-only 本机质量/速度控制；
- 固定 temperature=0.6、显式空 system、max_tokens=2048；
- Dynamic Tree 使用 `TREE_ATTN`、logical tree view、repeat-0 和真实节点 audit；
- 将 candidate coverage 从正向乘法奖励改为 parent-level `coverage_gate`，避免
  候选越多时所有兄弟 token 自动获得更高松弛分数；
- 报告检查 prompt manifest、协议、真实字段名和 tree-native metrics；
- 明确动态树是 approximate target-dominant path selection，不是 lossless。

新增文件：

- `remtp/fastmtp_checkpoint.py`；
- `remtp/fastmtp_verified_worker.py`；
- `remtp/fastmtp_verified_report.py`；
- `scripts/serve_fastmtp_verified.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `tests/test_fastmtp_checkpoint.py`；
- `tests/test_fastmtp_verified_report.py`。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `tests/test_dynamic_mtp_tree.py`；
- `tests/test_cactus_mtp.py`。

验证：

```text
checkpoint check: FastMTP logical width=1, physical layers=1, layer IDs=[0]
shell syntax: passed
Python compile: passed
unit/reference tests: 33 passed
GPU smoke: not run, because an old user experiment still occupied ~22 GB/96% GPU
```

当前运行入口：

```bash
SAMPLES=10 RUN_TAG=fastmtp_verified_pilot \
  ./scripts/run_fastmtp_verified_comparison.sh
```

该脚本包含 Target-only 控制和四个正式方法：Native、Cactus、SpecCascade、
Dynamic Tree。输出写入 `results/<RUN_TAG>/comparison.md`。

## 4. 当前方法与代码索引

| 方向 | 核心文件 | 当前状态 |
|---|---|---|
| Qwen MTP trace | `remtp/trace.py`, `scripts/serve.sh` | 已验证 |
| 完整概率 MTP | `remtp/probabilistic_mtp.py` | 已验证 |
| Cactus | `remtp/cactus_mtp.py` | Qwen 已验证；FastMTP 新入口待 GPU |
| SpecCascade | `remtp/speculative_cascade.py` | Qwen 已验证；FastMTP 新入口待 GPU |
| Exact-TV/block | `remtp/target_anchored_mtp.py` | 历史探索 |
| Regret Router | `remtp/regret_router.py` 及训练脚本 | 负结果 |
| ReMTP frozen chain | `remtp/risk_entropy_mtp.py` | 已冻结 |
| Prefix Credit | `remtp/prefix_credit_mtp.py` | 负结果/消融 |
| Proposal Calibration | `remtp/proposal_calibration.py` | 已验证、收益有限 |
| Fixed-6 tree | `remtp/fixed6_vllm.py` | 负结果、停止 |
| MiMo dynamic tree | `remtp/dynamic_tree_vllm.py` | 实验性 |
| FastMTP verified | `remtp/fastmtp_verified_worker.py` | 代码已验证 |

## 5. 已知无效或不得混用的结果

- `results/fastmtp_four_way_50_20260809*` 中旧 Cactus/SpecCascade 行：Worker
  实际为 Native；
- MiMo Base/physical 012 的深层结果：额外 head 不是后训练质量基线；
- trace 模式的 tok/s：包含 GPU/CPU 同步和 JSONL 输出，不得作为吞吐；
- Fixed-6 correctness reference 的速度：逐节点状态复制，只能作为负结果和
  kernel 上限诊断；
- 不同 system prompt、temperature、max_tokens、manifest 或代码抽取协议的结果；
- HumanEval generation-only summary 在 Docker evaluation 完成前的质量字段。

## 6. 后续日志模板

以后每次修改在本文末尾追加：

```markdown
### YYYY-MM-DD：修改标题

状态：`代码已验证 / 已验证 / 实验性 / 负结果 / 已废弃`

动机：

- 为什么修改。

修改：

- 算法或代码发生了什么变化。

文件：

- `path/to/file`。

验证：

- 实际运行的测试和结果；未运行 GPU 测试时必须明确写出。

限制与下一步：

- 尚未解决的问题。
```

## 7. 文档维护记录

### 2026-08-09：建立持续研究总日志

状态：`已验证`

修改：

- 汇总从最小 Qwen MTP、完整 Q、论文复现、Exact-TV、遗憾、Router、
  Proposal Calibration、Fixed-6、MiMo 到 FastMTP 审计的全部主线；
- 增加 `AGENTS.md` 和 `CLAUDE.md`，要求 Codex/Claude Code 后续每次修改同步
  更新本日志；
- 在 README 和过期 FastMTP 文档增加总日志入口或历史警告。
- 核对 Claude Code 所称多层 MTP 训练文件，确认文件不存在，并将旧声明降级为
  未实现的训练方案。

验证：文档链接和 Markdown 结构已检查；没有启动新的 GPU 实验。

### 2026-08-09：FastMTP verified pilot 慢速诊断

状态：`已验证（pilot 运行中的局部结果）`

现象：

- `run_fastmtp_verified_comparison.sh` 的第一组是不使用推测解码的
  Target-only 控制组，不是 Native FastMTP；
- Target-only 在 RTX 4090 上稳定约 `60 tok/s`，运行期间 GPU
  持续工作，没有服务停滞；
- GSM8K 10 条共生成 `10,197` tokens，平均 `1,019.7`，生成用时
  `169.85 s`；
- HumanEval 10 条共生成 `17,371` tokens，平均 `1,737.1`，
  生成用时 `290.61 s`，7/10 达到 `max_tokens=2048`；
- HumanEval Docker 评测仅用约 2 秒，不是主要延迟来源；
- 每切换一种方法都会重启服务；当次 Target/Native 冷启动分别约
  31 秒，这是次要固定成本。

根因：

1. FastMTP/MiMo RL 会生成很长的推理和重复文本。请求中虽传入
   `chat_template_kwargs={"enable_thinking": false}`，但 FastMTP 的
   `chat_template.jinja` 不读取该参数。本地 tokenizer 渲染测试证明，
   传入与不传入 `enable_thinking=False` 得到完全相同的 prompt；
2. 比较脚本将 GSM8K/HumanEval 的 `max_tokens` 都设为 2048，且
   没有任务级 early-stop，放大了模型冗长输出的延迟；
3. 整个实验串行执行 `target/native/cactus/spec_cascade/dynamic_tree`，
   每种方法又串行跑 GSM8K 和 HumanEval；
4. 当前 vLLM 入口固定 `--enforce-eager --no-async-scheduling
   --max-num-seqs 1`，用于可审计、可调试的 Worker 补丁，关闭了 CUDA Graph。
   FastMTP 官方基准使用 SGLang EAGLE、CUDA Graph 和高频词表压缩，
   不能将当前 vLLM eager 路径直接等同于官方优化路径。

对照证据：

- 同一 pilot 的 Native FastMTP GSM8K 已完成：`128.778 tok/s`，
  MAL `3.083`，draft acceptance `69.4%`；
- Native 相对 Target-only 的 decode 加速是 `2.145×`，说明训练好的
  MTP head 及主要接线正在正常工作。

结论：

- 当前“慢”不是 Docker 卡住，也不是 Native FastMTP 核心失效；
- 主因是 Target-only 基线本来只有约 60 tok/s，再叠加平均上千
  token 的冗长生成；
- 次因是串行运行五种方法、每种重启模型，以及 eager 调试配置；
- 本次只完成诊断和记录，没有中断用户正在运行的实验，也没有
  修改基准协议或代码。

### 2026-08-09：FastMTP 可选 non-thinking chat template

状态：`代码已验证；GPU 生成待验证`

动机：

- FastMTP checkpoint 携带的旧 `chat_template.jinja` 忽略
  `enable_thinking=False`，导致 MiMo RL 仍输出大量 `<think>` 内容；
- 本地 `MiMo-7B-RL-0530` 的新版官方模板已通过在 assistant
  generation prompt 中预填空的 `<think>\n\n</think>` 实现关闭思考。

修改：

- 新增 `configs/fastmtp_no_think_chat_template.jinja`，保留 FastMTP 原模板
  其余语义，只增加 `enable_thinking` 分支；
- `scripts/serve_fastmtp_verified.sh` 新增 `FAST_MTP_NO_THINK=1`
  可选开关，通过 vLLM `--chat-template` 加载兼容模板，并用
  `--default-chat-template-kwargs` 将 `enable_thinking=false` 设为请求默认值；
- `scripts/run_fastmtp_verified_comparison.sh` 传递并记录该开关；
- 默认值为 `0`，避免改变 2026-08-09 当时仍在运行的
  `fastmtp_verified_pilot` 协议。

使用：

```bash
FAST_MTP_NO_THINK=1 \
SAMPLES=10 \
RUN_TAG=fastmtp_no_think_pilot \
./scripts/run_fastmtp_verified_comparison.sh
```

验证：

- shell syntax：通过；
- tokenizer template reference tests：`2 passed`；
- `enable_thinking=False` 渲染结果以空的已闭合 think block 结束；
- `enable_thinking=True` 渲染结果以开放 think block 结束；
- 未运行 GPU 生成对照，原因是用户的五方法 pilot 仍在占用 GPU。

限制与下一步：

- 该方式是将 MiMo 已有 non-thinking 模板语义移植到 FastMTP，
  不修改任何模型或 MTP 权重；
- 需在当前 pilot 完成后，使用新 `RUN_TAG` 跑小样本 A/B，确认
  实际输出长度、质量和各方法的分布仍然可公平比较。

### 2026-08-09：FastMTP 动态树低接受长度审计

状态：`负结果诊断；未修改算法代码`

动机：

- `fastmtp_verified_pilot` 中动态树的 MAL 低于 Native FastMTP，需检查
  “候选更多应提高命中率”的预期为何没有实现。

结果：

| 数据集 | Native MAL | Cactus MAL | 动态树 MAL | 动态树始终选最深存活前缀的离线 MAL 上限 |
|---|---:|---:|---:|---:|
| GSM8K-10 | 3.083 | 3.419 | 2.996 | 3.228 |
| HumanEval-10 | 2.887 | 3.390 | 2.773 | 2.971 |

根因：

1. 动态树几乎没有实际变宽。GSM8K/HumanEval 每轮平均目标节点仅为
   `3.294/3.474`，而深度为 3 的单链本身已有 3 个节点；根节点出现多个
   候选的轮次仅为 `2.57%/4.42%`，最终选择任何非 top-1 sibling 的轮次仅为
   `3.74%/5.01%`；
2. FastMTP 的 Q 极尖锐。三个深度的平均 top-1 概率在 GSM8K 为
   `0.934/0.913/0.895`，在 HumanEval 为 `0.902/0.867/0.843`。当前阈值
   `q_max * exp(-kappa * U)` 配合按全词表大小归一化的 entropy，使阈值长期
   接近 q_max，绝大多数父节点只保留一个 token；
3. `surviving_paths` 指标把一条链的深度 1、2、3 前缀分别计作三个 endpoint，
   不能解释为三条独立分支；
4. 路径选择器在所有存活前缀 endpoint 之间按“几何平均可信度 × 长度奖励”
   采样，而不是优先选择最长存活路径。GSM8K 有 `20.08%` 的轮次、HumanEval
   有 `17.52%` 的轮次选择了比当前可用最深路径更短的前缀，分别损失约
   `0.231/0.198` MAL；
5. 动态树使用 `relative_target_support >= 0.5` 的 target-dominant 非保分布
   路径规则；Cactus 则直接放宽单链候选的概率接受。两者的“候选覆盖”与
   “单候选接受强度”不是同一个量，因此多出少量 sibling 不保证超过 Cactus。

文件：

- 审计 `remtp/dynamic_mtp_tree.py`、`remtp/dynamic_tree_vllm.py`、
  `scripts/serve_fastmtp_verified.sh`；
- 读取本地 `results/fastmtp_verified_pilot/*/dynamic_tree/tree_rounds.jsonl`
  及各方法 `summary.json`；
- 更新 `docs/research_and_development_log.md`。

验证：

- 对 GSM8K 的 `2,838` 轮和 HumanEval 的 `6,450` 轮 tree audit 做了完整
  离线重放统计；
- 本次没有修改 Worker、验证语义或服务脚本，也没有启动新的 GPU 实验；
- “始终选择最深存活前缀”的数字是基于原有 target forward 结果的离线
  反事实上限，不是重新生成后的正式质量结果。

限制与下一步：

- 第一优先级应修正 endpoint 选择目标，至少做 `longest-surviving-prefix`
  消融；
- 第二优先级应改用 top-k 局部 entropy、top-1/top-2 margin 或显式候选数
  控制构树，避免全词表归一化 entropy 把树压回单链；
- 修改后必须重新报告真正的根分支数、叶路径数和非 top-1 sibling 选中率，
  不能继续用旧 `surviving_paths` 作为树宽指标。

### 2026-08-09：动态树最长前缀优先与受控微扩宽

状态：`代码已验证；单请求 GPU 冒烟通过；正式质量实验待运行`

动机：

- 修复路径评分器在已有更深存活路径时仍选择短前缀的问题；
- 在不形成宽树的前提下，为 Q 中具有实际概率质量的 top-2 提供少量候选覆盖。

修改：

1. 验证后只把“没有存活孩子的节点”视作真实路径 endpoint；
2. 路径选择改为字典序目标：先最大化存活深度，仅在相同最大深度的路径间
   根据目标可信度和 `T_path` 选择；
3. 构树主阈值保持不变，但当 top-2 同时满足 `q_2 >= tau_min` 和
   `q_2/q_1 >= 0.25` 时，允许增加一个 guarded sibling；
4. 每个父节点硬限制最多 2 个孩子，默认整树节点上限从实验性的 32 收紧为 6；
5. 分层分配宽度预算时，为每个尚未到达的后续深度预留一个节点，避免前两层
   siblings 耗尽预算并截断原有三层主路径；
6. `surviving_paths` 改为真实 maximal surviving endpoints 数，不再把同一链的
   中间前缀重复算成独立路径；
7. audit construction 增加 `threshold_eligible` 与 `sibling_added` 字段。

文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `tests/test_dynamic_mtp_tree.py`；
- `scripts/serve_fastmtp_verified.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/serve_mimo_dynamic_tree.sh`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/fastmtp_usage_guide.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关 Python 单元测试：`15 passed`；
- 四个修改后的 shell 入口通过 `bash -n`；
- `git diff --check` 通过；
- FastMTP/vLLM 0.18/TREE_ATTN 真实 GPU 冒烟完成 64 个输出 token：27 个
  speculative rounds，目标调用始终为每轮 1 次，平均/最大目标节点为
  `4.741/6`，全部 27 轮均保留到深度 3，5 轮触发了 `2+3+1` 的预留预算
  形状，4 轮选择了含非 top-1 sibling 的路径；
- 初始试验的 `N_max=8, q_2/q_1>=0.10` 在相同短请求上平均使用 5.923 个
  节点，被判断为扩宽过多，未保留为默认值；最终默认改为
  `N_max=6, q_2/q_1>=0.25`。

限制与下一步：

- GPU 冒烟只证明变量宽度、TreeAttention、路径提交和单次 target forward
  接线可运行；`MAL=2.370` 来自一个 64-token 请求，不应与旧 benchmark
  数字比较；
- 动态树验证仍是非保目标分布的 target-dominant relaxation，正式结果必须
  同时报告 GSM8K/HumanEval 质量；
- 下一步只需用新 `RUN_TAG` 跑小样本动态树，不必重跑已冻结 baseline；重点看
  MAL、平均/最大节点、真实叶路径数和非 top-1 sibling 选中率。

### 2026-08-09：最长前缀动态树 FastMTP-10 结果分析

状态：`小样本正向信号；不能据此确认质量优势`

协议：

- `fastmtp_longest_tree_n6_smoke`，GSM8K/HumanEval 各 10 条；
- temperature `0.6`、seed `42`、显式空 system、所有方法均启用同一
  no-thinking chat template；
- 动态树 `D=3, N_max=6, children_max=2, sibling_ratio=0.25,
  tau_relax=0.50`。

结果：

| 数据集 | 方法 | 质量 | MAL | E2E tok/s | nodes/round |
|---|---|---:|---:|---:|---:|
| GSM8K-10 | Native | 100% | 3.058 | 126.364 | 3.000 |
| GSM8K-10 | Cactus | 100% | 3.396 | 141.442 | 3.000 |
| GSM8K-10 | Dynamic tree | 100% | 3.323 | 115.909 | 3.701 |
| HumanEval-10 | Native | 70% | 3.067 | 125.728 | 3.000 |
| HumanEval-10 | Cactus | 60% | 3.344 | 136.836 | 3.000 |
| HumanEval-10 | Dynamic tree | 80% | 3.073 | 106.933 | 3.967 |

分析：

1. 最长前缀修改修复了旧负结果：动态树 MAL 在 GSM8K 已高于 Native
   `+0.265`，只比 Cactus 低 `0.073`；HumanEval 与 Native 基本相同；
2. 当前速度瓶颈不能归因于松弛不足。GSM8K 的 MAL 仅比 Cactus 低 `2.14%`，
   E2E 仍低 `18.05%`，对应目标节点多 `23.37%`；HumanEval 目标节点多
   `32.22%`；
3. `tau_relax=0.50` 附近的拒绝呈离散簇：现有 audit 中只有 GSM8K 18 个、
   HumanEval 33 个已到达节点位于约 `0.49935`。离线保守 replay 将阈值降至
   `0.49` 只预测 MAL 增加 `0.010/0.013`，不能弥合 HumanEval 对 Cactus 的
   `0.271` 差距；继续降到 `0.30` 的保守增益也只有 `0.017/0.025`；
4. HumanEval 质量只有 10 条：动态树相对 Native 仅有 1 个 discordant win、
   相对 Cactus 仅 2 个 discordant wins，均无足够统计功效。`80/70/60%`
   是正向信号，不是已确认的质量排序；
5. 当前额外 sibling 的路径选择率仅约 `3.8%/4.4%`。进一步加宽会优先增加
   节点成本，而不保证提高最长可提交前缀。

结论：

- 不建议全局大幅降低 `tau_relax`，也不建议继续加宽；
- 如需一个最小消融，可测试 `tau_relax=0.49`，但预期 MAL 增益很小；
- 若目标是同时接近 Cactus MAL 且保持 Native 质量，应实现“首次失败前沿的
  单次受限 rescue”：先使用当前目标主导树规则，只在当前深度无任何孩子存活
  时，对目标支持最高的一个候选使用一次小预算 Cactus-style rescue；不对所有
  sibling 全局放宽；
- 若目标还包括 Cactus 级实际速度，必须同时减少每轮节点/动态运行时开销；
  单纯放宽接受条件无法解释或消除 GSM8K 已存在的约 18% 系统速度差。

验证与文件：

- 读取 `results/fastmtp_longest_tree_n6_smoke` 的 summary、HumanEval evaluation
  与 2,773 轮 tree audit；
- 检查五种方法 server log，均确认 `no_think=enabled`；
- 本次未修改算法代码、未启动新 GPU 实验；仅更新本研究日志。

### 2026-08-09：动态树首次失败前沿的单次受限救援

状态：`代码与真实 GPU 冒烟已通过；GSM8K/HumanEval-50 正式质量实验待运行`

动机：

- `fastmtp_longest_tree_n6_smoke` 中 GSM8K 的动态树 MAL 已接近 Cactus，
  HumanEval 仍有约 `0.271` 差距；
- 离线审计显示 `tau_relax=0.50` 附近候选很少，全局降阈值预计只能增加约
  `0.01–0.03` MAL，同时会在所有位置累积质量风险；
- 当前阶段按用户要求优先优化 MAL 与准确率，系统吞吐优化后置。

算法修改：

1. 保留现有构树、`tau_relax=0.50`、最长存活路径优先、EOS 保护和目标 anchor；
2. 仅当某个已到达父节点没有任何正常存活孩子时，才允许一次 failed-frontier
   rescue；
3. 只考虑目标相对支持最高的一个 sibling，默认要求
   `p(y)/max(P)>=0.1` 且 `p(y)>=0.001`；
4. 对该候选使用有限 Cactus-style 标量提升：
   `h(y)=min(1,p(y)+sqrt(2*delta*p(y)*(1-p(y))))`，并以
   `min(1,h(y)/q(y))` 做随机接受，默认 `delta=0.5`；
5. rescue 状态沿路径传播，因此最终提交路径最多包含一次 rescue；同深度
   路径比较时正常路径优先于 rescue 路径；
6. 该树验证仍是近似松弛解码，不宣称恢复目标分布；完整 Q 只用于救援接受
   概率，目标 P 决定候选语义优先级。

可观测性与实验入口：

- 每个节点新增 `target_probability`、`draft_probability`、
  `rescue_eligible`、`rescue_accept_probability`、`rescue_random` 和
  `rescued`；可读 trace 显示 rescue 的 `a/u/ACCEPT|REJECT`；
- tree metrics 新增 rescue 尝试、成功、最终选中 token 数和选中 rescue 的轮次
  比例；统一对比表新增 `rescue rounds`；
- 新增 `scripts/run_fastmtp_mal_quality_50.sh`，默认在 GSM8K 与 HumanEval
  各跑 50 条，并比较 target / Native / Cactus / SpecCascade / 动态树；
- 默认启用 no-thinking 模板，结果写入单一 `comparison.md`。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `remtp/tree_audit.py`；
- `remtp/mimo_tree_report.py`；
- `remtp/fastmtp_verified_report.py`；
- `tests/test_dynamic_mtp_tree.py`；
- `scripts/serve_fastmtp_verified.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_mal_quality_50.sh`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/fastmtp_usage_guide.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关单元测试：`18 passed`；
- 修改后的 Python 模块通过 `py_compile`；四个 shell 入口通过 `bash -n`；
- `git diff --check` 通过；
- FastMTP/vLLM 0.18/TREE_ATTN 真实 GPU 冒烟生成 128 tokens，共 52 个
  speculative rounds；每轮 target forward 始终为 1；
- 冒烟平均 MAL `2.442`、平均节点 `4.712`；4 轮最终路径使用 rescue，
  `max_rescues_per_selected_path=1`，验证了单次约束与在线审计接线。

限制与下一步：

- 单请求冒烟的 MAL 不能与正式 benchmark 比较，也不能证明准确率；
- rescue 使用 top-Q 树候选而非从完整 Q 独立采样，因此即使采用
  Cactus-style 比例也不构成严格分布保持；
- 下一步运行 `scripts/run_fastmtp_mal_quality_50.sh`。主要判断动态树是否在
  HumanEval/GSM8K 上提高 MAL，同时质量与 Native FastMTP 的置信区间相容；
  若质量下降明显，先把 `DYNAMIC_RESCUE_DELTA` 从 `0.5` 降到 `0.25`，而不是
  继续增加全局阈值或树宽。

### 2026-08-09：FastMTP 大样本脚本跳过 Target 并启用逐样本实时日志

状态：`代码与报告测试通过；等待用户运行`

动机：

- Target-only 约 `60 tok/s`，明显延长四方法 MAL/质量实验，但当前阶段不需要
  再用它估计相对自回归加速；
- 原脚本默认每 10 条打印一次，使长时间 GSM8K/HumanEval 运行难以观察。

修改：

1. `run_fastmtp_verified_comparison.sh` 默认 `INCLUDE_TARGET=0`，只运行
   Native / Cactus / SpecCascade / Dynamic tree；仍可显式设为 1 恢复旧控制；
2. `run_fastmtp_mal_quality_50.sh` 固定跳过 Target，并默认
   `PROGRESS_EVERY=1`；
3. Python 使用 unbuffered stdout，GSM8K 每题、HumanEval 每次生成和隔离判题
   都经 `tee` 实时输出，同时保存在 `logs/$RUN_TAG/`；
4. 增加方法启动、服务健康、当前数据集阶段和日志目录提示；
5. 报告器允许 Target 目录缺失，速度列改为 `speed vs Native`，不再引用
   不存在的 Target 行；四个 FastMTP 方法仍为强制完整对照组。

修改文件：

- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_mal_quality_50.sh`；
- `remtp/fastmtp_verified_report.py`；
- `tests/test_fastmtp_verified_report.py`；
- `docs/fastmtp_usage_guide.md`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关测试：`19 passed`；
- 两个 shell 入口通过 `bash -n`；
- 报告模块通过 `py_compile`；
- `git diff --check` 通过；
- 新增无 Target fixture，确认报告仅包含四个方法且 Native 速度倍率为 1。

限制与下一步：

- 实时日志不包含每轮所有树节点，避免详细 token 解码干扰正式吞吐；每条任务
  进度和最终 tree rescue 指标仍完整保留；
- 若要临时跟踪某个日志，可使用
  `tail -f logs/$RUN_TAG/<method>_gsm8k.log` 或对应 HumanEval 日志。

### 2026-08-09：FastMTP-100 前沿救援动态树结果与路径/宽度诊断

状态：`正式 100 条结果已分析；当前版本保留，下一轮修改待实施`

协议：

- `results/fastmtp_live_n100`；GSM8K 与 HumanEval 各 100 条；
- temperature `0.6`、seed `42`、显式空 system、no-thinking；
- Native / Cactus / SpecCascade / Dynamic tree 使用同一 FastMTP checkpoint；
- 动态树 `D=3, N_max=6, children_max=2, sibling_ratio=0.25`，启用单次
  failed-frontier rescue，`delta=0.5`。

原始结果：

| 数据集 | 方法 | 质量 | MAL | E2E tok/s | nodes/round |
|---|---|---:|---:|---:|---:|
| GSM8K | Native | 90% | 3.040 | 125.649 | 3.000 |
| GSM8K | Cactus | 90% | 3.434 | 141.529 | 3.000 |
| GSM8K | SpecCascade | 93% | 3.229 | 132.601 | 3.000 |
| GSM8K | Dynamic tree | 82% | 3.232 | 111.664 | 3.850 |
| HumanEval | Native | 61% | 3.065 | 126.552 | 3.000 |
| HumanEval | Cactus | 69% | 3.342 | 137.904 | 3.000 |
| HumanEval | SpecCascade | 66% | 3.143 | 128.953 | 3.000 |
| HumanEval | Dynamic tree | 69% | 3.150 | 109.094 | 3.937 |

统计与错误分析：

1. GSM8K 动态树相对 Native/Cactus 的配对 win/loss 均为 `2/10`，精确
   McNemar/binomial `p=0.0386`；相对 SpecCascade 为 `0/11`，`p=0.0010`。
   当前 82% 下降不能继续按小样本噪声解释；
2. HumanEval 动态树与 Cactus 为 `7/7` discordance，质量相同；相对 Native
   为 `15/7`，但 `p=0.134`，仍需第二 seed；
3. GSM8K 动态树 18 个错误中 16 个是 `max_tokens=2048` 截断。未截断样本为
   `78/80=97.5%` 正确，Native 未截断为 `86/88=97.7%`。因此主要问题是近似
   路径选择把少数请求带入不终止的长轨迹，而非普通完成后的算术错误；
4. 截断路径从未遇到高 EOS 概率节点，简单降低 EOS veto 阈值无效；偏移发生后
   target 本身已不想结束，风险必须在较弱路径 token 提交前控制；
5. rescue 在 GSM8K 正确/错误请求中为每 100 轮 `3.73/4.48` 次；HumanEval
   为 `2.53/4.77` 次。非 top-1 branch 选择也在错误请求中更高，说明 target
   支持和 rescue 信号有风险辨识力，但 rescue 不是唯一根因；
6. GSM8K 正确与错误请求的 MAL 几乎相同（`3.233/3.232`），所以不能简单按
   长度截断；应根据路径可信度决定什么时候放弃额外深度。

路径公式离线 replay：

- 最初设计为
  `C(path)=exp(mean(log S_i))`，`Score(path)=C(path)*L^beta`；当前实现只在
  最大深度相同的 endpoint 之间使用该分数，实际近似“无条件最长”；
- 让所有存活前缀重新参与选择，并用 `beta=0.75`、确定性最高分选择时，离线
  replay 的 GSM8K/HumanEval MAL 为 `3.217/3.138`，仅比当前少
  `0.015/0.012`；分别有 `1.57%/1.27%` 的轮次主动选择较短路径；
- 同时所选路径累计负对数支持风险代理下降约 `38%/35%`。该配置是下一轮最
  值得测试的准确率—MAL 平衡点；`beta=0.25` 过于保守，`beta>=2` 又基本退化
  回最长路径。

树宽诊断：

- `N_max=6` 只在 GSM8K/HumanEval 的 `19.1%/21.4%` 轮次用满，平均节点仅
  `3.85/3.94`，节点总量不是首要瓶颈；
- 根节点多分支仅 `13.5%/14.3%`，最终选择 branch-1 仅 `4.3%/4.5%`；
- 前两层出现至少 3 个 eligible 候选的比例仅约 `0.17%–0.85%`，所以把
  `max_children` 从 2 提到 3 基本无收益；
- useful-node ratio 只有 `58.0%/54.6%`，盲目加节点会先增加无效验证；
- 仍可测试小幅扩宽：保持每父节点最多 2 个孩子，将 `N_max=8`、
  `min_sibling_ratio=0.20`，让更多父路径进入第三层；不要直接使用已否定的
  `N=8, ratio=0.10` 激进配置。

建议顺序：

1. 先恢复所有存活前缀的原始 `C(path)*L^beta` 选择，默认 `beta=0.75`、
   `T_path=0`；rescue 节点继续使用 `S*a_rescue`，从而自动承担额外风险惩罚；
2. 在该选择器稳定质量后，再做唯一的宽度消融：`N=6, ratio=.25` 对比
   `N=8, ratio=.20`，`max_children=2` 不变；
3. 若仍出现长轨迹，再加入通用的跨轮风险债务，债务只降低下一轮长度奖励和
   rescue 许可，不做数学任务特化；
4. 进入标准对比的目标应为：GSM8K 不低于 Native 90%，HumanEval 与 Cactus
   69% 相容，同时 MAL 明显高于 SpecCascade `3.229/3.143`。

验证与文件：

- 读取两数据集共 `39,234` 个动态树 rounds、800 个四方法请求记录和 HumanEval
  隔离判题结果；
- 完成请求级 audit 分段、截断/正确性、rescue/branch、节点预算和原公式的离线
  replay；
- 本次仅更新 `docs/research_and_development_log.md`，未修改在线算法。

### 2026-08-09：Balanced Path 与 8 节点小幅扩宽版本

状态：`实现、测试与真实 GPU 冒烟通过；等待 100 条正式结果`

目标：

- 解决旧实现“只要存在更深存活节点就无条件选择最长路径”的长轨迹风险；
- 在不增加每父节点分支上限的条件下，把整树节点预算从 6 小幅提高到 8，检验
  候选覆盖是否能够提升 MAL；
- 不重复运行已经冻结的 Native / Cactus / SpecCascade 100 条结果。

算法修改：

1. `DynamicTreeConfig` 新增 `path_selection_mode`：
   - `longest` 完整保留旧语义；
   - `balanced` 令每个存活前缀参与
     `Score(path)=geomean(S_i)*L^beta` 比较，不再要求必须到达最深 endpoint；
2. 新实验固定 `beta=0.75, T_path=0`。因此长路径仍有明确奖励，但低可信尾部会
   降低几何平均支持度，可能被更可靠的短前缀击败；
3. rescue 节点的 `S_i` 继续乘其实际 rescue acceptance probability，所以
   balanced selector 会自动对由救援得到的尾部承担额外惩罚；
4. 树配置改为 `D=3, N_max=8, max_children=2, sibling_ratio=0.20`。扩宽来自给
   更多强父路径分配深层节点，而不是允许每个位置 top-3/top-4 泛滥；
5. 正常松弛阈值、frontier rescue delta、EOS 保护和 target anchor 均不变，便于
   将差异归因到路径选择与小幅候选扩宽。

运行入口：

```bash
SAMPLES=100 \
BASELINE_RUN_ROOT=results/fastmtp_live_n100 \
RUN_TAG=fastmtp_balanced_wide_n100 \
  ./scripts/run_fastmtp_balanced_wide_n100.sh
```

该入口只运行 `dynamic_tree`。它把旧 run 中 `native/cactus/spec_cascade` 的结果
目录只读式链接到新 run，再由原报告器检查相同 sample manifest 和协议，最终
仍生成包含四种方法的一个 `comparison.md`。底层比较脚本新增了受校验的
`METHODS_CSV` 子集运行参数。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_verified.sh`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/serve_mimo_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_balanced_wide_n100.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/fastmtp_usage_guide.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关单元测试 `20 passed`；新增测试明确验证两个 token 都存活时，balanced
  selector 会因第二 token 目标相对支持仅 0.25 而选择安全的一 token 前缀；
- Python 模块通过 `py_compile`，5 个 shell 入口通过 `bash -n`，
  `git diff --check` 通过；
- FastMTP/vLLM 0.18/TREE_ATTN 真实 GPU 冒烟生成 256 tokens，共 98 个 round；
  每轮 target forward 始终为 1，验证节点最大 8、平均 5.041；
- 冒烟平均接受草稿深度 1.612，对应 MAL 2.612；单请求中没有出现 balanced
  主动短停，符合离线估计中约 1%–2% round 才触发的稀有率。

限制与判据：

- 8 节点会增加 tree target forward 延迟，因此当前优先判断 MAL/accuracy，吞吐
  仅作为审计项；
- balanced 的离线 replay 预测 MAL 会轻微下降约 0.01，但可能显著减少低可信
  路径风险。扩宽能否补回并超过这部分 MAL 必须由正式结果回答；
- 主判据：GSM8K accuracy 至少恢复到 Native 的 90% 且截断率明显低于旧动态树
  20%；HumanEval pass@1 不低于旧动态树/Cactus 的 69%；MAL 应超过
  SpecCascade 的 GSM8K 3.229 与 HumanEval 3.143。若 MAL 没有提升，不再继续
  增加节点，而应回到候选质量或分支利用率分析。

### 2026-08-09：Balanced/Wide-100 结果与节点利用率诊断

状态：`100 条结果完成；证明质量风险有所缓解，但 8 节点上限没有形成真正宽树`

原始结果：

| 数据集 | 方法 | 质量 | MAL | nodes/round | useful-node | truncation |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | 旧 Dynamic N6/longest | 82% | 3.232 | 3.850 | 58.0% | 20% |
| GSM8K | 新 Dynamic N8/balanced | 88% | 3.230 | 4.171 | 53.5% | 17% |
| HumanEval | 旧 Dynamic N6/longest | 69% | 3.150 | 3.937 | 54.6% | 9% |
| HumanEval | 新 Dynamic N8/balanced | 67% | 3.166 | 4.268 | 50.7% | 11% |

统计边界：

- GSM8K 新方法 88/100 的 Wilson 95% CI 为 `80.2%–93.0%`；HumanEval
  67/100 为 `57.3%–75.4%`，100 条不足以区分约 2–3 pp 的质量差异；
- 新旧动态树配对质量变化：GSM8K `9 wins / 3 losses`，HumanEval
  `6 wins / 8 losses`，均未达到显著；
- 请求级 paired bootstrap 的新旧 MAL 差 95% CI 分别为
  GSM8K `[-0.041, +0.036]`、HumanEval `[-0.015, +0.046]`。当前不能宣称
  小幅扩宽稳定提高 MAL；
- balanced selector 仅在 GSM8K/HumanEval 的 `1.37%/1.32%` round 主动停在
  更短存活前缀，符合它是长轨迹风险控制而非主要 MAL 增长器的定位。

节点利用率：

- GSM8K/HumanEval 平均节点仅 `4.171/4.268`；恰好 8 节点的 round 只有
  `8.08%/9.02%`，至少 6 节点也只有 `22.86%/24.51%`；
- 超过一半 round 仍是 3 节点单链：GSM8K 10,161/18,023，HumanEval
  9,991/18,930；
- 根层仅 `15.82%/16.25%` 的父状态保留两个候选；深度 2 为
  `20.17%/22.43%`，深度 3 为 `19.92%/22.20%`。因此 `N_max=8` 大部分时间
  没有候选可以填入；
- 已保留的第二候选并非无用：三个深度中，被实际访问的 sibling 约
  `29.7%–33.0%` 能通过验证；根 sibling 中约 29% 还是 target top-1。该信号
  支持继续提高候选覆盖，但不能证明任意低 Q 候选都有相同价值；
- 从 N6 到 N8 后，GSM8K MAL 基本不变，HumanEval 仅 `+0.016`，而 useful-node
  ratio 下降约 4 pp。单独继续提高 `N_max` 不会解决候选在阈值前被删除的问题。

下一阶段建议（仅优化 MAL 与质量，暂不以吞吐为首要目标）：

1. 将构树从“阈值决定节点数”改为“质量下限 + 全局预算填充”：每个父节点
   暴露 top-2 备选及其 Q 概率，使用全树优先队列选择最高价值的 8–10 个节点；
   保留极低 Q 硬下限，避免为了填满预算验证纯噪声；
2. 最大深度从 3 增到 4。D=3 即使无限加宽，MAL 理论上限也只有
   `3 draft + 1 anchor = 4`；增加宽度只能提高到达深度 3 的概率，无法产生
   第四个草稿 token。建议第一版 `D=4, N=10`；
3. 节点价值应改成近似整块期望 MAL 的前缀概率收益，而不是仅使用路径 Q
   几何平均。早期备选可以解锁多个后续 token，应高于孤立的深层 sibling；
4. 路径提交保留 balanced 风险控制，但将 relaxed 风险由平均值改为可调的累计
   项作为消融，防止多枚边缘 token 因几何平均和长度奖励被连续提交；
5. 新构树先在 100 条上验证“平均节点达到 8–10、target calls=1、MAL 增长”；
   质量确认改为 GSM8K 至少 500 条、HumanEval 全部 164 条并增加第二 generation
   seed。100 条只用于淘汰明显失败配置。

结论：

> 当前瓶颈确实包含候选树覆盖不足，但原因不是 `N_max` 数字太小，而是 Q 阈值
> 使节点预算长期空置；同时 D=3 限制了 MAL 天花板。最有价值的下一版是
> `D=4 + N=10 + 全局预算填充式 top-2 候选分配`，而不是简单改成 `N_max=10`。

### 2026-08-09：D=3/D=4 概率守卫候选扩宽完整实验入口

状态：`实现、单元测试和 D3/D4 真实 GPU 冒烟通过；等待完整实验`

根据用户要求，本版本没有实现强制最小节点数，也不会用低质量 token 机械填满
预算。修改后的构树规则为：

1. 动态熵阈值产生 core candidates；
2. 在 `max_children` 范围内，top-2/top-3 可以作为 guarded backups；
3. 每个 backup 必须同时满足
   `q >= min_draft_prob` 和 `q/q_top1 >= min_sibling_ratio`；
4. 所有通过概率守卫的候选再按现有路径扩展价值竞争全局节点预算；
5. 候选不足时保留实际节点数，不进行预算补齐。

完整实验固定相同验证节点上限，对比：

| 配置 | D | N_max | max children | q floor | q/q1 floor | kappa |
|---|---:|---:|---:|---:|---:|---:|
| relaxed tree D3 | 3 | 10 | 3 | 0.002 | 0.02 | 1.25 |
| relaxed tree D4 | 4 | 10 | 3 | 0.002 | 0.02 | 1.25 |

两者均保留 `coverage_gate`、`tau_relax=0.50`、balanced path、`beta=0.75`、
`T_path=0`、单次 frontier rescue 和 target anchor。D4 只是额外递归使用一次
FastMTP 的单个已训练物理 MTP layer，不宣称存在第四个独立已训练 head。

统一运行入口：

```bash
RUN_TAG=fastmtp_depth34_relaxed_full \
  ./scripts/run_fastmtp_depth34_relaxed_tree.sh
```

默认协议：

- GSM8K 500 条；
- HumanEval 全部 164 条；
- phase 1 运行 Native / Cactus / SpecCascade / D3；
- phase 2 只运行 D4，并通过符号链接复用 phase 1 的三组 baseline；
- phase 3 生成一个 `depth34_comparison.md/json`，同时显示五种方法。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/fastmtp_depth34_report.py`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_depth34_relaxed_tree.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `tests/test_fastmtp_depth34_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/fastmtp_usage_guide.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关测试 `23 passed`；新增测试验证 guarded top-3 能增加两个 backup，同时
  继续服从每父节点硬上限；
- Python 模块通过 `py_compile`，shell 入口通过 `bash -n`，
  `git diff --check` 通过；
- D3 真实 FastMTP/vLLM/TREE_ATTN 冒烟：37 rounds，平均/最大节点
  `9.459/10`，MAL `2.324`，每轮 target calls=1；
- D4 冒烟：54 rounds，平均/最大节点 `9.889/10`，MAL `2.352`，实际最大接受
  草稿深度为 4，每轮 target calls=1；
- 两个 chat 冒烟均启用 no-thinking，响应不包含 `<think>` 或 `</think>`；
- 冒烟 MAL 来自单个短提示，只用于验证接线，不能作为算法收益结论。

限制：

- 放宽 Q 守卫后 target 节点几乎达到 10，符合本阶段候选覆盖目标，但 useful-node
  ratio 与正式准确率必须由完整数据决定；
- D4 使用权重共享递归，第四步候选质量可能下降；正式报告必须分开检查各深度
  survival、首个失败位置、MAL 与任务质量；
- 当前阶段优先验证 MAL/accuracy。吞吐仍被 eager TreeAttention 和更多 target
  nodes 影响，不作为否定算法正确性的首要标准。

### 2026-08-10：D3 完整实验 MAL 回退诊断

状态：`D3 的 GSM8K-500/HumanEval-164 已完成；D4 继续运行，未干预`

原始结果：

| 数据集 | 方法 | 质量 | MAL | nodes/round | accepted/nodes | truncation |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | Native | 88.2% | 3.046 | 3.000 | 68.2% | 12.8% |
| GSM8K | Cactus | 87.4% | 3.450 | 3.000 | 81.7% | 11.4% |
| GSM8K | SpecCascade | 87.0% | 3.213 | 3.000 | 73.8% | 12.4% |
| GSM8K | Relaxed Tree D3 | 88.6% | 3.169 | 6.838 | 31.7% | 14.6% |
| HumanEval | Native | 61.6% | 3.071 | 3.000 | 69.0% | 11.0% |
| HumanEval | Cactus | 68.3% | 3.344 | 3.000 | 78.1% | 9.1% |
| HumanEval | SpecCascade | 65.2% | 3.169 | 3.000 | 72.3% | 11.0% |
| HumanEval | Relaxed Tree D3 | 66.5% | 3.098 | 7.015 | 29.9% | 11.6% |

统计：

- 所有质量配对差异均不显著：D3 对 Native/Cactus/SpecCascade 的 exact paired
  p-value 在 GSM8K 为 `0.888/0.441/0.312`，HumanEval 为
  `0.230/0.711/0.860`；当前准确率没有显示系统性崩坏；
- 请求级 paired bootstrap 显示 GSM8K D3 MAL 相对 Native 增加区间
  `[+0.099,+0.146]`，但相对 SpecCascade 为 `[-0.067,-0.022]`、相对 Cactus
  为 `[-0.303,-0.258]`；HumanEval 对 Native 差异包含 0，但相对
  SpecCascade/Cactus 分别为 `[-0.099,-0.043]`、`[-0.276,-0.214]`；
- 因此“节点更多但 MAL 更低”不是 100 条小样本噪声，至少相对 Cactus 和
  SpecCascade 已是稳定方向。

原因分解：

1. 新节点大幅增加，但低 rank 节点质量下降：
   - rank-0 在三个深度的 `survive/visited` 约 `72%–76%`；
   - rank-1 仅约 `23%–26%`；
   - rank-2 仅约 `13%–15%`；
   - 因此 useful-node ratio 降至 GSM8K `31.7%`、HumanEval `29.9%`。
2. 宽度挤占了深度：
   - 每轮分配到深度 3 的节点为 `2.552/2.597`，但只有 `42.2%/40.4%` 真正
     到达可验证父前缀；
   - `30.2%/30.7%` 的深度 3 节点位于已经失败、逻辑上不可达的深度 2 父路径；
   - 在已经存活的深度 2 节点中，`8.55%/9.41%` 完全没有分配深度 3 子节点，
     说明早期 top-2/top-3 消耗预算后，部分真正有效的父路径反而无法继续。
3. 失败位置：
   - root 无存活候选占 `15.53%/17.02%` round；
   - 深度 1 的孩子全部失败占 `9.67%/10.61%`；
   - 深度 2 的孩子全部失败占 `9.88%/10.15%`；
   - 深度 2 已存活但没有子节点占 `5.83%/6.28%` round。
4. balanced path 不是主要原因：
   - 主动选择短于最大存活深度的 round 只有 `1.10%/0.99%`；
   - 假设总是提交最长存活路径，MAL 天花板也仅为 `3.180/3.108`，只比实际
     `3.169/3.098` 高约 0.01，仍明显低于 Cactus。
5. 现有 proposal priority 使用路径 Q 几何平均并奖励深度。几何平均不会充分
   惩罚早期低 Q 分支，导致低 reach backup 的后代与高 reach 主路径争夺相同
   节点预算；这是宽度转化不成 MAL 的核心结构问题。

后续建议：

- 先让当前 D4 完整运行，不在中途改变协议；D4 可以判断额外深度能否部分抵消
  D3 的宽度挤占；
- 下一版不要继续降低 Q floor 或增加 max children；改为 progressive widening：
  先给高价值父路径分配 rank-0 continuation，再全局分配 rank-1，最后才考虑
  rank-2；
- 节点优先级从路径 Q 几何平均改为累计 prefix probability（Q 概率乘积）乘以
  可解锁剩余长度，更接近整块期望 MAL；
- 对每个被保留的高 reach 父路径提供 continuation reserve，避免存活深度 2
  节点没有深度 3/4 子节点；
- balanced selector 保持不变。若在修复分配后仍需增加 MAL，再单独搜索
  `tau_relax`，不要与构树修复同时修改。

一句话结论：

> D3 不是“树还不够宽”，而是“宽度分配给了低 reach 分支，并挤掉了有效路径的
> 延伸节点”；当前应优化节点预算的层级与路径分配，而不是继续无差别加候选。

### 2026-08-10：D4 完整结果、公平深度审计与 Reach-first D3

状态：`D4 完整实验已分析；Reach-first D3 已实现并通过 CPU 测试；等待 GPU 正式实验`

> 后续修订：本节的硬 rank 分层分配未进入正式 GPU 实验，已由下一节的
> `soft_reach` 连续效用分配取代；历史诊断和 D3/D4 数据继续有效。

D3/D4 原始结果如下。D4 与 D3 使用相同 10 节点上限和验证器，但 D4 可额外
提交一个草稿 token，因此不能只比较原始 MAL。

| 数据集 | 方法 | 质量 | MAL | nodes/round | accepted/nodes |
|---|---|---:|---:|---:|---:|
| GSM8K-500 | Native | 88.2% | 3.046 | 3.000 | 68.2% |
| GSM8K-500 | Cactus | 87.4% | 3.450 | 3.000 | 81.7% |
| GSM8K-500 | SpecCascade | 87.0% | 3.213 | 3.000 | 73.8% |
| GSM8K-500 | Relaxed Tree D3 | 88.6% | 3.169 | 6.838 | 31.7% |
| GSM8K-500 | Relaxed Tree D4 | 88.0% | 3.382 | 8.575 | 27.8% |
| HumanEval-164 | Native | 61.6% | 3.071 | 3.000 | 69.0% |
| HumanEval-164 | Cactus | 68.3% | 3.344 | 3.000 | 78.1% |
| HumanEval-164 | SpecCascade | 65.2% | 3.169 | 3.000 | 72.3% |
| HumanEval-164 | Relaxed Tree D3 | 66.5% | 3.098 | 7.015 | 29.9% |
| HumanEval-164 | Relaxed Tree D4 | 67.7% | 3.320 | 8.637 | 26.9% |

公平深度审计：

| 数据集 | D3 MAL | D4 原始 MAL | D4 截断到前三层 | D4 第四层贡献 |
|---|---:|---:|---:|---:|
| GSM8K | 3.169 | 3.382 | 3.031 | 0.351 |
| HumanEval | 3.098 | 3.320 | 2.987 | 0.334 |

关键结论：

1. D4 相比 D3 的原始增益为 `+0.213/+0.222`，但其第四层本身贡献
   `+0.351/+0.334`；去掉天然的第四层后，D4 前三层反而比 D3 低
   `0.138/0.111`。因此 D4 的表面优势不能归因于更好的树候选覆盖。
2. D4 有 `46.2%/46.8%` 的已分配节点最终未访问，高于 D3 的
   `41.0%/42.4%`；`41.9%/41.8%` 的第四层节点还挂在不可达的深度 3
   父路径下。
3. D4 的 cap=10 命中率达到 `67.1%/68.9%`，但前三层 reach 全面下降：
   GSM8K 从 D3 的 `0.845/0.744/0.580` 降到 `0.814/0.697/0.519`，
   HumanEval 从 `0.830/0.720/0.549` 降到 `0.803/0.684/0.500`。
4. 各层 rank-0 节点的 visited survival 约为 `69%–73%`，rank-1 约
   `22%–26%`，rank-2 约 `13%–16%`。继续平等扩展低 rank 分支会浪费
   有限节点，而不是提高最长可提交前缀。

实现改动：

- 新增 `allocation_mode=reach_first`，旧 `geometric` 模式保持不变以复现实验；
- reach-first 用整条路径的累计 Q 概率作为 reach proxy，不再使用 Q 几何平均；
- 同一深度先竞争 rank-0 continuation，剩余节点再给 rank-1、rank-2；每个
  rank 档位内按累计前缀 Q 排序；
- 正式入口固定 `D=3`，与三组 baseline 使用相同三步 MTP 递归深度；D4 只保留
  为深度消融；
- 新入口默认继续使用 10 节点上限、Q 概率守卫、balanced path、target anchor
  和 EOS 保护，并将 `tau_relax` 从 0.50 调到 0.35。现有日志显示 0.35–0.50
  区间新增候选仅约 `0.050/0.042 node/round`，属于受控放宽，而非无条件提交。

涉及文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_reach_first_tree.sh`；
- `scripts/run_fastmtp_soft_reach_tree.sh`（当前方法的公开入口）；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：

- `tests/test_dynamic_mtp_tree.py`、`tests/test_fastmtp_verified_report.py`、
  `tests/test_fastmtp_depth34_report.py`：`24 passed`；
- Python `py_compile`、三个 shell 入口 `bash -n`、相关文件 `git diff --check`
  通过；
- 尚未运行 Reach-first 的真实 GPU 冒烟或正式 benchmark，因此不宣称其已超过
  SpecCascade/Cactus。正式目标是 D3 MAL 高于 SpecCascade、接近 Cactus，且
  质量接近 Native。

运行入口：

```bash
RUN_TAG=fastmtp_reach_first_d3_full \
  ./scripts/run_fastmtp_reach_first_tree.sh
```

该脚本只运行新方法，并复用
`results/fastmtp_depth34_relaxed_full/d3` 中协议一致的 Native/Cactus/SpecCascade
结果，避免重复测试已有 baseline。

### 2026-08-10：取消层/父路径硬配额，改为 Soft-reach 联合路径评分

状态：`已实现并通过 CPU 测试；等待 GPU 正式实验`

根据用户反馈，未测试的硬 progressive rank 顺序不再作为当前方法。新规则没有
“每层必须几个节点”“每个父节点必须一个 rank-0 子节点”等约束：

```text
proposal_utility(path)
  = product(Q along path) / (1 + lambda_rank * local_rank)
```

- 只有通过动态 Q 阈值、绝对 Q floor 和相对 Q/Q1 floor 的候选才能竞争；
- rank 是连续软惩罚，默认 `lambda_rank=2.0`，不是 rank 档位；
- 高 reach 的 rank-1/rank-2 可以超过低 reach 的 rank-0；
- 每层和每个父节点都没有节点数下限，`N_max=10` 仍只表示全树上限；
- level reserve 只防止早期宽度耗尽全部物理槽位，不会强制生成或提交未来节点。

最终路径继续使用 balanced 目标：

```text
confidence(path) = exp(mean_i(log S_i))
score(path) = confidence(path) * length(path)^beta
```

所有存活前缀都可以参与评分，因此不是默认选择最长路径。默认 `beta=0.75`、
`T_path=0`：确定性选择置信度—长度联合分数最高的前缀。现有单元测试覆盖“弱
尾部虽然存活，但高置信短前缀胜出”的情况。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_reach_first_tree.sh`；
- `scripts/run_fastmtp_soft_reach_tree.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：

- 动态树、报告相关测试共 `25 passed`；
- `py_compile`、三个 shell 入口的 `bash -n` 及环境配置读取通过；
- soft-reach 单测明确验证高效用 rank-1 可以超过低效用 rank-0；
- balanced path 既有单测明确验证弱长尾可输给高置信短前缀。

限制：尚未运行真实 GPU 冒烟或 benchmark，不能宣称 soft-reach 已提高 MAL。

### 2026-08-10：Soft-reach 拒绝样本审计与目标确认式放宽

状态：`分析完成；dual/confirmed 已实现；GPU 冒烟通过但未显示 MAL 正收益；建议先跑 100 条筛选`

Soft-reach D3 完整结果：

| 数据集 | 质量 | Native 质量 | MAL | Cascade MAL | Cactus MAL |
|---|---:|---:|---:|---:|---:|
| GSM8K-500 | 89.2% | 88.2% | 3.166 | 3.213 | 3.450 |
| HumanEval-164 | 65.2% | 61.6% | 3.112 | 3.169 | 3.344 |

质量仍有一定余量，但 MAL 分别比 Cascade 低 `0.047/0.057`。拒绝原因审计：

- coverage 拒绝 `58,580/22,950` 个节点，但其中 relative>=0.10 的只有
  `3/0` 个；该门控几乎只删除目标概率接近 0 的噪声，不应放宽；
- coverage 通过后，relative 位于 `0.20–0.35` 的拒绝为每轮
  `0.019/0.018`，位于 `0.10–0.20` 的拒绝为每轮 `0.039/0.034`；
- 因此无条件阈值 0.10 会新增约 `0.069/0.061 node/round`，数量级足以覆盖
  当前 Cascade MAL 缺口，但不能保证后续 MTP 仍然对齐；
- 当前规则还拒绝了 `2,450/747` 个 `P(y)>=Q(y)` 节点，即每轮
  `0.028/0.024`。这说明单独依赖 target top-1 relative 会比 proposal-overlap
  信号更保守。

5 个提示、146 rounds 的详细 token 审计发现以下代表性案例：

| candidate | target top-1 | P | Q | relative | 后续证据 | 判断 |
|---|---|---:|---:|---:|---|---|
| `reducing` | `allowing` | 0.200 | 0.449 | 0.249 | 子候选 `the` 是 target top-1 | 可救 |
| `string` | `sequence` | 0.110 | 0.999 | 0.125 | 子候选 `that` 是 target top-1 | 可救但需谨慎 |
| `options` | `paths` | 0.143 | 0.059 | 0.249 | P/Q>1 | 可救 |
| `and` | `,` | 0.150 | 0.425 | 0.176 | 后续未与 target top-1 对齐 | 不应仅凭局部概率救 |
| `\\` | 空格 | 0.059 | 0.811 | 0.062 | 后续 top-1 对齐但当前格式风险较高 | 边界样本 |

实现了两个独立消融：

1. `dual`：coverage/EOS 通过后，`relative>=0.10 OR P/Q>=1`；
2. `confirmed`（推荐筛选）：保留 `relative>=0.35` 主规则，额外允许
   `relative>=0.05 AND (P/Q>=1 OR retained child == target next top-1)`。

相同 5 提示真实 GPU 冒烟：

| 版本 | rounds | MAL | 说明 |
|---|---:|---:|---|
| soft-reach relative-0.35 | 146 | 2.712 | 参考 |
| unconditional dual/relative-0.10 | 152 | 2.605 | 反向，不推荐直接全量 |
| target-confirmed | 149 | 2.644 | 优于无条件放宽，但仍低于参考 |

三组生成总 token 数相同；每轮 target forward 均为 1。该 5 提示结果不是质量
统计，但证明“放宽当前 token”可能改变真实轨迹并降低后续接受，因此不能根据
局部新增节点数直接推断 MAL 必然上升。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_reach_first_tree.sh`；
- `scripts/run_fastmtp_dual_support_tree.sh`；
- `scripts/run_fastmtp_confirmed_support_tree.sh`；
- `scripts/run_fastmtp_confirmed_support_n100.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：相关单元测试 `29 passed`；Python/shell 静态检查通过；dual 与 confirmed
均完成真实 FastMTP/vLLM/TREE_ATTN 冒烟，服务正常退出且端口释放。

结论：确实存在值得救援的拒绝 token，但证据不支持“把所有验证条件大幅放开”。
下一步只应在 100 条同协议样本上筛选 target-confirmed；若其 MAL 未超过现有
soft-reach 或质量明显下降，则停止该放宽，而不是继续降低全局阈值。

### 2026-08-10：Confirmed N=100 复盘与 MAL-first 经验阈值

状态：`结果审计完成；新增 MAL-first 压力实验入口；尚未运行新 GPU 实验`

用户完成的 N=100 结果表明，target-confirmed tree 在 GSM8K/HumanEval 上分别为
`88.0% / MAL 3.184` 和 `68.0% / MAL 3.100`；对应 Cactus 为
`90.0% / 3.434` 和 `69.0% / 3.342`。树每轮验证 `6.904/7.096` 个节点，
但草稿利用率只有 `31.6%/29.6%`。

逐轮全量审计发现：

- confirmed 规则额外保留 `0.082/0.070 node/round`，最终进入提交路径的只有
  `0.017/0.013 node/round`，所以它没有形成足够的有效松弛；
- relative 位于 `0.01–0.05` 的拒绝候选只有 `0.122/0.106 node/round`；
- relative `<0.01` 的拒绝达到 `1.639/1.775 node/round`，但目标概率中位数仅
  `7.6e-14/1.2e-16`，大量额外节点实际上是目标强反对的分支；
- 基于旧日志可见节点的保守 replay 表明，`coverage>=0.001`、
  `relative>=0.001`、balanced `beta=5` 的 MAL 下界约为 `3.273/3.179`。

据此新增 `scripts/run_fastmtp_empirical_relaxation.sh`。默认 `PROFILE=mal_first`
固定 D=3 和原 soft-reach 构树，只把 target coverage/relative 下限降到 `0.001`
并把联合路径评分的长度权重升到 `beta=5`；它仍综合置信度和长度，不强制最长。
同时提供 `plausible`（0.01/0.01/beta=3）与只测上限的 `ceiling` 档。由于直接
存活规则已足够宽，关闭 stochastic frontier rescue，避免混入第二套接受机制。

修改文件：

- `scripts/run_fastmtp_empirical_relaxation.sh`；
- `scripts/run_fastmtp_reach_first_tree.sh`（允许压力实验显式关闭 frontier rescue）；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证计划：执行 shell 语法与 `--help` 检查；新 profile 尚未完成真实 GPU 运行，
不能预先声称 MAL 或质量收益。限制：`mal_first` 明确允许质量下降；若它仍无法
接近 Cactus MAL，则主要瓶颈是 proposal tree 的分支质量，而非验证阈值。

### 2026-08-10：Cactus-calibrated tree 松弛强度对齐

状态：`实现与单元测试完成；待用户运行 N=100 GPU 实验`

进一步对照实现后确认，旧树与 Cactus 的“松弛强度”不在同一尺度。Cactus
`delta=1` 先把候选概率提升为
`h=p+sqrt(2*p*(1-p))`，再以 `min(1,h/q)` 接受；旧树则把低于 relative
阈值的候选直接剪枝。confirmed N=100 被拒日志换算后，Cactus 对这些节点的额外
期望接受质量达到 `0.218/0.189 per round`，数量级与双方 MAL 差距接近。

实现 `support_mode=cactus`：GPU 并行计算每个节点的 boosted probability、
Cactus acceptance 和随机存活事件；路径评分信号使用
`relative^w * A_cactus^(1-w)`，默认 `w=0.25`，然后继续乘长度奖励。这样节点
存活强度与链式 Cactus 的 `delta=1` 一致，但最终路径不会只因长度更长而胜出。
EOS 保护、树节点上限和单次 target forward 均保持不变。由于兄弟候选具有不同
的临时 Cactus 分布，整个多路径选择仍是近似分布，必须同时报告质量。

新增运行入口 `scripts/run_fastmtp_cactus_calibrated_tree.sh`：固定 D=3、复用现有
N=100 baselines，默认 `delta=1`、target score weight `0.25`、`beta=1.5`，并
关闭重复的 frontier rescue。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_cactus_calibrated_tree.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：相关动态树、metrics 与 report 测试 `27 passed`；Python/shell 静态检查
通过；真实 FastMTP/vLLM/TREE_ATTN 64-token 冒烟完成，26 rounds 中记录 125 个
已评估节点、51 个 Cactus 存活节点、39 个最终选中节点，服务正常退出。该单提示
MAL=2.50 只证明执行与审计链路正确，不能预先声称 MAL 高于旧树或质量保持。

### 2026-08-10：Direct Cactus + Tree N=100 负结果归因

状态：`完整结果与逐轮日志分析完成；当前设计未达到 Chain Cactus MAL；后续主干救援方案待实现`

用户完成的 N=100 结果：Direct Cactus + Tree 在 GSM8K 为
`89.0% / MAL 3.251`，在 HumanEval 为 `67.0% / MAL 3.173`；Chain Cactus
分别为 `90.0% / 3.434` 和 `69.0% / 3.342`。因此把相同局部 Cactus 公式直接
应用到独立树节点，并未复现链式 Cactus 的整体接受长度。

逐轮审计结论：

- 路径评分仅在 `0.80%/0.74%` 的轮次主动缩短存活路径，MAL 损失均约 `0.008`，
  不是主要原因；
- all-zero deterministic top-Q 主路径自身只有 `MAL 2.835/2.761`；额外树分支
  实际贡献 `+0.415/+0.412`，说明树覆盖有效，但起始主干弱于 sampled-Q chain；
- 每轮 drafted `6.776/6.975` 个节点，真正位于存活祖先下并被验证器访问的只有
  `4.408/4.354` 个，约 35%–38% 节点被祖先拒绝连带作废；
- 替代分支约在 12% 轮次被选择。其后一轮 root 存活率为 `71.3%/65.8%`，低于
  主分支后的 `85.7%/83.8%`。该关联可能受困难上下文选择偏差影响，不能解释为
  严格因果，但提示侧分支会进入更难维持 MTP 对齐的轨迹。

核心设计错误是把“更多候选”实现为重新生成并重新选择整棵 top-Q 树，而不是在
标准 Chain Cactus sampled-Q 主干上增加候选。Cactus 的主干 token 来自完整 Q
随机采样并由标准 rejection sampler 处理；当前树采用 deterministic top-Q 路径、
对兄弟节点独立做候选特定 Cactus 存活采样，整体不具备相对 Chain Cactus 的单调
接受保证。

下一设计决定：保留 Chain Cactus 主干的候选、随机数与验证结果，只在主干第一次
拒绝的位置启用侧分支救援，即 `Cactus trunk + rejection-only tree rescue`。
主干已接受时侧分支不得替换它。当前 Direct Cactus + Tree 代码和结果保留为负例，
暂不删除。

受影响文件：

- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：解析 GSM8K/HumanEval 共 `37,761` 个 tree rounds，核对 summary、
tree_metrics、逐深度存活、主路径、最长存活路径、分支选择和相邻轮统计。未修改
运行时代码，未运行新的 GPU benchmark。限制：相邻轮关联未记录 request 边界且
存在上下文难度混杂，只用于定位风险，不作为因果结论。

### 2026-08-10：启动 Cactus-calibrated N=100 并新增 Cactus-guided 动态树

状态：`v1 N=100 正在运行；v2 实现与 CPU 测试完成；v2 尚未做 GPU 冒烟或 benchmark`

运行修复：8000 端口被本仓库遗留的 vLLM 动态树进程占用。只读核验确认 PID
`2058482` 使用 `FastMTPVerifiedDynamicTreeWorker`、FastMTP Tree10 和约 22 GB
显存后，以 `SIGTERM` 正常停止该进程；随后成功启动
`fastmtp_cactus_calibrated_tree_n100_v1`。服务健康检查通过并进入 GSM8K 生成，
实时日志位于 `logs/fastmtp_cactus_calibrated_tree_n100_v1/`。该实验尚未结束，
不能记录完整双数据集结论。其 GSM8K-100 已完成：accuracy `89.0%`、MAL
`3.251`、target nodes/round `6.776`、每轮 target forward `1`。同一已冻结基线
为 Native `90.0% / 3.040`、Cactus `90.0% / 3.434`、SpecCascade
`93.0% / 3.229`。因此 v1 的 GSM8K MAL 略高于 SpecCascade，但仍比 Cactus 低
`0.183`，且质量比 Native/Cactus 低 1 个百分点；HumanEval 仍在运行，不能据此
形成最终方法结论。

算法审计确认，`support_mode=cactus` 会对每个树节点直接执行链式 Cactus 强度的
随机存活，因此只在候选覆盖和路径选择上保留树方法差异。为避免主方法退化成
Cactus，新增 `support_mode=cactus_guided`：

- soft-reach 的 Q 驱动动态构树、全局节点上限和无硬层级配额保持不变；
- 正常节点继续使用目标 relative 阈值；
- 仅在存活父节点没有正常后继时启用 Cactus 概率；
- rescue 还需 coverage/EOS/目标概率门控，以及 P/Q、下一步 target-top1 命中或
  中等 relative 支持中的至少一种树证据；
- 每条路径最多一次 guided rescue，之后仍可沿正常目标支持节点继续；
- guided 候选使用 `relative^w * A_cactus^(1-w)`，推荐 `w=0.65`，最终路径仍按
  几何平均置信度和 `length^beta` 联合评分。

新增 N=100 入口 `scripts/run_fastmtp_cactus_guided_tree.sh`，默认 D=3、
`N_max=10`、`tau_relax=0.35`、coverage 0.05、Cactus delta 1、目标权重 0.65、
`beta=1.5`，复用原 Native/Cactus/SpecCascade 结果。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `tests/test_dynamic_mtp_tree.py`；
- `scripts/run_fastmtp_cactus_guided_tree.sh`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：`tests/test_dynamic_mtp_tree.py` 为 `27 passed`；加入 tree metrics 与 verified
report 后的相关测试集为 `31 passed`；相关 Python 文件 `py_compile` 通过；三个树
实验入口 `bash -n` 通过。新增测试覆盖 guided rescue、Cactus 概率较高但缺少
目标/树证据的拒绝、正常后继优先和每路径最多一次 rescue。

限制：v2 尚未加载到真实 vLLM/GPU，不能声称运行链路或 MAL/准确率收益已经通过。
当前正在运行的 v1 进程加载的是修改前代码，不受本次 v2 文件改动影响。

### 2026-08-10：Direct Cactus+Tree 完成与双树松弛统一实验入口

状态：`direct v1 双数据集完成；统一脚本与报告器完成；guided v2 待 GPU 运行`

本条更新替代上一条中“v1/HumanEval 仍在运行”的临时状态。端口 8000 已释放，
没有残留 vLLM 或实验进程。`fastmtp_cactus_calibrated_tree_n100_v1` 的最终结果：

| 数据集 | Direct Cactus+Tree 质量 | MAL | Native 质量/MAL | Cactus 质量/MAL | Cascade 质量/MAL |
|---|---:|---:|---:|---:|---:|
| GSM8K-100 | 89.0% | 3.251 | 90.0% / 3.040 | 90.0% / 3.434 | 93.0% / 3.229 |
| HumanEval-100 | 67.0% | 3.173 | 61.0% / 3.065 | 69.0% / 3.342 | 66.0% / 3.143 |

Direct 版本每轮 target forward 为 1，平均节点数为 `6.776/6.975`。它的 MAL 在
两个任务上均略高于 SpecCascade，但仍比链式 Cactus 低 `0.183/0.169`。该版本
逐节点执行 Cactus 存活，因此正式语义固定为 `Cactus+Tree ablation`，不作为我们
目标主导树方法的最终定义。

新增 `scripts/run_fastmtp_tree_relaxation_ablation.sh`，串行执行或恢复：

1. direct Cactus+Tree；
2. target-dominant `cactus_guided` tree（ours）。

默认每个数据集 100 条、逐样本打印、复用冻结的 Native/Cactus/SpecCascade
基线。可通过 `DIRECT_RUN_TAG=fastmtp_cactus_calibrated_tree_n100_v1` 直接复用已
完成的 direct 结果，只运行 guided 部分。新增报告器会校验两组 dynamic tree 的
协议字段和 sample manifest，并将五个方法汇总为同一个 Markdown/JSON。

修改文件：

- `scripts/run_fastmtp_tree_relaxation_ablation.sh`；
- `remtp/fastmtp_tree_relaxation_report.py`；
- `tests/test_fastmtp_tree_relaxation_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

为防止恢复运行时混淆两种语义，统一脚本新增运行时变体核验：direct 和 guided
必须使用不同 RUN_TAG；每个子实验结束后从 `dynamic_tree_server.log` 检查实际
`support_mode=cactus` 或 `support_mode=cactus_guided`，再写入本地
`tree_variant.json`。统一报告器拒绝缺少或模式不匹配的变体标记。

验证：动态树、统一报告、原 verified report 和 tree metrics 测试合计
`34 passed`；Python `py_compile`、三个运行入口 `bash -n` 和统一脚本 `--help`
通过；报告器早期版本已对真实 direct v1 结果目录完成一次只读生成检查；当前
direct server log 的 `support_mode=cactus` 标记也已只读确认。

限制：guided v2 尚未进行真实 GPU 冒烟或 N=100 benchmark，不能预先宣称它的
MAL 或质量优于 direct 版本。统一脚本运行 guided 时会重新加载修改后的 Worker。

### 2026-08-10：Direct Cactus+Tree 逐路径后验审计

状态：`只读分析完成；确认多候选有正收益但当前树不是 Chain Cactus 的单调扩展`

在 direct v1 完整结果基础上进一步解析 `37,761` 个 tree rounds。路径评分只在
`0.80%/0.74%` 的轮次缩短最长存活路径，造成的 MAL 损失约 `0.008/0.008`；
因此提高 beta 或强制最长不能解释与 Chain Cactus 的 `0.183/0.169` 差距。

all-zero top-Q 主路径 MAL 仅为 `2.835/2.761`，动态分支将其提高到
`3.251/3.173`，实际分支收益为 `+0.415/+0.412`。树并非无效；问题是 direct
实现用 deterministic top-Q 树替换了 Chain Cactus 的 sampled-Q 主干。每轮
`6.776/6.975` 个 drafted 节点中，只有 `4.408/4.354` 位于存活祖先之下，
其余约 35%–38% 随父节点拒绝作废。

替代分支之后的下一轮 root 存活率为 `71.3%/65.8%`，低于主分支后的
`85.7%/83.8%`。该统计存在困难上下文选择偏差，只能说明替代路径集中在低对齐
状态，不能证明替代路径造成后续下降。

设计区分：现有 `cactus_guided` 是目标规则死前沿上的受控救援，已经实现但尚未
GPU 验证；它仍不等同于严格保留 Chain Cactus sampled-Q 主干。若目标是让“更多
候选”相对 Chain Cactus 具有逐轮非负的接受机会，后续需要另行实现
`Cactus trunk + rejection-only tree rescue`：主干候选、共享随机数和已接受结果
完全复用 Chain Cactus，侧分支仅在主干首次拒绝时接管。

受影响文件：

- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：只读核对 summary、tree_metrics 和逐轮 JSONL；未修改算法代码，未运行新
GPU 实验。限制：日志没有显式 request boundary，相邻轮分析含最多 99 个跨请求
边界，且不能排除上下文难度混杂。

### 2026-08-10：Cactus-guided wide v3 双救援实验档位

状态：`实现、本地测试与短请求 GPU 冒烟通过；N=100 benchmark 尚未运行`

在不替换现有动态树和正常目标相对验证的前提下，新增更宽的死前沿救援档位。
默认 `cactus_guided` 仍保持每路径最多一次救援，以保证历史 v2 结果和配置语义
不变；新增 `max_guided_rescues_per_path` 配置允许实验档位显式设置为 2。救援计数
沿选中父子路径继承，正常存活不增加计数，只有成功的 Cactus-guided rescue 才
增加计数。第二次救援也只能发生在后续前沿没有任何正常存活候选时。

新增 v3 入口将正常 relative 阈值设为 `0.25`、coverage 下限 `0.01`、guided
relative 下限 `0.01`、`P/Q` 证据阈值 `0.25`、Cactus delta `1`、目标/Cactus
排序权重各 `0.5`、每路径最多 2 次救援和 `beta=2.0`。最低目标概率 `0.001`、
EOS 保护、D=3、soft-reach、最多 10 节点、无硬层级配额及每轮一次 target
forward 均保持不变。

受影响文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_cactus_guided_wide_tree.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

新增审计字段 `guided_rescue_count_before` 和服务启动标记
`max_guided_rescues`。测试新增“默认最多一次”向后兼容检查之外的“双次救援成功”
用例。相关单元与报告测试共 `35 passed`；Python `py_compile`、四个 Shell 入口
`bash -n` 及新脚本 `--help` 通过。

真实 FastMTP/vLLM/TREE_ATTN 短请求生成 64 token 成功：服务标记确认 D=3、
N_max=10、`support_mode=cactus_guided`、`tau_relax=0.25`、
`max_guided_rescues=2`、`beta=2` 和每轮一次 target forward；31 个审计 round 中
出现 2 次 guided rescue，并且下游节点记录到 `guided_rescue_count_before=1`。
该单提示 MAL 为 `2.032`，只证明执行链路和审计字段有效，不能作为 benchmark
效果结论。

冒烟时还发现旧独立入口 `serve_fastmtp_dynamic_tree.sh` 未设置 TREE_ATTN 且只按
depth 分配 speculative slots，首个请求报 `dynamic MTP tree requires TREE_ATTN
metadata`。该入口已改为委托给与正式实验相同的 audited verified server，避免
手动服务和正式脚本使用不同运行语义。

限制：尚未运行 N=100，不能声称 MAL 已达到 Cactus 或准确率保持不变；该算法仍
改变采样分布，并且更宽参数可能降低任务质量，必须用相同样本和基线联合评估。

### 2026-08-10：Cactus-guided wide v3 完成与 MAL 停滞归因

状态：`N=100 双数据集完成；该阈值/双救援方向停止继续放宽`

v3 最终结果为 GSM8K `88.0% / MAL 3.172`、HumanEval `64.0% / MAL 3.133`；
同一冻结 Chain Cactus 为 `90.0% / 3.434` 和 `69.0% / 3.342`。相对 Direct
Cactus+Tree，v3 MAL 还下降 `0.079/0.040`，质量下降 `1/3` 个百分点。100 条任务
的质量差异仍有抽样和随机生成波动，但 MAL 来自约 1.8 万/1.9 万 round，停滞不是
单纯报告精度问题。

逐轮只读审计发现：新开放的 relative `[0.25,0.35)` 区间只有 GSM8K 1 个正常
存活节点、HumanEval 0 个，且没有进入最终路径；第二次 guided rescue 仅发生
`10/1` 次；最终选路从未舍弃更长存活路径。最终选择 rescue 的 round 比例仅为
`3.37%/2.53%`，其反事实 MAL 贡献上界约为 `0.060/0.045`。每轮虽构造约 7 个
节点，但仅约 59% 位于存活祖先之下，useful-node ratio 约为 31%/30%。因此降低
normal threshold、提高 beta 和允许第二次救援三项修改都几乎没有可作用的事件。

决策：停止继续调低当前树的 relative 阈值或增加 guided rescue 次数。若继续树
方向，应把 Chain Cactus 的 sampled-Q 主干、共享随机数和逐位置接受语义原样保留，
仅在主干首次拒绝时用侧分支接管；现有 Direct/Guided 树均保留为负结果。

受影响文件：

- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：只读解析 v3 的两个 `comparison.json`、`tree_metrics.json` 和总计 37,317
个 `tree_rounds.jsonl` round，并与 balanced-wide、confirmed-support、Direct
Cactus+Tree 和冻结 Chain Cactus 结果交叉核对。没有修改算法或运行新 GPU 实验。
限制：日志没有显式 request boundary，无法把每次救援严格归因到单个任务的最终
正确性；跨方法生成轨迹不同，反事实救援贡献是上界而非严格配对估计。

### 2026-08-11：sampled-Q Cactus 主干与首次拒绝树救援

状态：`实现、CPU 测试和真实 FastMTP/vLLM/TREE_ATTN 冒烟通过；正式 benchmark 待用户运行`

针对近几版放宽阈值但 MAL 基本不变的问题，停止继续修改原 deterministic top-Q
动态树的 relative 阈值。新增独立模式 `support_mode=cactus_trunk_rescue`：

1. 每个深度用与链式 probabilistic MTP 相同的 exponential-race primitive 从完整
   Q 采样一个主干 token；
2. 每层额外保留至多两个高 Q 兄弟，但只递归展开主干，默认 D=3、节点上限 9；
3. 主干逐位置使用与 Chain Cactus 相同的候选特定 h、`min(1,h(y)/q(y))` 和
   `(h-q)+` correction residual；
4. 只有主干第一次拒绝时才查看同父兄弟，按 residual target support 和
   residual-Cactus acceptance 选择一个候选；
5. 救援成功提交已接受主干、救援 token 和该节点后的原始 target bonus；救援失败
   则从原 Cactus residual 采样 correction；
6. 主干通过时兄弟不参与路径竞争，避免树选择器改变 Chain Cactus 已接受轨迹。

该模式是“Cactus 主干 + 增量树救援”，不再用整棵树替换链式主干。主干概率语义
与 Cactus 对齐，但救援候选由 Q 确定性保留，并非来自严格联合 multi-proposal
采样，因此整体仍是近似松弛解码，不能宣称保持原目标或 Cactus 输出分布。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：新增验证器、residual rescue、EOS 保护和诊断字段；
- `remtp/dynamic_tree_vllm.py`：新增 sampled-Q 主干、毛毛虫动态拓扑、分支 KV
  物化与运行时 dispatch；
- `scripts/run_fastmtp_cactus_trunk_rescue_tree.sh`：新增 20/100 条一键实验入口；
  样本数匹配时复用冻结基线，不匹配时自动重跑三组基线以保持同表协议一致；
- `scripts/run_fastmtp_reach_first_tree.sh`：根据新 support mode 打印准确的 verifier
  与 node cap；
- `scripts/serve_fastmtp_verified.sh`：新增准确的 verify mode 和启动语义标记；
- `tests/test_dynamic_mtp_tree.py`：覆盖主干全接受、拒绝救援和 residual fallback，
  并给两个既有随机 fallback 测试补显式 generator，消除测试顺序依赖；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：`py_compile` 通过；新运行脚本和服务脚本 `bash -n` 通过；动态树、Cactus、
Fixed-6 primitive 和 tree audit 相关测试共 `46 passed`。真实 RTX 4090 冒烟加载
TencentBAC/FastMTP、TREE_ATTN、D=3、N_max=9 和新 Worker 后成功生成 48 tokens：
24 rounds 均为一次 target forward，实际节点数 5–9；3 次 trunk bonus、16 次
Cactus correction、5 次 rejection rescue bonus；无 cache/state/TreeAttention
错误，服务已正常停止。冒烟单提示 MAL `2.083` 仅用于链路验证。

剩余限制：尚未运行相同任务清单的 GSM8K/HumanEval 正式对比，不能声称 MAL 或
质量优于 Chain Cactus；新树每轮 target nodes 多于三 token 链，吞吐可能下降；
rescue 会改变输出分布，必须先看 20 条 pilot 的 accuracy/pass@1 与 MAL，再决定
是否运行 100 条。

### 2026-08-11：trunk-rescue pilot 失败与 tree target 双重温度修复

状态：`20 条失败结果完成诊断；CPU 修复、回归测试和修复后 GPU smoke 通过；pilot 待重跑`

用户完成 `fastmtp_cactus_trunk_rescue_pilot`。修复前原始结果为：

| 数据集 | Native | Chain Cactus | SpecCascade | trunk-rescue tree |
|---|---:|---:|---:|---:|
| GSM8K-20 质量/MAL | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 90.0% / 2.638 |
| HumanEval-20 质量/MAL | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 75.0% / 2.252 |

树方法 e2e 仅 `89.862/77.445 tok/s`，相对 Native 为 `0.707x/0.605x`；实际
target nodes/round 为 `4.621/4.401`。逐轮解析发现主干 root 接受率仅
`59.9%/47.6%`；目标候选 P(y) 小于 `1e-6` 的比例为 `24.2%/36.5%`，大于
`0.999` 的比例为 `52.5%/51.0%`，与 Chain Cactus 日志中约 `92%–95%` 的首位置
接受率明显矛盾。救援分别成功 `1096/735` rounds，但只能补一个 draft token，
无法弥补主干退化。

代码审计定位到 `remtp.fixed6_vllm._strict_tree_sampler`：它先调用 vLLM
`apply_sampling_constraints`，该函数已经执行 temperature scaling 和 top-k/top-p；
随后又把 processed logits 传给 `_temperature_probs`，导致第二次除以温度。
实验 `T=0.6` 的 tree target 因而约等价于 `T=0.36`，而 draft Q、Native、Cactus
和 SpecCascade 均保持 `T=0.6`。此前所有通过该 sampler 完成的随机 tree benchmark
均不再是公平对比，保留结果只作历史调试；贪心和状态隔离测试不因该错误失效。

修复内容：新增 `_processed_target_probs`。随机 sampling constraints 之后只执行
softmax；greedy 请求仍转换为 target top-1 one-hot。新增两项回归测试，一项验证
结果严格等于 `softmax(raw/T)` 且不同于 `softmax(raw/T/T)`，另一项验证 temperature
0 的 one-hot 行为。

受影响文件：

- `remtp/fixed6_vllm.py`；
- `tests/test_fixed6_microtree.py`；
- `docs/mimo_dynamic_mtp_tree.md`（添加旧随机树结果失效通知和 pilot 诊断）；
- `docs/research_and_development_log.md`。

验证：相关 dynamic tree、Cactus、Fixed-6 和 audit CPU 测试共 `48 passed`，Python
`py_compile` 通过。修复后真实 GPU smoke 使用相同通用提示、3 个 seed、每个 128
output tokens 分别运行 tree 与 Chain Cactus。Tree 为 183 rounds、MAL `2.082`，
Chain Cactus 服务窗口 MAL 约 `2.11–2.15`；tree 每轮 target forward 为 1，且无
TreeAttention/cache 错误。该通用提示不是正式任务，不能据此宣称质量或 MAL
收益；下一步应重跑同一 20 条协议，不应直接运行 N=100。

### 2026-08-11：目标逐节点最长路径与 Cactus 阈值 continuation rescue

状态：`历史 v1；20 条 pilot 已完成；验证语义已被同日 v2 单-token extension 取代`

修复温度后的 20 条 sampled-trunk pilot 为 GSM8K `95.0% / MAL 3.044`、HumanEval
`80.0% / MAL 3.070`，质量未暴露退化，但 MAL 仍低于 Chain Cactus 的
`3.509/3.361`。审计确认 sampled trunk 接受时旁支全部 `BACKUP_UNUSED`，第一次
拒绝后的旁支只能替代一个 correction，不能沿旁支继续；因此实际验证节点增加但
完整候选路径没有增加。

新增 `support_mode=target_path_rescue`，其语义为：

1. 使用原 dynamic-Q proposer 构造真正的多分支树，不再使用 sampled trunk
   caterpillar；
2. target 一次 TreeAttention forward 后，每个节点使用其父前缀对应的原始 target
   分布，按 `P(y)/max(P) >= tau_normal` 普通存活；
3. 所有普通存活兄弟都可继续到下一深度，coverage 仅作诊断；
4. 未到最深层时，每个父节点可额外保留一个普通失败孩子。救援分数为 target
   relative support 与 candidate-specific Cactus acceptance 的加权几何平均，
   需超过显式阈值，并受最低 target support、EOS 和每路径一次救援约束；
5. 救援为确定性 continuation 资格，不消耗随机数，也不会直接保证提交；
6. 最终先选择最大连续存活深度，同长度再最大化原始 target relative support
   的几何平均；Cactus 分数不进入最终可信度；
7. 每轮仍提交叶后原始 target bonus，target forward 次数保持 1。

默认 pilot 参数为 D=3、`N_max=9`、每父最多 3 children、普通 relative 阈值
`0.20`、救援 score 阈值 `0.18`、relative floor `0.02`、target probability floor
`1e-4`、Cactus delta `1.0`、target weight `0.65`、每路径最多一次救援。节点数是
上限而非强制填满。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：新增模式、阈值、逐节点多分支存活、确定性
  continuation rescue、最长优先和 target-confidence tie break；
- `remtp/dynamic_tree_vllm.py`：读取/打印新增阈值并使用原 dynamic-Q proposer；
- `remtp/tree_audit.py`：实时日志支持 deterministic score/threshold rescue；
- `scripts/run_fastmtp_target_path_rescue_tree.sh`：新增一键 20/100 条入口；
- `scripts/run_fastmtp_reach_first_tree.sh`：允许调用方选择 longest/balanced；
- `scripts/run_fastmtp_verified_comparison.sh`、`scripts/serve_fastmtp_verified.sh`、
  `scripts/serve_fastmtp_dynamic_tree.sh`：传递新参数和准确运行时标记；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`、`docs/research_and_development_log.md`。

验证：Python compile 与 5 个 shell 入口 `bash -n` 通过；dynamic-tree、Fixed-6、
Cactus、report 和 tree-audit 相关测试 `57 passed`。真实 RTX 4090 冒烟成功加载 TencentBAC/FastMTP、
TREE_ATTN、D=3、N_max=9，完成三个 temperature=0.6、96-output-token 请求，共
89 rounds；每轮 target forward 为 `1.000`，实际 target nodes 为 3--9（均值
6.674），无 KV/cache/TreeAttention 错误，服务正常停止。冒烟 MAL `3.247`；6 个
救援节点获得继续资格但 0 个进入最终路径，因为后继没有形成最大深度路径。

剩余限制：尚未运行同一任务清单的 GSM8K/HumanEval pilot，不能声称质量或 MAL
优于 Native、Chain Cactus 或 SpecCascade；该方法是确定性树路径选择和近似松弛，
不保持目标分布；树 target nodes 多于三 token 链，吞吐可能下降。实验 artifacts
仅保留在 ignored `results/target_path_rescue_gpu_smoke/`。

### 2026-08-11：target-path rescue 改为选路后的单-token extension（v2）

状态：`实现完成；CPU 回归和真实 FastMTP/vLLM/TREE_ATTN 分支冒烟通过；默认阈值 20 条质量实验待运行`

用户完成 v1 的 20 条实验后指出，普通节点与 target 比较本身已经是松弛验证，
“救援”应当是其后的独立一次延长。v1 的 GSM8K/HumanEval 质量/MAL 分别为
`95.0%/3.173` 与 `75.0%/3.132`；虽然分别有 `284/183` 个候选达到救援资格，
最终仅有 `76/65` 个救援 token 被选中，直接 MAL 增益只有 `+0.024/+0.020`。
旧语义只是让节点重新参加整树竞争，获救节点可能输给其他路径，因而日志看不出
稳定延长效果。

v2 保持普通松弛阶段不变：逐节点按各自父前缀的 `P(y)/max(P)` 判断存活，先选择
普通存活深度最长的基础路径，同长度再按 target-relative 几何平均决胜。之后新增
独立 extension 阶段：只有基础路径未达到最大深度时，才检查它的直接前沿；从普通
失败孩子中按 `relative^w * A_cactus^(1-w)` 选择达到阈值的最高分候选，并直接追加
恰好一个 token。extension 不再进入全树路径竞争，也不会解锁第二个 token；target
probability floor、EOS 保护和每轮一次上限保留。

报告新增普通接受 draft 数/深度、extension token 数及其直接 MAL 贡献。旧日志的
反事实 replay 预测：救援阈值 `0.18` 的增益约为 `+0.022/+0.019`，降到 `0.08`
约为 `+0.050/+0.034`，因此一键 pilot 的默认值改为 `0.08`。这只是离线阈值依据，
不是新方法质量结论。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：实现普通选路后的一次前沿 extension；
- `remtp/mimo_tree_report.py`：拆分 ordinary depth 和 rescue extension MAL；
- `remtp/fastmtp_verified_report.py`：在总对比表直接显示 ordinary depth 与 extension MAL；
- `scripts/run_fastmtp_target_path_rescue_tree.sh`：更新两阶段说明与默认阈值；
- `scripts/serve_fastmtp_verified.sh`：运行时 marker 明确标记为选路后单-token extension；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_mimo_tree_report.py`；
- `tests/test_fastmtp_verified_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`：给 v1 添加 superseded 通知并记录 v2；
- `docs/research_and_development_log.md`。

验证：Python compile、shell syntax 和 dynamic-tree/report/tree-audit/Fixed-6 相关
CPU 测试为 `52 passed`。
真实默认阈值 GPU smoke 完成 89 rounds、每轮 target forward `1.000`，该单提示未
触发 extension；为验证代码分支而运行的强制零阈值 smoke 完成 109 rounds，其中
85 rounds 出现 extension，extension 节点全部标记 `SELECTED`，terminal 为
`target-path-extension-bonus`，每轮 target forward 仍为 `1.000`。服务正常停止。
强制零阈值仅验证执行语义，不是可用于任务质量的参数或 benchmark。

剩余限制：新默认 `0.08` 尚未运行 GSM8K/HumanEval 同任务清单，不能声称准确率、
pass@1 或 MAL 改善；v2 仍是改变输出分布的近似松弛方法；树 target node 开销仍
高于三 token 链。GPU artifacts 仅保留在 ignored
`results/target_path_extension_gpu_smoke/` 与
`results/target_path_extension_forced_gpu_smoke/`。

### 2026-08-11：前部多路径重开救援与普通后继延伸

状态：`实现、CPU 回归和真实 FastMTP/TREE_ATTN 冒烟通过；20 条质量 pilot 待用户运行`

用户完成 v2 的 20 条结果：GSM8K 为 `85.0% / MAL 3.181`，HumanEval 为
`80.0% / MAL 3.123`；相同冻结基线中 Chain Cactus 为 `90.0% / 3.509` 和
`75.0% / 3.361`，SpecCascade 为 `95.0% / 3.192` 和 `75.0% / 3.160`。v2 的
一次选路后 extension 仅贡献 `+0.043/+0.027` MAL，root selection 只有
`85.3%/83.8%`。因此瓶颈不是 extension 阈值完全没有触发，而是 root 失败后
extension 固定只补一个 token，无法继续利用同一 target tree forward 已验证的
后继。

新增实验模式 `support_mode=prefix_reopen_rescue`：

1. 每个节点仍只用自己父前缀下的 `P(y)/max(P)` 做普通松弛验证，不以整条 target
   概率乘积决定节点生死；
2. 每个可达路径最多使用一次确定性 rescue，所有达到阈值的普通失败兄弟均重开，
   不再每父节点只选一个；
3. 获救节点的后代可以继续执行普通验证，但继承已使用 rescue 的状态，不能再次
   跨越失败节点；
4. 默认 rescue score 阈值随深度为 `0.08/0.12/0.16`，优先将风险用于能够解锁
   更多后继的 root/前部位置；最低 relative `0.02`、最低 target probability
   `1e-4` 和 EOS 保护保持；
5. 最终最大连续深度优先，同长度用 target-relative 几何平均决胜；
6. 报告新增 `rescue_unlocked_tokens`、`rescue_unlocked_mal_gain` 和
   `pre_rescue_mean_accepted_depth`，区分旧 `+1` extension 与新模式实际解锁的
   多个所选 token。

同时澄清 per-position 指标语义：vLLM 输出的是累计 reach rate，MAL 应按
`1+r1+r2+r3` 求和，不能再次连乘。构树器中的累计 Q 乘积只作为 proposal
前缀 reach 和全局节点预算排序的代理，不进入 target 放行条件；恢复早期的 Q
几何平均会重新高估低 reach backup，故本次没有改回该规则。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：新 support mode、深度阈值、逐路径一次 rescue
  状态与普通后继重开；
- `remtp/dynamic_tree_vllm.py`：读取、打印 `rescue_depth_penalty`；
- `remtp/mimo_tree_report.py`：统计 rescue 解锁 token、pre-rescue depth 并澄清
  legacy extension 字段；
- `remtp/fastmtp_verified_report.py`：总表读取通用 rescue MAL；
- `scripts/serve_fastmtp_dynamic_tree.sh`、`scripts/serve_fastmtp_verified.sh`、
  `scripts/run_fastmtp_verified_comparison.sh`：传递新模式和运行参数；
- `scripts/run_fastmtp_prefix_reopen_rescue_tree.sh`：新增一键 20 条 pilot；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_mimo_tree_report.py`、
  `tests/test_fastmtp_verified_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`、`docs/research_and_development_log.md`。

验证：相关 Python `py_compile` 和四个 shell 入口 `bash -n` 通过；dynamic tree、
report、tree audit、Fixed-6 状态相关 CPU 测试共 `56 passed`。真实 RTX 4090
使用 TencentBAC/FastMTP、vLLM 0.18、TREE_ATTN、D=3、N_max=9 运行三个
temperature=0.6、96-output-token 相同提示请求，共 122 rounds；target forward
固定 `1.000/round`，无 KV/cache/TreeAttention 错误，MAL `2.361`，实际 rescue
解锁贡献 `+0.205`。同提示 v2 对照为 124 rounds、MAL `2.315`、extension 贡献
`+0.097`。两个服务均已正常停止。

限制：匹配提示 smoke 只验证执行链路和机制活性，输出轨迹导致 round 数略有不同，
不是配对 benchmark；尚未运行新模式的 GSM8K/HumanEval 质量实验，不能声称准确率
稳定或 MAL 已接近 Cactus。新模式仍改变输出分布，且每轮 target nodes 多于三
token 链，吞吐仍可能低于链式基线。实验 artifacts 仅保留在 ignored
`results/prefix_reopen_rescue_gpu_smoke/` 与
`results/target_path_extension_matched_smoke/`。

### 2026-08-11：HumanEval 跨轮漂移诊断与 target-margin 自适应救援

状态：`实现、CPU 回归和真实 FastMTP/TREE_ATTN 冒烟通过；同任务 20 条 pilot 待用户运行`

用户完成 `fastmtp_prefix_reopen_rescue_pilot`。原始结果：

| 数据集 | Native | Cactus | SpecCascade | prefix-reopen tree |
|---|---:|---:|---:|---:|
| GSM8K 质量/MAL | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 95.0% / 3.262 |
| HumanEval 质量/MAL | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 75.0% / 3.135 |

HumanEval pass@1 与三条链式基线相同，短板是 MAL。prefix-reopen 对
GSM8K/HumanEval 的 rescue 解锁贡献为 `+0.105/+0.090`，但 HumanEval 普通接受
深度由 v2 的 `2.096` 下降到 `2.045`。逐轮相邻日志显示，HumanEval 在 ordinary
path 后的下一轮 MAL 为 `3.058`，rescue path 后为 `2.627`；选择根 branch
0/1/2 后的下一轮 MAL 为 `3.100/2.796/2.281`。GSM8K 出现相同但较弱趋势。
请求边界没有显式记录，数字不能解释为严格因果效应，但它们共同反对“继续全局
降阈值”这一调整。

新增 target-margin 自适应救援。每个父前缀计算
`m=log P(top1)-log P(top2)`，救援阈值乘以
`1 + margin_penalty * min(m/margin_reference,1)`。新 pilot 默认基础阈值 `0.06`、
深度惩罚 `0.35`、margin reference `1.5`、margin penalty `1.0`、relative floor
`0.015`：target 犹豫时比旧版更松，target 明确时最多加倍阈值。该规则只使用
通用目标分布信号，不读取任务类型、代码 token 或 benchmark 标签。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：batched target top-2 margin、margin multiplier、
  自适应救援阈值与逐节点诊断；
- `remtp/dynamic_tree_vllm.py`：新增 margin reference/penalty 环境配置与启动标记；
- `remtp/mimo_tree_report.py`：报告所选救援的平均 target margin 和实际阈值倍率；
- `scripts/serve_fastmtp_dynamic_tree.sh`、`scripts/run_fastmtp_verified_comparison.sh`：
  传递新配置；
- `scripts/run_fastmtp_margin_adaptive_rescue_tree.sh`：新增同协议一键 pilot；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_mimo_tree_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`、`docs/research_and_development_log.md`。

验证：Python `py_compile`、四个 shell 入口 `bash -n` 和相关 dynamic-tree、report、
tree-audit、Fixed-6 回归共 `57 passed`。真实 RTX 4090、TencentBAC/FastMTP、
vLLM 0.18、TREE_ATTN 冒烟在两个代码提示和一个通用提示上完成 121 rounds；
target forward 固定 `1.000/round`，MAL `2.380`，rescue 解锁贡献 `+0.116`，启动
标记确认 `margin_reference=1.5`、`margin_penalty=1`，服务已停止。

限制：20 条任务只有一个 seed，HumanEval 单题翻转会改变 5 个百分点；相邻轮分析
混入请求边界与上下文难度选择效应。新 margin 参数尚未运行相同任务清单，不能
声称 pass@1 或 MAL 改善；树仍是近似松弛且 target nodes 多于三 token 链。GPU
artifact 仅保留在 ignored `results/margin_adaptive_rescue_gpu_smoke/`。

### 2026-08-11：停止 margin 调参，改为 sampled-primary 拒绝后树恢复

状态：`实现、CPU 回归和真实 FastMTP/TREE_ATTN 机制冒烟通过；20 条质量 pilot 待用户运行`

用户完成 `fastmtp_margin_adaptive_rescue_pilot`。GSM8K 为
`95.0% / MAL 3.263`，HumanEval 为 `70.0% / MAL 3.101`；相同基线的
SpecCascade HumanEval 为 `75.0% / 3.160`。GSM8K/HumanEval 的 selected rescue
只贡献 `+0.071/+0.074` MAL，且所选救援的平均 target log-margin 仍高达
`1.735/1.798`。相邻轮诊断中，HumanEval 普通路径后的下一轮 MAL 为 `3.015`，
救援路径后为 `2.639`；GSM8K 为 `3.233/2.805`。这些相关性不是严格反事实，但
重复确认旧 deterministic tree/path selector 会在救援后降低下一轮 MTP 对齐。
target-margin 版本因此标记为机制负结果，不再继续扩大/缩小其阈值。

新增 `support_mode=sampled_primary_reopen`：每个深度 local rank 0 从完整 MTP Q
采样，形成与 Chain Cactus 同语义的 primary trunk；primary 使用相同 Cactus
接受概率，正常接受时 backup 不参与选路。仅在第一次 primary rejection 时，
residual-supported sibling 可执行一次近似恢复；恢复成功后，其树后代只能继续
通过 target-relative 规则，不能二次 rescue；恢复失败严格回到原 Cactus residual
correction。九节点预算在剩余深度间均摊，并以 reach-first 次序尽量为三个 root
候选都保留后继，避免浅层宽度耗尽后续节点。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：新 support mode、Cactus 主干拒绝后的恢复子树扫描、
  EOS 保护、恢复解锁 token 诊断；
- `remtp/dynamic_tree_vllm.py`：sampled primary proposal、主干节点硬保留、跨深度
  节点预算均摊、运行时 proposal marker；
- `scripts/serve_fastmtp_verified.sh`：新增准确 verifier marker；
- `scripts/run_fastmtp_reach_first_tree.sh`：识别新 verifier 摘要；
- `scripts/run_fastmtp_sampled_primary_reopen_tree.sh`：新增匹配基线的一键 pilot；
- `tests/test_dynamic_mtp_tree.py`：新增拒绝、恢复及两层后继延伸回归；
- `docs/mimo_dynamic_mtp_tree.md`、`docs/research_and_development_log.md`。

验证：`python -m py_compile` 通过；两个 shell 入口 `bash -n` 通过；dynamic-tree、
tree metrics 和 FastMTP report 相关 CPU 测试 `44 passed`。真实 RTX 4090 使用
TencentBAC/FastMTP、vLLM 0.18、TREE_ATTN 完成 3 个非 benchmark 提示、83 rounds，
target forward 固定 `1.000/round`、target nodes 平均 `8.494`；15 轮进入
`sampled-primary-reopen-bonus`，恢复解锁 22 个 draft token，服务正常停止。

剩余限制：真实 smoke 的提示、生成长度和 benchmark 不匹配，MAL `2.313` 只证明
机制可执行，不能解释为质量或性能提升；新树恢复仍是近似分布；HumanEval/GSM8K
20 条同任务结果尚未运行。实验 artifacts 仅保留于 ignored
`results/sampled_primary_reopen_smoke.jsonl` 和
`results/sampled_primary_reopen_smoke_metrics/`。

### 2026-08-12：停止先选兄弟的近似救援，新增 exact-residual-hit 六路线套件

状态：`实现、CPU 回归与真实 FastMTP/TREE_ATTN 冒烟通过；正式质量实验待用户运行`

用户完成 `fastmtp_sampled_primary_reopen_pilot`。同一 20 条协议下：GSM8K 的
Native/Cactus/SpecCascade/tree 为 `90%/3.080`、`90%/3.509`、`95%/3.192`、
`70%/3.316`；HumanEval 为 `75%/3.110`、`75%/3.361`、`75%/3.160`、
`70%/3.098`。树在 HumanEval 的 pre-rescue depth 仅 `1.718`，救援贡献
`+0.380` MAL；GSM8K 为 `2.049/+0.267`。但救援尝试 938 次、接受 913 次，
约 `97.3%`，说明旧流程“先从兄弟中挑最优 token，再做一次宽松接受”几乎把
救援变成确定性改写。GSM8K 中四个 Native 正确请求被翻错，且出现重复 `!!!`、
数量级错误和截断；HumanEval 出现一个净 pass@1 损失。该方向因此标记为质量负
结果，不再继续调整 target-relative rescue 阈值。

新增 `sampled_primary_shadow`、`residual_hit_anchor`、
`residual_hit_strict` 和 `residual_hit_cactus`。共同语义是：primary token 从完整
FastMTP `Q` 采样并用 Chain Cactus 的 `H` 验证；第一次拒绝后，必须先从
`normalize(max(H-Q,0))` 采样精确 correction；只有 sibling token ID 与该
correction 完全相同，树才可复用已验证状态。shadow 永不复用；anchor 只追加一个
原始 target token；strict 沿独立采样的 branch-local Q 做标准 `P/Q` 验证；
cactus 沿该 Q 使用同一 Cactus 规则。branch-local RNG 由请求初始 seed、轮次和
路径派生，不推进 primary/request RNG。顺手修复 `_sample_q_token` 在提供 request
generator 时先额外调用一次无用 `exponential_()` 的问题。

新增 `residual_coverage` 节点分配：先覆盖 sampled primary 父前缀下的 correction
兄弟，再保留已存在 recovery branch 的 local-rank-0 Q 样本，最后才分配非 primary
额外 backup。统一脚本一次运行 shadow/anchor/strict/cactus 及 strict/cactus wide
六条路线；第一个 route 只运行一次 Native/Cactus/SpecCascade，之后通过结果目录
symlink 复用。合并报告额外给出 correction hit rate、全轮复用率和每次命中解锁
token 数。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：精确 residual-hit 四类 verifier、诊断字段和
  `residual_coverage` 分配次序；
- `remtp/dynamic_tree_vllm.py`：非 primary Q 的独立分支采样、四类 runtime dispatch、
  sampled-primary 构树与无用 RNG 调用修复；
- `remtp/fastmtp_residual_hit_report.py`：新增跨路线合并报告及 correction-hit 审计；
- `scripts/serve_fastmtp_verified.sh`、`scripts/run_fastmtp_reach_first_tree.sh`：新增
  mode marker/摘要并修复 allocator 被硬编码成 `soft_reach` 的问题；
- `scripts/run_fastmtp_residual_hit_suite.sh`：新增一键六路线、共享 baseline、实时日志；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_fastmtp_residual_hit_report.py`：新增语义、
  分配和报告测试；
- `docs/exact_residual_hit_tree.md`、`docs/mimo_dynamic_mtp_tree.md`、
  `docs/research_and_development_log.md`。

验证：三个 Python 文件 `py_compile`、三个 shell 入口 `bash -n` 通过；相关测试
`49 passed`。真实 RTX 4090、TencentBAC/FastMTP、vLLM 0.18、TREE_ATTN 冒烟先后
验证 9-node reach-first 和 15-node residual-coverage；后者生成 32 tokens、14 rounds，
target forward 始终 `1/round`、实际 target nodes 平均 `12.0`，审计观察到一次
depth-2 exact correction hit，随后严格接受两个 recovery branch token，终止为
`residual-hit-strict-continuation-bonus`。服务均正常停止。

剩余限制：GPU smoke 使用单个非 benchmark 提示，只证明执行与审计语义；六路线尚
未跑 GSM8K/HumanEval，不能声称质量或 MAL 改善。15/20 节点路线可能显著降低吞吐，
wide 仅用于判断候选覆盖上限。shadow 与 Chain Cactus 的正式配对等价性仍需在统一
benchmark/seed 下审计。实验 artifacts 仅保留于 ignored
`results/residual_hit_gpu_smoke.jsonl` 与
`results/residual_hit_gpu_smoke_n15.jsonl`。

### 2026-08-12：100 条套件中断审计、请求级断点续跑与 shadow 负控制

状态：`断点续跑已实现并通过 CPU 测试；最后一个 HumanEval 待用户恢复；方法控制未通过`

`fastmtp_residual_hit_n100` 在 `cactus_wide` HumanEval 第 `75/100` 条后因会话断开。
现场检查确认没有残留 vLLM/benchmark 进程且 8000 端口空闲。shadow、anchor、strict、
cactus、strict_wide 两个数据集均完整，cactus_wide GSM8K 完整；只有
`cactus_wide/humaneval/dynamic_tree` 缺少 summary/requests。旧 HumanEval benchmark
将所有 records 保存在内存并在最后统一写 `requests.jsonl`，因此已经生成的 75 条只
能从 stdout 确认任务 ID/token 数，无法可靠重建候选代码或质量；本次需重跑最后
HumanEval 100 条，但此前约五个半 route 均直接复用。

新增通用 `requests.checkpoint.jsonl`：GSM8K/HumanEval 每完成一条请求立即 append、
flush 和 `fsync`。`--resume` 会校验现有 sample manifest 完全相同，并要求 checkpoint
严格构成 manifest 前缀；更换任务、seed 或 manifest 时拒绝混合。FastMTP comparison
新增 `RESUME_PARTIAL=1`：完整 summary 自动跳过，partial dataset 从 checkpoint 下一个
样本继续。动态树同时保存跨服务器会话的 `tree_rounds.checkpoint.jsonl`；如果像本次
旧运行一样没有 request checkpoint，则把无法配对的旧 audit 移到带
`unrecoverable_<timestamp>` 的历史文件，避免误算进重跑指标。server 历史日志同样
改名保留而非覆盖。残差六路线入口默认开启该恢复模式。

已完成数据还暴露出更重要的负控制。统一 100 条的原始结果为：

| 方法 | GSM8K 质量/MAL | HumanEval 质量/MAL |
|---|---:|---:|
| Native | 90.0% / 3.040 | 61.0% / 3.065 |
| Chain Cactus | 90.0% / 3.434 | 69.0% / 3.342 |
| SpecCascade | 93.0% / 3.229 | 66.0% / 3.143 |
| shadow | 87.0% / 3.079 | 67.0% / 2.979 |
| residual anchor | 87.0% / 3.094 | 62.0% / 2.977 |
| residual strict | 78.0% / 3.268 | 64.0% / 2.812 |
| residual Cactus continuation | 84.0% / 3.230 | 65.0% / 2.945 |
| strict wide | 87.0% / 2.951 | 61.0% / 2.871 |
| Cactus wide | 86.0% / 2.996 | 未完成 |

shadow 理应在树不参与输出时复现 Chain Cactus，但 MAL 分别低 `0.355/0.363`，因此
控制失败，不可把后续路线的质量/MAL 差异只归因于 residual hit。动态树 primary 的
条件接受率约为 shadow GSM8K `0.827/0.836/0.810`、HumanEval
`0.810/0.814/0.773`，明显不足以复现 Chain Cactus 的累计 per-position 曲线。wide
将 rejection correction hit rate 从约 `0.38--0.43` 提到约 `0.54--0.56`，但验证
节点升至 `19.6/round` 且 MAL 反而下降，说明单纯增加树宽已经是明确负方向。目前
最优先事项不是继续放松或增加节点，而是逐轮对齐 Chain Cactus 与 shadow 的 primary
Q、父前缀 target P、随机数消费顺序和 branch/cache 状态。

受影响文件：

- `remtp/benchmark_checkpoint.py`：新增 manifest-safe 请求级 durable checkpoint；
- `remtp/gsm8k_benchmark.py`、`remtp/humaneval_benchmark.py`：新增 `--resume`、逐请求
  checkpoint 和前缀跳过；
- `scripts/run_fastmtp_verified_comparison.sh`：新增 partial dataset、server 日志和动态树
  audit 的安全恢复/归并；
- `scripts/run_fastmtp_residual_hit_suite.sh`：默认开启恢复并补充帮助；
- `tests/test_benchmark_checkpoint.py`：checkpoint 前缀和 manifest 冲突测试；
- `docs/exact_residual_hit_tree.md`、`docs/research_and_development_log.md`。

验证：三个 Python 文件 `py_compile`、两个 shell 脚本 `bash -n` 通过；checkpoint、
GSM8K/HumanEval helper、动态树 verifier 和报告相关测试共 `64 passed`。没有声称恢复
后的 GPU benchmark 已通过；最后一个 HumanEval 仍需用户以相同 RUN_TAG 实际恢复。

限制：当前运行的前 75 条没有 candidate checkpoint，不能从第 76 条接续；恢复时会
从该 partial HumanEval 的第 1 条重跑。结果只有单 seed，且 shadow 控制失败意味着
exact-residual tree 尚不具备正式方法比较资格。所有结果和中断 audit 继续仅保留于
ignored `results/`、`logs/`。

### 2026-08-12：20 题树＋松弛定向优化达到小样本门槛

状态：`固定 20 题/单 seed 目标已达到；仅为 pilot，不代表正式结论`

先汇总同一 20 题 manifest 下的历史结果。此前最接近的
`prefix_reopen_rescue` 为 GSM8K `95.0% / MAL 3.262`、HumanEval
`75.0% / MAL 3.135`；后者比 SpecCascade `3.160` 少约 `0.026`。日志显示 HumanEval
有 244 个 rescue score 位于旧阈值的 50%--100%，但统一放平深度阈值后虽然 rescue
MAL 从 `0.090` 增至 `0.117`，普通深度同时下降，最终 MAL 降为 `3.131`。这确认额外
松弛会通过真实前缀破坏后续 MTP 对齐，不能用“更多 rescue 事件”代替端到端 MAL。

新增 continuation-certified rescue：只有第 2 层失败候选已经拥有一个按原普通规则
可接受的第 3 层直接后继时，才允许对其 rescue threshold 施加小折扣。该证书完全来自
同一次 tree target forward 已计算的正确父前缀分布，不增加 target 调用，不使用跨轮
状态。较大的 33.3% 折扣仅得到 GSM8K `90.0% / 3.211`、HumanEval
`75.0% / 3.101`，说明后继证书也不能支持大幅放松。

候选覆盖审计发现 9-node soft-reach 在 HumanEval 的 56.5% 轮次填满上限，每轮平均
仍有约 5.1 个构树合格节点未保留。但固定扩到 10/12 nodes 非单调：soft-reach 10
nodes 为 `85.0% / 3.237` 与 `70.0% / 3.113`；12 nodes 为
`85.0% / 3.177` 与 `85.0% / 3.160`。对 12-node 日志进一步审计发现，最终实际采用的
额外节点约 85% 是 local rank-0、约 90% 位于 depth 3、97% 以上直接通过普通验证。
因此缺口主要是分配器把预算过早用于横向 sibling，而不是验证阈值过紧。

将 allocator 改为已有的 `reach_first` 后，9 nodes 达到 GSM8K
`95.0% / 3.216`、HumanEval `80.0% / 3.146`。10 nodes 的 HumanEval 质量降到
65%，再次说明固定增加一个节点也不稳定。12-node reach-first 在不加后继折扣时为
GSM8K `95.0% / 3.300`、HumanEval `75.0% / 3.160095`；HumanEval 比级联
`3.160455` 仅少约 `0.00036`。最终只对有普通后继证书的 depth-2 rescue 使用 5%
折扣，固定 20 题结果为：

| 数据集 | Native 质量/MAL | SpecCascade 质量/MAL | 选定树＋松弛 质量/MAL |
|---|---:|---:|---:|
| GSM8K | 90.0% / 3.080 | 95.0% / 3.192 | **90.0% / 3.319** |
| HumanEval | 75.0% / 3.110 | 75.0% / 3.160 | **75.0% / 3.176** |

故在该固定小样本上，两项质量均与 Native 相等，MAL 均严格高于 SpecCascade。选定
参数为 D=3、N_max=12、children_max=3、`reach_first`、普通 relative threshold
0.18、rescue base 0.08、depth penalty 0.5、depth-2 continuation discount 0.05、
longest-first 路径选择。同一次运行的 nodes/round 为 GSM8K 8.425、HumanEval
8.363；吞吐仍低于链基线，本阶段只验证质量/MAL。

同时实现了一个 draft-entropy 自适应 9→12 节点上限实验接口。高熵阈值 0.05 在
GSM8K 达到 `95.0% / 3.339`，但 HumanEval 只有 `75.0% / 3.111`，未入选。该接口
保留为消融，不应描述为主方法。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：continuation-certified rescue 字段/语义；draft-entropy
  自适应 node limit helper；
- `remtp/dynamic_tree_vllm.py`：环境参数接入、按层 MTP entropy 计算 active node cap；
- `scripts/run_fastmtp_verified_comparison.sh`、`scripts/serve_fastmtp_verified.sh`、
  `scripts/serve_fastmtp_dynamic_tree.sh`：新参数透传与运行摘要；
- `scripts/run_fastmtp_prefix_reopen_rescue_tree.sh`：允许覆盖 allocator/path selector；
- `scripts/run_fastmtp_selected_tree_20.sh`：固化选定 20 题配置；
- `tests/test_dynamic_mtp_tree.py`：后继证书折扣与自适应节点上限单元测试；
- `docs/research_and_development_log.md`。

验证：`python -m py_compile` 两个 Python runtime 文件通过，四个 shell 入口
`bash -n` 通过；`tests/test_dynamic_mtp_tree.py` 为 `47 passed`。真实 RTX 4090、
TencentBAC/FastMTP、vLLM 0.18 上实际完成上述 GSM8K 20 + HumanEval 20 运行与隔离
HumanEval 测试；最终本地报告为
`results/fastmtp_prefix_n12_reachfirst_cont005_pilot/comparison.md`。

限制：20 条、单 seed 的准确率粒度为 5pp，MAL 差值也可能受输出轨迹和 RNG 影响；
最终配置是在同一 pilot 上迭代选择，存在选择偏差，必须在未参与调参的新 seed/更大样本
上复现后才能形成论文结论。GSM8K truncation 为 25%，高于 Native/SpecCascade，虽然
本次答案准确率未下降，仍是需要审计的质量风险。结果 artifacts 继续只保留于 ignored
`results/`、`logs/`，未发布到 GitHub。

### 2026-08-12：修复树路径跨轮 target hidden 路由，Exact Residual-Hit MAL 达到 Cactus 区间

状态：`结构缺陷已修复；固定 20 题 pilot 达标，等待独立 100 题复现`

为回答“树方法的 MAL 能否至少接近 Cactus”，先按 DSE 流程继续检查已有固定 20 题
manifest。统一降低普通 relative threshold 到 0.15/0.10 分别得到
GSM8K `90%/3.337`、`85%/3.332`，HumanEval 均为 `75%/3.172`；把 rescue 改为
root 0.06、随后按深度增至 0.12/0.18，则只有 `90%/3.262` 与 `75%/3.159`。
这些负结果说明继续全局放宽阈值不能解释与 Cactus 的 MAL 缺口。

随后增加 `sampled_primary_shadow` 控制：proposal 和验证仍走树 runtime，但树永远
不能改变 Cactus 已采样的 token。9-node 树最初只有 GSM8K `85%/2.978`、HumanEval
`60%/2.982`，而同协议 Cactus 是 `90%/3.509`、`75%/3.361`；将树缩成三节点单链
后 MAL 恢复到约 `3.487/3.330`。逐轮对照发现：在完整接受一条非连续 BFS path 后，
下一轮 target 分布仍基本对齐，但 MTP root Q 已经分叉。根因是 vLLM
`EagleProposer.prepare_inputs_padded` 按“接受 token 数”选择 hidden row，适用于链但
不适用于 BFS 树。例如选中节点 `[0,2,5]` 后，旧逻辑取 row 3，正确值应为实际叶节点
`5 + 1 = 6`。此前尝试对 MTP attention KV cache 做 snapshot/isolation 没有改善，
得到 `95%/3.152` 与 `70%/3.016`，该补丁已撤回并只保留负结果。

当前 runtime 在动态树提交后显式把下一轮 MTP root 的 target hidden 索引改为
`selected_leaf_index + 1`。修复后的 9-node shadow 控制 MAL 提升到
GSM8K `3.430`、HumanEval `3.324`。直接 sibling Cactus rescue 虽达到
`95%/3.560` 与 `60%/3.515`，但 HumanEval 质量坍塌，证实“先挑 sibling 再接受”
会改变 correction 分布，未被选用。

最终选定 `residual_hit_anchor`：主干仍使用完整 Q 的 Cactus 验证；第一次拒绝时先从
精确 `normalize(max(H-Q,0))` 残差分布采样 correction，之后才检查树中是否已有相同
token ID。仅在精确命中时复用该 sibling 的已验证状态，并从该前缀下的原始 target
分布追加一个 anchor。固定 pilot 实测：

| 数据集 | Cactus 质量/MAL | Exact residual-hit anchor 质量/MAL | MAL 差值 |
|---|---:|---:|---:|
| GSM8K | 90.0% / 3.509 | 90.0% / 3.569 | +0.060 |
| HumanEval | 75.0% / 3.361 | 80.0% / 3.540 | +0.178 |

树平均节点数为 `6.359/6.085`，精确命中救援约占全部 speculative rounds 的
`21.0%/20.9%`，贡献 MAL 约 `0.210/0.209`。该机制没有让树替换已经采样的残差
correction，避免了直接 sibling rescue 的质量问题。当前 Python/vLLM 工程仍未优化：
E2E 为 `116.814/116.530 tok/s`，低于 Cactus `144.468/138.267 tok/s`；本阶段只证明
算法层面的质量/MAL 可行性。

受影响文件：

- `remtp/fixed6_vllm.py`：新增 selected tree leaf 到 target hidden row 的映射，并在
  下一轮 MTP input preparation 中覆盖链式索引；
- `remtp/dynamic_tree_vllm.py`：动态树安装时接入上述 selected-leaf hook；
- `tests/test_fixed6_microtree.py`：覆盖空路径和 `[0,2,5] -> row 6` 映射；
- `remtp/fastmtp_verified_report.py`、`tests/test_fastmtp_verified_report.py`：按实际
  `verification_mode` 标注 exact residual-hit 语义，不再统一误写为近似路径选择；
- `scripts/run_fastmtp_selected_residual_anchor.sh`：新增可恢复的 100 题正式实验入口；
- `docs/exact_residual_hit_tree.md`、`docs/mimo_dynamic_mtp_tree.md`：增加旧结果失效通知、
  修复说明、pilot 结果及推荐入口；
- ignored `dse_results/`：记录阈值负结果、结构控制、收敛配置与 `DSE_REPORT.md`；
- `docs/research_and_development_log.md`：本条记录。

验证：使用仓库 `.venv` 完成三个 Python 文件 `py_compile`、两个 shell 脚本
`bash -n`，相关动态树、Fixed-6 和报告测试共 `60 passed`。在 RTX 4090、vLLM
0.18.0、TencentBAC/FastMTP 上实际完成 shadow 修复前后控制、直接 Cactus sibling
救援和最终 residual-hit anchor 的 GSM8K 20 + HumanEval 20 运行及隔离代码测试。

限制：这是在同一固定 20 题/单 seed 上经过多轮选择得到的 pilot，存在明显选择偏差，
且 5pp 的质量粒度不足以证明准确率提升。必须用未参与调参的更大样本和至少第二个 seed
验证；在此之前只能称“达到进入正式实验的门槛”，不能声称普遍优于 Cactus。树每轮
验证约两倍于三 token chain 的节点，吞吐仍低，工程优化尚未开始。实验 artifacts 仍
只位于 ignored `results/`、`logs/`、`dse_results/`，未上传报告数据。

### 2026-08-12：形成独立的“残差对齐树式 MTP”方法叙事

状态：`方法说明完成；未修改 runtime 或实验结果`

新增 `docs/residual_aligned_tree_mtp.md`，将当前选定机制整理为一套独立、连贯的算法：
MTP-only 稀疏纠正覆盖树、单次 Tree Attention 目标验证、候选条件化受控松弛、先残差
采样后树查询的精确命中、原始目标分布 anchor，以及选中 BFS 叶节点的跨轮状态提交。
文档明确给出 \(Q/P/H/R\) 定义、概率提升与接受公式、残差公式、MAL 收益分解、完整
伪代码、当前参数和方法边界。对外方法名暂定为 **Residual-Aligned Tree MTP**。

本次仅新增算法文档和本日志，没有修改验证器、worker、服务脚本、默认参数或 benchmark
协议。验证包括人工核对文档公式与当前 `residual_hit_anchor` runtime 语义，并检查新
文档不包含被替代的外部方法名称；未运行新的 GPU smoke 或 benchmark，也未声称新增
实验结果。限制是文档中的 20 题数值仍来自上一条记录的单 seed pilot，正式方法结论仍
需独立 100 题和第二 seed 复现。

### 2026-08-17：整理并准备上传当前 Residual-Aligned Tree MTP 代码快照

状态：`代码快照已通过 CPU 测试并准备推送；本条不声称新增 GPU 实验`

鉴于工作区已经从早期 proposal calibration 扩展为 FastMTP/MiMo 适配、动态树验证、
精确残差命中和完整 benchmark 工具链，新建分支
`research/residual-aligned-tree-mtp`，保留原有研究分支不变。此次快照包含当前所有
实现代码、运行脚本、配置、单元测试和方法/研发文档；继续排除模型、数据集、checkpoint、
`results/`、`logs/` 和 ignored `dse_results/`。同时将 `.vscode/` 加入 `.gitignore`，
避免上传机器本地编辑器配置。纯历史实验报告
`docs/four_way_comparison_summary.md` 按既定规则仅保留本地并加入忽略列表。
未发现私钥、GitHub/Hugging Face token 或 API key。

主要受影响内容：

- `remtp/`：概率 MTP、MiMo/FastMTP checkpoint 与 worker 适配、动态树/Fixed-6
  Tree Attention、残差对齐验证、benchmark/report/evaluator 和历史消融实现；
- `scripts/`：模型下载、服务启动、断点恢复、GSM8K/HumanEval 对比及选定
  residual-anchor 正式实验入口；
- `tests/`：上述 runtime、概率语义、状态路由、报告和评测协议测试；
- `configs/`：静态 proposal calibration 与无思考 chat template；
- `docs/`、`README.md`、`AGENTS.md`、`CLAUDE.md`：算法说明、历史/失效通知、使用说明
  和强制研发记录规则；
- `.gitignore`、`docs/research_and_development_log.md`：本次上传边界和记录。

提交前另对三份 FastMTP 历史文档和本日志做了纯机械的行尾空格
清理，并删除 `remtp/dynamic_tree_humaneval_report.py` 的多余文件末空行；无任何
算法、协议或 runtime 语义变化。

验证：仓库 `.venv` 下完整执行 `python -m pytest -q`，结果为 `334 passed`；执行
`bash -n scripts/*.sh`，所有 shell 脚本语法检查通过；运行敏感信息模式扫描，未发现
私钥或常见 token/API key。没有在本次上传工作单元重新运行 GPU smoke、GSM8K 或
HumanEval，因此 GPU 可运行性和历史性能仅以此前相应日志条目为准。

限制：当前分支包含研究过程中保留的历史实现和负结果对应代码，尚未压缩为最小发布包；
部分文档记录 pilot 数值，但原始结果 artifacts 仍只在本地 ignored 目录。上传不会包含
模型权重、数据、实验输出或本地 `.vscode` 配置。正式质量结论仍需选定方法的独立大样本
与第二 seed 验证。
