# RAVEN 分支在 Jetson R39.2.1 上的适配结果

> 历史结果说明（2026-10-07）：下文100题五方法对照为2026-09-30的旧数值协议记录。之后Target/MTP数值路径已修复，它不代表当前配置的正式质量/速度结果；已观察题目也不能再充作独立测试集。[当前全项目审查](REVIEW-ADAPTIVE-RAVEN-PROJECT-2026-10-07.md)与[最新阶段结果](RESULT-RAVEN-PROBABILITY-CAP-2026-10-07.md)。旧结果保留作历史证据。

状态：**核心方法已在 GPU 实机跑通；GSM8K 100 题 × 五方法正式对照已完成，HumanEval 尚未运行**。工作目录 `/home/apex/ReMTP-residual`，本地分支 `jetson-r39-residual-adapt`，基于远端 `research/residual-aligned-tree-mtp` 的 `3afafa1`。原 `/home/apex/ReMTP` 的 Gemma 4 工作目录保留。

## 选型与环境

本次以 RAVEN 方法为主。该分支的树验证代码绑定 MiMo/FastMTP 的一个物理 MTP 层及 vLLM 0.18 的内部接口。沿用官方 `TencentBAC/FastMTP` 权重、只对 Target 使用 bitsandbytes 4 位加载，MTP 草稿层保持 BF16。由此保留分支的树构造、一次 Tree Attention 验证、候选条件化接受、残差 correction 和实际 TV 自适应松弛实现。

官方权重固定在 revision `70b34477e1f274516ddadf778dd96a7f037bf176`。四个 BF16 分片共 15,666,870,944 字节（14.59 GiB），接近本机 15.17 GiB 的统一内存，尚未尝试完整 BF16 服务。隔离环境为 Python 3.12、PyTorch 2.10.0+cu130、vLLM 0.18.0+cu130、bitsandbytes 0.50.2；初期冒烟服务用 256 MiB KV、1024 token 最大上下文，正式评测改为 1 GiB KV、4096 token 最大上下文；均用 eager 模式和单请求调度。实测 Native/RAVEN 运行时系统 `MemAvailable` 约 2.8 GiB。

Gemma 4 W4A16 + 独立 assistant 的既有实现不具备这里的 MiMo 物理 MTP 层与 vLLM 0.18 树接口，不能直接换进本分支方法。

## 已验证的运行路径

| 检查 | 结果 |
| --- | --- |
| 模型完整性 | 四个分片下载完成；checkpoint 检查确认 451 个权重、一个物理 MTP 层。 |
| CPU 回归 | `pytest -q`：340 通过、2 跳过；4 位 BF16 CUDA 小算子通过。 |
| Target | 4 位 HTTP `/health` 为 200，32-token 请求成功；vLLM 报告权重加载 5.72 GiB。 |
| Native MTP | 4 位 Target + BF16 草稿层 HTTP 请求成功；32-token 请求有 16 轮、48 个草稿 token、17 个接受。 |
| Adaptive RAVEN | HTTP 请求成功；深度 3、最多 9 节点、`residual_hit_anchor`、adaptive actual-TV。32-token 请求 16 轮均只调用一次 Target forward，4 轮命中 residual target anchor。 |
| GSM8K 流程 | 官方 test 的固定 3 条样本在 Native、uniform RAVEN、adaptive RAVEN 均完成。Adaptive 105 轮、uniform 116 轮均为每轮一次 Target forward；残差命中分别 32、35 次。 |

以下为正式评测前的历史小样本冒烟协议：sample seed `20260730`、temperature `0.6`、generation seed `42`、空 system prompt、no-thinking 模板、每条最多 128 token。数据和请求结果位于忽略的 `results/`。该 3 题结果仅验证执行链路，正式结果见下文。

| 方法 | 正确/样本 | 截断 | decode tok/s | E2E tok/s | MAL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native MTP | 1/3 | 3/3 | 2.553 | 2.476 | 2.777 |
| Uniform RAVEN | 1/3 | 2/3 | 2.573 | 2.485 | 3.181 |
| Adaptive RAVEN | 1/3 | 2/3 | 2.648 | 2.555 | 3.229 |

原始结果：`results/raven_jetson_gsm8k3_{native,uniform,adaptive}/summary.json`；RAVEN 逐轮审计：`results/raven_jetson_uniform_rounds.jsonl` 和 `results/raven_jetson_smoke_rounds.jsonl`。审计包含实际树节点、一次 Target forward、残差命中和 adaptive TV 开关，证明执行了本分支方法路径，而非普通 Native MTP。三组的样本 manifest 一致。

## 本机改动与运行

`scripts/install.sh`、`scripts/cuda_env.sh` 提供 R39 ARM64 CUDA 环境；`scripts/serve_fastmtp_verified.sh` 接收可选量化和 KV 参数；`scripts/serve_fastmtp_jetson.sh` 固定本机可运行的默认参数。官方本地 `config.json` 的单元素 `tokenizer_class` 列表在 Transformers 4.57 中不能解析，因此 `scripts/prepare_fastmtp_jetson.sh` 保存 `config.hf-original.json` 并把该字段规范化为字符串。目标 4 位时，`remtp/jetson_bf16_draft.py` 只在草稿层构造期间关闭继承的量化配置，以按原始 BF16 形状加载草稿权重。树算法本身未改。逐文件说明见 [改动记录](upstream-changes/CHANGELOG.md)。

```bash
cd /home/apex/ReMTP-residual
./scripts/serve_fastmtp_jetson.sh
```

入口默认 `METHOD=dynamic_tree`；可通过环境变量设置 `METHOD=native` 或 `METHOD=target`，以及覆盖 `MAX_MODEL_LEN`、KV 和 RAVEN 参数。当前 256 MiB KV 的实测容量为 1760 token，因此默认最大上下文为 1024；提高上下文前需要同步增加 KV 预算并复测内存。

## 剩余限制

- 五方法 GSM8K 100 题已完成；HumanEval 100 题及其执行评分尚未运行。因此尚未完成仓库推荐的 GSM8K/HumanEval 100+100 全套对比。
- Docker socket 属于 `root:docker`，当前用户不在 `docker` 组；HumanEval 受限执行评分尚不能运行。官方数据已下载：GSM8K test 1319 条、HumanEval 164 条。
- RAVEN 的自适应松弛改变输出分布，不能称为 lossless speculative decoding。量化 Target 与原始 BF16 Target 的质量差异也尚未量化。

后续步骤见[计划](../plan/raven-jetson-r39.md)。

## 正式 GSM8K 对照：诊断与最终结果（2026-09-29—30）

曾启动设备版的固定 **100 题、五方法** 对照：Native、Cactus、SpecCascade、uniform RAVEN、adaptive RAVEN。**该次 768-token 协议已暂停；仅完成 Native 第 1 题，不能作为正式结果。**结果目录为本地忽略的 `results/raven_jetson_gsm8k_n100_r39/`，协议固定在其中的 `protocol.json`，逐题断点在各方法的 `requests.checkpoint.jsonl`。运行入口是 `scripts/run_jetson_raven_gsm8k.sh`，按方法自动启动/关闭服务并跳过已完成结果。

为适应本机 1024 token 上下文，首轮输出上限设为 768 token；固定抽样的 100 条 GSM8K 提示最长约 209 token。此前 3 题冒烟上限是 128 token；仓库通用五方法脚本默认上限为 2048 token。Native 第一题在 768 token 截断，耗时约 299 秒，输出有重复自我修正且没有可靠最终答案。按这一速度和截断情况，不宜在未经质量检查时自动续跑全套。4096 上下文、2048 输出上限的 Target 单题也已完成：**输出满 2048 token 仍截断**；末尾只是重复题干，评测器从末尾提取数字 20，碰巧与正确答案一致，不能当作可靠的解答。1 GiB KV 提供 7280 token 容量，服务运行时系统 `MemAvailable` 约 2.9–3.2 GiB。HumanEval 评分仍等待 Docker 访问权限。

独立诊断表明：同一题改成“只给答案”的提示词时，默认 FP4 Target 可在 97 token 正常结束且答对；另一道题即使使用该提示词仍到 256 token 截断。开启原生思考模板的同一题在 384 token 也截断。隔离的 NF4 诊断入口能加载 Target 并执行 RAVEN，但 NF4 的几道短测试仍有截断；未证实质量收益，因此未保留 NF4 仓库改动。

仓库既有的 [2026-08-09 诊断](../docs/research_and_development_log.md) 曾在 RTX 4090 上观察到 FastMTP/MiMo RL 的冗长与重复输出：GSM8K 10 条平均约 1,020 token，HumanEval 10 条平均约 1,737 token，原版上限本来就是 2048。因此较长回答并非 Jetson 独有。质量优先的下一轮选用原版 2048 输出上限，设备最大上下文扩至 4096、KV 预算 1 GiB；量化 Target 仍是与方法已验证的默认 FP4，单独报告量化限制和实际截断率。

设备评测报告并列给出仓库原有准确率和更严格的正确数：后者要求服务正常结束，且最后一行恰为正确的 `#### <number>`。这样被截断输出末尾的偶然数字不会被当成可靠结论。

**2026-09-30 完成正式 GSM8K 对照。** 用户级 `deploy/remtp-raven-gsm8k.service` 已自动停止，`Result=success`、退出码 0；完成标记时间为 `2026-09-30 21:09:56 UTC`。运行目录 `results/raven_jetson_gsm8k_n100_fp4_2048/` 保存协议、五组各 100 条逐题记录、相同的样本 manifest、汇总、RAVEN 树审计与[完整对照报告](../results/raven_jetson_gsm8k_n100_fp4_2048/comparison.md)。Target 使用默认 bitsandbytes 4 位，MTP 草稿层 BF16；固定 100 题、temperature 0.6、seed 42、2048 输出上限、4096 上下文及 1 GiB KV。服务在最后一组结果写出后曾以 127 退出，systemd 自动重启并跳过所有已完成方法，随后成功生成报告与 `COMPLETE`；没有重复计入样本。

| 方法 | 原仓库判对 | 正常结束且判对 | 严格最终行正确 | 截断 | E2E tok/s | MAL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 88/100 | 82/100 | 70/100 | 16/100 | 2.659 | 2.911 |
| Cactus | 83/100 | 81/100 | 63/100 | 12/100 | 2.988 | 3.272 |
| SpecCascade | 88/100 | 81/100 | 69/100 | 14/100 | 2.799 | 3.068 |
| uniform RAVEN | 91/100 | 83/100 | 68/100 | 13/100 | 2.837 | 3.469 |
| adaptive RAVEN | 86/100 | 80/100 | 64/100 | 15/100 | 2.828 | 3.476 |

### 结果分析

1. **质量差异取决于判分口径。** 原仓库解析器会在找不到标准答案时退回输出末尾的数字，甚至可能把截断输出判对；五组中被判对但已截断的题数依次为 6、2、7、8、6。排除截断后，五方法仅在 80–83 题之间；要求最后一行恰为正确 `#### <数字>` 时仅为 63–70 题。后一口径还会排除可能有效的 boxed 等非标准格式，因此它反映答案格式和结束状态，不等同于全部数学推理质量。应优先结合三列阅读，不能单看 91% 的最高原始准确率。
2. **adaptive 没有在这次测试中胜过 uniform。** 两者使用同一 100 题，原仓库口径分别为 86 与 91 题；逐题配对时，uniform 独有判对 5 题，adaptive 独有判对 0 题。正常结束且判对分别是 80 与 83 题。MAL 为 3.476 与 3.469，adaptive 仅多 0.007，但吞吐分别为 2.828 与 2.837 tok/s，差异极小且方向相反。树审计显示 adaptive 有 51,231 个节点的实际松弛权重不同于基础权重、uniform 为 0，说明自适应路径确实执行了；现有数据并未显示它带来质量或速度收益。
3. **接受长度变长不保证整机吞吐同比提高。** 相对 Native 的 2.659 tok/s，Cactus、uniform、adaptive 的总吞吐分别高约 12.4%、6.7%、6.3%；RAVEN 的 MAL 约 3.47，高于 Native 的 2.91。两组 RAVEN 审计分别有 19,878 和 19,108 轮，每轮一次 Target forward，但平均验证约 6.75 个树节点。树构造与验证成本可能抵消部分接受长度收益；这是依据审计和吞吐作出的解释，尚无逐组件耗时实验。各方法平均输出长度也不同，Cactus 的单题耗时中位数为 98.8 秒，反而高于 Native 的 80.6 秒，因此总体 tok/s 不能直接理解为每一道题都更快。
4. **结论范围有限。** 这是一次固定抽样与生成种子的 100 题设备内对照，没有重复种子或 HumanEval；小幅差异尚不能视为稳定排序。Target 是 4 位加载，尚未定量比较 BF16 Target 的质量。

2026-10-01 的[进一步诊断](ANALYSIS-RAVEN-GSM8K-2026-10-01.md)补充了机制证据：Adaptive 的同块实际 TV 比 uniform 变换控制低约 30.3%，但 61.6% 的轮次没有重新分配额度；anchor 模式约 20.1% 的备选后代节点不会被选中。轻量审计记录构建/写入回放约 0.38 ms/轮，不能解释约 134 ms 的整体轮成本差距。旧实验未记录各组件独立耗时，优化优先级和验证步骤已更新到计划。

查看进度和日志：

```bash
cd /home/apex/ReMTP-residual
./scripts/status_jetson_raven_gsm8k.sh
journalctl --user -u remtp-raven-gsm8k.service -n 30 --no-pager
```

控制器已停止，模型内存和端口 8000 已释放。若需按原协议检查或恢复，可运行 `systemctl --user start remtp-raven-gsm8k.service`；它会跳过已有完整结果。
