# RAVEN 分支在 Jetson R39.2.1 上的适配计划

目标：优先验证 `research/residual-aligned-tree-mtp` 的 RAVEN 树验证方法；允许为设备内存改用 4 位权重，保持验证算法与对比协议不变。

| 步骤 | 操作 | 预期效果 | 状态 |
| --- | --- | --- | --- |
| 1 | 在独立 worktree 固定分支提交，复核方法和依赖。 | 不覆盖此前 Gemma 4 适配，明确所需的 MiMo/FastMTP 接口。 | 已完成，`3afafa1` |
| 2 | 建立 R39 CUDA 13.0 / vLLM 0.18.0 / bitsandbytes 环境并跑方法单元测试。 | 树算法与 Python 接口在本机通过。 | 已完成：完整套件 340 通过、2 跳过；4 位 CUDA 小算子通过 |
| 3 | 下载固定 revision 的 FastMTP 权重，校验单物理 MTP 层。 | 获得与分支方法相容的 target 和草稿层。 | 已完成：4 分片、451 权重、单物理 MTP 层 |
| 4 | 先验证 4 位 Target 和原生 MTP，再验证 RAVEN Tree Attention。 | HTTP 请求成功，审计日志确认树节点、接受及残差命中。 | 已完成：三条服务路径成功，RAVEN 逐轮审计确认一次 Target forward 与 residual 命中 |
| 5 | 对比 Native、uniform RAVEN、adaptive RAVEN，记录质量、速度与内存。 | 给出同模型同设备下可解释的结果及剩余限制。 | 已完成 3 题链路冒烟；正式 GSM8K 100 题结果见步骤 6。 |
| 6 | 按固定 100 条 GSM8K 运行 Native、Cactus、SpecCascade、uniform 和 adaptive RAVEN；同时报告原版与严格正确数、格式和截断率。 | 获得五方法同模型同样本、原版 2048 输出上限的质量与速度数据。 | 已完成：五组各 100 题，报告、逐题记录和树审计齐全。HumanEval 评分仍受 Docker 权限限制。 |
| 7 | 用用户级 systemd 托管经验证的正式协议。 | 对话关闭后继续运行，失败自动重启并从逐题断点续跑，成功后自动生成报告。 | 已完成：服务成功结束并自动停止；报告与 `COMPLETE` 已生成。 |

原始 BF16 四分片合计 14.59 GiB，几乎占满本机 15.17 GiB 统一内存；实机路径先用 bitsandbytes 4 位动态加载。Gemma 4 现有模型没有此分支要求的 MiMo 物理 MTP 层和 vLLM 0.18 树适配接口，不能直接替换。

核心适配结论与启动命令见 [结果文档](../doc/RESULT-RAVEN-R39.2.1.md)。

## 2026-10-01：Adaptive RAVEN 质量与吞吐改进

适配和首轮 GSM8K 对照已完成。原优化步骤 8–12 的概要已由[Adaptive RAVEN 优化执行计划](adaptive-raven-optimization.md)细化并取代，以新计划作为执行入口。

优化目标是正确率尽量不降低、实际速度提高；论文主基线为 Native、Cactus、SpecCascade，uniform 为消融。先验证保持概率规则的实现优化，再使用独立校准数据选择参数，最后冻结配置做多种子正式对照。诊断依据见[日志分析](../doc/ANALYSIS-RAVEN-GSM8K-2026-10-01.md)。
