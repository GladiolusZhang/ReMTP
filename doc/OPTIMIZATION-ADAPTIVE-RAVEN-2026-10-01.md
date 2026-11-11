# Adaptive RAVEN 优化实施与验证

> 2026-10-02 历史状态说明：下文是实施过程快照。v4 质量校准现已完成，
> 门槛未通过并主动停止；完整结果见[中文报告](RESULT-ADAPTIVE-RAVEN-OPTIMIZATION.md)。
> 下一阶段按[截断与公平对照优化计划](../plan/adaptive-raven-next-optimization-2026-10-02.md)执行。

状态快照：v4 短测门槛通过，后台完整质量校准已启动；独立正式复测尚未完成。
本文件保存实施时的手动进度；最新进度以 `status.json` 和
[自动中文结果报告](RESULT-ADAPTIVE-RAVEN-OPTIMIZATION.md)为准，不把阶段进度当作最终正确率结论。

## 目标与约束

保持 FastMTP FP4 Target、BF16 MTP、R39 环境及原来的接受/残差修正规则。
优先保持原评测正确数和正常结束且判对数，提高实际吞吐；不通过缩短质量测试输出上限提速。

## 本次改动

| 改动 | 原因 | 验证与状态 |
| --- | --- | --- |
| anchor 最终计算前删除备选分支后代（v1，不采用） | 旧正式日志中约 20.1% 节点不会被使用 | CPU 同概率下输出/RNG 等价；GPU 短测吞吐 -3.29%，未改善轮成本 |
| 稳定 topk 加无并列快速路径（v3） | 最多选少量候选，避免全排序 | 首版 v2 GPU 算子反而更慢，已停止；v3 GPU 20 案例等价通过，算子约 0.50 ms vs 全排序 0.74 ms，端到端收益待测 |
| 只构建主路径和直接 sibling（v3，单独不采用） | 不为不可复用的后代继续计算 Q，把预算用于能复用的分支 | CPU 模拟检查通过；D3/N9 普通短测 -4.73%，GPU 单行 head 变慢抵消节省 |
| 3 行 head 输入与 D4/N12（v4，当前候选） | 实测单行 head 较慢；多验证一级主路径摊薄 Target 轮成本 | 复制相同 head 输入、只取原行数，不增 KV/RNG；4 层仍复用物理 head 0；正在普通测速与质量验收 |
| 显式空路径可关闭树审计 | 正式速度应排除审计影响 | profiling、测速、质量分开；全部对照使用相同审计设置 |
| 可选组件计时 | 确认主要瓶颈 | CUDA 同步计时仅诊断使用，记录嵌套关系，普通测速关闭 |
| systemd 后台控制器 | 对话关闭后仍可续跑 | 代码与协议冻结，逐题落盘，有限重试；门槛不通过自动停止，不自动采用失败候选 |

优化开关均默认关闭：`REMTP_DYNAMIC_TREE_PRUNE_ANCHOR_DESCENDANTS=1`、
`REMTP_DYNAMIC_TREE_EXACT_TOPK=1`、`REMTP_DYNAMIC_TREE_ANCHOR_ONLY_PROPOSAL=1`。
裁剪和 anchor-only 只允许 `residual_hit_anchor`，后者仅支持固定节点预算。
v4 仅启用后两项，max_depth=4、max_nodes=12、max_children=3；head 使用 3 行复制。
anchor-only 会改变候选分配、覆盖与 GPU 行数，
不能用主路径模拟检查代替完整质量校准。
旧协议和旧 100 题结果保持独立保存。

## 已得到的证据

- 当前完整 CPU 回归：356 passed、2 skipped。
- 初版 baseline 的 64 token 同步计时：Target 平均 937.6 ms/调用、proposal 216.1 ms/调用、
  验证 23.5 ms/调用、缓存更新 15.3 ms/调用；规划 3.2 ms 已包含在验证中，
  每次 draft_tree 约 15.3 ms，其时间已包含在 proposal 中。
- 一次短请求的裁剪版 Target 为 932.4 ms，proposal 为 218.4 ms；这个变化很小，
  不足以证明裁剪提速。仍需要关闭计时的普通测速和更多工作负载。
- 当前设备 `nvpmodel -q` 为 15W。性能结果限定于此设备状态。
- 实际 CUDA 合成校验通过：TREE_ATTN vs FP32 参考 max_abs=0.0037183，保留节点
  裁剪前后 BF16 输出 max_abs=0；8 个主路径/sibling 的 KV 搬运逐位一致。
  这项检查不代表完整模型概率或答案质量已经相同。
- v4 普通短测全部完成：两次 baseline 合计 E2E 2.643 → candidate 2.941 token/s，
  提高 11.29%；MAL 3.325 → 3.700。Cactus D4 为 2.614 token/s，候选高 12.53%。
  两条固定提示仅作性能筛选，不能据此宣称在正式任务上稳定胜出。
  首次 profile 中，D4 proposal 约 195.6 ms，低于旧 D3 的约 220 ms；
  新 head 平均 20.8 ms/调用。这是组合候选的结果，不能把收益全部归因于某一项。

## 后台校准协议

1. 固定两个 train 提示、同种子进行短测速；baseline 额外重跑以检查时间漂移。
2. 只在候选实际吞吐超过 baseline 时进入质量校准；短测速不评价正确率。
3. GSM8K train 独立固定抽取 24 题，seed=42/43，优化前后都用 2048 输出上限。
4. 总体和每个种子均检查原评测正确数、正常结束且判对数；保存配对变动，
   下降则停止，并保留错误输出。通过校准也不等于通过独立正式复测。

train 来自[官方 GSM8K 仓库](https://github.com/openai/grade-school-math)，本地校验 7473 条。
数据哈希、题目 ID、代码哈希与依赖版本在各运行的 `protocol.json` 中保存。

## 文件与运行状态

- 修改前快照：`results/adaptive_optimization_20261001/baseline_code/`。
- 第一轮裁剪验证：`results/adaptive_optimization_20261001/prune_v1/`。
- 首版 topk 诊断：`results/adaptive_optimization_20261001/topk_v2/`，因算子变慢主动停止。
- 三层 anchor-only 负结果：`results/adaptive_optimization_20261001/anchor_v3/`。
- 当前冻结代码：`results/adaptive_optimization_20261001/candidate_code_v4/`。
- 当前验证：`results/adaptive_optimization_20261001/anchor_d4_v4/`；阶段结束自动生成中文
  [结果与分析](../results/adaptive_optimization_20261001/anchor_d4_v4/report.md)。
- `doc/RESULT-ADAPTIVE-RAVEN-OPTIMIZATION.md` 为上述自动报告的本机链接，校准完成后自动更新内容。
- 服务：`remtp-adaptive-optimization.service`。
- 查询：`systemctl --user status remtp-adaptive-optimization.service`；
  查看运行目录中的 `status.json`、`smoke_gate.json`、`quality_gate.json`。

完整实验未结束。正式论文结论仍需三个主基线和优化前后方法的独立题目、多种子对照。
当前短测 Cactus 也使用 D4；正式对照须加入相同深度的 Native 与 SpecCascade，
并补做 D3 + head 优化消融、基线共享 head 优化控制，区分方法与工程实现收益。
