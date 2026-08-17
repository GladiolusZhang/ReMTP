# Proposal-Calibrated MTP

## 1. 方法概览

Proposal-Calibrated MTP 的目标不是放宽目标模型的验证规则，而是提高 MTP
提议分布与目标分布的重叠。

设第 `i` 个 MTP head 的原始提议分布为 `Q_i`，目标模型在相同草稿前缀下的
分布为 `P_i`。标准概率推测解码在该位置的期望严格接受率为：

```text
A_i = sum_x min(P_i(x), Q_i(x)).
```

方法使用只依赖 `Q_i` 和 head 编号的固定变换 `C_i`：

```text
Q_tilde_i = C_i(Q_i).
```

随后统一使用 `Q_tilde_i`：

1. 从 `Q_tilde_i` 采样 MTP 草稿 token；
2. 使用 `min(1, P_i(y_i) / Q_tilde_i(y_i))` 验证草稿；
3. 拒绝时从 `max(P_i - Q_tilde_i, 0)` 的归一化残差分布采样纠正 token。

因此校准只影响提议效率，不改变目标分布。单 token 的输出概率可以写为：

```text
accepted mass = min(P, Q_tilde)
rejected mass = 1 - sum_x min(P(x), Q_tilde(x))
residual      ∝ max(P - Q_tilde, 0)

accepted mass + rejected mass * residual = P.
```

按自回归位置递归应用后，完整输出序列仍服从目标模型分布。

## 2. 当前冻结算法

正式版本使用静态 head-wise temperature scaling：

| MTP head | 校准温度 |
|---:|---:|
| 1 | 0.7 |
| 2 | 0.5 |
| 3 | 0.3 |
| 4 | 0.5 |
| 5 | 0.4 |
| 6 | 0.4 |

若用户请求温度为 `T_request`，第 `i` 个 MTP head 实际使用：

```text
Q_tilde_i = softmax(MTP_logits_i / (T_request * T_head_i)).
```

深度处理规则：

- MTP=2：使用前两个 head 配置；
- MTP=4：使用前四个 head 配置；
- MTP=6：使用完整配置；
- MTP=8：第 7、8 个递归位置复用第 6 个 head 的温度 0.4。

MTP=8 的后两项没有单独使用目标日志调参，避免为某个评测深度引入额外
target 信息或后验选择。

## 3. 为什么优化整块 MAL

离线配置选择不以平均 token 接受率为目标，而是最大化六 token 草稿块的期望
接受长度：

```text
E[MAL] = 1 + a1 + a1*a2 + ... + a1*a2*...*aK.
```

其中 `a_i = sum_x min(P_i(x), Q_tilde_i(x))`。这个目标自动给予早期 head
更高权重：第一个位置的改善会提高后续所有位置被到达的概率。

离线工具也实现了以下 target-free 候选变换，用于比较但不一定进入最终配置：

- head-wise temperature scaling；
- top-2/top-3 内部概率质量摊平；
- top-2/top-3 次候选质量向 top-1 集中；
- `Q` 与 top-k 均匀分布的小比例混合。

所有策略的变换只读取 `Q`。目标 `P` 只用于离线评价策略，不参与在线档位选择。

## 4. vLLM 中的运行流程

### 4.1 Worker 安装

服务使用：

```text
remtp.worker.ProposalCalibratedMTPWorker
```

Worker 初始化时依次安装：

```text
install_probabilistic_mtp()
install_proposal_calibration()
```

顺序不能颠倒。第一步让 vLLM 暴露完整 MTP 分布，第二步在该分布被采样之前
执行校准。

### 4.2 MTP proposal

vLLM 0.18 的 Qwen3.5 MTP 路径默认对每个递归位置执行 argmax，并可能向验证器
传入 `draft_probs=None`。`remtp/probabilistic_mtp.py` 对以下位置进行运行时适配：

```text
EagleProposer._greedy_sample
EagleProposer.propose
RejectionSampler.forward
```

每个 MTP 递归位置执行：

```text
MTP hidden
  → shared LM head
  → MTP logits
  → request temperature × head calibration temperature
  → Q_tilde
  → exponential-race categorical sampling
  → draft token
```

生成的完整 `Q_tilde` 行按 D0、D1、……的顺序缓存。草稿 token 和概率行数量
必须完全一致，否则直接报错。

### 4.3 Target verification

目标模型仍然只执行一次标准 speculative target forward，得到草稿路径上各位置的
target logits。Proposal-Calibrated MTP 不修改：

- target hidden state；
- target logits；
- target KV cache；
- target sampling constraints；
- target forward 的节点数量和调用次数。

在 `RejectionSampler.forward` 收到 `draft_probs=None` 时，适配器把本轮缓存的
`Q_tilde` 传回 vLLM 原生 rejection sampler。

### 4.4 严格接受与残差纠正

目标 logits 仍由 vLLM 原生逻辑完成 temperature、top-k、top-p 等约束并形成 `P`。
随后 vLLM 原生概率 rejection sampler 使用同一个 `Q_tilde` 完成：

```text
acceptance(y) = min(1, P(y) / Q_tilde(y))
residual(x)   ∝ max(P(x) - Q_tilde(x), 0)
```

proposal sampling、acceptance 和 residual 三处使用的是同一份 `Q_tilde`，这是
严格分布一致性的核心条件。

## 5. 计算复杂度

最终静态配置只改变 softmax 的温度分母。实现把校准温度合并进 vLLM 原本已经
执行的 proposal softmax，没有新增完整词表 softmax、top-k 或 CPU/GPU 同步。

```text
原生：softmax(logits / T_request)
本方法：softmax(logits / (T_request * T_head))
```

每个 head 只增加一次标量温度选择和一个单元素乘法。紧凑 P/Q trace 和网格搜索
只在离线采集模式开启，禁止用于正式吞吐测试。

## 6. 文件与职责

| 文件 | 作用 |
|---|---|
| `configs/proposal_calibration_static.json` | 冻结的 head-wise 校准参数 |
| `remtp/probabilistic_mtp.py` | 从完整 MTP 分布采样并把相同 Q 送入验证器 |
| `remtp/proposal_calibration.py` | 校准变换、配置加载、在线 hook、紧凑 P/Q trace |
| `remtp/proposal_calibration_offline.py` | P/Q 错配分析、整块 MAL 搜索、5+1 oracle |
| `remtp/proposal_calibration_report.py` | 小样本筛选报告 |
| `remtp/proposal_depth_compare.py` | 正式多深度、多方法统一报告 |
| `remtp/worker.py` | 注册 `ProposalCalibratedMTPWorker` |
| `scripts/serve_proposal_calibrated_mtp.sh` | 启动在线校准服务 |
| `scripts/run_proposal_calibrated_mtp.sh` | 紧凑 trace、离线搜索和小样本筛选 |
| `scripts/run_formal_proposal_calibration_depths.sh` | MTP=2/4/6/8 正式比较矩阵 |
| `tests/test_proposal_calibration.py` | 归一化、Q 一致性、分布恢复及块目标测试 |

## 7. 限制与实验边界

- 当前概率 MTP 适配器要求 `--max-num-seqs 1`，正式结果属于单请求延迟/吞吐协议；
- 该方法针对随机采样。greedy 解码只取 argmax，正温度缩放不会改变 argmax；
- MTP=8 的第 7、8 个位置复用第 6 个配置，需要通过正式深度实验判断泛化；
- 修改 Q 会改变固定 seed 下的具体随机轨迹，但不会改变无条件目标输出分布；
- trace 模式包含完整词表归约和同步，trace 数据不能用于报告吞吐。

## 8. 运行入口

启动单个服务：

```bash
MTP_TOKENS=6 \
REMTP_PC_CONFIG=configs/proposal_calibration_static.json \
./scripts/serve_proposal_calibrated_mtp.sh
```

正式多深度比较：

```bash
./scripts/run_formal_proposal_calibration_depths.sh
```

最终统一结果写入运行目录中的 `comparison.md`、`comparison.json` 和
`comparison.csv`。
