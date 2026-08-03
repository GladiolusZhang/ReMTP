# Exact-TV + Expected-Regret Router

该分支实现两部分互补机制：

1. 当前块仍使用 Exact-TV：从 CACTUS 六个位置的总 TV 出发，在候选达到
   `A=1` 后停止投入，回收饱和预算，再按前缀价值、目标 margin、静态
   head 可靠度重新分配；future support 只是否决项，不再用 hidden
   consistency 作为正奖励。
2. 跨块使用期望遗憾：每个已提交草稿位置的责任为
   `max(A_relaxed-A_strict, 0)`。它与实际 TV、目标偏好集中度、预算重分配
   比例共同形成标量 debt；`P-H` 中被压低的目标偏好形成只保留一块的
  方向状态。

小型低秩 Router 冻结目标模型和 MTP，只输出每个 MTP head 的三个有界
控制量：方向强度、logit/置信度缩放、下一块 Exact-TV budget scale。
budget scale 始终位于 `[0,1]`，所以 Router 只能削减 Exact-TV 已分配的
TV，无法突破 CACTUS 总 TV 上限。debt 为零时三个控制量严格退化为恒等映射。

## 因果边界

草拟下一块之前不能无代价获得“下一块验证后的目标分布”。因此 proposal
通道使用上一块最后可用的目标 entropy/margin，加上当前 causal root
hidden。当前块目标 `P_i` 在一次目标验证后才可用，此时 Router 只将它用于
budget scale。实现没有额外增加目标模型 forward。

## 1. 准备无标签 Router 语料

不要使用 HumanEval 测试题或 GSM8K 测试答案训练 Router。准备普通文本、
代码或指令 JSON/JSONL；每行支持以下任一格式：

```json
{"prompt": "Explain how a hash table handles collisions."}
{"instruction": "Write a Python merge-sort implementation."}
{"messages": [{"role": "user", "content": "Summarize this paragraph..."}]}
```

## 2. 收集 Exact-TV trace

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

ROUTER_CORPUS=/absolute/path/to/router_corpus.jsonl \
SAMPLES=1000 \
TEMPERATURE=0.7 \
MAX_TOKENS=256 \
./scripts/collect_regret_router.sh
```

trace 默认保存在 `data/regret_router/traces/`，只包含推理期 `P/Q/H` 摘要、
hidden/direction 和验证标量，不读取任务答案标签。该目录被 Git 忽略。

## 3. 训练小型 Router

```bash
EPOCHS=10 \
BATCH_SIZE=64 \
./scripts/train_regret_router.sh
```

默认 checkpoint 为 `checkpoints/regret_router.pt`，目标函数联合优化：下一块
`P/Q` 对齐、干预幅度、entropy 漂移、严格接受收益和期望遗憾风险。
目标模型及 MTP 权重不参与优化。

## 4. 单独启动

纯 Exact-TV（无 Router 额外开销）：

```bash
TARGET_ANCHORED_VARIANT=tv_router ./scripts/serve_target_anchored_mtp.sh
```

训练后的双通道遗憾反馈：

```bash
REGRET_ROUTER_MODE=learned \
REGRET_ROUTER_CHECKPOINT=checkpoints/regret_router.pt \
REGRET_ROUTER_AUDIT_INTERVAL=100 \
./scripts/serve_exact_tv_regret_router.sh
```

## 5. HumanEval 对比

先跑 10 条链路测试：

```bash
SAMPLES=10 \
PROFILES="native_mtp cactus exact_tv exact_tv_regret_router" \
./scripts/run_humaneval_regret_router.sh
```

再跑全部 164 条，并包含之前质量或吞吐较有代表性的方案：

```bash
SAMPLES=164 \
TEMPERATURE=0.7 \
SEED=42 \
./scripts/run_humaneval_regret_router.sh
```

默认比较 Native MTP、CACTUS、SpecCascade、Exact-TV head、历史
hidden-veto Exact-TV、当前 target-only Exact-TV，以及训练后的 Regret
Router。生成代码只在受限 Docker 容器中执行。结果和日志保存在被 Git
忽略的 `results/`、`logs/`，不会上传报告或实验数据。

## 建议判据

Router 不是为某一类答案格式单独优化。先检查：

- HumanEval `pass@1` 高于 CACTUS；
- MAL 高于 CACTUS；
- E2E 不低于 CACTUS，或差距在预先设定的 3% 容忍范围内；
- strict-acceptable 比例提高、单位接受 token 的 TV 降低；
- 使用第二个生成 seed 复现后再扩大训练或测试规模。
