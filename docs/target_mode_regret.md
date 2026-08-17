# Target-Mode Rescue + Within-Block Regret

这个方法不再使用跨块 hidden steering 或 learned Router。它针对完整
`MTP=6` 草稿块执行标准概率验证，并提供三种互斥的候选资格规则：

1. `probability` 安全基线：草稿 token 的目标概率至少为 `0.5`，因此无需
   额外 reduction 就能保证它是目标分布的 mode；
2. `top1_margin` 扩展：草稿 token 必须精确等于目标 top-1，且处理后目标
   分布的 `log P(top1)-log P(top2)` 不低于给定 margin。
3. `target_band` 扩展：草稿 token 必须位于目标 top-K，并满足
   `log P(top1)-log P(y)` 不超过给定 gap。前部位置消耗的 TV 越多，后续
   位置允许的 gap 越小。

设当前位置目标概率为 `p=P(y)`、MTP 概率为 `q=Q(y)`：

1. 当 `q<=p` 时，严格接受率已经为 1，不使用预算；
2. 根据当前 eligibility 规则判断候选是否属于可松弛的目标头部；
3. 符合条件时把候选概率从 `p` 提高到 `min(q, p+b)`，其中 `b` 同时受
   单位置 TV 上限和整块 TV 上限约束；
4. 非候选 token 等比例缩小，拒绝后仍从 `(H-Q)+` 采样纠正 token；
5. bonus token 始终来自原始目标分布。

已经分配给前部位置的潜在 TV 构成当前块的 `regret debt`。债务越高，
后续位置必须具有越高的目标概率才允许继续松弛。这是当前块内的负反馈，
不会把旧 token、旧 hidden 或旧残差搬到下一块。

实现直接复用标准 rejection sampler 必需的目标 softmax。相比旧的
Cactus/Exact-TV 适配器，它不会先构造 `H` 的 logits 再让 vLLM 重算一次
softmax；目标 softmax 只计算一次。

## 单独启动

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

MTP_TOKENS=6 ./scripts/serve_target_mode_regret_mtp.sh
```

默认参数：

```text
MODE_REGRET_MIN_TARGET_PROB=0.50
MODE_REGRET_PER_TOKEN_TV=0.49
MODE_REGRET_BLOCK_TV=0.60
MODE_REGRET_DEBT_SLOPE=0.50
MODE_REGRET_DEPTH_SLOPE=0.00
MODE_REGRET_ELIGIBILITY=probability
MODE_REGRET_MIN_TOP1_MARGIN=0.00
MODE_REGRET_MARGIN_DEBT_SLOPE=0.50
MODE_REGRET_MARGIN_DEPTH_SLOPE=0.00
MODE_REGRET_MAX_TARGET_RANK=4
MODE_REGRET_MAX_CANDIDATE_GAP=0.50
MODE_REGRET_BAND_DEBT_SLOPE=0.75
MODE_REGRET_BAND_DEPTH_SLOPE=0.00
MODE_REGRET_COMPILE=1
```

`probability` 路径不会执行 top-k。`top1_margin` 路径用一次 GPU
`topk(k=2)` 同时处理六个 head，不做逐位置 CPU 循环；候选概率重分配图
默认由 `torch.compile` 融合，若本地 Torch/Triton 无法编译会自动回退到
等价的 eager 张量实现。

`target_band` 同样只执行一次批量 GPU top-k。它并不是无条件扩大预算：
候选必须位于目标头部、与 top-1 足够接近，并受到 per-token TV、block TV
和块内 debt 三重约束。

当前用于完整复验的均衡候选可以单独启动：

```bash
MTP_TOKENS=6 ./scripts/serve_target_band_mtp.sh
```

其参数是 `rank<=4`、`gap<=0.75`、单 token TV `0.30`、block TV `0.85`、
debt slope `0.50`。这只是筛选后的候选配置，必须以 164 题和第二个 seed
复验后才能称为最终版本。

## HumanEval 同协议对比

不要提前启动服务。下面的脚本会依次启动和关闭每个方法：

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

SAMPLES=164 \
TEMPERATURE=0.7 \
SEED=42 \
MTP_TOKENS=6 \
./scripts/run_humaneval_target_mode_regret.sh
```

默认对比包含纯概率 MTP、同一 fused kernel 但 `block_tv=0` 的严格 identity
对照、Cactus、SpecCascade TokenV3、此前的 Exact-TV 以及当前方法。identity
对照用于检查自定义 kernel 的 RNG、纠正采样和输出是否与标准严格验证一致。
生成代码只在隔离 Docker 容器中执行；结果和日志写入本地忽略目录，不会
上传到 GitHub。

## 完整夜跑：四组安全基线 + top-1 margin

不要提前启动服务，直接运行：

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

./scripts/run_humaneval_target_mode_nightly.sh
```

默认使用 164 道 HumanEval、`temperature=0.7`、`MTP=6`，依次运行纯 MTP、
Cactus、SpecCascade TokenV3、四组 `p>=0.5` 参数以及四组 top-1 margin。
模型生成在单 GPU 上串行计时；每个服务停止后用 4 个隔离 Docker worker
并行判题。最终统一表格在：

```text
results/humaneval_comparison_<RUN_TAG>/comparison.md
```

## Target-Band 优化筛选

下面的脚本比较纯 MTP、Cactus、SpecCascade、最佳 top-1 配置以及四档
Target-Band。默认先跑 40 题，确认质量—速度趋势后再把 `SAMPLES` 改成 164：

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

SAMPLES=40 ./scripts/run_humaneval_target_band_screen.sh
```

完整测试：

```bash
SAMPLES=164 \
RUN_TAG=target_band_full_01 \
./scripts/run_humaneval_target_band_screen.sh
```
