# Target-Mode Rescue + Within-Block Regret

这个方法不再使用跨块 hidden steering 或 learned Router。它针对完整
`MTP=6` 草稿块执行标准概率验证，只在一个严格受限的情形下松弛：当前
草稿 token 在目标分布中的概率至少为 `0.5`。因此该 token 必然也是目标
分布的 mode（并列时为 mode 之一）。

设当前位置目标概率为 `p=P(y)`、MTP 概率为 `q=Q(y)`：

1. 当 `q<=p` 时，严格接受率已经为 1，不使用预算；
2. 当 `p<0.5` 时，保持标准推测解码；
3. 其余情况把候选概率从 `p` 提高到 `min(q, p+b)`，其中 `b` 同时受
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
```

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
