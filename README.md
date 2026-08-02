# ReMTP：最小可观察 Qwen3.5 MTP

这个仓库只做一件事：用 vLLM 启动 `Qwen/Qwen3.5-4B` 的 MTP，并在服务端终端逐轮打印：

1. MTP 草稿 token；
2. 目标模型对每个草稿位置生成的 token；
3. 每个草稿 token 的接受或拒绝；
4. 这一轮最终提交到输出序列的 token。

Trace 代码不修改 vLLM，也不改变解码结果，可观察贪心和非贪心验证。默认配置面向单张 RTX 4090。

## 1. 安装

要求 Linux、Python 3.10–3.13 和可用的 NVIDIA GPU。仓库固定使用已经在 RTX 4090、NVIDIA 535.247 驱动上实测通过的 `vLLM 0.18.0`。

```bash
git clone https://github.com/GladiolusZhang/ReMTP.git
cd ReMTP
./scripts/install.sh
```

安装脚本等价于：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U uv
uv pip install 'vllm==0.18.0' --torch-backend=auto
```

不要在这台 535 驱动的机器上直接换成当前 x86 nightly：截至本仓库验证时，nightly wheel 使用 CUDA 13，启动后会报 `CUDA driver version is insufficient for CUDA runtime version`。

## 2. 单独下载模型

激活环境后，用 HF Mirror 把完整 checkpoint 下载到仓库的 `models/` 目录：

```bash
source .venv/bin/activate
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir ./models/Qwen3.5-4B \
  --max-workers 4
```

也可以直接运行同一命令的封装：

```bash
source .venv/bin/activate
./scripts/download_model.sh
```

需要下载的是 `Qwen/Qwen3.5-4B` 的完整仓库，尤其不能排除 `model-*.safetensors` 和 `model.safetensors.index.json`：目标模型、视觉模块和 `mtp.*` 权重位于同一个 checkpoint。启动时的 `--language-model-only` 会跳过视觉模块加载，但最简单可靠的下载方式仍是下载完整 snapshot。

当前固定 revision 包含两个权重分片，完整 snapshot 约 8.70 GiB：

```text
model.safetensors-00001-of-00002.safetensors  5,329,398,688 bytes
model.safetensors-00002-of-00002.safetensors  3,990,429,408 bytes
model.safetensors.index.json
config/tokenizer/preprocessor 等小文件
```

默认固定 revision：

```text
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

如果 `hf-mirror.com` 临时不可用或返回跨域 308，先等待镜像恢复；确需临时改用官方端点时只需：

```bash
HF_ENDPOINT=https://huggingface.co ./scripts/download_model.sh
```

下载可中断后重跑，`hf download` 会复用已完成内容。

下载后可独立检查两个分片是否齐全：

```bash
PYTHONPATH=. python -m remtp.checkpoint ./models/Qwen3.5-4B
```

## 3. 启动 MTP

终端 1：

```bash
source .venv/bin/activate
./scripts/serve.sh
```

脚本只读取本地模型，不会在启动服务时继续下载。默认设置：

```bash
MODEL_PATH=./models/Qwen3.5-4B
SERVED_MODEL_NAME=Qwen/Qwen3.5-4B
MTP_METHOD=mtp
MTP_TOKENS=2
```

4090 启动参数等价于：

```bash
vllm serve ./models/Qwen3.5-4B \
  --served-model-name Qwen/Qwen3.5-4B \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.82 \
  --worker-cls remtp.worker.ReMTPWorker \
  --language-model-only \
  --enforce-eager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

只验证单步 MTP：

```bash
MTP_TOKENS=1 ./scripts/serve.sh
```

如果模型下载到了其他磁盘：

```bash
MODEL_PATH=/data/models/Qwen3.5-4B ./scripts/serve.sh
```

## 4. 发送请求并看中间过程

终端 2：

```bash
./scripts/request.sh
```

终端 1 会出现类似日志：

```text
[ReMTP][round 001][request cmpl-...]
  MTP draft       : D0=3891(' future')  D1=315(' of')
  TARGET verify   : T0=3891(' future')  T1=315(' of')  bonus=1492(' AI')
  VERIFY          : D0==T0 ✓  D1==T1 ✓  -> accepted 2/2
  COMMIT          : 3891(' future') 315(' of') 1492(' AI')
```

若第二个草稿不匹配：

```text
  VERIFY          : D0==T0 ✓  D1!=T1 ✗  -> accepted 1/2
  COMMIT          : <D0> <目标模型在失败位置选择的 token>
```

这里 `TARGET verify` 是目标模型一次 forward 在各草稿位置产生的候选 token；全接受时还会产生一个 `bonus` token。验证只接受从 `D0` 开始的最长连续合法前缀，首次失败后其余草稿作废。`COMMIT` 是这一轮真正加入输出序列的 token。

为了让 `Dk==Tk` 的对照关系完全直观，示例请求固定使用 `temperature=0`，且不设置惩罚项、top-k 或 top-p。Trace 最多打印 64 轮，可通过 `REMTP_TRACE_MAX_ROUNDS=10` 调小。

### 非贪心验证

服务端启动方式不变。在终端 2 设置非零温度和固定随机种子：

```bash
TEMPERATURE=0.8 SEED=42 ./scripts/request.sh
```

终端 1 会直接显示概率接受过程：

```text
[ReMTP][round 001][request 0][stochastic]
  MTP draft       : D0=1558(' model')  D1=1179(' then')
  TARGET verify   : p(D0)=0.8000  p(D1)=0.3500
  VERIFY D0       : q=1.0000  α=min(1,p/q)=0.8000  u=0.3000  -> ACCEPT ✓
  VERIFY D1       : q=1.0000  α=min(1,p/q)=0.3500  u=0.6000  -> REJECT ✗
  DRAFT q         : deterministic MTP proposal, q(Dk)=1
  RECOVER         : 264(' can')
  COMMIT          : 1558(' model') 264(' can')
```

其中 `p(Dk)` 是目标模型给草稿 token 的概率，`q(Dk)` 是草稿分布给它的概率，`α=min(1,p/q)` 是接受概率，`u` 是本轮均匀随机数。原生 `serve.sh` 路径中的 vLLM MTP proposer 固定取草稿 top-1，等价于确定性草稿分布，所以日志中的 `q(Dk)=1`。若 `u≤α` 就接受；首次拒绝后从残差分布采样 `RECOVER` 并停止验证后续草稿。两个草稿全接受时则显示并提交 `BONUS`。需要完整 MTP 分布时使用第 7 节新增的概率 MTP 路径。

## 5. 常见问题

- 服务启动时报 MTP method 不支持：确认安装的是仓库固定的 `vLLM 0.18.0`，并保持默认 `MTP_METHOD=mtp`。
- 启动提示 `Local model not found`：先运行 `./scripts/download_model.sh`，或通过 `MODEL_PATH=/实际目录` 指定本地 snapshot。
- CUDA/驱动报错：确认 `python -c 'import vllm; print(vllm.__version__)'` 输出 `0.18.0`；若不是，删除并重建 `.venv`。
- 显存不足：保持 `--language-model-only`，再将 `GPU_MEMORY_UTILIZATION=0.75` 或 `MAX_MODEL_LEN=2048`。
- 没有 Trace 日志：必须通过 `./scripts/serve.sh` 启动；该脚本会加载 `remtp.worker.ReMTPWorker`，使只读 hook 进入实际执行 GPU 验证的 EngineCore 进程。

## 6. Spec-Bench 子集测速

测试默认固定为：

- `temperature=0.7`；
- `MTP_TOKENS=2`；
- `translation`、`summarization`、`math_reasoning`、`rag` 四个任务；
- 每个任务用固定随机种子抽样 20 条，共 80 条；
- 每条最多生成 128 tokens，单请求串行运行。

数据集不包含在本仓库中。你只需自行把 Spec-Bench 官方
`data/spec_bench/question.jsonl` 放到：

```text
data/spec_bench/question.jsonl
```

例如可自行执行：

```bash
mkdir -p data/spec_bench
curl -L \
  https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl \
  -o data/spec_bench/question.jsonl
wc -l data/spec_bench/question.jsonl
```

最后一条命令应显示 480 行。

测速时不要使用逐轮 Trace，因为打印中间 tensor 会触发 GPU 到 CPU
同步。测速脚本保留 `torch.compile`，但会关闭当前软件栈中与 Qwen3.5 GDN
不兼容的 CUDA Graph；它没有使用 `--enforce-eager`。先停止当前观察用
服务，再在终端 1 启动原生 vLLM MTP：

```bash
source .venv/bin/activate
MTP_TOKENS=2 ./scripts/serve_benchmark.sh
```

首次启动需要完成 `torch.compile` 和 Triton kernel 预热。看到服务监听
8000 端口后，
在终端 2 运行：

```bash
./scripts/benchmark_specbench.sh
```

脚本会先做一次不计入结果的短预热，然后逐条请求。结果写到
`results/spec_bench_mtp_t0.7_时间戳/`：

```text
summary.md             人类可读汇总
summary.csv            每个任务及 overall 的表格
summary.json           配置和完整汇总
requests.jsonl         每个请求的输出、耗时和指标增量
sample_manifest.json   实际抽到的 question_id
```

汇总同时报告三个核心指标：

- `decode tok/s`：vLLM 生成 token 增量 ÷ 服务端 decode 时间增量，作为原生
  MTP 解码吞吐；
- `e2e output tok/s`：输出 token 数 ÷ 客户端墙钟时间，包含 prefill、调度和
  HTTP 开销；
- `mean acceptance length`：`1 + 接受的草稿 token 数 / MTP 验证轮数`。
  其中 `1` 是每轮目标模型保证产生的 token，因此它的范围是
  `[1, MTP_TOKENS + 1]`；当 `MTP_TOKENS=2` 时范围为 `[1, 3]`。

常用覆盖参数：

```bash
SAMPLES_PER_TASK=20 TEMPERATURE=0.7 SEED=42 \
MAX_TOKENS=128 MTP_TOKENS=2 \
./scripts/benchmark_specbench.sh
```

如果数据放在其他位置：

```bash
SPEC_BENCH_DATA=/data/spec_bench/question.jsonl \
./scripts/benchmark_specbench.sh
```

如需显式写出测速脚本的稳定编译配置：

```bash
REMTP_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}' \
MTP_TOKENS=2 ./scripts/serve_benchmark.sh
```

## 7. Speculative Cascade + MTP

实验分支还实现了论文 *Faster Cascades via Speculative Decoding*
的 TokenV3 机制，并继续使用 Qwen3.5 自带的 MTP 作为草稿器。

当前 vLLM 0.18 的 Qwen3.5 hybrid runner 默认只向验证器提供 MTP argmax
草稿 token。本分支新增了完整分布适配：

```text
MTP logits -> q=softmax(logits/temperature) -> D~q
标准 MTP：min(1, p(D)/q(D))
TokenV3：先由完整 p、q 构造 π，再用 min(1, π(D)/q(D))
```

先测试完整 `q` 的标准概率 MTP：

```bash
MTP_TOKENS=2 ./scripts/serve_probabilistic_mtp.sh
# 另一个终端
./scripts/benchmark_probabilistic_mtp.sh
```

再启动 TokenV3、`alpha=0.5` 的服务：

```bash
CASCADE_RULE=token_v3 CASCADE_ALPHA=0.5 \
MTP_TOKENS=2 ./scripts/serve_spec_cascade.sh
```

在另一个终端运行与原生 MTP 完全相同的 Spec-Bench 子集：

```bash
CASCADE_RULE=token_v3 CASCADE_ALPHA=0.5 \
./scripts/benchmark_spec_cascade.sh
```

统一三方法、同一批 80 条样本的固定种子实验中，TokenV3 相对标准概率 MTP
的整体 decode 吞吐为 `149.371 → 150.029 tok/s`（`+0.44%`，基本持平），
平均接受长度为 `2.519 → 2.564`，草稿接受率为
`75.97% → 78.18%`。完整机制说明见
[docs/speculative_cascade_mtp.md](docs/speculative_cascade_mtp.md)，逐任务结果和
实验限制见
[reports/spec_cascade_mtp_specbench_t0.7.md](reports/spec_cascade_mtp_specbench_t0.7.md)。

## 8. Cactus + 概率 MTP

本分支还将 ICLR 2026 论文 *Cactus: Accelerating Auto-Regressive
Decoding with Constrained Acceptance Speculative Sampling* 适配到完整概率
MTP。对当前草稿 token `D`：

```text
gamma = min(p(D) + sqrt(2 delta p(D)(1-p(D))), 1)
h(D) = gamma，其余 p(v) 按比例缩放
接受率 = min(1, h(D) / q_mtp(D))
拒绝恢复分布 = normalize(max(h - q_mtp, 0))
```

`delta=0` 精确恢复第 7 节的标准非松弛概率 MTP。论文在 Spec-Bench
使用 `delta=1` 且不做任务级调参，本仓库也将其作为默认值。

启动和测试：

```bash
CACTUS_DELTA=1.0 MTP_TOKENS=2 ./scripts/serve_cactus_mtp.sh

# 另一个终端
CACTUS_DELTA=1.0 ./scripts/benchmark_cactus_mtp.sh
```

统一三方法、同一批 80 条样本的固定种子实验中，Cactus 相对标准概率 MTP
的整体 decode 吞吐为 `149.371 → 163.999 tok/s`（`+9.79%`），e2e
吞吐为 `128.957 → 139.747 tok/s`（`+8.37%`），平均接受长度为
`2.519 → 2.784`。机制和适配说明见
[docs/cactus_mtp.md](docs/cactus_mtp.md)，逐任务结果和实验限制见
[reports/cactus_mtp_specbench_t0.7.md](reports/cactus_mtp_specbench_t0.7.md)。

三种方法的统一协议对比表见
[reports/three_way_mtp_specbench_t0.7.md](reports/three_way_mtp_specbench_t0.7.md)。
当前总体结果如下：

| 方法 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 |
|---|---:|---:|---:|---:|
| 标准概率 MTP | 149.371 | 128.957 | 2.519 | 75.97% |
| SpecCascade [TokenV3] | 150.029 | 129.174 | 2.564 | 78.18% |
| Cactus + MTP | **163.999** | **139.747** | **2.784** | **89.18%** |

## 9. GSM8K 子集质量—效率评测

下载 OpenAI 官方 GSM8K test split：

```bash
mkdir -p data/gsm8k
curl -L \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
  -o data/gsm8k/test.jsonl
```

启动任意一种服务后，在另一个终端运行对应入口：

```bash
# 标准概率 MTP
./scripts/benchmark_gsm8k_probabilistic_mtp.sh

# SpecCascade [TokenV3]
./scripts/benchmark_gsm8k_spec_cascade.sh

# Cactus
./scripts/benchmark_gsm8k_cactus_mtp.sh
```

默认从 test split 固定抽样 100 条，使用 `temperature=0.7`、
`MTP_TOKENS=2` 和 384-token 上限。可覆盖参数：

```bash
SAMPLES=200 SAMPLE_SEED=20260730 TEMPERATURE=0.7 \
SEED=42 MAX_TOKENS=384 MTP_TOKENS=2 \
./scripts/benchmark_gsm8k_probabilistic_mtp.sh
```

当前 100 条统一子集结果：

| 方法 | 准确率 | decode tok/s | e2e tok/s | 平均接受长度 |
|---|---:|---:|---:|---:|
| 标准概率 MTP | **89.0%** | 151.222 | 144.663 | 2.548 |
| SpecCascade [TokenV3] | 88.0% | 153.602 | 146.389 | 2.611 |
| Cactus + MTP | 88.0% | **166.779** | **158.784** | **2.819** |

完整协议、置信区间、截断率与配对质量检查见
[reports/gsm8k_three_way_t0.7.md](reports/gsm8k_three_way_t0.7.md)。

完成一次可运行实验后，建议记录确切版本：

```bash
python -c 'import vllm; print(vllm.__version__)'
uv pip freeze > environment.lock.txt
```

论文实验应固定 vLLM wheel/commit、模型 revision 和启动参数。

## 10. Target-Anchored Exact-TV MTP（MTP=6）

本分支用目标模型 margin、MTP head 可靠度和当前 hidden cosine 分配
CACTUS 的块级 TV；未来位置仅能否决前部预算。实现不再计算词表级 JS、
top-K 并集或 tail bucket。

统一入口依次测试五种方法：

```text
CACTUS
SpecCascade TokenV3
CACTUS + h(y) <= q(y)
上述上限 + 精确 TV 块重分配 + head reliability
上述方法 + 当前 hidden cosine + future target-support veto
```

一次运行相同的 200 条 GSM8K 样本：

```bash
SAMPLES=200 TEMPERATURE=0.7 SEED=42 MTP_TOKENS=6 \
./scripts/run_gsm8k_mtp6_five_way.sh
```

只启动完整方法：

```bash
TARGET_ANCHORED_VARIANT=tv_hidden_veto \
MTP_TOKENS=6 ./scripts/serve_target_anchored_mtp.sh
```

默认 head 先验为 `1.0,0.85,0.70,0.55,0.40,0.30`，可用
`HEAD_RELIABILITY=a,b,c,d,e,f` 替换为校准集统计值。对一份独立校准集
的标准概率 MTP=6 服务日志，可直接计算六个 head 的条件可靠度：

```bash
python -m remtp.head_reliability \
  logs/calibration_server.log --mtp-tokens 6
```

实验输出仅写入本地 `results/` 和 `logs/`，这两个目录不会上传到
GitHub。

## 11. Budget-Induced Regret Feedback（MTP=6）

该实验保持上一节的 `tv_hidden_veto` 验证器不变。只有在同一个随机数
`u` 下发生“严格验证拒绝、松弛验证接受”时，才从 `P -> H` 的实际 TV
转移构造短期遗憾方向。记忆最多跨两个 block，并且只注入下一轮 MTP
Module 1 使用的目标 hidden 副本；目标模型 hidden、KV、logits 和验证
分布均不修改。

单独启动：

```bash
MTP_TOKENS=6 \
REGRET_ALPHA=0.03 \
REGRET_TOKEN_DECAY=0.90 \
./scripts/serve_regret_feedback_mtp.sh
```

把新方法跑在同一批 200 条 GSM8K 上，并追加到已经完成的五方法本地
表格：

```bash
BASELINE_RUN_ROOT=results/gsm8k_mtp6_five_way_20260730_202927 \
SAMPLES=200 TEMPERATURE=0.7 SEED=42 MTP_TOKENS=6 \
./scripts/run_gsm8k_mtp6_regret_append.sh
```

若省略 `BASELINE_RUN_ROOT`，脚本会选择最新的完整五方法结果。运行结束
后，本地结果除六行对比表外还包含 `regret_mechanism.json`，其中记录
严格可接受比例、causal relaxed acceptance、单位接受 token 的 TV、
注入次数和六个 MTP head 的机制统计。

该跨块 hidden-steering 版本保留在
`research/future-only-regret-exploration`，作为 future-only 机制的探索记录。

## 12. 当前块验证债务控制（MTP=6）

新方法不再把主要风险反馈推迟到下一块。它把松弛造成的接受概率增量

```text
debt_i = A_relaxed(i) - A_strict(i)
```

作为当前位置的验证债务，在 token 提交前限制单位置预算、目标明显反对
的候选和块内累计债务；被裁掉的 Exact-TV 预算仍可回收到更安全且前缀
价值更高的位置。达到风险阈值后，主方法停止后续松弛并退回严格验证，
另提供 CACTUS fallback 消融。

在两个固定 GSM8K seed 上执行 Pareto 门槛：

```bash
SAMPLES=100 TEMPERATURE=0.7 SEED=42 MTP_TOKENS=6 \
./scripts/run_gsm8k_current_block_debt_gate.sh
```

第一 seed 只有同时达到 `Accuracy > CACTUS`、`MAL > CACTUS`、
`E2E >= CACTUS` 的配置才进入第二 seed。加入低成本 posterior scale 和
top-1 bias 辅助消融：

```bash
INCLUDE_WEAK_FEEDBACK=1 SAMPLES=100 \
./scripts/run_gsm8k_current_block_debt_gate.sh
```

算法和参数说明见
[docs/current_block_verification_debt.md](docs/current_block_verification_debt.md)。
