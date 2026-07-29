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

其中 `p(Dk)` 是目标模型给草稿 token 的概率，`q(Dk)` 是草稿分布给它的概率，`α=min(1,p/q)` 是接受概率，`u` 是本轮均匀随机数。当前 vLLM 的 MTP proposer 固定取草稿 top-1，等价于确定性草稿分布，所以日志中的 `q(Dk)=1`。若 `u≤α` 就接受；首次拒绝后从残差分布采样 `RECOVER` 并停止验证后续草稿。两个草稿全接受时则显示并提交 `BONUS`。

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

当前 vLLM 0.18 的 Qwen3.5 hybrid runner 只向验证器提供 MTP argmax
草稿 token，因此实现采用确定性提议 `q(D)=1`。对每个草稿 `D`：

```text
如果 p_unscaled(D) >= (1 - alpha) * max(p_unscaled)：
    接受 MTP 草稿 D
否则：
    按目标模型 p 做标准验证和恢复采样
```

启动 TokenV3、`alpha=0.5` 的服务：

```bash
CASCADE_RULE=token_v3 CASCADE_ALPHA=0.5 \
MTP_TOKENS=2 ./scripts/serve_spec_cascade.sh
```

在另一个终端运行与原生 MTP 完全相同的 Spec-Bench 子集：

```bash
CASCADE_RULE=token_v3 CASCADE_ALPHA=0.5 \
./scripts/benchmark_spec_cascade.sh
```

本机单次固定种子实验中，整体 decode 吞吐从 `152.48` 提升到
`165.48 tok/s`（`+8.53%`），平均接受长度从 `2.403` 提升到
`2.637`。完整机制说明见
[docs/speculative_cascade_mtp.md](docs/speculative_cascade_mtp.md)，逐任务结果和
实验限制见
[reports/spec_cascade_mtp_specbench_t0.7.md](reports/spec_cascade_mtp_specbench_t0.7.md)。

完成一次可运行实验后，建议记录确切版本：

```bash
python -c 'import vllm; print(vllm.__version__)'
uv pip freeze > environment.lock.txt
```

论文实验应固定 vLLM wheel/commit、模型 revision 和启动参数。
