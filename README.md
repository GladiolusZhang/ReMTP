# RAVEN / ReMTP

> 第三层开销优化与通宵结果（2026-11-11，UTC+8）：通宵 GSM8K-100 上优化后 RAVEN 89/100、E2E 2.773 tok/s、MAL 3.471（原 RAVEN 2.612，Cactus 2.833）；Spec-Bench 70 次请求 E2E 2.336 对 Cactus 2.306（单次运行，不作显著性声明）。下午完成第三层（跳过死的草稿 KV、树元数据缓存、原地路径提交），三道 GSM8K 题逐 token 等价、每轮 1.2385 → 1.2155 s；4090 未测。汇总与移植步骤见 `doc/PORTING-RAVEN-OVERHEAD-OPTIMIZATIONS-2026-11-11.md`，全天实验输出 `results/day_20261111/`。

> 每轮开销优化（2026-11-10，UTC+8）：定位出 RAVEN 比 Cactus 每轮多约 172 ms 的来源——canonical 树注意力逐节点扫前缀（随上下文增长）与 Python 编排里约 100 次主机/设备同步（固定）。两层等价优化已实机验证：`REMTP_CANONICAL_TREE_ATTENTION=0`（原生 qq_bias 树注意力，探针确认掩码正确、输出至多差 1 个 bf16 ulp）和 `REMTP_DYNAMIC_TREE_FAST_SELECT=1`/`REMTP_DYNAMIC_TREE_FAST_VERIFY=1`（输出与参考逐 token 相同）。三题实测每轮 1.306→1.241 s，仍未追平 Cactus；第三层候选与全部数字见 [结果文档](doc/RESULT-RAVEN-OVERHEAD-OPTIMIZATION-2026-11-10.md)。默认开关全部保持原值。
>
> 采用结论更新（2026-10-14，UTC+8）：按用户转述的组长意见，deep4_conservative（D4/N12）因树预算大于原Adaptive RAVEN（D3/N9），不予采用。下方D4对D3的数值改善保留为历史配置实验，不能作为相同树预算的方法改进证据。后续固定D3/N9优化，目标仍为正确率不下降且速度超过Native、Cactus、SpecCascade。[相同树预算优化计划](plan/adaptive-raven-equal-budget-2026-10-14.md)。

> 完整结果（2026-10-13，UTC+8）：100+100五方法20:59:11正常结束，1000生成/500判题、exit0，独立CPU核验通过。deep在两任务点结果上超过Native；对Cactus仍少1/2道正确且吞吐低5.41%/6.63%，对Spec整题时间更长，目标尚未达到。原Adaptive的GSM8K正确率最高88%；deep的HumanEval63%、相对原方法整题−15.05%。[中文完整结果与分析](doc/RESULT-RAVEN-100X100-2026-10-13.md)，[CSV](results/raven_full_100x100_20261010/summary.csv)。当前无后台实验，下方恢复/运行描述为历史快照。

> 续跑更新（2026-10-12，UTC+8）：断电重启后600生成/230判题完整核验并保留，12:57:46后台恢复；关机前的Spec44候选代码OOM误分类已单独修复登记，原冻结生成代码/数据/种子不变。SpecCascade判题已补齐，总判题300/500，正在继续两RAVEN的余400生成。[恢复记录与再次续跑命令](doc/RESUME-RAVEN-100X100-2026-10-12.md)，[实时报告](doc/RESULT-RAVEN-100X100-LIVE.md)。下方启动/停止状态均保留为当时快照。

> 当前正式复现（2026-10-10）：用户确认原协议GSM8K100+HumanEval100及原种子，11:22:57后台已启动；100参考解与错误控制判题通过，Native health/预热/Worker核对通过，正式生成进行中。[启动配置与验收](doc/RESULT-RAVEN-100X100-2026-10-10.md)，[实时中文报告](doc/RESULT-RAVEN-100X100-LIVE.md)，进度`bash scripts/status_raven_full_benchmark.sh`。本轮只运行三基线、优化前原方法和deep4_conservative，不自动继续优化；下方“无新后台”等状态均为历史快照。

> 最新完成（2026-10-10，UTC+8）：优势搜索00:32:59正常结束，112/112生成、7启动预热、exit0、0重试。当前修复配置下deep4_balanced对原B正确12→16/16、吞吐+2.36%、整题−4.99%；conservative14/16、整题−16.56%。两个D4通过对B/Native开发点门槛，仍未同时超过Cactus吞吐与SpecCascade整题耗时。独立CPU核验112payload/64复现/12比较通过。[中文结果与分析](doc/RESULT-RAVEN-DOMINANCE-2026-10-10.md)。当前无新后台；以下运行/准备状态按当时历史保留。

> 当前后台（2026-10-09 11:24 UTC+8核验）：新目标112完整生成搜索已运行，原B健康/预热通过、首题1532/42生成中，尚无新质量/速度结论；22项CPU/真实preflight/source hash通过。[实时中文报告](doc/RESULT-RAVEN-DOMINANCE-LIVE.md)，查询bash scripts/status_raven_dominance.sh。以下“预检中”为启动前历史状态。

> 当前目标更新（2026-10-09）：用户明确不要求减少截断，只以原评分正确率与实际速度超过Native/Cactus/SpecCascade评价。新门槛和三个Adaptive候选的112完整请求搜索已实现，CPU预检中；[新执行计划](plan/adaptive-raven-dominance-2026-10-09.md)，[目标/优化范围](doc/OBJECTIVE-RAVEN-DOMINANCE-2026-10-09.md)。以下截断/正常结束硬门槛属于旧目标，历史结果保留。

> 最新研究（2026-10-09）：重新以原Adaptive RAVEN（B）为基准。CPU核验定位5处真实验收翻转，26组共同前缀完整帧P/Q/RNG一致；新计划先验证B行为等价计算优化，再独立研究完整分布质量约束。[失败机制与截断解释](doc/RESEARCH-RAVEN-BASELINE-2026-10-09.md)；[步骤、预期效果与有限预算](plan/adaptive-raven-baseline-optimization-2026-10-09.md)。研究完成，新GPU尚未启动，最终三基准目标仍未达成。

> 最新完结（2026-10-09核验）：根因与target_odds修复单元已在10月8日正常完成，exit0、0重试。80条五方法校准完整：原B12判对/10正常结束判对/6截断，候选10/8/8；候选对B吞吐-1.16%、整题+16.87%，对三基准联合门槛均未通过，未采用。两单元各12现场Target raw、48 CUDA验收及独立CPU核验通过。[中文结果与分析](doc/RESULT-RAVEN-TARGET-ODDS-2026-10-09.md)。以下准备/运行状态为历史；当前没有新后台实验，最终目标仍未达成。

> 当前定位与修复（2026-10-08）：原B根因诊断09:26 UTC+8冻结启动，10:11完成4/6自然轨迹；已证明预算可能放大局部偏差。Target相对支持约束已实现，130项算法/87项最终控制器CPU检查和冻结check-only通过，10:17:42后台轻量等待依赖成功后自动重跑。[中文根因](doc/ROOT-CAUSE-ADAPTIVE-RAVEN-2026-10-08.md)；[修复计划](plan/adaptive-raven-target-odds-repair-2026-10-08.md)；[实时中文报告](doc/RESULT-RAVEN-TARGET-ODDS-REPAIR-LIVE.md)，查询bash scripts/status_raven_risk_repair.sh。下方“当前无后台”只描述上一单元完结时，最终质量/速度目标仍未达成。

> 最新完成（2026-10-08）：复审证据单元06:25 UTC+8正常结束，74主HTTP完成；五个实际晚段raw门槛和B16输出签名复现通过。四方法校准B原判对12/16、截断6/16，吞吐低于Cactus9.70%，质量/速度目标未达成。[中文结果与原因分析](doc/RESULT-RAVEN-REVIEW-EVIDENCE-2026-10-08.md)；[控制器报告](doc/RESULT-RAVEN-REVIEW-LIVE.md)。当前无后台实验，`bash scripts/status_raven_review_unit.sh`用于查询已结束单元。

> 当前审查入口（2026-10-07）：[全项目审查、目标与优化判断](doc/REVIEW-ADAPTIVE-RAVEN-PROJECT-2026-10-07.md)；[新的执行优先级与有限预算](plan/adaptive-raven-project-review-optimization-2026-10-07.md)。MTP修复已完成，懒残差没有实测加速；cap2因质量门槛拒绝。当前仍未完成新数值配置下的三基准正式比较。

本分支的当前方法是面向 native MTP 的 RAVEN：以 FastMTP 的完整 proposal
distribution 构造三步主路径和稀疏备选树，在一次目标模型 Tree Attention
验证后进行候选条件化概率接受；主路径拒绝时先从残差分布采样 correction，随后通过
token-ID 精确匹配复用已验证 sibling 状态。当前推荐配置进一步加入接受饱和截断和
块内 actual-TV 自适应分配。

研究过程、失效路线、协议变化和验证状态统一记录在
[研究与开发总日志](docs/research_and_development_log.md)。模型、数据、实验结果与运行日志
保存在本地忽略目录，不随代码上传。

Jetson Orin NX（R39.2.1）的 4 位 FastMTP + BF16 草稿层适配、实测结果和
运行限制见 [设备适配记录](doc/RESULT-RAVEN-R39.2.1.md)。该设备的服务入口是
`./scripts/serve_fastmtp_jetson.sh`。

2026-10-04 的历史 Target 数值一致性修复见
[中文修复记录](doc/FIX-TARGET-PROBABILITY-CONSISTENCY-2026-10-04.md)。
已拒绝cap2单元的历史状态入口为 `bash scripts/status_raven_probability_cap.sh`；
该单元结果见 [历史中文阶段分析](doc/RESULT-RAVEN-PROBABILITY-CAP-2026-10-07.md)。
第三轮查询脚本仅用于历史单元。新的审查计划第一单元已完成；当前数据未满足三个主基线的质量/速度门槛，独立正式测试尚未执行。

## 当前推荐流程：FastMTP + RAVEN

### 1. 获取代码与安装环境

```bash
git clone --branch research/residual-aligned-tree-mtp \
  https://github.com/GladiolusZhang/ReMTP.git
cd ReMTP
./scripts/install.sh
source .venv/bin/activate
```

当前适配使用 vLLM 0.18.0、单请求执行和 NVIDIA GPU。实验脚本会检查端口、服务
Worker 标记和输出完整性，端口 8000 被其他服务占用时会停止运行。

### 2. 下载 FastMTP checkpoint

默认从 Hugging Face 官方端点下载到 `models/FastMTP`：

```bash
source .venv/bin/activate
HF_ENDPOINT=https://huggingface.co ./scripts/download_fastmtp.sh
```

如果 checkpoint 已经位于其他目录，运行实验时设置：

```bash
MODEL_PATH=/absolute/path/to/FastMTP
```

FastMTP checkpoint 约 15.7GB，包含一个训练后的物理 MTP layer。实验中的三个逻辑
proposal 位置递归复用该层。

### 3. 准备 GSM8K 与 HumanEval

```bash
mkdir -p data/gsm8k
curl -L \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
  -o data/gsm8k/test.jsonl

./scripts/setup_humaneval.sh
./scripts/download_humaneval.sh
docker pull python:3-slim
```

HumanEval 生成和执行相互分离；候选代码在受限 Docker 容器中运行，判题时间不计入
模型生成吞吐。

### 4. 环境检查与 CPU 测试

```bash
./scripts/validate_fastmtp_setup.sh

python -m pytest -q \
  tests/test_dynamic_mtp_tree.py \
  tests/test_raven_adaptive_relaxation_report.py \
  tests/test_raven_depth_relaxation_report.py
```

### 5. 快速冒烟测试

运行 GSM8K 5 条和 HumanEval 5 条：

```bash
SAMPLES=5 \
PROGRESS_EVERY=1 \
RUN_TAG=raven_adaptive_smoke \
./scripts/run_fastmtp_raven_adaptive_relaxation.sh
```

脚本依次比较：

- Native FastMTP；
- Cactus + FastMTP；
- SpecCascade TokenV3 + FastMTP；
- uniform RAVEN；
- saturation-aware adaptive RAVEN。

Target-only 不在默认对比中。所有方法使用同一 FastMTP checkpoint、相同抽样题目、
temperature 0.6、generation seed 42、空 system prompt 和 no-thinking 模板。

### 6. 运行 100+100 正式对比

```bash
SAMPLES=100 \
PROGRESS_EVERY=1 \
RUN_TAG=raven_adaptive_n100 \
./scripts/run_fastmtp_raven_adaptive_relaxation.sh
```

运行中断后，重复完全相同的命令即可从请求级 checkpoint 继续；完成的方法和数据集会
自动跳过。若端口 8000 已被其他服务占用，先停止该服务再重新运行。

结果写入本地：

```text
results/raven_adaptive_n100/comparison.md
results/raven_adaptive_n100/{uniform,adaptive}/gsm8k/
results/raven_adaptive_n100/{uniform,adaptive}/humaneval/
logs/raven_adaptive_n100/{uniform,adaptive}/
```

其中汇总表报告任务质量、MAL、decode/E2E tok/s、草稿接受率、nodes/round、逐深度
acceptance 和块级 actual-TV。`results/` 与 `logs/` 已被 `.gitignore` 排除。

### 7. 关键参数

推荐默认值由统一入口固定：

```text
MTP depth                 3
maximum tree nodes        9
maximum children          3
base relaxation delta     1.0
adaptive mix              0.5
per-position TV cap       1.5 x baseline
target-support exponent   0.25
allocation steps          16
```

可以在命令行覆盖自适应松弛参数：

```bash
SAMPLES=20 \
RAVEN_BASE_DELTA=1.0 \
ADAPTIVE_MIX=0.5 \
ADAPTIVE_MAX_RATIO=1.5 \
ADAPTIVE_TARGET_POWER=0.25 \
ADAPTIVE_STEPS=16 \
RUN_TAG=raven_custom \
./scripts/run_fastmtp_raven_adaptive_relaxation.sh
```

## 历史功能：可观察 Qwen3.5 MTP

这个仓库最初用于用 vLLM 启动 `Qwen/Qwen3.5-4B` 的 MTP，并在服务端终端逐轮打印：

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

后续统一实验主表始终包含纯概率 MTP、参考论文方法和本仓库方法；如果还
保留旧版本，它只作为额外消融，不替代纯 MTP 基线。

## 13. Cactus-dominant target surplus（MTP=6）

该版本保持每个位置的 Cactus 候选接受率作为下界，回收已经超过
`q_mtp(y)`、无法继续增加该位置接受率的候选概率质量。回收的 Exact-TV
surplus 按前缀顺序分配给目标模型相对 log-prob gap 不超过2.0且尚未
饱和的位置。严格 top-1 版本保留为消融；真实 audit 表明在 `delta=1`
时它几乎没有可接收位置，因此主版本采用 target-support margin。

正式四方法测试包含纯概率 MTP、Cactus、旧 balanced debt 和新方案，
并强制关闭 audit：

```bash
SAMPLES=100 TEMPERATURE=0.7 SEED=42 MTP_TOKENS=6 \
./scripts/run_gsm8k_target_surplus_gate.sh
```

单独运行10条机制审计：

```bash
SAMPLES=10 ./scripts/run_gsm8k_target_surplus_audit.sh
```

## 14. Target-anchored Cactus risk swap（MTP=6）

Risk swap 取消全块 stop：目标支持位置保留 Cactus 有效预算，目标反对
越强则连续削减；削减预算和候选饱和浪费一起，按草稿前缀顺序重新投入
安全位置。

主测试仍固定包含纯概率 MTP、Cactus、旧质量消融以及新方案：

```bash
SAMPLES=100 TEMPERATURE=0.7 SEED=42 MTP_TOKENS=6 \
./scripts/run_gsm8k_target_anchored_gate.sh
```

正式测试始终关闭机制审计；结果和日志仅写入本地忽略目录。

同一入口还测试 target-recovery 消融：松弛接受和目标 bonus 保持不变，
但第一次拒绝后的纠正 token 从原始目标分布采样，而不是松弛残差
`(H-Q)+`。该消融用于隔离接受决策与 RECOVER 分布造成的质量变化。

## 15. HumanEval 质量—速度评测（MTP=6）

HumanEval 采用官方功能测试计算 `pass@1`，允许生成实现与参考代码不同。

最新的 Exact-TV + Expected-Regret Router 保持 CACTUS 总 TV 上界，使用
期望接受责任形成短期遗憾，并通过冻结模型外的小型低秩 Router 同时校准
下一块 MTP proposal 与 Exact-TV 预算。实现和运行方法见
[docs/exact_tv_regret_router.md](docs/exact_tv_regret_router.md)。
Router 混合语料、统一 benchmark 去污染和请求级训练/验证切分见
[docs/regret_router_data.md](docs/regret_router_data.md)。
生成和执行完全分离：模型代码只在无网络、只读、限资源的临时 Docker
容器内执行，判题时间不计入生成吞吐。

```bash
./scripts/setup_humaneval.sh
./scripts/download_humaneval.sh

# 先验证三种方法和完整执行链路
SAMPLES=5 \
PROFILES="native_mtp cactus regret_calibrated_block" \
./scripts/run_humaneval_comparison.sh

# 再跑默认九方法、全部 164 题
SAMPLES=164 TEMPERATURE=0.7 SEED=42 MTP_TOKENS=6 \
./scripts/run_humaneval_comparison.sh
```

默认主表包含纯概率 MTP、Cactus、SpecCascade、Block Verification 论文
基线、此前具有代表性的 Exact-TV 变体以及当前 Regret-Calibrated Block
方法。完整说明见 [docs/humaneval.md](docs/humaneval.md)。数据、生成代码、
判题结果和日志均只保存在本地忽略目录。

## 16. Target-Mode Rescue + 块内遗憾（MTP=6）

当前大改版本停止跨块 Router/hidden steering。它只救援目标概率不低于
`0.5` 的 MTP 候选，并把前部位置使用的 TV 作为块内遗憾债务，逐步提高
后续位置的松弛门槛。自定义验证路径只计算一次目标 softmax。

完整算法与运行命令见
[docs/target_mode_regret.md](docs/target_mode_regret.md)。一键 HumanEval 对比：

```bash
SAMPLES=164 TEMPERATURE=0.7 SEED=42 MTP_TOKENS=6 \
./scripts/run_humaneval_target_mode_regret.sh
```

若只研究“Cactus verifier 不变、遗憾残差仅引导下一块 MTP 草稿”，运行：

```bash
# 40 条固定子集筛查
./scripts/run_humaneval_cactus_regret.sh

# 锁定参数后的全量测试
SAMPLES=164 CACTUS_REGRET_ALPHA=0.03 \
./scripts/run_humaneval_cactus_regret.sh
```

This profile leaves the Cactus verifier unchanged and projects the previous
block's target-side `P-H` regret through the shared output head to correct
only the next block's first MTP proposal. Its
default posterior causal responsibility is continuous; use
`CACTUS_REGRET_RESPONSIBILITY=realized` for the binary-event ablation.
Experimental feature-space variants are available with
`CACTUS_REGRET_INJECTION_SITE=root` and
`CACTUS_REGRET_RESIDUAL_SPACE=boundary_hidden` or `hidden`.

该表固定包含纯概率 MTP、Cactus、SpecCascade 和 residual regret 方法。
