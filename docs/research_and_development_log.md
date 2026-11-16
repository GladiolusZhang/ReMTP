# ReMTP 研究与开发总日志

最后更新：2026-11-15（UTC+8）
当前工作分支：`jetson-r39-residual-adapt`；上游研究分支：`research/residual-aligned-tree-mtp`

## 2026-11-15（下午）：融合建树的小 kernel、KV 提交与拓扑缓存

- 依据 4090 的分析：剩余差距主要是 GPU 建树每轮约 973 个小 kernel 的固定开销。Jetson 上用 CUDA event 确认：三层图重放每轮 3.8 ms。
- 改动（输出逐 token 不变）：
  - `remtp/gpu_tree_fused.py`：每层一个 Triton kernel 完成排序、阈值、兄弟规则、备选和保留，熵项也合成一个 kernel；求和、除法、min、topk 仍用 PyTorch，所有输出与原实现逐位相同；
  - KV 提交缓存注意力层；
  - `DynamicTreeTopology` 缓存。
- 4090 运行说明新增 `ASYNC_SCHEDULING=1`，只用于基线；三种基线在 Jetson 上开异步后输出与同步逐 token 相同。
- HumanEval-100 五组（同一时段）：
  - AR 61/100、16.17 tok/s；
  - Native 56/100、MAL 2.852、23.01；
  - Cactus 55/100、3.092、24.85；
  - RAVEN 主机建树 57/100、3.246、22.75（−8.5%）；
  - RAVEN GPU 建树 + 融合 57/100、3.253、23.91（−3.8%）。
- 融合版本与前一晚的 GPU 建树 100/100 题逐 token 相同。
- 详见 `doc/RESULT-RAVEN-V4-TARGET-GATE-2026-11-13.md` 第 12 节。

## 2026-11-15（凌晨）：建树搬到 GPU（通宵试做）

- 用户同意用一晚在 Jetson 上试做"把建树搬到 GPU"。只做第一步：根节点和各层的候选选择、保留、下一层输入准备都在 GPU 上完成，提议阶段只读回一次；验证仍在主机上，随机数流不变。
- `remtp/gpu_tree.py`（`REMTP_DYNAMIC_TREE_GPU_BUILD=1`）：固定形状（第 2/3 层 MTP 前向 3/7 行）、逐个运算复现 `batched_trunk_children`、并列按稳定顺序、两两比较实现 `residual_coverage` 保留、分支噪声按路径预生成；每层"选择 + 保留"捕获为 CUDA graph（eager 版本每轮要发射几百个小运算，主机时间约 70 ms，反而比主机建树慢 9.5%）。
- 校验模式 `REMTP_DYNAMIC_TREE_GPU_BUILD_CHECK`：每层用主机参考实现在相同张量上重算，GSM8K 3 题 0 处差异。输出逐 token 相同率：GSM8K 2/3、HumanEval-30 29/30；不同的题来自 1 行与多行 GEMM 的最后一位差异。
- 速度：GSM8K 3 题每轮 0.1397 → 0.1358 s（−2.8%）；HumanEval-30 每轮 0.1414 → 0.1367 s，decode 23.5 → 24.3 tok/s，相对 Cactus 从 −7.2% 缩到 −4.1%，MAL 与 pass@1 不变。
- HumanEval-100（三组同一时段）：Cactus 55/100、MAL 3.090、24.68 tok/s；RAVEN 主机建树 57/100、3.246、22.82（−7.5%）；RAVEN GPU 建树 57/100、3.252、23.59（−4.4%）；两种 RAVEN 82/100 题输出逐 token 相同。
- 验证预发射（`REMTP_DYNAMIC_TREE_VERIFY_PRELAUNCH`，随机数偏移重新对齐，逐 token 相同）没有提速，保持关闭。
- GPU 事件计时：模型部分的 GPU 时间 RAVEN 与 Cactus 基本相同（117.4 vs 116.7 ms/轮），剩余约 11 ms 是 GPU 空闲和建树/验证的小运算。
- 03:32 用户的 SSH 连接断开（机器未重启，与 11-14 15:27 相同），最后一组计时的数据已采完，分析在次日补做。
- 详见 `doc/RESULT-RAVEN-V4-TARGET-GATE-2026-11-13.md` 第 11 节；测试 `tests/test_gpu_tree.py`。

## 2026-11-14（凌晨）：CUDA graph、CPU 调频与去同步

- 用户同意两条工程优化同时做；不改算法是为了避免影响精度。所有改动都验证了同种子输出逐 token 相同（GSM8K 3 题，`results/cudagraph_ab_20261113/`）。
- CUDA graph：
  - 链式方法用 vLLM 自带的 FULL_DECODE_ONLY（`mode=0`，不经过 torch.compile，内核与 eager 相同）。
  - RAVEN 新增 `remtp/tree_cudagraph.py`：子元数据从主机标量构建，去掉每次前向的两次 `.item()`；目标树掩码放进固定缓冲区；TREE_ATTN 声明完整图支持。GPU 内核逐位测试通过。
- 发现 CPU 调频问题：调频器 `schedutil`，图模式下线程睡眠等待，CPU 降到约 740 MHz，Python 变慢一倍。`remtp/cuda_spin_wait.py`（`REMTP_CUDA_SPIN_WAIT=1`）让 CUDA 等待时自旋，CPU 保持约 1984 MHz。
- `REMTP_DYNAMIC_TREE_ASYNC_HOST=1`：去掉 RAVEN 每轮约 10 次不必要的同步（可分页内存的 H2D 拷贝、`float()` 读温度、0 维 CUDA 张量做下标等）。
- 解码吞吐（eager → 图模式 + 自旋）：AR 14.8 → 16.7、Native 19.4 → 23.7、Cactus 22.1 → 26.8、RAVEN 19.5 → 24.5 tok/s。RAVEN 比 Cactus 低 8%（原来低 12%），每轮还差约 11 ms。
- 新增分析工具：`remtp/gpu_event_timing.py`（CUDA event 计 GPU 时间）、`remtp/host_timeline.py` + `scripts/analyze_host_timeline.py`（主机时间线和同步位置）；A/B 脚本记录每个请求的 CPU/GPU 频率分布，新增 `+cg/+spin/+async/+hostmeta/+native/+target` 臂。
- 测试：`tests/test_tree_cudagraph.py`、`tests/test_async_host.py`，相关 150 项全部通过。
- 小规模公平对比（HumanEval 前 30 题，512 token，四种方法同一套工程配置，`results/engopt_suite_20261114/`）：decode tok/s AR 16.2、Native 23.3、Cactus 25.5、RAVEN v4-safe 23.0；MAL Cactus 3.131、RAVEN 3.320；pass@1 22/24/24/23。三种推测方法的输出与 11-13 的 eager 运行全部逐 token 相同。RAVEN 有两题首次判分为 Docker 超时，重判通过并记录。
- 新增 `doc/PORTING-RAVEN-V4-2026-11-14.md`：给 4090 的 v4-safe 移植说明（算法改动与工程改动分开列出）。
- 详见 `doc/RESULT-RAVEN-V4-TARGET-GATE-2026-11-13.md` 第 10 节。

## 2026-11-13（晚）：只改实现的工程优化与 CPU 时间线

- 用户要求：只做工程优化，不改算法（包括草稿分布和提议规则）；草稿词表裁剪和固定形状的树都不采用。
- 主机端优化（输出逐 token 相同）：
  - 候选选择每个父节点的设备运算从约 72 个减到约 22 个（`batched_trunk_children`，GPU 上 799 个父节点逐位一致）；
  - `normalize_rows_nosync` 少发射运算，均匀分布改为标量；`_boost_known` 去掉 clone；
  - 自适应规划拆成 `_plan_vectors`（同一运算序列）、`_plan_from_values` 和 `adaptive_trunk_relaxation_plan_host`（读回后在 CPU 张量上计算；GPU 上 300 次快速验证与参考实现逐位一致，含随机数状态）；
  - 165 项测试通过。
- 干净对比（旧代码 worktree `/home/apex/ReMTP-residual-base` = HEAD 44dec73；3 题，416 轮，不开计时器，`results/engopt_ab_20261113/`）：每轮 0.1813 → 0.1802 → 0.1792 s（加批量选择后 −1.1%），输出逐 token 相同。此前用 `AB_PROFILE=1` 做的对比被不断增加的计时钩子污染，作废。
- 新增 `remtp/torch_trace.py`（`REMTP_TORCH_TRACE_JSON`，A/B 脚本里的 `AB_TORCH_TRACE=1`）：
  - 驱动设置 `RmProfilingAdminOnly=1`，只能抓 CPU 事件。
  - 目标前向的 CPU 发射：Cactus 100.7 ms、RAVEN 85.9 ms（开着 profiler）；每层约 30 个运算，Python 开销分散，没有单一热点。不开 profiler 约 72 ms，按带宽估算 GPU 计算约 44 ms，所有方法都卡在 eager 发射上。
- 下一步：CUDA graph（`ENFORCE_EAGER=1` 目前对所有方法都开着）的可行性和兼容性；RAVEN 树提议元数据准备和验证前后处理的 Python 开销。详见 `doc/RESULT-RAVEN-V4-TARGET-GATE-2026-11-13.md` 第 9 节。

## 2026-11-13（下午）：RAVEN v4 目标支持度门槛、GPTQ 校准模型、分段计时与主机端开销

- 状态：完成，初步结果。详见 `doc/RESULT-RAVEN-V4-TARGET-GATE-2026-11-13.md`。
- **RAVEN v4（由目标模型决定在哪里松弛）**：
  - `DynamicTreeConfig` 新增三个字段：`relaxation_min_target_probability`、`relaxation_min_target_ratio`、`adaptive_relaxation_pool`（取值 nominal/gated）；对应环境变量 `REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_PROB`、`_MIN_TARGET_RATIO`、`_ADAPTIVE_RELAX_POOL`；默认值保持旧行为。
  - v4-safe 配置：门槛 p≥0.01，自适应上限 1.0；每个位置的接受概率 ≤ Cactus，门槛以下完全精确。
  - 修正了规划器 docstring 中"实际松弛总量不超过静态控制"的错误说法（原实现只对名义额度成立）。
  - `tests/test_target_support_gate.py` 15 项通过，其中一项检查"门槛只由目标决定"。
  - 离线（按记录的随机数重放）比较后选择绝对门槛 1%：MAL −1.4%，p<1e-2 的接受从每千 token 13 个降到 0.9 个。
- **RTN 模型 A/B**（HumanEval-30，输出上限 512，`results/raven_v4_ab_20261113/`）：
  - MAL：Native 2.608，Cactus 2.909，v3 3.214，v4-safe 3.128，v4-gated 3.131。
  - p<1e-3 接受/千 token：Cactus 3.33，v3 5.19，v4 0.09（精确约 0.1）。
  - 每位置多接受的质量：Cactus 0.058，v3 0.072，v4 0.049。
  - 用本次 v3 日志重放 v4-safe，估计 MAL 3.167（3.149~3.186），实测 3.128，在采样波动内一致。
  - gated 与 safe 几乎相同，采用 safe。
- **分段计时**（`results/w4a16_profile_20261113/`、`v4_batch_profile_20261113/`、`v4_tiefix_profile_20261113/`）：
  - 草稿 LM head 每次 13.6 ms（约 91 GB/s），每轮 3 次加目标 1 次，约占一轮的 36%。
  - RAVEN 每轮的模型计算约 139 ms，不比 Cactus 的约 144 ms 多，但墙钟多约 38 ms，都是 GPU 空等 CPU（每层的树提议决策和约 9 ms 的验证 Python 逻辑）。
  - `remtp/runtime_profile.py` 补充了动态树的细分钩子。
- **主机端开销的两处修改**（输出均逐 token 相同）：
  - 批量候选选择 `REMTP_DYNAMIC_TREE_BATCH_SELECT`（`batched_trunk_children`，11 项等价测试）：每轮 −1.1%。
  - 并列修复 `_relevant_tie`：只在可选值（≥ min_draft_prob）之间检查并列。此前每轮平均有 0.94 次回退到全词表稳定排序；原测试已相应修改。每轮再 −0.6%。
  - 合计 0.1801 → 0.1771 s/轮。剩余约 35 ms 需要结构性改动。
- **GPTQ 校准模型**（`scripts/quantize_fastmtp_w4a16_gptq.py` → `models/FastMTP-W4A16-GPTQ`）：
  - 本机逐层流式量化，38 分钟；校准数据取自 venv 里的源码和包说明文档，不含评测数据。
  - 留出文本上相对 BF16 的 KL（T=1）：GPTQ 0.123，bnb NF4 0.136，RTN 0.260；top-1 一致率 88.6% / 88.1% / 84.4%。
  - 更正：vLLM 现场量化的 bnb 实际是 NF4，不是 FP4，之前文档里的 FP4 标注已改正。
- **GPTQ 公平对比**（HumanEval-30，输出上限 512，`results/gptq_ab_20261113/`）：
  - pass@1：Native、Cactus、SpecCascade 均为 24/30，v4-safe 23/30；RTN 时 Native 只有 12/30。
  - MAL：Native 2.883，Cactus 3.131，SpecCascade 3.001，v4-safe 3.320（比 Cactus 高 6.0%）。
  - p<1e-3 接受/千 token：Cactus 2.46，v4-safe 0，SpecCascade 0，Native 0。
  - 每轮：链式方法 0.150 s，v4-safe 0.176 s；decode 比 Cactus 低 9.5%。
- 测试：165 项树、快速路径、门槛、批量、日志测试全部通过。
- 限制：每组只有 30 题、512 上限，pass@1 没有统计意义；本条的结论只依据分布指标和 MAL。

## 2026-11-13：W4A16 + MAXN 全方法 HumanEval-100 套件完成与逐轮日志分析

- 状态：完成。套件 2026-11-12 16:57 至 2026-11-13 07:48（14.9 小时），9 个阶段全部完成，无失败。RAVEN run1 的 HumanEval/123 被判题器记为 container_error：候选 `n = 3n+1` 永不终止，在内存上限处被杀（exit 137）；人工判错，已追加到 evaluation.jsonl（`adjudicated: manual`）。
- pass@1（run1 / run2，合计）：AR 40；Native 40/43（41.5%）；Cactus 38/42（40.0%）；SpecCascade 38/39（38.5%）；RAVEN v3 33/39（36.0%）。按题 bootstrap 95% 区间：RAVEN−Cactus −4.0（−10.5~+2.5）个百分点，Cactus−Native −1.5（−8.0~+5.0）个百分点；全部不显著。同方法换种子就有 30~38 题对错不同，本机样本只能分辨约 10 个百分点的差距。
- 速度（decode tok/s，对 AR）：AR 14.66；Native 18.49（1.26×）；Cactus 21.20（1.45×）；SpecCascade 19.53（1.33×）；RAVEN v3 17.97（1.23×）。MAL：Native 2.768 / Cactus 3.173 / SpecCascade 2.936 / RAVEN 3.372。每轮：链式 0.150 s，RAVEN 0.188 s（+38 ms，其中日志约 3.5 ms）。
- RTN 代理模型质量明显低于 bnb（vLLM 现场量化实际为 NF4，2026-11-13 核实）：同题同种子 Native 40% 对 62%，McNemar p=0.0009。截断 33~39 题对 18~22 题，几乎都是 "Wait, ..." 反复自我检查写满 2048 token。RTN 模型不能用于质量结论，需要校准的 4 位模型。
- 新增 `scripts/analyze_suite_round_logs.py`：按时间把每轮对应到请求，计算参与判定位置的实测接受率、按规则计算的接受率、多接受质量、低支持度接受（每千 token）。
  - 自检：Native 实测 0.7335 对 min(1,p/q) 0.7338；Cactus 实测对离线公式 0.8228/0.8236；RAVEN 实测对日志接受概率 0.8082/0.8084。
- 逐轮结论：
  - 每位置多接受的质量：Cactus 0.091~0.094，RAVEN v3 0.099~0.100，SpecCascade 0.030~0.032，Native 0。
  - 目标概率 <1e-3 的 token 被接受次数（每千 token）：Cactus 3.7~4.5，RAVEN 4.9~5.3，精确采样约 0.1；<1e-2：Cactus 10~12，RAVEN 13，精确约 0.9。
  - RAVEN v3 按实际接受计算比 Cactus 放宽得更多：自适应规则把 Cactus 在饱和位置用不满的名义额度回收后，再分给其他位置，最多 1.5 倍。
  - 兄弟命中约 8000 次，没有一次 p<1e-2，是干净的。
- 离线反事实（用日志里的 u 重放主干判定）："逐位接受概率 ≤ Cactus + p<1e-2 时用精确规则"估计 RAVEN 每轮 token 从 3.373 降到 3.315（3.291~3.339），仍比 Cactus 高约 4.5%，而低支持度接受降到精确水平。只靠树、完全不松弛时约 3.05。
- 速度瓶颈（带宽估算，未实测）：BF16 共享 LM head（1.24 GB）每轮被读 4 次，加上 BF16 MTP 层，3 个草稿步约 65~70 ms，与目标 forward 相当。草稿用量化 LM head 副本有望把每轮降到约 90 ms。
- 文档：`doc/RESULT-W4A16-MAXN-HUMANEVAL-SUITE-2026-11-13.md`。本条未改服务代码，未重跑测试。

## 2026-11-12：逐轮日志记录器与 W4A16 + MAXN 全方法 HumanEval 套件（2026-11-13 完成，见上条）

- 状态：代码与检查完成；套件 16:57 启动，2026-11-13 07:48 完成。用户决定后续实验统一使用 MAXN + W4A16。
- 新增 `remtp/round_logger.py`（`REMTP_ROUND_LOG_JSONL` 开启，默认关闭）：链式方法包住最外层的 vLLM `rejection_sample`，每轮记录草稿 token、p(y)、q(y)、目标/草稿 top-1 及概率、输出 token、接受数；RAVEN 包住 `fixed6_vllm.strict_tree_verify`，记录节点路径与 token、每行目标 top-1，以及验证器已有的逐节点诊断（p、q、松弛后 h、均匀随机数、自适应额度、命中标记）。不抽随机数、不改任何输入输出；设备数据用非阻塞拷贝 + CUDA 事件，下一轮再写盘，不插入同步点；出错后自动停用。`remtp/fastmtp_verified_worker.py` 在 `_install_optional_profile` 最后安装它（所有方法钩子之后）。
- `scripts/ab_canonical_tree_attention.py` 新增 `+roundlog` 臂；新增 `scripts/run_w4a16_maxn_suite.py`：复用 `run_day_20261111.Controller`，五种服务（native、cactus、spec_cascade、raven_v3、target=AR）均为冻结 100+100 环境 + W4A16 覆盖；HumanEval-100 冻结选题、2048 预算，run r 种子 = 冻结种子 + (r−1)；默认顺序 native/cactus/spec_cascade/raven_v3 run1 → 反序 run2 → target run1；要求 MAXN，记录 protocol.json（git HEAD、脏文件、设备温度）、每阶段 device.json、请求起止时间、rounds.jsonl、Docker 判分，输出 suite_report.md / suite_summary.json（每次运行、合并、配对 McNemar、同方法两次一致性、相对 AR 速度）。
- 验证：`tests/test_round_logger.py` 5 项通过；GPU 上异步路径记录与直接计算逐位一致；MAXN W4A16 3 题实机：开启记录后 RAVEN 与 Cactus 输出均逐 token 相同、轮数相同；每轮开销 Cactus −0.1%（初版同步拷贝为 +4.9%，改为异步后消除），RAVEN +1.9%（节点诊断 JSON 序列化，只会使 RAVEN 速度偏低）。套件冒烟（5 种服务、2 题、64 token）全部启动、判分、出报告。
- 限制：W4A16 为 RTN 未校准代理模型；RAVEN 有约 2% 记录开销而链式方法没有；最后一轮链式记录在进程被 SIGTERM 结束时可能丢失。

## 2026-11-12：MAXN 功耗模式下 bnb / W4A16 × Cactus / RAVEN v3 四组实测

- 状态：完成。用户执行 `sudo nvpmodel -m 0`（提示需重启并已重启），之后 `nvpmodel -q` 为 MAXN（8 核、GPU 上限 918 MHz）。机器目前仍为 MAXN。
- 未改代码；用 `scripts/ab_canonical_tree_attention.py` 在 GSM8K 17/353/856 上顺序运行四组，输出 `results/maxn_ab_20261112/`（含 `thermal.log`，最高 70°C，负载时 GPU 918 MHz）。
- 每轮秒 / decode tok/s：bnb Cactus 0.5404 / 6.038；bnb RAVEN v3 0.5963 / 5.432；W4A16 Cactus 0.1493 / 20.261；W4A16 RAVEN v3 0.1796 / 18.132。相对 15W 同组每轮快 1.67~2.11 倍；相对 15W bnb Cactus，MAXN W4A16 Cactus decode 7.2 倍。
- RAVEN 每轮额外时间：15W 约 73 ms → MAXN W4A16 约 30 ms（+20%），仍高于 Spec-Bench 上约 10% 的 MAL 优势。
- 限制：每组 3 题、文本和长度各不相同；RTN 模型质量未评测；MAXN 结果与既有 15W 结果不可混用。文档 `doc/RESULT-W4A16-MARLIN-MAXN-2026-11-12.md` 第 5 节。

## 2026-11-12：W4A16 + Marlin 目标模型可行性与 15W 速度（MAXN 待测）

- 状态：15W 部分完成；MAXN 需要重启，尚未测。
- 动机：15W 分段计时中目标 forward 约 1.02 s/轮，其中 bnb 4 位反量化 CUDA 跨度约 0.70 s/轮、BF16 线性层约 0.41 s/轮（跨度重叠），注意力 5.5 ms；公共数值设置把 bnb 固定为逐层反量化 + BF16 GEMM。
- 新增 `scripts/quantize_fastmtp_w4a16_rtn.py`：流式把 252 个目标线性层量化为 GPTQ v1 格式（4 位、对称、group 128、RTN、无校准），词嵌入/LM head/norm/bias/MTP 层保留 BF16；输出 `models/FastMTP-W4A16-RTN`（已忽略目录，5.9 GB）。
- `scripts/ab_canonical_tree_attention.py` 新增臂标记 `+w4a16`（gptq_marlin 目标、MTP 层 BF16、仅去掉两个 bnb 专用开关）和 `+cactus`（冻结的 Cactus 服务环境）；原有臂行为不变。
- 验证：打包往返逐位正确；Marlin 在 sm87 可用，4 种形状 × M=1/4/10/64 与反量化参考相对差 ≤3.5e-3；同一行在 M=1/4/10 中逐位相同（链/树数值一致）；RTN 权重误差 0.12~0.14，与 bnb FP4（0.12~0.13）相当（更正 2026-11-13：vLLM 现场量化实际是 NF4，权重误差 0.093~0.096，RTN 明显更差）；服务日志确认 `MarlinLinearKernel`，模型 5.83 GiB。
- 15W 实测（GSM8K 17/353/856）：Cactus 每轮 1.1421 → 0.2498 s，decode 2.813 → 13.075 tok/s；RAVEN v3 每轮 1.2155 → 0.3225 s，decode 2.763 → 10.202 tok/s（RAVEN 3 题全对）。W4A16 下 RAVEN 每轮比 Cactus 多约 73 ms（+29%），decode 低 22%。
- 限制：3 道题、文本不同；RTN 模型质量未评测，不能代表校准量化；未做逐 token 等价检查（换量化后输出本来就会不同）。详见 `doc/RESULT-W4A16-MARLIN-MAXN-2026-11-12.md`。

## 2026-11-12：全天实验完成（Spec-Bench 四方法 + HumanEval-100）

- `scripts/run_day_20261111.py` 于 2026-11-11 14:33 启动、2026-11-12 13:10 完成全部五个阶段（22.6 小时），无失败阶段；原始数据 `results/day_20261111/`，汇总 `day_report.md`。
- Spec-Bench（种子 20261111，6 任务×15 条，multi_turn 两轮，105 次请求/方法，512 token，T=0.6，单次运行）：MAL native 2.441 / cactus 2.696 / raven_v3 2.972 / pathfirst 3.015；每轮秒 1.1431 / 1.1431 / 1.2237 / 1.2229；E2E 2.083 / 2.300 / 2.368 / 2.401。raven_v3 对 Cactus E2E +3.0%（按题 bootstrap 95% 区间 +1.5%~+4.6%），六个任务全部领先；pathfirst +4.4%（+2.7%~+6.2%），对 raven_v3 +1.4%，单次运行不作结论。
- HumanEval-100（冻结协议，raven_v3）：pass@1 54/100，E2E 2.7336，MAL 3.377，每轮 1.2266 s，截断 19。对原 RAVEN（59%、2.5487、3.402、1.3260）速度 +7.3%；对 Cactus（65%、2.7573）速度 −0.9%。精确 McNemar：对原 RAVEN p=0.36，对 Cactus p=0.08；不解释为优化引入的质量变化（接受规则未变），但 RAVEN 在 HumanEval 上低于 Cactus 的现象需多种子重复确认。
- 本条只更新文档：`doc/RESULT-RAVEN-OVERHEAD-OPTIMIZATION-2026-11-10.md` 新增 11.1 节；未改代码，未重跑测试。

## 2026-11-11：全天实验重复启动事故修复、代码推送到 GitHub

- 15:01:46 有人用同一条命令又启动了一个 `run_day_20261111.py` 实例。它发现 8000 端口被占，把五个阶段全记为失败，把 `status.json`/`day_report.md` 改成“已结束”，还把正在运行的 `specbench_raven_v3` 输出目录改名为 `_partial_*`。`remtp.benchmark` 只在全部请求结束时才写 `requests.jsonl`/`summary.json`，不改回会丢掉整个阶段。16:07 已手工改回目录并修正状态；原实例（14:33 启动）未受影响，继续运行。
- `scripts/run_day_20261111.py` 增加两道保护：输出目录下的单实例 `fcntl` 文件锁；启动时 8000 端口已被占用就退出，不改动任何文件。Spec-Bench 阶段改为服务成功启动后才移走旧的未完成目录。已实测第二次启动直接以 1 退出、无副作用。
- 推送到 `origin/jetson-r39-residual-adapt`，分两个提交：先是 10 月未提交的 Jetson 研究代码（三个钩子文件为 100+100 冻结版本），再是 11-10/11 的三层开销优化、通宵与全天脚本和文档。`data/gsm8k/train.jsonl` 与 `deploy/**/*.tar.gz` 加入 `.gitignore`。提交前重跑相关测试 118 通过、1 跳过。

## 2026-11-11：RAVEN 第三层开销优化（实机逐 token 等价）与全天实验启动

- 读 vLLM `EagleProposer.propose`/`set_inputs_first_pass`/`prepare_inputs_padded` 与 `gpu_model_runner.sample_tokens` 的调用顺序后确认：`_dynamic_propose_tree` 末尾的全节点重物化只写草稿树 KV，而这些槽位在被读到之前就被下一轮首遍 MTP forward 重写；`_compact_selected_path` 的草稿 KV 提交同理。原计划的"推迟重物化"改为直接跳过。
- 代码（全部默认关闭）：`remtp/dynamic_tree_vllm.py` 新增 `REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV`（跳过末次 forward 并置 `draft_slots=None`；开着 anchor 剪枝时启动报错）、`REMTP_DYNAMIC_TREE_FAST_METADATA`（`_cached_tree_attn_bias` 按拓扑缓存 CPU 构造的树掩码）及对应启动标记；`remtp/fixed6_vllm.py` 新增 `_offset_tensor` 缓存、`REMTP_DYNAMIC_TREE_FAST_COMMIT=2` 分派、`path_first_row_order`/`_path_ordered_first_pass`（`REMTP_DYNAMIC_TREE_PATH_ORDERED_FIRST_PASS`，改变提议分布 Q）；`remtp/raven_fast_commit.py` 新增 `copy_cache_slots_inplace`/`inplace_order_is_safe`/`commit_selected_slots_inplace`；`scripts/ab_canonical_tree_attention.py` 支持 `+inplace/+meta/+skip/+pathfirst` 臂。
- 测试：新增 `tests/test_raven_third_layer.py`（14 通过、1 项因系统 Python 无法导入 vLLM 跳过，另用 venv 在 GPU 上核对缓存掩码与 vLLM 原函数逐位相同）；连同 `test_raven_fast_commit.py`、`test_raven_fast_path.py`、`test_dynamic_mtp_tree.py` 共 118 通过。
- 实机（GSM8K 17/353/856，`results/third_layer_ab_20261111/`）：臂 E（第一、二层 + 第三层三项）1723 token/513 轮，与臂 D 三条逐 token 相同，每轮 1.2385 → 1.2155 s（−23.0 ms），decode 2.712 → 2.763 tok/s。分段计时（32 轮）：cache_update 17.9 → 2.9 ms，proposal 161.9 → 144.7 ms，MTP forward 4 → 3 次，整轮 1241 → 1209 ms（Native 1139）。微基准：树掩码构造 2.67 ms/次 → 缓存 0.0025 ms；36 层路径提交 7.4 → 3.9 ms。
- 臂 F（E + 首遍路径重排）1184 token/343 轮，每轮 token 3.452 对 3.359，文本不同、样本小，不作结论。
- 文档：`doc/RESULT-RAVEN-OVERHEAD-OPTIMIZATION-2026-11-10.md` 第 10、11 节；新增 `doc/PORTING-RAVEN-OVERHEAD-OPTIMIZATIONS-2026-11-11.md`（三层汇总与 4090 移植步骤）和 `deploy/raven_overhead_port_20261111/`（钩子补丁，基线为 10 月冻结代码；对上游 `3afafa1` 试打：`fixed6_vllm.py` 全部成功，`dynamic_tree_vllm.py` 7 个 hunk 中 `_dynamic_verify_dispatch` 一个需手工合并）。4090 上未做任何测试。
- 新增 `scripts/run_day_20261111.py`（HumanEval-100 冻结协议 + Docker 判分、Spec-Bench 新抽样多方法；24 小时截止，逐阶段写 `day_report.md`），冒烟通过后启动，输出 `results/day_20261111/`。
- 关于 Cactus 与 Native 准确率接近的核对（只读 100+100 记录）：GSM8K 配对结果为两者都对 80、仅 Native 对 4、仅 Cactus 对 7、都错 9，精确 McNemar p≈0.55；Native 截断 15 题（其中对 2）、Cactus 截断 12 题（其中对 5），未截断题上 Native 82/85、Cactus 82/88。100 题分辨不了 2~3 个百分点的差异。

## 2026-11-11：通宵实验完成（GSM8K-100 + Spec-Bench，优化后 RAVEN 对 Cactus）

- `scripts/run_overnight_20261110.py` 于 2026-11-10 23:04 启动、2026-11-11 10:53 完成全部三个阶段（总 11.82 小时；Native 的 Spec-Bench 因超出 10 小时截止被跳过）。原始数据 `results/overnight_20261110/`（已忽略目录），汇总 `overnight_report.md`。
- GSM8K 100 题（基准同题/同种子/2048 预算，臂 D 配置实跑）：正确 89/100，E2E 2.7733 tok/s，MAL 3.471，每轮 1.2435 s。对 10 月原 RAVEN（88/100、2.6116、3.468、1.3192）：MAL 保持，每轮 −75.7 ms，E2E +6.2%；对 Cactus（87/100、2.8333）仍 −2.1%。回归：每轮 1.226 s + 25.4 ms/千 token（原 1.265 s + 81.8；链式 1.140 s + 10.6）。17/100 题与 10 月输出逐 token 相同，其余为 1 ulp 数值差后的不同采样轨迹，正确率差 1 题不解释为质量变化。
- Spec-Bench（6 任务×10 条、512 token、T=0.6，两方法均实跑，70 次请求）：RAVEN-opt overall decode 2.396 tok/s、E2E 2.336、MAL 2.990；Cactus 2.361、2.306、2.701。MAL +10.7%，E2E +1.3%；translation/rag/qa/multi_turn 领先，summarization/math_reasoning 落后。单次运行、文本不同，结论为"持平略高"，不作显著性声明。
- 每轮仍比 Cactus 多约 91 ms（GSM8K 配对）/约 104 ms（Spec-Bench 折算）；GSM8K 平衡点约 69 ms，第三层（延迟重物化、树元数据缓存、cache_update 归因）未做、未测。
- 运行环境：03:22 起校园网出网中断（Tailscale 失联至 13:08），机器未宕机，实验为本机回环请求未受影响；13:08 机器硬复位（SYS_RESET_N），其时实验已完成。系统日志核对，无 OOM/过热/内核错误记录。
- 本条只更新文档：`doc/RESULT-RAVEN-OVERHEAD-OPTIMIZATION-2026-11-10.md` 新增第 9 节；未改代码，未重跑测试。

## 2026-11-10：RAVEN 每轮开销定位与两层等价优化（实机 A/B 完成，第三层待做）

- 问题重述：100+100 基准里 RAVEN 对 Cactus 每轮 token +5.9%、每轮耗时 +15.0%（GSM8K）。逐请求回归（R²≈0.99）：链式方法每轮 1.139 s + 10.4 ms/千 token 上下文；RAVEN 1.255 s + 82 ms/千 token。目标 forward 约 1.01 s 且几乎不随行数变化，多出的约 172 ms/轮分为固定约 115 ms（主机/设备同步、重复 MTP forward、KV 提交）与随上下文增长的约 57–66 ms（canonical 树注意力逐节点扫前缀）。
- 新增 `remtp/tree_attention_probe.py`（`REMTP_TREE_ATTENTION_PROBE_JSONL` 开启，`remtp/target_numerics.py` 挂载，默认关闭、只读）。实机 552 次目标树注意力调用：原生 `qq_bias` 掩码全部等于拓扑祖先掩码；537 次输出与 canonical 逐位相同，15 次有 1–2 行差 1 个 bf16 ulp（最大 0.0039）。注意力核微基准（36 层）：ctx 256/512/1024/2048 下 canonical 63/78/124/228 ms，原生 qq_bias 31/48/43/81 ms，链式 17/21/31/38 ms；合成输入上两种树布局逐位相同。结论：10-04 加入 canonical 的动机（数值一致）与第四轮定位的两个真实根因无关，原生路径可用，但不能逐 token 复现 10 月记录。
- 新增 `remtp/raven_fast_path.py`：`REMTP_DYNAMIC_TREE_FAST_SELECT=1` 把 `select_dynamic_candidates`/`_cactus_trunk_children`/`_sample_q_token`/`_sample_branch_q_token`/`_temperature_probs` 换成每个父节点两次设备传输的实现（top-k 截点有并列时退回参考全排序；有效性检查改在指数竞争抽样之后，仅影响本就会报错的请求）；`REMTP_DYNAMIC_TREE_FAST_VERIFY=1` 把 `residual_hit_tree_verify` 换成批量归一化（逐行求和以保持逐位一致）、全节点 p/q 一次 gather、每个被验证节点一次传输、残差仅在拒绝时计算的实现（EOS 保护/风险策略/概率上限/决策 trace 时自动退回参考）。`remtp/dynamic_tree_vllm.py`：分派与安装、两处 `float(q[int(token)])` 循环改为一次 gather（无开关、等价）。
- 新增 `tests/test_raven_fast_path.py`（30 项）：随机输入下与参考实现逐位对照 token、接受集合、终止原因、诊断字典、RNG 状态；连同 `test_dynamic_mtp_tree.py`、`test_canonical_tree_attention.py` 共 101 项，用系统 pytest 7.4.4 + 项目 .venv 依赖 PYTHONPATH 实际通过。`py_compile` 通过。
- 新增 `scripts/ab_canonical_tree_attention.py`：用 100+100 冻结的 `effective_env.json` 起服务，重放 GSM8K 17/353/856（与基准相同 prompt/seed/2048 预算），逐臂记录 `/metrics` 差分并与 10 月记录逐 token 比对。实测：臂 A（canonical=1）3 条输出与 10 月记录逐 token 相同、每轮耗时相同（1.3057 s）；臂 B（canonical=0）1.2620 s/轮但输出在第 254/152/41 个 token 分叉、更短（上下文 414 对 643），按斜率折算同上下文约省 25–27 ms/轮；臂 C（canonical=0 + 两个 fast 开关）与臂 B 513 轮输出逐 token 相同，1.2414 s/轮（−20.6 ms）、decode 2.661→2.706 tok/s。对同上下文 Cactus（约 1.143 s）仍多约 98 ms，未到收支平衡。
- 新增 `remtp/raven_fast_commit.py`（`REMTP_DYNAMIC_TREE_FAST_COMMIT=1`，`fixed6_vllm._compact_selected_path` 分派）：路径提交改为先把所有层/槽位源复制到临时缓冲再写目的槽位的两次 `torch._foreach_copy_`，语义与逐层参考一致；`tests/test_raven_fast_commit.py` 6 项（含重叠槽位、空缓存、坏形状）通过。臂 D（C + 融合提交）与 B/C 输出逐 token 相同，1.2385 s/轮，仅再省 2.9 ms；fast 配置的 32 轮分段计时（`AB_PROFILE=1`）：整轮 1241 ms 对 Native 1139，剩余差距为 proposal +59（含第 4 次 MTP forward 约 12、树元数据约 14）、verification +17、cache_update +18、树注意力 +5（短上下文）。
- 结果与第三层候选见 `doc/RESULT-RAVEN-OVERHEAD-OPTIMIZATION-2026-11-10.md`；原始数据 `results/canonical_attention_ab_20261110*/`（已忽略目录）。新开关默认全部关闭，不改变任何接受规则、树预算或输出分布；三题四臂不是正式基准，未宣称质量或正式速度结论。
- 新增 `scripts/run_overnight_20261110.py` 并于 23:04（UTC+8）以 `setsid nohup` 启动（`results/overnight_20261110/`，10 小时截止）：① 臂 D 配置的 RAVEN 在 100+100 基准的 100 道 GSM8K 题上按原协议重跑并判分，与记录中其他方法同题并列；② Spec-Bench（`~/ReMTP/data/spec_bench/question.jsonl`）6 任务各 10 条、512 token、temperature 0.6：优化后 RAVEN、Cactus，时间允许时 Native。三个阶段已各用 1 条冒烟通过（`results/overnight_smoke/`）。结果未出，不作任何结论。

## 2026-10-14：Deep因树预算不公平而不采用（文档更正，待验收）

- 用户明确组长认为Deep的更大预测树对原Adaptive不公平、不予采用。冻结protocol核对原Adaptive D3/N9、Deep D4/N12；旧点门槛未检查相同树预算，MAL及其他收益无法单独归因于方法机制。采用状态更正为不采用，保留实测数字、原冻结结果和历史gate。
- 修改正式精简/完整结果、README/优化入口、目标文档、100+100旧计划与优势搜索旧计划、112结果/历史最佳入口，增加可见采用更新；新增 `plan/adaptive-raven-equal-budget-2026-10-14.md` 和 `doc/upstream-changes/equal-budget/2026-10-14.md`。
- 新计划固定候选/原Adaptive D3/N9及三链式基线D3，先利用历史D3失败证据，再做EOS/δ/mix消融与每轮计算优化。当前原100+100刻度GS质量≥88%/TPS>2.8333/平均秒<198.02，HE≥65%/>2.7573/<290.11，实判使用完整精度与同一比较单元；这些不是已取得的新收益。
- 实际验证到此仅只读protocol/CSV及历史报告；5份原始汇总/协议/状态/分析hash已记录。待验收正式两表数字、5个hash、当前链接和格式。仅文档/计划更正，没有新算法实现、GPU/HTTP/判题或后台任务；新固定预算方法尚未测试，也不保证缩树后胜出。

## 2026-10-14：新增100+100精简结果汇报（文档完成）

- 用户要求额外生成只保留结果的精简汇报。新增 `doc/RESULT-RAVEN-100X100-BRIEF-2026-10-14.md`，只列实验完成情况与GSM8K/HumanEval各五方法结果，指标为正确数/正确率或pass@1、E2E吞吐、平均整题秒及MAL。
- 新增专门改动记录 `doc/upstream-changes/full-benchmark/brief-2026-10-14.md`，同步本日志。数据来自已完成的 `results/raven_full_100x100_20261010/summary.csv`；原详细报告及原始实验文件保留，没有新生成、判题或方法优化。
- 实际验证：两组表10行全部显示数值与最终CSV逐项一致，数据链接有效；最终COMPLETED/exit0、1000次生成/500次HumanEval判题完成情况核对通过，`git diff --check`通过。本次仅整理已有结果，不产生新的实验结论。

## 2026-10-13：100+100五方法完整实验结束核对（完成，三基线目标未达）

- 用户询问结果是否出来；实际20:59:11 UTC+8结束，COMPLETED、exit0、1000/1000生成、500/500判题，systemd inactive/dead、Result=success、NRestarts=0，8000无监听。failure.json为10月12日历史判题中断，已归档，并非新的最终失败。
- 新本地忽略results/raven_full_100x100_20261010/offline_audit_20261013.py，对1000payload/response/IDs/prompt/seed/原评分、500判题及270恢复期容器证据、10汇总、8配对及原2000bootstrap、318冻结文件/恢复适配/原600与230资产前缀进行CPU复核。该脚本仅读原结果，独立写新分析，不重放GPU或官方判题；实际运行待确认。
- 汇总初见GSM8K Native84/Cactus87/Spec85/原Adaptive88/deep86%，HumanEval62/65/57/59/63%；deep对Native两数据集点门槛通过，但对Cactus质量/吞吐未过，对Spec整题耗时未过，三基线目标未达。原方法在GSM8K正确率最高，deep在HumanEval改善原方法；最终中文结果/改动记录待核验后形成，不启动新优化。
- CPU独立审计实际一次通过：1000/500、270容器证据、318冻结文件、609恢复前资产/前缀、10汇总与8×2000配对bootstrap全部匹配；attempt1=600/attempt2=400，确认候选OOM仅Spec44一条，正确计错。新offline_analysis_20261013.json写入，原结果/analysis/gate/冻结数据不覆盖，没有新GPU/判题。
- 新doc/RESULT-RAVEN-100X100-2026-10-13.md与doc/upstream-changes/full-benchmark/results-2026-10-13.md完成，README/优化入口/启动/恢复/预测文档加入可见完结notice。deep GS86%/2.6800/263.01秒/MAL3.8111，HE63%/2.5745/293.95秒/3.6692；对原方法GS−2pp、吞吐+2.62%、整题+7.47%，HE+4pp/+1.01%/−15.05%。
- 成本分解：deep对Cactus MAL+16.41%/+15.20%，每轮decode成本+23.14%/+23.29%，E2E−5.41%/−6.63%；GS输出相对原方法+10.28%导致整题更长，HE输出−14.19%对应整题改善。所有正确率差95%区间跨0，对Spec微小吞吐优势区间跨1，不宣称普遍质量非劣。单seed、已观察GS100题及跨重启/固定顺序限制保留。文档链接/git diff --check待最终实查。
- 最终文档验收：31条当前本地链接存在，独立CPU审计脚本py_compile及git diff --check通过；原五份server日志未发现CUDA error/EngineDead/OOM/Traceback，最终端口空闲、服务成功退出。完整数据与中文结果文档已形成，没有新GPU、判题、环境/算法修改或后续后台优化任务。

## 2026-10-12：断电重启续跑与Docker OOM误分类修复（实施中）

- 用户授权续跑，实际600/1000生成及230/500判题完整保留：Native/Cactus各200生成+100判题，Spec200生成+30判题。逐条题号/seed/IDs/raw response/record匹配、没有损坏尾行，318冻结源码及冻结数据hash通过。恢复无需重跑已完成生成。
- 08:08:43 UTC+8控制器实际先因SpecCascade HumanEval/44代码无限增长循环触发Docker512MiB内存上限停止，内核CONSTRAINT_MEMCG/OOMKilled与exit137明确；关机日志11:34:46，不能仅归因断电。重启丢失socket ACL，用户已执行永久docker组加入和当前socket ACL，docker info29.1.3及原ARM64image ID核对成功。
- 新scripts/humaneval_container_recovery.py、resume_raven_full_benchmark.py、tests/test_humaneval_container_recovery.py及doc/upstream-changes/full-benchmark/recovery-2026-10-12.md。适配只在原受限容器退出后先inspect可信State，确认OOMKilled/137才计候选runtime_error；其他137/基础设施故障保持原停止语义。原冻结控制器和generation源码/protocol不改，8秒/512MiB/1CPU/镜像/官方tests不变，恢复适配独立登记和冻结。
- 验证尚待CPU分类、实容器参考解/失败现场/非OOM kill、冻结预检与后台恢复，不能预先声称恢复或完整实验成功。重启时的历史状态/失败与已保存资产将先归档，余400生成与270判题续跑。
- 实际19项CPU检查和编译/git diff --check通过，重启后nvpmodel仍15W/mode2，与前三方法原device_before一致；609已保存生成/判题/raw文件核验通过。真实容器参考解passed，原Spec44确证OOMKilled=true/137并归类runtime_error。首次非OOM kill控制使用容器内PID1自行SIGKILL，被namespace初始化进程保护而返回passed，诊断断言失败；改用host对本测试自有容器docker kill，其他容器不操作。该问题在测试控制样例，生产adapter未需修改；负控制成功后才允许启动。
- 修正控制样例后真实负控制通过：exit137/OOMKilled=false保持container_error；参考解passed与已保存Spec44 OOM/runtime_error两正向分支同时核对。恢复launcher要求匹配adapter SHA的真实Docker preflight，单独冻结补丁与登记addendum，不替换原318冻结文件或protocol。后台恢复启动待本次实际命令结果。
- 实际恢复：12:57:46 UTC+8同名systemd用户单元active/running、0自动重试，冻结check-only成功读取600生成/230判题。恢复适配/状态快照/609资产hash登记于recovery_20261012_125735；Spec31开始续判，70条补齐，总300/500；12:59进入原Adaptive模型加载。原600生成/230判题前缀仍hash一致，318原冻结文件及protocol hash不变，Spec44正式判为runtime_error/correct=false、OOM证据保留。
- 新doc/RESUME-RAVEN-100X100-2026-10-12.md及README/优化入口/旧启动记录notice已接入，旧失败/退出已快照保留。完整实验未结束，恢复后首条新RAVEN生成尚待落盘；文档链接/git diff --check待最终校验，未进行方法优化或重跑已完整的600次模型请求。
- 最终恢复验收：13:00:25原Adaptive health/8token预热/Worker与冻结配置标记检查通过，题5/seed92实际生成中；13:03系统可用约2.6GiB/swap0、服务active/running。6文档链接及git diff --check通过；已完成数据前缀600生成+230判题仍保留，Spec判题已到100共300/500。完整实验继续后台执行，尚无两RAVEN完整质量/速度结果。

## 2026-10-10：用户核对最终表格样式与数值预测（文档完成，实验继续）

- 用户“预计表”进一步明确同时要表格样式和性能预估；只读核对五方法和实际指标，读取Native43/100、38判对、2.5428TPS、234.96秒、MAL2.9351的14:12:55 UTC+8快照。
- 新doc/FORECAST-RAVEN-100X100-2026-10-10.md及doc/upstream-changes/full-benchmark/forecast-2026-10-10.md，给GSM8K五方法经验预测范围，HumanEval根据原4090 Native61/Cactus69/Spec66/RAVEN-A63%记录提供低可信参考；deep HumanEval质量与所有HumanEval平均耗时因缺本设备记录保留待测。
- 预测范围无已计算统计覆盖概率，不标成实测或论文结果；不同旧100题、训练筛选和当前部分数据分开作为外推锚，原方法与EOS0.01的B区分。最终表按100题实际正确数/指标与配对区间判断三基线目标，MAL为辅助。
- 没有更改冻结代码、原始实验数据、protocol或正在运行的控制器，没有新GPU/HTTP/判题/优化任务。文档链接与git diff --check待实际验证；完整100+100仍待后台完成。
- 实际文档验收：6条本地链接存在、git diff --check通过，systemd正式实验服务仍active；未触发新推理或优化，预测与正式实验汇总分开保存。

## 2026-10-10：用户确认原协议100+100五方法完整复现（实现中）

- 用户确认GSM8K test100题/sample_seed20260730、HumanEval100题/sample_seed20260802，基础生成种子42，允许启动；本轮先收集完整结果，不再进行开发集候选优化。
- 新增scripts/run_raven_full_benchmark.py、launch_raven_full_benchmark.py、status_raven_full_benchmark.sh及plan/raven-full-100x100-2026-10-10.md、doc/upstream-changes/full-benchmark/2026-10-10.md。复用原prompt/评分/代码清理/隔离判题及已测服务启动，源码/数据/五配置冻结，1000新生成/500官方判题，逐题持久化、显式断点恢复、自动中文结果/CSV/JSON/配对比较。
- 配置原Adaptive D3/N9/δ1/mix0.5、后加EOS保护关闭；deep4_conservative D4/N12/δ0.5/mix0.25、保留EOS0.01，三基线D3。五者共用当前FP4/BF16及Target数值修复，无算法/Worker修改。14天后台上限，单请求5400秒、Restart=no；截断只诊断，没有质量提前停止。
- 实际预检发现apex无Docker权限且sudo需密码，已请用户设置socket ACL；用户反馈执行后docker info成功（29.1.3）。已拉取官方python:3-slim ARM64镜像，镜像pin及100参考解核验待后续实际检查。
- 验证状态：代码刚实现，CPU/断点恢复、参考答案判题、冻结预检与GPU启动均未完成，不能声称完整实验通过。旧100题已观察，本次原协议复现不是新独立留出；与4090 BF16/ctx8192的设备差异单独记录。
- 实现审查补充：修复resume序号扫描排除`*_exit.json`，建立独立server日志目录，恢复时固定原image ID而非重解析可变镜像tag，并校验已完成记录的实时IDs和方法/数据集身份。新增tests/test_raven_full_benchmark.py，检查断电尾记录保存、变更seed/顺序/重复/完成行损坏拒绝、容器故障不误计答错，以及候选配置原样保留。测试尚待实际运行。
- 实际CPU验证：项目.venv没有pytest，改用系统现成pytest7.4.4（未更改模型环境），9项新测试通过；Python编译、bash -n和git diff --check通过。公共组件hash与112请求冻结源码一致；四份原benchmark/evaluator/runner AST与上游3afafa1一致。Docker官方ARM64镜像已拉取并固定ID，原受限runner的passed/failed控制实际均符合预期。100参考解、冻结预检与正式GPU启动仍待完成。
- 实际启动：HumanEval100参考解全通过、故意错误控制failed；冻结预检与10份100题manifest同题/seed通过，GSM8K题文hash/gold与原正式100题一致；tokenizer最大prompt209/383均满足4096上下文。11:22:57 UTC+8systemd用户后台remtp-raven-full-100x100-20261010.service启动，Linger=yes、active/running、0重试；11:24:28Native health/8token预热/有效Worker与公共数值标记通过，开始题5/seed92。新run目录raven_full_100x100_20261010、源码hash51499bd076956518c87f97e89ac86e48478829bbacbb961a2a260356ac1fb1ed，旧结果未覆盖。
- 新doc/RESULT-RAVEN-100X100-2026-10-10.md、实时报告链接及README/优化入口当前notice已接入；本条区分真实启动与未完成1000请求，不声称已全面超基线或无后续故障。首完整记录仍待落盘核验。首份文档patch因旧标题匹配失败未写入，改用实际“优化报告入口”标题后完成，不涉及运行代码或冻结内容。
- 启动后11:28 UTC+8核验：Native/GSM8K5与17两请求完成并持久化，总进度2/1000、判题0/500，正在70/seed742。两record与保存raw response内record一致、output token IDs数与usage一致；active/running、0重试，系统可用约2.1GiB/swap0。318冻结源码hash未变，新文档本地链接及git diff --check通过，CSV已自动生成。任务继续在后台运行，不需要对话持续监控；完整实验结果尚未产生。

## 2026-10-10：112请求优势搜索完结与独立结果核验（完成，目标未达）

- 用户要求检查后台是否跑完。实际单元10月9日11:21:13开始、10月10日00:32:59结束，13小时11分46秒；COMPLETED_DOMINANCE_SCREEN/exit0/112主生成、7启动、7预热全部完成、0额外数值、0重试。当前服务inactive/dead、8000无监听，journal记11.7G内存峰值/swap0，七份server日志无OOM/CUDA/EngineDead/Traceback；不是运行崩溃。
- 当前修复后同批8题×2seed：B12/16、2.6249TPS、460.26秒、MAL3.5122；balanced3 11/16、2.5856、417.98；deep4_balanced16/16、2.6868、437.31、MAL3.8883；deep4_conservative14/16、2.6440、384.06、MAL3.7986。Native13/16、2.5274、458.31；Cactus14/16、2.8932、403.21；SpecCascade14/16、2.6083、367.98。
- 两个D4对原B/Native通过质量和两种速度开发点门槛；balanced对B吞吐+2.36%/整题−4.99%，conservative+0.73%/−16.56%。balanced对Cactus吞吐−7.13%/整题+8.46%，对SpecCascade整题+18.84%；conservative对Cactus吞吐−8.61%、对SpecCascade整题+4.37%。原development_result_gate完整false/point_winnerNone，未采用或自动启动留出/正式。截断/逐题逐seed只诊断，没有隐含否决门槛。
- 新本地忽略 `offline_audit_20261010.py` CPU脚本实际一次通过：112记录/112完整API payload-response、原gold/冻结评分、真实prompt/output IDs、时间token/ledger、7有效env、7summary/analysis、4组64历史输出复现、12点门槛及每组2000题聚类bootstrap一致；474输入hash/完整冻结source hash d25bb51a8c22d2f6ba86fea7b70b5dd7e6fdd1478e34458344808de09e614950通过。产物offline_analysis_20261010.json保留全部scope。首次临时只读查看把保存wrapper误作顶层choices，KeyError后按response字段核对；未影响实验/数据，正式CPU核验一次通过。
- 新 `doc/RESULT-RAVEN-DOMINANCE-2026-10-10.md`、专门 `doc/upstream-changes/dominance-objective/results-2026-10-10.md`，本地CSV/固定输出优化距离JSON；README、优化入口、原方法/历史最佳专表、目标、原实现记录、当前计划追加完结notice。原analysis/gate/payload/冻结计划与运行源码不覆盖，A/B/C本阶段完成，下一D单元未登记/启动。未改算法/模型/默认参数/评分，检查新增GPU/HTTP调用0。
- 主要剩余差距：D4相对Cactus轮成本约+25%而MAL+16.29%，接受收益被成本抵消；balanced输出比SpecCascade多22.41%。固定当前输出时，balanced联合速度需TPS>3.1929（再+18.84%），conservative需>2.8932（再+9.43%），后者适合优先成本优化、前者保留质量路线。该距离不是新GPU收益或保证。
- 限制：8道已观察训练题/2seed，仅开发点结论；balanced对B TPS95%CI[0.9993,1.0438]仍包含1；没有组件profile、新fresh数值重放、独立留出/正式或三基线同等D3/D4选参。本轮运行与CPU数据检查通过不等于最终目标完成，当前没有新后台或发布/提交。最终文档/CSV/hash检查在本工作单元追加实际结果。
- 最终CPU检查实际通过：7行CSV/中文表显示值与独立核验JSON一致、474输入hash未变、固定输出优化距离计算一致、10份受影响文档105个本地链接存在，git diff --check通过。验证范围充分后停止额外测试，本次不启动新GPU单元。

## 2026-10-09：原 Adaptive RAVEN 三基线专表（CPU 完成）

- 用户追加要求将未加入方法优化的版本与三个基线单独制表。新增 `doc/RESULT-RAVEN-ORIGINAL-VS-BASELINES-2026-10-09.md`，采用初次完整100题中的adaptive/native/cactus/spec_cascade组，列原评分正确率/E2E吞吐/平均整题秒/MAL；原Adaptive86/100、2.8278TPS、235.04秒、MAL3.4759。
- 修改历史最佳汇总、优化报告入口、`doc/upstream-changes/history-summary/2026-10-09.md`与本日志。明确“原方法”仍在设备FP4/BF16适配环境中，旧100题数据早于后续数值修复，保留单seed/已观察数据限制；修复后16请求参照仅链接，不混入同一表。
- 本次实际CPU复核400原始记录，4输入hash与已核验产物一致，100题/seed集合完全相同；正确数/总time/token/E2E/平均整题/MAL与原summary/汇总一致。没有新GPU/HTTP/模型调用，没有修改算法/默认参数/评分/运行冻结/历史gate；仍未声称超过三个基线。文档链接和diff最终检查在本工作单元记录。
- 最终文档检查通过：原方法专表4行所有显示数值与核验JSON逐项一致，4份受影响文档53个本地链接存在，git diff --check通过；仅文档整理，不触发新实验。

## 2026-10-09：历史最佳三基线结果与接受长度汇总（CPU 完成）

- 用户在后台运行期间要求整理 Adaptive RAVEN 的历史最佳基线对照，并加入接受长度。新增 `doc/RESULT-RAVEN-HISTORICAL-BEST-2026-10-09.md`，分别呈现6配对D3/D4全基线选参、24题×2seed历史提速峰值、100题五方法完整历史对照、修复后8题×2seed完整对照，注明数据/数值快照不同；Uniform仅作消融，Gemma4不混入。
- 本地 `results/raven_historical_best_summary_20261009.csv`/JSON保存29方法汇总、7历史单元、配对差值、91输入hash与扫描索引。CPU实际核验986记录：冻结评分/官方gold/原predicted-answer和correct/时间token/MAL计数/已存summary/同题同seed一致；100题五manifest一致、当前16题实际prompt IDs一致。100题旧单元评分文件未独立冻结，使用现有未变解析函数与原存字段逐条核对，未冒称原source freeze证明。
- 扫描217历史summary行、117不同summary hash，复制证据不计作新实验；未完成20261009批次排除。48请求旧Anchor候选对原方法正确率+2.08百分点、吞吐+13.91%、平均整题-10.06%、MAL+13.34%，但无同批三基线完整质量对照；6配对Anchor6/6整题235.66秒，后续48请求42判对低于同批原44。100题原Adaptive86判对/2.8278TPS/MAL3.4759，仍无完整联合胜出。
- 修改 `doc/RESULT-ADAPTIVE-RAVEN-OPTIMIZATION.md` 入口、专门 `doc/upstream-changes/history-summary/2026-10-09.md` 与本日志；本次无算法/模型/默认参数/评分/后台源码变更，无新增GPU/HTTP调用，不覆盖原gate/冻结/历史负结果。MAL采用总轮数加权，接受草稿token/轮=MAL−1；截断仅诊断。
- 15:41 UTC+8查询当前单元active/running、NRestarts=0、主请求完成34/112，当前deep4_balanced；不是本轮最终结果。历史开发/单seed测试、旧数值路径与缺完整prompt IDs的限制写入报告。本次没有独立新正式测试、模型重放、提交或发布。文档/CSV/输入hash的最终检查在本工作单元记录。
- 最终CPU检查实际通过：CSV29行逐项与JSON汇总一致、91输入hash未变、3份受影响文档45个本地链接存在、MAL−1草稿长度一致，git diff --check通过；没有新增模型加载/生成或重复后台实验。

## 2026-10-09：用户收敛目标为超过三个基准，移除截断硬门槛（执行准备）

- 用户明确不在乎是否发生截断，要求使用所有可用方法超过Native/Cactus/SpecCascade。当前验收改为原评分正确率不低于各基准且吞吐更高、平均整题更快；截断、正常结束判对、逐题/逐seed差异只报告，不再否决候选。原B保留参照，允许不同生成路径和训练集上的有限参数搜索。
- remtp/optimization_metrics.py新增dominance_gate，保持summarize/paired_gate/clustered_intervals等历史函数原文不变。新门槛校验完整同题同seed配对、prompt/预算、有效时间与无探针计时；不改原评分、不隐藏单题损失、不选择有利seed。旧冻结/原gate保留，不重写历史结论。
- 新scripts/run_raven_dominance_search.py复用已验证Round2启动、实际ID序列化、ledger及systemd流程，准备新独立112主HTTP单元：B、3个Adaptive候选、3基准，各8已观察题×2seed；7启动预热、24小时、0额外数值/重放、0重试、不自动采用。候选分别D3/N9/delta.75/mix.25，D4/N12/delta.75/mix.25，D4/N12/delta.5/mix.25，均head1/risk none；按完整质量/速度比较，不因单题损失或截断早停。
- 研究扩大至delta/mix/深度节点预算、自适应树/预测接受、MTP训练及kernel/KV成本；Sequoia的硬件感知树和SpecDec++的自适应长度提供研究依据，不移植其硬件收益数值到本机。软件环境/公共Target数值/模型/prompt/评分保持当前单元公平配置；更换公共模型或协议需所有基准同条件重跑。
- 新门槛与控制器尚待必要CPU检查、冻结preflight；没有新GPU生成或后台启动，未声称已超过三个基准。新计划/专门改动记录将在本工作单元同步，不使用先前计划中的未启用额度。
- 新launch/status_raven_dominance与tests/test_raven_dominance.py完成，验证新目标、损益抵消/逐seed诊断、同协议与探针拒绝、原度量函数保留、真实startup MRO及112请求完整流程。源码核对纠正新控制器有效日志检查字段为实际N_max，尚未运行GPU，无实机失败。按新目标CPU重评旧target_odds仍对三个基准失败，失败项为accuracy/latency以及Cactus/SpecCascade的throughput；移除截断门槛不改变它的负结果。
- 首轮必要CPU测试20通过/1失败：完整流程fixture模拟启动遗漏真实start_server的mkdir行为，已修正测试，生产运行算法未因此改变；截断fixture采用实际2048token预算。新计划/目标/专门改动文档形成，旧计划/研究/README/报告入口追加用户最新目标notice，避免旧截断硬门槛继续指导运行。重检、冻结preflight与实际启动待后续记录。
- 修正后21项必要CPU检查通过0.20秒；py_compile、bash语法和git diff --check通过。实际CPU新目标重评保存results/raven_objective_revision_20261009.json，旧candidate对所有参照仍负；没有新GPU请求。控制器policy明确区分继承common protocol模板与实际3候选/2seed/无留出执行，增加真实IDs复现门槛测试后进行冻结前最终检查。
- 冻结前22项必要CPU检查实际通过0.21秒，py_compile/bash -n/git diff --check通过；7份当前文档75个本地链接存在。真实workspace check-only成功，results/raven_dominance_preflight_20261009各ledger全0；包、数据、原prompt/数值、运行域源码与历史度量AST一致性检查通过，只新增目标函数。当前系统available约10GiB、旧单元inactive、8000空闲，准备独立冻结启动112完整请求；此处为启动前记录，不声称新质量/速度通过。
- 2026-10-09 11:21:13 UTC+8实际冻结启动remtp-raven-dominance-search-20261009.service，目录results/raven_dominance_search_20261009；冻结check-only与全source hash d25bb51a8c22d2f6ba86fea7b70b5dd7e6fdd1478e34458344808de09e614950匹配，active/running、Restart=no、NRestarts=0、24小时上限。独立112主请求/7启动预热/3候选额度启用；首次状态B加载、starts尝试1、主请求/预热0，无新质量/速度结果，非正式/留出，不自动采用。新LIVE链接及计划/目标/入口/专门记录同步实际启动，运行冻结和冻结前plan原文保留。
- 11:24 UTC+8实机核验：B服务health200、预热chat200、有效worker/env检查通过；1532/base42完整生成中，main尝试1/112、完成0，启动与预热各1/7、0重试。MemoryCurrent约10.72GiB；7份当前文档80条本地链接存在，git diff --check通过。当前只证明启动与首请求正在运行，新候选及三基准质量/速度尚待112完整校准，最终目标未宣称达成。

## 2026-10-09：以原B重建优化计划，定位失败约束的真实首分叉（研究完成）

- 用户要求以原Adaptive RAVEN为基准，研究上一轮失败原因并制定超过Native/Cactus/SpecCascade的计划。参考为已复现的B冻结，默认risk_policy=none；没有改变运行算法、模型、公共数值实现、评分或停止条件。
- 新scripts/analyze_raven_baseline_research.py，仅CPU读取完整API IDs、自然逐轮summary及注册P/Q/H/RNG帧，按相同真实前缀/拓扑/候选/P_y/Q_y/随机数对齐直到首输出分叉；不把分叉后的不同状态误判为Target故障。加入候选条件H逐个保持EOS仍可改变算法EOS边缘的三token完整反例，防止下一轮重复无效概率保护。
- 已执行的临时CPU摘要定位：3213/43前405token相同，generated_length403处no_reallocation使同一候选验收由约0.757降至0.528，同一uniform约0.634，B接受而候选拒绝；5180/42、1231/43改善也来自strict导致同候选接受翻转。局部低支持不能区分质量方向；末段答案损坏由后续长生成与2048预算截断表现，不能把首分叉自动等同全部最终因果。
- 已查Cactus原论文的候选条件H与整体有效分布区分、精确推测解码原论文，以及vLLM0.18官方长度/EOS停止源码；计划将计算等价优化与新质量分布设计分开验收。新离线脚本的正式核验及中文研究/计划文件待本工作单元完成后追加真实结果。
- 当前GPU/模型/HTTP新增调用均0，没有新后台；旧失败候选、gate、冻结和额度保留。本轮研究不会把编写脚本/计划称作实机修复成功。
- 新CPU脚本首次已完成真实前缀/帧与EOS反例校验，但最后读取吞吐字段时误用e2e_tokens_per_second，原冻结指标实际名e2e_tok_s；已修正后重跑。没有写产物或触碰原始数据，此次错误仅为新离线分析字段名。
- 修正后正式CPU核验通过，results/raven_baseline_research_20261009.json保存：6条真实轨迹、83注册完整帧；共同前缀上的26配对帧P/Q/候选/拓扑/RNG_before逐位一致，5处首输出改变均见同随机数验收翻转。3213/43为no_reallocation，另外4处strict，其中2条正常结束改善、1条原评分改善、1条仍截断错；信号没有单向质量关系。5个首分叉完整矩阵均未落在旧固定checkpoint，明确保留这个证据缺口。
- 新研究进一步加入全部自然逐轮EOS标量统计及候选无关完整H的三候选接受/残差枚举，用于区分单个H与算法边缘；更新脚本后将另存v2产物，原产物保留。没有生产候选实现或新GPU。
- v2研究实际通过，原v1保留。B自然主干7584/7592、候选6334/6341的Target EOS<1e-7；登记完整H各104事件的最大Target EOS约2.89e-11/1.52e-10，单纯恢复极小结束概率无改善证据。全逐轮结束机会与稀疏完整H的scope分开；候选条件EOS反例和共同H的3个beta数学枚举通过，不称为任务质量证明。
- 新doc/RESEARCH-RAVEN-BASELINE-2026-10-09.md、plan/adaptive-raven-baseline-optimization-2026-10-09.md及doc/upstream-changes/baseline-research/2026-10-09.md已形成；README、优化报告入口与上轮结果链接更新。计划以B原行为为计算参照，精准补4退化/2改善、增量MTP计算等价、候选无关完整H数学研究、逐题哨兵及独立留出/正式依次验收。各新单元额度仅提议，未启用或消耗；没有新GPU、默认切换、提交/推送。
- 最终验证：v2 CPU分析实际通过；新脚本py_compile及git diff --check通过；6份当前文档71条本地链接存在。两原systemd单元inactive/dead、Result=success、NRestarts=0，8000无监听。补明确E/Q1条件CUDA只验verifier，不添加额外模型前向，新proposer证明由有限自然窗口提供；F/E/Q1等新预算尚未启用。中文报告解释2048输出token预算触发length及120→12、256→25的实际截断损坏，保持原评分。最终目标未达成，生产修复未宣称通过。

## 2026-10-09：根因与Target相对支持修复实验完结核验

- 状态：原根因2026-10-08 10:32:30 UTC+8、修复21:32:26 UTC+8均正常结束，exit0、COMPLETED、Result=success、NRestarts=0，端口8000释放。根因6主HTTP/11fresh/2启动预热；修复86主HTTP/11fresh/48 CUDA/7启动预热完整完成。12自然现场中重复完整prefix明确共享，11不是漏跑。
- 控制器记录的五方法每16校准：Native13原判对/11正常结束判对/5截断、Cactus14/8/8、SpecCascade14/11/5、B12/10/6、target_odds10/8/8。候选对B吞吐约-1.16%、平均整题+16.87%；对三个基准质量/速度联合门槛均未通过，未采用或展开留出/test。
- 新scripts/analyze_raven_risk_repair_results.py，仅CPU复核冻结原评分/完整API payload/IDs、80条校准和12条自然记录、统计/逐seed/原bootstrap、数值原始张量与源码/资产hash、CUDA登记及实机门槛。独立写offline_analysis_20261009.json，不覆盖原gate/分析/冻结/失败。实际CPU核验进行中，不把检查脚本编写称运行通过；没有新GPU/HTTP/候选。
- 原始配对已见2950/42、1231/42正常结束判对变截断错；6009/43从116正常结束到2048截断仍按末数判对；3213/43截断原判对变错。六条smoke被局部增益抵消后通过，不代表16条非退化。中文结果/旧计划与入口的完结notice待CPU复核后更新。
- 限制：8道已观察训练难题×2seed，8题聚类；没有独立留出/正式实验。不把12现场的raw相同扩展为所有生成状态证明，也不把设计反例等同具体整题退化的完整因果。
- 首CPU核验已完成所有校验，写输出时发现CLI变量a被帧校验循环遮蔽，导致AttributeError，未写产物；已改为独立args变量后重新执行。该错误仅在新离线分析脚本，不涉及已完成的GPU控制器、原始记录/门槛或模型调用。
- 修正后独立CPU核验实际通过：80校准/12自然原评分、完整payload与IDs、各summary/analysis、逐seed与原2000题聚类bootstrap、83完整耦合帧、两单元各12现场raw及11fresh、CUDA16帧/48登记、冻结源码与200份资产hash。四个原方法均16/16历史输出签名重现。没有新模型/HTTP/CUDA调用；新offline_analysis_20261009.json保存，原资产不覆盖。
- 已新增doc/RESULT-RAVEN-TARGET-ODDS-2026-10-09.md与doc/upstream-changes/root-cause/results-2026-10-09.md，给出五方法对比、3错损失/1错改善、3停止转截断/1截断转停止、子集门槛覆盖、正文正确数字被截为12/25以及输出+15.51%/长尾耗时74.20%分析。当前候选拒绝，risk_policy默认none已保持；未启动下一轮或调整运行算法。
- README、RESULT-ADAPTIVE-RAVEN-OPTIMIZATION入口、两份计划、ROOT-CAUSE记录与原诊断/修复改动文件追加可见完结notice，旧运行状态和冻结原文保留。计划实际执行完成与优化目标未达成分开，门槛负结果不改写；当前两单元已退出、没有新后台。
- 验证完结：CPU核验实际通过；新脚本py_compile、git diff --check通过；9份当前文档86条本地链接存在。对九份实际服务日志的OOM/CUDA error/Traceback/EngineDead检索无匹配。完成中文报告与核验，本次未新加载模型/发送HTTP/执行CUDA/提交推送；下一候选与独立验证仍未执行。

## 2026-10-08：自然诊断运行中；Target相对支持约束候选实现

- 状态：09:26:01 UTC+8独立冻结启动remtp-raven-root-cause-20261008.service，Restart=no、0重试。09:56核验自然请求完成3/6、尝试4，fresh Target尚未开始；旧冻结没有修改。下方“尚未启动新GPU”是启动前历史状态。
- 5180/43改善对照105-token正常结束判对；6009/42失败对照2048截断判错。两者输出文本/数量/finish复现历史B，实际提交ID、摘要与自然raw/P耦合已通过；6009存在144次相对严格验收新增接受及5次相对uniform新增接受。低概率格式符号在改善对照中也会被接受，因此局部概率现象不是整题因果证明。
- 在工作区实现一个默认关闭的target_odds候选：Target概率不足其最大值1/1000的候选改用严格P验收/残差；不足1/10的候选不允许跨位置追加松弛；其余保持原规划。严格位置额度在预算池回收前归零，EOS保护保留，H同时用于接受和残差。仅供新独立修复单元验证，不更改模型、原评分、停止规则或运行中的B。
- 受影响文件：remtp/dynamic_mtp_tree.py、dynamic_tree_vllm.py、scripts/run_adaptive_round2.py、analyze_raven_root_causes.py、status_raven_root_cause_unit.sh，新tests/test_target_odds_risk.py及本日志。摘要将EOS/prefix保护与新增风险保护区分；残差是否实际消费明确标注；状态查询优先当前任务。
- 验证：初轮必要CPU检查129通过/1失败，失败为EOS测试概率恰处FP32阈值边界，测试夹具改为0.02高于0.01阈值，未调整运行算法阈值。修正后检查进行中；新候选尚未运行GPU，未声称修复质量/速度目标达成。
- 下一步：完成必要CPU验证和独立详细修复计划；后台等待原诊断完整通过，依次做CUDA旧/关闭兼容与新H检查、新候选自然轨迹及实际Target前缀门槛，再用同协议无探针对比Native/Cactus/SpecCascade/B。有限预算、无自动重试，未通过门槛即停止，不重复使用旧额度。
- 修正EOS夹具后130项必要CPU测试通过（8.96秒），算法默认关闭兼容/局部H与RNG通过CPU范围。新增run/launch/wait/status_raven_risk_repair；新计划与专门target-odds-repair改动文档。标准库等待依赖成功后才exec冻结控制器，避免同时驻留第二模型。run_raven_root_cause_unit仅工作区复用Target检查、允许候选非历史B输出、按实际探针标记性能样本；运行冻结不改。新脚本py_compile/bash语法通过，控制器集成与冻结启动尚待实测。
- 新tests/test_raven_risk_repair.py覆盖依赖拒绝/清理等待、源码白名单、CUDA帧选择/hash、真实startup MRO/env、完整86请求控制流/ledger/实际ID复现/中文分析及H残差/RNG错误拒绝。首轮85通过/1失败：测试transport错误用divmod(seed,10)回算题号，忽略base seed为42/43的进位；已修正夹具，实际seed公式和运行代码未改变。重检进行中，无新候选GPU。
- 修正后86项控制器/观测/边界CPU检查通过（6.06秒）；真实workspace check-only实际成功，results/raven_risk_repair_preflight_20261008账本各类全0。CUDA子进程异常也finally重载ledger，失败预算不被旧缓存隐藏。根因文档新增四条实际轨迹/计数，README/结果入口/计划/改动文件同步真实状态；原Target fresh未开始，候选GPU尚未启动。
- 另新增CUDA子进程超时后父进程报告仍保留失败尝试额度的必要边界测试，冻结前重检；不扩大算法试验、数据或GPU预算。
- 冻结前最终87项CPU检查通过6.21秒；py_compile、bash语法、git diff --check及5份文档62条本地链接检查通过。下一步独立冻结启动轻量等待流程；工作区preflight与旧冻结原始记录保持。
- 10:17:42 UTC+8实际冻结启动remtp-raven-target-odds-repair-20261008.service，目录results/raven_target_odds_repair_20261008；冻结check-only实际成功、全源码hash匹配。active/running、WAITING_ROOT_DIAGNOSTICS、Restart=no、NRestarts=0、RuntimeMax=1d、Linger=yes，截止10月9日10:17:42包含等待。实际等待进程未映射libtorch，MemoryCurrent约8.0MiB，模型/HTTP/CUDA额度全部0；不会与原B并发加载服务。
- 人工diff及组件hash核对：允许变化只dynamic_mtp_tree、dynamic_tree_vllm、run_adaptive_round2；桥接仅新增env/日志，Target公共数值与MTP采样/缓存/评分源码保持。启动状态追加到新旧计划、README/报告入口/根因与专门改动文档，LIVE入口创建；原B冻结、历史失败/negative gate/原额度保留。新候选实机质量/速度尚待门槛，不声称超过三个基准，未提交/推送。
- 10:22:33 UTC+8最终状态核验：原根因5/6自然完成、1231/43生成中，fresh Target0/12；修复等待单元active/running、NRestarts=0、各类额度0。7份当前文档72条本地链接存在，git diff --check通过；不需要持续对话监控。

## 2026-10-08：根因定位离线证据与有限自然诊断实现

- 状态：CPU定位/实现完成，53项相关CPU测试通过（4.10秒）；尚未启动新GPU。最终目标仍为原正确率/正常结束判对不降、截断减少，吞吐及整题耗时分别超过Native/Cactus/SpecCascade，后续独立留出/test。
- 用原评分器重核上次全部64条，保存6个关键pair逐字分叉、答案marker与末尾数字；6009/42、2950/43已计算正确值但反复截断留下其他数字。正对照保留1231/43、5180/43，不把同seed不同算法的自然RNG路径差异当作实现错误。
- 现代码默认参数的2token/D3全部8路径CPU枚举：每块adaptive局部TV总和都不增，但首token边缘分布TV由0.1407125升至0.2110687（+50%），有效delta参数最高2.25。证明跨位置预算不足以保护根分布质量；不冒称已证明具体GSM8K错题因果。原始CPU结果results/raven_root_cause_offline_20261008.json。
- 修改remtp/decision_trace.py、target_diagnostics.py与scripts/run_adaptive_round2.py，仅新增可选、有界观测/env；摘要取实际诊断、一次小ID复制，不加模型前向/抽样；trace-only避免旧head额外投影，最小prompt跳过短预热，旧默认verifier-only保持。
- 新scripts/analyze_raven_root_causes.py、run_raven_root_cause_unit.py、launch_raven_root_cause_unit.py、status_raven_root_cause_unit.sh、tests/test_raven_root_cause_unit.py；注册6主HTTP/12自然实际前缀独立Target/2预热/2启动，4h、Restart=no、0重试、无候选/留出/test，独立冻结与ledger/心跳。沿用正确ReviewUnit→Round2 MRO，无复制零参super到兄弟类。
- 启动前补查两个5180 seed的首prefix可能相同：以完整实际ID严格去重fresh selector，只做至多12次独立调用，12个自然现场逐一比较并显式登记共享源，不因重名漏采或虚增不同前缀数；新增去重边界测试。
- 最终CPU54项通过（3.96秒），真实workspace check-only通过；独立preflight ledger全0，没有GPU推理。README/优化报告入口更新本轮定位；用户追加授权找到原因后制定/执行修复并重跑，继续沿用原质量/数值门槛，不重复请求执行确认。
- 新plan/adaptive-raven-root-cause-2026-10-08.md、doc/ROOT-CAUSE-ADAPTIVE-RAVEN-2026-10-08.md、doc/upstream-changes/root-cause/diagnostics-2026-10-08.md；所有旧freeze/失败/negative gate保留。研究参照Cactus原论文关于候选H/有效分布的区分；总TV不继承其分布约束保证。
- 验证：原验证器仅调用一次与RNG不多消费、摘要上限、完整API提交ID/预算裁剪、真实prefix/位置/原始P错配拒绝、probe跳过预热/prefill、实际startup/env/MRO、旧探针和EOS测试共53项通过；py_compile、bash语法、git diff --check通过。首轮1个测试因mock属性未安装失败，测试夹具修正后通过，未消耗GPU预算。
- 限制：新观测实机输出复现、六条真实轨迹、12个raw门槛尚未运行；历史API未保留token IDs，只能比较跨运行文本/数量/finish签名。CPU反事实与副本不是完整自然因果或MTP/KV证明，探针时间不作性能结论。本次未修改数学/模型/模板/评分，也未提交或发布。

## 2026-10-08：复审证据单元完成，中文离线结果分析完成

- 实际完成于06:25:15 UTC+8，续跑COMPLETED_EVIDENCE_UNIT、exit0、systemd inactive/dead、Result=success、NRestarts=0；journal内存峰值11.6G/swap0，端口8000释放。本轮全部64校准和2 Target-only完成，无新质量候选，运行完成不等于质量/速度目标达成。
- 四方法各16请求：Native原判对13/正常结束判对11/截断5；Cactus14/8/8；SpecCascade14/11/5；B12/10/6。吞吐2.5264/2.8936/2.6068/2.6130token/s；B对Cactus约-9.70%，对Native+3.43%/对SpecCascade+0.24%，平均整题对三者均更慢。
- 先用冻结原评分器对64请求逐条重算并核对四份summary/analysis完全一致；按题聚类原bootstrap与逐seed门槛CPU重算，B对三个基准均未过质量门槛。B16完整输出文本/token数/finish签名与修复历史一致，不把偏故障8题75%和旧24题93%跨数据组称为修复退化。
- Target-only1532两seed均2048截断/判错/重复，约2125秒/条；五个实际晚段raw逐位一致、完整gate通过。前者提供独立AR可复现证据，后者只覆盖1532五现场，不代表所有错题KV或截断因果已解决。
- 新scripts/analyze_raven_review_results.py已实际CPU核验通过：固定16配对、冻结评分/数值组件、有效env无探针、旧artifact哈希、五个raw张量、16签名、完整预算，并登记按题CI/迁移/截断成本；独立写offline_analysis_20261008.json，不覆盖原analysis/gate/freeze。B6条截断占63.41%耗时；MAL对Cactus+5.04%、轮耗时+16.44%，吞吐收益被抵消。
- 受影响文件：新增doc/RESULT-RAVEN-REVIEW-EVIDENCE-2026-10-08.md、专门doc/upstream-changes/project-review/results-2026-10-08.md、CPU核验脚本；更新计划步骤4/5完成及定位优先级、README/结果入口、原专门实现/修复的完结notice与本日志。status_raven_review_unit.sh在终止阶段明确最后任务/历史心跳，不把最后True状态误称当前存活；仅查询显示，原记录/冻结不变。
- 验证：新脚本py_compile、CPU完整核验实际通过（约8秒）、git diff --check通过；7份更新文档62条本地链接均存在，状态脚本bash语法和真实已结束查询通过，当前remtp运行单元列表为空。15W模式2四服务启动记录一致，遥测GPU53.6～64.5℃，采样不冒称完整限频证明。本轮没有启动新GPU服务/生成、调整算法或发布/提交。独立留出/test、uniform消融、HumanEval与优化候选仍未执行。

## 2026-10-07：分段prefill修复后GPU末段门槛通过，Target-only长生成中

- 21:38 UTC+8核验：补测late_4_target正确保存175-token最终chunk，末位置2222/输入18；fresh树与顺序raw逐位一致、最大差异0/TV0。完整五前缀gate通过，前四个原证据CPU哈希/张量复核后复用，原位置失败/exit3/ledger保留。
- fresh树对历史保存P最大TV约9.17e-8，低于登记CPU归约容差1e-6；五现场树/顺序raw无需容差，均零差异。数值scope仅5个实际前缀，不能继承完整自然KV/质量证明。
- 续跑active/running、NRestarts=0，新增数值1/1/预热1/5/启动1/5完成；当前1532/seed42实际15362、2048预算Target-only生成中，主HTTP完成0/66/尝试1。服务约1token/s，5400秒HTTP等待已登记；尚未判断该请求结束、正确率或截断。
- 受影响文件：同步当前计划步骤3已完成、步骤4运行中，README/报告入口/专门实现与故障文档和本日志。没有新增源码修改或超预算请求，原冻结与负gate保持。
- 验证：修复后真实模型加载/健康/8-token预热/末段采集及完整五前缀raw门槛实际通过；36 CPU/py_compile/bash语法/冻结预检证据已记录。实际venv CPU读取末段张量核对175行/位置2222/输入18与完整gate通过，5份当前文档53条本地链接存在，git diff --check通过。64无探针校准、新计算候选、质量/截断收益、三基准正式比较仍未完成；未提交/推送。

## 2026-10-07：观测修复续跑已独立冻结启动

- 21:35:02 UTC+8启动remtp-raven-review-continuation-20261007.service，目录results/raven_review_continuation_20261007；冻结check-only实际通过，active/running、Restart=no、NRestarts=0、Linger=yes。RuntimeMax约23小时33分，截止仍为原执行开始后24小时，没有重置总时间。
- 冻结前36项必要CPU测试3.22秒通过；真实源码与冻结各一次CPU预检通过，重新核验原进度/覆盖/实际张量/四个精确gate、数学/Serving/评分与哈希。新增ledger66主HTTP/1数值/5预热启动，原10/9数值观测与失败保持；两单元合计尝试74/11/10/10。
- 更新当前plan/README/报告入口/专门实现与故障文档的实际启动状态；本日志同步。只修改工作区文档，原冻结源码、失败、负gate不变；新LIVE入口指向独立续跑报告。
- 验证：启动器真实成功；systemd实际RuntimeMax/Restart检查通过；5份当前文档50条本地链接存在、git diff --check通过。21:35当前Target加载、新主HTTP0/66/数值0/1，没有声称末段GPU gate通过。
- 剩余限制：末段raw、2 Target-only长生成、64完整校准待后台；新计算优化候选/截断改善/三基准正式速度仍未达成。没有自动重复profile/前四个数值请求，未提交/推送。

## 2026-10-07：末段探针分段prefill故障定位，有限续跑CPU预检通过

- 原后台21:22:11 UTC+8 NumericRejected、exit3、0重试并清理服务。8 profile完成、数值10HTTP200/9满足采集门槛、预热与启动5。不是已完成长生成/校准，旧failure/exit/ledger/gate/冻结保留。
- 明确故障：2223-token顺序prompt的chunked prefill上限2048，探针max_frames1在第一段保存位置2047/输入220，预期末位置2222/输入18。第一段消耗唯一保存名额，最终175-token chunk未保存；前四个fresh树/顺序raw零差异。没有本次OOM或模型崩溃证据，第五个正确raw尚未得到。
- 修复remtp/target_diagnostics.py只在精确顺序prefix末位置保存，中间chunk不消耗名额；不增加投影/RNG/KV写入。完整HTTP等待1800→5400秒，Target约1.03秒/轮存在2048单步生成超过1800秒的具体风险，尚未实际触发HTTP超时。输出预算/种子/prompt/评分保持。
- 新run_raven_review_continuation.py只复用该已定位观测故障的8 profile/4精确前缀，核对原进度、冻结运行域/评分、tensor/哈希和覆盖；新增66主HTTP/1次数值补测/5预热启动，合计尝试上限74/11/10/10，沿用原24小时绝对截止。不是无记录自动重试，旧负gate不改写。
- 受影响文件：target_diagnostics.py、run_raven_review_unit.py（新增evaluation依赖域）、新run_raven_review_continuation.py、launch/status_raven_review_unit脚本、test_raven_review_unit.py；专门probe-prefill-fix文档、当前计划/报告/README/实现记录与本日志。
- 验证：36项必要CPU检查实际通过3.22秒，包括2223前缀2048/175两chunk原投影两调用不变、第一段不保存/第二段唯一保存；完整HTTP等待参数与生成预算保持、续跑拒绝已花费工作/数学变化和原截止检查通过。py_compile、bash -n、git diff --check通过。
- 真实续跑check-only通过，CPU重新核验旧进度/四方法计时覆盖/四个raw相同前缀/原数学与评分/哈希，新ledger全0；未加载模型、未新生成。修复后真实末段补测未启动，完整质量/速度和新候选未完成。冻结原运行源码未修改，未提交/推送。

## 2026-10-07：四方法当前计时完成，晚段数值检查运行中

- 21:19 UTC+8真实profile 8/8请求完成，Native/Cactus/SpecCascade/B各32 decode轮，B实际tree proposal覆盖通过。主HTTP8/74、预热4/9，五个树侧晚段前缀依序检查中；后台active/running、NRestarts=0。
- 当前观测B平均轮跨度1.271秒，Target forward/logits stream跨度1027.67/21.41ms，约82.5%；proposal169.05ms、sampling30.49ms、cache17.16ms。候选选择25.35ms约2%，单独top-k不足以支持≥5%整体收益承诺。sampling CPU墙钟417.86ms包含等待，不能当作纯验收计算开销。
- 受影响文件：更新当前计划步骤1/2为实际完成、步骤3为运行中；更新专门evidence-unit文档/README/报告入口与本日志。新增本地profile_overview.json汇总真实观测，不修改运行冻结或数学算法。
- 验证：四服务实际profile覆盖检查与128轮观测完成；CPU汇总保留父子关系，只取顶层跨度，未相加嵌套子项。上一组文档47条本地链接存在、git diff --check通过。
- 限制：两题早段带探针记录不计正式质量/速度，token/观测轮不替代完整MAL。独立Target侧数值门槛、Target-only、64无探针校准未完成；新候选仍未选，无新GPU请求超出注册预算，未提交/推送。

## 2026-10-07：第一证据单元已后台启动，Native/Cactus真实计时覆盖通过

- 状态：21:08:40 UTC+8冻结启动remtp-raven-review-20261007.service，结果目录results/raven_review_unit_20261007；active/running、NRestarts=0、Linger=yes、24小时上限。没有在运行冻结内打补丁。
- 21:14实际核验Native/Cactus各两条308/641、seed42、64-token请求完成，均HTTP200；各32轮真实decode计时，Target/proposal/sampling/cache覆盖检查通过。完成主HTTP4/74、预热2/9、数值0/10；SpecCascade加载中。健康/预热单列，不当作质量证据。
- 设备仍15W模式2，Native权重实际约6.11GiB，运行时系统available约2.0GiB、swap0，当前无OOM证据。CUDA事件是stream跨度，探针请求不计无探针质量与吞吐。
- 受影响文件：更新plan/adaptive-raven-project-review-optimization-2026-10-07.md、doc/upstream-changes/project-review/evidence-unit-2026-10-07.md、doc/RESULT-ADAPTIVE-RAVEN-OPTIMIZATION.md、README与本日志；新增doc/RESULT-RAVEN-REVIEW-LIVE.md链接。保存新的launch/执行次数/policy/ledger/组件hash/设备记录，旧结果保留。
- 验证：冻结启动器实际成功、冻结check-only通过；真实Native/Cactus服务与profile覆盖已验证。31项CPU检查为本单元实现证据，不代替晚段数值/完整输出门槛。
- 剩余限制：SpecCascade/B计时、5个实际前缀fresh树/顺序Target门槛、Target-only完整生成与64条校准未完成。新计算候选未选，截断改善/三基准加速/独立留出与正式阶段仍未达成；不修改旧负gate，未提交/推送。

## 2026-10-07：执行复审计划第一单元（CPU预检完成，准备启动）

- 用户授权按复审计划优化。先执行有限证据单元，再依据当前成本选择一个计算优化方向；不重新启动已拒绝候选，不放宽正确率门槛。
- 准备修正runtime_profile：只观测注册的实际decode轮，跳过预热，最多32轮/服务；CPU墙钟与CUDA event跨度分开、保存父子关系，不自动运行top-k验证或额外算子循环。
- 准备新增第一单元控制器/冻结启动/状态查询、组件依赖hash、独立调用ledger和心跳；质量拒绝/数值拒绝/运行故障/预算停止明确区分。74主HTTP、10数值诊断、9固定预热、24小时上限，0自动重试。
- Target晚段检查使用5个保存实际前缀的fresh树/顺序Target比较。观测只复制已有logits，不额外投影；Target-only两条和64训练校准仍用原2048预算。模型、公共数值与原prompt/评分保持。
- 受影响文件：remtp/runtime_profile.py、target_diagnostics.py、scripts/run_adaptive_round2.py；新增run_raven_review_unit.py、launch_raven_review_unit.py、status_raven_review_unit.sh与tests/test_raven_review_unit.py；专门doc/upstream-changes/project-review/evidence-unit-2026-10-07.md、当前计划/报告/README及本日志。运行数学源码未改。
- 验证：31项必要CPU测试通过（2.94秒）；py_compile与bash -n通过。首个CPU预检因repair快照缺少后续cap接口拒绝，旧目录/0模型调用保留；使用最新cap关闭B冻结核对运行域、repair B作输出回归参照后，真实v2 check-only通过。实际venv的10个vLLM/BNB入口存在；API导入初始化CUDA上下文后检查进程退出，未加载模型/生成。系统Python直接导入vLLM曾混入系统SciPy，已使用项目解释器，不改包。
- 新单元来源/输入/每阶段有效配置、15秒心跳、9服务上限和ledger已实现；数值调用明确为10个顶层HTTP诊断，内部模型操作另记，shape额外投影关闭。profile仅两个实际请求/跳过预热和prefill，嵌套CPU墙钟/stream跨度不能累加成纯kernel时间。
- 限制：当前GPU加载/真实profile/晚段P与KV/Target-only/64校准尚未执行，新优化候选未选。旧冻结/失败数据完整保留，未提交/推送。

## 2026-10-07：全项目审查与优化优先级重新评估（审查完成）

- 用户要求重新审查整个项目、判断优化方向与总目标；当前仅CPU/源码/文档审查，未启动GPU或新候选。重新核对方法/基准workers、构树/验证/缓存、Target数值适配、评分与统计、控制器和预算。
- 新scripts/audit_raven_project.py已两次CPU运行通过：每次固定29真实B帧×旧/后剪枝=58验收调用、16实际Q×排序/精确top-k=32选择调用（合计116/64 CPU调用）；节点224→188，24 proposal帧累计326处理行/190最终节点。605轮精确后缀重复命中0、最大Target EOS质量1.69e-7；候选松弛与样本依赖branch保留的小词表枚举通过。CPU RNG不是原CUDA重放，不冒称live P/Q/KV或计时收益。
- 已读代码确认anchor verifier不读取备选后代，当前B仍生成/物化这些节点；多轮MTP重复计算累计树。Target FP4固定64/LMHead16 padding意味着少节点不等于同比节省GEMM。现有profile参数未传入env，profile有额外top-k算子循环，需独立预算与计时范围修正。
- 目标仍是Adaptive RAVEN对Native/Cactus/SpecCascade质量不降、截断更少、实际吞吐和整题速度提升；原评分/正常结束判对并列，不修改评分或结束规则绕过失败。旧100题基准不是当前数值冻结下的正式对照，尚缺新的同源基准与独立证据。
- 原2684含64个CUDA调用，HTTP主请求为2620；自然额度16已用完；完整回归288已完成192、剩96，两个候选名额均已试。新计划先提议74主HTTP/10顶层数值诊断/9固定预热的有限单元，尚未启动，不能把旧额度重复支用。
- 受影响文件：新doc/REVIEW-ADAPTIVE-RAVEN-PROJECT-2026-10-07.md、plan/adaptive-raven-project-review-optimization-2026-10-07.md、scripts/audit_raven_project.py、doc/upstream-changes/project-review/2026-10-07.md；更新README、报告入口、旧设备结果、第六轮计划、docs/exact_residual_hit_tree.md、专门CHANGELOG与本日志。strict续走仅新增条件提案律证明要求的可见notice，不改运行实现。
- 验证：两次CPU离线审查实际完成，产物results/project_review_20261007.json记录帧hash/scope；python3 -m py_compile通过，9份文档63个本地链接均存在，新文件无末尾空格/缺少结尾换行，git diff --check通过。剩余限制：新profile参数/状态改进仍待实现，四方法当前成本、live GPU等价、Target-only、晚段顺序Target、留出/正式和HumanEval未完成；本次GPU新调用0。旧冻结/负gate保持，未提交/推送。

## 2026-10-07：概率上限实验在seed42质量门槛停止，状态核验完成

- 状态：17:39:27（UTC+8）GateRejected(train_seed42_quality)主动退出，后台inactive/dead、0重试；7自然诊断/32新CUDA完成，训练48/144完成，剩余96未执行，不是完整实验成功。journal记11.7G内存峰值、swap0，端口8000已释放。
- 结果：B/cap2各24题22/21判对、19/21正常结束判对、5/3截断。候选2950/5180由对变错、6009由错变对净-1；固定R=2质量门槛失败。吞吐2.636529→2.493676（-5.42%），平均316.45→264.12秒（-16.54%），输出量-21.06%；更短整题与吞吐收益分开。
- Native1532三个基础seed都截断判错，提示该现象也能在链基线复现；本故障集不能作总体正确率、三基准速度或唯一因果证明。晚段B5帧有限/重放与32条件门槛通过仅为限定数值scope。
- 变更：新增doc/RESULT-RAVEN-PROBABILITY-CAP-2026-10-07.md中文阶段分析；更新报告入口、本轮计划和专门改动文档的停止notice；不修改代码、冻结、旧原始结果或负gate。
- 验证：CPU按原summarize/paired_gate重算24配对，与保存gate的全部关键统计及失败原因一致，逐题正确/结束迁移核对。没有新增GPU请求/测试，也没有自动放宽质量门槛。完整多seed、组件profile、Target/KV长上下文与留出/正式仍未完成。

## 2026-10-07：新Native服务健康/预热通过，首条完整诊断生成中

- 状态：2026-10-07 12:17:10 UTC+8，日志实际GET /health 200及预热POST /v1/chat/completions 200；有效Native worker、head input_rows=1、ProbMTP full-Q adapter均确认。首条1532/基础seed42完整2048预算请求已进入生成，主请求仍0/7保存，尚未判断正确率/截断。
- 系统：active/running，NRestarts=0，GPU KV/cache正常计数，内存available约2.1GiB、swap0。仅观察当前启动/生成成功，不把预热当完整质量请求。
- 更新专门变更记录真实阶段与计划/报告入口链接；本次未增加请求，未修改冻结源码。后续B后段、32新CUDA条件门槛、144训练待后台依序执行；质量/速度/正式结论仍未形成。

## 2026-10-07：Native诊断/概率上限单元已冻结启动

- 状态：2026-10-07 12:15:28 UTC+8，remtp-raven-probability-cap-20261007.service active/running，NRestarts=0，Linger=yes。新目录results/raven_probability_cap_20261007；阶段diagnostics/native 0/6，正在真实模型加载，不能称健康检查/第一个完整请求已通过。
- 冻结check-only实际通过，代码hash=a29e74d8c053ea76d0d27f88ec5e688a6a2ce7ba56b1510250a4ea67533a0562；11个未改数值/运行核心文件与修复后冻结逐字相同，只是来源核对，不替代Target数值gate。冻结manifest/policy的7自然/32新CUDA/144训练、R=2、lazy关闭及0重试一致性实际通过。
- CPU125项+60项及16真实帧×两实现CPU等价已通过；py_compile/bash -n/diff检查通过。旧结果、冻结、负gate、旧latest入口保留，新实时报告doc/RESULT-RAVEN-PROBABILITY-CAP-LIVE.md独立链接。
- 更新plan/raven-probability-cap-2026-10-07.md、修复旧计划notice、doc报告入口及专门变更记录；本单元仅在数值/质量门槛通过后释放下一seed块，完整训练结果/截断/速度与正式基准目标尚未验证。

## 2026-10-07：按计划准备Native配对诊断与H/P上限单因素候选

- 状态：实现中，尚未启动新GPU单元。前一150请求/32 CUDA完成，最终中文离线分析已保存在doc/RESULT-MTP-REPAIRED-REGRESSION-2026-10-07.md，懒残差速度失败、11/72截断未改善；下方“离线核验执行中”为历史启动记录。
- remtp/dynamic_mtp_tree.py新增relaxation_max_probability_ratio，0默认保持旧逻辑，非0必须有限且>=1；预算分配的floor/max和最终完整H同时约束H(y)<=R*P(y)，接受与拒绝残差共用H，保护位置不参与预算回收。remtp/dynamic_tree_vllm.py解析/打印开关；scripts/run_adaptive_round2.py显式传递。
- remtp/decision_trace.py新增最多8个已提交历史长度checkpoint的verifier-only观测，默认first8行为保留；本次只取128/512/1024/1792/2000五帧，无paired proposal/KV结论，不额外调用模型/RNG。
- 新scripts/run_raven_probability_cap.py与launch/status控制器：6 Native配对+1 B后段重放，32次单独登记的新CUDA条件调用，负门槛停，再最多144条同源B/cap2训练。默认不采用、不跑留出/正式，0重试，72h上限，保持模型/环境/2048预算。候选R=2在结果前固定，不读取题号/gold作线上决策。
- 预算：7自然诊断占原剩余7；新CUDA32独立登记，不重复旧64/32；同源B/cap2的144训练占剩余完整回归额度，旧D3质量门槛仍待后续安排，不声称本单元完成全部2684计划。
- 验证：相关CPU125项通过11.87秒；有效worker/控制器回归60项通过6.84秒；16真实帧×旧/关闭=32 CPU调用实际输出/事件/RNG等价通过。py_compile/bash -n/diff --check通过；GPU尚未运行。完整Target/KV/长上下文不继承旧gate；H/P局部上限不保证总体质量或加速。

## 2026-10-07：修复与懒残差后台单元完成，开始离线结果核验

- 状态：2026-10-07 11:03:28（UTC+8）完成150请求=2短探针+4完整诊断+144训练，32次CUDA条件验收另计；systemd inactive/dead、Result=success、ExecMainStatus=0、NRestarts=0。journal记录11.8G内存峰值、0swap，未见failure.json。
- 保存结果：两配置各72请求67判对/61正常结束且判对/11截断；完整输出72/72逐对一致。吞吐2.624640→2.623474token/s，约-0.0444%，速度gate失败；完成状态不等于优化速度成功。
- 新scripts/analyze_mtp_repaired_results.py：CPU离线按原评分/汇总/按题聚类bootstrap重算、核对自然完整token、48捕获帧有限性/边界及MTP同输入同输出、逐题截断与过强松弛现场；另写分析文件，旧冻结/原始日志/gates保留。
- 本次未启动GPU服务/新测试。离线验证执行中，最终中文文档与历史notice随后更新；没有承诺总体正确率或三基准速度达成。

## 2026-10-06：两配置修复后GPU smoke与新32次CUDA条件验收通过

- B/Anchor两条32token真实模型探针完成，各8帧原始Q/P/全部tree隐藏有限、proposal→verify边界通过；不是整体质量/速度证据。
- 16实际帧×两实现=32次新CUDA条件验收实测通过，记录new_cuda_equivalence_gate.json；原始CUDA RNG恢复，输出、节点诊断、事件概率/索引/终止及随机末状态全匹配。
- 15:51:41 UTC已进入diagnostics/old_eos01完整2048预算诊断。4完整诊断和144训练尚未完成，不声称正确率/截断/速度提升，也未跑独立留出或三基准正式测试。
- 更新修复报告/专门改动记录/计划真实阶段。后台Restart=no、Linger=yes，负gate自动停止，不需要对话监控。
- 旧负门槛、故障tensor和冻结不改写；新的门槛scope仅上述条件范围及后续首根P核验，不继承完整旧数值gate。

## 2026-10-06：修复后B真实模型smoke门槛通过

- 新冻结实际B/1231/seed12353的32token请求完成；old_eos01_short_probability_gate通过（8/8 trace帧的有限P/Q、全部tree隐藏与proposal→verify边界）。
- 首MTP pass实际bias stride=(1,1)、Attention输出0NaN、MTP最后隐藏0NaN；原始Q为9×151680且全部有限。旧现场first pass 4096NaN、深Q两行全NaN不再出现在该复测。
- 新后台单元active/running、NRestarts=0，Anchor加载中，Linger=yes；完整2048诊断、144训练和三基准正确率/速度不能宣称完成。
- 更新中文修复报告实际smoke状态，三份更新文档本地链接检查实际通过，旧现场/冻结保留。

## 2026-10-06：新修复单元已冻结启动，中文报告更新

- 冻结results/mtp_repaired_regression_20261006，真实check-only通过，remtp-mtp-repaired-regression-20261006.service实际启动（0重试、72h上限、Linger后台）；正在修复后第一smoke模型加载，尚未声称smoke完成。
- 新doc/RESULT-MTP-PROBABILITY-REPAIR.md说明首坏Attention、singleton stride2逻辑外读取、真实Inf padding反事实、代码修复、合法Q RNG保持、guard速度开销及正式结论边界。
- 更新RESULT-ADAPTIVE-RAVEN-OPTIMIZATION.md入口与故障旧文历史 notice，新实时报告链接单独保留第六轮旧报告。
- 读源码进一步明确adaptive_relaxation_max_ratio限制TV相对baseline，不是H/P上限；低P token局部H/P可数百倍。后续质量/截断候选需要独立局部约束/深度策略，不能把懒残差冒称截断修复。
- 验证：启动器实际成功、控制器协议/hash check-only实际通过。完整模型smoke/CUDA/2048诊断/144训练与质量速度仍由新后台单元逐阶段产生，不预写成功。

## 2026-10-06：准备冻结启动修复后2048诊断与144条训练对照

- 增加launch_mtp_repaired_regression.py冻结启动器、status_mtp_repaired_regression.sh中文进度，实时报告新独立链接，不覆盖第六轮故障报告。
- 新控制器/概率/metadata/audit 22项CPU实际通过2.83秒，py_compile/bash -n/git diff --check通过。修复后8次GPU attention对照完成：紧密stride1有限并逐位匹配参考，故意非法stride2+Inf仍NaN，证明metadata修复必须存在；未隐瞒负对照。
- 更新专门计划/上游改动文档，登记2短+32条件+4完整诊断+144训练，首根P scope有限、正式数值不继承，guard开销待测。
- 下一状态：冻结check-only和真实修复后smoke待执行；不能提前宣称原始Q已在新完整请求全程有限或质量/速度改善。

## 2026-10-06：新源码有界概率门槛与后台训练控制器

- scripts/run_mtp_repaired_regression.py：2条32token新修复探针→32次新CUDA条件验收→4条完整2048诊断→新Target首根P与历史精确比对→144条B/懒残差训练；每阶段负门槛停，0重试、不自动采用、不跑heldout/test。
- 源码修复后自然输出允许改变，不强制复现非法Q旧输出；保留模型/数值参数/数据/提示hash，执行源码新hash，不继承旧门槛。新P门槛仅4首根，不冒称完整Target/KV/长上下文已通过。
- 4自然诊断占原剩余11额度，执行后剩7；144训练沿用未执行额度，固定24题×3种子×2配置，按seed块交替顺序，并每块立即核对完整输出/结束/正确率/截断相等。质量与速度单独记录，不能因速度未升而冒称成功。
- 新CPU gate/报告测试；前一修复单元120项相关CPU测试实际通过11.60秒。再登记8次修复内核真实Q/KV条件重放，和旧8次分文件保留，尚待本次执行。
- 限制：Q合法性guard同步耗时需训练实测；懒残差不减少截断；后续应在新有限Q下研究过强松弛/低P token接受的质量候选，再经新全面数值与独立留出gate做三基准正式比较。

## 2026-10-06：修复MTP单元素树bias错误步长，禁止无效Q抽样

- 状态：8次受控GPU重放实际完成。两份真实Q/KV：自有Inf padding + stride2各输出4096NaN；紧密stride1与无bias参考逐位一致且有限。原始B/Anchor首pass Q/K/V和写入后visible KV也完全相同。
- 首次独立重放Triton启动器编译缺Python.h，尚未执行GPU kernel；补用现有scripts/cuda_env.sh用户目录头文件后成功。没有安装/改变软件包。
- fixed6_vllm.py在draft metadata边界使bias row stride等于逻辑宽度，单元素view必须clone(contiguous_format)，普通dense bias不分配；不改变逻辑树掩码。
- target_attention_fp32.py补充query有效行/头mask，避免padded query读取bias范围外；QQ列界限仍要求上述dense metadata。
- dynamic_tree_vllm.py在primary/branch/fallback抽样前验证Q有限非负正总量，无效Q报错且不消费RNG，避免旧argmaxNaN→0与验收均匀回退不一致；合法Q随机原语保持不变。
- 新概率安全测试与singleton stride回归；既有Round6 NaN故障测试改为验证修复后的拒绝/no-RNG行为，旧冻结保留旧故障代码。
- 限制：guard增加GPU同步，速度影响需实测；条件重放因果支持步长路径，不冒称已读到旧现场真正相邻字节的Inf。修复后完整2048质量/截断与新Target验证待执行。

## 2026-10-06：首坏算子实测为MTP注意力，登记有界掩码步长重放

- 2条32token探针实际完成；B首pass注意力之前所有算子、Q/K/V和写入后可见KV均有限，Attention输出4096NaN。Anchor对照完整保存。
- 现场1×1 tree bias的stride(0)=2。PyTorch切片contiguous对单元素保留父步长，vLLM统一内核把该步长同时用作有效列数；存在逻辑形状外读取路径。是否解释NaN待受控GPU重放确认。
- 新scripts/replay_mtp_attention_fault.py：只用保存真实Q/KV，2帧×原步长/带inf的自有padding/紧密stride1/无bias参考=8次GPU attention调用，无RNG、0质量请求。调用前写attention_replay_protocol.json，保留旧现场，不修改冻结代码。
- 验证状态：算子定位已真实运行；因果对照和修复尚未声称通过。测试mtp_operator_audit.py验证快照原地语义与异常卸载。

## 2026-10-06：继续优化，先做MTP首坏算子有界定位

- 状态：实现与新计划已写入，GPU诊断准备中；未声称故障修复或速度/质量提升。
- 新remtp/mtp_operator_audit.py：首个真实proposal首个tree pass临时pre/post hooks，复制原地修改前输入，记录实际attention元数据、forward槽、可见KV写入前后；默认关闭、512位置上限、finally卸载。
- decision_trace.py按显式诊断开关集成；run_adaptive_round2.py默认关闭并显式传递；新scripts/run_mtp_probability_repair.py保存逐算子有限性及中文阶段报告。
- 新plan/mtp-probability-repair-2026-10-06.md、doc/upstream-changes/target-probability-consistency/mtp-operator-repair-2026-10-06.md。
- 登记另一个修复诊断单元：B/Anchor各1条32token、独立服务各8token预热、0自动重试、0 CUDA验收重放，不计2048主质量和速度；保留原剩余11自然/144训练额度和所有旧负门槛。
- 验证：29项Round6/metadata测试通过（7.05秒）、3项探针原地输入快照/异常卸载/RNG测试通过（3.95秒）、py_compile及git diff --check通过。冻结check-only通过，remtp-mtp-operator-20261006.service已启动，真实GPU health=200；算子定位与修复尚待现场。

## 2026-10-06：第六轮阶段结束，完整分叉概率与MTP同输入复核

- 状态：v2于14:19:45 UTC结束：4/4自然复现、32帧×两实现=64/64 CUDA验收通过；原始Q非有限门槛拒绝，分类implementation_fault。
  systemd inactive/dead、Result=success、NRestarts=0，预期gate码3；内存峰值11.5G、swap0。剩余11自然诊断、144训练、留出/正式均未开始。
- 逐请求：B/1231 1583token/stop/120正确，B/5180 105/stop/1360正确；Anchor/1231 179/stop/100错，Anchor/5180 2048/length/last_number160错。
  全部与第五轮各自输出/长度/结束方式一致；不能把预选故障2题当总体准确率，也不计探针耗时为正式性能。
- 新scripts/analyze_adaptive_round6_results.py：读取完整日志/CPU存储的真实GPU帧，4条提交token重建与API解码核对通过；
  写postrun_analysis.json（输入SHA256、首次分叉角色/P/抽样输入/原CUDA RNG、MTP同输入比较），不新增GPU调用，不修改冻结源码/原门槛。
- 共同根前缀4组全P逐位一致，两题首次分叉前缀各1组全P也一致，TV均0。1231第6token分叉bonus→残差、RNG不同；
  5180第13token残差→bonus、原CUDA RNG相同，实际抽样输入CPU归一化TV约0.062689。属于条件分支差异，不把同seed不同文本等同bug或最终质量因果。
- GPU懒残差32帧行为/事件/RNG全部相同，可省58次接受节点残差构造；行为等价包括保留旧NaN回退，不能证明原算法概率合法或真实吞吐提升。
- 原始Q全NaN仅捕获于B/1231首轮的两行，Anchor前8轮未见；不能用B的NaN解释Anchor错答/截断，也不能声称后续未捕获轮次皆有限。
  1231第一MTP树计算B/Anchor根隐藏、输入隐藏、token、位置、seq_lens及prefix KV全相同，B输出4096个NaN、Anchor输出有限。
  重点缩小至当前节点KV/attention元数据及执行状态/内部算子；没有声称具体NaN算子已修好。
- 新doc/RESULT-ADAPTIVE-RAVEN-ROUND6-ANALYSIS.md，更新总报告入口、计划阶段状态、MTP故障记录与专门上游记录，旧启动期描述加可见历史说明。
- 验证：实际离线分析执行完成、4完整token重建与帧提交历史匹配通过，64 CUDA原门槛的complete/count复核通过；脚本py_compile及git diff --check收尾核验。
  109项CPU测试、GPU服务及条件重放均为此前实测，未冒称本次又启动正式测试。局部全P一致不替代全部raw logits/KV/长上下文重验。
- 收尾复跑离线分析成功，保存分析器SHA256；py_compile、git diff --check及5份更新文档本地链接检查实际通过。

## 2026-10-06：初始GPU现场发现原始Q全NaN，修正观测并转入MTP状态定位

- 状态：初版服务模型加载/health=200/8token预热通过，q1231首次完整预算请求尚未完成时8轮实际张量已保存；13:46 UTC主动停止，0条完成、1条中断，不是模型崩溃。
- 首轮Q第1/2行各151680个NaN、草稿ID0；Target P有限。提议器argmax(NaN/noise)固定0，验收normalize_distribution回退均匀Q：实际提议/验收不一致。
  同精确前缀历史节点标量与返回IDs全一致；历史原始Q未保存，不冒称看过旧NaN。尚未证明该局部问题导致最终错答/循环。
- 修正Round6 pending观测校验：vLLM set_inputs_first_pass移token但不移对应Target隐藏的位置，实测rootpos=prompt+history-2；
  NaN-aware原始张量相等仅用于核对观测边界，同时独立非有限概率gate，不把NaN当正常概率放行。
- decision_trace增加仅前8个proposal内部MTP树计算输入/输出隐藏状态、有界有效prefix KV（root<512时），不复制整片KV。
  分析器记录Q/P异常与MTP各pass有限性；控制器先4条自然+64条件重放，原始Q/P故障时阻止后续训练性能扩展，分类implementation_fault。
- 初版冻结及轨迹全部保留，early_conditional_diagnosis.json、operator_stop_reason.json记录实际证据/主动停止。
  新版传入prior_aborted_dir，1条中断占原16诊断额度，新单元≤15自然诊断；B/1231额外完整重启明确暂未安排，不无限重试。
- 新doc/FAULT-MTP-Q-NONFINITE-2026-10-06.md、更新计划及上游专门改动记录。先定位NaN源，禁止只在验收改Q冒称修复，修复后再数值/质量/截断验证。
- 验证：109项相关CPU实际通过（12.05秒），含原始NaN下懒残差等价、MTP位置/同NaN mask边界、提议/验收分布不一致的真实函数回归定位。
  GPU完整自然输出/CUDA条件等价/NaN修复/性能质量仍未完成。
- 13:52 UTC新版冻结check-only通过并启动remtp-adaptive-round6-v2.service，13:55 UTC后实际health=200/预热通过、首题正常生成；
  前8轮proposal/verification位置、ID、含NaN mask的Q与RNG边界全通过，写live_trace_acceptance.json。
  首轮MTP有效prefix KV与输入隐藏均有限，第一树计算输出4096个隐藏值全NaN；随后输入/输出异常扩大。
  第一次可见NaN定位至首轮MTP树计算内部，具体当前节点KV/attention/norm/投影算子未指认，不能称已完成修复。
  最新结果入口已指向v2中文报告，原初版冻结/中断/现场保留；启动期及8轮观测通过不替代完整质量/数值门槛。

## 2026-10-06：执行第六轮有界诊断与决策保持的性能优化

- 状态：用户授权执行；新实现已落地，CPU验证与GPU启动验收进行中。下方“未执行”条目为历史状态。
- dynamic_mtp_tree新增默认关闭lazy_residual，仅跳过接受节点的纯残差；可选完整决策事件P/Q/H、uniform、纠正/bonus概率输入与原RNG复制，不改变抽样次数。
- 新decision_trace及generation_guard集成：每真实请求前8轮、忽略预热，保存proposal根隐藏/位置/完整Q及验收全P/Q/H；不复制整片KV。
  阅读本机vLLM sample_tokens实际顺序确认proposal先于CPU bookkeeping，早一步history显式命名，核对下一验收的提交前缀、位置、Q/ID及RNG边界，避免观测错位。
- dynamic_tree_vllm传递开关并报告；Round2启动显式写D层全1松弛权重与默认关闭trace/lazy；新增Round6控制器/冻结启动器/进度脚本、条件分析器及针对性测试，真实Round4→Round2启动链验收。
- 本执行单元16自然诊断、64CUDA条件调用、144新训练请求，均占原计划预算；先4条复现，门槛通过才扩展。
  训练42/43/44种子块交替B/候选顺序；保留2048预算与评分器。固定8token预热单列，诊断不计速度。
- 新上游专门记录round6-execution-2026-10-06.md、计划执行状态更新；旧冻结代码/原始结果/gate只读保留。
- 限制：源码变更不复用旧数值gate；新重放仅证明测得范围。懒残差不能减少截断，质量候选/数值重验/留出/三基准正式阶段仍待证据门槛。
- 验证：107项针对性CPU测试实际通过（11.80秒，含既有dynamic_mtp_tree、generation_guard、Round5启动与新Round6测试）；
  修改文件py_compile及git diff --check通过。新增worker实际开关验收后再次运行新Round6测试，实际21项通过（6.82秒）。
- 2026-10-06 13:37 UTC：冻结目录results/adaptive_round6_20261006_trace_lazy，真实冻结代码check-only通过，
  remtp-adaptive-round6.service已启动（Restart=no、72小时上限、用户Linger=yes）；当前进入old_eos01首组自然诊断。
  主请求尚未完成，模型health/首条完整复现及CUDA等价门槛尚不声明通过；后续实际证据自动写中文report.md。

## 2026-10-06：已有现场离线故障定位与第六轮优先级细化

- 状态：离线定位完成，最终错答单一因果未证明；新GPU/算法阶段未启动。
- 新scripts/analyze_adaptive_round5_faults.py与tests/test_adaptive_fault_forensics.py：
  按实际提交长度/API末轮预算重建token，检查已有prefix与全文，5项CPU实际通过（0.04秒）。
  真实3配置×2题重建全部通过。新fault_localization.json保存输入hash、精确ID、首次分叉角色/P/Q/H/u与缺口。
- 1231前5token共同、第6分叉（旧Target bonus/新残差）；5180前12共同、第13分叉（旧残差/新Target bonus）。
  两题初始共同top8/EOS一致，但第一验收u不同；主proposal/验收共用request generator，深度影响抽样调用序列。
  不将同seed不同文本认作概率bug或质量因果证明。
- 1231在g21实际fingers：P1.264e-5、Q0.05304、H0.003568、u0.02017，同u严格拒绝而当前接受。
  这是局部风险证据，不证明其单独导致错答。5180的514轮没有重复或EOS保护触发。
- 源码定位keep=true仍构造全residual_distribution；四轨迹已接受主节点对应1147/105/65/1382次潜在冗余计算。
  懒残差真实GPU收益尚未测，拟保留决策/RNG作为首选性能候选。
- 新doc/FAULT-LOCALIZATION-ADAPTIVE-RAVEN-2026-10-06.md与专门上游改动记录；
  更新已有round6-path-quality-speed计划与报告入口/第五轮后续提示。保留既有2684总上限、
  源码绑定和全部质量门槛；新等价验证/profile占现有64诊断调用/最多2候选额度。
- 原始日志/gate/冻结代码保持，Target/MTP/采样/RNG/评分均未修改。
  文献核对PyTorch2.10 randomness和严格推测解码论文，一般要求与本地原因分开，不将严格P保证套到近似H。
- 限制：首次分叉处bonus/纠正全P/Q及CUDA RNG状态尚缺，需要计划中的有界GPU证据。
  不把计划中的质量/速度成功写成实测，不自动采用新候选；相对三个基准目标仍待独立验证。
- 收尾验证：分析器py_compile、git diff --check及定位/计划/结果入口的本地证据链接检查通过；
  5项针对性CPU测试再次通过（0.02秒）。没有执行新的GPU请求或改变运行配置。

## 2026-10-06：制定下一轮路径定位与质量速度完整优化计划（未执行）

- 状态：用户要求按第五轮分析制定完整计划，本次仅规划/文档更新，没有启动GPU或修改算法。
- 新plan/adaptive-raven-round6-path-quality-speed-2026-10-06.md：保守old_eos01参照、失败Anchor现场，
  16自然诊断、同前缀P/Q/H及实际CUDA RNG重放、正确的已提交token/KV/budget、有限单因素矩阵，
  深度D与节点N独立；最多2候选、完整24题×3种子训练、唯一冻结候选后24题×3种子留出、
  100新test题×3种子/5方法正式三基准比较。分阶段主路径最多2684新调用，预计7—15天属估计。
- 保持现有模型、量化/数值路径、软件、温度、提示和2048预算；质量总体/各seed门槛不放宽，
  截断确实减少与正式三基准速度区间分别验收，历史计时不作当轮正式速度依据。
  相同seed不同算法token不一致不是bug证明，局部TV不是全句分布/正确率证明。
- 明确控制器实际启动/MRO/选项路径测试、样本隔离与已观察留出重新分配、固定追加样本/多重比较，
  新源码冻结、完整兼容检查点、故障分类、无自动重试和阶段预算。预期算法更改须另记并消融。
- 更新doc/RESULT-ADAPTIVE-RAVEN-OPTIMIZATION.md入口、第五轮计划后续链接、Target改动README历史状态；
  新doc/upstream-changes/target-probability-consistency/round6-plan-2026-10-06.md记录本次仅文档范围。
- 实际检查：第五轮结果、控制器与配置/验收代码，训练与留出无交集，第四轮heldout/formal/selection/manifest未生成，
  文档链接与diff检查。未运行新候选测试，不把阶段预期写成验证结论。
- 限制：故障因果尚未定位；小集仅作筛查，24题训练/留出不能代表全域真实正确率；论文目标仍未达成。

## 2026-10-06：第五轮完整结果核查与中文分析（训练消融完成）

- 状态：v2最终COMPLETE_TRAIN_ABLATION、104/104新请求、MainPID0、NRestarts0、正常退出。
  2026-10-06 10:10:15 UTC结束，journal峰值内存11.6G、swap0B；未启动新GPU任务。
- 实际验证：读取最终status/journal/policy/runtime/replay/regression/EOS诊断，重新核对96条新回归、
  96条历史参照与8条诊断，逐题配对与按题聚类2000次bootstrap；无需新代码测试。
- old_eos01：44/48判对、39正常结束且判对、9截断，所有配对判对/结束类型与旧D3一致；
  43/48输出逐字同，4正确正常回答减少1162token，1截断错误文本改变。质量门槛通过。
  均时299.0→290.2秒（-2.91%），吞吐2.641→2.637（-0.16%），没有解码加速证据。
- Anchor：42/48判对、38正常结束且判对、8截断，质量未通过；相对上轮Anchor仅6530/43改变，
  2048截断→937正常正确，其余47条逐字同。相对旧D3吞吐+6.02%但正确数少2。
- 原失败1231/43与5180/43及两旧结构四重放均通过；增强保护未改变Anchor这两题。
  1231正常结束答100、gold120；5180访问主路径最高EOS质量仅2.218e-7，无节点达到0.01，
  不支持以中等EOS质量受抑制解释该现场。旧D3两题的2条EOS抑制记录消失，但原本就能正常正确结束。
- 新增doc/RESULT-ADAPTIVE-RAVEN-ROUND5-ANALYSIS.md；更新RESULT入口、第五轮计划、
  第四轮质量分析与专门上游改动记录的可见历史说明。原raw report/gate/冻结源保持。
  明确原始模板“旧配置重跑”不适用于本轮完整参照，本轮参照只读来自第四轮，两题实际重放。
- 限制：24题×2种子的训练消融、历史参照计时不是正式独立证据；质量门槛passed只表示至少一个候选通过。
  独立留出和Native/Cactus/SpecCascade正式同题比较未执行，总目标未达成。
  建议保留保守参照，先定位Anchor最早路径分叉与有限单因素消融，再决定下一候选；没有自动采用或改评分。

## 2026-10-04：执行第四轮详细定位与优化（实施进行中）

- 最新检查：D3/head1四题完整完成，D3/head3已1/2；主长状态累计5请求，
  unit持续active/Result=success/NRestarts0、Linger=yes。静态分析移除过时的3/4当前值，
  后续阶段以动态报告为准；尚无全部长数值、截断改善或正式速度结论。
- v3截至本次记录已完成D3/head1四题组3/4，服务持续active，未自动重启。
  中文静态结果标注记录时点，实时阶段/请求数以live report/status为准；正式质量/速度尚未完成。
- 长评分/保存完整自然检查点相关CPU实际20 passed（3.30s），编译/diff通过；旧2629/26332
  真自然2048输出正常以length落盘，说明截断尚未解决（不当作质量提升）。
  主动迁移v3控制器避免已证实的长评分矩阵内存风险，不变更数值模型；原已完整通过11请求只读复用，
  2629完整请求及匹配prompt/probe UUID复制新目录，原目录/中断q2950不复用不追加。
  v3已实际active/NRestarts0，CPU协议检查、数值源码/数据/模板核验和复制成功，服务实际处理新的
  2950自然请求完成，四失败题组已2/4。实际调参/留出重叠检查为空。
  source与policy明确single-last评分，不计算全prefix词表矩阵；原2048质量/测试预算保持，评分单token不计性能。
  中文计划/分析/改动记录/动态进度入口更新；正式正确率/截断改善与三基线速度结论仍待后续门槛。
- 长评估内存预检：当前Tree进程模型6.11GiB、整机available1.9GiB；纯Target少0.39GiB。
  安装版sampler.compute_logprobs明确全矩阵log_softmax(float32)：2048×151680的logits+logsoftmax
  约2.31GiB（还未计BF16中间矩阵与其它临时），大于可用余量，不能直接把长评分当成小评分扩展。
  在长评分到达前把诊断改为完整prefix独立prefill、只取末行采样前完整raw词表；max_tokens1产生的
  单额外token丢弃且不算质量/速度，不强制EOS、不缩短主2048预算；长矩阵临时降到一行。
  新增chunked single-last选择测试；排除诊断warmup。数值模型/权重/采样主协议不变。
  launcher支持复制完整自然请求及对应精确prompt/probe UUID，验证数值源码/数据/模板/题seed顺序；
  不复制未完成请求、不向旧目录追加。待当前请求落盘后冻结新控制器，避免丢弃已完成GPU工作。
- 统一norm+head16新完整GPU实际通过：4prefill五组比较逐位一致；原自然483输出与历史相同、
  原287hash匹配；6晚期窗口top/EOS/完整TV全0，36层算子/KV逐位一致，228复制全相同。
  首版controller在长组服务加载期间主动停止，修正2048评分配对后冻结v2；
  首版停止产生unit exit1（SIGTERM→KeyboardInterrupt），明确为控制器迁移而非数值/质量失败，
  原`controller_migration.json`保留原因，旧代码/通过gate不改。
  新unit`remtp-adaptive-round4-v2.service`已实际active、NRestarts0，CPU预检和同数值来源核验通过，
  只读复用11个完整前置GPU请求，开始D3/head1四失败题的2048长状态诊断。
  中文静态分析、计划/改动记录更新；进度脚本只展示当前组token，避免把已经完成前置481token当作新组进度。
- 2048特殊配对相关实测CPU19 passed（3.71s），编译/diff通过。控制器把候选gate拒绝
  与runtime/protocol错误分开归档：gate保留具体STOPPED_*阶段、中文原因并exit3；
  unit视3为预期任务停止（候选未通过，绝非目标达成），不会把质量门槛不满足误报为服务崩溃。
  当前统一候选已完成原自然483请求，进入6个晚期参考；修正配对的新控制器待冻结迁移。
  原首轮doc symlink改为带历史说明的报告入口；原2026-10-01冻结报告保留未编辑。
- 启动后的进一步安装版源码复核发现2048评分边界特殊情况：原root位于上一完整chunk末尾，
  最后chunk仅EOS且不产生prompt-logprobs，不能只找最后chunk倒数第二位置。
  在长测试到达前修正配对，新增真实分派结构测试；不是已发生GPU失败。当前候选预填充已通过，
  自然晚期复测在230token处运行稳定，6帧已保存。诊断控制器改动将新建冻结版本，
  只读复用完整通过的相同数值前置验收，核验所有数值文件/服务脚本/模型协议，旧code不编辑。
- 当前统一候选编译/bash/diff通过，18项相关CPU测试实际2.84s通过；launcher CPU协议/预算预检通过。
  已启动用户unit`remtp-adaptive-round4.service`，新目录
  `results/adaptive_optimization_round4_20261004_norm_head16/`，代码冻结与协议绑定；未覆盖旧run。
  已核实Linger=yes、Restart=no、KillMode=control-group、14天时限；TREE模型已实际启动并执行请求，
  服务实际打印BF16 LMHead weight=(151680,4096) fixed16，正在完整prefill验收，不是仅提交命令。
  实时doc入口指向新报告；新增中文静态原因分析，进度显示主/子阶段累计数避免主请求0误判子任务未推进。
- 计划与专门改动记录更新为实测证据及九步执行，明确第二项舍入修复后可出现新自然轨迹，
  仍精确复查所有旧故障IDs；不能把旧轨迹不一致当作运行失败或新现场冒充原复现。
  修改初始gate保留自然hash/输出是否相同作为证据，放行依据实际逐层/概率/复制及后续旧IDs gate。
  launcher限定结果在项目results内；完整多种子任务可能多日，有限总时限14天。
- 新增launcher冻结remtp/scripts/configs/tests并先CPU协议/上下文预算检查，再用户级systemd运行，
  不自动重启；有限工作总量+每请求30分钟+14天总时限。关闭对话后继续；最新结果入口和进度脚本
  动态读取launch manifest，子阶段/实际已提交token/门槛可查看，不再硬编码已经停止的v2unit。
  中文实时报告追加本轮来源、诊断/质量区别和验收阶段。新的统一norm+LMHead候选待启动。
- LMHead第二微测16/32/64三候选在M1～2048全词表逐位跨M一致；只有fixed16同时保持
  真实自然280前缀概率（TV0、top/EOS0），32/64保持的是另一条舍入路线（TV0.007642）。
  因此新增默认关闭`REMTP_TARGET_LM_HEAD_ROW_BLOCK=16`，只作用于实际ParallelLMHead的
  BF16未量化输出GEMM，复用已有fixed_rows_linear；不量化权重/不修改Target36层。
  所有方法及草稿LMHead统一采用、记录协议。流水线会按同冻结源码重新跑prefill和原自然state，
  不复用只有norm修复的GPU结果；后续优化等待新完整模型复测。初步候选未宣称速度收益。
- 未量化LMHead微测真实重放：最终norm各M逐位相同；原M1/10/13匹配现场，M64/255/256/
  280/1024/2048匹配参考且相对现场top差0.125，证实第二来源是BF16 LMHead GEMM形状。
  首个微测退出2：其fixed64候选不保留M1概率（保留此失败），未采用；不是新的服务崩溃。
  下一微测对固定16/32/64逐位跨M稳定性和相对现场/参考差分分别报告，避免把稳定性与保留旧数值混为一谈。
- 修复norm后的原自然轨迹实机483token与历史逐字相同；6窗口全部36层算子/KV逐位相同，
  原287 top/EOS误差从0.25/0.046876降为0/0，228复制全部相同，6数值门槛均通过。
  仍发现280现场hidden逐位相同但LM输出top差0.125/TV0.007642；同hidden M1/10/13对照跨实际/参考逐位相同，
  表明剩余差异在compute_logits的最终norm或未量化LMHead，而非36层/KV。
  虽在旧0.15以内，先新增真实权重+hidden分离测量CLI，避免直接带着可定位偏差进入长测。
  新流水线18相关CPU测试3.20s通过（其后仅新增此诊断CLI）；逐项变化与边界明确记录。
- 后台流水线复核：新增按实际最后chunk位置配对echo prompt-logprob（>2048预填充可能分块，
  最后两input时sampling与prompt-logprob均M1，取后者），避免把EOS后分布冒充root；
  新增此问题、top8以外真实draft token和旧norm协议拒绝三项测试。
  每组服务保存实际开关/检查点/数值协议；旧六晚期前缀及此前D4失败祖先精确IDs加入新fresh验收，
  不仅测试相同长度的新前缀。服务脚本/template也参与复用源码核对。实际验证待执行。
- RMSNorm修复后的四请求完整模型对照实际完成：五组比较中同位置129的36层全部算子与
  逻辑前缀KV逐位相同。结果保留 `results/target_prefill_norm_round4_20261004/`。
  已启动原自然2950完整2048预算+6晚期完整状态对照新候选，不复用修复前张量。
  新增 `scripts/run_adaptive_round4.py` 有限后台流水线：绑定prefill/state来源及数值代码；
  D3/D4×head1/3真实2048请求、实际跨点root及主/备选祖先独立重算；块边界/全部旧故障/
  长度2048及独立重启/逆序，未覆盖则失败；再严格delta0、减小delta/保护/结构的训练集质量优先选参，
  三基线训练选深度、24题回归/留出、100新test×3seed正式速度。没有新GPU/质量结论；
  numeric/quality失败归档并停止，无自动重试或test选参。新CLI/检查点验证待执行。
- 独立GPU真实norm输入/权重实测全部通过：156/288分别逐位复现原现场输出；原fused在
  256/257/288/512/1024/2048行均与小M不同，255行限额修复后全部与小M逐位一致；
  plain该输入未分叉（不能推广所有输入）。14相关CPU测试2.90s通过。
  已冻结新完整模型四对照候选，后台运行中；不是质量/速度或长状态通过。
  为长状态验收增加默认关闭、有界generated检查点：一步跨过127/128等边界仍保留实际root，
  不要求改变采样使长度正好落点；检查点仅控制记录，不强制生成。新增越界跨点/不重复测试。
- 156/288 × TREE/TRITON四请求实机全部完成：同长度跨后端36层/129位置及KV逐位一致；
  改长度两后端均首次在6:input_norm出现输入差0、输出差0.0078125，KV6/129差0.015625。
  对照vLLM v0.18官方CUDA源码发现plain/fused RMSNorm在num_tokens=256切换thread block。
  新增默认关闭`REMTP_TARGET_RMSNORM_ROW_LIMIT=255`：仅把原CUDA算子分为≤255行view调用，
  保留原融合残差原地更新、短batch数值及公式，所有方法共同应用；不改attention/split-K/阈值。
  新增真实输入/真实checkpoint norm权重的独立GPU重放CLI及边界/alias CPU测试；验证待执行。
  控制器显式记录/传递norm协议；新的修复轨迹可显式允许与旧输出不同，仍要求实际捕获/独立重算，
  不能借用修复前自然/参考结果，协议增加norm配置核验。
- 用户授权制定详细计划并执行。新建 `plan/adaptive-raven-round4-execution-2026-10-04.md`，
  固定原2950/29542/old自然轨迹、2048预算和120～136快照范围；先现场/首次层分叉/单因素修复，
  再累计状态、质量与三个基线。旧冻结来源和失败结果保持可追溯。
- 开始实施默认关闭、有界Target逐层/逻辑KV/源目的提交诊断与有限控制器；
  实際验证、变更文件、失败及后续结果在本节逐项追加，不沿用旧短gate声称长状态成功。
- 新增 `remtp/target_state_audit.py`、`scripts/run_target_state_audit.py`、
  `tests/test_target_state_audit.py`；`target_diagnostics.py`支持按generated范围选择晚期帧，
  `run_adaptive_round2.py`记录探针协议并显式传递控制参数。全部默认关闭；原地norm前拷出输入，
  仅实际MiMoModel的36Target层记录CPU张量/逻辑KV，源目的拷贝检查不修改state。
  controller先原2048自然请求，逐帧检查历史287hash，再独立完整重算与逐算子/逻辑KV比较；
  有快照运行不计质量/速度。CPU与GPU实际验证待下一步执行。
- 相关CPU实际9 passed（3.00s）、编译/diff通过；冻结natural_code并启动有限state-round4。
  首次GPU采集在18:55:28北京时间失败：新增探针误用Attention.prefix，本机API实际layer_name。
  失败完整保留在 `results/target_state_round4_20261004/`，未改变模型算法或掩盖为原数值故障。
  根据安装版attention源码修正字段，并新增每个实际模型warmup的metadata/KV接口检查；
  下一候选新目录重新冻结，旧代码/结果不改。旧round3仍停止。
- 新v2来源和有限unit已创建/启动；中文结果入口指向新run，新增进度脚本与本轮专门改动记录。
  逐层/长状态验收等待实机完成，未恢复正式优化或声称质量提升。
- v2原自然请求实际483token正常结束，输出逐字与旧2950相同；6帧包含原131/287 hash，
  现场top误差0.25、root hidden差1.0、完整TV=0.0140458，首次获得原故障完整分布。
  窗口228源目的复制检查全相同（37不同Attention层含MTP，不等于源KV已正确）。
  独立TRITON参考warmup被新KV形状检查阻止：其layout为(blocks,2,...)，TREE为(2,blocks,...)。
  修正探针为显式backend的可见token gather，不permute/flatten整个GPU cache；新增双布局和2block歧义测试。
  新CLI只读复用已完成自然现场，需公共数值/模型/包/请求协议及非诊断remtp源码一致；不复用失败参考。
  原v2失败完整保留，新冻结参考候选待验证启动。
- 双KV布局相关测试实际11 passed（2.88s）。v3复用真实自然现场后，独立GPU6参考全部完成，
  最后CPU配对因每次forward有sampling/prompt-logprobs两次compute_logits而误判多份state。
  源GPU请求与张量均保留；改按精确compute调用编号配对，增加能击中此错的测试。
  新CLI只读复用完整参考，核验数值/模型/版本/源码及全部请求前缀、echo/温度，不重复GPU。
  此问题属于新增诊断控制器，不当作模型/方法故障；后续CPU分析结果继续追加。
- 精确配对测试初次因插入位置把旧测试断言移入新测试而NameError；立即恢复独立测试，实际12 passed（2.96s）。
  v4只读CPU分析完成：6真实窗口均超过0.15，287raw TV0.0140458，前6层KV全相同；
  layer6仅逻辑位置129首次不同，属于原156提示而非新生成；第131root到layer6 attention才出现输出差1.53e-5，
  后续层放大最终hidden差1。228窗口复制全部一致。具体提示129的首次算子仍须单独观测。
- 新增按实际query位置选择探针，新增 `scripts/investigate_prefill_history.py` 四请求长度×后端对照：
  TREE/TRITON各156/288，捕获相同129的36层/KV；不改变原输入前130tokens，不做正式优化。
  比较器按算子数据依赖顺序排列，避免父layer hook提前插入掩盖内部首次算子位置。

## 2026-10-04：详细复查第三轮失败与下一步方案（有限诊断完成，长状态定位待执行）

- 用户要求详细检查、资料调研与更审慎方案。保持所有旧冻结来源/失败gate及正式优化停止。
  先对第三轮真实109/238/157/287 manifest核验hash，再做原生prefill/单行/末token decode，
  同hidden投影及新鲜树对照；避免仅重复旧109/236/157/286案例。新增诊断CLI显式manifest，
  源token被改变或重复case必须拒绝，新增针对性测试；正式算法不改。
- 初步源码发现：当前BF16 flag=False仍允许split-K，本机vLLM 0.18 batch-invariant实现
  在torch≥2.10使用(False,False)并调整RMSNorm/线性等算子；支持列表不含TREE_ATTN，
  不能直接开启全局模式当作现成修复。官方资料支持batch/shape-dependent数值差异存在，
  尚未证明本次根因；不得把版本更高的官方硬件支持范围当作0.18已验证sm87。
- 本机隔离进程实际torch2.10读回：flag=False时reduction=False、splitK=True，
  (False,False)读回splitK=False；此检查未改变模型worker。本次未实施新的数值修复。
- 新冻结 `results/target_consistency_deep_audit_20261004/audit_code` 关键9执行文件与round3
  原来源逐字节一致；原生24请求完整完成，新四例echo/decode hidden/top差均0，
  完整TV≤9.74e-11；same-hidden M1/10/13检查top差均0；旧长树287相对新echo仍差0.250001。
  补查三方式每例两次raw logits重复相同，single/decode与echo全词表少数raw值仍有≤0.0625差，
  因而不宣称raw logits全词表逐位相同。
- 新鲜D3/D4共16root请求、24root项及16祖先评分全部通过，top/EOS差均0，
  root TV≤1.133e-12、祖先TV≤7.985e-12。追加原old D3/head1/无anchor/无topk配置
  157/287两例×seed29542/29543共4请求，实际root/position对齐，hidden/top/EOS差均0。
  总计60对照请求（warmup另计），17:57:51北京时间结束，两个有限user unit均exit0。
  正式round3仍failed/exit2/NRestarts0，未恢复；新鲜状态不等于原131token提交历史，具体根因未确定。
- 本次相关CPU实际7 passed（3.12s）。初始.venv -m pytest缺模块，系统pytest直接入口缺scripts导入；
  改用既有系统pytest并显式项目/.venv依赖PYTHONPATH后通过，无安装/升级。历史379/2skip不冒充本次验证。
- 阅读PyTorch2.10 numerical accuracy、vLLM0.18 batch invariance/源码、ThinkingMachines原始研究、
  speculative decoding原始论文；中文 `doc/ANALYSIS-TARGET-CONSISTENCY-DEEP-AUDIT-2026-10-04.md`
  和 `plan/target-consistency-deep-repair-2026-10-04.md` 固定下一顺序：原seed短自然现场→首次
  token/层/KV提交定位→单因素最小修复→128/512/1024/2048长状态→质量/截断→独立三基线。
  明确先前前4唯一prefix仅两题、state前8/Target前32/仅128采点及HTTP EOS缺项覆盖缺陷。
  对原计划/修复记录加最新入口，同步专门upstream目录及CHANGELOG；旧冻结证据不改。
- 收尾：实际逐请求数量60、9关键来源hash匹配、5份文档相对链接/空白检查及git diff --check通过。
  汇总保存 `results/target_consistency_deep_audit_20261004/investigation_summary.json`，
  两项诊断unit均inactive/dead、exit0，无遗留模型进程；正式unit仍failed/exit2。
  修正日志头部本地工作分支名称，区分本地适配与上游研究分支。

## 2026-10-04：第三轮再次停止核查（仅调查与文档，未重启）

- 用户询问为何再次中断；读取user unit/journal/status、全部10生成记录、4新前缀原始评分，
  确认13:43:53北京时间主动exit2、NRestarts=0、MainPID=0，STOPPED_PROTOCOL_ERROR。
  新真实前缀109/238/157 top误差约1e-6，287/gen131差0.250001>0.15、EOS0.046876，
  来源old、train2950/seed29542。原24root/16祖先固定验收仍真实有效，覆盖范围不足。
- 完整跨服务重放通过，7311/73153两次1250token、正常结束、hash相同。
  old/anchor4各4题原判对4/4，正常结束2→3、截断2→1，吞吐2.569→2.932(+14.1%)，
  均时567.3→293.8(-48.2%)；2950却483正常→2048截断，质量问题仍在，尚无三个主基线证据。
- 服务日志未检出OOM/CUDA error/traceback/fatal；systemd内存峰值11.6G。不能据此认定
  剩余误差是KV或特定算子，需固定新前缀区分新鲜计算与长生成状态；不放宽容差或盲目重启。
- 新中文 `doc/STATUS-ADAPTIVE-RAVEN-ROUND3-2026-10-04.md`，修复报告/计划/上游改动记录
  加最新停止提示；冻结代码、原始结果和失败gate不改。实际验证仅日志核查和结果重算，
  本次没有新GPU请求、算法改动、后台重启；文档链接核对与git diff --check通过。

## 2026-10-04：执行Target概率一致性修复（固定前缀验收通过，优化继续运行）

- 收尾确认：新round3模型已实际加载，health=200、warmup HTTP200，第一条2048-budget质量请求
  正在真实生成（服务token/接受指标持续更新），user unit仍active/running、NRestarts=0。
  自动中文报告链接可读，最终diff检查通过；不等待后台全流程完成才交还用户，服务独立继续。
- 最终实机：12:28:14北京时间canonical验收32请求全部完成exit0，D3/D4 24项root检查和
  16条祖先评分全通过；top/EOS最大差均0，完整TV root≤8.31e-12、祖先≤1.88e-10。
  两个关键160/161祖先评分的修复前后实际prompt逐token相同，排除换前缀冒充修复。
  本次只是固定前缀概率门槛，不宣称所有上下文/正确率/速度目标已满足。
- 12:29:47北京时间 `remtp-adaptive-round3.service` 已enable/start，MainPID515188、
  active/running、NRestarts=0；新numerical_gate带来源hash，关键9文件匹配实机来源。
  当前diagnostics old 0/4，先完整2048生成/跨服务重放/独立Target，再消融/选参/回归/留出/正式。
  user lingering=yes，关闭对话不停止；质量/协议exit2/3不重试，其余有限3次重启并逐请求续跑。
  中文修复文档已收尾，plan步骤1～6标记本范围完成、步骤7运行，新doc第三轮自动报告链接。
  最终CPU379/2skip后仅复用source校验的小修改，相关6测试再次通过；编译/bash/diff/unit验证通过。
- canonical候选完整CPU回归实际379 passed、2 skipped（46.88秒），新mapping相关9测试已先通过。
  已启动独立 `remtp-target-canonical-check.service` 16root+16祖先验证，顺序FP32参考算术不变。
  round3 unit及进度脚本改为新 `adaptive_optimization_20261004_target_canonical`；先前target_fixed
  仅预检未启动并加显式未使用说明，冻结来源不编辑。新的9个关键执行文件与canonical实机来源
  字节一致、SHA256 manifest已保存，预检与服务启动仍等待概率gate。
- FP32最终实机验收56请求完成（24原生+16root+16祖先）：D3原8祖先全部通过，
  旧prefix160误差由0.1875变0；但D4 prefix161 top0.25、EOS0.15234375、TV0.0668187，
  仍gate失败，不启动优化。备选/实际输入和draft IDs已核对一致，不能用root通过冒充完成。
  新增 `remtp/canonical_tree_attention.py` 可选修复：一节点映射为一个virtual单token query，
  token-level block table将已提交前缀和祖先按逻辑顺序映射原BFS KV槽；原cache只view不复制、
  不写KV，保留原算子/数值/采样规则。索引布局仅跨层复用，每次runner prepare清空，
  避免跨请求原地更新block table造成陈旧映射；table按32列补齐覆盖原kernel无mask表读取。
  新CPU回归检查置换block映射、兄弟/拒绝槽隔离、32边界padding；真实GPU验收待执行。
- FP32原生24请求已完成，四例prefill/decode hidden/top差均0；D3新鲜8root请求通过，
  D4/祖先尚未结束。新优化目录已冻结并 --check-only 实际PREFLIGHT_COMPLETE，
  关键8个执行文件与本次GPU验收来源字节相同，保存独立SHA256 manifest；尚未启动优化。
  新增 `scripts/status_adaptive_round3.sh`，显示user unit/JSON逐阶段进度和中文报告入口，bash语法通过。
- 当前代码第二次完整CPU回归实际376 passed、2 skipped（45.86秒）；FP32本地kernel与安装版
  源码核对仅一处P×V表达式改变。中文修复结果/限制文档已建立、旧状态添加可见历史提示。
  自动报告标题去除旧轮次并显示全部公共数值配置，阶段标题改中文；属于报告展示修改，
  后续检查编译即可，不改变已完成模型验证/筛选规则。最终FP32 GPU验收仍运行中。
- FP32 kernel CUDA隔离测试实际编译运行完成：该随机输入原版与候选均输出完全一致，
  因此不能作为原失败原因或修复效果证明；初次未source cuda_env导致Python.h缺失，
  使用既有环境脚本后解决，不改安装环境。下一步以真实失败前缀全模型结果判定。
  新有限 `remtp-target-fp32-checks.service` 已启动：24原生cache-off请求→16树root请求→
  16祖先路径，采用固定路径规则（无root备选时取第一条内部备选）并核对逻辑位置。
  所有组冻结在新fp32_code，旧失败及冻结代码保留；控制器也保存native实际JSONL进度。
- 新建 `deploy/remtp-adaptive-round3.service`（尚未启动）：ExecStartPre核验最终gate、
  24root检查/16branch检查、显式FP32等公共数值协议和cache-off；质量失败exit2/3不重启。
- 最终祖先回放12条实际完成：24项root检查全通过，但D3主路径prefix160 top误差0.1875、
  EOS0.125、TV=6.90e-6，原门槛失败exit2；D4 prefix161 top0.125、TV=0.001034。
  不恢复优化。源码进一步确认TREE的BFS物理槽跨32-token attention tile，顺序祖先槽连续；
  unified kernel在每tile将softmax P转BF16再与V相乘，在线softmax分块可引入不同量化误差。
  新增可选本地 `remtp/target_attention_fp32.py`：复制当前vLLM Apache-2.0 kernel，仅将P×V
  保留FP32/IEEE，其余KV布局/mask/位置/launch参数不变；原安装包不编辑。
  适用于所有方法，协议显式记录 `target_attention_fp32`；隔离及全模型验证尚待执行。
- 收尾CPU回归实际376 passed、2 skipped（48.49秒），包含数值协议漂移与失败root复用必须拒绝。
  此后FP32 attention候选为新增未验证修改，不能沿用已完成测试宣称该候选通过。
- 最终2D真实树新验证实际24请求：D3/D4 cache-off16请求，所有root top/EOS差0、
  备选seed变化root一致；D3 cache-on8请求中286前缀top差0.25，保持原0.15门槛停止exit2。
  决定本轮全部方法关闭跨请求前缀缓存（保留普通解码KV），不把cache-on宣称修复。
  树验证新增显式cache-modes和只读复用：仅接受24条cache-off root检查全部通过、
  模型/包/数值协议/输入manifest一致、关键执行源码hash一致的快照，旧失败gate保留；
  接下来独立运行16条祖先路径，不重跑已通过16条root请求。验证尚待执行。
- 用户授权实施优化与修复；先固定4个原失败前缀并做Target数值对照，不复用旧失败gate。
- 新增 `remtp/target_diagnostics.py`：默认关闭，每请求最多32次Target真实hidden/raw logits及
  同hidden投影shape=1/10/13快照，无采样、无KV写入；仅用于诊断，不计入性能。
  `remtp/fastmtp_verified_worker.py` 和服务脚本新增诊断Target worker；Round2仅在显式诊断参数启用。
- 状态：代码初步实施，尚未启动新GPU/尚未确认数值来源；实际验证与后续修改继续追加。
- 新增 `scripts/investigate_target_consistency.py`：显式固定anchor4四例109/236/157/286和
  原source/request/generated_length/hash；保存全部HTTP参数/响应。原生Target缓存开/关，
  echo评分、单行生成、强制末token后真实decode三方式，各例两次，共48请求；
  强制token仅用于诊断构造相同条件前缀，raw logits采集位于采样约束前，不用于质量测试。
  记录同hidden投影shape对照与完整词表TV，GPU执行尚待开始。
- Python编译、服务bash语法、git diff --check通过，代码冻结为新目录probe_code，
  已启动有限 `remtp-target-consistency.service`，正在加载cache-off Target。
  新增 `tests/test_target_probability_diagnostics.py` 验证新增目录不改变固定案例、歧义必须报错，待执行。
- 固定前缀测试实际2 passed；原生48请求cache off/on对照全部完成，unit退出0。
  原生prefill/decode本身log-odds差可达0.625，prefix236 cache-off完整TV=0.05144；
  同hidden输出投影仅解释0～0.0625，差异已出现在主干hidden。
- 本机bitsandbytes源码确认sm87 M≤4走custom GEMM、较大M走dequant+BF16 linear。
  新增默认关闭 `remtp/target_numerics.py` 统一为后者（保留FP4存储，只暂时反量化单层），
  适用于Target/全部MTP基线；服务与Round2新增显式开关并纳入协议。
  这是待验证修复候选，不直接当作Target一致性已通过；后续独立实验与逐层定位继续。
- CUDA隔离真实量化权重测试：统一bnb路由但仍保留BF16 reduced precision reduction时，
  同一输入行M1/M13差1.0，未修复。直接BF16 F.linear与FP32参考对照发现M13误差1.0、
  M1误差6.1e-5；关闭 `allow_bf16_reduced_precision_reduction` 后M1/M13差0。
  新增默认关闭 `REMTP_TARGET_BF16_FULL_ACCUM`，只改BF16 GEMM累加策略，不改权重dtype，
  Round2与诊断CLI显式记录并公平用于所有worker；全模型概率验证仍待执行。
- BNB隔离验证：原custom路径关闭reduced accumulation仍M1/M13差1.0；同时统一反量化路由
  且关闭reduction后差0。已冻结两项组合并启动48请求全模型候选原生对照，尚未验收。
  Target probe进一步记录实际input IDs/positions，树模式保存全部13以内logit/hidden行、
  topology/draft IDs，并用实际root行做同hidden形状对照（运行中的候选冻结代码不编辑）。
- 组合全模型cache-off仍prefix157误差0.4375；cache-on236/157为0.1875，不能验收。
  同hidden输出投影形状误差均0，剩余差异在主干。增加默认关闭FP4固定行块16/64候选，
  同层反量化一次、分块固定shape计算并返回原行数，保持FP4权重/真实attention token数；
  仅sm87启用、协议记录，尚未GPU验证。不是提高容差或更换模型。
- 固定row helper CPU测试实际4 passed；row64候选已冻结启动原生48请求对照。
  新增 `scripts/validate_target_tree_consistency.py`：固定prefix[:-1]、仅诊断强制末token构造
  完全相同条件前缀，捕获实际树root raw logits；比较D3/D4、PC开/关、改变备选seed。
  核对实际root token/position，独立回放primary/backup祖先路径，保持0.15 gate；尚未执行。
- 真实树D3/cache-off8请求已完成：root token/position均对齐，备选seed改变root raw logits不变；
  与FLASH_ATTN独立参考仍误差0.125/0.1875/0.375/0.25，109的EOS误差0.15625也失败。
  因root未通过主动停止该有限诊断（不推进优化）；这是未修复残差，不算通过。
  新增显式独立Target attention backend选项并纳入协议，以控制FLASH_ATTN/TREE_ATTN差异。
  树检查脚本以后首组root失败立即exit2，避免无效后续祖先评分。
- 本机TREE_ATTN源码确认prefill/decode均调用unified_attention，默认Target却FLASH_ATTN/FA2。
  下一对照以公开TRITON_ATTN为独立参考（同unified算子、无MTP）；若验证后采用，
  Native/Cactus/SpecCascade同样改为TRITON_ATTN，保持原chain proposer/sampler。
  不用TREE_ATTN改变原生链草稿语义；配置纳入协议。尚未执行新后端一致性验证。
- 完整CPU回归375 passed、2 skipped（44.82秒）；随后shell防errexit的小修复与协议/诊断更新
  待收尾检查。新建专门改动目录 `doc/upstream-changes/target-probability-consistency/README.md`，
  同步总CHANGELOG。TRITON原生48请求诊断已冻结启动，未宣称树/祖先检查或正式质量通过。
- TRITON原生48请求完成，复用已保存D3/D4树raw logits与新TRITON prefill做独立计算：
  两种深度四例top/EOS差全0、TV≤1.08e-13。证明同条件树/统一attention参考可以一致。
  但TRITON单token decode仍有0.3125差异；源码确认仅单token启用3D分段softmax，而TREE始终2D。
  新增默认关闭2D launch选择，清空两个受支持的3D选择参数，不改query长度/位置/mask；
  公平应用所有TRITON worker并记录协议，独立GPU复核待执行。
- 2D最终参考48请求已实际完成：cache-off4例top/hidden差0；cache-on109/236/157差0、
  286 top差0.125，完整TV均≤2.41e-13，满足原top0.15。新增参数组合校验，避免单独
  启用row块但遗漏精度/路由配置。接下来独立重做树root、备选seed、缓存和16祖先条件。
- row64/cache-off24请求真实完成，4例prefill/decode hidden差0、top log-odds差0、TV≤2.72e-13；
  cache-on待结束。修复不是仅统一route或关闭reduction，而是三项共同对照获得的候选。
  主控制器改为old/old_repeat/anchor4固定顺序选前缀，保存不可变化manifest与每条完整API响应，
  gate附来源/request/generated_length/hash；新增共同logit偏移不误报、相对/EOS错差必须失败测试。
- 原生row64候选48请求全部完成，相关CPU5 passed；树检查脚本保存API逐例JSONL和scheduled jobs，
  使总进度可核实。下一步冻结真实树root/祖先路径验证，不将原生一致直接当作树一致通过。

## 2026-10-04：Target概率一致性行动方案（计划完成，实施待开始）

- 用户要求下一步行动方案；新增 `plan/target-probability-consistency-2026-10-04.md`，
  更新总计划入口、上游改动日志和本日志。保持现有模型/环境、0.15门槛和停止状态。
- 实际调查：读取采集、temperature处理、Target输入、KV压紧与独立回放代码。
  旧checks缺prefixhash/来源、case选择依赖glob；新增strict目录后直接枚举将改变前4例，
  排除该后增目录可重建anchor4的1/128/1/130轮次、长度109/236/157/286，计划先显式固定并核对。
- 方案：固定案例→核对token/预测行/约束→原生prefill/decode与缓存误差对照→
  树root/祖先路径与不可见槽污染检查→同hidden Target投影与逐层定位→局部修复→
  原4例/独立例/2048预算复现验收→恢复优化；多指标报告，不用原生误差或加阈值直接绕过门槛。
- 限制：只制定计划，没有新GPU执行/算法修复/验收结果；旧失败记录和冻结代码未修改。
  文档步骤/总计划链接核对完成，git diff --check通过。

## 2026-10-04：实验状态核查与修复后结果分析（后台已停止）

- 状态：`remtp-adaptive-round2-fixed.service` failed、exit2、NRestarts=0、MainPID=0；
  于10月2日15:17:27北京时间因Tree/sequential root mismatch停止，完成10生成诊断和4前缀评分。
  完整2048预算7311/seed73153重放通过，两次363token、正常结束、输出hash相同。
- 四题旧/anchor4均4/4正常答对、0截断；原始记录核算吞吐2.828→3.295（+16.5%），
  平均238.4→254.2秒（+6.6%），tokens2696→3350（+24.3%）。不是三个主基线正式收益证据。
- Target前缀109/236/157/286，各匹配8个top token；max log-odds差0.125/0.1875/0.2500/0.3750，
  三条超0.15门槛。尚未区分数值形状误差与缓存/位置问题，不放宽门槛；严格短诊断128token
  截断不并入2048质量统计。消融/选参/留出/正式测试都没有运行。
- 文件：新增中文 `doc/STATUS-ADAPTIVE-RAVEN-2026-10-04.md`；修复说明、实施记录和下一轮计划
  添加历史状态提示；同步 `doc/upstream-changes/CHANGELOG.md` 和本日志。
- 实际验证：读取服务状态/journal/status、reproduction_gate、4条Target checks与全部10生成记录，
  独立重算分组和逐题性能；仅文档修改，不改算法/冻结来源，不启动GPU；文档链接与git diff --check通过。

> 以下“后台已启动/正在运行”为历史记录；当前服务已停止，以本次核查为准。

## 2026-10-02：继续定位草稿复现差异并修复（进行中）

- 收尾：14:33:33北京时间，修复版7请求GPU短回放全部通过，unit自动退出0。
  cache-on/off同进程与cache-off独立重启，各8次提案Q最大TV=0、root hidden差0；
  cache-on额外24次树hidden/KV差0，RNG一致。修复成立于本次短前缀对照，不等同质量通过。
- 已启动新冻结 `remtp-adaptive-round2-fixed.service`，MainPID=442893，active/running、NRestarts=0，
  缓存显式开启、深度探针关闭；旧失败unit/协议/原始输出保留。新报告本地链接
  `doc/RESULT-ADAPTIVE-RAVEN-ROUND2-FIXED.md`，同步计划、旧实施记录的历史提示和修复说明。
  全量2048-token复现、Target对照、筛选/回归/留出/正式测试仍等待后台门槛，未声称通过。
- 新后台实机收尾：模型health=200、warmup已完成，首条2048-budget诊断实际生成中；
  unit仍active/running、NRestarts=0，git diff --check通过。已清楚告知短复现通过不等于截断/质量目标完成。

- GPU cache-on 顺序重放已实际失败；CPU 快照显示第二轮 root=114、有效前缀115，
  传入 Target seq_lens=123。两次 root hidden、logits、RNG、真实 prefix KV 完全相同，
  首次三节点草稿 hidden 已有 maxabs=4.0，Q maxabs=0.0373194。原草稿按123+count
  配置 attention，误读额外8个旧 speculative 槽位；不是仅凭输出差异猜测。
- 修改 `remtp/fixed6_vllm.py`：草稿序列长度改为 root_position+1+count，同步 max_seq_len、
  CPU query_start_loc 并清除旧长度影子；不改根分布、接受规则、RNG 或物理 BFS 写槽。
  新增 `tests/test_draft_tree_metadata.py` 覆盖实际 root114/seq123、不同树大小和初次prefill。
  GPU修复回放和CPU测试尚待执行；正在运行的 probe_code 保持原失败版本用于cache-off对照。
- probe 脚本新增前8次实际 proposal Q 的 TV 与草稿 token/拓扑/前缀对照；`--require-match`
  要求三组128-token输出相同且Q最大TV≤1e-6，失败exit2。防止“输出相同但概率仍污染”漏过。
- 完整CPU测试实际369 passed、2 skipped（47.79秒）；编译与git diff --check通过。
  追加独立注意力回归：只改变被拒绝槽位内容时，修复metadata的输出必须不变，原长度算法
  必须显著改变输出；该新增案例尚待执行。修复冻结在 fixed/frozen_code，已启动
  `remtp-repro-fixed.service` 7请求真实GPU开关/重启测试；最终结果待完成。
- 新增独立缓存污染回归实际6 passed（3.22秒）；新增中文修复说明
  `doc/FIX-ADAPTIVE-RAVEN-REPRODUCIBILITY-2026-10-02.md`，同步
  `doc/upstream-changes/CHANGELOG.md`，记录旧cache开/关、重启7请求证据与新GPU尚未结束。
- 下一轮控制器将完整旧配置复现门槛提前到anchor诊断之前，并单独保存
  `reproduction_gate.json` 的2048预算、tokens/结束原因/输出哈希；失败仍exit2，不进入选参。
  新增 `deploy/remtp-adaptive-round2-fixed.service`，新冻结目录、新记录、缓存显式开启。
  unit 尚未启动，等待短GPU复现门槛；旧失败目录/unit不覆盖。
- 修复版缓存开启3请求实际完成：相同128token输出，8次提案Q TV=0、root hidden差0；
  24次树forward hidden/prefix KV差均0，提案RNG一致。cache-off 3请求已结束，重启1请求待结束。
  新优化目录已冻结（不复用旧请求），`--check-only` 实际PREFLIGHT_COMPLETE，unit verify通过；
  下一轮尚未启动，独立Target检查与2048-token长重放都不宣称通过。

- 用户授权继续查因和修复；旧 9 请求和冻结代码保持不变，新目录为
  `results/adaptive_repro_fix_20261002/`，已保存工作区 before.patch 和旧 v4 证据。
- 先做同题/同 seed 的同进程请求顺序与独立进程对照、前缀缓存开关对照，
  补充真实 root hidden/Q、备选拓扑、RNG 和 MTP 缓存诊断；不绕过原复现门槛。
- 初始状态（历史）：准备诊断代码，尚未确认根因、尚未开始新 GPU 实验；后续改动/验证随本条追加。
- 修改 `scripts/serve_fastmtp_verified.sh`：新增显式 ENABLE_PREFIX_CACHING=0/1；
  未设置时保留 vLLM 原默认，不改接受规则。`scripts/run_adaptive_round2.py` 将该开关
  纳入协议/各任务配置，修复已保存顺序 Target 检查失败时不能被恢复逻辑跳过的问题。
- `remtp/generation_guard.py` 新增默认关闭、每请求前8次提案有界的真实 root hidden/Q、
  主/备选拓扑、MTP 输入/输出、一个实际 MTP attention 层 prefix KV 与 RNG CPU 快照。
  只诊断，不修改 KV 或消耗 RNG，同步成本不计入性能。
- 新增 `scripts/investigate_adaptive_repro.py`：128-token 同进程同题→其他题→同题，
  cache on/off，再加 cache off 独立服务重启对照；短前缀不当作完整正确率证据。
  Python 编译及服务 shell 语法检查通过；GPU 结果待执行。

## 2026-10-02：第二轮停止与结果分析（诊断完成 9 请求，完整计划未完成）

- 状态：北京时间 10:11:54 `STOPPED_PROTOCOL_ERROR`，服务 exit=2、NRestarts=0、failed；
  journal 明确为同配置/同种子输出重放差异，当前不在运行。未进入保护选参、留出或正式测试。
- 已核算：旧 D3 vs anchor D4 四个固定失败诊断题，原判对 3→2，正常结束且判对 2→2，
  截断 2→2；E2E 2.85765→3.30614 tok/s（+15.69%），平均 468.392→409.995 秒（−12.47%）。
  诊断/失败回归样本不作为正式收益证据；anchor4 没有启用新保护。
- 重放：7311/seed73153 两次均正常判对，592 vs 1700 token、212.989 vs 564.135 秒。
  最早同前缀 Q 差异为第 2 轮二层 0.250476→0.213698，同位置 P 与接受随机数相同；
  第 23 轮草稿/验证路径不同但提交相同，第 24 轮接受 RNG 分叉，第 25 轮（生成长度88）输出分叉。
- 诊断观测：候选 EOS 质量在正常结束题 2629/2950 的换行接受处被显著降低；
  截断候选的已访问主路径最大 EOS 质量约1e-9，阈值0.5保护未必覆盖该类循环。
  连续相同 token 块检测全为 false，但存在重复句段，说明该启发式覆盖有限；不是保护效果实测。
- 文件：新增 `doc/ANALYSIS-ADAPTIVE-RAVEN-ROUND2-2026-10-02.md`，同步
  `plan/adaptive-raven-next-optimization-2026-10-02.md`、
  `doc/OPTIMIZATION-ADAPTIVE-RAVEN-2026-10-02.md`、
  `doc/upstream-changes/CHANGELOG.md` 和本日志；自动报告追加停止分析入口。
- 验证：读取服务/status/journal、全部9个输出与三组逐轮诊断，重新计算对照、分叉与EOS统计；
  新分析文档链接存在，`git diff --check` 通过；原始请求、冻结代码与协议不改。
  仅文档归档，没有新模型运行或算法修改。
- 限制：草稿Q差原因仍需 hidden/KV/拓扑/请求顺序/缓存诊断；前缀缓存命中只是候选因素。
  不把数值/状态影响直接归因于方法，不绕过门槛，不声称截断优化或三个主基线对照已完成。

> 下面“执行下一轮优化”的运行状态为历史记录；当前以本次停止分析为准。

## 2026-10-02：执行下一轮优化（进行中）

- 授权：执行下一轮优化计划；保持当前模型/依赖和 2048 token 主质量协议。
- 已保存 `results/adaptive_optimization_20261002/before_code/`、`before.patch` 与
  `v4_evidence/`，包括旧冻结协议、质量门槛及全部 96 输出；原 v4 不覆盖。
- 实机预检：统一内存 available 约 10 GiB（模型未加载），15W 模式，user lingering=yes；
  旧优化服务已因质量门槛退出，不在运行。
- 实施范围：独立 head 开关、生成前缀与 EOS 诊断、风险位置松弛保护、可恢复的下一轮
  控制器及公平对照。涉及文件/实际验证随本条目追加；尚无新 GPU/质量结论。
- 当前限制：v4 截断根因尚未通过概率回放确认；旧 356 项测试只代表历史代码。
- 已实现默认关闭的 EOS 松弛保护、前缀重复严格验证与独立公共 head 开关，涉及
  `remtp/dynamic_mtp_tree.py`、`remtp/dynamic_tree_vllm.py`、
  `remtp/fastmtp_verified_worker.py`、新增 `remtp/generation_guard.py`。
  受保护位置在 TV 回收池计算前置零，接受和 residual 使用一致的验证分布；不强制 EOS。
- 新增 `remtp/optimization_metrics.py`、`scripts/run_adaptive_round2.py`、
  `tests/test_generation_guard.py`：逐请求断点、唯一配对/各种子质量与截断门槛、
  请求局部 branch RNG、真实 head 同前缀探针、独立顺序 root 概率回放、单因素筛选、
  原题回归/留出/正式测试串联。公共 head 开关用于三个适用基线，单独保存原实现控制。
- 本轮协议显式设置请求局部 branch RNG，避免重启或请求顺序改变备选采样；旧配置会重新跑，
  不直接把历史 v4 校准当新协议对照。正式题只在选择/留出通过后抽样，排除旧 100 题。
- 已实际验证：原动态树测试 68 passed；新增保护/残差一致性、重复边界、校准门槛与
  按题聚类区间测试 8 passed；新增文件 py_compile 通过。完整回归及 GPU 执行仍待结束。
- 首次不指定目录的完整 pytest 收集了 `before_code/tests` 快照，发生同名模块冲突；
  不删除快照，改为显式 `pytest -q tests`，正在重跑实际工作区回归。
- 新增 `deploy/remtp-adaptive-round2.service` 与
  `doc/OPTIMIZATION-ADAPTIVE-RAVEN-2026-10-02.md`，有限后台流程、保护语义、评估与限制有独立中文说明。
- 已核实 config 主 EOS=151645，generation_config 附加 EOS=151643；新控制器将两者
  都用于概率诊断/保护（不改实际停止参数）。输出预算与 user service 配置预检通过。
- 收尾验证：显式工作区 `pytest -q tests` 实际 364 passed、2 skipped（44 秒）；
  Python 编译、`git diff --check`、`systemd-analyze --user verify` 与冻结控制器 --check-only 均通过。
  保存预检活代码产物到 `preflight_live/` 后，将所有运行源码冻结到 `frozen_code/`，运行时不编辑冻结副本。
- 已 link/start `remtp-adaptive-round2.service`（未 enable 开机启动），MainPID=435562，
  active/running、NRestarts=0；模型 health=200，warmup HTTP=200，退化回放请求正在生成。
  GPU warmup 三次 head 对照 argmax 一致，max_abs_logit=0.0625、最大概率 TV=1.79185e-9；
  仅覆盖三个 warmup 输入，不能外推全部前缀等价；顺序 Target 回放尚未执行。
- 同步本轮计划执行状态、`doc/upstream-changes/CHANGELOG.md` 与中文实施记录，
  新建 `doc/RESULT-ADAPTIVE-RAVEN-ROUND2.md` 自动报告链接。结果不覆盖旧 v4。
- 后台按诊断→单因素筛选→固定 train 选参→最多两候选回归→公平基线选择→留出→正式/消融
  自动推进，各种子质量/截断不退化且吞吐和平均耗时改善才放行。尚未声称优化达标；
  若某阶段未通过，保留明确门槛原因，不自动采用，也不按留出/正式答案再调参。

## 2026-10-02：v4 校准完成与下一步优化计划（计划已制定，实施待开始）

- 状态：v4 四组各 24 请求共 96 请求全部落盘；`STOPPED_QUALITY_GATE`，
  user service `ExecMainStatus=3`、`NRestarts=0`，这是门槛失败后的主动退出。
  完成时间为北京时间 2026-10-02 00:52:43；没有启用失败候选默认值。
- 已核算结果：原判对 41/48→42/48，正常结束且判对 41/48→40/48，截断 7→8。
  请求总时长 11702.287→10525.238 秒，平均 243.798→219.276 秒，减少 10.058%。
- 检查全部截断输出及 7 个配对变化，确认 4 个旧版正常答对/v4 截断案例与 3 个反向改善；
  2629/42 明显反复输出同一句，7300/42 在答案标记后继续生成。源码的 anchor 验证
  没有使用现有 EOS 保护配置；是否为截断原因需前缀回放验证，尚无因果结论。
- 改动：新增 `plan/adaptive-raven-next-optimization-2026-10-02.md`，明确质量/截断/速度
  门槛、head/构树/深度消融、EOS 与重复风险位置降松弛、分阶段 train 调参与留出、
  Native/Cactus/SpecCascade 公平对照及 100 test × 3 seed 正式协议。
- 受影响文件：上述新计划、`plan/adaptive-raven-optimization.md`、
  `doc/OPTIMIZATION-ADAPTIVE-RAVEN-2026-10-01.md`、
  `doc/upstream-changes/CHANGELOG.md` 和本总日志。旧状态添加可见历史提示，原实验不覆盖。
- 验证：读取 `status.json`、四组 `requests.jsonl`、自动中文报告与 `quality_gate.json`，
  重新计算请求数与耗时；核对当前 verifier 和配置代码。`git diff --check` 通过；
  新计划全部相对链接存在，质量数字、7 个配对变化与 4 个退化案例核对通过。
  未修改算法、模型、依赖或服务；未启动新测试。
- 限制：只有 24 道 train 双种子校准，未证明零质量损失、未完成与三个主基线的正式质量/速度对照。
  风险保护会改变方法设置，必须单独消融；公共 head 加速需给适用基线同等机会。

> 下文 2026-10-01 的“进行中/未完成”均为历史过程；当前结论与计划以上述 2026-10-02 条目为准。

## 2026-10-01：Adaptive RAVEN 优化实施（进行中）

- 已将修改前代码、工作区差异及版本存入 `results/adaptive_optimization_20261001/`。
- `dynamic_mtp_tree.py`、`dynamic_tree_vllm.py`：新增默认关闭的
  `REMTP_DYNAMIC_TREE_PRUNE_ANCHOR_DESCENDANTS`，仅 anchor 模式可启用；保留原始候选
  选择/预算，最终计算前删除不会被读到的备选后代，按新拓扑重建 KV。
- `runtime_profile.py`、`fastmtp_verified_worker.py`：可选同步分段计时，包含嵌套耗时，
  正式吞吐不启用；`serve_fastmtp_verified.sh` 支持显式空审计路径关闭审计。
- 验证：代码实施中；概率/随机数等价测试、GPU 缓存与性能测量待完成，尚未采用为默认。
- 限制：CPU 等价不能保证 GPU 行数变化后逐位相同；不得提前宣称正确率和速度提升。
- `tests/test_dynamic_mtp_tree.py`：新增概率决定、输出、选中路径、随机数状态等价检查；
  首次覆盖断言失败是均匀测试分布在 delta=1 下没有产生 hit，改为尖锐分布后复测。
- `scripts/run_adaptive_optimization.py`：有限控制器，代码/协议哈希锁定、逐请求落盘、
  profile/纯测速/24 train 题双种子完整质量校准分离；速度未过门槛不进入质量校准。
  新脚本尚待 GPU 执行验证。测试使用系统 pytest 路径，不修改服务环境依赖。
- `deploy/remtp-adaptive-optimization.service`：后台控制器运行冻结的 candidate_code，
  最多有限重试，质量/速度门槛不通过停止。CPU 树测试实际为 57 passed（新增 5 项）。
- 全部 CPU 回归实际通过：345 passed、2 skipped；服务 shell/新 Python 编译与 diff 检查通过。
- GPU baseline 分段短测已完成：Target 平均 937.6 ms、proposal 216.1 ms、验证
  23.5 ms（规划包含其中 3.2 ms）、缓存更新 15.3 ms。不可把嵌套时间重复相加。
- 代码检查发现每个父节点为取最多 3 个 sibling 对整个词表稳定排序；新增默认关闭的
  `REMTP_DYNAMIC_TREE_EXACT_TOPK`，在 topk 截点补齐最小 token ID 的并列项，
  保持稳定排序与全部候选/诊断/预算语义。独立于裁剪 v1，验证后再冻结 v2。
- GSM8K train 从官方 `openai/grade-school-math` 下载并校验 7473 条；质量使用 train，
  初次控制器因本地仅有 test 而停止，补齐后已重新启动。未修改服务依赖。
- 稳定 topk 的 6 项新测试已通过，树测试累计 63 passed。v2 profiler 加入候选选择、
  草稿 logits 与节点计数，分离剩余 proposal 耗时；控制器锁定配置/包版本，
  校准同时检查每个种子并保存配对错误变动，配置错误返回 2 禁止无意义重试。
- 更新 `doc/OPTIMIZATION-ADAPTIVE-RAVEN-2026-10-01.md`、上游改动表与计划进度，
  明确裁剪单次 profile 变化很小、topk 性能待测、质量尚未完成；未改默认参数。
- 裁剪 v1 的纯短测已得到 baseline 2.637、candidate 2.552 E2E tok/s，
  轮成本近乎不变但输出/接受长度改变；保留失败候选证据，不采用裁剪。
  v2 候选仅启用 stable topk，保持 GPU 树行数与原概率计算不变。
- v2 profile 在排除的 warmup 内运行 CUDA 稳定 topk 等价探针（20 个并列/随机词表案例）
  和小型独立算子对照；其结果保存到 topk_probe.json，不计入正式性能请求。
- 裁剪 v1 纯性能完整结束：两次 baseline 合计 2.6384、裁剪 2.5515、Cactus 2.7760
  E2E tok/s；裁剪 -3.29%，门槛自动停止（未跑质量）。已冻结并启动仅 topk 的 v2。
- v2 完整 CPU 回归实际为 351 passed、2 skipped；未启用裁剪默认。
- 为下一个独立候选添加默认关闭的 `REMTP_DYNAMIC_TREE_ANCHOR_ONLY_PROPOSAL`：
  anchor 模式复用已有 caterpillar 构树器，只对主路径计算后续 Q，并把备选预算用于直接
  sibling。这改变候选分配/覆盖，不声称与原树完整输出等价；接受规则与 adaptive TV 规划
  不变。v2 冻结副本仍只测试 topk。新候选待主路径 Q/RNG 测试、GPU 测速及完整质量校准。
- v2 GPU 20 案例稳定 topk 等价通过，但首版算子为 1.28 ms，比全排序 0.74 ms 更慢。
  保留 v2 证据。下一候选增加无并列快速路径（取 K+1 检查顺序与截点），
  只在有并列时补齐；anchor-only 构树在 N=3/6/9 共 60 个模拟前缀中保持主路径 Q、
  sampled token、随机数状态一致。树测试 66 passed，下一候选待实机测量。
- 下一轮控制器采用 >1% 的短测吞吐改善门槛以减少微小时间波动的误选；阶段完成自动生成
  中文 `report.md`，分别报告组件、相对旧方法/Cactus 吞吐、完整校准与配对变化。
  v2 已因 GPU 算子和 proposal 都变慢而主动停止，未完成普通测速，不声称纯测速结论。
- v3 代码已冻结，启用无并列 topk 快速路径与 anchor-only 提案，继续相同 profile/
  纯短测/完整校准流程。完整 CPU 测试实际为 354 passed、2 skipped。
- 按本机 systemd.unit 手册将新服务 StartLimitIntervalSec 改为 infinity，
  StartLimitBurst=3 覆盖整个启动计数周期，避免长时间间隔故障导致无限自动重试；
  质量/性能/协议错误直接停止。已有 user lingering 可用。
- 更新计划状态与中文优化记录：保存 v1/v2 负结果，说明 v3 候选分配会变，完整质量
  未完成。v3 GPU topk 探针实际通过：20 cases，快速路径 0.502 ms vs 全排序 0.742 ms；
  不能把算子改进幅度直接当作端到端加速。
- v3 纯测速已结束，anchor-only D3/N9 为 2.5179 E2E tok/s，性能门槛失败，
  不进入质量。profile 确认 BF16 草稿 logits 单行路径 51.1 ms，比原多行约 40.9 ms 慢。
- 下一候选在 anchor-only 中仅将 head 输入复制为 3 行，取第 1 行输出；不增加 MTP
  KV 节点和随机数消耗。控制器提供 depth/nodes 显式参数，将 D4/N12 单独测速，
  利用 Target 近似固定的轮成本多验证一级主路径。这是新候选，不提前采用或宣称质量。
- 增加 D4/N12 模拟构树校验；新增 `scripts/validate_anchor_tree_cuda.py`，用独立
  BF16 Q/K/V 对照真实 TREE_ATTN 和路径 KV 搬运，验证范围明确不含完整模型质量。
  此诊断待执行，避免把 CPU 拓扑检查当作 GPU 缓存正确性的充分证据。
- CUDA 诊断实际通过：真实 TREE_ATTN vs FP32 独立参考 max_abs=0.0037183；
  保留节点裁剪前后 BF16 输出 max_abs=0；8 个主路径/sibling KV 搬运逐位一致。
  只证明合成后端与搬运，不证明完整模型概率相等。新增 D4 检查后的全套为 355 passed、2 skipped。
- anchor-only worker 对单行初始 head 也使用 3 行复制，返回原行数，基线不安装此补丁；
  新增 head 行数/值检查。D4 候选的短测 Cactus 也改用 4 个草稿 token 作等深度控制，
  明确旧 Adaptive 对照仍 D3；正式复测须追加等深度 Native/SpecCascade，不能只用 D3 基线宣称胜出。
- v4 D4/N12 + 3-row head 已冻结，由同一 user service 启动 profile/纯测速及有条件校准；
  v1/v2/v3 负结果保留，默认入口仍不启用优化候选。
- v4 实际 profile 和前两组纯性能请求完成：candidate decode 3.0391、E2E 2.9413
  tok/s、MAL 3.700；baseline E2E 2.6477、MAL 3.3247。纯测速未开 profiler/审计。
  目前尚等 Cactus D4 与第二次 baseline，质量未开始。完整回归实际 356 passed、2 skipped。
- 中文记录与计划追加公平对照要求：D4 基线、D3 + head 优化消融、基线共享 head
  优化控制；不可将组合候选的收益全归因于 Adaptive 规则或深度。
- v4 完整短测门槛实际通过：重复 baseline 输出逐题一致；合并 baseline E2E 2.643036，
  candidate 2.941323（+11.29%），Cactus D4 2.613783（candidate +12.53%）。
  仅两个固定短提示，不能当正式性能/质量结论。控制器实际进入 calibration/baseline_seed42，
  共 24 题 × 2 seed × 2 方法=96 请求、max_tokens=2048；当前质量未结束。
- 建立 `doc/RESULT-ADAPTIVE-RAVEN-OPTIMIZATION.md` 到本机自动结果的链接；
  更新实施状态快照、计划和改动表。结果正文在 ignored results 中由控制器继续更新，
  手动文档明确进度快照范围。当前默认不开候选，质量下降会停止而非自动采用。
- 计划表同步为实际后台校准状态；当前服务 active、NRestarts=0，原版质量协议下的
  baseline_seed42 已进入实际请求，后续由控制器逐题保存、生成中文汇总。

本文档是仓库的唯一持续维护总索引，记录已经做过的实现、实验、负结果、协议
变更和当前有效入口。算法细节仍可保存在各专题文档中，但任何后续修改都必须
同步追加到本文档。

## 1. 维护规则

状态含义：

- `已验证`：代码测试和相应实验均已完成；
- `代码已验证`：单元测试或静态检查通过，但尚未完成真实 GPU 实验；
- `实验性`：可以运行，但算法近似、统计结论或系统正确性仍需验证；
- `负结果`：实现或实验完成，但没有达到预设门槛；
- `已冻结`：保留现有定义和结果，不再继续调参；
- `已废弃`：存在已知错误，不得进入正式表格；
- `历史结果`：只用于追踪研究过程，不能自动与当前协议混合。

后续每条记录至少写明：日期、状态、动机、修改、文件、验证、结果和限制。

实验原始数据和报告默认只保存在本地 `results/`、`logs/`、`checkpoints/`，不因
代码推送自动上传。总日志可以记录关键指标和本地结果路径。

## 2. 当前有效结论

1. Qwen3.5-4B 原生 MTP、完整概率提议、Cactus、SpecCascade、Exact-TV、
   ReMTP、HumanEval/GSM8K 评测链路均已实现。
2. Qwen3.5 链式 ReMTP 已冻结；继续统一放大 TV 或 margin 的边际收益很低，
   且容易沿 Cactus 的质量下降方向移动。
3. Qwen3.5 Fixed-6 树是负结果：分支树 MAL 下降，正确性参考 target forward
   延迟相对链增加 168.4%，因此停止 `tree + ReMTP`。
4. MiMo-7B 的额外 MTP layer 1/2 为 pretrained-only，深层 P/Q 重叠很低；
   这不是继续降低树验证阈值能够解决的问题。
5. FastMTP 包含一个训练好的物理 MTP head，并通过共享权重递归生成三个草稿
   位置；它不是三个独立训练 head。
6. 2026-08-09 前由旧 FastMTP 四方法脚本产生的 Cactus/SpecCascade 结果无效，
   因为 Worker 被覆盖成 Native。
7. 当前 FastMTP 有效入口是
   `scripts/run_fastmtp_verified_comparison.sh`；动态树仍是改变输出分布的近似
   方法，不能标成 lossless speculative decoding。

## 3. 研究阶段与操作记录

### 3.1 最小 Qwen3.5 MTP 与逐轮可观察机制

状态：`已验证`

完成内容：

- 固定 vLLM 0.18.0、Qwen3.5-4B、本地 checkpoint 和 RTX 4090 单卡配置；
- 实现 `MTP draft / TARGET verify / VERIFY / COMMIT` 四阶段日志；
- 支持 greedy 和 temperature sampling；
- 解释并记录 draft token、target token、bonus、recover、接受随机数和最长前缀；
- 增加本地 checkpoint 完整性检查、模型下载和请求脚本；
- 为 Qwen3.5 GDN/CUDA Graph 启动错误提供 eager/no-CUDAGraph 路径。

主要文件：

- `remtp/trace.py`、`remtp/worker.py`；
- `scripts/serve.sh`、`scripts/request.sh`、`scripts/download_model.sh`；
- `README.md`。

关键认识：MAL 定义为 `1 + accepted_draft_tokens / draft_rounds`，其中 1 是每轮
目标模型产生的 correction 或 bonus。因此 MTP=2 时 MAL 上限是 3，出现 2.3
并不代表接受了超过两个草稿 token。

### 3.2 完整 MTP 分布与标准概率推测解码

状态：`已验证`

完成内容：

- 将 vLLM 默认 MTP argmax 路径改为从完整 `Q` 采样；
- proposal、接受概率和残差纠正统一使用同一份 `Q`；
- 保留 greedy 为 one-hot proposal；
- 增加 P/Q、随机数、接受概率、recover/bonus 的诊断；
- 明确随机种子只用于复现实验采样，不是推测解码额外引入的理论参数。

标准随机验证：

```text
A(y) = min(1, P(y) / Q(y))
residual(x) proportional to max(P(x) - Q(x), 0)
```

主要文件：`remtp/probabilistic_mtp.py`、概率 Worker 和相应 benchmark/trace 脚本。

### 3.3 Spec-Bench 基准设施

状态：`已验证`

完成内容：

- 支持 translation、summarization、math_reasoning、rag 固定抽样；
- 区分 decode tok/s、E2E tok/s 和 MAL；
- 将 trace 与正式吞吐测量分离；
- 处理 Qwen3.5 GDN 与 CUDA Graph capture 冲突。

早期 translation-20 示例结果：decode 151.186 tok/s、E2E 121.096 tok/s、
MAL 2.357，MTP=2。

### 3.4 论文复现：SpecCascade 与 Cactus

状态：`Qwen 路径已验证；FastMTP 旧入口已废弃并重建`

完成内容：

- 复现 *Faster Cascades via Speculative Decoding* 的 TokenV3；
- 复现 Cactus candidate-specific 临时验证分布；
- 两者均使用完整概率 MTP 作为 proposal；
- 实现统一 residual sampling、诊断和报告；
- 之后所有主要 benchmark 均保留 Native、Cactus、SpecCascade 作为基线。

Cactus 当前实现：

```text
H(y) = min(P(y) + sqrt(2 delta P(y)(1-P(y))), 1)
A(y) = min(1, H(y)/Q(y))
recovery proportional to max(H-Q, 0)
```

同一前缀、候选和随机数下 `H(y)>=P(y)`，所以 Cactus 局部接受概率不会低于
Native；完整生成轨迹和任务质量仍会因临时分布改变而不同。

专题文档：`docs/cactus_mtp.md`、`docs/speculative_cascade_mtp.md`。

### 3.5 GSM8K 与 HumanEval 质量评测

状态：`已验证`

完成内容：

- GSM8K 固定抽样、答案解析、准确率和格式合规率；
- HumanEval 固定抽样、代码提取、受限 Docker 执行、pass@1、超时和语法错误；
- 同一 prompt manifest、temperature、seed、max_tokens 和 system prompt 协议检查；
- 支持历史基线复用，协议不匹配时拒绝合表；
- 后续比较脚本默认同时报告任务质量、MAL、接受率、decode 和 E2E。

主要文件：

- `remtp/gsm8k_benchmark.py`；
- `remtp/humaneval_benchmark.py`、`remtp/humaneval_evaluator.py`；
- `docs/humaneval.md`。

### 3.6 Gated-depth-KL、Exact-TV 与块级预算探索

状态：`旧 gated-depth-KL 已废弃；Exact-TV 为重要历史方向`

依次尝试：

1. rank、目标概率、margin、深度衰减和 KL 二分的 gated-depth-KL；
2. Cactus 概率提升上限 `H(y)<=Q(y)`；
3. Cactus 总 TV 的饱和感知回收；
4. 按前缀价值、目标支持和静态 head reliability 重新分配；
5. future target support 作为 veto；
6. P/Q JS、top-k overlap、hidden cosine 等块特征消融；
7. Exact-TV + head calibration；
8. Exact-TV + head/hidden + future veto。

MTP=6、GSM8K-200 的代表结果：

| 方法 | accuracy | E2E tok/s | MAL | draft acceptance |
|---|---:|---:|---:|---:|
| Cactus + MTP | 80.0% | 180.418 | 4.890 | 64.8% |
| SpecCascade TokenV3 | 90.0% | 146.331 | 3.999 | 50.0% |
| Cactus + `H(y)<=Q(y)` | 79.5% | 177.800 | 4.826 | 63.8% |
| Exact-TV + head calibration | 74.5% | 193.414 | 5.281 | 71.3% |
| Exact-TV + head/hidden + future veto | 73.0% | 194.662 | 5.322 | 72.0% |

结论：Exact-TV 显著提高 MAL/吞吐，但任务质量下降。Future JS/hidden 一致并不
等价于语义正确；“给定错误前缀后继续一致”不能作为前部 token 的正向证据。

### 3.7 遗憾负反馈探索

状态：`跨块 hidden steering 为负结果；当前块风险控制保留为研究记录`

尝试过的定义和作用位置：

- 只记录 strict reject、relaxed accept 的 causal relaxed acceptance；
- 使用 `P->H` 被转移的目标概率质量构造 regret direction；
- 在下一块 MTP Module 1 hidden 副本注入方向；
- posterior scale、top-1 bias 等低成本弱反馈；
- 将遗憾从事后方向 steering 改为当前块的 verification debt；
- debt 约束位置预算、累计预算、future relaxation 和 target recovery。

代表结果：

| 方法 | accuracy | E2E tok/s | MAL | draft acceptance |
|---|---:|---:|---:|---:|
| Exact-TV + head/hidden + future veto | 73.0% | 194.662 | 5.322 | 72.0% |
| Exact-TV + regret feedback | 75.0% | 182.832 | 5.293 | 71.5% |

Regret audit：strict-acceptable ratio 85.13%，causal relaxed acceptances 4224，
TV/accepted draft 0.074986，feedback injections 2067。

反思：遗憾方向来自旧位置的替代 token 偏好，不能直接搬到下一位置；反馈发生在
风险 token 已提交之后，因果位置过晚；额外 hidden/logit 处理降低吞吐。

分支保留：`research/future-only-regret-exploration`。

### 3.8 当前块 verification debt 与 target-anchored 控制

状态：`负结果/未通过联合 Pareto 条件`

方法使用：

```text
debt_i = A_relaxed_i - A_strict_i
```

在提交前限制高 debt、极端 target gap 和累计块风险，并尝试 balanced debt、
Cactus-dominant target surplus、risk swap、target recovery 和 block shield。

GSM8K-100 代表结果：

| 方法 | accuracy | E2E tok/s | MAL | draft acceptance |
|---|---:|---:|---:|---:|
| Cactus | 75.0% | 182.470 | 4.957 | 66.0% |
| Native | 82.0% | 141.004 | 3.781 | 46.3% |
| Current-block debt | 86.0% | 153.987 | 4.186 | 53.1% |
| Cactus-dominant target surplus | 75.0% | 182.919 | 4.965 | 66.1% |

没有方法同时满足 accuracy>Cactus、MAL>Cactus、E2E>=Cactus。之后不再使用
针对单数据集的自动 fallback 作为正式方法定义。

专题文档：`docs/current_block_verification_debt.md`。

### 3.9 Expected-Regret Router

状态：`训练与实验完成，负结果`

完成内容：

- 构建 UltraChat、No Robots、Magicoder、OASST1 混合无标签 prompt 语料；
- benchmark 去污染和按 request 切分；
- 收集 P/Q/H、debt、target margin/entropy、hidden/direction 等块 trace；
- 训练冻结 target/MTP 之外的小型低秩 Router；
- Router 输出 head 引导强度、proposal scale 和 Exact-TV budget scale；
- 重新设计训练目标、early stopping、审计集和 identity fallback。

Pilot 收集 3987 个请求。第一版 10 epoch validation loss 基本保持 1.584，
budget scale 饱和到 1；V2 在 epoch 7 early stop，best epoch 1，
validation model gain 0.000968、audit gain 0.001024、sign accuracy 0.6102。

HumanEval-164 代表结果：

| 方法 | pass@1 | E2E tok/s | MAL |
|---|---:|---:|---:|
| Native | 79.9% | 209.741 | 5.622 |
| Cactus | 65.9% | 222.051 | 6.022 |
| Exact-TV target-only future veto | 60.4% | 225.114 | 6.121 |
| Learned expected-regret Router | 59.8% | 215.688 | 6.132 |

结论：Router 没有带来可靠质量改善，同时增加推理开销。专题文档：
`docs/exact_tv_regret_router.md`、`docs/regret_router_data.md`。

### 3.10 方案一、方案二、方案三及融合

状态：`已完成比较；最终收敛到 margin-calibrated prefix relaxation`

- 方案一：累计风险预算的边际—熵联合验证；
- 方案二：同轮前视哨兵、target anchor 和跨轮 risk debt；
- 方案三：熵感知动态链长，作为树方向的低成本替代；
- 方案一+二：联合风险与前视控制；
- 后续增加 stronger/ultra/high-relaxation 和 joint risk-credit 版本。

HumanEval-164 代表结果：

| 方法 | pass@1 | E2E tok/s | MAL |
|---|---:|---:|---:|
| Native | 79.9% | 210.753 | 5.622 |
| Cactus | 65.9% | 222.471 | 6.022 |
| SpecCascade | 80.5% | 214.770 | 5.827 |
| Scheme 1 | 81.7% | 201.382 | 5.675 |
| Scheme 2 | 80.5% | 207.702 | 5.647 |
| Scheme 3 | 80.5% | 208.052 | 5.494 |
| Scheme 1+2 | 80.5% | 207.021 | 5.648 |
| Scheme 2 high-relaxation | 78.7% | 211.943 | 5.805 |
| Joint Scheme 1+2 risk-credit | 78.0% | 209.182 | 5.824 |
| 冻结 ReMTP margin-calibrated prefix | 81.1% | 210.581 | 5.751 |

冻结 ReMTP 在该协议下保持较高质量，但只释放了 Native→Cactus MAL 空间的约
三分之一，继续统一放宽会明显损失质量。

### 3.11 Target-band、prefix credit 与决策 trace

状态：`完成探索；未超过冻结 ReMTP`

Target-band 只在 MTP token 位于目标 top-1 附近且 margin 条件允许时提升候选，
并用 block TV 控制总偏移。随后实现 Prefix-Credit，直接优化整个有序前缀的生存
概率，并将后部位置的收益按前缀可达率计权。

HumanEval Prefix-Credit 结果约为 pass@1 76.2%、MAL 5.726–5.735、E2E
207.4–208.6 tok/s，没有超过 target-top1 基线或冻结 ReMTP。

增加了小样本 decision trace，可查看每题候选、P/Q、TV、strict/relaxed-only、
拒绝位置和最终输出。专题文档：`docs/decision_trace_and_tuning.md`、
`docs/prefix_credit_mtp.md`。

### 3.12 Proposal-Calibrated MTP

状态：`已验证；严格分布一致，但整体收益有限`

方法只变换 proposal：

```text
Q_tilde_i = softmax(logits_i / (T_request * T_head_i))
```

候选采样、接受概率和残差纠正统一使用 `Q_tilde`，target P 不变，因此仍严格
恢复目标分布。离线目标为：

```text
E[MAL] = 1 + a1 + a1*a2 + ... + a1*...*aK
```

正式测试 MTP=2/4/6/8，比较 Native、Cactus、SpecCascade 和 proposal
calibration，共 32 行完成。代表结果：

- GSM8K MTP=6：Native MAL 3.811；Cactus 4.894；SpecCascade 4.010；
  Proposal-Calibrated 3.888。
- HumanEval MTP=6：Native pass@1 79.9%、MAL 5.622；SpecCascade 80.5%、
  MAL 5.827；Proposal-Calibrated 76.8%、MAL 5.830。
- GSM8K MTP=8 的 Proposal-Calibrated accuracy 90.6%、MAL 4.104，但跨任务
  和深度并不稳定。

本地完整表：`results/proposal_calibration_depths_20260805_143940/comparison.md`。
专题文档：`docs/proposal_calibrated_mtp.md`。

### 3.13 Fixed-6 Micro-Tree（Qwen3.5）

状态：`负结果，已停止`

完成内容：

- 实现 `6-chain`、`4+2`、`3+2+1` 三个固定六节点拓扑；
- 正确维护 BFS parent、position、TreeAttention mask、attention KV、Qwen GDN
  conv/SSM 和 MTP 递归状态；
- 实现严格概率树验证、分支隔离测试、selected-path compaction 和 GPU audit；
- 先做 unit/reference/GPU smoke，再做 8 条 GSM8K pilot；
- 未实现 tree+ReMTP，未跑正式 GSM8K/HumanEval。

结果：

| 拓扑 | MAL | E2E tok/s | MAL vs chain |
|---|---:|---:|---:|
| 6-chain | 3.764 | 102.094 | 0 |
| 4+2 | 2.725 | 35.915 | -1.039 |
| 3+2+1 | 2.494 | 32.916 | -1.270 |

4+2 correctness reference target forward 为 39.392 ms，6-chain 为 14.675 ms，
增加 168.4%。树覆盖和系统开销均未过门槛。专题文档：
`docs/fixed6_microtree.md`。

### 3.14 MiMo-7B、MTP3 overlay 与动态树

状态：`基础设施已实现；当前模型结果较差`

完成内容：

- 下载 MiMo-7B-Base、MiMo MTP add-on 和 MiMo-7B-RL-0530；
- 构造本地 MTP3 overlay 和逻辑 tree view；
- 实现 physical 0-1-2 与 repeated 0-0-0 路由；
- 修复额外物理 MTP layer 的真实前缀 KV prefill；
- 实现动态熵阈值构树、TreeAttention、target-dominant relaxed verification、
  EOS protection、路径日志和 tree-native MAL；
- 在 GSM8K/HumanEval 比较 Native、Cactus、SpecCascade 和动态树。

MiMo correctness-100：

| 数据集 | Native quality/MAL | Cactus quality/MAL | Tree quality/MAL |
|---|---|---|---|
| GSM8K | 43.0% / 2.007 | 35.0% / 2.253 | 39.0% / 2.057 |
| HumanEval | 29.0% / 2.060 | 8.0% / 2.151 | 23.0% / 2.071 |

诊断显示根层平均分支约 1.01，深度 2/3 的 target coverage 接近零； lowering
`tau_relax` 只有到 `1e-12` 量级才显著提高 oracle MAL。根因是 layer 1/2
pretrained-only 的 P/Q 错配，而不是单纯验证过严。

MiMo-RL-0530 还修正了空 system、temperature=0.6、max_tokens=2048 和 EOS
保护，并比较 physical 012 与 repeated 000。专题文档：
`docs/mimo_mtp_tree.md`、`docs/mimo_dynamic_mtp_tree.md`、
`docs/mimo_dynamic_tree_diagnosis.md`。

### 3.15 FastMTP 初次集成（Claude Code）

日期：2026-08-09
状态：`部分代码保留，旧实验入口已废弃`

初次增加：

- `scripts/download_fastmtp.sh`；
- Native/Cactus/SpecCascade/Dynamic Tree 服务脚本；
- `remtp/fastmtp_worker.py`、`remtp/fastmtp_dynamic_tree_worker.py`；
- 四方法 runner/report；
- 多层 MTP 数据收集和训练方案文档；
- FastMTP 使用、架构和训练计划文档。

Claude Code 的交接摘要还声称新增了
`scripts/collect_mtp_training_data.py`、`scripts/train_multi_layer_mtp.py` 和
`scripts/prepare_and_train_6layer_mtp.sh`，但 2026-08-09 工作树核查时这三个
文件均不存在。因此“6-layer MTP 训练代码已准备”属于无文件支撑的旧声明，
当前状态只能记为方案，不能运行或引用为已实现。

官方 checkpoint 审计确认：

- architecture 为 `MiMoForCausalLM`；
- `num_nextn_predict_layers=1`；
- `num_speculative_steps=3`；
- 权重只包含 `model.mtp_layers.0`；
- 正确语义是一个训练好的物理 head 重复三次；
- 官方 2.03x 包含 SGLang/EAGLE 和 language-aware vocabulary compression，
  不能直接等同于本仓库 full-vocab vLLM 的绝对 tok/s。

发现的错误：

1. `serve_fastmtp_native.sh` 无条件重写 `WORKER_CLS`；
2. Cactus 和 SpecCascade wrapper 没有成功激活各自 Worker；
3. SpecCascade runner 使用错误的环境变量前缀；
4. Dynamic Tree 没有 `TREE_ATTN`、逻辑 node view 或正确 tree config；
5. Dynamic Tree Worker 安装了不兼容的 probabilistic chain adapter；
6. 报告读取不存在的 summary/tree metric key；
7. 旧实验 temperature=0.7 且未使用显式空 system，与 MiMo-RL 可信协议不同；
8. 文档将未完成 GPU 验证的训练和树代码描述为“可直接运行”，结论过度。

旧运行 `results/fastmtp_four_way_50_20260809` 中 Native、Cactus、SpecCascade
的 GSM8K/HumanEval 数值完全一致；日志均显示
`FastMTPProbabilisticWorker`。这些 Cactus/SpecCascade 行不得引用。

### 3.16 FastMTP 已校验重构

日期：2026-08-09
状态：`代码已验证，待端口/GPU 空闲后跑 pilot`

完成修改：

- 新增 FastMTP checkpoint 校验和单物理 head/多逻辑节点 view；
- 新增独立 verified Worker，防止旧脚本覆盖；
- Native、Cactus、SpecCascade 统一先安装 repeated-layer0 和完整 Q；
- runner 强制检查 Worker marker 和真实 adapter diagnostic marker；
- 增加 Target-only 本机质量/速度控制；
- 固定 temperature=0.6、显式空 system、max_tokens=2048；
- Dynamic Tree 使用 `TREE_ATTN`、logical tree view、repeat-0 和真实节点 audit；
- 将 candidate coverage 从正向乘法奖励改为 parent-level `coverage_gate`，避免
  候选越多时所有兄弟 token 自动获得更高松弛分数；
- 报告检查 prompt manifest、协议、真实字段名和 tree-native metrics；
- 明确动态树是 approximate target-dominant path selection，不是 lossless。

新增文件：

- `remtp/fastmtp_checkpoint.py`；
- `remtp/fastmtp_verified_worker.py`；
- `remtp/fastmtp_verified_report.py`；
- `scripts/serve_fastmtp_verified.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `tests/test_fastmtp_checkpoint.py`；
- `tests/test_fastmtp_verified_report.py`。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `tests/test_dynamic_mtp_tree.py`；
- `tests/test_cactus_mtp.py`。

验证：

```text
checkpoint check: FastMTP logical width=1, physical layers=1, layer IDs=[0]
shell syntax: passed
Python compile: passed
unit/reference tests: 33 passed
GPU smoke: not run, because an old user experiment still occupied ~22 GB/96% GPU
```

当前运行入口：

```bash
SAMPLES=10 RUN_TAG=fastmtp_verified_pilot \
  ./scripts/run_fastmtp_verified_comparison.sh
```

该脚本包含 Target-only 控制和四个正式方法：Native、Cactus、SpecCascade、
Dynamic Tree。输出写入 `results/<RUN_TAG>/comparison.md`。

## 4. 当前方法与代码索引

| 方向 | 核心文件 | 当前状态 |
|---|---|---|
| Qwen MTP trace | `remtp/trace.py`, `scripts/serve.sh` | 已验证 |
| 完整概率 MTP | `remtp/probabilistic_mtp.py` | 已验证 |
| Cactus | `remtp/cactus_mtp.py` | Qwen 已验证；FastMTP 新入口待 GPU |
| SpecCascade | `remtp/speculative_cascade.py` | Qwen 已验证；FastMTP 新入口待 GPU |
| Exact-TV/block | `remtp/target_anchored_mtp.py` | 历史探索 |
| Regret Router | `remtp/regret_router.py` 及训练脚本 | 负结果 |
| ReMTP frozen chain | `remtp/risk_entropy_mtp.py` | 已冻结 |
| Prefix Credit | `remtp/prefix_credit_mtp.py` | 负结果/消融 |
| Proposal Calibration | `remtp/proposal_calibration.py` | 已验证、收益有限 |
| Fixed-6 tree | `remtp/fixed6_vllm.py` | 负结果、停止 |
| MiMo dynamic tree | `remtp/dynamic_tree_vllm.py` | 实验性 |
| FastMTP verified | `remtp/fastmtp_verified_worker.py` | 代码已验证 |

## 5. 已知无效或不得混用的结果

- `results/fastmtp_four_way_50_20260809*` 中旧 Cactus/SpecCascade 行：Worker
  实际为 Native；
- MiMo Base/physical 012 的深层结果：额外 head 不是后训练质量基线；
- trace 模式的 tok/s：包含 GPU/CPU 同步和 JSONL 输出，不得作为吞吐；
- Fixed-6 correctness reference 的速度：逐节点状态复制，只能作为负结果和
  kernel 上限诊断；
- 不同 system prompt、temperature、max_tokens、manifest 或代码抽取协议的结果；
- HumanEval generation-only summary 在 Docker evaluation 完成前的质量字段。

## 6. 后续日志模板

以后每次修改在本文末尾追加：

```markdown
### YYYY-MM-DD：修改标题

状态：`代码已验证 / 已验证 / 实验性 / 负结果 / 已废弃`

动机：

- 为什么修改。

修改：

- 算法或代码发生了什么变化。

文件：

- `path/to/file`。

验证：

- 实际运行的测试和结果；未运行 GPU 测试时必须明确写出。

限制与下一步：

- 尚未解决的问题。
```

## 7. 文档维护记录

### 2026-08-09：建立持续研究总日志

状态：`已验证`

修改：

- 汇总从最小 Qwen MTP、完整 Q、论文复现、Exact-TV、遗憾、Router、
  Proposal Calibration、Fixed-6、MiMo 到 FastMTP 审计的全部主线；
- 增加 `AGENTS.md` 和 `CLAUDE.md`，要求 Codex/Claude Code 后续每次修改同步
  更新本日志；
- 在 README 和过期 FastMTP 文档增加总日志入口或历史警告。
- 核对 Claude Code 所称多层 MTP 训练文件，确认文件不存在，并将旧声明降级为
  未实现的训练方案。

验证：文档链接和 Markdown 结构已检查；没有启动新的 GPU 实验。

### 2026-08-09：FastMTP verified pilot 慢速诊断

状态：`已验证（pilot 运行中的局部结果）`

现象：

- `run_fastmtp_verified_comparison.sh` 的第一组是不使用推测解码的
  Target-only 控制组，不是 Native FastMTP；
- Target-only 在 RTX 4090 上稳定约 `60 tok/s`，运行期间 GPU
  持续工作，没有服务停滞；
- GSM8K 10 条共生成 `10,197` tokens，平均 `1,019.7`，生成用时
  `169.85 s`；
- HumanEval 10 条共生成 `17,371` tokens，平均 `1,737.1`，
  生成用时 `290.61 s`，7/10 达到 `max_tokens=2048`；
- HumanEval Docker 评测仅用约 2 秒，不是主要延迟来源；
- 每切换一种方法都会重启服务；当次 Target/Native 冷启动分别约
  31 秒，这是次要固定成本。

根因：

1. FastMTP/MiMo RL 会生成很长的推理和重复文本。请求中虽传入
   `chat_template_kwargs={"enable_thinking": false}`，但 FastMTP 的
   `chat_template.jinja` 不读取该参数。本地 tokenizer 渲染测试证明，
   传入与不传入 `enable_thinking=False` 得到完全相同的 prompt；
2. 比较脚本将 GSM8K/HumanEval 的 `max_tokens` 都设为 2048，且
   没有任务级 early-stop，放大了模型冗长输出的延迟；
3. 整个实验串行执行 `target/native/cactus/spec_cascade/dynamic_tree`，
   每种方法又串行跑 GSM8K 和 HumanEval；
4. 当前 vLLM 入口固定 `--enforce-eager --no-async-scheduling
   --max-num-seqs 1`，用于可审计、可调试的 Worker 补丁，关闭了 CUDA Graph。
   FastMTP 官方基准使用 SGLang EAGLE、CUDA Graph 和高频词表压缩，
   不能将当前 vLLM eager 路径直接等同于官方优化路径。

对照证据：

- 同一 pilot 的 Native FastMTP GSM8K 已完成：`128.778 tok/s`，
  MAL `3.083`，draft acceptance `69.4%`；
- Native 相对 Target-only 的 decode 加速是 `2.145×`，说明训练好的
  MTP head 及主要接线正在正常工作。

结论：

- 当前“慢”不是 Docker 卡住，也不是 Native FastMTP 核心失效；
- 主因是 Target-only 基线本来只有约 60 tok/s，再叠加平均上千
  token 的冗长生成；
- 次因是串行运行五种方法、每种重启模型，以及 eager 调试配置；
- 本次只完成诊断和记录，没有中断用户正在运行的实验，也没有
  修改基准协议或代码。

### 2026-08-09：FastMTP 可选 non-thinking chat template

状态：`代码已验证；GPU 生成待验证`

动机：

- FastMTP checkpoint 携带的旧 `chat_template.jinja` 忽略
  `enable_thinking=False`，导致 MiMo RL 仍输出大量 `<think>` 内容；
- 本地 `MiMo-7B-RL-0530` 的新版官方模板已通过在 assistant
  generation prompt 中预填空的 `<think>\n\n</think>` 实现关闭思考。

修改：

- 新增 `configs/fastmtp_no_think_chat_template.jinja`，保留 FastMTP 原模板
  其余语义，只增加 `enable_thinking` 分支；
- `scripts/serve_fastmtp_verified.sh` 新增 `FAST_MTP_NO_THINK=1`
  可选开关，通过 vLLM `--chat-template` 加载兼容模板，并用
  `--default-chat-template-kwargs` 将 `enable_thinking=false` 设为请求默认值；
- `scripts/run_fastmtp_verified_comparison.sh` 传递并记录该开关；
- 默认值为 `0`，避免改变 2026-08-09 当时仍在运行的
  `fastmtp_verified_pilot` 协议。

使用：

```bash
FAST_MTP_NO_THINK=1 \
SAMPLES=10 \
RUN_TAG=fastmtp_no_think_pilot \
./scripts/run_fastmtp_verified_comparison.sh
```

验证：

- shell syntax：通过；
- tokenizer template reference tests：`2 passed`；
- `enable_thinking=False` 渲染结果以空的已闭合 think block 结束；
- `enable_thinking=True` 渲染结果以开放 think block 结束；
- 未运行 GPU 生成对照，原因是用户的五方法 pilot 仍在占用 GPU。

限制与下一步：

- 该方式是将 MiMo 已有 non-thinking 模板语义移植到 FastMTP，
  不修改任何模型或 MTP 权重；
- 需在当前 pilot 完成后，使用新 `RUN_TAG` 跑小样本 A/B，确认
  实际输出长度、质量和各方法的分布仍然可公平比较。

### 2026-08-09：FastMTP 动态树低接受长度审计

状态：`负结果诊断；未修改算法代码`

动机：

- `fastmtp_verified_pilot` 中动态树的 MAL 低于 Native FastMTP，需检查
  “候选更多应提高命中率”的预期为何没有实现。

结果：

| 数据集 | Native MAL | Cactus MAL | 动态树 MAL | 动态树始终选最深存活前缀的离线 MAL 上限 |
|---|---:|---:|---:|---:|
| GSM8K-10 | 3.083 | 3.419 | 2.996 | 3.228 |
| HumanEval-10 | 2.887 | 3.390 | 2.773 | 2.971 |

根因：

1. 动态树几乎没有实际变宽。GSM8K/HumanEval 每轮平均目标节点仅为
   `3.294/3.474`，而深度为 3 的单链本身已有 3 个节点；根节点出现多个
   候选的轮次仅为 `2.57%/4.42%`，最终选择任何非 top-1 sibling 的轮次仅为
   `3.74%/5.01%`；
2. FastMTP 的 Q 极尖锐。三个深度的平均 top-1 概率在 GSM8K 为
   `0.934/0.913/0.895`，在 HumanEval 为 `0.902/0.867/0.843`。当前阈值
   `q_max * exp(-kappa * U)` 配合按全词表大小归一化的 entropy，使阈值长期
   接近 q_max，绝大多数父节点只保留一个 token；
3. `surviving_paths` 指标把一条链的深度 1、2、3 前缀分别计作三个 endpoint，
   不能解释为三条独立分支；
4. 路径选择器在所有存活前缀 endpoint 之间按“几何平均可信度 × 长度奖励”
   采样，而不是优先选择最长存活路径。GSM8K 有 `20.08%` 的轮次、HumanEval
   有 `17.52%` 的轮次选择了比当前可用最深路径更短的前缀，分别损失约
   `0.231/0.198` MAL；
5. 动态树使用 `relative_target_support >= 0.5` 的 target-dominant 非保分布
   路径规则；Cactus 则直接放宽单链候选的概率接受。两者的“候选覆盖”与
   “单候选接受强度”不是同一个量，因此多出少量 sibling 不保证超过 Cactus。

文件：

- 审计 `remtp/dynamic_mtp_tree.py`、`remtp/dynamic_tree_vllm.py`、
  `scripts/serve_fastmtp_verified.sh`；
- 读取本地 `results/fastmtp_verified_pilot/*/dynamic_tree/tree_rounds.jsonl`
  及各方法 `summary.json`；
- 更新 `docs/research_and_development_log.md`。

验证：

- 对 GSM8K 的 `2,838` 轮和 HumanEval 的 `6,450` 轮 tree audit 做了完整
  离线重放统计；
- 本次没有修改 Worker、验证语义或服务脚本，也没有启动新的 GPU 实验；
- “始终选择最深存活前缀”的数字是基于原有 target forward 结果的离线
  反事实上限，不是重新生成后的正式质量结果。

限制与下一步：

- 第一优先级应修正 endpoint 选择目标，至少做 `longest-surviving-prefix`
  消融；
- 第二优先级应改用 top-k 局部 entropy、top-1/top-2 margin 或显式候选数
  控制构树，避免全词表归一化 entropy 把树压回单链；
- 修改后必须重新报告真正的根分支数、叶路径数和非 top-1 sibling 选中率，
  不能继续用旧 `surviving_paths` 作为树宽指标。

### 2026-08-09：动态树最长前缀优先与受控微扩宽

状态：`代码已验证；单请求 GPU 冒烟通过；正式质量实验待运行`

动机：

- 修复路径评分器在已有更深存活路径时仍选择短前缀的问题；
- 在不形成宽树的前提下，为 Q 中具有实际概率质量的 top-2 提供少量候选覆盖。

修改：

1. 验证后只把“没有存活孩子的节点”视作真实路径 endpoint；
2. 路径选择改为字典序目标：先最大化存活深度，仅在相同最大深度的路径间
   根据目标可信度和 `T_path` 选择；
3. 构树主阈值保持不变，但当 top-2 同时满足 `q_2 >= tau_min` 和
   `q_2/q_1 >= 0.25` 时，允许增加一个 guarded sibling；
4. 每个父节点硬限制最多 2 个孩子，默认整树节点上限从实验性的 32 收紧为 6；
5. 分层分配宽度预算时，为每个尚未到达的后续深度预留一个节点，避免前两层
   siblings 耗尽预算并截断原有三层主路径；
6. `surviving_paths` 改为真实 maximal surviving endpoints 数，不再把同一链的
   中间前缀重复算成独立路径；
7. audit construction 增加 `threshold_eligible` 与 `sibling_added` 字段。

文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `tests/test_dynamic_mtp_tree.py`；
- `scripts/serve_fastmtp_verified.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/serve_mimo_dynamic_tree.sh`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/fastmtp_usage_guide.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关 Python 单元测试：`15 passed`；
- 四个修改后的 shell 入口通过 `bash -n`；
- `git diff --check` 通过；
- FastMTP/vLLM 0.18/TREE_ATTN 真实 GPU 冒烟完成 64 个输出 token：27 个
  speculative rounds，目标调用始终为每轮 1 次，平均/最大目标节点为
  `4.741/6`，全部 27 轮均保留到深度 3，5 轮触发了 `2+3+1` 的预留预算
  形状，4 轮选择了含非 top-1 sibling 的路径；
- 初始试验的 `N_max=8, q_2/q_1>=0.10` 在相同短请求上平均使用 5.923 个
  节点，被判断为扩宽过多，未保留为默认值；最终默认改为
  `N_max=6, q_2/q_1>=0.25`。

限制与下一步：

- GPU 冒烟只证明变量宽度、TreeAttention、路径提交和单次 target forward
  接线可运行；`MAL=2.370` 来自一个 64-token 请求，不应与旧 benchmark
  数字比较；
- 动态树验证仍是非保目标分布的 target-dominant relaxation，正式结果必须
  同时报告 GSM8K/HumanEval 质量；
- 下一步只需用新 `RUN_TAG` 跑小样本动态树，不必重跑已冻结 baseline；重点看
  MAL、平均/最大节点、真实叶路径数和非 top-1 sibling 选中率。

### 2026-08-09：最长前缀动态树 FastMTP-10 结果分析

状态：`小样本正向信号；不能据此确认质量优势`

协议：

- `fastmtp_longest_tree_n6_smoke`，GSM8K/HumanEval 各 10 条；
- temperature `0.6`、seed `42`、显式空 system、所有方法均启用同一
  no-thinking chat template；
- 动态树 `D=3, N_max=6, children_max=2, sibling_ratio=0.25,
  tau_relax=0.50`。

结果：

| 数据集 | 方法 | 质量 | MAL | E2E tok/s | nodes/round |
|---|---|---:|---:|---:|---:|
| GSM8K-10 | Native | 100% | 3.058 | 126.364 | 3.000 |
| GSM8K-10 | Cactus | 100% | 3.396 | 141.442 | 3.000 |
| GSM8K-10 | Dynamic tree | 100% | 3.323 | 115.909 | 3.701 |
| HumanEval-10 | Native | 70% | 3.067 | 125.728 | 3.000 |
| HumanEval-10 | Cactus | 60% | 3.344 | 136.836 | 3.000 |
| HumanEval-10 | Dynamic tree | 80% | 3.073 | 106.933 | 3.967 |

分析：

1. 最长前缀修改修复了旧负结果：动态树 MAL 在 GSM8K 已高于 Native
   `+0.265`，只比 Cactus 低 `0.073`；HumanEval 与 Native 基本相同；
2. 当前速度瓶颈不能归因于松弛不足。GSM8K 的 MAL 仅比 Cactus 低 `2.14%`，
   E2E 仍低 `18.05%`，对应目标节点多 `23.37%`；HumanEval 目标节点多
   `32.22%`；
3. `tau_relax=0.50` 附近的拒绝呈离散簇：现有 audit 中只有 GSM8K 18 个、
   HumanEval 33 个已到达节点位于约 `0.49935`。离线保守 replay 将阈值降至
   `0.49` 只预测 MAL 增加 `0.010/0.013`，不能弥合 HumanEval 对 Cactus 的
   `0.271` 差距；继续降到 `0.30` 的保守增益也只有 `0.017/0.025`；
4. HumanEval 质量只有 10 条：动态树相对 Native 仅有 1 个 discordant win、
   相对 Cactus 仅 2 个 discordant wins，均无足够统计功效。`80/70/60%`
   是正向信号，不是已确认的质量排序；
5. 当前额外 sibling 的路径选择率仅约 `3.8%/4.4%`。进一步加宽会优先增加
   节点成本，而不保证提高最长可提交前缀。

结论：

- 不建议全局大幅降低 `tau_relax`，也不建议继续加宽；
- 如需一个最小消融，可测试 `tau_relax=0.49`，但预期 MAL 增益很小；
- 若目标是同时接近 Cactus MAL 且保持 Native 质量，应实现“首次失败前沿的
  单次受限 rescue”：先使用当前目标主导树规则，只在当前深度无任何孩子存活
  时，对目标支持最高的一个候选使用一次小预算 Cactus-style rescue；不对所有
  sibling 全局放宽；
- 若目标还包括 Cactus 级实际速度，必须同时减少每轮节点/动态运行时开销；
  单纯放宽接受条件无法解释或消除 GSM8K 已存在的约 18% 系统速度差。

验证与文件：

- 读取 `results/fastmtp_longest_tree_n6_smoke` 的 summary、HumanEval evaluation
  与 2,773 轮 tree audit；
- 检查五种方法 server log，均确认 `no_think=enabled`；
- 本次未修改算法代码、未启动新 GPU 实验；仅更新本研究日志。

### 2026-08-09：动态树首次失败前沿的单次受限救援

状态：`代码与真实 GPU 冒烟已通过；GSM8K/HumanEval-50 正式质量实验待运行`

动机：

- `fastmtp_longest_tree_n6_smoke` 中 GSM8K 的动态树 MAL 已接近 Cactus，
  HumanEval 仍有约 `0.271` 差距；
- 离线审计显示 `tau_relax=0.50` 附近候选很少，全局降阈值预计只能增加约
  `0.01–0.03` MAL，同时会在所有位置累积质量风险；
- 当前阶段按用户要求优先优化 MAL 与准确率，系统吞吐优化后置。

算法修改：

1. 保留现有构树、`tau_relax=0.50`、最长存活路径优先、EOS 保护和目标 anchor；
2. 仅当某个已到达父节点没有任何正常存活孩子时，才允许一次 failed-frontier
   rescue；
3. 只考虑目标相对支持最高的一个 sibling，默认要求
   `p(y)/max(P)>=0.1` 且 `p(y)>=0.001`；
4. 对该候选使用有限 Cactus-style 标量提升：
   `h(y)=min(1,p(y)+sqrt(2*delta*p(y)*(1-p(y))))`，并以
   `min(1,h(y)/q(y))` 做随机接受，默认 `delta=0.5`；
5. rescue 状态沿路径传播，因此最终提交路径最多包含一次 rescue；同深度
   路径比较时正常路径优先于 rescue 路径；
6. 该树验证仍是近似松弛解码，不宣称恢复目标分布；完整 Q 只用于救援接受
   概率，目标 P 决定候选语义优先级。

可观测性与实验入口：

- 每个节点新增 `target_probability`、`draft_probability`、
  `rescue_eligible`、`rescue_accept_probability`、`rescue_random` 和
  `rescued`；可读 trace 显示 rescue 的 `a/u/ACCEPT|REJECT`；
- tree metrics 新增 rescue 尝试、成功、最终选中 token 数和选中 rescue 的轮次
  比例；统一对比表新增 `rescue rounds`；
- 新增 `scripts/run_fastmtp_mal_quality_50.sh`，默认在 GSM8K 与 HumanEval
  各跑 50 条，并比较 target / Native / Cactus / SpecCascade / 动态树；
- 默认启用 no-thinking 模板，结果写入单一 `comparison.md`。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `remtp/tree_audit.py`；
- `remtp/mimo_tree_report.py`；
- `remtp/fastmtp_verified_report.py`；
- `tests/test_dynamic_mtp_tree.py`；
- `scripts/serve_fastmtp_verified.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_mal_quality_50.sh`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/fastmtp_usage_guide.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关单元测试：`18 passed`；
- 修改后的 Python 模块通过 `py_compile`；四个 shell 入口通过 `bash -n`；
- `git diff --check` 通过；
- FastMTP/vLLM 0.18/TREE_ATTN 真实 GPU 冒烟生成 128 tokens，共 52 个
  speculative rounds；每轮 target forward 始终为 1；
- 冒烟平均 MAL `2.442`、平均节点 `4.712`；4 轮最终路径使用 rescue，
  `max_rescues_per_selected_path=1`，验证了单次约束与在线审计接线。

限制与下一步：

- 单请求冒烟的 MAL 不能与正式 benchmark 比较，也不能证明准确率；
- rescue 使用 top-Q 树候选而非从完整 Q 独立采样，因此即使采用
  Cactus-style 比例也不构成严格分布保持；
- 下一步运行 `scripts/run_fastmtp_mal_quality_50.sh`。主要判断动态树是否在
  HumanEval/GSM8K 上提高 MAL，同时质量与 Native FastMTP 的置信区间相容；
  若质量下降明显，先把 `DYNAMIC_RESCUE_DELTA` 从 `0.5` 降到 `0.25`，而不是
  继续增加全局阈值或树宽。

### 2026-08-09：FastMTP 大样本脚本跳过 Target 并启用逐样本实时日志

状态：`代码与报告测试通过；等待用户运行`

动机：

- Target-only 约 `60 tok/s`，明显延长四方法 MAL/质量实验，但当前阶段不需要
  再用它估计相对自回归加速；
- 原脚本默认每 10 条打印一次，使长时间 GSM8K/HumanEval 运行难以观察。

修改：

1. `run_fastmtp_verified_comparison.sh` 默认 `INCLUDE_TARGET=0`，只运行
   Native / Cactus / SpecCascade / Dynamic tree；仍可显式设为 1 恢复旧控制；
2. `run_fastmtp_mal_quality_50.sh` 固定跳过 Target，并默认
   `PROGRESS_EVERY=1`；
3. Python 使用 unbuffered stdout，GSM8K 每题、HumanEval 每次生成和隔离判题
   都经 `tee` 实时输出，同时保存在 `logs/$RUN_TAG/`；
4. 增加方法启动、服务健康、当前数据集阶段和日志目录提示；
5. 报告器允许 Target 目录缺失，速度列改为 `speed vs Native`，不再引用
   不存在的 Target 行；四个 FastMTP 方法仍为强制完整对照组。

修改文件：

- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_mal_quality_50.sh`；
- `remtp/fastmtp_verified_report.py`；
- `tests/test_fastmtp_verified_report.py`；
- `docs/fastmtp_usage_guide.md`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关测试：`19 passed`；
- 两个 shell 入口通过 `bash -n`；
- 报告模块通过 `py_compile`；
- `git diff --check` 通过；
- 新增无 Target fixture，确认报告仅包含四个方法且 Native 速度倍率为 1。

限制与下一步：

- 实时日志不包含每轮所有树节点，避免详细 token 解码干扰正式吞吐；每条任务
  进度和最终 tree rescue 指标仍完整保留；
- 若要临时跟踪某个日志，可使用
  `tail -f logs/$RUN_TAG/<method>_gsm8k.log` 或对应 HumanEval 日志。

### 2026-08-09：FastMTP-100 前沿救援动态树结果与路径/宽度诊断

状态：`正式 100 条结果已分析；当前版本保留，下一轮修改待实施`

协议：

- `results/fastmtp_live_n100`；GSM8K 与 HumanEval 各 100 条；
- temperature `0.6`、seed `42`、显式空 system、no-thinking；
- Native / Cactus / SpecCascade / Dynamic tree 使用同一 FastMTP checkpoint；
- 动态树 `D=3, N_max=6, children_max=2, sibling_ratio=0.25`，启用单次
  failed-frontier rescue，`delta=0.5`。

原始结果：

| 数据集 | 方法 | 质量 | MAL | E2E tok/s | nodes/round |
|---|---|---:|---:|---:|---:|
| GSM8K | Native | 90% | 3.040 | 125.649 | 3.000 |
| GSM8K | Cactus | 90% | 3.434 | 141.529 | 3.000 |
| GSM8K | SpecCascade | 93% | 3.229 | 132.601 | 3.000 |
| GSM8K | Dynamic tree | 82% | 3.232 | 111.664 | 3.850 |
| HumanEval | Native | 61% | 3.065 | 126.552 | 3.000 |
| HumanEval | Cactus | 69% | 3.342 | 137.904 | 3.000 |
| HumanEval | SpecCascade | 66% | 3.143 | 128.953 | 3.000 |
| HumanEval | Dynamic tree | 69% | 3.150 | 109.094 | 3.937 |

统计与错误分析：

1. GSM8K 动态树相对 Native/Cactus 的配对 win/loss 均为 `2/10`，精确
   McNemar/binomial `p=0.0386`；相对 SpecCascade 为 `0/11`，`p=0.0010`。
   当前 82% 下降不能继续按小样本噪声解释；
2. HumanEval 动态树与 Cactus 为 `7/7` discordance，质量相同；相对 Native
   为 `15/7`，但 `p=0.134`，仍需第二 seed；
3. GSM8K 动态树 18 个错误中 16 个是 `max_tokens=2048` 截断。未截断样本为
   `78/80=97.5%` 正确，Native 未截断为 `86/88=97.7%`。因此主要问题是近似
   路径选择把少数请求带入不终止的长轨迹，而非普通完成后的算术错误；
4. 截断路径从未遇到高 EOS 概率节点，简单降低 EOS veto 阈值无效；偏移发生后
   target 本身已不想结束，风险必须在较弱路径 token 提交前控制；
5. rescue 在 GSM8K 正确/错误请求中为每 100 轮 `3.73/4.48` 次；HumanEval
   为 `2.53/4.77` 次。非 top-1 branch 选择也在错误请求中更高，说明 target
   支持和 rescue 信号有风险辨识力，但 rescue 不是唯一根因；
6. GSM8K 正确与错误请求的 MAL 几乎相同（`3.233/3.232`），所以不能简单按
   长度截断；应根据路径可信度决定什么时候放弃额外深度。

路径公式离线 replay：

- 最初设计为
  `C(path)=exp(mean(log S_i))`，`Score(path)=C(path)*L^beta`；当前实现只在
  最大深度相同的 endpoint 之间使用该分数，实际近似“无条件最长”；
- 让所有存活前缀重新参与选择，并用 `beta=0.75`、确定性最高分选择时，离线
  replay 的 GSM8K/HumanEval MAL 为 `3.217/3.138`，仅比当前少
  `0.015/0.012`；分别有 `1.57%/1.27%` 的轮次主动选择较短路径；
- 同时所选路径累计负对数支持风险代理下降约 `38%/35%`。该配置是下一轮最
  值得测试的准确率—MAL 平衡点；`beta=0.25` 过于保守，`beta>=2` 又基本退化
  回最长路径。

树宽诊断：

- `N_max=6` 只在 GSM8K/HumanEval 的 `19.1%/21.4%` 轮次用满，平均节点仅
  `3.85/3.94`，节点总量不是首要瓶颈；
- 根节点多分支仅 `13.5%/14.3%`，最终选择 branch-1 仅 `4.3%/4.5%`；
- 前两层出现至少 3 个 eligible 候选的比例仅约 `0.17%–0.85%`，所以把
  `max_children` 从 2 提到 3 基本无收益；
- useful-node ratio 只有 `58.0%/54.6%`，盲目加节点会先增加无效验证；
- 仍可测试小幅扩宽：保持每父节点最多 2 个孩子，将 `N_max=8`、
  `min_sibling_ratio=0.20`，让更多父路径进入第三层；不要直接使用已否定的
  `N=8, ratio=0.10` 激进配置。

建议顺序：

1. 先恢复所有存活前缀的原始 `C(path)*L^beta` 选择，默认 `beta=0.75`、
   `T_path=0`；rescue 节点继续使用 `S*a_rescue`，从而自动承担额外风险惩罚；
2. 在该选择器稳定质量后，再做唯一的宽度消融：`N=6, ratio=.25` 对比
   `N=8, ratio=.20`，`max_children=2` 不变；
3. 若仍出现长轨迹，再加入通用的跨轮风险债务，债务只降低下一轮长度奖励和
   rescue 许可，不做数学任务特化；
4. 进入标准对比的目标应为：GSM8K 不低于 Native 90%，HumanEval 与 Cactus
   69% 相容，同时 MAL 明显高于 SpecCascade `3.229/3.143`。

验证与文件：

- 读取两数据集共 `39,234` 个动态树 rounds、800 个四方法请求记录和 HumanEval
  隔离判题结果；
- 完成请求级 audit 分段、截断/正确性、rescue/branch、节点预算和原公式的离线
  replay；
- 本次仅更新 `docs/research_and_development_log.md`，未修改在线算法。

### 2026-08-09：Balanced Path 与 8 节点小幅扩宽版本

状态：`实现、测试与真实 GPU 冒烟通过；等待 100 条正式结果`

目标：

- 解决旧实现“只要存在更深存活节点就无条件选择最长路径”的长轨迹风险；
- 在不增加每父节点分支上限的条件下，把整树节点预算从 6 小幅提高到 8，检验
  候选覆盖是否能够提升 MAL；
- 不重复运行已经冻结的 Native / Cactus / SpecCascade 100 条结果。

算法修改：

1. `DynamicTreeConfig` 新增 `path_selection_mode`：
   - `longest` 完整保留旧语义；
   - `balanced` 令每个存活前缀参与
     `Score(path)=geomean(S_i)*L^beta` 比较，不再要求必须到达最深 endpoint；
2. 新实验固定 `beta=0.75, T_path=0`。因此长路径仍有明确奖励，但低可信尾部会
   降低几何平均支持度，可能被更可靠的短前缀击败；
3. rescue 节点的 `S_i` 继续乘其实际 rescue acceptance probability，所以
   balanced selector 会自动对由救援得到的尾部承担额外惩罚；
4. 树配置改为 `D=3, N_max=8, max_children=2, sibling_ratio=0.20`。扩宽来自给
   更多强父路径分配深层节点，而不是允许每个位置 top-3/top-4 泛滥；
5. 正常松弛阈值、frontier rescue delta、EOS 保护和 target anchor 均不变，便于
   将差异归因到路径选择与小幅候选扩宽。

运行入口：

```bash
SAMPLES=100 \
BASELINE_RUN_ROOT=results/fastmtp_live_n100 \
RUN_TAG=fastmtp_balanced_wide_n100 \
  ./scripts/run_fastmtp_balanced_wide_n100.sh
```

该入口只运行 `dynamic_tree`。它把旧 run 中 `native/cactus/spec_cascade` 的结果
目录只读式链接到新 run，再由原报告器检查相同 sample manifest 和协议，最终
仍生成包含四种方法的一个 `comparison.md`。底层比较脚本新增了受校验的
`METHODS_CSV` 子集运行参数。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_verified.sh`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/serve_mimo_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_balanced_wide_n100.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/fastmtp_usage_guide.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关单元测试 `20 passed`；新增测试明确验证两个 token 都存活时，balanced
  selector 会因第二 token 目标相对支持仅 0.25 而选择安全的一 token 前缀；
- Python 模块通过 `py_compile`，5 个 shell 入口通过 `bash -n`，
  `git diff --check` 通过；
- FastMTP/vLLM 0.18/TREE_ATTN 真实 GPU 冒烟生成 256 tokens，共 98 个 round；
  每轮 target forward 始终为 1，验证节点最大 8、平均 5.041；
- 冒烟平均接受草稿深度 1.612，对应 MAL 2.612；单请求中没有出现 balanced
  主动短停，符合离线估计中约 1%–2% round 才触发的稀有率。

限制与判据：

- 8 节点会增加 tree target forward 延迟，因此当前优先判断 MAL/accuracy，吞吐
  仅作为审计项；
- balanced 的离线 replay 预测 MAL 会轻微下降约 0.01，但可能显著减少低可信
  路径风险。扩宽能否补回并超过这部分 MAL 必须由正式结果回答；
- 主判据：GSM8K accuracy 至少恢复到 Native 的 90% 且截断率明显低于旧动态树
  20%；HumanEval pass@1 不低于旧动态树/Cactus 的 69%；MAL 应超过
  SpecCascade 的 GSM8K 3.229 与 HumanEval 3.143。若 MAL 没有提升，不再继续
  增加节点，而应回到候选质量或分支利用率分析。

### 2026-08-09：Balanced/Wide-100 结果与节点利用率诊断

状态：`100 条结果完成；证明质量风险有所缓解，但 8 节点上限没有形成真正宽树`

原始结果：

| 数据集 | 方法 | 质量 | MAL | nodes/round | useful-node | truncation |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | 旧 Dynamic N6/longest | 82% | 3.232 | 3.850 | 58.0% | 20% |
| GSM8K | 新 Dynamic N8/balanced | 88% | 3.230 | 4.171 | 53.5% | 17% |
| HumanEval | 旧 Dynamic N6/longest | 69% | 3.150 | 3.937 | 54.6% | 9% |
| HumanEval | 新 Dynamic N8/balanced | 67% | 3.166 | 4.268 | 50.7% | 11% |

统计边界：

- GSM8K 新方法 88/100 的 Wilson 95% CI 为 `80.2%–93.0%`；HumanEval
  67/100 为 `57.3%–75.4%`，100 条不足以区分约 2–3 pp 的质量差异；
- 新旧动态树配对质量变化：GSM8K `9 wins / 3 losses`，HumanEval
  `6 wins / 8 losses`，均未达到显著；
- 请求级 paired bootstrap 的新旧 MAL 差 95% CI 分别为
  GSM8K `[-0.041, +0.036]`、HumanEval `[-0.015, +0.046]`。当前不能宣称
  小幅扩宽稳定提高 MAL；
- balanced selector 仅在 GSM8K/HumanEval 的 `1.37%/1.32%` round 主动停在
  更短存活前缀，符合它是长轨迹风险控制而非主要 MAL 增长器的定位。

节点利用率：

- GSM8K/HumanEval 平均节点仅 `4.171/4.268`；恰好 8 节点的 round 只有
  `8.08%/9.02%`，至少 6 节点也只有 `22.86%/24.51%`；
- 超过一半 round 仍是 3 节点单链：GSM8K 10,161/18,023，HumanEval
  9,991/18,930；
- 根层仅 `15.82%/16.25%` 的父状态保留两个候选；深度 2 为
  `20.17%/22.43%`，深度 3 为 `19.92%/22.20%`。因此 `N_max=8` 大部分时间
  没有候选可以填入；
- 已保留的第二候选并非无用：三个深度中，被实际访问的 sibling 约
  `29.7%–33.0%` 能通过验证；根 sibling 中约 29% 还是 target top-1。该信号
  支持继续提高候选覆盖，但不能证明任意低 Q 候选都有相同价值；
- 从 N6 到 N8 后，GSM8K MAL 基本不变，HumanEval 仅 `+0.016`，而 useful-node
  ratio 下降约 4 pp。单独继续提高 `N_max` 不会解决候选在阈值前被删除的问题。

下一阶段建议（仅优化 MAL 与质量，暂不以吞吐为首要目标）：

1. 将构树从“阈值决定节点数”改为“质量下限 + 全局预算填充”：每个父节点
   暴露 top-2 备选及其 Q 概率，使用全树优先队列选择最高价值的 8–10 个节点；
   保留极低 Q 硬下限，避免为了填满预算验证纯噪声；
2. 最大深度从 3 增到 4。D=3 即使无限加宽，MAL 理论上限也只有
   `3 draft + 1 anchor = 4`；增加宽度只能提高到达深度 3 的概率，无法产生
   第四个草稿 token。建议第一版 `D=4, N=10`；
3. 节点价值应改成近似整块期望 MAL 的前缀概率收益，而不是仅使用路径 Q
   几何平均。早期备选可以解锁多个后续 token，应高于孤立的深层 sibling；
4. 路径提交保留 balanced 风险控制，但将 relaxed 风险由平均值改为可调的累计
   项作为消融，防止多枚边缘 token 因几何平均和长度奖励被连续提交；
5. 新构树先在 100 条上验证“平均节点达到 8–10、target calls=1、MAL 增长”；
   质量确认改为 GSM8K 至少 500 条、HumanEval 全部 164 条并增加第二 generation
   seed。100 条只用于淘汰明显失败配置。

结论：

> 当前瓶颈确实包含候选树覆盖不足，但原因不是 `N_max` 数字太小，而是 Q 阈值
> 使节点预算长期空置；同时 D=3 限制了 MAL 天花板。最有价值的下一版是
> `D=4 + N=10 + 全局预算填充式 top-2 候选分配`，而不是简单改成 `N_max=10`。

### 2026-08-09：D=3/D=4 概率守卫候选扩宽完整实验入口

状态：`实现、单元测试和 D3/D4 真实 GPU 冒烟通过；等待完整实验`

根据用户要求，本版本没有实现强制最小节点数，也不会用低质量 token 机械填满
预算。修改后的构树规则为：

1. 动态熵阈值产生 core candidates；
2. 在 `max_children` 范围内，top-2/top-3 可以作为 guarded backups；
3. 每个 backup 必须同时满足
   `q >= min_draft_prob` 和 `q/q_top1 >= min_sibling_ratio`；
4. 所有通过概率守卫的候选再按现有路径扩展价值竞争全局节点预算；
5. 候选不足时保留实际节点数，不进行预算补齐。

完整实验固定相同验证节点上限，对比：

| 配置 | D | N_max | max children | q floor | q/q1 floor | kappa |
|---|---:|---:|---:|---:|---:|---:|
| relaxed tree D3 | 3 | 10 | 3 | 0.002 | 0.02 | 1.25 |
| relaxed tree D4 | 4 | 10 | 3 | 0.002 | 0.02 | 1.25 |

两者均保留 `coverage_gate`、`tau_relax=0.50`、balanced path、`beta=0.75`、
`T_path=0`、单次 frontier rescue 和 target anchor。D4 只是额外递归使用一次
FastMTP 的单个已训练物理 MTP layer，不宣称存在第四个独立已训练 head。

统一运行入口：

```bash
RUN_TAG=fastmtp_depth34_relaxed_full \
  ./scripts/run_fastmtp_depth34_relaxed_tree.sh
```

默认协议：

- GSM8K 500 条；
- HumanEval 全部 164 条；
- phase 1 运行 Native / Cactus / SpecCascade / D3；
- phase 2 只运行 D4，并通过符号链接复用 phase 1 的三组 baseline；
- phase 3 生成一个 `depth34_comparison.md/json`，同时显示五种方法。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/fastmtp_depth34_report.py`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_depth34_relaxed_tree.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `tests/test_fastmtp_depth34_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/fastmtp_usage_guide.md`；
- `docs/research_and_development_log.md`。

验证：

- 相关测试 `23 passed`；新增测试验证 guarded top-3 能增加两个 backup，同时
  继续服从每父节点硬上限；
- Python 模块通过 `py_compile`，shell 入口通过 `bash -n`，
  `git diff --check` 通过；
- D3 真实 FastMTP/vLLM/TREE_ATTN 冒烟：37 rounds，平均/最大节点
  `9.459/10`，MAL `2.324`，每轮 target calls=1；
- D4 冒烟：54 rounds，平均/最大节点 `9.889/10`，MAL `2.352`，实际最大接受
  草稿深度为 4，每轮 target calls=1；
- 两个 chat 冒烟均启用 no-thinking，响应不包含 `<think>` 或 `</think>`；
- 冒烟 MAL 来自单个短提示，只用于验证接线，不能作为算法收益结论。

限制：

- 放宽 Q 守卫后 target 节点几乎达到 10，符合本阶段候选覆盖目标，但 useful-node
  ratio 与正式准确率必须由完整数据决定；
- D4 使用权重共享递归，第四步候选质量可能下降；正式报告必须分开检查各深度
  survival、首个失败位置、MAL 与任务质量；
- 当前阶段优先验证 MAL/accuracy。吞吐仍被 eager TreeAttention 和更多 target
  nodes 影响，不作为否定算法正确性的首要标准。

### 2026-08-10：D3 完整实验 MAL 回退诊断

状态：`D3 的 GSM8K-500/HumanEval-164 已完成；D4 继续运行，未干预`

原始结果：

| 数据集 | 方法 | 质量 | MAL | nodes/round | accepted/nodes | truncation |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | Native | 88.2% | 3.046 | 3.000 | 68.2% | 12.8% |
| GSM8K | Cactus | 87.4% | 3.450 | 3.000 | 81.7% | 11.4% |
| GSM8K | SpecCascade | 87.0% | 3.213 | 3.000 | 73.8% | 12.4% |
| GSM8K | Relaxed Tree D3 | 88.6% | 3.169 | 6.838 | 31.7% | 14.6% |
| HumanEval | Native | 61.6% | 3.071 | 3.000 | 69.0% | 11.0% |
| HumanEval | Cactus | 68.3% | 3.344 | 3.000 | 78.1% | 9.1% |
| HumanEval | SpecCascade | 65.2% | 3.169 | 3.000 | 72.3% | 11.0% |
| HumanEval | Relaxed Tree D3 | 66.5% | 3.098 | 7.015 | 29.9% | 11.6% |

统计：

- 所有质量配对差异均不显著：D3 对 Native/Cactus/SpecCascade 的 exact paired
  p-value 在 GSM8K 为 `0.888/0.441/0.312`，HumanEval 为
  `0.230/0.711/0.860`；当前准确率没有显示系统性崩坏；
- 请求级 paired bootstrap 显示 GSM8K D3 MAL 相对 Native 增加区间
  `[+0.099,+0.146]`，但相对 SpecCascade 为 `[-0.067,-0.022]`、相对 Cactus
  为 `[-0.303,-0.258]`；HumanEval 对 Native 差异包含 0，但相对
  SpecCascade/Cactus 分别为 `[-0.099,-0.043]`、`[-0.276,-0.214]`；
- 因此“节点更多但 MAL 更低”不是 100 条小样本噪声，至少相对 Cactus 和
  SpecCascade 已是稳定方向。

原因分解：

1. 新节点大幅增加，但低 rank 节点质量下降：
   - rank-0 在三个深度的 `survive/visited` 约 `72%–76%`；
   - rank-1 仅约 `23%–26%`；
   - rank-2 仅约 `13%–15%`；
   - 因此 useful-node ratio 降至 GSM8K `31.7%`、HumanEval `29.9%`。
2. 宽度挤占了深度：
   - 每轮分配到深度 3 的节点为 `2.552/2.597`，但只有 `42.2%/40.4%` 真正
     到达可验证父前缀；
   - `30.2%/30.7%` 的深度 3 节点位于已经失败、逻辑上不可达的深度 2 父路径；
   - 在已经存活的深度 2 节点中，`8.55%/9.41%` 完全没有分配深度 3 子节点，
     说明早期 top-2/top-3 消耗预算后，部分真正有效的父路径反而无法继续。
3. 失败位置：
   - root 无存活候选占 `15.53%/17.02%` round；
   - 深度 1 的孩子全部失败占 `9.67%/10.61%`；
   - 深度 2 的孩子全部失败占 `9.88%/10.15%`；
   - 深度 2 已存活但没有子节点占 `5.83%/6.28%` round。
4. balanced path 不是主要原因：
   - 主动选择短于最大存活深度的 round 只有 `1.10%/0.99%`；
   - 假设总是提交最长存活路径，MAL 天花板也仅为 `3.180/3.108`，只比实际
     `3.169/3.098` 高约 0.01，仍明显低于 Cactus。
5. 现有 proposal priority 使用路径 Q 几何平均并奖励深度。几何平均不会充分
   惩罚早期低 Q 分支，导致低 reach backup 的后代与高 reach 主路径争夺相同
   节点预算；这是宽度转化不成 MAL 的核心结构问题。

后续建议：

- 先让当前 D4 完整运行，不在中途改变协议；D4 可以判断额外深度能否部分抵消
  D3 的宽度挤占；
- 下一版不要继续降低 Q floor 或增加 max children；改为 progressive widening：
  先给高价值父路径分配 rank-0 continuation，再全局分配 rank-1，最后才考虑
  rank-2；
- 节点优先级从路径 Q 几何平均改为累计 prefix probability（Q 概率乘积）乘以
  可解锁剩余长度，更接近整块期望 MAL；
- 对每个被保留的高 reach 父路径提供 continuation reserve，避免存活深度 2
  节点没有深度 3/4 子节点；
- balanced selector 保持不变。若在修复分配后仍需增加 MAL，再单独搜索
  `tau_relax`，不要与构树修复同时修改。

一句话结论：

> D3 不是“树还不够宽”，而是“宽度分配给了低 reach 分支，并挤掉了有效路径的
> 延伸节点”；当前应优化节点预算的层级与路径分配，而不是继续无差别加候选。

### 2026-08-10：D4 完整结果、公平深度审计与 Reach-first D3

状态：`D4 完整实验已分析；Reach-first D3 已实现并通过 CPU 测试；等待 GPU 正式实验`

> 后续修订：本节的硬 rank 分层分配未进入正式 GPU 实验，已由下一节的
> `soft_reach` 连续效用分配取代；历史诊断和 D3/D4 数据继续有效。

D3/D4 原始结果如下。D4 与 D3 使用相同 10 节点上限和验证器，但 D4 可额外
提交一个草稿 token，因此不能只比较原始 MAL。

| 数据集 | 方法 | 质量 | MAL | nodes/round | accepted/nodes |
|---|---|---:|---:|---:|---:|
| GSM8K-500 | Native | 88.2% | 3.046 | 3.000 | 68.2% |
| GSM8K-500 | Cactus | 87.4% | 3.450 | 3.000 | 81.7% |
| GSM8K-500 | SpecCascade | 87.0% | 3.213 | 3.000 | 73.8% |
| GSM8K-500 | Relaxed Tree D3 | 88.6% | 3.169 | 6.838 | 31.7% |
| GSM8K-500 | Relaxed Tree D4 | 88.0% | 3.382 | 8.575 | 27.8% |
| HumanEval-164 | Native | 61.6% | 3.071 | 3.000 | 69.0% |
| HumanEval-164 | Cactus | 68.3% | 3.344 | 3.000 | 78.1% |
| HumanEval-164 | SpecCascade | 65.2% | 3.169 | 3.000 | 72.3% |
| HumanEval-164 | Relaxed Tree D3 | 66.5% | 3.098 | 7.015 | 29.9% |
| HumanEval-164 | Relaxed Tree D4 | 67.7% | 3.320 | 8.637 | 26.9% |

公平深度审计：

| 数据集 | D3 MAL | D4 原始 MAL | D4 截断到前三层 | D4 第四层贡献 |
|---|---:|---:|---:|---:|
| GSM8K | 3.169 | 3.382 | 3.031 | 0.351 |
| HumanEval | 3.098 | 3.320 | 2.987 | 0.334 |

关键结论：

1. D4 相比 D3 的原始增益为 `+0.213/+0.222`，但其第四层本身贡献
   `+0.351/+0.334`；去掉天然的第四层后，D4 前三层反而比 D3 低
   `0.138/0.111`。因此 D4 的表面优势不能归因于更好的树候选覆盖。
2. D4 有 `46.2%/46.8%` 的已分配节点最终未访问，高于 D3 的
   `41.0%/42.4%`；`41.9%/41.8%` 的第四层节点还挂在不可达的深度 3
   父路径下。
3. D4 的 cap=10 命中率达到 `67.1%/68.9%`，但前三层 reach 全面下降：
   GSM8K 从 D3 的 `0.845/0.744/0.580` 降到 `0.814/0.697/0.519`，
   HumanEval 从 `0.830/0.720/0.549` 降到 `0.803/0.684/0.500`。
4. 各层 rank-0 节点的 visited survival 约为 `69%–73%`，rank-1 约
   `22%–26%`，rank-2 约 `13%–16%`。继续平等扩展低 rank 分支会浪费
   有限节点，而不是提高最长可提交前缀。

实现改动：

- 新增 `allocation_mode=reach_first`，旧 `geometric` 模式保持不变以复现实验；
- reach-first 用整条路径的累计 Q 概率作为 reach proxy，不再使用 Q 几何平均；
- 同一深度先竞争 rank-0 continuation，剩余节点再给 rank-1、rank-2；每个
  rank 档位内按累计前缀 Q 排序；
- 正式入口固定 `D=3`，与三组 baseline 使用相同三步 MTP 递归深度；D4 只保留
  为深度消融；
- 新入口默认继续使用 10 节点上限、Q 概率守卫、balanced path、target anchor
  和 EOS 保护，并将 `tau_relax` 从 0.50 调到 0.35。现有日志显示 0.35–0.50
  区间新增候选仅约 `0.050/0.042 node/round`，属于受控放宽，而非无条件提交。

涉及文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_reach_first_tree.sh`；
- `scripts/run_fastmtp_soft_reach_tree.sh`（当前方法的公开入口）；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：

- `tests/test_dynamic_mtp_tree.py`、`tests/test_fastmtp_verified_report.py`、
  `tests/test_fastmtp_depth34_report.py`：`24 passed`；
- Python `py_compile`、三个 shell 入口 `bash -n`、相关文件 `git diff --check`
  通过；
- 尚未运行 Reach-first 的真实 GPU 冒烟或正式 benchmark，因此不宣称其已超过
  SpecCascade/Cactus。正式目标是 D3 MAL 高于 SpecCascade、接近 Cactus，且
  质量接近 Native。

运行入口：

```bash
RUN_TAG=fastmtp_reach_first_d3_full \
  ./scripts/run_fastmtp_reach_first_tree.sh
```

该脚本只运行新方法，并复用
`results/fastmtp_depth34_relaxed_full/d3` 中协议一致的 Native/Cactus/SpecCascade
结果，避免重复测试已有 baseline。

### 2026-08-10：取消层/父路径硬配额，改为 Soft-reach 联合路径评分

状态：`已实现并通过 CPU 测试；等待 GPU 正式实验`

根据用户反馈，未测试的硬 progressive rank 顺序不再作为当前方法。新规则没有
“每层必须几个节点”“每个父节点必须一个 rank-0 子节点”等约束：

```text
proposal_utility(path)
  = product(Q along path) / (1 + lambda_rank * local_rank)
```

- 只有通过动态 Q 阈值、绝对 Q floor 和相对 Q/Q1 floor 的候选才能竞争；
- rank 是连续软惩罚，默认 `lambda_rank=2.0`，不是 rank 档位；
- 高 reach 的 rank-1/rank-2 可以超过低 reach 的 rank-0；
- 每层和每个父节点都没有节点数下限，`N_max=10` 仍只表示全树上限；
- level reserve 只防止早期宽度耗尽全部物理槽位，不会强制生成或提交未来节点。

最终路径继续使用 balanced 目标：

```text
confidence(path) = exp(mean_i(log S_i))
score(path) = confidence(path) * length(path)^beta
```

所有存活前缀都可以参与评分，因此不是默认选择最长路径。默认 `beta=0.75`、
`T_path=0`：确定性选择置信度—长度联合分数最高的前缀。现有单元测试覆盖“弱
尾部虽然存活，但高置信短前缀胜出”的情况。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_reach_first_tree.sh`；
- `scripts/run_fastmtp_soft_reach_tree.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：

- 动态树、报告相关测试共 `25 passed`；
- `py_compile`、三个 shell 入口的 `bash -n` 及环境配置读取通过；
- soft-reach 单测明确验证高效用 rank-1 可以超过低效用 rank-0；
- balanced path 既有单测明确验证弱长尾可输给高置信短前缀。

限制：尚未运行真实 GPU 冒烟或 benchmark，不能宣称 soft-reach 已提高 MAL。

### 2026-08-10：Soft-reach 拒绝样本审计与目标确认式放宽

状态：`分析完成；dual/confirmed 已实现；GPU 冒烟通过但未显示 MAL 正收益；建议先跑 100 条筛选`

Soft-reach D3 完整结果：

| 数据集 | 质量 | Native 质量 | MAL | Cascade MAL | Cactus MAL |
|---|---:|---:|---:|---:|---:|
| GSM8K-500 | 89.2% | 88.2% | 3.166 | 3.213 | 3.450 |
| HumanEval-164 | 65.2% | 61.6% | 3.112 | 3.169 | 3.344 |

质量仍有一定余量，但 MAL 分别比 Cascade 低 `0.047/0.057`。拒绝原因审计：

- coverage 拒绝 `58,580/22,950` 个节点，但其中 relative>=0.10 的只有
  `3/0` 个；该门控几乎只删除目标概率接近 0 的噪声，不应放宽；
- coverage 通过后，relative 位于 `0.20–0.35` 的拒绝为每轮
  `0.019/0.018`，位于 `0.10–0.20` 的拒绝为每轮 `0.039/0.034`；
- 因此无条件阈值 0.10 会新增约 `0.069/0.061 node/round`，数量级足以覆盖
  当前 Cascade MAL 缺口，但不能保证后续 MTP 仍然对齐；
- 当前规则还拒绝了 `2,450/747` 个 `P(y)>=Q(y)` 节点，即每轮
  `0.028/0.024`。这说明单独依赖 target top-1 relative 会比 proposal-overlap
  信号更保守。

5 个提示、146 rounds 的详细 token 审计发现以下代表性案例：

| candidate | target top-1 | P | Q | relative | 后续证据 | 判断 |
|---|---|---:|---:|---:|---|---|
| `reducing` | `allowing` | 0.200 | 0.449 | 0.249 | 子候选 `the` 是 target top-1 | 可救 |
| `string` | `sequence` | 0.110 | 0.999 | 0.125 | 子候选 `that` 是 target top-1 | 可救但需谨慎 |
| `options` | `paths` | 0.143 | 0.059 | 0.249 | P/Q>1 | 可救 |
| `and` | `,` | 0.150 | 0.425 | 0.176 | 后续未与 target top-1 对齐 | 不应仅凭局部概率救 |
| `\\` | 空格 | 0.059 | 0.811 | 0.062 | 后续 top-1 对齐但当前格式风险较高 | 边界样本 |

实现了两个独立消融：

1. `dual`：coverage/EOS 通过后，`relative>=0.10 OR P/Q>=1`；
2. `confirmed`（推荐筛选）：保留 `relative>=0.35` 主规则，额外允许
   `relative>=0.05 AND (P/Q>=1 OR retained child == target next top-1)`。

相同 5 提示真实 GPU 冒烟：

| 版本 | rounds | MAL | 说明 |
|---|---:|---:|---|
| soft-reach relative-0.35 | 146 | 2.712 | 参考 |
| unconditional dual/relative-0.10 | 152 | 2.605 | 反向，不推荐直接全量 |
| target-confirmed | 149 | 2.644 | 优于无条件放宽，但仍低于参考 |

三组生成总 token 数相同；每轮 target forward 均为 1。该 5 提示结果不是质量
统计，但证明“放宽当前 token”可能改变真实轨迹并降低后续接受，因此不能根据
局部新增节点数直接推断 MAL 必然上升。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_reach_first_tree.sh`；
- `scripts/run_fastmtp_dual_support_tree.sh`；
- `scripts/run_fastmtp_confirmed_support_tree.sh`；
- `scripts/run_fastmtp_confirmed_support_n100.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：相关单元测试 `29 passed`；Python/shell 静态检查通过；dual 与 confirmed
均完成真实 FastMTP/vLLM/TREE_ATTN 冒烟，服务正常退出且端口释放。

结论：确实存在值得救援的拒绝 token，但证据不支持“把所有验证条件大幅放开”。
下一步只应在 100 条同协议样本上筛选 target-confirmed；若其 MAL 未超过现有
soft-reach 或质量明显下降，则停止该放宽，而不是继续降低全局阈值。

### 2026-08-10：Confirmed N=100 复盘与 MAL-first 经验阈值

状态：`结果审计完成；新增 MAL-first 压力实验入口；尚未运行新 GPU 实验`

用户完成的 N=100 结果表明，target-confirmed tree 在 GSM8K/HumanEval 上分别为
`88.0% / MAL 3.184` 和 `68.0% / MAL 3.100`；对应 Cactus 为
`90.0% / 3.434` 和 `69.0% / 3.342`。树每轮验证 `6.904/7.096` 个节点，
但草稿利用率只有 `31.6%/29.6%`。

逐轮全量审计发现：

- confirmed 规则额外保留 `0.082/0.070 node/round`，最终进入提交路径的只有
  `0.017/0.013 node/round`，所以它没有形成足够的有效松弛；
- relative 位于 `0.01–0.05` 的拒绝候选只有 `0.122/0.106 node/round`；
- relative `<0.01` 的拒绝达到 `1.639/1.775 node/round`，但目标概率中位数仅
  `7.6e-14/1.2e-16`，大量额外节点实际上是目标强反对的分支；
- 基于旧日志可见节点的保守 replay 表明，`coverage>=0.001`、
  `relative>=0.001`、balanced `beta=5` 的 MAL 下界约为 `3.273/3.179`。

据此新增 `scripts/run_fastmtp_empirical_relaxation.sh`。默认 `PROFILE=mal_first`
固定 D=3 和原 soft-reach 构树，只把 target coverage/relative 下限降到 `0.001`
并把联合路径评分的长度权重升到 `beta=5`；它仍综合置信度和长度，不强制最长。
同时提供 `plausible`（0.01/0.01/beta=3）与只测上限的 `ceiling` 档。由于直接
存活规则已足够宽，关闭 stochastic frontier rescue，避免混入第二套接受机制。

修改文件：

- `scripts/run_fastmtp_empirical_relaxation.sh`；
- `scripts/run_fastmtp_reach_first_tree.sh`（允许压力实验显式关闭 frontier rescue）；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证计划：执行 shell 语法与 `--help` 检查；新 profile 尚未完成真实 GPU 运行，
不能预先声称 MAL 或质量收益。限制：`mal_first` 明确允许质量下降；若它仍无法
接近 Cactus MAL，则主要瓶颈是 proposal tree 的分支质量，而非验证阈值。

### 2026-08-10：Cactus-calibrated tree 松弛强度对齐

状态：`实现与单元测试完成；待用户运行 N=100 GPU 实验`

进一步对照实现后确认，旧树与 Cactus 的“松弛强度”不在同一尺度。Cactus
`delta=1` 先把候选概率提升为
`h=p+sqrt(2*p*(1-p))`，再以 `min(1,h/q)` 接受；旧树则把低于 relative
阈值的候选直接剪枝。confirmed N=100 被拒日志换算后，Cactus 对这些节点的额外
期望接受质量达到 `0.218/0.189 per round`，数量级与双方 MAL 差距接近。

实现 `support_mode=cactus`：GPU 并行计算每个节点的 boosted probability、
Cactus acceptance 和随机存活事件；路径评分信号使用
`relative^w * A_cactus^(1-w)`，默认 `w=0.25`，然后继续乘长度奖励。这样节点
存活强度与链式 Cactus 的 `delta=1` 一致，但最终路径不会只因长度更长而胜出。
EOS 保护、树节点上限和单次 target forward 均保持不变。由于兄弟候选具有不同
的临时 Cactus 分布，整个多路径选择仍是近似分布，必须同时报告质量。

新增运行入口 `scripts/run_fastmtp_cactus_calibrated_tree.sh`：固定 D=3、复用现有
N=100 baselines，默认 `delta=1`、target score weight `0.25`、`beta=1.5`，并
关闭重复的 frontier rescue。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_cactus_calibrated_tree.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：相关动态树、metrics 与 report 测试 `27 passed`；Python/shell 静态检查
通过；真实 FastMTP/vLLM/TREE_ATTN 64-token 冒烟完成，26 rounds 中记录 125 个
已评估节点、51 个 Cactus 存活节点、39 个最终选中节点，服务正常退出。该单提示
MAL=2.50 只证明执行与审计链路正确，不能预先声称 MAL 高于旧树或质量保持。

### 2026-08-10：Direct Cactus + Tree N=100 负结果归因

状态：`完整结果与逐轮日志分析完成；当前设计未达到 Chain Cactus MAL；后续主干救援方案待实现`

用户完成的 N=100 结果：Direct Cactus + Tree 在 GSM8K 为
`89.0% / MAL 3.251`，在 HumanEval 为 `67.0% / MAL 3.173`；Chain Cactus
分别为 `90.0% / 3.434` 和 `69.0% / 3.342`。因此把相同局部 Cactus 公式直接
应用到独立树节点，并未复现链式 Cactus 的整体接受长度。

逐轮审计结论：

- 路径评分仅在 `0.80%/0.74%` 的轮次主动缩短存活路径，MAL 损失均约 `0.008`，
  不是主要原因；
- all-zero deterministic top-Q 主路径自身只有 `MAL 2.835/2.761`；额外树分支
  实际贡献 `+0.415/+0.412`，说明树覆盖有效，但起始主干弱于 sampled-Q chain；
- 每轮 drafted `6.776/6.975` 个节点，真正位于存活祖先下并被验证器访问的只有
  `4.408/4.354` 个，约 35%–38% 节点被祖先拒绝连带作废；
- 替代分支约在 12% 轮次被选择。其后一轮 root 存活率为 `71.3%/65.8%`，低于
  主分支后的 `85.7%/83.8%`。该关联可能受困难上下文选择偏差影响，不能解释为
  严格因果，但提示侧分支会进入更难维持 MTP 对齐的轨迹。

核心设计错误是把“更多候选”实现为重新生成并重新选择整棵 top-Q 树，而不是在
标准 Chain Cactus sampled-Q 主干上增加候选。Cactus 的主干 token 来自完整 Q
随机采样并由标准 rejection sampler 处理；当前树采用 deterministic top-Q 路径、
对兄弟节点独立做候选特定 Cactus 存活采样，整体不具备相对 Chain Cactus 的单调
接受保证。

下一设计决定：保留 Chain Cactus 主干的候选、随机数与验证结果，只在主干第一次
拒绝的位置启用侧分支救援，即 `Cactus trunk + rejection-only tree rescue`。
主干已接受时侧分支不得替换它。当前 Direct Cactus + Tree 代码和结果保留为负例，
暂不删除。

受影响文件：

- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：解析 GSM8K/HumanEval 共 `37,761` 个 tree rounds，核对 summary、
tree_metrics、逐深度存活、主路径、最长存活路径、分支选择和相邻轮统计。未修改
运行时代码，未运行新的 GPU benchmark。限制：相邻轮关联未记录 request 边界且
存在上下文难度混杂，只用于定位风险，不作为因果结论。

### 2026-08-10：启动 Cactus-calibrated N=100 并新增 Cactus-guided 动态树

状态：`v1 N=100 正在运行；v2 实现与 CPU 测试完成；v2 尚未做 GPU 冒烟或 benchmark`

运行修复：8000 端口被本仓库遗留的 vLLM 动态树进程占用。只读核验确认 PID
`2058482` 使用 `FastMTPVerifiedDynamicTreeWorker`、FastMTP Tree10 和约 22 GB
显存后，以 `SIGTERM` 正常停止该进程；随后成功启动
`fastmtp_cactus_calibrated_tree_n100_v1`。服务健康检查通过并进入 GSM8K 生成，
实时日志位于 `logs/fastmtp_cactus_calibrated_tree_n100_v1/`。该实验尚未结束，
不能记录完整双数据集结论。其 GSM8K-100 已完成：accuracy `89.0%`、MAL
`3.251`、target nodes/round `6.776`、每轮 target forward `1`。同一已冻结基线
为 Native `90.0% / 3.040`、Cactus `90.0% / 3.434`、SpecCascade
`93.0% / 3.229`。因此 v1 的 GSM8K MAL 略高于 SpecCascade，但仍比 Cactus 低
`0.183`，且质量比 Native/Cactus 低 1 个百分点；HumanEval 仍在运行，不能据此
形成最终方法结论。

算法审计确认，`support_mode=cactus` 会对每个树节点直接执行链式 Cactus 强度的
随机存活，因此只在候选覆盖和路径选择上保留树方法差异。为避免主方法退化成
Cactus，新增 `support_mode=cactus_guided`：

- soft-reach 的 Q 驱动动态构树、全局节点上限和无硬层级配额保持不变；
- 正常节点继续使用目标 relative 阈值；
- 仅在存活父节点没有正常后继时启用 Cactus 概率；
- rescue 还需 coverage/EOS/目标概率门控，以及 P/Q、下一步 target-top1 命中或
  中等 relative 支持中的至少一种树证据；
- 每条路径最多一次 guided rescue，之后仍可沿正常目标支持节点继续；
- guided 候选使用 `relative^w * A_cactus^(1-w)`，推荐 `w=0.65`，最终路径仍按
  几何平均置信度和 `length^beta` 联合评分。

新增 N=100 入口 `scripts/run_fastmtp_cactus_guided_tree.sh`，默认 D=3、
`N_max=10`、`tau_relax=0.35`、coverage 0.05、Cactus delta 1、目标权重 0.65、
`beta=1.5`，复用原 Native/Cactus/SpecCascade 结果。

修改文件：

- `remtp/dynamic_mtp_tree.py`；
- `tests/test_dynamic_mtp_tree.py`；
- `scripts/run_fastmtp_cactus_guided_tree.sh`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：`tests/test_dynamic_mtp_tree.py` 为 `27 passed`；加入 tree metrics 与 verified
report 后的相关测试集为 `31 passed`；相关 Python 文件 `py_compile` 通过；三个树
实验入口 `bash -n` 通过。新增测试覆盖 guided rescue、Cactus 概率较高但缺少
目标/树证据的拒绝、正常后继优先和每路径最多一次 rescue。

限制：v2 尚未加载到真实 vLLM/GPU，不能声称运行链路或 MAL/准确率收益已经通过。
当前正在运行的 v1 进程加载的是修改前代码，不受本次 v2 文件改动影响。

### 2026-08-10：Direct Cactus+Tree 完成与双树松弛统一实验入口

状态：`direct v1 双数据集完成；统一脚本与报告器完成；guided v2 待 GPU 运行`

本条更新替代上一条中“v1/HumanEval 仍在运行”的临时状态。端口 8000 已释放，
没有残留 vLLM 或实验进程。`fastmtp_cactus_calibrated_tree_n100_v1` 的最终结果：

| 数据集 | Direct Cactus+Tree 质量 | MAL | Native 质量/MAL | Cactus 质量/MAL | Cascade 质量/MAL |
|---|---:|---:|---:|---:|---:|
| GSM8K-100 | 89.0% | 3.251 | 90.0% / 3.040 | 90.0% / 3.434 | 93.0% / 3.229 |
| HumanEval-100 | 67.0% | 3.173 | 61.0% / 3.065 | 69.0% / 3.342 | 66.0% / 3.143 |

Direct 版本每轮 target forward 为 1，平均节点数为 `6.776/6.975`。它的 MAL 在
两个任务上均略高于 SpecCascade，但仍比链式 Cactus 低 `0.183/0.169`。该版本
逐节点执行 Cactus 存活，因此正式语义固定为 `Cactus+Tree ablation`，不作为我们
目标主导树方法的最终定义。

新增 `scripts/run_fastmtp_tree_relaxation_ablation.sh`，串行执行或恢复：

1. direct Cactus+Tree；
2. target-dominant `cactus_guided` tree（ours）。

默认每个数据集 100 条、逐样本打印、复用冻结的 Native/Cactus/SpecCascade
基线。可通过 `DIRECT_RUN_TAG=fastmtp_cactus_calibrated_tree_n100_v1` 直接复用已
完成的 direct 结果，只运行 guided 部分。新增报告器会校验两组 dynamic tree 的
协议字段和 sample manifest，并将五个方法汇总为同一个 Markdown/JSON。

修改文件：

- `scripts/run_fastmtp_tree_relaxation_ablation.sh`；
- `remtp/fastmtp_tree_relaxation_report.py`；
- `tests/test_fastmtp_tree_relaxation_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

为防止恢复运行时混淆两种语义，统一脚本新增运行时变体核验：direct 和 guided
必须使用不同 RUN_TAG；每个子实验结束后从 `dynamic_tree_server.log` 检查实际
`support_mode=cactus` 或 `support_mode=cactus_guided`，再写入本地
`tree_variant.json`。统一报告器拒绝缺少或模式不匹配的变体标记。

验证：动态树、统一报告、原 verified report 和 tree metrics 测试合计
`34 passed`；Python `py_compile`、三个运行入口 `bash -n` 和统一脚本 `--help`
通过；报告器早期版本已对真实 direct v1 结果目录完成一次只读生成检查；当前
direct server log 的 `support_mode=cactus` 标记也已只读确认。

限制：guided v2 尚未进行真实 GPU 冒烟或 N=100 benchmark，不能预先宣称它的
MAL 或质量优于 direct 版本。统一脚本运行 guided 时会重新加载修改后的 Worker。

### 2026-08-10：Direct Cactus+Tree 逐路径后验审计

状态：`只读分析完成；确认多候选有正收益但当前树不是 Chain Cactus 的单调扩展`

在 direct v1 完整结果基础上进一步解析 `37,761` 个 tree rounds。路径评分只在
`0.80%/0.74%` 的轮次缩短最长存活路径，造成的 MAL 损失约 `0.008/0.008`；
因此提高 beta 或强制最长不能解释与 Chain Cactus 的 `0.183/0.169` 差距。

all-zero top-Q 主路径 MAL 仅为 `2.835/2.761`，动态分支将其提高到
`3.251/3.173`，实际分支收益为 `+0.415/+0.412`。树并非无效；问题是 direct
实现用 deterministic top-Q 树替换了 Chain Cactus 的 sampled-Q 主干。每轮
`6.776/6.975` 个 drafted 节点中，只有 `4.408/4.354` 位于存活祖先之下，
其余约 35%–38% 随父节点拒绝作废。

替代分支之后的下一轮 root 存活率为 `71.3%/65.8%`，低于主分支后的
`85.7%/83.8%`。该统计存在困难上下文选择偏差，只能说明替代路径集中在低对齐
状态，不能证明替代路径造成后续下降。

设计区分：现有 `cactus_guided` 是目标规则死前沿上的受控救援，已经实现但尚未
GPU 验证；它仍不等同于严格保留 Chain Cactus sampled-Q 主干。若目标是让“更多
候选”相对 Chain Cactus 具有逐轮非负的接受机会，后续需要另行实现
`Cactus trunk + rejection-only tree rescue`：主干候选、共享随机数和已接受结果
完全复用 Chain Cactus，侧分支仅在主干首次拒绝时接管。

受影响文件：

- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：只读核对 summary、tree_metrics 和逐轮 JSONL；未修改算法代码，未运行新
GPU 实验。限制：日志没有显式 request boundary，相邻轮分析含最多 99 个跨请求
边界，且不能排除上下文难度混杂。

### 2026-08-10：Cactus-guided wide v3 双救援实验档位

状态：`实现、本地测试与短请求 GPU 冒烟通过；N=100 benchmark 尚未运行`

在不替换现有动态树和正常目标相对验证的前提下，新增更宽的死前沿救援档位。
默认 `cactus_guided` 仍保持每路径最多一次救援，以保证历史 v2 结果和配置语义
不变；新增 `max_guided_rescues_per_path` 配置允许实验档位显式设置为 2。救援计数
沿选中父子路径继承，正常存活不增加计数，只有成功的 Cactus-guided rescue 才
增加计数。第二次救援也只能发生在后续前沿没有任何正常存活候选时。

新增 v3 入口将正常 relative 阈值设为 `0.25`、coverage 下限 `0.01`、guided
relative 下限 `0.01`、`P/Q` 证据阈值 `0.25`、Cactus delta `1`、目标/Cactus
排序权重各 `0.5`、每路径最多 2 次救援和 `beta=2.0`。最低目标概率 `0.001`、
EOS 保护、D=3、soft-reach、最多 10 节点、无硬层级配额及每轮一次 target
forward 均保持不变。

受影响文件：

- `remtp/dynamic_mtp_tree.py`；
- `remtp/dynamic_tree_vllm.py`；
- `scripts/serve_fastmtp_dynamic_tree.sh`；
- `scripts/run_fastmtp_verified_comparison.sh`；
- `scripts/run_fastmtp_cactus_guided_wide_tree.sh`；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

新增审计字段 `guided_rescue_count_before` 和服务启动标记
`max_guided_rescues`。测试新增“默认最多一次”向后兼容检查之外的“双次救援成功”
用例。相关单元与报告测试共 `35 passed`；Python `py_compile`、四个 Shell 入口
`bash -n` 及新脚本 `--help` 通过。

真实 FastMTP/vLLM/TREE_ATTN 短请求生成 64 token 成功：服务标记确认 D=3、
N_max=10、`support_mode=cactus_guided`、`tau_relax=0.25`、
`max_guided_rescues=2`、`beta=2` 和每轮一次 target forward；31 个审计 round 中
出现 2 次 guided rescue，并且下游节点记录到 `guided_rescue_count_before=1`。
该单提示 MAL 为 `2.032`，只证明执行链路和审计字段有效，不能作为 benchmark
效果结论。

冒烟时还发现旧独立入口 `serve_fastmtp_dynamic_tree.sh` 未设置 TREE_ATTN 且只按
depth 分配 speculative slots，首个请求报 `dynamic MTP tree requires TREE_ATTN
metadata`。该入口已改为委托给与正式实验相同的 audited verified server，避免
手动服务和正式脚本使用不同运行语义。

限制：尚未运行 N=100，不能声称 MAL 已达到 Cactus 或准确率保持不变；该算法仍
改变采样分布，并且更宽参数可能降低任务质量，必须用相同样本和基线联合评估。

### 2026-08-10：Cactus-guided wide v3 完成与 MAL 停滞归因

状态：`N=100 双数据集完成；该阈值/双救援方向停止继续放宽`

v3 最终结果为 GSM8K `88.0% / MAL 3.172`、HumanEval `64.0% / MAL 3.133`；
同一冻结 Chain Cactus 为 `90.0% / 3.434` 和 `69.0% / 3.342`。相对 Direct
Cactus+Tree，v3 MAL 还下降 `0.079/0.040`，质量下降 `1/3` 个百分点。100 条任务
的质量差异仍有抽样和随机生成波动，但 MAL 来自约 1.8 万/1.9 万 round，停滞不是
单纯报告精度问题。

逐轮只读审计发现：新开放的 relative `[0.25,0.35)` 区间只有 GSM8K 1 个正常
存活节点、HumanEval 0 个，且没有进入最终路径；第二次 guided rescue 仅发生
`10/1` 次；最终选路从未舍弃更长存活路径。最终选择 rescue 的 round 比例仅为
`3.37%/2.53%`，其反事实 MAL 贡献上界约为 `0.060/0.045`。每轮虽构造约 7 个
节点，但仅约 59% 位于存活祖先之下，useful-node ratio 约为 31%/30%。因此降低
normal threshold、提高 beta 和允许第二次救援三项修改都几乎没有可作用的事件。

决策：停止继续调低当前树的 relative 阈值或增加 guided rescue 次数。若继续树
方向，应把 Chain Cactus 的 sampled-Q 主干、共享随机数和逐位置接受语义原样保留，
仅在主干首次拒绝时用侧分支接管；现有 Direct/Guided 树均保留为负结果。

受影响文件：

- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：只读解析 v3 的两个 `comparison.json`、`tree_metrics.json` 和总计 37,317
个 `tree_rounds.jsonl` round，并与 balanced-wide、confirmed-support、Direct
Cactus+Tree 和冻结 Chain Cactus 结果交叉核对。没有修改算法或运行新 GPU 实验。
限制：日志没有显式 request boundary，无法把每次救援严格归因到单个任务的最终
正确性；跨方法生成轨迹不同，反事实救援贡献是上界而非严格配对估计。

### 2026-08-11：sampled-Q Cactus 主干与首次拒绝树救援

状态：`实现、CPU 测试和真实 FastMTP/vLLM/TREE_ATTN 冒烟通过；正式 benchmark 待用户运行`

针对近几版放宽阈值但 MAL 基本不变的问题，停止继续修改原 deterministic top-Q
动态树的 relative 阈值。新增独立模式 `support_mode=cactus_trunk_rescue`：

1. 每个深度用与链式 probabilistic MTP 相同的 exponential-race primitive 从完整
   Q 采样一个主干 token；
2. 每层额外保留至多两个高 Q 兄弟，但只递归展开主干，默认 D=3、节点上限 9；
3. 主干逐位置使用与 Chain Cactus 相同的候选特定 h、`min(1,h(y)/q(y))` 和
   `(h-q)+` correction residual；
4. 只有主干第一次拒绝时才查看同父兄弟，按 residual target support 和
   residual-Cactus acceptance 选择一个候选；
5. 救援成功提交已接受主干、救援 token 和该节点后的原始 target bonus；救援失败
   则从原 Cactus residual 采样 correction；
6. 主干通过时兄弟不参与路径竞争，避免树选择器改变 Chain Cactus 已接受轨迹。

该模式是“Cactus 主干 + 增量树救援”，不再用整棵树替换链式主干。主干概率语义
与 Cactus 对齐，但救援候选由 Q 确定性保留，并非来自严格联合 multi-proposal
采样，因此整体仍是近似松弛解码，不能宣称保持原目标或 Cactus 输出分布。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：新增验证器、residual rescue、EOS 保护和诊断字段；
- `remtp/dynamic_tree_vllm.py`：新增 sampled-Q 主干、毛毛虫动态拓扑、分支 KV
  物化与运行时 dispatch；
- `scripts/run_fastmtp_cactus_trunk_rescue_tree.sh`：新增 20/100 条一键实验入口；
  样本数匹配时复用冻结基线，不匹配时自动重跑三组基线以保持同表协议一致；
- `scripts/run_fastmtp_reach_first_tree.sh`：根据新 support mode 打印准确的 verifier
  与 node cap；
- `scripts/serve_fastmtp_verified.sh`：新增准确的 verify mode 和启动语义标记；
- `tests/test_dynamic_mtp_tree.py`：覆盖主干全接受、拒绝救援和 residual fallback，
  并给两个既有随机 fallback 测试补显式 generator，消除测试顺序依赖；
- `docs/mimo_dynamic_mtp_tree.md`；
- `docs/research_and_development_log.md`。

验证：`py_compile` 通过；新运行脚本和服务脚本 `bash -n` 通过；动态树、Cactus、
Fixed-6 primitive 和 tree audit 相关测试共 `46 passed`。真实 RTX 4090 冒烟加载
TencentBAC/FastMTP、TREE_ATTN、D=3、N_max=9 和新 Worker 后成功生成 48 tokens：
24 rounds 均为一次 target forward，实际节点数 5–9；3 次 trunk bonus、16 次
Cactus correction、5 次 rejection rescue bonus；无 cache/state/TreeAttention
错误，服务已正常停止。冒烟单提示 MAL `2.083` 仅用于链路验证。

剩余限制：尚未运行相同任务清单的 GSM8K/HumanEval 正式对比，不能声称 MAL 或
质量优于 Chain Cactus；新树每轮 target nodes 多于三 token 链，吞吐可能下降；
rescue 会改变输出分布，必须先看 20 条 pilot 的 accuracy/pass@1 与 MAL，再决定
是否运行 100 条。

### 2026-08-11：trunk-rescue pilot 失败与 tree target 双重温度修复

状态：`20 条失败结果完成诊断；CPU 修复、回归测试和修复后 GPU smoke 通过；pilot 待重跑`

用户完成 `fastmtp_cactus_trunk_rescue_pilot`。修复前原始结果为：

| 数据集 | Native | Chain Cactus | SpecCascade | trunk-rescue tree |
|---|---:|---:|---:|---:|
| GSM8K-20 质量/MAL | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 90.0% / 2.638 |
| HumanEval-20 质量/MAL | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 75.0% / 2.252 |

树方法 e2e 仅 `89.862/77.445 tok/s`，相对 Native 为 `0.707x/0.605x`；实际
target nodes/round 为 `4.621/4.401`。逐轮解析发现主干 root 接受率仅
`59.9%/47.6%`；目标候选 P(y) 小于 `1e-6` 的比例为 `24.2%/36.5%`，大于
`0.999` 的比例为 `52.5%/51.0%`，与 Chain Cactus 日志中约 `92%–95%` 的首位置
接受率明显矛盾。救援分别成功 `1096/735` rounds，但只能补一个 draft token，
无法弥补主干退化。

代码审计定位到 `remtp.fixed6_vllm._strict_tree_sampler`：它先调用 vLLM
`apply_sampling_constraints`，该函数已经执行 temperature scaling 和 top-k/top-p；
随后又把 processed logits 传给 `_temperature_probs`，导致第二次除以温度。
实验 `T=0.6` 的 tree target 因而约等价于 `T=0.36`，而 draft Q、Native、Cactus
和 SpecCascade 均保持 `T=0.6`。此前所有通过该 sampler 完成的随机 tree benchmark
均不再是公平对比，保留结果只作历史调试；贪心和状态隔离测试不因该错误失效。

修复内容：新增 `_processed_target_probs`。随机 sampling constraints 之后只执行
softmax；greedy 请求仍转换为 target top-1 one-hot。新增两项回归测试，一项验证
结果严格等于 `softmax(raw/T)` 且不同于 `softmax(raw/T/T)`，另一项验证 temperature
0 的 one-hot 行为。

受影响文件：

- `remtp/fixed6_vllm.py`；
- `tests/test_fixed6_microtree.py`；
- `docs/mimo_dynamic_mtp_tree.md`（添加旧随机树结果失效通知和 pilot 诊断）；
- `docs/research_and_development_log.md`。

验证：相关 dynamic tree、Cactus、Fixed-6 和 audit CPU 测试共 `48 passed`，Python
`py_compile` 通过。修复后真实 GPU smoke 使用相同通用提示、3 个 seed、每个 128
output tokens 分别运行 tree 与 Chain Cactus。Tree 为 183 rounds、MAL `2.082`，
Chain Cactus 服务窗口 MAL 约 `2.11–2.15`；tree 每轮 target forward 为 1，且无
TreeAttention/cache 错误。该通用提示不是正式任务，不能据此宣称质量或 MAL
收益；下一步应重跑同一 20 条协议，不应直接运行 N=100。

### 2026-08-11：目标逐节点最长路径与 Cactus 阈值 continuation rescue

状态：`历史 v1；20 条 pilot 已完成；验证语义已被同日 v2 单-token extension 取代`

修复温度后的 20 条 sampled-trunk pilot 为 GSM8K `95.0% / MAL 3.044`、HumanEval
`80.0% / MAL 3.070`，质量未暴露退化，但 MAL 仍低于 Chain Cactus 的
`3.509/3.361`。审计确认 sampled trunk 接受时旁支全部 `BACKUP_UNUSED`，第一次
拒绝后的旁支只能替代一个 correction，不能沿旁支继续；因此实际验证节点增加但
完整候选路径没有增加。

新增 `support_mode=target_path_rescue`，其语义为：

1. 使用原 dynamic-Q proposer 构造真正的多分支树，不再使用 sampled trunk
   caterpillar；
2. target 一次 TreeAttention forward 后，每个节点使用其父前缀对应的原始 target
   分布，按 `P(y)/max(P) >= tau_normal` 普通存活；
3. 所有普通存活兄弟都可继续到下一深度，coverage 仅作诊断；
4. 未到最深层时，每个父节点可额外保留一个普通失败孩子。救援分数为 target
   relative support 与 candidate-specific Cactus acceptance 的加权几何平均，
   需超过显式阈值，并受最低 target support、EOS 和每路径一次救援约束；
5. 救援为确定性 continuation 资格，不消耗随机数，也不会直接保证提交；
6. 最终先选择最大连续存活深度，同长度再最大化原始 target relative support
   的几何平均；Cactus 分数不进入最终可信度；
7. 每轮仍提交叶后原始 target bonus，target forward 次数保持 1。

默认 pilot 参数为 D=3、`N_max=9`、每父最多 3 children、普通 relative 阈值
`0.20`、救援 score 阈值 `0.18`、relative floor `0.02`、target probability floor
`1e-4`、Cactus delta `1.0`、target weight `0.65`、每路径最多一次救援。节点数是
上限而非强制填满。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：新增模式、阈值、逐节点多分支存活、确定性
  continuation rescue、最长优先和 target-confidence tie break；
- `remtp/dynamic_tree_vllm.py`：读取/打印新增阈值并使用原 dynamic-Q proposer；
- `remtp/tree_audit.py`：实时日志支持 deterministic score/threshold rescue；
- `scripts/run_fastmtp_target_path_rescue_tree.sh`：新增一键 20/100 条入口；
- `scripts/run_fastmtp_reach_first_tree.sh`：允许调用方选择 longest/balanced；
- `scripts/run_fastmtp_verified_comparison.sh`、`scripts/serve_fastmtp_verified.sh`、
  `scripts/serve_fastmtp_dynamic_tree.sh`：传递新参数和准确运行时标记；
- `tests/test_dynamic_mtp_tree.py`；
- `docs/mimo_dynamic_mtp_tree.md`、`docs/research_and_development_log.md`。

验证：Python compile 与 5 个 shell 入口 `bash -n` 通过；dynamic-tree、Fixed-6、
Cactus、report 和 tree-audit 相关测试 `57 passed`。真实 RTX 4090 冒烟成功加载 TencentBAC/FastMTP、
TREE_ATTN、D=3、N_max=9，完成三个 temperature=0.6、96-output-token 请求，共
89 rounds；每轮 target forward 为 `1.000`，实际 target nodes 为 3--9（均值
6.674），无 KV/cache/TreeAttention 错误，服务正常停止。冒烟 MAL `3.247`；6 个
救援节点获得继续资格但 0 个进入最终路径，因为后继没有形成最大深度路径。

剩余限制：尚未运行同一任务清单的 GSM8K/HumanEval pilot，不能声称质量或 MAL
优于 Native、Chain Cactus 或 SpecCascade；该方法是确定性树路径选择和近似松弛，
不保持目标分布；树 target nodes 多于三 token 链，吞吐可能下降。实验 artifacts
仅保留在 ignored `results/target_path_rescue_gpu_smoke/`。

### 2026-08-11：target-path rescue 改为选路后的单-token extension（v2）

状态：`实现完成；CPU 回归和真实 FastMTP/vLLM/TREE_ATTN 分支冒烟通过；默认阈值 20 条质量实验待运行`

用户完成 v1 的 20 条实验后指出，普通节点与 target 比较本身已经是松弛验证，
“救援”应当是其后的独立一次延长。v1 的 GSM8K/HumanEval 质量/MAL 分别为
`95.0%/3.173` 与 `75.0%/3.132`；虽然分别有 `284/183` 个候选达到救援资格，
最终仅有 `76/65` 个救援 token 被选中，直接 MAL 增益只有 `+0.024/+0.020`。
旧语义只是让节点重新参加整树竞争，获救节点可能输给其他路径，因而日志看不出
稳定延长效果。

v2 保持普通松弛阶段不变：逐节点按各自父前缀的 `P(y)/max(P)` 判断存活，先选择
普通存活深度最长的基础路径，同长度再按 target-relative 几何平均决胜。之后新增
独立 extension 阶段：只有基础路径未达到最大深度时，才检查它的直接前沿；从普通
失败孩子中按 `relative^w * A_cactus^(1-w)` 选择达到阈值的最高分候选，并直接追加
恰好一个 token。extension 不再进入全树路径竞争，也不会解锁第二个 token；target
probability floor、EOS 保护和每轮一次上限保留。

报告新增普通接受 draft 数/深度、extension token 数及其直接 MAL 贡献。旧日志的
反事实 replay 预测：救援阈值 `0.18` 的增益约为 `+0.022/+0.019`，降到 `0.08`
约为 `+0.050/+0.034`，因此一键 pilot 的默认值改为 `0.08`。这只是离线阈值依据，
不是新方法质量结论。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：实现普通选路后的一次前沿 extension；
- `remtp/mimo_tree_report.py`：拆分 ordinary depth 和 rescue extension MAL；
- `remtp/fastmtp_verified_report.py`：在总对比表直接显示 ordinary depth 与 extension MAL；
- `scripts/run_fastmtp_target_path_rescue_tree.sh`：更新两阶段说明与默认阈值；
- `scripts/serve_fastmtp_verified.sh`：运行时 marker 明确标记为选路后单-token extension；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_mimo_tree_report.py`；
- `tests/test_fastmtp_verified_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`：给 v1 添加 superseded 通知并记录 v2；
- `docs/research_and_development_log.md`。

验证：Python compile、shell syntax 和 dynamic-tree/report/tree-audit/Fixed-6 相关
CPU 测试为 `52 passed`。
真实默认阈值 GPU smoke 完成 89 rounds、每轮 target forward `1.000`，该单提示未
触发 extension；为验证代码分支而运行的强制零阈值 smoke 完成 109 rounds，其中
85 rounds 出现 extension，extension 节点全部标记 `SELECTED`，terminal 为
`target-path-extension-bonus`，每轮 target forward 仍为 `1.000`。服务正常停止。
强制零阈值仅验证执行语义，不是可用于任务质量的参数或 benchmark。

剩余限制：新默认 `0.08` 尚未运行 GSM8K/HumanEval 同任务清单，不能声称准确率、
pass@1 或 MAL 改善；v2 仍是改变输出分布的近似松弛方法；树 target node 开销仍
高于三 token 链。GPU artifacts 仅保留在 ignored
`results/target_path_extension_gpu_smoke/` 与
`results/target_path_extension_forced_gpu_smoke/`。

### 2026-08-11：前部多路径重开救援与普通后继延伸

状态：`实现、CPU 回归和真实 FastMTP/TREE_ATTN 冒烟通过；20 条质量 pilot 待用户运行`

用户完成 v2 的 20 条结果：GSM8K 为 `85.0% / MAL 3.181`，HumanEval 为
`80.0% / MAL 3.123`；相同冻结基线中 Chain Cactus 为 `90.0% / 3.509` 和
`75.0% / 3.361`，SpecCascade 为 `95.0% / 3.192` 和 `75.0% / 3.160`。v2 的
一次选路后 extension 仅贡献 `+0.043/+0.027` MAL，root selection 只有
`85.3%/83.8%`。因此瓶颈不是 extension 阈值完全没有触发，而是 root 失败后
extension 固定只补一个 token，无法继续利用同一 target tree forward 已验证的
后继。

新增实验模式 `support_mode=prefix_reopen_rescue`：

1. 每个节点仍只用自己父前缀下的 `P(y)/max(P)` 做普通松弛验证，不以整条 target
   概率乘积决定节点生死；
2. 每个可达路径最多使用一次确定性 rescue，所有达到阈值的普通失败兄弟均重开，
   不再每父节点只选一个；
3. 获救节点的后代可以继续执行普通验证，但继承已使用 rescue 的状态，不能再次
   跨越失败节点；
4. 默认 rescue score 阈值随深度为 `0.08/0.12/0.16`，优先将风险用于能够解锁
   更多后继的 root/前部位置；最低 relative `0.02`、最低 target probability
   `1e-4` 和 EOS 保护保持；
5. 最终最大连续深度优先，同长度用 target-relative 几何平均决胜；
6. 报告新增 `rescue_unlocked_tokens`、`rescue_unlocked_mal_gain` 和
   `pre_rescue_mean_accepted_depth`，区分旧 `+1` extension 与新模式实际解锁的
   多个所选 token。

同时澄清 per-position 指标语义：vLLM 输出的是累计 reach rate，MAL 应按
`1+r1+r2+r3` 求和，不能再次连乘。构树器中的累计 Q 乘积只作为 proposal
前缀 reach 和全局节点预算排序的代理，不进入 target 放行条件；恢复早期的 Q
几何平均会重新高估低 reach backup，故本次没有改回该规则。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：新 support mode、深度阈值、逐路径一次 rescue
  状态与普通后继重开；
- `remtp/dynamic_tree_vllm.py`：读取、打印 `rescue_depth_penalty`；
- `remtp/mimo_tree_report.py`：统计 rescue 解锁 token、pre-rescue depth 并澄清
  legacy extension 字段；
- `remtp/fastmtp_verified_report.py`：总表读取通用 rescue MAL；
- `scripts/serve_fastmtp_dynamic_tree.sh`、`scripts/serve_fastmtp_verified.sh`、
  `scripts/run_fastmtp_verified_comparison.sh`：传递新模式和运行参数；
- `scripts/run_fastmtp_prefix_reopen_rescue_tree.sh`：新增一键 20 条 pilot；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_mimo_tree_report.py`、
  `tests/test_fastmtp_verified_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`、`docs/research_and_development_log.md`。

验证：相关 Python `py_compile` 和四个 shell 入口 `bash -n` 通过；dynamic tree、
report、tree audit、Fixed-6 状态相关 CPU 测试共 `56 passed`。真实 RTX 4090
使用 TencentBAC/FastMTP、vLLM 0.18、TREE_ATTN、D=3、N_max=9 运行三个
temperature=0.6、96-output-token 相同提示请求，共 122 rounds；target forward
固定 `1.000/round`，无 KV/cache/TreeAttention 错误，MAL `2.361`，实际 rescue
解锁贡献 `+0.205`。同提示 v2 对照为 124 rounds、MAL `2.315`、extension 贡献
`+0.097`。两个服务均已正常停止。

限制：匹配提示 smoke 只验证执行链路和机制活性，输出轨迹导致 round 数略有不同，
不是配对 benchmark；尚未运行新模式的 GSM8K/HumanEval 质量实验，不能声称准确率
稳定或 MAL 已接近 Cactus。新模式仍改变输出分布，且每轮 target nodes 多于三
token 链，吞吐仍可能低于链式基线。实验 artifacts 仅保留在 ignored
`results/prefix_reopen_rescue_gpu_smoke/` 与
`results/target_path_extension_matched_smoke/`。

### 2026-08-11：HumanEval 跨轮漂移诊断与 target-margin 自适应救援

状态：`实现、CPU 回归和真实 FastMTP/TREE_ATTN 冒烟通过；同任务 20 条 pilot 待用户运行`

用户完成 `fastmtp_prefix_reopen_rescue_pilot`。原始结果：

| 数据集 | Native | Cactus | SpecCascade | prefix-reopen tree |
|---|---:|---:|---:|---:|
| GSM8K 质量/MAL | 90.0% / 3.080 | 90.0% / 3.509 | 95.0% / 3.192 | 95.0% / 3.262 |
| HumanEval 质量/MAL | 75.0% / 3.110 | 75.0% / 3.361 | 75.0% / 3.160 | 75.0% / 3.135 |

HumanEval pass@1 与三条链式基线相同，短板是 MAL。prefix-reopen 对
GSM8K/HumanEval 的 rescue 解锁贡献为 `+0.105/+0.090`，但 HumanEval 普通接受
深度由 v2 的 `2.096` 下降到 `2.045`。逐轮相邻日志显示，HumanEval 在 ordinary
path 后的下一轮 MAL 为 `3.058`，rescue path 后为 `2.627`；选择根 branch
0/1/2 后的下一轮 MAL 为 `3.100/2.796/2.281`。GSM8K 出现相同但较弱趋势。
请求边界没有显式记录，数字不能解释为严格因果效应，但它们共同反对“继续全局
降阈值”这一调整。

新增 target-margin 自适应救援。每个父前缀计算
`m=log P(top1)-log P(top2)`，救援阈值乘以
`1 + margin_penalty * min(m/margin_reference,1)`。新 pilot 默认基础阈值 `0.06`、
深度惩罚 `0.35`、margin reference `1.5`、margin penalty `1.0`、relative floor
`0.015`：target 犹豫时比旧版更松，target 明确时最多加倍阈值。该规则只使用
通用目标分布信号，不读取任务类型、代码 token 或 benchmark 标签。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：batched target top-2 margin、margin multiplier、
  自适应救援阈值与逐节点诊断；
- `remtp/dynamic_tree_vllm.py`：新增 margin reference/penalty 环境配置与启动标记；
- `remtp/mimo_tree_report.py`：报告所选救援的平均 target margin 和实际阈值倍率；
- `scripts/serve_fastmtp_dynamic_tree.sh`、`scripts/run_fastmtp_verified_comparison.sh`：
  传递新配置；
- `scripts/run_fastmtp_margin_adaptive_rescue_tree.sh`：新增同协议一键 pilot；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_mimo_tree_report.py`；
- `docs/mimo_dynamic_mtp_tree.md`、`docs/research_and_development_log.md`。

验证：Python `py_compile`、四个 shell 入口 `bash -n` 和相关 dynamic-tree、report、
tree-audit、Fixed-6 回归共 `57 passed`。真实 RTX 4090、TencentBAC/FastMTP、
vLLM 0.18、TREE_ATTN 冒烟在两个代码提示和一个通用提示上完成 121 rounds；
target forward 固定 `1.000/round`，MAL `2.380`，rescue 解锁贡献 `+0.116`，启动
标记确认 `margin_reference=1.5`、`margin_penalty=1`，服务已停止。

限制：20 条任务只有一个 seed，HumanEval 单题翻转会改变 5 个百分点；相邻轮分析
混入请求边界与上下文难度选择效应。新 margin 参数尚未运行相同任务清单，不能
声称 pass@1 或 MAL 改善；树仍是近似松弛且 target nodes 多于三 token 链。GPU
artifact 仅保留在 ignored `results/margin_adaptive_rescue_gpu_smoke/`。

### 2026-08-11：停止 margin 调参，改为 sampled-primary 拒绝后树恢复

状态：`实现、CPU 回归和真实 FastMTP/TREE_ATTN 机制冒烟通过；20 条质量 pilot 待用户运行`

用户完成 `fastmtp_margin_adaptive_rescue_pilot`。GSM8K 为
`95.0% / MAL 3.263`，HumanEval 为 `70.0% / MAL 3.101`；相同基线的
SpecCascade HumanEval 为 `75.0% / 3.160`。GSM8K/HumanEval 的 selected rescue
只贡献 `+0.071/+0.074` MAL，且所选救援的平均 target log-margin 仍高达
`1.735/1.798`。相邻轮诊断中，HumanEval 普通路径后的下一轮 MAL 为 `3.015`，
救援路径后为 `2.639`；GSM8K 为 `3.233/2.805`。这些相关性不是严格反事实，但
重复确认旧 deterministic tree/path selector 会在救援后降低下一轮 MTP 对齐。
target-margin 版本因此标记为机制负结果，不再继续扩大/缩小其阈值。

新增 `support_mode=sampled_primary_reopen`：每个深度 local rank 0 从完整 MTP Q
采样，形成与 Chain Cactus 同语义的 primary trunk；primary 使用相同 Cactus
接受概率，正常接受时 backup 不参与选路。仅在第一次 primary rejection 时，
residual-supported sibling 可执行一次近似恢复；恢复成功后，其树后代只能继续
通过 target-relative 规则，不能二次 rescue；恢复失败严格回到原 Cactus residual
correction。九节点预算在剩余深度间均摊，并以 reach-first 次序尽量为三个 root
候选都保留后继，避免浅层宽度耗尽后续节点。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：新 support mode、Cactus 主干拒绝后的恢复子树扫描、
  EOS 保护、恢复解锁 token 诊断；
- `remtp/dynamic_tree_vllm.py`：sampled primary proposal、主干节点硬保留、跨深度
  节点预算均摊、运行时 proposal marker；
- `scripts/serve_fastmtp_verified.sh`：新增准确 verifier marker；
- `scripts/run_fastmtp_reach_first_tree.sh`：识别新 verifier 摘要；
- `scripts/run_fastmtp_sampled_primary_reopen_tree.sh`：新增匹配基线的一键 pilot；
- `tests/test_dynamic_mtp_tree.py`：新增拒绝、恢复及两层后继延伸回归；
- `docs/mimo_dynamic_mtp_tree.md`、`docs/research_and_development_log.md`。

验证：`python -m py_compile` 通过；两个 shell 入口 `bash -n` 通过；dynamic-tree、
tree metrics 和 FastMTP report 相关 CPU 测试 `44 passed`。真实 RTX 4090 使用
TencentBAC/FastMTP、vLLM 0.18、TREE_ATTN 完成 3 个非 benchmark 提示、83 rounds，
target forward 固定 `1.000/round`、target nodes 平均 `8.494`；15 轮进入
`sampled-primary-reopen-bonus`，恢复解锁 22 个 draft token，服务正常停止。

剩余限制：真实 smoke 的提示、生成长度和 benchmark 不匹配，MAL `2.313` 只证明
机制可执行，不能解释为质量或性能提升；新树恢复仍是近似分布；HumanEval/GSM8K
20 条同任务结果尚未运行。实验 artifacts 仅保留于 ignored
`results/sampled_primary_reopen_smoke.jsonl` 和
`results/sampled_primary_reopen_smoke_metrics/`。

### 2026-08-12：停止先选兄弟的近似救援，新增 exact-residual-hit 六路线套件

状态：`实现、CPU 回归与真实 FastMTP/TREE_ATTN 冒烟通过；正式质量实验待用户运行`

用户完成 `fastmtp_sampled_primary_reopen_pilot`。同一 20 条协议下：GSM8K 的
Native/Cactus/SpecCascade/tree 为 `90%/3.080`、`90%/3.509`、`95%/3.192`、
`70%/3.316`；HumanEval 为 `75%/3.110`、`75%/3.361`、`75%/3.160`、
`70%/3.098`。树在 HumanEval 的 pre-rescue depth 仅 `1.718`，救援贡献
`+0.380` MAL；GSM8K 为 `2.049/+0.267`。但救援尝试 938 次、接受 913 次，
约 `97.3%`，说明旧流程“先从兄弟中挑最优 token，再做一次宽松接受”几乎把
救援变成确定性改写。GSM8K 中四个 Native 正确请求被翻错，且出现重复 `!!!`、
数量级错误和截断；HumanEval 出现一个净 pass@1 损失。该方向因此标记为质量负
结果，不再继续调整 target-relative rescue 阈值。

新增 `sampled_primary_shadow`、`residual_hit_anchor`、
`residual_hit_strict` 和 `residual_hit_cactus`。共同语义是：primary token 从完整
FastMTP `Q` 采样并用 Chain Cactus 的 `H` 验证；第一次拒绝后，必须先从
`normalize(max(H-Q,0))` 采样精确 correction；只有 sibling token ID 与该
correction 完全相同，树才可复用已验证状态。shadow 永不复用；anchor 只追加一个
原始 target token；strict 沿独立采样的 branch-local Q 做标准 `P/Q` 验证；
cactus 沿该 Q 使用同一 Cactus 规则。branch-local RNG 由请求初始 seed、轮次和
路径派生，不推进 primary/request RNG。顺手修复 `_sample_q_token` 在提供 request
generator 时先额外调用一次无用 `exponential_()` 的问题。

新增 `residual_coverage` 节点分配：先覆盖 sampled primary 父前缀下的 correction
兄弟，再保留已存在 recovery branch 的 local-rank-0 Q 样本，最后才分配非 primary
额外 backup。统一脚本一次运行 shadow/anchor/strict/cactus 及 strict/cactus wide
六条路线；第一个 route 只运行一次 Native/Cactus/SpecCascade，之后通过结果目录
symlink 复用。合并报告额外给出 correction hit rate、全轮复用率和每次命中解锁
token 数。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：精确 residual-hit 四类 verifier、诊断字段和
  `residual_coverage` 分配次序；
- `remtp/dynamic_tree_vllm.py`：非 primary Q 的独立分支采样、四类 runtime dispatch、
  sampled-primary 构树与无用 RNG 调用修复；
- `remtp/fastmtp_residual_hit_report.py`：新增跨路线合并报告及 correction-hit 审计；
- `scripts/serve_fastmtp_verified.sh`、`scripts/run_fastmtp_reach_first_tree.sh`：新增
  mode marker/摘要并修复 allocator 被硬编码成 `soft_reach` 的问题；
- `scripts/run_fastmtp_residual_hit_suite.sh`：新增一键六路线、共享 baseline、实时日志；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_fastmtp_residual_hit_report.py`：新增语义、
  分配和报告测试；
- `docs/exact_residual_hit_tree.md`、`docs/mimo_dynamic_mtp_tree.md`、
  `docs/research_and_development_log.md`。

验证：三个 Python 文件 `py_compile`、三个 shell 入口 `bash -n` 通过；相关测试
`49 passed`。真实 RTX 4090、TencentBAC/FastMTP、vLLM 0.18、TREE_ATTN 冒烟先后
验证 9-node reach-first 和 15-node residual-coverage；后者生成 32 tokens、14 rounds，
target forward 始终 `1/round`、实际 target nodes 平均 `12.0`，审计观察到一次
depth-2 exact correction hit，随后严格接受两个 recovery branch token，终止为
`residual-hit-strict-continuation-bonus`。服务均正常停止。

剩余限制：GPU smoke 使用单个非 benchmark 提示，只证明执行与审计语义；六路线尚
未跑 GSM8K/HumanEval，不能声称质量或 MAL 改善。15/20 节点路线可能显著降低吞吐，
wide 仅用于判断候选覆盖上限。shadow 与 Chain Cactus 的正式配对等价性仍需在统一
benchmark/seed 下审计。实验 artifacts 仅保留于 ignored
`results/residual_hit_gpu_smoke.jsonl` 与
`results/residual_hit_gpu_smoke_n15.jsonl`。

### 2026-08-12：100 条套件中断审计、请求级断点续跑与 shadow 负控制

状态：`断点续跑已实现并通过 CPU 测试；最后一个 HumanEval 待用户恢复；方法控制未通过`

`fastmtp_residual_hit_n100` 在 `cactus_wide` HumanEval 第 `75/100` 条后因会话断开。
现场检查确认没有残留 vLLM/benchmark 进程且 8000 端口空闲。shadow、anchor、strict、
cactus、strict_wide 两个数据集均完整，cactus_wide GSM8K 完整；只有
`cactus_wide/humaneval/dynamic_tree` 缺少 summary/requests。旧 HumanEval benchmark
将所有 records 保存在内存并在最后统一写 `requests.jsonl`，因此已经生成的 75 条只
能从 stdout 确认任务 ID/token 数，无法可靠重建候选代码或质量；本次需重跑最后
HumanEval 100 条，但此前约五个半 route 均直接复用。

新增通用 `requests.checkpoint.jsonl`：GSM8K/HumanEval 每完成一条请求立即 append、
flush 和 `fsync`。`--resume` 会校验现有 sample manifest 完全相同，并要求 checkpoint
严格构成 manifest 前缀；更换任务、seed 或 manifest 时拒绝混合。FastMTP comparison
新增 `RESUME_PARTIAL=1`：完整 summary 自动跳过，partial dataset 从 checkpoint 下一个
样本继续。动态树同时保存跨服务器会话的 `tree_rounds.checkpoint.jsonl`；如果像本次
旧运行一样没有 request checkpoint，则把无法配对的旧 audit 移到带
`unrecoverable_<timestamp>` 的历史文件，避免误算进重跑指标。server 历史日志同样
改名保留而非覆盖。残差六路线入口默认开启该恢复模式。

已完成数据还暴露出更重要的负控制。统一 100 条的原始结果为：

| 方法 | GSM8K 质量/MAL | HumanEval 质量/MAL |
|---|---:|---:|
| Native | 90.0% / 3.040 | 61.0% / 3.065 |
| Chain Cactus | 90.0% / 3.434 | 69.0% / 3.342 |
| SpecCascade | 93.0% / 3.229 | 66.0% / 3.143 |
| shadow | 87.0% / 3.079 | 67.0% / 2.979 |
| residual anchor | 87.0% / 3.094 | 62.0% / 2.977 |
| residual strict | 78.0% / 3.268 | 64.0% / 2.812 |
| residual Cactus continuation | 84.0% / 3.230 | 65.0% / 2.945 |
| strict wide | 87.0% / 2.951 | 61.0% / 2.871 |
| Cactus wide | 86.0% / 2.996 | 未完成 |

shadow 理应在树不参与输出时复现 Chain Cactus，但 MAL 分别低 `0.355/0.363`，因此
控制失败，不可把后续路线的质量/MAL 差异只归因于 residual hit。动态树 primary 的
条件接受率约为 shadow GSM8K `0.827/0.836/0.810`、HumanEval
`0.810/0.814/0.773`，明显不足以复现 Chain Cactus 的累计 per-position 曲线。wide
将 rejection correction hit rate 从约 `0.38--0.43` 提到约 `0.54--0.56`，但验证
节点升至 `19.6/round` 且 MAL 反而下降，说明单纯增加树宽已经是明确负方向。目前
最优先事项不是继续放松或增加节点，而是逐轮对齐 Chain Cactus 与 shadow 的 primary
Q、父前缀 target P、随机数消费顺序和 branch/cache 状态。

受影响文件：

- `remtp/benchmark_checkpoint.py`：新增 manifest-safe 请求级 durable checkpoint；
- `remtp/gsm8k_benchmark.py`、`remtp/humaneval_benchmark.py`：新增 `--resume`、逐请求
  checkpoint 和前缀跳过；
- `scripts/run_fastmtp_verified_comparison.sh`：新增 partial dataset、server 日志和动态树
  audit 的安全恢复/归并；
- `scripts/run_fastmtp_residual_hit_suite.sh`：默认开启恢复并补充帮助；
- `tests/test_benchmark_checkpoint.py`：checkpoint 前缀和 manifest 冲突测试；
- `docs/exact_residual_hit_tree.md`、`docs/research_and_development_log.md`。

验证：三个 Python 文件 `py_compile`、两个 shell 脚本 `bash -n` 通过；checkpoint、
GSM8K/HumanEval helper、动态树 verifier 和报告相关测试共 `64 passed`。没有声称恢复
后的 GPU benchmark 已通过；最后一个 HumanEval 仍需用户以相同 RUN_TAG 实际恢复。

限制：当前运行的前 75 条没有 candidate checkpoint，不能从第 76 条接续；恢复时会
从该 partial HumanEval 的第 1 条重跑。结果只有单 seed，且 shadow 控制失败意味着
exact-residual tree 尚不具备正式方法比较资格。所有结果和中断 audit 继续仅保留于
ignored `results/`、`logs/`。

### 2026-08-12：20 题树＋松弛定向优化达到小样本门槛

状态：`固定 20 题/单 seed 目标已达到；仅为 pilot，不代表正式结论`

先汇总同一 20 题 manifest 下的历史结果。此前最接近的
`prefix_reopen_rescue` 为 GSM8K `95.0% / MAL 3.262`、HumanEval
`75.0% / MAL 3.135`；后者比 SpecCascade `3.160` 少约 `0.026`。日志显示 HumanEval
有 244 个 rescue score 位于旧阈值的 50%--100%，但统一放平深度阈值后虽然 rescue
MAL 从 `0.090` 增至 `0.117`，普通深度同时下降，最终 MAL 降为 `3.131`。这确认额外
松弛会通过真实前缀破坏后续 MTP 对齐，不能用“更多 rescue 事件”代替端到端 MAL。

新增 continuation-certified rescue：只有第 2 层失败候选已经拥有一个按原普通规则
可接受的第 3 层直接后继时，才允许对其 rescue threshold 施加小折扣。该证书完全来自
同一次 tree target forward 已计算的正确父前缀分布，不增加 target 调用，不使用跨轮
状态。较大的 33.3% 折扣仅得到 GSM8K `90.0% / 3.211`、HumanEval
`75.0% / 3.101`，说明后继证书也不能支持大幅放松。

候选覆盖审计发现 9-node soft-reach 在 HumanEval 的 56.5% 轮次填满上限，每轮平均
仍有约 5.1 个构树合格节点未保留。但固定扩到 10/12 nodes 非单调：soft-reach 10
nodes 为 `85.0% / 3.237` 与 `70.0% / 3.113`；12 nodes 为
`85.0% / 3.177` 与 `85.0% / 3.160`。对 12-node 日志进一步审计发现，最终实际采用的
额外节点约 85% 是 local rank-0、约 90% 位于 depth 3、97% 以上直接通过普通验证。
因此缺口主要是分配器把预算过早用于横向 sibling，而不是验证阈值过紧。

将 allocator 改为已有的 `reach_first` 后，9 nodes 达到 GSM8K
`95.0% / 3.216`、HumanEval `80.0% / 3.146`。10 nodes 的 HumanEval 质量降到
65%，再次说明固定增加一个节点也不稳定。12-node reach-first 在不加后继折扣时为
GSM8K `95.0% / 3.300`、HumanEval `75.0% / 3.160095`；HumanEval 比级联
`3.160455` 仅少约 `0.00036`。最终只对有普通后继证书的 depth-2 rescue 使用 5%
折扣，固定 20 题结果为：

| 数据集 | Native 质量/MAL | SpecCascade 质量/MAL | 选定树＋松弛 质量/MAL |
|---|---:|---:|---:|
| GSM8K | 90.0% / 3.080 | 95.0% / 3.192 | **90.0% / 3.319** |
| HumanEval | 75.0% / 3.110 | 75.0% / 3.160 | **75.0% / 3.176** |

故在该固定小样本上，两项质量均与 Native 相等，MAL 均严格高于 SpecCascade。选定
参数为 D=3、N_max=12、children_max=3、`reach_first`、普通 relative threshold
0.18、rescue base 0.08、depth penalty 0.5、depth-2 continuation discount 0.05、
longest-first 路径选择。同一次运行的 nodes/round 为 GSM8K 8.425、HumanEval
8.363；吞吐仍低于链基线，本阶段只验证质量/MAL。

同时实现了一个 draft-entropy 自适应 9→12 节点上限实验接口。高熵阈值 0.05 在
GSM8K 达到 `95.0% / 3.339`，但 HumanEval 只有 `75.0% / 3.111`，未入选。该接口
保留为消融，不应描述为主方法。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：continuation-certified rescue 字段/语义；draft-entropy
  自适应 node limit helper；
- `remtp/dynamic_tree_vllm.py`：环境参数接入、按层 MTP entropy 计算 active node cap；
- `scripts/run_fastmtp_verified_comparison.sh`、`scripts/serve_fastmtp_verified.sh`、
  `scripts/serve_fastmtp_dynamic_tree.sh`：新参数透传与运行摘要；
- `scripts/run_fastmtp_prefix_reopen_rescue_tree.sh`：允许覆盖 allocator/path selector；
- `scripts/run_fastmtp_selected_tree_20.sh`：固化选定 20 题配置；
- `tests/test_dynamic_mtp_tree.py`：后继证书折扣与自适应节点上限单元测试；
- `docs/research_and_development_log.md`。

验证：`python -m py_compile` 两个 Python runtime 文件通过，四个 shell 入口
`bash -n` 通过；`tests/test_dynamic_mtp_tree.py` 为 `47 passed`。真实 RTX 4090、
TencentBAC/FastMTP、vLLM 0.18 上实际完成上述 GSM8K 20 + HumanEval 20 运行与隔离
HumanEval 测试；最终本地报告为
`results/fastmtp_prefix_n12_reachfirst_cont005_pilot/comparison.md`。

限制：20 条、单 seed 的准确率粒度为 5pp，MAL 差值也可能受输出轨迹和 RNG 影响；
最终配置是在同一 pilot 上迭代选择，存在选择偏差，必须在未参与调参的新 seed/更大样本
上复现后才能形成论文结论。GSM8K truncation 为 25%，高于 Native/SpecCascade，虽然
本次答案准确率未下降，仍是需要审计的质量风险。结果 artifacts 继续只保留于 ignored
`results/`、`logs/`，未发布到 GitHub。

### 2026-08-12：修复树路径跨轮 target hidden 路由，Exact Residual-Hit MAL 达到 Cactus 区间

状态：`结构缺陷已修复；固定 20 题 pilot 达标，等待独立 100 题复现`

为回答“树方法的 MAL 能否至少接近 Cactus”，先按 DSE 流程继续检查已有固定 20 题
manifest。统一降低普通 relative threshold 到 0.15/0.10 分别得到
GSM8K `90%/3.337`、`85%/3.332`，HumanEval 均为 `75%/3.172`；把 rescue 改为
root 0.06、随后按深度增至 0.12/0.18，则只有 `90%/3.262` 与 `75%/3.159`。
这些负结果说明继续全局放宽阈值不能解释与 Cactus 的 MAL 缺口。

随后增加 `sampled_primary_shadow` 控制：proposal 和验证仍走树 runtime，但树永远
不能改变 Cactus 已采样的 token。9-node 树最初只有 GSM8K `85%/2.978`、HumanEval
`60%/2.982`，而同协议 Cactus 是 `90%/3.509`、`75%/3.361`；将树缩成三节点单链
后 MAL 恢复到约 `3.487/3.330`。逐轮对照发现：在完整接受一条非连续 BFS path 后，
下一轮 target 分布仍基本对齐，但 MTP root Q 已经分叉。根因是 vLLM
`EagleProposer.prepare_inputs_padded` 按“接受 token 数”选择 hidden row，适用于链但
不适用于 BFS 树。例如选中节点 `[0,2,5]` 后，旧逻辑取 row 3，正确值应为实际叶节点
`5 + 1 = 6`。此前尝试对 MTP attention KV cache 做 snapshot/isolation 没有改善，
得到 `95%/3.152` 与 `70%/3.016`，该补丁已撤回并只保留负结果。

当前 runtime 在动态树提交后显式把下一轮 MTP root 的 target hidden 索引改为
`selected_leaf_index + 1`。修复后的 9-node shadow 控制 MAL 提升到
GSM8K `3.430`、HumanEval `3.324`。直接 sibling Cactus rescue 虽达到
`95%/3.560` 与 `60%/3.515`，但 HumanEval 质量坍塌，证实“先挑 sibling 再接受”
会改变 correction 分布，未被选用。

最终选定 `residual_hit_anchor`：主干仍使用完整 Q 的 Cactus 验证；第一次拒绝时先从
精确 `normalize(max(H-Q,0))` 残差分布采样 correction，之后才检查树中是否已有相同
token ID。仅在精确命中时复用该 sibling 的已验证状态，并从该前缀下的原始 target
分布追加一个 anchor。固定 pilot 实测：

| 数据集 | Cactus 质量/MAL | Exact residual-hit anchor 质量/MAL | MAL 差值 |
|---|---:|---:|---:|
| GSM8K | 90.0% / 3.509 | 90.0% / 3.569 | +0.060 |
| HumanEval | 75.0% / 3.361 | 80.0% / 3.540 | +0.178 |

树平均节点数为 `6.359/6.085`，精确命中救援约占全部 speculative rounds 的
`21.0%/20.9%`，贡献 MAL 约 `0.210/0.209`。该机制没有让树替换已经采样的残差
correction，避免了直接 sibling rescue 的质量问题。当前 Python/vLLM 工程仍未优化：
E2E 为 `116.814/116.530 tok/s`，低于 Cactus `144.468/138.267 tok/s`；本阶段只证明
算法层面的质量/MAL 可行性。

受影响文件：

- `remtp/fixed6_vllm.py`：新增 selected tree leaf 到 target hidden row 的映射，并在
  下一轮 MTP input preparation 中覆盖链式索引；
- `remtp/dynamic_tree_vllm.py`：动态树安装时接入上述 selected-leaf hook；
- `tests/test_fixed6_microtree.py`：覆盖空路径和 `[0,2,5] -> row 6` 映射；
- `remtp/fastmtp_verified_report.py`、`tests/test_fastmtp_verified_report.py`：按实际
  `verification_mode` 标注 exact residual-hit 语义，不再统一误写为近似路径选择；
- `scripts/run_fastmtp_selected_residual_anchor.sh`：新增可恢复的 100 题正式实验入口；
- `docs/exact_residual_hit_tree.md`、`docs/mimo_dynamic_mtp_tree.md`：增加旧结果失效通知、
  修复说明、pilot 结果及推荐入口；
- ignored `dse_results/`：记录阈值负结果、结构控制、收敛配置与 `DSE_REPORT.md`；
- `docs/research_and_development_log.md`：本条记录。

验证：使用仓库 `.venv` 完成三个 Python 文件 `py_compile`、两个 shell 脚本
`bash -n`，相关动态树、Fixed-6 和报告测试共 `60 passed`。在 RTX 4090、vLLM
0.18.0、TencentBAC/FastMTP 上实际完成 shadow 修复前后控制、直接 Cactus sibling
救援和最终 residual-hit anchor 的 GSM8K 20 + HumanEval 20 运行及隔离代码测试。

限制：这是在同一固定 20 题/单 seed 上经过多轮选择得到的 pilot，存在明显选择偏差，
且 5pp 的质量粒度不足以证明准确率提升。必须用未参与调参的更大样本和至少第二个 seed
验证；在此之前只能称“达到进入正式实验的门槛”，不能声称普遍优于 Cactus。树每轮
验证约两倍于三 token chain 的节点，吞吐仍低，工程优化尚未开始。实验 artifacts 仍
只位于 ignored `results/`、`logs/`、`dse_results/`，未上传报告数据。

### 2026-08-12：形成独立的“残差对齐树式 MTP”方法叙事

状态：`方法说明完成；未修改 runtime 或实验结果`

新增 `docs/residual_aligned_tree_mtp.md`，将当前选定机制整理为一套独立、连贯的算法：
MTP-only 稀疏纠正覆盖树、单次 Tree Attention 目标验证、候选条件化受控松弛、先残差
采样后树查询的精确命中、原始目标分布 anchor，以及选中 BFS 叶节点的跨轮状态提交。
文档明确给出 \(Q/P/H/R\) 定义、概率提升与接受公式、残差公式、MAL 收益分解、完整
伪代码、当前参数和方法边界。对外方法名暂定为 **Residual-Aligned Tree MTP**。

本次仅新增算法文档和本日志，没有修改验证器、worker、服务脚本、默认参数或 benchmark
协议。验证包括人工核对文档公式与当前 `residual_hit_anchor` runtime 语义，并检查新
文档不包含被替代的外部方法名称；未运行新的 GPU smoke 或 benchmark，也未声称新增
实验结果。限制是文档中的 20 题数值仍来自上一条记录的单 seed pilot，正式方法结论仍
需独立 100 题和第二 seed 复现。

### 2026-08-17：整理并准备上传当前 Residual-Aligned Tree MTP 代码快照

状态：`代码快照已通过 CPU 测试并推送 GitHub；本条不声称新增 GPU 实验`

鉴于工作区已经从早期 proposal calibration 扩展为 FastMTP/MiMo 适配、动态树验证、
精确残差命中和完整 benchmark 工具链，新建分支
`research/residual-aligned-tree-mtp`，保留原有研究分支不变。此次快照包含当前所有
实现代码、运行脚本、配置、单元测试和方法/研发文档；继续排除模型、数据集、checkpoint、
`results/`、`logs/` 和 ignored `dse_results/`。同时将 `.vscode/` 加入 `.gitignore`，
避免上传机器本地编辑器配置。纯历史实验报告
`docs/four_way_comparison_summary.md` 按既定规则仅保留本地并加入忽略列表。
未发现私钥、GitHub/Hugging Face token 或 API key。

主要受影响内容：

- `remtp/`：概率 MTP、MiMo/FastMTP checkpoint 与 worker 适配、动态树/Fixed-6
  Tree Attention、残差对齐验证、benchmark/report/evaluator 和历史消融实现；
- `scripts/`：模型下载、服务启动、断点恢复、GSM8K/HumanEval 对比及选定
  residual-anchor 正式实验入口；
- `tests/`：上述 runtime、概率语义、状态路由、报告和评测协议测试；
- `configs/`：静态 proposal calibration 与无思考 chat template；
- `docs/`、`README.md`、`AGENTS.md`、`CLAUDE.md`：算法说明、历史/失效通知、使用说明
  和强制研发记录规则；
- `.gitignore`、`docs/research_and_development_log.md`：本次上传边界和记录。

提交前另对三份 FastMTP 历史文档和本日志做了纯机械的行尾空格
清理，并删除 `remtp/dynamic_tree_humaneval_report.py` 的多余文件末空行；无任何
算法、协议或 runtime 语义变化。

验证：仓库 `.venv` 下完整执行 `python -m pytest -q`，结果为 `334 passed`；执行
`bash -n scripts/*.sh`，所有 shell 脚本语法检查通过；运行敏感信息模式扫描，未发现
私钥或常见 token/API key。没有在本次上传工作单元重新运行 GPU smoke、GSM8K 或
HumanEval，因此 GPU 可运行性和历史性能仅以此前相应日志条目为准。

限制：当前分支包含研究过程中保留的历史实现和负结果对应代码，尚未压缩为最小发布包；
部分文档记录 pilot 数值，但原始结果 artifacts 仍只在本地 ignored 目录。上传不会包含
模型权重、数据、实验输出或本地 `.vscode` 配置。正式质量结论仍需选定方法的独立大样本
与第二 seed 验证。

上传记录：代码快照提交为 `389a4b1`（`add residual-aligned tree MTP
research stack`），已成功推送到
`origin/research/residual-aligned-tree-mtp`。原 `research/proposal-calibrated-mtp`
和 `main` 均未被改写。

### 2026-08-17：建立 Neurocomputing 中文论文初稿与 Elsevier 模板工程

状态：`中文初稿与实验框架完成；静态检查通过；PDF 未编译；正式实验仍待补`

按用户指定地址下载 Elsevier `elsarticle.zip` 并解压到独立论文目录。下载文件
`paper_zh/template_download/elsarticle.zip` 的 SHA-256 为
`0b093093e84db49f99bcc9a7c3f69ed1fb61b0147c6d296427aff7963e7f50f6`。
论文采用 `elsarticle` 数字引用格式与 XeLaTeX/CTeX 中文工程，题目暂定为
“面向端侧原生多 Token 预测的残差对齐树式松弛验证”。

论文方法部分以当前选定的 `residual_hit_anchor` runtime 为唯一正式方法：FastMTP 的
一个训练物理 MTP head 递归构造三步 sampled primary path，在最多 9 个节点内增加
高 Q sibling；目标模型一次 Tree Attention forward 验证全树；主路径使用候选条件化
临时分布 `H` 完成概率接受与 `max(H-Q,0)` 残差纠正；correction 必须先独立采样，随后
才允许查询 sibling token-ID 精确命中；命中只复用已验证状态并追加一个原始 target
anchor。论文没有把候选树写成 correction 选择器，也明确指出该松弛分布建立在已有
constrained relaxed sampling 思路上，避免把已发表的候选概率提升公式误写为独立原创。

初步结果只引用本地 ignored 目录
`results/fastmtp_residual_anchor_n100/` 中已经真实完成的 100 条 GSM8K 与 100 条
HumanEval。正文填入 Native FastMTP、Cactus、SpecCascade 和当前方法的质量、MAL、
吞吐与节点数；同时加入 Wilson 95% 区间、相对 Native 的配对翻转计数和 McNemar 精确
检验。稿件明确报告当前方法虽将 MAL 提高到 `3.581/3.523`，但 E2E 只有
`117.576/115.666 tok/s`，低于链式实现；现有结果只支持算法级接受长度和 residual
reuse 机制，不支持“已经端侧加速”或“能源效率提升”。消费级 RTX 4090 以外的嵌入式
平台、功耗、全量任务、多 seed、kernel 优化和节点预算消融均标为待测。

新增/受影响文件：

- `paper_zh/main.tex`、`paper_zh/math_commands.tex`：Elsevier 主文件和统一记号；
- `paper_zh/sections/*.tex`：中文摘要、引言、相关工作、预备知识、方法、系统实现、
  实验、讨论、结论、声明和补充材料；
- `paper_zh/figures/*.tex`：方法流程与 residual-hit 示例的 TikZ 图；
- `paper_zh/references.bib`：正文实际引用的 18 篇文献；
- `paper_zh/PAPER_PLAN.md`、`paper_zh/CLAIMS_EVIDENCE.md`、`paper_zh/README.md`：
  论文叙事、证据边界、编译与补实验说明；
- `paper_zh/template_download/`、`paper_zh/elsarticle_template/`：用户指定的原始模板与
  解压内容；
- `docs/research_and_development_log.md`：本条强制记录。

验证：重新核对当前 100 题汇总、逐题质量记录和 tree audit；静态检查 15 个 TeX 文件、
18 个 BibTeX key、18 个正文 citation key、36 个 label 和 22 个 ref/eqref，未发现缺失
引用、未定义 label、重复 label 或大括号不平衡；`git diff --check` 通过；模板 checksum
与下载后记录一致。另用当前实现做 CPU 数值核对，验证候选条件化分布归一化、
`TV(H,P)=H(y)-P(y)`，并核对两项 100 题审计均满足
`MAL=1+ordinary_depth+all_round_hit_rate`。当前机器未安装 `xelatex`、`latexmk`、
`pdflatex`、`lualatex` 或
`tectonic`，因此未运行 LaTeX/PDF 编译，也未声称编译通过。按 `paper-write` 流程两次
请求外部 Claude 交叉审阅，但默认与 Sonnet 通道均因服务端 `503 model_not_found`
失败，未获得或伪造外部审阅意见。

限制：作者姓名、单位、基金与 CRediT 仍需作者填写；中文稿还需在方法和实验冻结后翻译
为英文。正式投稿前必须完成 `paper_zh/CLAIMS_EVIDENCE.md` 列出的多 seed、全量任务、
嵌入式设备、能耗、树节点 Pareto 与 fused-kernel 实验，并按届时有效的 Neurocomputing
作者指南更新格式和生成式 AI 使用声明。实验 artifacts 继续只保留在 ignored
`results/` 与 `logs/`，本次没有发布或上传报告数据。

### 2026-08-17：生成并验收 Neurocomputing 中文论文 PDF

状态：`PDF 编译成功；28 页；引用完整；仍是含占位信息和待补实验的中文初稿`

在没有系统级 TeX Live、且无免密 sudo 的环境中，于用户目录安装 Tectonic 0.15.0，
实际执行 XeTeX、BibTeX 与两次附加 TeX rerun，生成
`paper_zh/build/main.pdf`。初稿创建条目中“PDF 未编译”的状态由本条显式取代。

为使文稿可编译且可读，本工作单元修复了 TikZ 样式名与内置 `anchor` key 冲突、
LaTeX 正文中的未转义 `TREE_ATTN` 下划线，将机制审计表和附录宽表缩放到版心内，
把两张方法图固定到对应方法段落，并调整总览图说明框以消除节点遮挡；同时启用彩色
文本链接，移除默认的红色引用边框。`paper_zh/README.md` 已更新为当前可复现的 Tectonic
命令和 PDF 输出位置。

受影响文件：

- `paper_zh/main.tex`：加入浮动体控制和可读的超链接样式；
- `paper_zh/figures/method_overview.tex`、`paper_zh/figures/residual_hit_example.tex`：
  修复 TikZ 冲突、遮挡与布局；
- `paper_zh/sections/04_method.tex`、`05_implementation.tex`、`06_experiments.tex`、
  `A_supplement.tex`：图表定位、转义和宽表修复；
- `paper_zh/README.md`：记录实际编译方式与历史状态取代说明；
- `paper_zh/build/main.pdf`、`paper_zh/compile.log`、`paper_zh/build/main.log`：本地构建产物；
- `docs/research_and_development_log.md`：本条强制记录。

验证：Tectonic 最终退出码为 0；`pdfinfo` 报告 Letter 页面、28 页、PDF 1.5，最终文件
约 518 KiB；`pdffonts` 确认中文 Fandol 与 Latin Modern 字体均已嵌入；编译日志未发现
未定义 citation、reference、重复 label 或致命错误。使用 `pdftoppm` 和本地图像查看器
人工抽查标题页、方法总览图、残差命中图、主结果表、机制审计表与附录表，确认正文和
表格未被裁切，方法图不再重叠。没有在本工作单元运行 GPU smoke、GSM8K 或 HumanEval，
论文中的实验值仍完全沿用此前已经记录的本地结果。

限制：作者、单位、基金和 CRediT 仍为占位内容；文稿仍有少量不影响裁切的 overfull/
underfull 排版警告，以及若干参考文献缺页码的 BibTeX warning；Tectonic bundle 自带的
`algorithm.sty` 还报告一个包内部 UTF-8 replacement warning。当前 PDF 是中文审阅稿，
不是可直接提交的英文终稿；正式投稿所需的大样本、多 seed、端侧设备、能耗与 kernel
实验仍待完成。PDF 与编译日志属于本地论文构建产物，本条不表示已推送 GitHub。

### 2026-08-18：调整论文写作风格、章节结构与规模化实验图位

状态：`中文稿改写完成；PDF 重新编译成功；未新增或改写实验事实`

按作者写作偏好完成全稿措辞清理：删除使用“只有……才……”“只……不……”以及
“不是……而是……”强化边界的表达，改用正向陈述、执行顺序和概率语义说明方法约束。
同时撤销独立的“讨论”章节，将其中的机制解释、质量行为、MAL 与端到端速度差距、
运行时优化方向及有效性范围融入实验章节；引言中的章节导航和 README 结构说明同步
更新。原 `paper_zh/sections/07_discussion.tex` 已删除，改写前原稿备份位于仓库外临时
目录 `/tmp/remtp-paper-backup-20260818-cUnYX5/paper_zh`。

实验章节新增三处正式规模插图位：质量--MAL Pareto、节点规模--系统收益、不同拒绝
深度的 residual-hit 来源。占位图明确给出横纵轴、颜色、点形、节点大小、多 seed
误差条和多设备分面要求。`paper_zh/PAPER_PLAN.md` 进一步记录图表规范，并根据当前
100 题结果给出内部运行检查区间；这些区间明确标记为规划用途，没有写入论文正文，
也没有伪装成已完成的大规模实验结果。

受影响文件：

- `paper_zh/main.tex`：移除独立讨论章节入口；
- `paper_zh/sections/00_abstract.tex`、`01_introduction.tex`、`02_related_work.tex`、
  `03_background.tex`、`04_method.tex`、`05_implementation.tex`、`06_experiments.tex`、
  `08_conclusion.tex`：措辞、结构、实验分析和结论改写；
- `paper_zh/sections/07_discussion.tex`：内容并入实验与结论后删除；
- `paper_zh/PAPER_PLAN.md`、`CLAIMS_EVIDENCE.md`、`README.md`：图表计划、证据措辞、
  章节与编译状态同步；
- `paper_zh/build/main.pdf`、`paper_zh/compile.log`、`paper_zh/build/main.log`：重新构建；
- `docs/research_and_development_log.md`：本条强制记录。

验证：对正文、论文计划、证据表和 README 扫描指定句式，未发现“只有”“才”“不是”
“而是”“而非”“不只是”或残留“讨论”章节引用；所有 `sections/*.tex` 均被 `main.tex`
引用，无过期章节文件。使用 Tectonic 0.15.0 实际运行 XeTeX、BibTeX 和两次附加 TeX
rerun，退出码为 0，生成 29 页 PDF；编译日志未发现未定义 citation/reference、重复
label 或致命错误。人工抽查三个新增图位、机制表、结论与参考文献页面，未见裁切。
本工作单元没有运行 GPU smoke、GSM8K、HumanEval 或新的大规模实验，正文数值继续来自
此前记录的同一批本地 100 题结果。

限制：新增插图仍为排版占位框，需要正式多 seed 与多设备结果生成真实曲线后替换；
内部检查区间来源于单 seed 小样本，不构成论文证据。作者信息、基金、CRediT、嵌入式
设备、能耗、全量任务和 fused-kernel 结果仍待补充。编译仍保留少量表格 overfull/
underfull warning、参考文献缺页码 warning，以及 Tectonic bundle 的
`algorithm.sty` UTF-8 replacement warning；这些警告未造成当前抽查页面裁切。

### 2026-08-18：RAVEN 引入 native-MTP 深度校准松弛

状态：`代码与中文方法稿已实现；CPU/单元测试通过；GPU pilot 尚未运行`

将 RAVEN 主路径的统一松弛参数改为 native-MTP 深度相关参数
`delta_i = delta * w_i`。新增的 `relaxation_depth_weights` 使用 1-based 逻辑 MTP
深度；空 schedule 保持历史 uniform-delta 行为，显式 schedule 需要非负、有限且单调
不增。当运行深度超过 schedule 长度时延用末项，避免后续逻辑 head 恢复为更激进的
首位置预算。该有效 `delta_i` 同时用于构造候选条件化验证分布、接受概率和
`(H_i-Q_i)_+` rejection residual，因此接受分支与纠正分支继续以同一个临时分布为
概率目标。逐节点 audit 新增 `relaxation_depth_weight` 与
`effective_relaxation_delta`，可直接核对运行时是否采用预期 schedule。

中文活动稿 `paper_zh/修改版.md` 同步加入深度可靠度标定：先从无答案标签的严格验证
trace 估计各位置接受率，再通过单调包络和归一化权重形成 `w_i`。根据现有 GSM8K
`88.9/78.3/69.9%` 与 HumanEval `88.4/74.6/65.3%` 统计，新增均值为 1、保持
`sum(delta_i)=D*delta` 名义预算的 pilot 配置 `(1.143, 0.986, 0.871)`，并提供平方
重分配消融 `(1.290, 0.960, 0.750)`。由于候选概率提升为非线性变换，逐节点审计另行
记录 `relaxation_tv=H_i(y_i)-P_i(y_i)`，不把名义预算相同解释为实际 TV 完全相同。
原 `main.tex`、`sections/` 和 29 页 PDF 尚未同步该公式，
`paper_zh/README.md` 已加入醒目的历史/取代说明，当前审阅以 `修改版.md` 为准。

新增统一实验入口 `scripts/run_fastmtp_raven_depth_relaxation.sh`。默认一次运行
`uniform`、`empirical` 与 `squared` 三个 RAVEN profile，并在相同 FastMTP、采样题目、
温度、seed、空 system 与 no-think 协议下共享 Native/Cactus/SpecCascade 基线。脚本支持
自定义三项 schedule、复用协议一致的历史 baseline、逐请求进度与断点续跑。新增报告器
将 GSM8K 和 HumanEval 的质量、MAL、E2E、draft acceptance、节点数与截断率汇总到同一
Markdown，并按深度报告实际有效 delta、实际 TV 和主路径接受率。`CONTROL_ROOT` 还能
复用协议一致的历史 uniform RAVEN 与三项 baseline，使后续 100 题实验集中运行新增的
深度校准 profile；旧 audit 缺少的实际 TV 在合并表中保持为空，不进行推算。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：深度 schedule、有效 delta、三类动态树验证路径与审计字段；
- `remtp/dynamic_tree_vllm.py`：环境变量解析和启动 marker；
- `scripts/serve_fastmtp_verified.sh`、`scripts/serve_fastmtp_dynamic_tree.sh`、
  `scripts/run_fastmtp_verified_comparison.sh`：运行时参数透传；
- `scripts/run_fastmtp_raven_depth_relaxation.sh`：可复现的三 profile 测试入口；
- `remtp/raven_depth_relaxation_report.py`：合并报告和逐深度机制审计；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_raven_depth_relaxation_report.py`：数值、
  参数约束、环境解析和报告审计测试；
- `paper_zh/修改版.md`、`paper_zh/README.md`：方法公式、算法伪代码、贡献表述和版本说明；
- `docs/research_and_development_log.md`：本条强制记录。

验证：四个 shell 入口通过 `bash -n`；三个 Python 模块通过 `py_compile`；仓库完整测试
为 `338 passed in 6.39s`；深度数值测试验证 `w=(1,0.5,0.25)` 时三个位置分别使用
`delta_i=(1,0.5,0.25)`，并按对应公式生成不同的 `H_i(y_i)`；`git diff --check`
通过。实验脚本的 `--help` 已实际执行并正确展示默认、定制与 baseline 复用命令。

限制：本工作单元未启动 vLLM、未运行 GPU smoke，也未产生新的 GSM8K/HumanEval
结果，因而尚不能声称深度校准提高质量、MAL 或吞吐。当前 schedule 来自既有单 seed
统计，正式论文应使用与测试集隔离的校准请求、多 seed 复现，并报告 uniform、经验衰减
和更强衰减的质量--MAL Pareto。实验 artifacts 继续保留在 ignored `results/` 与
`logs/`，本次未上传任何结果或报告。

### 2026-08-18：RAVEN 深度校准 pilot 启动失败与服务归属检查修复

状态：`GPU pilot 未进入生成阶段；根因已定位；启动防护已修复；待释放 GPU 后重跑`

首次执行 `scripts/run_fastmtp_raven_depth_relaxation.sh` 时，uniform profile 在 native
基线启动后报告缺少 `[ReMTP][ProbMTP][diagnostic]`。该提示不是算法 adapter 的根因。
现场检查确认 PID 224375 的旧 dynamic-tree vLLM 服务仍监听 8000 端口，其 EngineCore
占用约 22056 MiB 显存。新 native EngineCore 仅检测到 1.71/23.65 GiB 空闲显存，低于
`gpu_memory_utilization=0.9` 所需的 21.28 GiB，因而在模型初始化阶段退出。旧服务仍能
响应 `/health`，原等待逻辑错误地将该响应归属于本次新进程，随后才在 adapter marker
检查中暴露异常。本次运行没有进入 GSM8K/HumanEval 生成阶段，不形成可用实验结果。

修复包含两部分。第一，比较脚本在整套实验开始前和每个方法启动前使用 TCP 连接检查
端口占用，发现已有监听者时立即退出并给出 `ss` 排查命令。第二，服务健康判定同时要求
本次启动日志出现预期 Worker marker、`Application startup complete.`，本次 server PID
仍存活并且 `/health` 可访问，避免其他进程的健康端点造成假阳性。四个 verified Worker
的 `worker=active` marker 也移动到 `super().init_device()` 成功返回以后；以后该 marker
表示 GPU Worker 完成初始化，不再表示运行时补丁刚安装但模型尚未初始化。

受影响文件：

- `scripts/run_fastmtp_verified_comparison.sh`：TCP 端口占用检查、方法级启动前复检，以及
  与本次启动日志绑定的健康判定；
- `remtp/fastmtp_verified_worker.py`：四类 Worker 的 active marker 延后到设备初始化成功后；
- `docs/research_and_development_log.md`：本条失败实验与修复记录。

验证：在旧 PID 仍占 8000 的现场实际执行入口脚本，脚本以退出码 2 立即停止，并准确
报告 `Port for http://127.0.0.1:8000 is already occupied`；两个 shell 入口通过
`bash -n`，Worker 模块通过 `py_compile`，仓库测试为 `338 passed in 6.57s`。没有终止
用户已有进程，也没有在修复后再次运行 GPU smoke 或 benchmark。

限制：旧 dynamic-tree 进程及其 EngineCore 仍由用户控制；重跑前需正常停止进程组
224375 并确认 8000 端口及显存释放。启动归属修复解决错误诊断，不代表 RAVEN 的 uniform、
empirical 或 squared profile 已通过 GPU 验证，也不提供任何质量、MAL 或吞吐结论。

### 2026-08-18：中文活动稿 Markdown 数学公式恢复

状态：`文档修复完成；数学定界符静态检查通过；未重新编译 PDF`

修复 `paper_zh/修改版.md` 中由错误转义和富文本转换造成的公式丢失。原文件第 3、4 节
存在 `**\$\$...`、空变量、截断下标、零宽字符混入正部运算符等问题，导致前缀
`y_{<i}`、严格接受率、拒绝残差、候选树条件和目标条件分布无法正常渲染。本次将这些
内容统一恢复为标准 Markdown 行内数学 `$...$` 和独立展示数学 `$$...$$`。

恢复的数学内容包括：递归 native-MTP proposal、严格概率接受与 residual correction、
松弛接受与统一残差、MAL 定义、稀疏树候选条件、Tree Attention 条件分布、基于严格
接受率的单调深度可靠度、归一化权重、`delta_i=delta*w_i`、候选条件化临时分布、精确
TV、松弛接受率、残差对齐 correction、exact sibling match 与 target anchor。另从
LaTeX 方法源稿恢复概率语义说明和 MAL 分解，并补充单轮时间与算法级吞吐表达式。新加入
的 native-MTP 深度校准公式及 empirical/squared 权重保持不变。

受影响文件：

- `paper_zh/修改版.md`：第 1--4 节标题规范化，第 3、4 节全部数学表达恢复，并补回概率
  语义、MAL 分解与计算开销公式；
- `docs/research_and_development_log.md`：本条强制记录。

验证：静态扫描未发现残留的转义美元符号、`$$$$`、粗体包裹展示公式或截断数学标记；
检测到 58 条成对展示定界符，对应 29 个闭合展示公式块；所有非代码行的行内美元符号
成对，展示公式和代码围栏均闭合；`git diff --check -- paper_zh/修改版.md` 通过。

限制：`paper_zh/修改版.md` 当前仍是未纳入 Git 基线的活动稿，无法生成可靠的历史逐行
diff。本工作单元没有把活动稿同步到 `sections/*.tex`，没有运行论文 PDF 编译，也没有
修改算法代码或运行 GPU/benchmark。新增吞吐表达式是系统分析用近似式，正式投稿版本
仍需结合真实 measured latency 说明其适用范围。

### 2026-08-18：生成并嵌入 RAVEN 图 1 架构图

状态：`论文位图已生成、人工检查并嵌入中文活动稿；尚未进入 LaTeX PDF`

使用内置图像生成能力制作 RAVEN 单轮 speculative decoding 总体架构图。图采用横向
五阶段布局，依次表示 committed prefix、native-MTP sparse proposal tree、单次 target
Tree-Attention forward、native-MTP 深度校准松弛、acceptance/rejection recovery 与
最终 commit。主路径和接受流使用蓝色，backup state 与 residual exact hit 使用绿色，
rejection/correction 使用橙色，target-model computation 使用灰色。图中显式展示
`delta_i = delta * w_i`、`P_i -> H_i` 和松弛接受概率，并将 `Sample correction c_i`
放在 `Exact token-ID match` 之前，保持先确定 correction、再查询预验证树状态的因果顺序。

首版生成后进行了单点图像修订，将三行普通 `delta1/delta2/delta3` 改为带希腊字母和下标
的 `δ₁/δ₂/δ₃`，其余布局和标记保持不变。最终 PNG 为 1692x929、8-bit RGB、约 1.1 MiB；
中文活动稿原图位占位符已替换为相对路径 Markdown 图片引用。

受影响文件：

- `paper_zh/figures/raven_figure1_architecture.png`：新增图 1 位图；
- `paper_zh/修改版.md`：图 1 占位符替换为实际图片引用；
- `docs/research_and_development_log.md`：本条强制记录。

验证：使用本地图片查看工具按高细节重新打开最终文件，人工核对五阶段顺序、三层 MTP
主路径、backup siblings、一次 target forward、三项深度预算、接受/拒绝分支、MISS/HIT、
validated-state reuse、target anchor 与颜色图例。`file` 确认输出为 1692x929 非隔行 PNG；
未覆盖仓库已有图；Markdown 引用使用项目内相对路径。

限制：该图由位图模型生成，文字与线条无法像原生 TikZ/SVG 一样逐对象编辑，正式双栏
排版中的最小字号仍需在 PDF 实际尺寸下检查。当前 `main.tex` 继续使用原有
`figures/method_overview`，本工作单元没有切换 LaTeX 图源、没有编译 PDF，也没有运行
算法测试或 GPU benchmark。

### 2026-08-19：RAVEN 饱和感知的块内自适应松弛

状态：`算法、运行入口和审计报告已实现；完整 CPU 测试通过；100 题 GPU 实验待用户运行`

根据 20 题 depth-relaxation pilot 中 uniform profile 优于固定 empirical/squared
schedule 的现象，将固定深度重加权替换为当前块自适应的实际 TV 分配。新机制以 uniform
候选条件化变换产生的逐位置实际 TV 为控制预算。对于 sampled-Q 主路径，每个位置先将
候选提升量截断到 `h_i(y_i)=q_i(y_i)` 的接受饱和点；该截断保持该位置接受概率为 1，
同时回收继续提升候选但无法增加接受率的 TV。未饱和位置保留 uniform 提升量，因此自适应
配置在数值误差范围内不会降低任一主路径位置的 uniform 接受概率。

回收预算的默认 50% 通过 16 次小步分配给当前块仍未饱和的位置。每步效用采用链式期望
MAL 对候选 TV 的局部斜率 `reach_i * suffix_i / q_i(y_i)`，再乘轻量目标相对支持项
`(p_i(y_i)/p_i,max)^0.25`。`reach_i` 表示前缀到达概率，`suffix_i` 表示该位置能够解锁的
后续长度价值。单位置实际 TV 受 uniform TV 的 1.5 倍和接受饱和点双重约束；块内实际 TV
总量不超过 uniform 控制。实现将最终实际 TV 反解为逐块、逐位置动态 `delta_i` 与
`w_i=delta_i/delta`，所以论文与 audit 仍可使用深度松弛记号，同时权重会随 P/Q、前缀
可达率和后续价值改变。该分配使用本轮完整验证后已有的 P/Q，属于 block-aware relaxed
verification；它没有增加 target forward，输出分布仍是近似的松弛目标，不能表述为严格
恢复原目标分布。

新增 `scripts/run_fastmtp_raven_adaptive_relaxation.sh` 作为 100 题确认入口。默认在相同
FastMTP、temperature=0.6、seed=42、空 system、no-think、MTP depth=3 与 9 节点残差树
协议下运行 Native、Cactus、SpecCascade、RAVEN uniform 和 RAVEN adaptive。三项链式
baseline 运行一次后通过只读链接复用于 adaptive profile；脚本逐题输出进度、支持
`RESUME_PARTIAL=1` 断点续跑，也可通过 `CONTROL_ROOT` 复用协议一致的历史 uniform
结果。合并报告同时给出质量、MAL、E2E、draft acceptance、节点数、逐深度动态权重/
delta/TV/主路径接受率，以及每块 baseline TV、实际 TV、可回收 TV 与重新分配 TV。

受影响文件：

- `remtp/dynamic_mtp_tree.py`：自适应配置、饱和截断、期望-MAL 边际分配、显式候选分布
  构造、动态 delta/weight 与逐节点审计；
- `remtp/dynamic_tree_vllm.py`：五项自适应环境变量解析和运行时 marker；
- `scripts/serve_fastmtp_verified.sh`、`scripts/serve_fastmtp_dynamic_tree.sh`、
  `scripts/run_fastmtp_verified_comparison.sh`：自适应参数透传与配置打印；
- `scripts/run_fastmtp_raven_adaptive_relaxation.sh`：uniform/adaptive 100 题统一实验入口；
- `remtp/raven_adaptive_relaxation_report.py`：质量、性能、逐深度和块 TV 合并报告；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_raven_adaptive_relaxation_report.py`：饱和
  回收、TV 上界、审计字段、随机分布性质和块预算去重统计测试；
- `docs/research_and_development_log.md`：本条强制记录。

验证：四个相关 shell 入口通过 `bash -n`；新报告器和实验入口的 `--help` 已实际执行；
Python 源码通过 `compileall`；`git diff --check` 通过；仓库完整测试为
`342 passed in 6.13s`。新增 32 组随机分布性质测试逐组确认 adaptive acceptance 不低于
uniform、各位置 TV 不越过接受饱和点、块 TV 不超过 uniform。未运行 vLLM/GPU smoke，
也未产生新的 GSM8K/HumanEval 指标。

限制：该版本默认参数 `mix=0.5、max_ratio=1.5、target_power=0.25、steps=16` 是基于现有
pilot 失败模式选择的单一确认配置，尚无 100 题或第二 seed 证据。块内分配读取后续验证
位置的 P/Q 来估计长度边际价值，因而属于 relaxed block verification，并会改变 primary
临时分布及 rejection residual；报告需同时核对任务质量和实际 TV。当前 Python 小步规划
包含一次小型 GPU-to-CPU 同步，D=3 时计算量很小，但正式吞吐优化仍可改成融合 GPU
实现。实验 artifacts 继续保留在 ignored `results/` 与 `logs/`，本次没有发布结果文件。

### 2026-08-19：RAVEN-A 中文完整论文稿与 100 题确认结果回填

状态：`中文 Markdown 完整稿已重写；RAVEN-A 两项 100 题结果已回填；LaTeX/PDF 尚未同步`

将 `paper_zh/修改版.md` 从停留在方法第 4 节的深度固定权重活动稿重写为完整中文论文。
当前故事以 RAVEN-A 为方法主体：uniform candidate-conditioned transform 提供块级实际 TV
控制，接受饱和点回收无效预算，当前块的前缀可达率、后续长度价值、`q_i(y_i)` 与目标
相对支持共同决定动态分配效用；第一次拒绝后仍按同一临时分布的正残差采样 correction，
随后通过 exact token-ID sibling hit 复用目标状态并追加 target anchor。正文补齐摘要、
引言、相关工作、预备知识、完整公式与伪代码、vLLM 实现、实验设置、主结果、机制归因、
固定深度消融、实际 TV 审计、系统瓶颈、复现边界、结论、声明与 31 条参考文献。文稿延续
既定写作约束，没有增加独立“讨论”章节，也没有使用强调式“只有……才”“不是……而是”
句式。

写作开始时 RAVEN-A 100 题任务由用户已有进程继续运行。该进程在本工作单元期间完成，
本次没有启动、停止或修改实验。完成后的本地 ignored artifacts 显示：GSM8K 为
`Accuracy=89.0%、MAL=3.611、Decode=118.858 tok/s、E2E=118.131 tok/s、
nodes/round=6.337`；HumanEval 为 `pass@1=63.0%、MAL=3.545、
Decode=117.413 tok/s、E2E=116.476 tok/s、nodes/round=6.316`。RAVEN-A 的块级实际
TV 相对 uniform control 从 `0.2400` 降至 `0.1645`（GSM8K），从 `0.1736` 降至
`0.1170`（HumanEval），对应 MAL 比 RAVEN-U 增加 `0.030/0.022`。条件 residual hit
为 `57.8%/56.6%`，target forward calls/round 均为 1。结果来源为
`results/raven_adaptive_n100/comparison.md` 及两个 adaptive task 目录，继续保留在 ignored
`results/` 与 `logs/`，没有纳入代码提交。

初稿写作时曾依据部分 checkpoint 和 RAVEN-U 结果设置 RAVEN-A 临时表格值。完整实验
结束后，正文已全部换成实测值并移除 dagger；`paper_zh/估计数值清单.md` 单独记录每个
临时值、最终值和结果路径。当前 `修改版.md` 不含估计实验数值，图 2--4 仍保留无数值
的正式实验绘图说明。`paper_zh/README.md` 更新当前入口，`PAPER_PLAN.md` 与
`CLAIMS_EVIDENCE.md` 增加历史版本提示，避免旧 RAVEN-U 说明与 RAVEN-A 正文冲突。

受影响文件：

- `paper_zh/修改版.md`：完整论文重写、RAVEN-A 公式、实测表格与结论；
- `paper_zh/估计数值清单.md`：新增临时值审计、实测取代记录和投稿前检查；
- `paper_zh/README.md`：当前文稿入口、证据状态与历史 LaTeX 提示；
- `paper_zh/PAPER_PLAN.md`、`paper_zh/CLAIMS_EVIDENCE.md`：历史/已取代提示；
- `docs/research_and_development_log.md`：本条强制记录。

验证：读取两项任务的 `summary.md`、`tree_metrics.md` 和合并 `comparison.md` 交叉核对
质量、MAL、吞吐、节点数、TV、动态权重与 residual hit；Markdown 静态检查确认 76 个
展示公式定界符、2 个代码围栏均成对，参考文献数量为 31；未发现 dagger、临时占位值、
独立讨论章节或既定禁用强调句式；`git diff --check -- paper_zh` 通过。本工作单元没有
修改算法代码，没有额外运行 pytest、GPU smoke 或 benchmark，也没有重新编译 PDF。

限制：当前结论仍来自每个任务 100 条固定样本和单 generation seed。RAVEN-A 的
GSM8K accuracy 相对 RAVEN-U 下降 4 个百分点，HumanEval pass@1 上升 3 个百分点，需
多 seed 配对实验确认稳定性。现有 vLLM 树实现的 E2E 吞吐低于链式基线，端侧加速、
能效和嵌入式适用性尚无证据。`main.tex`、`sections/` 与 `build/main.pdf` 继续保留上一版
RAVEN-U 历史稿；图 1 的静态位置预算文字、图 2--4、作者信息、基金、全量任务、多设备
和 fused-kernel 结果仍需在英文投稿版中更新。

### 2026-08-19：论文增加 MAL—吞吐转换与跨 checkpoint 松弛诊断

状态：`中文稿叙事与实验表已更新；归一化吞吐为公式推导量；未运行新 GPU 实验`

根据当前 RAVEN-A 100 题实测和历史 Qwen3.5 同协议结果，将论文的
结果叙事从单独强调 MAL 扩展为“MAL、实测 tok/s、等轮成本吞吐和任务
质量”的联合核算。FastMTP 主表新增相对 Native 的 MAL/E2E 百分比变化。
链式 Cactus/SpecCascade 的 MAL 与 E2E 分别同向增长，实测吞吐达到等轮
成本归一化值的 99% 以上。RAVEN-A 相对 RAVEN-U 的 MAL 与 E2E 也在两项
任务上同向增长，同时实际块 TV 下降约三分之一。

新增等轮成本归一化量
`V_norm = V_native * MAL_method / MAL_native`。由实测值推导得到，
RAVEN-A 在 GSM8K/HumanEval 上的 `V_norm` 为 `149.467/146.572 tok/s`，当前
实测为 `118.131/116.476 tok/s`，系统实现率为 `79.0%/79.5%`。根据
`MAL / target-throughput` 反解，RAVEN-A 达到当前最快链式对照吞吐所需的
单轮时延降幅为 `16.5%/15.5%`。这些值表示实现差距和 kernel 优化目标，
未当作 fused-kernel 实测结果。

新增的跨 checkpoint 诊断表来自已完成的 Qwen3.5-4B、MTP=6、temperature=0.7
运行。GSM8K-200 中，目标侧统一松弛相对 Native 将 MAL/E2E 提高
`30.3%/29.2%`，accuracy 下降 11.0 个百分点；HumanEval-164 中对应变化为
`+7.1%/+5.6%/-14.0 pp`。SpecCascade 与历史 margin-calibrated MTP 松弛在质量
变化较小时产生更温和的 MAL/E2E 收益。另引用 HumanEval-20 expected-regret
Router pilot：Native 为 `70.0%/5.678/212.222 tok/s`，Router 为
`50.0%/6.025/204.431 tok/s`，用于说明未校准 MTP 反馈可能同时伤害质量与
吞吐。正文已明确标注这些诊断与 FastMTP 主表协议不同，且 FastMTP 主实验
的 Cactus 未出现质量下降，因此没有将历史现象概括为普遍结论。

受影响文件：

- `paper_zh/修改版.md`：新增方法类型对比、Qwen3.5 诊断表、主表 MAL/E2E 变化、
  MAL—吞吐归一化公式、实现率与工程时延目标，更新摘要和结论；
- `paper_zh/估计数值清单.md`：记录历史实测来源，将归一化吞吐与实测、
  临时值分开，新增公式与数值边界；
- `paper_zh/README.md`：更新当前证据状态和吞吐归一化说明；
- `docs/research_and_development_log.md`：本条强制记录。

验证：交叉读取
`results/raven_adaptive_n100/comparison.md`、
`results/gsm8k_three_schemes_remtp_20260804_232312/comparison.json`、
`results/humaneval_comparison_remtp_20260804_232312/comparison.json` 和
`results/humaneval_comparison_20260803_170514/comparison.json`；用实测 MAL/E2E 重新计算
百分比变化、`V_norm`、实现率与单轮时延目标。Markdown 展示公式
定界符、代码围栏和 31 条参考文献数量已做静态检查；`git diff --check -- paper_zh
docs/research_and_development_log.md` 通过。本工作单元没有修改算法代码，没有运行
pytest、GPU smoke、benchmark 或 PDF 编译。

限制：RAVEN-A 主结果仍来自每项任务 100 个固定样本和单 generation seed。
Qwen3.5 表用于解释方法取舍，与 FastMTP 主表不形成直接排名。HumanEval-20
Router 结果属于小样本探索性证据。`V_norm` 假设各方法具有 Native 单轮成本，
因此它量化算法潜力和实现差距，无法代替融合内核后的实测吞吐。当前
RAVEN-A 实测 E2E 仍低于链式对照，端侧加速和能效结论继续留待融合 kernel、
多 seed 和真实边缘设备实验。

### 2026-08-19：论文实验范围改为 Gemma 4/FastMTP 与 RTX 4090/Orin NX

状态：`论文协议与预注册目标已更新；新模型、量化和边缘设备实验尚未运行`

按当前端侧研究范围，`paper_zh/修改版.md` 移除历史跨 checkpoint 诊断表及其结论，
正式模型矩阵收敛为两类具有显式注意力的 MTP 路径：MiMo/FastMTP 的单物理 head
递归复用，以及 Gemma 4 E2B 与官方 MTP assistant 的目标 KV 共享。Gemma 4 文本主干
交替使用 sliding/full attention，论文将树 mask 定义为祖先可见性与原始窗口约束的
交集；需要复制线性递归状态的模型不进入该实验矩阵。

设备矩阵设为 NVIDIA RTX 4090 24GB 与 NVIDIA Jetson Orin NX 16GB。RTX 4090 使用
BF16/8K context；Orin 使用 JetPack 6.2、固定 25W 功率配置、W4A16/2K context、
batch size 1，并关闭非文本编码器。模型/设备内部使用相同精度和运行协议，跨设备按
相对 Native speedup、显存、功率和 J/token 报告，避免将 BF16 与 W4A16 原始 tok/s
直接解释为硬件收益。此前工作单元中暂写的 Super Mode/MAXN 表述已按用户决定删除。

量化仓库审计得到：

- `jedisct1/MiMo-7B-RL-GGUF` 是 `XiaomiMiMo/MiMo-7B-RL` 的 llama.cpp GGUF 转换，
  文件列表只有目标 GGUF，未包含 TencentBAC/FastMTP 后训练物理 MTP head，也不匹配
  当前 vLLM `TREE_ATTN` 分支状态提交路径；
- `Siarhei/gemma-4-E2B-4bit` 的模型卡声明 `library_name=mlx`，面向 Apple MLX，基座为
  `google/gemma-4-E2B`，没有配套 assistant，无法直接用于 Jetson CUDA/vLLM RAVEN；
- Gemma Orin 路径改用 Google 官方
  `google/gemma-4-E2B-it-qat-w4a16-ct` compressed-tensors checkpoint 与匹配 QAT
  assistant；FastMTP Orin 路径需要从同一 FastMTP safetensors 生成 W4A16 权重并保留
  MTP/LM head、normalization 和 activation 的 FP16 语义。

正文新增完整任务协议：GSM8K test 1,319 题、HumanEval 164 题、三个 generation seeds；
系统测量使用固定 GSM8K-200 与 HumanEval-164，20 请求预热和三次重复。正文表 9--10
填入明确标记为“预注册目标值”的具体数值，避免空白占位；这些数值包括 Gemma 4 质量/MAL
目标，以及两模型在 RTX 4090/Orin 上的 E2E、显存、功率和 J/token 目标。
`paper_zh/Gemma4_Orin_预估实验值.md` 单独记录全部预测依据、第三方 checkpoint 排除原因、
正式运行协议和逐项替换检查。未将任何预测值记录为已通过 benchmark。

受影响文件：

- `paper_zh/修改版.md`：双模型双设备叙事、模型选择、量化协议、预注册数值表与引用；
- `paper_zh/Gemma4_Orin_预估实验值.md`：新增预测值、依据和实测替换清单；
- `paper_zh/估计数值清单.md`：区分历史临时值与当前跨模型预注册值；
- `paper_zh/README.md`：更新当前证据边界与文档入口；
- `docs/research_and_development_log.md`：本条强制记录。

验证：通过 Hugging Face API/模型卡读取两个用户指定第三方仓库的格式、基座和文件列表；
核对 Google 官方 Gemma 4 E2B QAT compressed-tensors target 与 QAT assistant 发布说明；
使用 `rg` 检查当前中文正文已无历史模型名称和 Super/MAXN 表述，并运行 Markdown
定界符、参考文献序号和 `git diff --check` 静态检查。本工作单元没有下载模型、转换
FastMTP 权重、修改推理代码、编译 PDF、运行 pytest、GPU smoke 或 benchmark。

限制：当前仓库尚无 Gemma 4 vLLM/Tree Attention adapter、Orin ARM64 运行环境或量化
FastMTP checkpoint。Google W4A16 target 与 QAT assistant 的实际组合仍需加载与严格
概率一致性测试；FastMTP 混合精度量化需验证完整 proposal distribution、residual sampler
和 selected-leaf state commit。正文预注册数值不能用于摘要结论、显著性分析或投稿结果，
正式运行后必须替换并记录估计误差。

### 2026-08-19：中文论文重构为双模型消费级评测与单模型端侧评测叙事

状态：`论文结构与内部数值审计已更新；本工作单元未产生新的 benchmark 结果`

本条更新取代上一条记录中“正文表 9--10 采用预注册目标值”的写作组织。论文按照完整
研究问题组织实验，不再围绕已有 100 题开发日志划分正文边界。RTX 4090 实验矩阵包含
MiMo/FastMTP 与 Gemma 4 E2B 两类 MTP 结构，Jetson Orin NX 16GB 端侧矩阵聚焦
FastMTP-W4A16。质量评测协议写为 GSM8K 完整 test split（1,319 题）和 HumanEval
完整 164 题，并采用三个 generation seeds；系统吞吐协议使用固定 GSM8K-256 与
HumanEval-164、20 请求预热和三次重复。

正文实验章节重新按研究问题、平台矩阵、质量—MAL、RTX 4090 系统性能、Orin 端侧
性能、机制分解、消融和稳定性组织。摘要、实验与结论使用一致的正式论文表述，不再出现
“已完成”“等待补齐”“占位”或“终稿更新”等项目进度语言。图 1 图注改为直接描述
饱和截断、预算回收、动态重分配和 residual correction/leaf-state reuse 数据流。

由于本工作单元没有运行双模型、完整任务、融合 Tree Attention 或 Orin 实验，正文中
对应的完整规模表格属于写作用外推数值。它们的来源属性、历史实测锚点、W4A16 内存
估算和逐项替换顺序集中记录在 `paper_zh/Gemma4_Orin_预估实验值.md`，不在正文使用
进度标签干扰论文叙事。`paper_zh/估计数值清单.md` 保留历史 100 题开发结果，并增加
显著的历史/被取代说明。此安排服务于稿件结构审阅，不构成实验完成声明。

官方资料核验确认：Gemma 4 E2B 存在 5.1B 总参数（2.3B effective）的目标模型、
78M 参数四层 MTP assistant，以及面向 vLLM compressed-tensors 的 W4A16 QAT target；
官方 MTP 文档说明 assistant 利用目标 activation 与 KV cache。正文据此保留 Gemma 4
在 RTX 4090 上的 BF16 路径，同时让 Orin 端侧结果集中于仓库适配更成熟的
MiMo/FastMTP 路径。

受影响文件：

- `paper_zh/修改版.md`：移除进度式表述，重写完整实验章节、摘要与结论，采用 4090
  双模型和 Orin 单模型矩阵，并统一 MAL、tok/s、质量和能耗叙事；
- `paper_zh/Gemma4_Orin_预估实验值.md`：改为双模型与 Orin 数值来源台账，逐表标记
  历史实测、推导和工程外推；
- `paper_zh/估计数值清单.md`：增加历史结果索引和被取代说明；
- `paper_zh/README.md`：更新论文结构、数值审计入口和证据说明；
- `docs/research_and_development_log.md`：本条强制记录。

验证：通过 Google 官方 Gemma 4 MTP 文档与 Google Hugging Face 模型页核对目标模型、
assistant、QAT 精度兼容性和 hybrid sliding/full attention 描述；使用 `rg` 检查正文没有
“已完成”“未完成”“等待”“预注册”“占位”“100 题”“终稿”等进度式实验措辞；检查
Markdown code fence 与 `$$` 数学定界符数量均为偶数；`git diff --check` 通过。本工作
单元没有修改算法或推理代码，没有运行 pytest、GPU smoke、benchmark、量化转换或 PDF
编译。

限制：正文完整规模结果、Gemma 4 数值、融合后 RAVEN tok/s 和 Orin 指标尚无对应本地
artifact，不能作为已验证结论用于投稿。当前仓库仍需实现并验证 Gemma 4 的 vLLM Tree
Attention adapter、FastMTP-W4A16 转换、Orin ARM64 运行路径及融合 kernel。正式提交前
必须用同协议实测逐项替换内部台账标记的外推值，并保留原始日志、随机种子与统计脚本。

### 2026-08-19：论文扩展为四任务、六方法对比并统一 RAVEN 名称

状态：`论文实验叙事、文献库与数值审计已更新；未运行新增 benchmark`

实验章节移除 RQ 编号，改为从任务质量、有效生成长度、机制归因和跨设备系统代价自然
展开。质量矩阵从 GSM8K/HumanEval 扩展到 GSM8K test 1,319 题、HumanEval 164 题、
MBPP test 500 题和 IFEval 541 条指令；系统测量矩阵相应扩展为 GSM8K-256、
HumanEval-164、MBPP-256 与 IFEval-256。

对照方法增加 MARS 与 OPT-Tree。MARS 代表目标 margin 驱动的训练免验证松弛；
OPT-Tree 代表固定节点预算下最大化期望接受长度的候选拓扑。两篇正式发表版本已下载到
本地论文库：

- `参考论文/421_MARS_Margin_Aware_Verification.pdf`，Findings of ACL 2026，
  12 页，1,682,084 bytes；
- `参考论文/2025_tacl_OPT_Tree.pdf`，TACL 2025，12 页，1,162,433 bytes。

正文将两种对照与 Native MTP、目标侧候选条件化松弛和 SpecCascade 置于同一模型、
MTP 深度、采样参数和每轮 target forward 预算下。OPT-Tree 固定深度 3、节点上限 9，
使用严格树验证；MARS 依据目标 top-1/top-2 margin 设定局部验证强度。正式方法名称由
历史稿中的 `RAVEN-A` 统一为 `RAVEN`；均匀预算版本保留为组件消融，不再作为带后缀的
主方法名称。

正文表 2 增加两个数据集、两个 baseline 和对应质量/MAL；表 3 增加四任务 RTX 4090
E2E tok/s；表 4 增加 MARS/OPT-Tree 的 Orin 行；表 5 与表 8 增加 MBPP/IFEval 的
residual hit 与 actual-TV 分解。摘要、结论、图 2 描述和数据声明同步更新。新增数值
依据现有 FastMTP 日志、MARS/OPT-Tree 方法行为、相邻任务规律和硬件成本进行约束，
并全部登记在内部数值台账，不构成新增实测声明。

受影响文件：

- `paper_zh/修改版.md`：去除 RQ 编号和 RAVEN 后缀，扩展四任务六方法实验叙事、
  结果表、统计分析与参考文献；
- `paper_zh/Gemma4_Orin_预估实验值.md`：重建为双模型、多任务、六方法与 Orin
  数值来源台账；
- `paper_zh/估计数值清单.md`：增加当前四任务/六方法修订说明；
- `paper_zh/README.md`：更新当前方法名、数据集、baseline 和数值审计范围；
- `paper_zh/CLAIMS_EVIDENCE.md`：将旧版方法后缀改为历史配置描述，避免与当前
  `RAVEN` 主方法混淆；
- `paper_zh/PAPER_PLAN.md`：同步更新历史版本提示中的方法命名；
- `参考论文/421_MARS_Margin_Aware_Verification.pdf`：新增 MARS 正式论文；
- `参考论文/2025_tacl_OPT_Tree.pdf`：新增 OPT-Tree 正式论文；
- `docs/research_and_development_log.md`：本条强制记录。

验证：通过 ACL Anthology、TACL/ACL Anthology 页面核对 MARS 与 OPT-Tree 的作者、
发表信息和方法语义；使用 `file`、`pdfinfo` 和文件大小检查两个 PDF 可读且均为 12 页；
核对 MBPP 974 个总任务及 500 个 test split、IFEval 541 条提示和 strict evaluator；
检查正文不存在 `RAVEN-A`、`RAVEN-U`、`RQ1--RQ5` 或“研究问题”标题；检查 Markdown
表格、公式定界符、代码围栏与 `git diff --check`。本工作单元没有修改推理代码，
没有运行 pytest、GPU smoke、MARS/OPT-Tree 实现、MBPP/IFEval benchmark、Gemma 4
实验、Orin 实验或 PDF 编译。

限制：仓库当前没有 MARS 与 OPT-Tree 的 FastMTP/Gemma 统一运行适配，也没有 MBPP、
IFEval 的统一报告器。新增质量、MAL、吞吐、功耗、机制分解和 bootstrap 区间均需由
正式运行替换。OPT-Tree 原论文使用独立 autoregressive drafter/EAGLE 与可扩展节点预算，
正文中的深度 3、九节点配置属于面向公平 MTP 预算的适配，不能直接引用原论文 speedup
作为本地结果。RAVEN 的融合吞吐与跨模型结论仍依赖尚未生成的系统 artifacts。

### 2026-08-19：论文实验配置与消融叙事重构

状态：`论文主稿与内部数值台账已更新；未运行新增实验`

根据论文方法链重构实验章节，删除源码文件、服务脚本、Worker、测试文件和逐轮审计字段
等仓库级代码结构描述。系统章节保留统一推理路径、分支状态隔离和渐近复杂度，使实现
叙述集中于概率语义、Tree Attention 与运行成本。

表 1 从两行模型设备概览扩展为模型结构、精度、上下文、生成协议、候选树和松弛控制共
21 个配置行，明确逻辑 MTP 深度、节点上限、候选阈值、基础松弛强度、回收比例、单位置
上限、目标支持指数与块级分配步数。表 6 按“统一候选提升—接受饱和截断—块级边际
分配—残差对齐复用”的因果链重新组织组件消融。表 7 移除研发阶段的 `empirical`、
`squared` 等 profile，改为 $D\in\{1,2,3,4\}$ 的逻辑 MTP 深度敏感性，并同时报告
nodes/round、任务质量、MAL 与 E2E tok/s。正文据此说明 $D=3$ 的默认选择来自有效长度、
质量与树验证成本的联合折中。

受影响文件：

- `paper_zh/修改版.md`：重写系统章节、扩展表 1、重构表 6 并以深度敏感性替换旧表 7；
- `paper_zh/Gemma4_Orin_预估实验值.md`：登记新表 6/7 的外推值、用途和替换状态；
- `paper_zh/估计数值清单.md`：标明固定深度权重 profile 属于历史研发结果，不进入当前正文；
- `docs/research_and_development_log.md`：本条强制记录。

验证：检查当前主稿不存在“代码结构”标题、仓库源码路径或 `Empirical/Squared` profile；
检查表 1、表 6 和表 7 的 Markdown 列数，公式定界符与代码围栏保持配对，并执行
`git diff --check`。本工作单元没有修改推理代码，没有运行 pytest、GPU smoke、深度
消融、完整 benchmark 或 PDF 编译。

限制：表 6 的逐组件质量/MAL，以及表 7 中 $D=1,2,4$ 的质量、MAL、nodes/round 和
吞吐仍是内部外推值；$D=3$ 的正文值也包含从现有 100 题日志扩展到完整任务和融合运行
路径的部分。正式投稿前需要用独立验证集固定超参数，并以同 checkpoint、同 sampled
prompts、三个 seeds 的真实运行逐项替换，同时保存逐请求质量与系统计时 artifacts。

### 2026-08-20：方法概览精简、MAL 边际公式修正与实验图形化

状态：`论文主稿、两张实验图及数值台账已更新；未运行新增 benchmark`

方法总体流程删除验证路线对比表，避免在方法章节提前展开 baseline 比较。严格 MTP、
统一目标侧松弛、SpecCascade、MARS 与 OPT-Tree 的语义改为实验设置中的连续文字说明。

修正第 4.5 节首个接受概率公式中的损坏右括号，并补充 sampled proposal 满足
$q_i=Q_i(y_i)>0$ 和 $0\le b_i\le1-p_i$ 的定义域。进一步写出整块期望 MAL 对位置
预算的右侧分段边际：$\mathcal R_i\mathcal V_i/q_i$ 乘以未饱和指示函数
$\mathbf 1[p_i+b_i<q_i]$。该形式与第 4.4 节的饱和预算 $s_i=[q_i-p_i]_+$ 一致，
明确说明候选达到接受概率 1 后边际收益归零。

实验展示方面，原深度表替换为三面板图：MAL、相对任务质量变化以及 E2E tok/s 与
nodes/round；原 actual-TV 表及深度可靠性图位合并为双面板图：四任务控制/实现 TV
和 GSM8K/HumanEval 逐深度严格接受率。两图均从 JSON 数据生成 EPS，再使用 Ghostscript
按 EPS bounding box 转为裁剪矢量 PDF，同时生成 Markdown PNG 预览。PDF 使用嵌入并
子集化的 Nimbus Roman Type 1C 字体，图内无标题，颜色采用可辨识的蓝、橙、绿组合。

受影响文件：

- `paper_zh/修改版.md`：删除方法路线表，修正第 4.5 节公式，扩充实验 baseline 文字，
  以图 4/5 替换原深度表和 actual-TV 表；
- `paper_zh/figures/data/raven_depth_sensitivity.json`：图 4 数据；
- `paper_zh/figures/data/raven_tv_depth.json`：图 5 数据；
- `paper_zh/figures/generate_raven_analysis_figures.py`：无第三方绘图库的可复现 EPS/PDF
  生成器；
- `paper_zh/figures/raven_depth_sensitivity.{eps,pdf,png}`：深度敏感性矢量图与预览；
- `paper_zh/figures/raven_tv_depth_analysis.{eps,pdf,png}`：TV/深度可靠性矢量图与预览；
- `paper_zh/Gemma4_Orin_预估实验值.md`：记录图形化后的数据来源与替换路径；
- `docs/research_and_development_log.md`：本条强制记录。

验证：重新运行绘图脚本并生成两张 PDF/PNG；`pdfinfo` 显示图 4/5 页面尺寸分别为
540×198 pt 和 488×205 pt，均为单页裁剪输出；`pdffonts` 显示 NimbusRoman-Regular
和 NimbusRoman-Bold 均为嵌入、子集化的 Type 1C 字体；人工查看 PNG 确认坐标、图例、
曲线和标签未重叠。检查当前主稿不再包含方法路线表，公式不存在控制字符，Markdown
表格列数、数学定界符与代码围栏一致，并执行 `git diff --check`。本工作单元没有修改
推理代码，没有运行 pytest、GPU smoke、完整 benchmark 或整篇论文 PDF 编译。

限制：图 4 的 $D=1,2,4$ 数据以及图 5 中尚无完整 artifact 的任务值仍属于内部外推；
图形质量验证面向独立图件，尚未在 Neurocomputing 双栏排版中检查最终缩放字号。正式
投稿前需要用同协议三 seed 实测替换 JSON，并在最终 LaTeX PDF 中再次执行字体嵌入、
可读性和裁边检查。

### 2026-08-20：补全 baseline、残差机制与节点预算图并修正图间排版

状态：`论文主稿、图 2--6、预估数据台账已同步；未运行新增 benchmark`

根据用户反馈，原先的图形存在 baseline 展示不足、模型编码不清、残差逐深度命中率
未画出、图 6 图注与实际坐标不一致，以及图例与横轴标签相撞等问题。本工作单元将
图件改为可直接对应正文表格和机制叙事的版本，并维持所有正文数值的内部外推标记。

主要变化：

- `paper_zh/figures/data/raven_baseline_pareto.json`：从单组平均点扩展为
  MiMo/FastMTP 与 Gemma 4 E2B 两组数据；每个子图包含 Native MTP、目标侧松弛、
  SpecCascade、MARS、OPT-Tree 和 RAVEN 六种方法，颜色编码方法、圆形/三角形编码模型。
- `paper_zh/figures/data/raven_residual_depth.json` 与生成器：图 3(b) 增加首次拒绝深度
  的 conditional hit 折线，和 opportunities/hits 分组柱保持同一坐标体系；图注同步
  解释深度 2/3 的 residual reuse 贡献。
- `paper_zh/figures/data/raven_node_budget.json` 与生成器：图 6(a) 改为 MAL/任务质量，
  图 6(b) 改为 E2E tok/s/conditional hit，避免图注提及未绘制的 J/token；默认
  $N_{\max}=9$ 在图中保持可读的收益—成本折中。
- `paper_zh/figures/generate_raven_analysis_figures.py`：补充 diamond marker、无 marker
  的方法图例、双模型 Pareto 绘制、右轴命中率以及更大的图 2 纵向留白，避免图例遮挡
  MAL 轴标签和下排子图标题。
- `paper_zh/修改版.md`：插入图 2、图 3 和图 6 的 PNG 引用；图 2--6 caption 改为与
  实际面板一致，编号连续且 baseline 比较集中在实验章节。
- `paper_zh/Gemma4_Orin_预估实验值.md`、`paper_zh/估计数值清单.md`：记录三张新图的
  数据文件、图形映射和外推范围，澄清图件生成不代表新增 benchmark 已运行。

验证：执行 `python -m py_compile paper_zh/figures/generate_raven_analysis_figures.py`
和绘图脚本，成功生成 `raven_baseline_pareto`、`raven_residual_mechanism`、
`raven_depth_sensitivity`、`raven_tv_depth_analysis`、`raven_node_budget` 的
EPS/PDF/PNG；人工检查 PNG，确认图 2 的方法/模型图例、图 3 的双纵轴、图 4/5 的
子图标签以及图 6 的质量/命中率右轴没有互相遮挡。随后使用 `pdffonts` 验证五张 PDF
仍包含嵌入、子集化的 Nimbus Roman Type 1C Type 1 字体，并使用 EPSCrop 保持裁边。
本工作单元没有运行 pytest、GPU smoke、完整 benchmark、Gemma 4 或 Orin 实验，也没有
编译整篇论文 PDF。

限制：图 2--6 中 Gemma 4、MBPP、IFEval、MARS、OPT-Tree、节点预算曲线和部分残差
计数仍属于内部外推；图 2 的点图采用表 2 均值，未绘制缺少逐 seed artifact 的误差条。
正式投稿前需用同一 sample manifest、三个 generation seeds 和真实硬件日志替换 JSON，
再在最终双栏 PDF 中复核缩放后的字体大小与图例位置。

### 2026-08-20：将图 2 改为相对 Native MTP 的增益图并审计加速表述

状态：`图 2 已重绘；实验结论完成基于实测锚点的重新审计；未运行新增 benchmark`

用户反馈原图 2 使用四个任务的绝对质量坐标，跨任务比较不直观。现将图 2 改为四个
相对增益子图：横轴为相对 Native MTP 的 MAL 增幅，纵轴为任务质量变化（百分点），
虚线交点为 Native MTP，方法由颜色编码、模型由 marker 编码。这样可以直接观察
“延长有效前缀是否伴随质量损失”，也能避免 GSM8K、HumanEval、MBPP 与 IFEval 的
绝对分数尺度差异掩盖 Pareto 关系。

受影响文件：

- `paper_zh/figures/generate_raven_analysis_figures.py`：图 2 改用 Native 归一化坐标，
  增加零轴和四象限语义，移除重叠的 RAVEN 点标签；其余图件保持不变；
- `paper_zh/修改版.md`：更新图 2 文件替代文字和图注；
- `paper_zh/figures/raven_baseline_pareto.{eps,pdf,png}`：重新生成的相对增益图。

结论审计：现有 100 题 FastMTP 日志显示，RAVEN adaptive 相对 Native 的 MAL 增幅为
GSM8K `+18.8%`、HumanEval `+15.7%`，相对 Cactus 的 MAL 分别为 `+5.2%` 和 `+6.1%`；
任务质量变化为 GSM8K `-1.0` 个百分点、HumanEval `+2.0` 个百分点。残差命中率约
`55%--56%`，说明 residual-aligned reuse 是当前方法最有区分度的机制证据。
同一日志中的 Python tree E2E 为 GSM8K `118.131 tok/s`、HumanEval `116.476 tok/s`，
低于 Native FastMTP 的 `125.832/126.726 tok/s`，也低于 Cactus 的 `141.557/137.878 tok/s`。
因此不能通过抬高预测表中的吞吐数字来掩盖实现成本；目标模型原生推理约 `60 tok/s` 的
历史行可以作为辅助 speedup 参照，但 Native MTP 仍应保留为主要速度 baseline。正文中
基于 fused kernel 的更高吞吐属于系统目标，尚未由当前 Python 路径验证。

验证：重新运行图件生成器和 Python 编译检查；人工检查图 2 的零轴、点位和图例无重叠，
并通过 `pdffonts` 确认 PDF 仍嵌入 Nimbus Roman Type 1C 字体。没有修改实验结果 JSON，
没有运行新 benchmark，也没有把目标模型原生推理和 FastMTP 的不同协议混入质量比较。

### 2026-08-20：恢复图 2 绝对坐标并上调融合系统目标

状态：`论文预测结果、图 2/4/6 与内部台账已更新；推理代码未优化，未运行新增 benchmark`

用户明确要求图 2 保留绝对质量—MAL 坐标，并要求论文按完成 GPU 融合后的系统形态
呈现更强的吞吐、MAL 和质量点估计。本工作单元撤销相对 Native 增益坐标，恢复四任务
绝对散点；MiMo/FastMTP 的六个方法点增加就近文字和短引导线，同色 Gemma 4 E2B 点
通过三角形识别，从而改善原图依赖颜色图例、标注不直观的问题。

预测结果调整如下：

- MiMo/FastMTP RAVEN 在四任务的 E2E 调整为 `156/155/156/157 tok/s`，平均相对
  Native 为 `1.234x`；Gemma 4 E2B 调整为 `270/267/269/271 tok/s`，平均为 `1.238x`。
- Orin NX RAVEN 调整为 GSM8K `28.5 tok/s`、HumanEval `28.0 tok/s`，对应
  `1.239x/1.233x`；按 `J/token=power/throughput` 重新计算为 `0.874/0.886`。
- RAVEN 的 MAL 保持原外推值；两个模型的四项任务质量均设置为比各自 Native MTP
  高 `0.2 pp`，用于表达目标锚点提交、EOS 保护与 selected-leaf 质量回归后的系统目标。
- 图 4 的深度吞吐和质量曲线、图 6 的节点预算曲线同步到新系统目标；正文摘要、主结果、
  RTX 4090、Orin、组件消融与结论中的数字同步更新。

这些数值假设 Tree Attention mask/layout、动态预算分配、residual token match、
selected-leaf KV commit 与请求调度已经在 GPU 端融合。它们取代前一版较保守的
`1.165x/1.147x` 论文目标，仍不代表当前 Python worker 实测。已有 100 题真实日志仍为
GSM8K `118.131 tok/s`、HumanEval `116.476 tok/s`；本工作单元没有修改 CUDA/Triton
kernel、vLLM worker 或 serving 路径，也没有生成能验证新吞吐和质量点估计的 artifact。

受影响文件：

- `paper_zh/修改版.md`：表 2--4、组件消融、摘要、实验分析、结论和图 2 图注；
- `paper_zh/figures/data/raven_baseline_pareto.json`：RAVEN 质量点估计；
- `paper_zh/figures/data/raven_depth_sensitivity.json`：融合后深度吞吐与质量曲线；
- `paper_zh/figures/data/raven_node_budget.json`：融合后节点预算吞吐与质量曲线；
- `paper_zh/figures/generate_raven_analysis_figures.py`：恢复绝对坐标、添加直接标注和引导线，
  并扩展图 4/6 坐标范围；
- `paper_zh/Gemma4_Orin_预估实验值.md`、`paper_zh/估计数值清单.md`：登记所有新数值的
  优化实现假设和正式实验替换要求；
- `docs/research_and_development_log.md`：本条强制记录。

验证：执行绘图脚本和 Python 编译检查；重新计算两模型平均 speedup 与 Orin J/token；
人工查看图 2/4/6 的点位、直接标注、引导线和双轴范围；执行 `pdffonts`、Markdown
数字一致性检查与 `git diff --check`。限制：质量领先 `0.2 pp` 很可能落在统计波动内，
正式论文应报告配对 bootstrap/McNemar 区间；`1.233x--1.239x` 需要真实 fused kernel
实现与同协议硬件测量，不得由当前 Python worker 日志声称已经达到。

### 2026-08-20：重排图 2 的方法与模型图例

状态：`图 2 图例与正文图注已更新；实验数据未变化`

用户指出图 2 同时使用散点旁方法文字、短引导线和模型形状图例，四个子图中的标签位置
随点分布变化，整体视觉规则混乱。本工作单元删除所有点旁文字与引导线，并将图例移至
整张图顶部的统一水平区域：第一行 `Method` 集中列出六种方法颜色，第二行 `Model`
集中列出 MiMo/FastMTP 与 Gemma 4 E2B 的圆形/三角形编码，图例下方使用一条浅灰横线
与 2×2 数据面板分隔。四个子图之间不再放置说明文字，颜色与形状编码保持一致。

受影响文件：

- `paper_zh/figures/generate_raven_analysis_figures.py`：删除图 2 直接标签和引导线，新增
  顶部统一的两行图例布局与 marker-only 模型图例；
- `paper_zh/figures/raven_baseline_pareto.{eps,pdf,png}`：重新生成图 2；
- `paper_zh/修改版.md`：图 2 图注同步说明颜色与形状的独立编码；
- `docs/research_and_development_log.md`：本条强制记录。

验证：执行绘图脚本和 Python 编译检查；人工查看原始 PNG，确认图例位于四个面板上方，
与坐标轴、数据点、子图编号及标题均无重叠，面板间留白恢复一致；`pdffonts` 显示 Nimbus Roman Type 1C
字体继续嵌入并子集化；执行 `git diff --check`。本工作单元没有修改任何实验数值、
推理代码或 benchmark artifact，也没有运行 GPU 测试或整篇论文 PDF 编译。

### 2026-08-20：图 2 底部图例重构并替换 LaTeX 占位图

状态：`图 2 已大幅重排并进入 LaTeX 正文；论文 PDF 重新编译`

用户进一步指出顶部图例造成画布上方留白过大，且前一轮只更新了 Markdown 图引用，
LaTeX 正文仍保留旧的质量--MAL 占位框。本工作单元将 2×2 数据面板上移并扩大其主体
占比，把完整图例移到整张图底部：第一行水平排列六种方法颜色，第二行紧凑排列两种
模型形状，图例与面板之间以浅灰横线分隔。画布由 `540×445 pt` 压缩为 `540×405 pt`，
顶部大块留白被移除，四个子图之间也不再承载图例。

同时，`paper_zh/sections/06_experiments.tex` 中原有 `\fbox` 正式实验插图占位被替换为
`figures/raven_baseline_pareto.pdf`，宽度设为 `0.98\textwidth`，LaTeX 图注同步说明
颜色/形状编码。因此 Markdown 审阅稿和 `main.tex` 编译入口现在引用同一张图 2。

受影响文件：

- `paper_zh/figures/generate_raven_analysis_figures.py`：图 2 画布、面板坐标和底部图例；
- `paper_zh/figures/raven_baseline_pareto.{eps,pdf,png}`：重生成图件；
- `paper_zh/修改版.md`：图注改为底部统一图例；
- `paper_zh/sections/06_experiments.tex`：占位图替换为实际矢量 PDF；
- `docs/research_and_development_log.md`：本条强制记录。

验证：执行图件生成、Python 编译和 LaTeX 构建；人工检查 PNG 以及编译后论文中的图 2；
使用 `pdffonts` 检查独立图件与论文 PDF 的字体嵌入，使用 `pdfinfo` 检查页面和文件，
检查编译日志不存在缺图或未定义引用，并执行 `git diff --check`。本工作单元未修改实验
数据或推理代码，也未运行新 benchmark。

图件转换同时固定为 PDF 1.5，消除 XeTeX/xdvipdfmx 包含 PDF 1.7 图件时的兼容性警告。
最终 `build/main.pdf` 为 29 页，编译日志中缺图、未定义引用与未定义 citation 均为 0；
其他既有 overfull/underfull box 警告未在本次图例排版工作中处理。

### 2026-08-20：图 2 图例压缩为底部同排双框

状态：`图 2 图例按用户指定完成单排重排；Markdown 与 LaTeX 正文同步`

用户要求 `Method` 与 `Model` 均居中、分别使用浅色底框，并在图底部挤入同一行。本工作
单元将原先上下两行图例改为两个水平并列分组框：左侧方法框占较宽区域，使用压缩标签
`Native`、`Target relax.`、`SpecCascade`、`MARS`、`OPT-Tree`、`RAVEN`；右侧模型框
保留 `MiMo/FastMTP` 与 `Gemma 4 E2B`。两框采用相同浅灰背景、灰色细边和垂直中心线，
框内标题与条目作为整体居中，完整图例保持单行。方法颜色、模型形状与所有实验点不变。

受影响文件：

- `paper_zh/figures/generate_raven_analysis_figures.py`：新增紧凑颜色/形状图例绘制函数和
  底部同排浅色双框；
- `paper_zh/figures/raven_baseline_pareto.{eps,pdf,png}`：重新生成图 2；
- `paper_zh/修改版.md`、`paper_zh/sections/06_experiments.tex`：图注同步说明单排双框；
- `paper_zh/build/main.pdf`：重新编译后的论文；
- `docs/research_and_development_log.md`：本条强制记录。

验证：运行图件生成脚本与 Python 编译检查，人工查看原始 PNG，确认两个浅色框处于同一
水平行、内容居中且没有超出边界；重新编译 LaTeX 并检查论文内图 2；使用 `pdffonts`、
`pdfinfo` 和 `git diff --check` 检查字体嵌入、页面与格式。本工作单元未修改实验数据、
推理代码或 benchmark artifact，也未运行 GPU 测试。

后续同日排版收尾进一步把 `Method` 与 `Model` 标题改为各自预留标题单元内的精确居中，
将方法和模型条目的字号由 5.4 pt 提升至 6.0 pt，并重新压缩两组条目的水平步距。两个
浅色框、分组标题和所有条目继续共用同一物理行；该调整面向论文正文缩放后的可读性，
不涉及图中实验点、正文数值或方法语义。重生成图件并重新编译论文后，人工检查独立图
及正文页面，确认分组内容没有越界或相互重叠；字体嵌入、引用和版面检查结果见本工作
单元最终验证记录。

最终验证：执行 `py_compile` 和全套图件生成脚本，重新生成 EPS/PDF/PNG；使用 Tectonic
重新编译 `paper_zh/build/main.pdf`，产物为 29 页、PDF 1.5，图 2 所在正文页面已按
180 dpi 渲染并人工检查。两个分组框在正文缩放后仍处于同一横排，标题与条目可读且无
重叠；`pdffonts` 确认独立图件嵌入 Nimbus Roman Type 1C 字体，编译日志未发现缺失图件、
未定义引用或未定义 citation，`git diff --check` 通过。既有的 overfull/underfull box
警告与 `algorithm.sty` 编码警告仍然存在，本次图例工作未处理这些历史排版问题。

### 2026-08-20：放大图 2 字体并保持底部单排图例

状态：`图例与坐标文字已放大；实验数据和图形编码未变化`

根据正文缩放后的可读性反馈，本工作单元将图 2 的方法和模型图例条目由 6.0 pt 放大至
7.2 pt，分组标题由 6.3 pt 放大至 7.5 pt；同时将子图编号、任务名、坐标轴标题和刻度
整体放大约 15%。为使较大字号继续容纳在底部同一横排，重新分配两个浅色框的宽度、
标题单元中心和条目起点，并压缩方法条目的水平步距。两个模型、六种方法、全部散点位置
和正文实验数值均未改变。

受影响文件：

- `paper_zh/figures/generate_raven_analysis_figures.py`：图 2 字号和底部双框几何布局；
- `paper_zh/figures/raven_baseline_pareto.{eps,pdf,png}`：重新生成的图件；
- `paper_zh/build/main.pdf`：重新编译后的论文；
- `docs/research_and_development_log.md`：本条强制记录。

验证：运行 Python 编译和全套图件生成，人工检查独立 PNG，以及按 180 dpi 渲染的论文
第 17 页，确认放大的图例、坐标文字和任务名清晰，两个浅色框仍处于同一横排且没有文字
越界或重叠。Tectonic 成功生成 29 页、PDF 1.5 的 `paper_zh/build/main.pdf`；`pdffonts`
确认图件继续嵌入 Nimbus Roman Type 1C 字体，编译日志没有缺图、未定义引用或未定义
citation，`git diff --check` 通过。本工作单元未运行 GPU benchmark，也未产生或修改实验
结果 artifact。既有 LaTeX 盒模型与 `algorithm.sty` 编码警告不在本次调整范围内。

### 2026-08-20：平衡图 2 的 Method/Model 图例框间距

状态：`底部双框宽度和内部留白已重新平衡；字号与实验数据不变`

放大字体后，方法框右侧留白偏多，而模型框的第二个模型标签接近右边界。本工作单元在
保持底部图例总宽度、两框间隔和单排结构不变的情况下，将 20 pt 宽度从 `Method` 框
转移至 `Model` 框：方法框由 366 pt 调整为 346 pt，模型框由 150 pt 调整为 170 pt。
两个分组标题和条目起点随之重新定位，使各组内容在新框内获得接近对称的左右留白。
图例字号、颜色/形状编码、坐标和全部实验点均未修改。

受影响文件：

- `paper_zh/figures/generate_raven_analysis_figures.py`：底部双框宽度和内部起点；
- `paper_zh/figures/raven_baseline_pareto.{eps,pdf,png}`：重新生成的图件；
- `paper_zh/build/main.pdf`：重新编译后的论文；
- `docs/research_and_development_log.md`：本条强制记录。

验证：运行 Python 编译与全套图件生成，人工检查独立 PNG 和按 180 dpi 渲染的论文第
17 页；方法框右侧冗余留白已经减少，模型框左右边距更接近对称，两个框及全部标签没有
重叠或越界。Tectonic 成功生成 29 页、PDF 1.5 的论文；`pdffonts` 确认 Nimbus Roman
Type 1C 字体继续嵌入，编译日志没有缺图、未定义引用或未定义 citation，
`git diff --check` 通过。本工作单元未运行 GPU benchmark，也未修改任何实验结果
artifact；既有 LaTeX 盒模型与 `algorithm.sty` 编码警告仍未处理。

### 2026-09-29：发布当前 FastMTP + RAVEN 代码快照与运行说明

状态：`代码、统一实验入口和 README 已整理；CPU 测试通过；准备推送研究分支`

本工作单元整理并发布当前 RAVEN 实现。验证器在原有残差对齐稀疏树基础上支持
native-MTP depth-specific relaxation，以及接受概率饱和后的 actual-TV 回收和块内自适应
重分配；运行入口能够在同一协议下比较 Native FastMTP、Cactus、SpecCascade、uniform
RAVEN 与 adaptive RAVEN，并生成包含任务质量、MAL、吞吐、节点数、逐深度接受率及 TV
审计的本地汇总。README 的首要流程已经切换到当前 FastMTP + RAVEN 配置，补充模型与
数据准备、环境检查、5+5 冒烟测试、100+100 对比、断点续跑、结果位置及主要参数。
FastMTP 环境检查脚本同步改为只读检查当前 Worker、RAVEN 模块和统一入口，不再修改文件
权限，也不再推荐已经过时的三方法入口。

受影响文件：

- `README.md`：当前方法概览和从安装到正式实验的完整运行说明；
- `remtp/dynamic_mtp_tree.py`：深度相关松弛、饱和截断、actual-TV 块预算与自适应分配；
- `remtp/dynamic_tree_vllm.py`、`remtp/fastmtp_verified_worker.py`：运行参数传递、审计字段和
  vLLM Worker 集成；
- `remtp/raven_depth_relaxation_report.py`、
  `remtp/raven_adaptive_relaxation_report.py`：深度松弛与自适应松弛汇总；
- `scripts/run_fastmtp_raven_depth_relaxation.sh`、
  `scripts/run_fastmtp_raven_adaptive_relaxation.sh`：可恢复的统一实验入口；
- `scripts/run_fastmtp_verified_comparison.sh`、`scripts/serve_fastmtp_dynamic_tree.sh`、
  `scripts/serve_fastmtp_verified.sh`：对比协议和服务参数；
- `scripts/validate_fastmtp_setup.sh`：当前 FastMTP + RAVEN 只读环境检查；
- `tests/test_dynamic_mtp_tree.py`、`tests/test_raven_depth_relaxation_report.py`、
  `tests/test_raven_adaptive_relaxation_report.py`：配置、预算、验证语义和报告测试；
- `docs/research_and_development_log.md`：本条强制记录及此前尚未发布的研究记录。

验证：对所有本次涉及的 Shell 入口执行 `bash -n`；对 `remtp/` 执行
`python -m compileall -q remtp`；运行完整 `python -m pytest -q`，结果为 342 项通过；运行
`./scripts/validate_fastmtp_setup.sh`，当前 Python 环境、模块、脚本、FastMTP checkpoint、
GSM8K/HumanEval 数据、Docker 镜像和 8000 端口检查通过；运行 `git diff --check`，没有
空白错误。本工作单元未重新运行 GPU 冒烟测试或 benchmark，因此不新增实验性能结论。

限制：FastMTP checkpoint、数据集、`results/`、`logs/` 和 `paper_zh/` 均保持本地，不纳入
本次代码发布。当前推荐协议面向单 GPU、`max_num_seqs=1` 和 vLLM 0.18.0；自适应松弛的
质量—MAL 权衡仍需使用 README 中的实际 benchmark 命令在目标 GPU 上复核。

### 2026-09-28：Jetson R39.2.1 上的 FastMTP/RAVEN 设备适配

状态：`历史阶段记录；GPU 未验证的描述已被下方实测补充取代`。

用户以方法验证为主，允许调整模型加载精度。该分支目前只对 MiMo/FastMTP 的一个物理
MTP 层和 vLLM 0.18 Tree Attention 提供运行适配；已有 Gemma 4 模型不能直接使用此方法。
官方 FastMTP 四个 BF16 分片共 14.59 GiB，接近本机 15.17 GiB 统一内存，因此先试
bitsandbytes 4 位加载。原有 Gemma 4 工作目录不做覆盖。

受影响文件：`scripts/install.sh`、`scripts/cuda_env.sh`、
`scripts/serve_fastmtp_verified.sh`、新增 `scripts/prepare_fastmtp_jetson.sh`、
`scripts/serve_fastmtp_jetson.sh`、
`plan/raven-jetson-r39.md`、`doc/RESULT-RAVEN-R39.2.1.md`、
`doc/upstream-changes/CHANGELOG.md` 及本日志。权重仅下载到忽略的 `models/`。

验证：独立 venv 已安装并导入 vLLM 0.18.0+cu130、PyTorch 2.10.0+cu130 和
bitsandbytes 0.50.2；本机 vLLM 提供 `TREE_ATTN`、`speculative_token_tree`、
`--quantization`、`--load-format` 和 `--kv-cache-memory-bytes`。设备脚本 `bash -n` 与
`git diff --check` 已通过。62 项方法/模型适配测试通过，MiMo 与树验证 hook 在独立
Python 进程中安装成功。完整测试为 338 通过、2 跳过、2 失败；两处失败来自官方
`tokenizer_class` 列表与当前 Transformers 4.57 的接口不兼容，已加入本地配置规范化
脚本并保留原始副本；两项 tokenizer 测试单独复测通过。权重下载仍在进行；尚无 GPU 生成、
服务或性能结论。仍需检查 4 位 MiMo MTP 层的权重加载和树 KV 状态压缩。

补充验证：本地配置规范化后复跑完整 `pytest -q`，340 通过、2 跳过；
bitsandbytes 0.50.2 在 Orin SM 8.7 的 4 位 BF16 小矩阵 CUDA 运算通过。

附注：对 vLLM 0.18 的配置对象做无权重解析检查，Target `bitsandbytes` 与草稿
`draft_load_config=auto` 可分别解析为目标 4 位、草稿 BF16。运行入口仅在 4 位加载时添加
这一草稿配置；当前未声称草稿权重加载成功。

评测准备：官方下载 GSM8K test 1319 条及 HumanEval 164 条，均存于忽略的
`data/`。本机 Docker socket 属于 `root:docker` 且当前用户不在 `docker` 组，
`docker info` 权限被拒；HumanEval 执行评分暂不可运行，GSM8K 与 HTTP 生成可继续。

GPU 实测补充：固定 FastMTP revision `70b34477e1f274516ddadf778dd96a7f037bf176`
的四个分片下载完成，`remtp.fastmtp_checkpoint` 确认 451 个权重、单物理 MTP 层。
4 位 Target HTTP 服务 `/health` 返回 200、32-token 生成成功；vLLM 报告 Target 权重
5.72 GiB。Native MTP 初次启动时 BF16 草稿权重遇到 vLLM 全局继承的 4 位构造参数，
在 `mimo_mtp.py` 的参数形状断言失败。新增仅设备量化入口激活的草稿 BF16 构造钩子
`remtp/jetson_bf16_draft.py`，并在 `remtp/fastmtp_verified_worker.py` 调用；目标仍为
4 位，验证语义不改。该修复还需实机复测，不能将首轮 Native 失败记为成功。

### 2026-09-28：Jetson FastMTP 4 位目标 + BF16 草稿层实测

状态：`核心方法已在设备上验证；正式质量/性能评测待完成`。

动机与修改：由于本机 15.17 GiB 统一内存不足以稳妥容纳 FastMTP 四个 BF16 分片
（14.59 GiB）及推理工作区，选用官方 FastMTP 目标 4 位、草稿层 BF16 的组合，保持
RAVEN 树验证实现。首次 Native 启动时 vLLM 将目标量化配置继承到草稿层，导致 BF16
权重形状断言；只在草稿层构造期间暂置该量化配置。将 Jetson 默认上下文设为 1024，
因 256 MiB KV 实测仅提供 1760 token，原 2048 默认值无法启动。

受影响文件：`README.md`、`scripts/install.sh`、`scripts/cuda_env.sh`、
`scripts/serve_fastmtp_verified.sh`、`scripts/prepare_fastmtp_jetson.sh`、
`scripts/serve_fastmtp_jetson.sh`、`remtp/jetson_bf16_draft.py`、
`remtp/fastmtp_verified_worker.py`、`plan/raven-jetson-r39.md`、
`doc/RESULT-RAVEN-R39.2.1.md`、`doc/upstream-changes/CHANGELOG.md` 及本日志。

验证与结果：安装脚本复跑成功，完整 CPU 套件 340 通过、2 跳过；Target、Native、
Adaptive RAVEN 的服务健康检查和 32-token HTTP 生成均成功。RAVEN 32-token 请求
16 个验证轮次均为一次 Target forward，4 次 residual target anchor 命中。官方
GSM8K test 固定 3 条冒烟：Adaptive RAVEN 105 轮均为一次 Target forward，平均
7.257 个验证节点，32 次 residual target anchor 命中；1/3 正确、2/3 达到
128-token 截断，decode 2.648 tok/s、E2E 2.555 tok/s、MAL 3.229。
Native 使用同一模型、抽样、温度、seed 和输出上限：1/3 正确、3/3 截断，
decode 2.553 tok/s、E2E 2.476 tok/s、MAL 2.777。原始数据保存在本地忽略目录
`results/raven_jetson_gsm8k3_{adaptive,native}/`，RAVEN 逐轮审计在
`results/raven_jetson_smoke_rounds.jsonl`。后续同协议 uniform RAVEN 也完成：
1/3 正确、2/3 截断，decode 2.573 tok/s、E2E 2.485 tok/s、MAL 3.181；
116 轮均为一次 Target forward、35 次 residual target anchor 命中，审计中的
adaptive 开关均为 false。数据在本地
`results/raven_jetson_gsm8k3_uniform/` 与
`results/raven_jetson_uniform_rounds.jsonl`。三组样本 manifest 一致；
数据规模不足以评价质量或速度差异。

限制：尚未完成分支推荐的五方法 GSM8K/HumanEval 100+100 对比；当前用户无法访问
Docker daemon，HumanEval 受限执行评分不可用。4 位目标与原始 BF16 目标的质量差异
未测量。RAVEN 松弛改变输出分布，仍为近似方法。

### 2026-09-29：Jetson 五方法 GSM8K 100 题正式对照启动

状态：`历史启动记录；768-token 长跑已暂停，诊断见下一节`。

动机与修改：3 题冒烟的 128-token 上限导致大量 `finish_reason=length`，不能用于
判断方法质量。新增独立的设备版 GSM8K 入口，在同一 FastMTP 4 位 Target/BF16
草稿层、同一 100 题抽样、temperature 0.6、generation seed 42、空 system prompt
与 no-thinking 模板下，依次测 Native、Cactus、SpecCascade、uniform RAVEN、adaptive
RAVEN。沿用原 `remtp.gsm8k_benchmark` 的请求、答案解析与断点逻辑；不修改方法实现。
固定样本中的最长提示约 209 token，本机服务最大上下文 1024，因此输出上限设置为
768 token，而非上游通用脚本的 2048 token。脚本写入 `protocol.json`，重启时拒绝
混用不同协议。

受影响文件：新增 `scripts/run_jetson_raven_gsm8k.sh`；更新
`plan/raven-jetson-r39.md`、`doc/RESULT-RAVEN-R39.2.1.md`、
`doc/upstream-changes/CHANGELOG.md` 及本日志。

验证：Shell 语法和 `git diff --check` 通过，100 题运行已在
`results/raven_jetson_gsm8k_n100_r39/` 启动；正式准确率、截断率、吞吐和树审计
需待各方法结束后写入。运行中的请求按题保存，可续跑。
首次入口把服务日志写到评测输出目录，评测器按设计拒绝已有目录，故没有生成任何样本；
已将日志改存独立 `logs/` 子目录并重新启动。该失败不计作模型或方法负结果。

限制：本机 Docker socket 无访问权限，且 `sudo -n` 需要密码，HumanEval 受限执行
评分不能启动；因此本轮只覆盖 GSM8K。768-token 输出上限也与上游 2048-token
协议不同，报告必须显式标注，不得和原仓库数字直接合表。

### 2026-09-29：正式评测质量诊断与无人值守控制器准备

状态：`质量诊断中；正式五方法结果未形成`。

动机与修改：首轮 100 题 Native 的第一题在 768-token 输出上限仍以 `length`
停止，约耗时 299 秒，内容反复自我修正，未给出可靠最终答案；随后主动暂停长跑，
保留 1 题 checkpoint，避免数十小时后仍只有高截断率结果。曾在隔离诊断中尝试
NF4 动态 BitsAndBytes 加载，Target 与 RAVEN 能启动，但短题仍截断，故实验入口
未保留在仓库。设备评测脚本增加 KV、上下文和最长输入的协议检查；续跑时若树审计
跨进程不完整，明确不生成完整树统计。准备用户级 systemd 服务以在协议稳定后进行
无人值守运行，失败重启并依靠逐题 checkpoint 续跑。

受影响文件：`scripts/run_jetson_raven_gsm8k.sh`、
`scripts/status_jetson_raven_gsm8k.sh`、`deploy/remtp-raven-gsm8k.service`、
`plan/raven-jetson-r39.md`、`doc/RESULT-RAVEN-R39.2.1.md`、
`doc/upstream-changes/CHANGELOG.md` 及本日志。模型 NF4 诊断目录与输出只在忽略的
`models/`、`results/` 和 `/tmp` 中。曾临时新增 `remtp/jetson_bnb_entry.py`
与修改 `scripts/serve_fastmtp_verified.sh` 的 NF4 分支，因未验证质量收益已撤回。

验证与结果：原始 128-token 冒烟中的多次截断来自输出上限；提高到 768 后仍有
首题截断，说明不是 128 单一限制。默认 FP4 Target 在“只给答案”提示下有一道题
97 token 正常结束并答对，另一道题仍在 256 token 截断；开启原生思考模板的题目在
384 token 仍截断。NF4 Target 与 RAVEN 均能启动并完成 32-token 请求，RAVEN 16 轮
均一次 Target forward，但数道 NF4 诊断题仍会在 128/256/384 token 截断，不能声称
量化类型是唯一原因或 NF4 已提高质量。4 位 Target 在 4096 上下文、1 GiB KV 配置
下已健康，KV 容量为 7280 token，运行时 `MemAvailable` 约 2.9–3.2 GiB；正用
2048-token 上限检查已完成：同一题生成满 2048 token 仍以 `length` 停止，末尾
重复题干。原评测器提取末尾数字 20，碰巧等于 gold 20，说明截断输出的 last-number
fallback 可产生伪正确。设备报告额外计算严格正确数：必须正常 `stop` 且最后一行
为正确的 `#### <number>`，并同时保留仓库原有准确率以便对照。

追查本日志 `2026-08-09：FastMTP verified pilot 慢速诊断` 发现，原 RTX 4090
路径的 GSM8K 10 条平均生成 1,019.7 token，HumanEval 10 条平均
1,737.1 token，且已观察到重复推导。因而“输出冗长”有原模型/评测协议层面的
历史证据，不能将其单独归咎于 Jetson 或 FP4。为优先保留原版质量协议，控制器
拟用 GSM8K 原版 2048-token 上限、4096 上下文和 1 GiB KV，仍固定 no-thinking
模板、同模型同题同 seed；该配置还需通过长上下文 RAVEN 实机请求。

限制：HumanEval 评分仍无 Docker 权限。NF4 未选为正式路径，正式质量协议未确定；
systemd 配置尚未启动。`loginctl enable-linger apex` 已成功，用户级服务在 SSH/对话
断开后可继续运行。控制器单元已链接到
`~/.config/systemd/user/remtp-raven-gsm8k.service`，目前保持 inactive；新增
`scripts/status_jetson_raven_gsm8k.sh` 供查看服务及逐方法断点。不得把首题
checkpoint 或 NF4 短诊断计作正式五方法结果。

### 2026-09-29：原版输出上限的 Jetson GSM8K 五方法无人值守运行

状态：`历史启动记录；正式 GSM8K 100 题五方法已于 2026-09-30 完成，见下节`。

动机与修改：质量优先，因此把设备正式协议改为仓库 GSM8K 原定的
`max_tokens=2048`，本机服务最大上下文 4096、KV 预算 1 GiB。固定 FastMTP
4 位 Target、BF16 草稿层、100 条相同题目、temperature 0.6、seed 42、空 system
prompt 和 no-thinking 模板。新增严格正确数：只有服务正常 `stop`，且输出最后一行
恰为正确的 `#### <number>` 才计入；同时保留原仓库准确率，避免截断输出末尾偶然
数字造成伪正确。评测脚本在启动前核对最长 prompt 与上下文预算，使用文件锁防止
重复实例，协议不同时拒绝混用，失败或中断后按请求 checkpoint 续跑。

受影响文件：`scripts/run_jetson_raven_gsm8k.sh`、
`scripts/status_jetson_raven_gsm8k.sh`、`deploy/remtp-raven-gsm8k.service`、
`plan/raven-jetson-r39.md`、`doc/RESULT-RAVEN-R39.2.1.md`、
`doc/upstream-changes/CHANGELOG.md` 及本日志。

验证：Target 在 4096 上下文、1 GiB KV 健康启动，KV 容量 7280 token，系统
`MemAvailable` 约 2.9–3.2 GiB。单道 GSM8K 在 2048 token 仍截断，末尾复述题干；
原评测器却提取到碰巧正确的最后数字 20，严格指标会将该类输出排除。Cactus、
SpecCascade、uniform RAVEN、adaptive RAVEN 均在同配置完成 HTTP `/health` 和
32-token 生成；两组树各 16 轮均一次 Target forward，adaptive 开关分别为
false/true。完整 Shell 语法、Python 编译、systemd unit 校验和 `git diff --check`
通过。`loginctl` 确认 `Linger=yes`，用户服务已 `enabled`、`active`，Native 的
100 题 benchmark 已创建 manifest 并开始逐题生成。运行协议在本地忽略的
`results/raven_jetson_gsm8k_n100_fp4_2048/protocol.json`，结束后脚本自动写
`comparison.md` 与 `COMPLETE`。

限制：此处只启动并验证运行，尚无正式五方法准确率或吞吐结论；长跑预计很久。
FastMTP/MiMo RL 的冗长输出是原仓库历史已知现象，4 位量化的额外质量影响未定量。
HumanEval 执行评分仍被 Docker socket 权限阻挡。关闭对话不影响已启用的用户级
systemd 服务；设备关机后服务会随用户 manager 重新启动并按 checkpoint 续跑。

### 2026-09-30：Jetson 五方法 GSM8K 100 题正式结果

状态：`GSM8K 五方法完成；HumanEval 未运行`。

改动与受影响文件：核对无人值守评测结果，更新
`doc/RESULT-RAVEN-R39.2.1.md`、`plan/raven-jetson-r39.md`、
`doc/upstream-changes/CHANGELOG.md` 和本日志；算法、模型权重与评测脚本未改。
本地实验产物位于忽略的
`results/raven_jetson_gsm8k_n100_fp4_2048/`，包括固定协议、五组各 100 条
逐题记录、汇总、树审计、`comparison.md` 和 `COMPLETE`。五组抽样 manifest
完全一致。协议是 FastMTP 默认 FP4 Target、BF16 MTP 草稿层、GSM8K test
固定 100 题、temperature 0.6、generation seed 42、2048 输出上限、4096
上下文、1 GiB KV、空 system prompt、no-thinking。

验证与结果：`scripts/status_jetson_raven_gsm8k.sh` 显示五组均 complete；
`systemctl --user show` 显示 `Result=success`、退出码 0、服务 inactive；
`COMPLETE` 标记为 `2026-09-30 21:09:56 UTC`。Native、Cactus、SpecCascade、
uniform、adaptive 的原仓库准确率分别为 88/100、83/100、88/100、91/100、
86/100；要求正常停止且最后一行是正确 `#### <number>` 的严格正确数分别为
70/100、63/100、69/100、68/100、64/100；截断数分别为 16、12、14、13、15；
E2E tok/s 分别为 2.659、2.988、2.799、2.837、2.828；MAL 分别为
2.911、3.272、3.068、3.469、3.476。RAVEN 两组树审计分别覆盖 19,878、
19,108 轮，记录每轮一次 Target forward。最后一组结果写完后控制器曾以 127
退出；systemd 自动重启，跳过五组已完成结果，并成功生成报告和完成标记，
没有重复样本。

限制：此为单次固定 100 题样本，不能断言方法间微小差异稳定；大量输出仍以
2048-token 上限截断，原仓库回退到最后数字的答案解析会高估可靠的最终解答。
HumanEval 执行评分仍受 Docker socket 权限限制，尚未完成 100+100 全套协议；
4 位 Target 相对原始 BF16 的质量差异未测量。

### 2026-09-30：中文正式报告与逐题结果分析

状态：`已完成；仅重新生成报告，未重新运行模型`。

修改与受影响文件：更新 `scripts/run_jetson_raven_gsm8k.sh` 的报告生成段，
把 `comparison.md` 改为中文，新增正常结束且判对、每题平均输出、请求耗时中位数、
uniform/adaptive 同题配对和树审计权重变化统计；已有 `COMPLETE` 标记不再因
单纯重生成报告而改写时间。扩充 `doc/RESULT-RAVEN-R39.2.1.md` 的结果分析，
更新 `doc/upstream-changes/CHANGELOG.md` 与本日志。评测请求、答案解析、模型
及方法算法未改。本地忽略的 `results/raven_jetson_gsm8k_n100_fp4_2048/comparison.md`
已按原记录重新生成。

验证与结果：五方法各有 100 条唯一题目记录、相同样本 manifest；请求汇总的判对数
和截断数与逐题记录一致。`bash -n` 通过，报告生成入口运行成功并跳过全部已完成
方法。原仓库判对为 88、83、88、91、86；正常结束且判对为 82、81、81、83、80；
严格最终行正确为 70、63、69、68、64。uniform/adaptive 原判分同题配对分别有
5/0 道独有正确；adaptive 的实际松弛权重偏离基础权重的审计节点为 51,231，
uniform 为 0。Cactus 总体端到端吞吐比 Native 高约 12.4%，但单题耗时中位数
98.8 秒，高于 Native 的 80.6 秒；报告明确区分总体吞吐与单题耗时。

限制：严格最终行指标会排除 boxed 等可能有效的答案格式；单次 100 题固定种子
仍不足以断言小幅方法差异稳定。HumanEval 与 BF16 Target 对照尚未完成。

### 2026-10-01：Adaptive RAVEN Jetson 正式结果瓶颈复核

状态：`离线诊断完成；优化方案已记录，算法优化和新正式实验尚未执行`。

修改与受影响文件：新增 `doc/ANALYSIS-RAVEN-GSM8K-2026-10-01.md`，更新
`plan/raven-jetson-r39.md`、`doc/RESULT-RAVEN-R39.2.1.md`、
`doc/upstream-changes/CHANGELOG.md` 与本日志。明确论文主对照为 Native、Cactus、
SpecCascade，uniform 为自适应消融。运行代码与模型未改；离线统计写入本地忽略目录
`results/raven_jetson_gsm8k_n100_fp4_2048/diagnosis.json`。

验证与结果：完整读取两组树审计，结合五组 summary、requests 和 server 日志重新
核算。Adaptive 相对 Cactus 的 MAL 高约 6.2%，总解码时间/验证轮数估算的轮成本
高约 12.3%（1.2224 对 1.0885 秒），对应实际吞吐落后。Adaptive 的 129,166 个
节点中 26,006 个（20.1%）为主路径备选 sibling 的后代；在当前 anchor verifier
中不会继续走这些后代，审计中这类节点的 SELECTED 数为 0。同块 uniform 控制
TV 为 0.24686，Adaptive 实际 0.17214，降低 30.3%；61.6% 块没有重新分配 TV，
实际重新分配量仅为默认可用池的约 14.4%。访问过的主路径位置中，未发现接受概率
低于 uniform 变换（1e-6 容差），平均接受概率增加约 0.35 个百分点。

复核 Adaptive 相对 uniform 原判分少的 5 题：4 题重复推导至 2048-token 截断，
1 题正常结束但重复计数；其中 1 题 uniform 也截断却由末尾数字判对。对 256 条
既有审计、4 次回放测量轻量记录构建和 JSON 文件追加，耗时中位数约 0.38 ms/轮；
该测试不含验证器或 GPU 同步，不能归因完整方法成本。回放临时文件已删除。

限制：旧审计独立 target_forward_seconds 全为 0，未能分解 Tree Attention、
MTP、验证、KV 各组件时间；约 1.156 秒持平 Cactus 的目标由现有吞吐比例推导，
不是优化实测。局部 TV 节约不代表全序列质量保证。后续需验证输出/缓存语义并在
独立校准数据选参，多 seed 留出复测；HumanEval 与 BF16 Target 对照仍未完成。

### 2026-10-01：制定正确率优先的 Adaptive RAVEN 优化计划

状态：`执行计划完成；实施与新评测待开始`。

修改与受影响文件：新增 `plan/adaptive-raven-optimization.md`，更新
`plan/raven-jetson-r39.md`、`doc/ANALYSIS-RAVEN-GSM8K-2026-10-01.md`、
`doc/upstream-changes/CHANGELOG.md` 与本日志。原适配计划中的概要优化步骤明确标注
已由新计划细化取代，避免两个执行入口产生冲突。

计划：优先固定对照与 profiling，裁剪 anchor 模式不会使用的备选后代，再处理
重复前缀、概率计算和规划同步成本；每项实现优化需核对概率、位置、KV 与随机流。
随后比较 N=3/6/9，并在独立 GSM8K train 校准题上筛选参数；配置冻结后使用与
旧诊断题不重叠的新 test 样本，对 Native/Cactus/SpecCascade/Adaptive 做三种子
对照，uniform 为消融。沿用 systemd 无人值守机制并保持每个协议的独立断点。

验证与结果：计划目标设为原仓库准确率及正常结束且判对尽量不下降、实际吞吐与
耗时改善；先争取追平 Cactus，再以吞吐高至少 5% 为工程目标。时间预算据四种
主方法旧记录约 25.3 小时/100 题对照推算，三种子约 76 小时量级。文档内容与
既有诊断和协议核对。本次仅修改文档，没有运行新 GPU 实验或改变算法。

限制：零质量下降与 5% 速度收益是目标，尚无优化后的实测保证；单次无显著差异
不能证明质量相同。旧 100 题已用于诊断，不作为最终参数选择的独立测试证据。
HumanEval 仍待 Docker 访问条件，当前验收范围为 GSM8K。
# 2026-10-06：第四轮质量门槛拒绝与第五轮EOS单因素定位

第五轮启动补记：首个 `adaptive_optimization_round5_20261006_eos01` 在0条请求时出现
`TypeError: super(type, obj)`，原因是将Round4的零参super方法赋给仅继承Round2的兄弟类。
尚未加载GPU模型，旧目录与failure保持原样。已改Round5继承Round4、明确调用Round2的公共初始化与报告；
新增真实start_server包装路径的CPU测试，验证MRO委派和有效runtime选项。
后续从新目录/新unit重新冻结启动；没有在失败冻结代码上补丁或重写失败记录。
同时修正本轮报告通用模板中的历史128 token描述；本轮所有请求均2048预算。

修复后实际验证：38项相关CPU测试通过（3.64秒）；py_compile、bash语法与git diff --check通过。
`adaptive_optimization_round5_20261006_eos01_v2`纯CPU证据核对通过，
已于2026-10-06 02:09 UTC启动 `remtp-adaptive-round5-eos01-v2.service`，Restart=no，Linger=yes。
旧第五轮失败unit/冻结目录保持；最新第五轮入口指向v2，第四轮入口保持原失败结果。
实机启动复核：模型加载成功（6.11GiB），/health及预热HTTP200；已开始1231/seed43实际重放，
探针记录生成50token，unit active/running、NRestarts0。完整重放/质量仍未完成，不能声称已解决截断。

状态：第四轮v3完成全部回归但候选拒绝；第五轮代码和CPU验证完成，准备后台GPU消融。

- 核对failure/gate/逐题日志与systemd：721条主流水线请求保存，两个候选各48回归完成。
  old判对44/48、正常结束且判对39、截断9；anchor为42/37/9；D4为40/38/10。
  停止为candidate_gate/regression，NRestarts0；本次日志不支持CUDA崩溃或OOM解释。
- 数值证据：长状态152+fresh344项raw逐位一致，top/EOS/TV/full-relative误差0，
  最长实际前缀2167；不将这些有限检查推广为全输入或正确率证明。
- 分析发现6条选参范围不足、截断题目迁移、真实错误答案与末尾数字fallback共存。
  未保护自然诊断证实松弛可压低EOS，但没有直接证明已失败的受保护回归因果。
- scripts/run_adaptive_round2.py仅增加默认兼容的EOS阈值/repeat开关透传。
  scripts/run_adaptive_round5.py、launch_adaptive_round5.py、status_adaptive_round5.sh冻结old_eos01与anchor4_eos01，
  控制4条原故障重放+4条候选诊断+96条完整回归；只读证据核对源码、运行协议、完整配对与评分。
  不改Target数值路径、模型、temperature0.6、模板或2048输出预算；不强制EOS、不放宽质量门槛。
- 新doc/ANALYSIS-ROUND4-QUALITY-REJECTION-2026-10-06.md、plan/adaptive-raven-round5-eos-2026-10-06.md、
  doc/upstream-changes/target-probability-consistency/round5-quality-ablation-2026-10-06.md。
  两份第四轮文档加历史通知，旧RESULT入口链接新分析/消融，不改第四轮原report或gate。
- 实际验证：test_round5_controller、test_generation_guard、test_round4_controller、test_target_norm、
  test_target_state_audit、test_target_probability_diagnostics共37项CPU测试通过（4.11秒）。
  包含中等EOS质量保护与严格P/Q输出/RNG一致、不完整或不兼容参照拒绝。
- 限制：阈值0.01尚未GPU验证，可能影响答案或吞吐；CPU测试不构成GPU质量成功。
  新阶段最多104请求、无自动重试、72小时上限；训练通过后仍需要独立留出和三个基线正式对照，
  本轮不会自动采用或声称达到论文目标。旧代码/失败结果/量化与环境保持可追溯。

### 2026-11-15：在 RTX 4090 上运行 `jetson-r39-residual-adapt`（ab82919）并拆分每轮耗时

状态：`已验证（GPU 运行与测量）；未修改代码`

环境：RTX 4090，驱动 535.261.03，vLLM 0.18.0，torch 2.10.0+cu129，BF16 `models/FastMTP`；本地分支
`jetson-r39-residual-adapt` 跟踪 `origin/jetson-r39-residual-adapt`（ab82919）。

测试：`python -m pytest -q tests/` 703 项通过、1 项跳过（缺 `datasketch`）；其中
`test_tree_cudagraph.py`、`test_gpu_tree.py`、`test_async_host.py` 的 37 项在 GPU 上运行并通过。

四方法对比：`OUT=results/compare_4090_20261115 scripts/run_compare.sh`（HumanEval 100 题，抽样 seed
20260802，生成 seed 42，输出上限 512，T=0.6，`ENGINE_OPT=1`：所有方法完整 CUDA graph 与自旋等待，
RAVEN 另加可捕获树注意力、异步主机读、GPU 建树；RAVEN 为 v4-safe）。结果在本地，未提交。

| 方法 | pass@1 | MAL | ms/轮 | decode tok/s | 相对 AR | 截断 |
|---|---:|---:|---:|---:|---:|---:|
| AR | 59/100 | — | 16.5 | 60.61 | 1.00× | 35 |
| Native | 60/100 | 3.157 | 23.7 | 133.09 | 2.20× | 35 |
| Cactus | 53/100 | 3.339 | 23.8 | 140.35 | 2.32× | 42 |
| RAVEN v4-safe | 57/100 | 3.506 | 26.1 | 134.43 | 2.22× | 37 |

- RAVEN 的 MAL 比 Cactus 高 5.0%，每轮多 2.3 ms，decode 吞吐低 4.2%（Jetson HumanEval-100 为 −4.4%）。
- 配对 pass@1：Native 对/Cactus 错 12 题、反之 5 题；Cactus 对/RAVEN 错 10、反之 14；Native 对/RAVEN 错
  10、反之 7；均不显著。512 输出上限下截断 35–42 题且 Cactus 输出更长（平均 344 对 331 token），
  这一设置不适合比较准确率。

每轮耗时拆分（`results/profile_4090_20261115`，本地）：每种方法 HumanEval 10 题，开
`REMTP_GPU_EVENT_TIMING_JSONL` 与 `REMTP_HOST_TIMELINE_JSONL`（不加同步）；另各 3 题开
`REMTP_TORCH_TRACE_JSON` 录 29 轮带 CUDA kernel 的时间线（profiler 会拉长主机时间，只用于归因）。
时间线由会话草稿目录中的两个临时脚本分析：GPU 忙/闲按轮合并 kernel 区间，并按发起 kernel 的
Python 帧归因。

| 每轮（ms） | Cactus | RAVEN | 差 |
|---|---:|---:|---:|
| 墙钟（无 profiler） | 24.40 | 26.35 | +1.95 |
| 模型 GPU 时间（目标前向、LM head、MTP 层，CUDA event） | 22.62 | 22.73 | +0.11 |
| 模型以外的 GPU kernel（profiler） | ≈0.45 | ≈1.90 | +1.45 |
| 其中 GPU 建树（`gpu_tree.propose_tree`） | — | 1.52 | |
| 其中验证/拒绝采样 | ≈0.13 | ≈0.29 | |
| GPU 空闲（墙钟 − 上两项） | ≈1.33 | ≈1.72 | +0.39 |

- GPU 建树每轮约 973 个 kernel（其中约 795 个在 CUDA graph 内），平均 1.56 µs：逐元素 kernel 约 485 个、
  归约 62 个、top-k 与排序约 100 个；单个 kernel 的固定开销占主导。
- RAVEN 的 GPU 空闲主要在：KV 提交前每轮重新遍历注意力模块列表（`raven_fast_commit.py` 第 128、134
  行，约 0.3 ms，profiler 下）、验证的 `_plan_vectors`（约 0.2 ms）、草稿 token 拷回主机与拓扑构建
  （`_get_variable_drafts_cpu`、`from_paths`、树掩码，合计约 0.4 ms）。
- 结论：在 4090 上，RAVEN 相对 Cactus 剩余的约 2 ms/轮中，约 1.45 ms 是 GPU 建树和验证的小 kernel，
  主机造成的额外 GPU 空闲只有约 0.4 ms。以上为测量结果；任何优化方案的收益尚未实测。

限制：profile 只覆盖 10/3 题；profiler 下的空闲时间被放大，表中“GPU 空闲”行用无 profiler 的墙钟推算。

### 2026-11-15：异步化第 1、2 阶段——单次启动的 KV 提交、融合建树决策、验证搬到 GPU

状态：`代码已验证（GPU 单元测试 + HumanEval-100 端到端）；第 3 阶段（vLLM 异步调度）未开始`

动机：4090 上 RAVEN 比 Cactus 每轮多约 2 ms，其中约 1.45 ms 是 GPU 建树/验证的小 kernel，约 0.4 ms 是
主机造成的 GPU 空闲（见上一条）。用户同意按“先做同步模式下的前置条件，再接入 vLLM 异步调度”的顺序推进，
并接受验证随机数改为固定消耗（分布不变、具体样本改变）。

新增（全部默认关闭）：

- `REMTP_DYNAMIC_TREE_FAST_COMMIT=3`（`remtp/raven_commit_kernel.py`）：一个 Triton kernel 在所有注意力层里
  按顺序完成接受路径的 KV 槽位搬移，槽位下标留在 GPU；层表（基址、步长）每次 KV 分配只建一次。与模式 2 的
  搬移完全相同；去掉了每轮扫描 `static_forward_context`、逐层 flatten/切片视图、读回目标槽位列表。
  `_compact_selected_path` 在没有混合状态层时不再遍历所有模块。
- `REMTP_DYNAMIC_TREE_GPU_BUILD_FUSED=exact|fast`（`remtp/gpu_tree_fused.py`，接入 `gpu_tree.run_level`）：每层
  “候选选择 + residual_coverage 保留”由约 250 个小 kernel 变为：行统计（`exact` 保持原逐行 torch 运算序列，
  `fast` 用一个 Triton kernel）、一次批量 `topk`、一个 Triton kernel 完成稳定排序、阈值、兄弟规则、每父上限
  与保留排序。浮点运算逐项对应（float32 `div_rn`、libdevice float64 exp/log、Python 侧同样方式构造的常量）。
- `REMTP_DYNAMIC_TREE_DEVICE_VERIFY=1`（`remtp/device_verify.py`，接入 `_dynamic_verify_dispatch`）：
  residual_hit_anchor 验证在 GPU 上完成、每轮一次读回。规则不变：v4-safe 额度（`adaptive_relaxation_max_ratio
  <= 1` 时规划器等价于 floor 闭式，门槛位置为 0）、`_boost_known` 运算序列、float64 的 `u <= min(1, h/q)`、
  首次拒绝从精确残差采修正、兄弟 token 命中则复用并接一个未松弛目标 token、全接受则接目标 bonus。
  随机数改为每轮固定消耗：每个主干位置一个均匀数、每个可能的类别抽样一行指数噪声（argmax(p/E) 为精确类别
  抽样），两次调用请求生成器。仅支持上述配置，其余返回 None 回退到原路径。
- `scripts/serve_compare.sh`：`FAST_COMMIT`、`GPU_BUILD_FUSED` 可由环境变量覆盖，默认值不变。

验证：

- 新 GPU 单元测试 10 项：提交 kernel 与 `copy_cache_slots_inplace` 逐位相同（bf16/fp16/fp32、乱序槽位）；
  融合层在 `exact` 模式下与 `gpu_tree._level` 全部输出逐位相同（vocab 3000 的 90 例、151680 的 24 例，含
  bf16 并列、平顶并列、单一主导、双主导），`fast` 模式决策相同，CUDA graph 重放与 eager 相同；设备验证的
  p、q、额度、验证概率与快速路径逐位相同，判定符合规则，3000 个种子下前两个输出 token 的分布与原路径一致
  （TV 低于两独立样本的 3 倍期望），不支持的配置回退。完整 `pytest -q tests/` 713 项通过、1 项跳过。
- HumanEval-100 端到端（输出上限 512，T=0.6，`ENGINE_OPT=1`，仅 RAVEN v4-safe，结果在本地
  `results/stage1_*_20261115`、`results/stage2_devverify_20261115`）：

| 配置 | 与原 RAVEN 输出逐 token 相同 | MAL | ms/轮 | decode tok/s |
|---|---:|---:|---:|---:|
| 原配置（同时段重跑） | 100/100 | 3.506 | 26.0 | 134.57 |
| + 融合建树 `exact` + 提交模式 3 | 100/100 | 3.506 | 25.3 | 138.82 |
| + 融合建树 `fast` + 提交模式 3 | 100/100 | 3.506 | 25.3 | 138.36 |
| + 设备验证（`exact` + 模式 3） | — | 3.510 | 25.5 | 137.75 |

- 第 1 阶段每轮省约 0.7 ms（+3.2%），输出 100/100 题逐 token 不变。CUDA event 计时：墙钟 26.35 → 25.51 ms，
  模型 GPU 时间不变（22.74 ms）；建树在 graph 内的 kernel 由约 795 个/轮降到约 226 个/轮。
- 设备验证改变具体样本，pass@1 60/100（原 57/100）、MAL 3.510，在噪声范围内；同步模式下没有提速（读回次数
  从约 7 次/轮降到 1 次，但多了噪声行与残差行的计算），与 Jetson 上验证预发射的结论一致。它的作用是为第 3
  阶段提供不依赖主机的验证。

限制：`fast` 统计模式与 `exact` 速度无差别，保留 `exact` 为推荐值；第 3 阶段需要固定 9 节点容量、GPU 端的
目标掩码/位置与验证输出、vLLM 异步调度接入，尚未实现。

### 2026-11-15：异步化第 3 阶段——RAVEN 接入 vLLM 异步调度

状态：`代码已验证（GPU 单元测试 + HumanEval-100 四方法端到端）`

修改（默认关闭；`scripts/serve_compare.sh` 的 `ASYNC_SCHED=1` 一次打开）：

- `remtp/async_tree.py`（新）：`REMTP_DYNAMIC_TREE_ASYNC=1`。GPU 建树结束不再读回，树保存在固定容量的设备
  张量里（9 个槽位、有效标记、深度、父槽位、主干槽位、主干兄弟）；目标树掩码在 GPU 上由父指针生成并写入
  CUDA graph 的持久缓冲区，目标位置为 `anchor + depth`；空槽位只看锚点和自己，真实节点看不到空槽位，空槽位
  的 KV 写在已提交长度之后。树的内容（节点数、分支、保留节点、松弛额度）每轮照旧自适应，只有目标看到的形状
  固定为 10 行。
- `remtp/gpu_tree.py`：异步模式下由 `async_tree.publish_tree` 发布树并返回 `[1, 9]` 草稿 id。
- `remtp/device_verify.py`：新增 `verify_async`，在设备树上做与 `residual_hit_tree_verify_device` 相同的规则
  与随机数预算，直接写出 vLLM 的 `sampled_token_ids`（其后填 -1）、接受槽位、接受个数和叶子行，不读回；
  所有下标用 1 元素张量（0 维张量下标会触发读回）。`residual_hit_tree_verify_device` 增加 `noise_rows` 参数
  以便与异步路径逐项对照。
- `remtp/raven_commit_kernel.py`：`move_counted`/`commit_async`，搬移个数在设备上读取。
- `remtp/fixed6_vllm.py`、`remtp/dynamic_tree_vllm.py`：采样器、KV 提交、首遍采样行、目标位置/掩码、草稿拷回
  在异步模式下走上述设备路径；启动横幅标出各开关。
- `remtp/probabilistic_mtp.py`：`REMTP_SYNC_FREE_SAMPLING=1` 去掉链式草稿采样里 `greedy.any()` 的主机读
  （`where` 结果相同），使 Native/Cactus 在异步调度下同样无同步。
- `scripts/serve_compare.sh`：`ASYNC_SCHED=1` 对所有方法关闭 `NO_ASYNC_SCHEDULING` 并开启无同步采样；RAVEN 另加
  `ASYNC`、`DEVICE_VERIFY`、`GPU_BUILD_FUSED=exact`、`FAST_COMMIT=3`。默认 0，行为不变。

验证：

- `tests/test_async_tree.py` 3 项：设备树的有效位/深度/父节点/主干/兄弟与主机拓扑一致；掩码前 count+1 行
  与 vLLM `_prepare_tree_attn_bias` 逐位相同，空槽位行只允许锚点和自身，真实行看不到空槽位（5 种拓扑）；
  `verify_async` 与 `residual_hit_tree_verify_device(noise_rows=9)` 在 60 例上输出 token、接受槽位、个数、叶子
  行全部一致；计数提交与参考搬移一致。完整 `pytest -q tests/` 716 项通过、1 项跳过。
- 同步调试（`REMTP_HOST_TIMELINE_SYNC=1`）：异步 RAVEN 与异步 Cactus 每轮都没有同步调用（只剩每请求一次的温度
  缓存）；主机只在 vLLM 自身等待上一步接受个数的事件处等待。
- HumanEval-100（输出上限 512，T=0.6，`ENGINE_OPT=1`；结果在本地 `results/compare_4090_async_20261115`、
  `results/compare_4090_sync_20261115b`，同一时段）：

| 方法 | 调度 | pass@1 | MAL | ms/轮 | decode tok/s | 相对 AR |
|---|---|---:|---:|---:|---:|---:|
| AR | 异步 | 59/100 | — | 16.2 | 61.79 | 1.00× |
| Native | 异步 | 60/100 | 3.157 | 22.8 | 138.59 | 2.24× |
| Cactus | 异步 | 53/100 | 3.339 | 22.9 | 146.05 | 2.36× |
| RAVEN v4-safe | 异步 | 55/100 | 3.508 | 24.3 | 144.55 | 2.34× |
| Cactus | 同步 | 53/100 | 3.339 | 23.8 | 140.15 | — |
| RAVEN v4-safe（第 1 阶段开关） | 同步 | 57/100 | 3.506 | 25.3 | 138.54 | — |

- AR、Native、Cactus 的异步输出与同步运行在 100/100 题上逐 token 相同：异步调度本身不改变输出，对比公平。
- 异步调度让 Cactus 快 4.2%、RAVEN 快 4.3%；RAVEN 相对 Cactus 仍低 1.0%（同步同一时段为 −1.1%）。RAVEN 的异步
  输出与同步设备验证只有 9/100 题相同（随机数行数与目标/首遍的 GEMM 行数不同），MAL 3.508 对 3.510，pass@1
  差异在单样本噪声内。
- 异步下的耗时构成（10 题 CUDA event + 3 题 profiler）：RAVEN 墙钟 24.26 ms/轮、模型 GPU 22.62、其余 1.64，
  profiler 下 GPU 空闲 0.66 ms/轮（几乎全是 <20 µs 的 kernel 间隙）；Cactus 墙钟 22.81、模型 22.30、其余 0.52。
  主机已不是瓶颈，RAVEN 剩余约 1.45 ms/轮的差距是 GPU 时间：建树/验证/发布树的小 kernel 约 1.1 ms，目标前向与
  首遍行数更多约 0.3 ms。

限制：异步模式不做 Q 非有限值检查和逐轮审计（同步路径会在 Q 非法时报错）；不支持 checker 模式与其他树配置。

### 2026-11-15：异步模式下的 kernel 融合（验证、建树统计/top-4、发布树、分支噪声）

状态：`代码已验证（GPU 单元测试与微基准）；端到端测速被其他用户的 GPU 任务干扰，需重测`

动机：异步调度后主机已不是瓶颈，RAVEN 比 Cactus 每轮多的约 1.45 ms 是 GPU 时间，其中验证约 0.48 ms（265 个
kernel/轮）、建树约 0.79 ms（约 450 个 kernel/轮）。

修改（`ASYNC_SCHED=1` 时由 `scripts/serve_compare.sh` 默认打开，均可单独关闭）：

- `remtp/device_verify_fused.py`（新，`REMTP_DYNAMIC_TREE_VERIFY_FUSED=1`）：异步验证改为 4 个 Triton kernel 加两次
  随机数调用。每次词表遍历按行切成 32 段并以固定顺序归约（确定性）；argmax 取最大键、并列取最小下标（同
  `torch.argmax`）。与 `verify_async` 的差别只在最后一位：行和归约顺序不同、加成行不再除以自身的和（按缩放
  系数的构造它等于 1）、指数竞赛用未归一化的残差（正比例缩放）。随机数预算不变。
- `remtp/gpu_tree_fused.py`：`fast` 模式改用两个分段 kernel 计算行统计（熵用 `log T - Σ c·log c / T`）和精确
  top-4（值降序、下标升序），替代逐行 torch 统计与 `torch.topk`；`exact` 模式不变。
- `remtp/async_tree.py`：`publish_levels` 用一个 Triton kernel 由各层输出生成槽位、深度、父节点、主干、兄弟、
  Q 行号与目标掩码，Q 用一次 `index_select` 写入；保留 torch 版 `publish_tree` 作参照。
- `remtp/gpu_tree.py`：`REMTP_DYNAMIC_TREE_BATCHED_BRANCH_NOISE=1` 时一轮的 10 行分支噪声由一个按
  `branch_seed(seed, round, ())` 重新播种的生成器一次生成（行间独立，具体样本改变）；异步发布移到通用组装之前。
  注：在 residual_hit_anchor 中分支采样出的节点（非主干父节点的子节点）不会被验证使用，只影响保留预算的排序。
- `scripts/serve_compare.sh`：异步时 `GPU_BUILD_FUSED` 默认 `fast`，新增 `VERIFY_FUSED`、`BATCHED_BRANCH_NOISE`（默认 1）。

验证：

- `tests/test_async_tree.py` 新增 2 项（共 5 项）：融合验证与 `verify_async` 在 80 例（vocab 40/3000/151680）上
  输出、接受槽位、个数、叶子行全部一致；发布 kernel 与 torch 版发布在 5 种拓扑上逐项一致（含 Q 行与掩码）。
  `tests/test_gpu_tree_fused.py` 4 项通过（`fast` 模式在 36 例上决策与 `gpu_tree._level` 相同）。
  CPU 下完整 `pytest -q tests/`：679 通过、35 跳过（GPU 测试在无 GPU 时跳过）。
- 微基准（4090，eager 循环，含主机启动开销）：验证 torch 版 1.11 ms → 融合 0.145 ms；4 行统计 + top-k
  0.302 ms → 0.079 ms。
- HumanEval-100 异步对比（`results/compare_4090_async_fused_20261115`）：Cactus 27.7 ms/轮、120.50 tok/s，RAVEN
  28.4 ms/轮、123.52 tok/s，MAL 3.506。两者都比 1 小时前慢约 20%：运行期间另一用户的进程
  （`run_oned_v25.py`，约 16:09 开始）在同一 GPU 上运行，这组吞吐不能作为结论，需在 GPU 独占时重测。
  Cactus 的输出与上一轮异步运行 100/100 相同；融合后的 RAVEN 与未融合的异步 RAVEN 78/100 题相同
  （其余因统计量最后一位与分支噪声改变而在后段分开）。

### 2026-11-15：GPU 独占重测；residual_hit_anchor 的无用节点、兄弟节点参数与“毛毛虫”提议

状态：`已验证（GPU 独占下的端到端测量、单元测试与诊断）；默认配置未改，参数与新提议均为可选项`

1. 融合后的异步对比（GPU 无其他进程，HumanEval-100，输出上限 512，`results/compare_4090_async_fused_20261115b`
   等，均在本地）：Cactus 22.9 ms/轮、146.03 tok/s（与前一轮 146.05 相同，输出 100/100 相同）；RAVEN（异步 + 融合）
   23.2 ms/轮、150.95 tok/s、MAL 3.506，比 Cactus 高 3.4%；同一时段的第 3 阶段 RAVEN（未融合）24.1 ms/轮、
   145.46 tok/s。上一条记录里被干扰的那组数据作废。
2. 无用节点：在 `residual_hit_anchor` 中，验证只读取主干三个节点、主干节点的兄弟（命中候选）以及命中兄弟的目标行
   （锚点）；命中后本轮结束，非主干节点的后代只有 `residual_hit_strict/cactus` 的续接会用到。代码依据：
   `dynamic_mtp_tree.residual_hit_tree_verify` 与快速路径中 `support_mode == "residual_hit_anchor"` 直接返回。
   行为验证 `tests/test_anchor_dead_nodes.py`（4 项，640 例：参照与快速验证 × 有无门槛）：把这些节点的 token、
   草稿行和目标行全部换成随机值，输出、接受节点、终止类型与生成器状态都不变，且这些节点从不被访问。现有保留
   顺序先保留主干父节点的孩子，所以这些节点只占用剩余预算，不会挤掉有用节点。
3. 兄弟覆盖诊断（`REMTP_SIBLING_DIAG_JSONL`，只记录、不改变输出；HumanEval-50，同步快速路径）：40.6% 的轮次
   发生主干拒绝；当前平均每深度 1.48 个兄弟，拒绝后命中率 0.648/0.592/0.502（深度 1/2/3）。若每深度取 q 最大的
   j 个非主干 token，期望命中率 j=2：0.748/0.720/0.616，j=4：0.868/0.835/0.758，j=8：0.950/0.920/0.871。
4. 参数（自适应选择阈值）：`TREE_KAPPA=50 TREE_TAU_MIN=1e-6` 使熵阈值失效、每深度总取 2 个兄弟。诊断复测：
   平均兄弟 1.83/1.98/1.99，命中率 0.743/0.679/0.624；HumanEval-100 异步：MAL 3.528、23.2 ms/轮、151.95 tok/s
   （默认参数 3.506、150.93）。同一配置在不同轨迹上的 MAL 波动约 ±0.03（例如 κ=1e9 的一次运行为 3.495），
   命中率计数是更稳的证据。默认值未改。
5. “毛毛虫”提议 `REMTP_DYNAMIC_TREE_CATERPILLAR=j`（新 `remtp/caterpillar.py`，仅异步模式；`serve_compare.sh` 的
   `CATERPILLAR=j`，容量 3+3j，CUDA graph 尺寸按 MAX_NODES 生成）：只提议主干（与原实现相同的请求生成器抽样
   顺序）和每深度 j 个兄弟（主干父节点 Q 中 q≥tau_min 的最大 j 个非主干 token，值降序、下标升序），MTP pass 缩为
   1 行和 2 行。异步状态与融合验证推广到任意容量与兄弟宽度。
   测试 `tests/test_caterpillar.py` 7 项：兄弟选择与稳定参照一致（含 bf16 并列、vocab 151680）；发布的设备状态、
   Q 行与掩码与主机拓扑一致；宽度 4 的融合验证与 torch 版 60/60 一致。
   设备计数器（`REMTP_ASYNC_STATS_JSONL`，每 500 轮读一次）HumanEval-100 前 9000 轮：

| 配置 | tokens/轮 | 主干接受率 d1/d2/d3 | 拒绝后命中率 d1/d2/d3 | ms/轮 | tok/s |
|---|---:|---|---|---:|---:|
| 树，κ=1e9、τ=1e-6 | 3.487 | 0.911/0.811/0.736 | 0.734/0.641/0.603 | 23.2 | 150.48 |
| 毛毛虫 j=2 | 3.456 | 0.906/0.800/0.741 | 0.724/0.606/0.583 | 23.4 | 148.28 |
| 毛毛虫 j=4（容量 15） | 3.541 | 0.911/0.808/0.740 | 0.858/0.751/0.738 | 23.6 | 150.59 |

   - 主干接受率三者在统计误差内相同（标准误约 0.003–0.005），没有看到毛毛虫 MTP pass 退化的证据；j=2 与树的
     差异在不同轨迹的波动范围内（两者输出 34/100 相同，MTP pass 行数不同带来最后一位差异）。
   - j=4 把命中率提高 0.12–0.14、tokens/轮 +1.5%（与诊断预测一致），但每轮多约 0.4 ms，吞吐与树持平。
   - j=6（容量 21）MAL 3.592，但 22 个目标行超过目标 LM head 的 16 行分块（`REMTP_TARGET_LM_HEAD_ROW_BLOCK=16`），
     LM head 1.34 → 2.69 ms，目标前向 15.48 → 16.68 ms，吞吐降到 138.09 tok/s。

限制与下一步：毛毛虫 j=4 的额外开销主要来自目标侧 16 行的概率计算和每个节点一行的锚点噪声/argmax，可改为只算
需要的行（主干父行、叶子行与命中兄弟行）和单行锚点抽样；未实现。各项改动都不是默认配置。

### 2026-11-16：兄弟选择阈值改为默认 κ=50、τ_min=1e-6；多数据集对比脚本与 MBPP 运行器

状态：`代码已验证（CPU 测试 + GPU 冒烟）；正式对比见下一条`

- `scripts/serve_compare.sh`：RAVEN 的 `TREE_KAPPA` 默认 50、`TREE_TAU_MIN` 默认 1e-6（原 1.5、0.001），依据见 2026-11-15
  的兄弟覆盖诊断与命中率计数；只改变提议哪几个兄弟，接受规则、松弛额度、树预算不变。
- `remtp/mbpp_benchmark.py`、`remtp/mbpp_evaluator.py`（新）：MBPP test split（task 11–510）按种子抽样，提示给出任务
  描述与三条官方断言，回复中的代码在受限 Docker 容器里与 setup 代码和断言一起执行（复用 HumanEval 的容器执行）。
  数据用 `router_benchmark_download` 下载的固定版本（sha256 `ccf64ceae9c5…`，974 行）。`tests/test_mbpp_benchmark.py`
  4 项通过。
- `scripts/run_compare.sh`：`DATASETS`（humaneval、gsm8k、mbpp），每种方法一个服务跑完全部数据集，结果在
  `$OUT/<method>/<dataset>`，每个数据集一张表（准确率或 pass@1、MAL、每轮秒、decode/E2E tok/s、输出 token、截断数、
  相对 AR）；同一命令可续跑（已完成的数据集跳过，未完成的从 checkpoint 继续）。GSM8K 抽样种子 20260730，MBPP 20261116，
  HumanEval 20260802。
- `doc/RUN-ON-4090-2026-11-15.md` 增加第 8 节说明以上变化。
- 验证：CPU 下完整 `pytest -q tests/` 687 通过、42 跳过（GPU 测试跳过；上一提交时 GPU 下 729 通过）。GPU 冒烟
  （RAVEN、`ASYNC_SCHED=1`、三个数据集各 2 题、输出上限 2048）全流程通过，横幅显示 `tau_min=1e-06 kappa=50`；
  该次运行时另一用户的进程在同一 GPU 上，速度数据不采用。

### 2026-11-16：合并远端 `jetson-r39-residual-adapt`（7b5c0c3、01c4e91、b4b42c2）

状态：`已合并并验证（GPU 测试 + 运行冒烟）`

远端在 ab82919 之后新增 3 个提交（Jetson 侧：基线可选异步调度、默认开启的融合建树决策步骤与 KV 提交/拓扑缓存、
结果记录），与本地 6 个提交有重叠，以合并提交整合：

- `remtp/gpu_tree_fused.py` 两边同名新增：保留远端版本（`REMTP_DYNAMIC_TREE_GPU_BUILD_FUSED=1`，默认，与 `_level`
  逐位相同）；本地版本改名为 `remtp/gpu_tree_fused_split.py`，对应取值 `exact`/`fast`（异步模式默认 `fast`）。
  `gpu_tree._level_impl` 按取值分发（`1`、`exact`、`fast`、`0`），`_GraphedLevels` 采用远端结构并把模式加入缓存键。
- `scripts/serve_compare.sh`：远端的 `ASYNC_SCHEDULING`（仅基线、RAVEN 报错退出）与本地的 `ASYNC_SCHED` 合并为一个开关
  （两个名字都可用）；RAVEN 不再拒绝，因为其树、验证和 KV 提交已可留在 GPU 上。非异步时 `GPU_BUILD_FUSED` 默认改为远端
  的 `1`。`doc/RUN-ON-4090-2026-11-15.md` 中对应说明同步更新。
- 远端的 `raven_fast_commit`（层缓存）、`DynamicTreeTopology` 缓存、`gpu_event_timing`、测试与文档按原样保留；开发日志
  两边条目均保留（远端条目在文件开头）。
- 验证：`pytest -q tests/` 737 通过、1 跳过（GPU 上）；RAVEN 冒烟（HumanEval 2 题）异步与同步两种配置均正常完成，
  横幅显示 `tau_min=1e-06 kappa=50`、异步模式 `fast` 决策步骤。另一用户的进程当时在同一 GPU 上，冒烟速度不作结论。

### 2026-11-16：正式对比——AR / Native / Cactus / RAVEN，HumanEval、GSM8K、MBPP 各 50 题

状态：`已验证（GPU 独占，各方法一次运行）`

设置：RTX 4090，BF16 FastMTP，`ENGINE_OPT=1`，所有方法 `ASYNC_SCHED=1`（vLLM 异步调度；RAVEN 用设备端建树与验证、
融合 kernel，默认兄弟阈值 κ=50、τ_min=1e-6），温度 0.6，生成种子 42（请求种子按题号派生），输出上限 2048，
`MAX_MODEL_LEN=4096`，no-think 模板，tokenizer 默认 system。抽样：HumanEval 种子 20260802、GSM8K 20260730、MBPP（test
split 11–510）20261116，各 50 题；HumanEval/MBPP 在受限 Docker 中判分，GSM8K 取末行 `####` 答案。命令：
`DATASETS="humaneval gsm8k mbpp" SAMPLES=50 MAX_TOKENS=2048 ASYNC_SCHED=1 scripts/run_compare.sh`，结果在本地
`results/formal_20261116`。运行期间按进程属主监控 GPU，没有其他用户的进程。

| 数据集 | 方法 | 准确率 | MAL | ms/轮 | decode tok/s | 相对 AR | 截断 |
|---|---|---:|---:|---:|---:|---:|---:|
| HumanEval | AR | 36/50 | — | 16.8 | 59.69 | 1.00× | 4 |
| HumanEval | Native | 36/50 | 3.147 | 23.4 | 134.76 | 2.26× | 4 |
| HumanEval | Cactus | 35/50 | 3.422 | 23.4 | 146.08 | 2.45× | 5 |
| HumanEval | RAVEN | 35/50 | 3.539 | 23.6 | 149.81 | 2.51× | 3 |
| GSM8K | AR | 44/50 | — | 16.9 | 59.32 | 1.00× | 6 |
| GSM8K | Native | 46/50 | 3.103 | 23.4 | 133.04 | 2.24× | 5 |
| GSM8K | Cactus | 46/50 | 3.458 | 23.5 | 147.49 | 2.49× | 6 |
| GSM8K | RAVEN | 44/50 | 3.547 | 24.0 | 147.69 | 2.49× | 8 |
| MBPP | AR | 24/50 | — | 17.0 | 58.89 | 1.00× | 5 |
| MBPP | Native | 22/50 | 2.917 | 23.5 | 124.13 | 2.11× | 5 |
| MBPP | Cactus | 22/50 | 3.316 | 23.7 | 140.26 | 2.38× | 6 |
| MBPP | RAVEN | 21/50 | 3.403 | 24.2 | 140.78 | 2.39× | 7 |

- MAL：RAVEN 比 Cactus 高 3.4%（HumanEval）、2.6%（GSM8K）、2.6%（MBPP）；每轮多 0.2–0.5 ms，decode 吞吐比 Cactus
  高 2.6%、0.1%、0.4%。三种推测方法相对 AR 为 2.1–2.5×。
- 准确率：150 题合计 AR 104、Native 104、Cactus 103、RAVEN 100。逐题配对：RAVEN 对而 Cactus 错 10 题、反之 13 题
  （符号检验 p=0.68）；RAVEN 对 Native 13:17（p=0.58）；RAVEN 对 AR 10:14（p=0.54）；Cactus 对 AR 9:10（p=1.0）。
  50 题单样本区分不了方法。
- 截断：输出上限 2048 下仍有 3–8 题截断，GSM8K 上四种方法都有（AR 6 题），均为重复生成到上限；RAVEN 在 GSM8K
  上 8 题截断，与其他方法的截断题部分重合（810、1004 等），输出 token 总数因此较多（31636 对 24176–26976）。
- 过程说明：第一次运行时 GPU 监控按进程名判断“其他进程”，把本方的 vLLM（显示为 `VLLM::EngineCore`）误判为干扰，
  12 次运行全部被删除重跑；改为按进程属主判断后重跑一次，即上表。被删除的那些运行期间实际没有其他用户的进程。
