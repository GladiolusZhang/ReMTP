# 相对 research/residual-aligned-tree-mtp 的本机改动

## 2026-10-07：全项目审查与优化优先级调整

- 新增CPU离线审查工具、中文全项目审查和有限优化计划；具体文件、验证和边界见[专门记录](project-review/2026-10-07.md)。
- 两次CPU审查通过：每次29帧/58 verifier、16 Q/32 selector；节点224→188但无GPU加速证明；605轮EOS/重复信号与小词表条件分布枚举完成。
- 更新当前入口与历史notice；当前三基准正式质量/速度目标尚未达到。算法、模型、默认数值设置、评分器和旧冻结未改，本次新GPU调用0。

## 2026-10-04：深入复查第三轮失败（诊断完成，正式优化未恢复）

- 专门记录：[Target深入复查改动](target-probability-consistency/deep-audit-2026-10-04.md)。
- 诊断CLI新增真实manifest/hash校验和canonical参数，相关7项测试通过；不改正式算法或数值开关。
- 60条有限GPU对照已结束：原生24、新鲜D3/D4 root16、祖先16、原old/head1新鲜root4。
  新鲜条件通过，历史287长状态仍差0.25；具体首次差异层/提交操作待定位。
- 调研固定PyTorch2.10/vLLM0.18版本，本机验证False不关闭split-K；全局不变模式不能直接用于本TREE/sm87路径。
- 新中文调查报告及逐步长状态修复计划；旧文件加最新入口，冻结失败结果不覆盖，round3未重启。

## 2026-10-02：质量校准结论与下一轮计划

- 第二轮停止更新：9 次诊断完成，同配置 seed=73153 输出 592→1700 token，
  `STOPPED_PROTOCOL_ERROR`、exit=2、NRestarts=0。下文“GPU 已加载/后台运行”是启动时状态。
  新建 `doc/ANALYSIS-ADAPTIVE-RAVEN-ROUND2-2026-10-02.md`，补充 Q 最早分叉、
  EOS 质量变化和重复检测覆盖分析；更新计划、实施记录及总日志。本次未重启或改算法。
- 执行更新：新增 `remtp/generation_guard.py`、`remtp/optimization_metrics.py`、
  `scripts/run_adaptive_round2.py`、`tests/test_generation_guard.py`、
  `deploy/remtp-adaptive-round2.service` 与本轮中文实施/结果入口。
  verifier/adapter/worker 新增默认关闭的 EOS/重复保护、独立公共 head 开关和前缀/RNG 观测。
  保护位置不回收 TV，接受/残差一致；旧配置在请求局部 RNG 新协议下重新对照。
  工作区 CPU 回归实际 364 passed、2 skipped，编译、diff、systemd 与冻结协议预检通过。
  GPU 已加载，health=200，三个 warmup 同输入 head 探针产生结果；完整质量尚未完成。
  计划执行状态和研究日志同步；新报告 `doc/RESULT-ADAPTIVE-RAVEN-ROUND2.md` 链接到
  `results/adaptive_optimization_20261002/report.md`，旧 v4 证据保留。
- v4 的 96 次校准全部结束：原评测 41→42、正常结束且判对 41→40、截断 7→8；
  平均请求耗时降低 10.1%。质量门槛失败，控制器按设计退出；未采用为默认。
- 新建 `plan/adaptive-raven-next-optimization-2026-10-02.md`：退化前缀回放、实现/树深消融、
  EOS 与重复风险保护、小范围选参、独立留出校准、三个主基线公平多种子对照。
- 更新旧计划与 `doc/OPTIMIZATION-ADAPTIVE-RAVEN-2026-10-01.md` 的历史状态提示，
  同步 `docs/research_and_development_log.md`。下文的“待质量验收/运行中”是此前快照。
- 制定计划时验证仅为已落盘结果核算、代码路径与文档检查；之后的执行状态见本节执行更新。

基线：`3afafa1`，本地分支 `jetson-r39-residual-adapt`。当前状态与验证见 [`RESULT-RAVEN-R39.2.1.md`](../RESULT-RAVEN-R39.2.1.md)。

| 文件 | 改动 | 验证状态 |
| --- | --- | --- |
| `scripts/install.sh` | R39 ARM64 用 uv 安装 vLLM 0.18.0+cu130、PyTorch 2.10.0+cu130 和 bitsandbytes 0.50.2；缺 Python 头文件时只解包到用户目录。 | 独立 `.venv` 已安装并导入；脚本复跑通过。 |
| `scripts/cuda_env.sh` | R39 缺 `Python.h` 时设置用户目录头文件路径。 | `bash -n` 通过；方案沿用原工作目录的实测配置。 |
| `scripts/serve_fastmtp_verified.sh` | 可选地传递 `QUANTIZATION`、`LOAD_FORMAT`、`KV_CACHE_MEMORY_BYTES`；目标 4 位加载时为 BF16 草稿层显式设置 `draft_load_config=auto`；默认调用不变。 | Target、Native、RAVEN 三种 GPU HTTP 请求通过。 |
| `scripts/prepare_fastmtp_jetson.sh` | 规范化本地 FastMTP `tokenizer_class`，保留原始配置副本；不改权重。 | 完整测试 340 通过、2 跳过；GPU 模型加载通过。 |
| `scripts/serve_fastmtp_jetson.sh` | 添加 4 位、256 MiB KV、1024 token 上下文及 RAVEN 参数入口。 | Target、Native、RAVEN 启动与请求通过；逐轮审计确认 RAVEN 执行。 |
| `remtp/jetson_bf16_draft.py`、`remtp/fastmtp_verified_worker.py` | 仅目标 4 位时将 MiMo 草稿层的构造量化配置暂置空，使 BF16 草稿权重按原始形状加载。 | 初次 Native 启动形状断言失败；修复后 Native 和 RAVEN 均通过 GPU HTTP 请求。 |
| `README.md`、`plan/raven-jetson-r39.md`、`doc/RESULT-RAVEN-R39.2.1.md` | 记录设备入口、适配计划、实测证据和剩余限制。 | 与正式 GSM8K 五方法各 100 题结果及本机服务状态核对。 |
| `scripts/run_jetson_raven_gsm8k.sh` | 新增设备版 GSM8K 五方法固定抽样入口；保留原评测 prompt、seed、温度、答案解析和断点，记录 KV/上下文协议，预检输入输出长度；增加严格答案指标，支持中断续跑。报告生成改用中文，并加入正常结束且判对、单题耗时、同题配对和树审计分析；重生成报告时保留原完成时间。 | `bash -n` 通过；五方法各 100 题完成；中文报告从既有结果重新生成，未重跑模型。 |
| `deploy/remtp-raven-gsm8k.service`、`scripts/status_jetson_raven_gsm8k.sh` | 用户级 systemd 控制器及进度查询；原版 2048 输出上限、4096 上下文、1 GiB KV，失败自动重启并按题断点续跑。 | 本机 user systemd 与 lingering 可用；最后一组写出后故障自动重启，最终 `Result=success`、服务自动停止。 |
| `docs/research_and_development_log.md` | 按分支工作规则追加本工作单元记录。 | 本记录。 |
| `doc/ANALYSIS-RAVEN-GSM8K-2026-10-01.md`、计划和结果文档 | 从上次正式记录重新核算轮成本、无用备选后代、实际 TV 与错误案例，形成优化与复测顺序。 | 完整读取两组树审计；256 条记录、4 次轻量审计回放；未改算法或重跑正式实验。 |
| `plan/adaptive-raven-optimization.md`、适配计划与诊断文档 | 细化正确率优先的优化顺序、阶段验收、独立校准/测试、三个主基线及多种子对照、无人值守归档。 | 文档与现有实验记录核对；实施及新的 GPU 测试待开始。 |

NF4 动态量化曾通过隔离的 `/tmp` 诊断入口启动 Target 与 RAVEN，但短题仍截断；该实验入口未保留在仓库，正式路径维持已验证的默认 FP4 加载。

## 2026-10-04：执行Target一致性修复（固定前缀验收通过，优化已恢复）

专门记录见 [Target概率一致性改动](target-probability-consistency/README.md)。
新增有界Target探针、固定前缀/祖先路径回放、可选公共数值适配；控制器前缀固定与来源记录修复。
已完成默认/组合/row64/TRITON/统一2D五组各48原生请求，树真实多组root和12祖先回放；
cache-off root通过但祖先仍有0.1875残差，新增本地可选FP32 P×V kernel正在验收。
CPU376 passed/2 skipped；修改日志和状态文档添加历史提示，新中文修复报告已建立。
最终新增canonical祖先索引，不复制/写入KV；24 root项和16祖先评分全部通过，top/EOS差均0。
379 CPU通过/2跳过；两个关键失败祖先前后实际prompt逐token一致。
新round3用户级unit核验gate后已启用并启动，公共数值配置公平应用所有基线。
完整2048复现、质量、截断与正式加速仍由后台继续验证，门槛0.15保持不变。

## 2026-10-04：修复后诊断状态核查（仅文档，无新模型运行）

- 后续新增 `plan/target-probability-consistency-2026-10-04.md`：固定4例/评分对象、原生形状和缓存对照、
  树状态/逐层定位、最小修复及验收，再恢复正式优化；更新总计划入口和研究日志。
  已阅读本地采集/比较/采样/压紧实现，发现旧case选择依赖glob且缺来源hash，
  纳入首步诊断可靠性修复。仅制定行动方案，没有改算法、阈值或启动GPU。

- 新增 `doc/STATUS-ADAPTIVE-RAVEN-2026-10-04.md`，核实服务在10月2日15:17:27北京时间
  Target树/顺序概率门槛失败退出；完整输出重放通过，10生成请求和4前缀评分检查已完成。
- 从原始逐请求记录重算：四题两配置均4/4正常答对、0截断；候选吞吐+16.5%，
  平均时延+6.6%，输出token+24.3%。不合并128-token严格短诊断的截断/错误进入主质量统计。
- 为修复说明、实施记录、计划添加可见历史提示，同步研究日志。
  原始输出、冻结代码、协议、控制器保持不变；没有重启后台、改容差或声称正式目标达成。

## 2026-10-02：MTP 草稿有效前缀长度修复（7请求GPU短回放通过，长回放进行中）

| 文件 | 改动 | 实际验证 |
| --- | --- | --- |
| `remtp/fixed6_vllm.py` | 草稿metadata以root+1构造有效前缀，排除上一轮拒绝节点；同步长度/CPU影子 | 真实失败快照定位；CPU回归及7请求GPU短回放通过，长检查进行中 |
| `remtp/generation_guard.py` | 默认关闭、每请求前8提案真实hidden/Q/KV/RNG快照 | 旧版本缓存开/关和重启7个GPU请求实际完成 |
| `scripts/serve_fastmtp_verified.sh` | 新增显式前缀缓存开关 | shell语法检查及实际服务开/关对照通过 |
| `scripts/run_adaptive_round2.py` | 记录缓存策略、诊断开关；恢复时拒绝跳过已保存的失败顺序检查 | 编译与冻结预检通过；新后台长回放已启动 |
| `scripts/investigate_adaptive_repro.py` | 独立短回放及Q分布对照门槛，不作为质量统计 | 旧7请求诊断完成，新7请求修复检查全部通过 |
| `tests/test_draft_tree_metadata.py` | 不同树大小、初次prefill、改变拒绝KV槽内容时注意力输出不变 | 6 passed；修改前完整CPU 369 passed、2 skipped |
| `doc/FIX-ADAPTIVE-RAVEN-REPRODUCIBILITY-2026-10-02.md`、研究日志 | 中文根因、修复、证据与限制 | 已与本机快照和日志核对 |

收尾实测：修复冻结版7请求全部完成，cache-on/off同进程与cache-off重启均输出相同，
各自前8次提案Q最大TV=0、root hidden差0；cache-on额外24次树hidden/KV差0，RNG一致。
新 `deploy/remtp-adaptive-round2-fixed.service` 已启动，冻结目录/新协议与旧失败记录分开；
完整2048复现门槛前移并保存独立gate，失败不能进入选参。新报告链接
`doc/RESULT-ADAPTIVE-RAVEN-ROUND2-FIXED.md`；长回放、正确率、截断与正式收益尚未完成。

## 2026-10-01：Adaptive RAVEN 优化候选（待质量验收）

| 文件 | 改动 | 当前验证 |
| --- | --- | --- |
| `remtp/dynamic_mtp_tree.py`、`remtp/dynamic_tree_vllm.py` | 可选裁剪 anchor 不可使用的后代；可选精确稳定 topk，保留接受规则和候选选择语义 | CPU 输出/RNG/诊断等价检查通过；GPU 裁剪短请求已完成，实际测速进行中 |
| `remtp/runtime_profile.py`、worker | 默认关闭的同步分段计时，明确嵌套关系 | GPU baseline/candidate 两次 profile 实际完成 |
| `scripts/serve_fastmtp_verified.sh` | 支持显式空审计路径 | shell 检查通过；GPU 测速无审计 |
| `scripts/run_adaptive_optimization.py`、`deploy/remtp-adaptive-optimization.service` | 锁定冻结代码、逐题断点、速度和质量门槛、有限重试 | systemd 已启动；尚未完成长质量测试 |
| `tests/test_dynamic_mtp_tree.py` | 新增裁剪等价与稳定 topk 并列/预算检查 | 树测试 63 passed；初版全套 345 passed、2 skipped |
| `doc/OPTIMIZATION-ADAPTIVE-RAVEN-2026-10-01.md`、计划、研究日志 | 中文记录实施、证据、限制与进度 | 随运行更新，尚无最终正确率结论 |

后续实测：裁剪 v1 的短测吞吐 -3.29%，不采用；首版 topk v2 算子变慢，主动停止。
v3 增加无并列快速路径，并提供默认关闭的 `REMTP_DYNAMIC_TREE_ANCHOR_ONLY_PROPOSAL`，
复用已有主路径/direct sibling 构树器，省掉备选后代的 Q 计算，保留验证规则。
该项改变候选分配，必须重新校准质量。N=3/6/9 模拟检查主路径 Q/token/RNG 一致；
当前全套 354 passed、2 skipped。GPU 快速 topk 20 案例通过，v3 纯测速与质量待完成。

v3 随后的完整短测为 -4.73%，未进入质量。v4 在 anchor-only worker 内将单行
`MiMoMTP.compute_logits` 输入复制为 3 行、返回原行数，并测试 D4/N12；主基线未安装
此补丁，短测 Cactus 使用相同 D4。收益必须在后续消融与共享 head 优化控制中拆分。
新增真实 CUDA TreeAttention/KV 搬运诊断通过；当前全套 356 passed、2 skipped。
v4 普通短测正在完成，初步 E2E 2.648 → 2.941 token/s；完整质量尚未开始。

后续状态：v4 短测门槛已通过，两次 baseline 合并 2.6430 → candidate 2.9413 E2E tok/s
（+11.29%）；Cactus D4 2.6138。已自动进入 24 train 题 × 2 seed 的优化前后质量对照，
共 96 请求，输出上限 2048。中文自动报告经 `doc/RESULT-ADAPTIVE-RAVEN-OPTIMIZATION.md`
本机链接查看。当前没有新的完整正确率结论；各开关仍为候选验证入口。
