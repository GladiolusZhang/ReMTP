# 执行全项目复审计划：有限证据单元

> 完结更新（2026-10-08）：原位置故障修复后续跑正常结束，74主HTTP全部完成，64校准已分析；五个raw门槛和B16输出签名复现通过，三个质量门槛未过。[中文结果](../../RESULT-RAVEN-REVIEW-EVIDENCE-2026-10-08.md)。以下21点进展为历史记录，当前无后台实验；原失败/冻结保留。

日期：2026-10-07（UTC+8）。状态：**四方法计时及五前缀raw门槛通过；原观测故障保留，独立续跑后台执行中。**

> 21:38修复后实机更新：第五个最终chunk位置/输入正确，fresh raw逐位一致；完整五前缀门槛全部通过。Target-only第一条1532/seed42、2048预算生成中，后续64校准待执行。范围/新增预算见[观测修复记录](probe-prefill-fix-2026-10-07.md)。以下21:22为旧单元原始停止记录。

> 21:22停止：第五个2223-token前缀的probe误保存第一段位置2047，真实HTTP200；前四个raw完全一致，末段完整gate未形成。[故障修复与新增预算](probe-prefill-fix-2026-10-07.md)。下面后台进展为原单元历史记录，原冻结/失败保持。

## 实际执行状态

- 21:08:40 UTC+8启动 `remtp-raven-review-20261007.service`，目录 `results/raven_review_unit_20261007`；冻结后CPU预检通过，systemd active/running、NRestarts=0、Linger=yes。
- 21:14核验：Native和Cactus各完成健康检查、8-token预热及308/641两条64-token真实请求；各保存32轮真实分段计时，覆盖检查通过。主HTTP4/74、数值0/10、预热2/9，SpecCascade加载中；无OOM/崩溃证据。
- 运行模型约6.11GiB，15W模式2；Native运行时系统available约2.0GiB、swap0。计时带观测开销，不能作为无探针方法吞吐。
- [实时中文报告](../../RESULT-RAVEN-REVIEW-LIVE.md)由控制器更新；`bash scripts/status_raven_review_unit.sh`可查当前阶段、具体题/种子、心跳和预算。
- 新优化候选尚未选择。晚段数值一致性、Target-only两条完整生成及64条校准仍待结果，不提前宣称质量/速度达标。

### 21:19更新：四方法真实分段计时完成

8条注册profile请求全部完成，四服务各32轮；B的真实`propose_tree`覆盖通过。离线汇总保存在`results/raven_review_unit_20261007/profile_overview.json`。

| 方法 | 观测轮平均墙钟ms | 观测轮平均提交token |
| --- | ---: | ---: |
| Native | 1139.25 | 2.344 |
| Cactus | 1139.98 | 3.125 |
| SpecCascade | 1141.41 | 2.969 |
| B / Adaptive RAVEN | 1271.22 | 3.281 |

- B的Target forward/logits stream跨度约1027.67/21.41ms，占轮跨度约82.5%；proposal169.05ms、sampling30.49ms、cache17.16ms。均为当前带观测开销的跨度，不能直接当作纯kernel占比或正式吞吐。
- proposal内候选选择25.35ms、MTP forward48.22ms、MTP head63.66ms；候选选择包含归一化，不能再相加。仅删除全部候选选择的理想上限约2%，不支持以top-k单项承诺≥5%整体收益；实际完整回归的成本仍需无探针证据。
- sampling CPU墙钟417.86ms与stream30.49ms差别很大，CPU等待会承接此前Target未完成的工作；不能按417.86ms直接给验收计算划出33%可优化空间。
- 数据只覆盖两个训练题的早段，提交token/轮也不等于完整请求MAL或正确率。五个前缀的树侧观测正在运行；完整数值门槛仍待独立Target侧。

## 相对上游的修改

- `remtp/runtime_profile.py`：移除profile内的自动top-k验证/额外计时循环。仅观测一个预热后的两个注册请求，跳过prefill，每请求最多16轮、每服务最多32轮；CPU墙钟、CUDA event跨度、父子span和轮末显式同步分别记录。
- `scripts/run_adaptive_round2.py`：profile参数真实传入env，记录有效env；显式传递Cactus/SpecCascade参数；预热成为可记录的正常继承方法。原默认非profile调用保持关闭观测。
- `remtp/target_diagnostics.py`：新增精确prompt/output IDs选择表和可关闭的shape投影探针；只在选中的帧复制实际输入/位置，未选中轮不触发这些GPU→CPU读取。旧调用不提供选择表/shape开关时仍按原范围保存。
- 新 `scripts/run_raven_review_unit.py`、`launch_raven_review_unit.py`、`status_raven_review_unit.sh`：冻结、组件hash、调用ledger、15秒心跳、设备记录、明确停止原因和中文报告；9次服务启动以内、24小时上限、0重试。
- 新 `tests/test_raven_review_unit.py`，覆盖预算消耗/重复拒绝、状态分类、真实启动env、观测不额外投影、CPU RNG/调用保持、分层计时、warmup/prefill/后续数值请求排除。
- 本记录、执行计划、报告入口、README与强制研究日志同步。

模型、公共Target/MTP数值算法、树验收、默认cap/懒残差、prompt、评分和旧冻结结果未改。质量/速度优化候选尚未选择。

## 第一单元实际范围

| 类型 | 上限 | 内容 |
| --- | ---: | --- |
| 主生成HTTP | 74 | 8条64-token四方法profile + 2条Target-only/1532完整2048生成 + 64条8训练题×2种子×4方法校准 |
| 数值诊断HTTP | 10 | 5保存实际晚段前缀×fresh树/独立顺序Target；树max_tokens2强制已知末token以获取实际root，顺序Target只取末行/生成1token |
| 预热HTTP | 9 | 每服务1条8-token；单列，不计质量/性能 |
| 服务启动 | 9 | profile4 + Target1 + calibration4 |

数值诊断按10个顶层请求计数，每次内部可能包含prefill、树前向及MTP；不是10个CUDA kernel或10次单独verifier。没有echo/prompt_logprobs，不构造长prefix×全词表logits；shape探针关闭，不额外做LMHead投影。

原动态/顺序Target raw门槛在本次更明确：fresh raw任一差异停止定位，旧0.15仅作描述；fresh树与保存P的TV允许1e−6跨CPU/GPU softmax归约误差，超过则停止。五个前缀的通过仅覆盖这些现场，不能继承为完整KV证明。

profile顺序Native→Cactus→SpecCascade→B，校准反序；各方法服务内部交替种子顺序。限9次启动，因此该8题开发诊断不声称实现正式全块方法顺序平衡；正式阶段仍按原计划独立安排。

## 预检证据与限制

- 31项必要CPU测试通过（2.94秒）；新增脚本编译与状态脚本bash语法通过。
- 首个CPU参照检查拒绝：修复快照少了后续已关闭的cap接口，dynamic adapter/verifier源码不同。旧预检目录保留，0模型调用。当前运行域改与最新cap冻结B比较，修复单元仅作16个完整B输出签名参照；并未改旧gate/hash。
- 第二个真实CPU check-only通过，数据/模板/公共数值选项、5帧输入、组件源码和预算核对通过。未加载模型，未请求生成。
- 使用实际`.venv`检查10个vLLM/BNB hook入口均存在；导入会初始化CUDA上下文，检查进程已退出，没有模型前向/生成。直接用系统Python拼接venv路径导入vLLM会混入系统SciPy，导入检查已改用项目解释器，未安装/升级包。
- CUDA events是stream时间跨度，会包含host发射间隙；CPU墙钟包含等待；不是纯kernel时间和，不将嵌套项相加。观测会改变时序，不能当作无探针吞吐。计时设计参照[PyTorch 2.10 CUDA语义](https://docs.pytorch.org/docs/2.10/notes/cuda.html)。
- GPU实际加载和四方法profile覆盖已通过；晚段raw一致、Target-only与完整校准尚未完成。本单元不自动采用候选或执行留出/正式阶段。
