# 晚段观测的分段预填充故障与有限续跑

> 完结更新（2026-10-08）：末段GPU采集门槛与全部后续生成完成；06:25:15 UTC+8 exit0、0重试。Target-only两条实际约2125秒正常HTTP返回；64校准完成但未达到方法质量/速度目标。[结果分析](../../RESULT-RAVEN-REVIEW-EVIDENCE-2026-10-08.md)。以下加载/运行状态为原始历史记录，旧位置故障与ledger保留。

日期：2026-10-07（UTC+8）。状态：**修复后末段GPU观测/完整五前缀门槛通过；Target-only长生成运行中。**

> 21:38实机核验：第五个顺序帧保存最终chunk的175个实际输入，末位置2222/输入18正确；fresh树/顺序raw逐位相同、最大差异0、TV0。完整五前缀gate通过；与历史保存P最大TV约9.17e-8，在登记的CPU归约容差1e-6内。当前进入1532/seed42、2048输出的Target-only生成，主HTTP完成0/66、数值1/1；模型加载/健康/预热均通过，0重试。只覆盖这五个实际前缀，不能延伸为所有自然KV状态或整体质量证明。

> 实际启动：21:35:02 UTC+8，`remtp-raven-review-continuation-20261007.service`，目录`results/raven_review_continuation_20261007`。冻结check-only通过，active/running、NRestarts=0、Restart=no；RuntimeMax约23小时33分，绝对截止仍为原单元开始后24小时。21:35核验正在Target加载，新增主HTTP0/66、数值0/1，不能提前称末段门槛通过。[实时中文报告](../../RESULT-RAVEN-REVIEW-LIVE.md)。

## 实际停止原因

原单元`remtp-raven-review-20261007.service`在21:22:11以`NUMERIC_REJECTED`停止，exit3、0重试，服务清理完成。四方法8条profile完成；10次数值HTTP均返回，但只有9次满足采集位置门槛。旧failure、exit、ledger、冻结和不完整gate保留在`results/raven_review_unit_20261007`。

第五个顺序Target前缀长度2223，服务启用`max_num_batched_tokens=2048`的chunked prefill。原探针在第一段就消耗`max_frames=1`，保存位置2047/输入220，预期最终位置2222/输入18。真实请求随后处理175-token末段并HTTP200，控制器因保存帧位置错误停止。没有这次模型崩溃、OOM或新raw不一致的证据；前四个fresh树/顺序raw完全相同、TV=0，第五个正确末行仍待补测。

## 相对上游的最小修复

- `remtp/target_diagnostics.py`：精确前缀选择表中的顺序prefill案例，要求实际末位置等于完整prompt末位置才保存。中间chunk不消耗保存名额，不增加LMHead投影、采样或改变KV。旧未使用选择表的诊断范围保持。
- `scripts/run_raven_review_unit.py`：完整生成HTTP超时1800→5400秒，仅延长等待；2048输出/种子/评分保持。当前Target前向观测约1.03秒/轮，单步2048生成存在超过1800秒的具体风险；尚未出现这次HTTP超时。
- 加入评分/benchmark/汇总组件hash核对。公共数学、树验收、Serving和评分实现必须与原冻结一致。
- 新`scripts/run_raven_review_continuation.py`：只允许复用此次已确认的chunk观测故障；核对原进度、全部运行域、profile覆盖、实际帧、四个精确gate与artifact哈希。拒绝已有长生成/校准、源码变化或不同停止原因；不修改旧ledger/gate。
- 启动器支持独立续跑冻结和原绝对截止时间，状态/中文报告分别显示复用证据与新增调用。不会自动重启失败单元。

## 注册的补测范围

| 调用 | 原单元实际尝试/完成 | 续跑上限 | 两单元尝试上限 |
| --- | ---: | ---: | ---: |
| 主HTTP | 8/8 | 66：2 Target-only +64校准 | 74 |
| 数值HTTP | 10/9 | 1：仅late_4_target末段 | 11 |
| 固定预热 | 5/5 | 5 | 10 |
| 服务启动 | 5/5 | 5 | 10 |

增加1次明确注册的数值补测及1次Target重启/预热，计入真实总账；没有重复profile或前四个数值请求，没有自动重试。截止时间固定为原执行开始后24小时：2026-10-08 21:08:49 UTC+8。续跑也最多1次执行，失败保留现场停止。

## 验证要求与限制

36项必要CPU检查实际通过（3.22秒），包括真实2223-token几何下的2048/175两段观测：第一段不保存、第二段只保存一次、原投影调用仍两次、输出不变；续跑拒绝重复工作/数学变化、原绝对截止、5400秒HTTP等待与原生成参数均检查通过。py_compile、bash -n与git diff --check通过。

真实续跑check-only通过：CPU重新核验旧进度、四方法profile、四个逐位相等前缀、公共数学/状态/Serving/评分源码与artifact哈希；预检时新ledger全部0、未加载模型或生成。随后独立冻结启动的实机补测已通过，详见顶部21:38记录。

这是观测和控制器修复，不是新质量/速度候选。模型、2048输出、公共数值与旧负gate保持；完整质量、截断改善及三基准速度仍未达成。
