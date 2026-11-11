# Adaptive RAVEN 优化报告入口

> 采用结论更新（2026-10-14）：deep4_conservative（D4/N12）因大于原Adaptive的D3/N9树预算，不予采用。历史D4对D3的MAL/质量/速度比较仅作配置实验记录。当前路线改为固定D3/N9，验证策略与计算优化需在相同预算下证明收益；[新计划与每步预期效果](../plan/adaptive-raven-equal-budget-2026-10-14.md)。本次更新未启动新实验。

> 完整结果（2026-10-13）：原协议100+100五方法20:59:11 UTC+8正常完成，1000生成/500判题，原始记录与统计独立CPU核验通过。deep的MAL两任务最高、对Native点门槛通过，但尚未同时超过Cactus和Spec。对原方法GS正确率−2百分点/整题+7.47%，HE+4百分点/整题−15.05%。[完整实测与分析](RESULT-RAVEN-100X100-2026-10-13.md)，[CSV](../results/raven_full_100x100_20261010/summary.csv)。当前没有后台实验，以下状态为历史。

> 续跑更新（2026-10-12）：600条生成和230条判题检查点完整；12:57:46 UTC+8恢复后台，SpecCascade余70判题已完成，正在继续两种RAVEN。原冻结生成/数据/种子不变，OOM候选错误归类修复另有登记。[恢复情况与命令](RESUME-RAVEN-100X100-2026-10-12.md)，[实时中文报告](RESULT-RAVEN-100X100-LIVE.md)。完整实验仍未结束。

> 当前正式复现（2026-10-10）：用户已确认原GSM8K100+HumanEval100及相同种子；五方法1000请求后台11:22:57启动，Native health/预热通过，正式生成中。官方100参考解和错误控制判题通过。[冻结配置与启动验收](RESULT-RAVEN-100X100-2026-10-10.md)，[自动更新中文报告](RESULT-RAVEN-100X100-LIVE.md)，进度`bash scripts/status_raven_full_benchmark.sh`。本轮先完成数据，不自动优化；以下“无新后台”等描述为历史。

> 最新完结（2026-10-10，UTC+8）：112/112优势搜索00:32:59正常完成、exit0/0重试，独立CPU核验通过。deep4_balanced原判对16/16、吞吐2.6868、平均437.31秒、MAL3.8883；conservative14/16、2.6440、384.06秒。两个D4在本批点结果上超过原B/Native，仍未同时超过三个基线。[中文完结结果、区间与下一优化重点](RESULT-RAVEN-DOMINANCE-2026-10-10.md)。当前没有新后台，以下“运行中”状态为历史快照。

> 原方法专表（2026-10-09）：[初始 Adaptive RAVEN 与 Native/Cactus/SpecCascade 同批100题比较](RESULT-RAVEN-ORIGINAL-VS-BASELINES-2026-10-09.md)，包含正确率、吞吐、整题耗时、平均接收长度；400条原始记录CPU复核通过，属于旧数值版本的历史结果。

> 历史最佳结果整理（2026-10-09）：[同批三基线表、D4 提速峰值与接收长度 MAL](RESULT-RAVEN-HISTORICAL-BEST-2026-10-09.md)。CPU 重算986条/29组汇总通过；48请求旧候选对原B吞吐+13.91%，尚无完整三基线联合胜出。15:41 UTC+8后台完成34/112、deep4_balanced运行中，以下旧“首题”状态为当时快照；本整理没有新增GPU调用。

> 当前后台（2026-10-09 11:24 UTC+8核验）：按用户新目标运行112完整生成搜索，B＋3个Adaptive候选＋3基准，原B健康/预热通过，首题1532/42生成中。22项CPU/真实preflight/source hash通过，active/running、0重试；[实时中文报告](RESULT-RAVEN-DOMINANCE-LIVE.md)，进度bash scripts/status_raven_dominance.sh。截断仅诊断，尚无新质量/速度结果；下文“预检中”为启动前状态。

> 当前目标更新（2026-10-09）：取消截断、正常结束判对和单题非退化硬门槛；原评分正确率、token吞吐与整题耗时决定是否超过三个基准。[新目标与路线](OBJECTIVE-RAVEN-DOMINANCE-2026-10-09.md)，[112完整请求/三个Adaptive候选的新计划](../plan/adaptive-raven-dominance-2026-10-09.md)。当前CPU预检中，尚无新GPU结果；以下旧门槛按当时要求保留。

> 最新研究（2026-10-09）：回到原B基准，上一轮失败的真实分叉与质量门槛覆盖已重新核验。[中文研究、EOS边缘反例与截断解释](RESEARCH-RAVEN-BASELINE-2026-10-09.md)；[新优化计划及每步预期效果](../plan/adaptive-raven-baseline-optimization-2026-10-09.md)。6轨迹/83帧CPU研究通过，26组共同前缀完整P/Q/RNG一致；当前没有新后台实验，生产优化尚待执行。

> 最新完结（2026-10-09核验）：原根因10月8日10:32、修复21:32正常完成，exit0、0重试；86/86主HTTP、48/48 CUDA、80条无探针五方法校准完整。原B12判对/10正常结束判对/6截断，候选10/8/8，吞吐-1.16%、整题+16.87%，候选拒绝，最终目标仍未达成。[本轮中文结果、逐题退化与原因分析](RESULT-RAVEN-TARGET-ODDS-2026-10-09.md)。两单元各12现场raw/11fresh及CPU原始数据核验通过；当前没有新后台实验，以下等待/运行状态为历史。

> 当前定位与修复（2026-10-08）：[中文证据](ROOT-CAUSE-ADAPTIVE-RAVEN-2026-10-08.md)、[修复与重跑计划](../plan/adaptive-raven-target-odds-repair-2026-10-08.md)。原B自然诊断已启动并完成4/6；候选已实现，130项算法/87项最终控制器CPU检查和冻结check-only通过，10:17:42后台轻量等待完整门槛后自动重跑。[实时中文报告](RESULT-RAVEN-TARGET-ODDS-REPAIR-LIVE.md)，查询bash scripts/status_raven_risk_repair.sh。最终三基准质量/速度目标仍未达成。以下“无新后台”为上一单元结束时的历史状态。

> 最新完结（2026-10-08）：复审证据单元06:25:15 UTC+8正常结束，退出0、0重试；64条四方法校准全部完成。B原判对12/16、正常结束判对10、截断6，吞吐2.6130；三个质量门槛均未过，Cactus吞吐2.8936。Target-only两条都2048反复截断；五个晚段raw及16历史B输出签名复现通过。[中文结果与原因分析](RESULT-RAVEN-REVIEW-EVIDENCE-2026-10-08.md)。以下运行进展为历史记录；当前没有新后台实验或采用的新候选。

> 当前执行（2026-10-07 21:38 UTC+8）：四方法计时完成；分段prefill观测修复后，五个实际晚段前缀的fresh树/顺序raw全部逐位一致、最大差异0。36项CPU/冻结预检通过，后台正执行Target-only完整生成，随后64条无探针校准。[实时中文报告](RESULT-RAVEN-REVIEW-LIVE.md)；[故障定位/追加预算](upstream-changes/project-review/probe-prefill-fix-2026-10-07.md)。下面启动状态为历史记录，质量/截断/三基准速度目标仍未达成。

> 执行复审计划（2026-10-07）：第一证据单元21:08:40 UTC+8已冻结启动，31项CPU/真实check-only通过；四方法8条真实请求与各32轮计时覆盖通过，晚段数值检查和后续校准后台运行中。[实时中文报告](RESULT-RAVEN-REVIEW-LIVE.md)；[修改与验证](upstream-changes/project-review/evidence-unit-2026-10-07.md)；进度入口 `bash scripts/status_raven_review_unit.sh`。当前尚无新的优化收益/正式基准结论。

> 全项目复审（2026-10-07）：[目标、差距与优化判断](REVIEW-ADAPTIVE-RAVEN-PROJECT-2026-10-07.md)；[新的执行顺序和有限预算](../plan/adaptive-raven-project-review-optimization-2026-10-07.md)。审查/CPU证据已完成；第一单元补当前成本与基准证据，再选择一个优化方向。

> 已停止的上一单元（2026-10-07，UTC+8）：概率上限单元17:39:27在seed42质量门槛停止；7诊断/32 CUDA完成，训练48/144。cap2截断5→3、正常结束判对19→21，但原判对22→21，吞吐约-5.42%。[该单元结果与分析](RESULT-RAVEN-PROBABILITY-CAP-2026-10-07.md)。

> 最新结果（2026-10-07）：MTP修复单元正常完成150请求与32次CUDA条件验收。B/懒残差各72请求均67判对、61正常结束判对、11截断，输出72/72相同；吞吐2.624640→2.623474token/s（约-0.04%），速度gate未通过。数值修复已通过本轮复测，截断与三基准速度目标仍未达成。

- [最新完成：修复后训练结果与中文分析](RESULT-MTP-REPAIRED-REGRESSION-2026-10-07.md)
- [修复记录：具体原因、验证与优化边界](RESULT-MTP-PROBABILITY-REPAIR.md)
- [已完成后台单元控制器报告](RESULT-MTP-PROBABILITY-REPAIR-LIVE.md)
- [已停止：Native诊断与H/P上限控制器报告](RESULT-RAVEN-PROBABILITY-CAP-LIVE.md)
- [已拒绝候选的历史计划：Native配对诊断与局部H/P上限](../plan/raven-probability-cap-2026-10-07.md)
- [概率修复历史计划](../plan/mtp-probability-repair-2026-10-06.md)

> 历史说明（2026-10-04）：2026-10-01的短测与训练回归结果是第一轮历史记录，不能代表当前数值修复候选或正式基线对比。

- [历史：第六轮阶段结束与中文分析](RESULT-ADAPTIVE-RAVEN-ROUND6-ANALYSIS.md)
- [第六轮原始控制器报告（概率故障门槛停止）](RESULT-ADAPTIVE-RAVEN-ROUND6.md)
- [原始故障现场：MTP原始Q非有限值（后续已修复）](FAULT-MTP-Q-NONFINITE-2026-10-06.md)
- [最近完成：第五轮中文结果与分析](RESULT-ADAPTIVE-RAVEN-ROUND5-ANALYSIS.md)
- [后续故障定位与详细优化思路](FAULT-LOCALIZATION-ADAPTIVE-RAVEN-2026-10-06.md)
- [原完整优化计划：路径定位、质量与速度](../plan/adaptive-raven-round6-path-quality-speed-2026-10-06.md)
- [上一轮失败原因与结果分析](ANALYSIS-ROUND4-QUALITY-REJECTION-2026-10-06.md)
- [第五轮原始控制器报告](RESULT-ADAPTIVE-RAVEN-ROUND5.md)
- [第五轮执行计划](../plan/adaptive-raven-round5-eos-2026-10-06.md)
- [第四轮完整结果（已因质量门槛拒绝停止）](RESULT-ADAPTIVE-RAVEN-ROUND4.md)
- [本轮原因分析与已验证修复](ANALYSIS-ROUND4-NUMERICAL-REPAIR-2026-10-04.md)
- [新一轮详细执行计划](../plan/adaptive-raven-round4-execution-2026-10-04.md)
- [2026-10-01原历史报告（完整保留）](../results/adaptive_optimization_20261001/anchor_d4_v4/report.md)

第五轮已完成104/104请求：old_eos01通过训练质量门槛，Anchor未通过；尚未达成三个基线的正式质量与速度目标。
第六轮已完成4/4自然诊断与64/64 CUDA条件调用，随后因原始Q非有限值门槛停止；144条训练与正式比较没有开始。
共同前缀及两题首次分叉的完整Target P均一致；懒残差行为等价通过，真实速度和截断收益尚未验证。
核对本轮状态：`bash /home/apex/ReMTP-residual/scripts/status_adaptive_round6.sh`。
核对新修复单元：`bash /home/apex/ReMTP-residual/scripts/status_mtp_repaired_regression.sh`。
第五轮历史状态：`bash /home/apex/ReMTP-residual/scripts/status_adaptive_round5.sh`。

本轮进度：`bash /home/apex/ReMTP-residual/scripts/status_raven_review_unit.sh`。
已停止的概率上限单元：`bash /home/apex/ReMTP-residual/scripts/status_raven_probability_cap.sh`。
