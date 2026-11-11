# 原版改动：Target相对支持约束与有限重跑

> 完结更新（2026-10-09核验）：修复单元10月8日21:32:26 UTC+8完整完成，86主HTTP/11fresh/48 CUDA/7启动与预热，exit0、0重试。legacy/关闭兼容、完整H/残差/RNG、候选实际prefix raw和有/无探针IDs复现均通过；16条校准候选10原判对/8正常结束判对/8截断，较B12/10/6退化，联合门槛拒绝，默认none保持。[中文结果与分析](../../RESULT-RAVEN-TARGET-ODDS-2026-10-09.md)、[完结核验改动](results-2026-10-09.md)。以下待启动/运行为历史，没有自动采用或正式测试。

日期：2026-10-08（UTC+8）。状态：候选实现并独立冻结，10:17:42后台轻量等待原B根因诊断，未验证新GPU收益。

## 数学改动

- remtp/dynamic_mtp_tree.py：新增默认none的relaxation_risk_policy；仅adaptive residual_hit_anchor支持target_odds。p/p_max小于0.001时H=P且预算先归零；小于0.1时不允许超过基础松弛；其余保持原分配。接受与拒绝残差共用完整H；EOS/prefix保护与新增风险保护区分。
- remtp/dynamic_tree_vllm.py：读取可选REMTP_DYNAMIC_TREE_RELAX_RISK_POLICY，启动日志报告实际值。
- scripts/run_adaptive_round2.py：将固定策略传入env，旧默认none；模型、Target/MTP算子、原评分/停止规则、2048预算不改。
- scripts/analyze_raven_root_causes.py：登记候选反例枚举，标注残差是否实际消费；同树uniform反事实只按原EOS/prefix保护计算。

风险与限制：此约束修复一个可证明的局部放大缺口，不提供整体分布/任务正确率保证。不改写全局cap2失败，也不把现场额外接受当整题因果。

## 控制器与观察

- 新run_raven_risk_repair.py、launch_raven_risk_repair.py、wait_raven_risk_repair.py、status_raven_risk_repair.sh。一个固定候选，86主HTTP/12数值HTTP/48 CUDA/7启动与预热，24小时含依赖等待，0重试/无采用/无留出test。
- run_raven_root_cause_unit.py：复用实际自然prefix检查方法；候选可记录与历史B输出不同，原根因默认仍要求复现；请求记录按实际探针标志区分性能样本。运行中的旧冻结没有改动。
- 等待进程仅标准库；根因完整完成、exit0且释放端口后才导入Torch/加载GPU。旧运行资产/source hash/失败/额度保留，新单元单独冻结登记。
- 16 CUDA帧预定为12个实际首/末帧+4个按采集顺序最早剩余帧，不按候选正确答案挑选。候选概率、接受、残差及RNG完整链独立检查；默认关闭必须与旧实机保存结果位相等。
- 六条自然观察不计速度；五方法80条校准全部关闭探针并用实际ID复现检查，质量与速度门槛单列，报告中文。

## 已实际验证

130项算法/根因相关CPU检查通过8.96秒。初轮EOS测试在0.01边界受FP32归一化影响，修正测试夹具为0.02后通过，运行阈值未改变。新脚本py_compile与状态脚本bash语法通过。

86项控制器/观测/算法边界CPU检查通过6.06秒。本机真实check-only通过，独立preflight ledger全0；包含完整86请求transport模拟、服务/预热预算、同协议ID回归和中文分析生成。模拟不算GPU/HTTP实机实验。CUDA子进程失败或超时也在finally重载真实ledger，避免报告隐去已花费额度。

最终87项CPU检查通过6.21秒，包含CUDA子进程超时后父报告保持已花费额度。脚本编译、bash语法、git diff --check通过；5份文档62条本地链接存在。

冻结check-only、实机启动待追加；当前不声称新GPU/质量/速度门槛通过。详细顺序见[计划](../../../plan/adaptive-raven-target-odds-repair-2026-10-08.md)，所有改动同步[研发日志](../../../docs/research_and_development_log.md)。

> 实际启动更新：10:17:42 UTC+8，remtp-raven-target-odds-repair-20261008.service，冻结check-only成功、源码hash匹配；active/running、Restart=no、NRestarts=0、RuntimeMax=1d。等待进程约8MiB，实际没有映射libtorch，所有模型/HTTP/CUDA账本仍0。上句待启动是历史预检状态。Root原冻结保持，runtime变化只登记dynamic_mtp_tree、dynamic_tree_vllm和run_adaptive_round2三个允许文件；人工diff核对桥接仅env/日志，数学仅本候选。关闭对话后等待成功依赖并自动执行，有失败仍停止且保留记录。[实时中文报告](../../RESULT-RAVEN-TARGET-ODDS-REPAIR-LIVE.md)。
