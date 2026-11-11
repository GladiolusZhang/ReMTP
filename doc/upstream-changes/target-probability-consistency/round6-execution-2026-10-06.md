# 第六轮执行改动记录

日期：2026-10-06。状态：实现及CPU验收完成，后台GPU单元已启动；完整GPU门槛尚未形成结论。

> 收尾更新：v2已完成4条自然复现和64次CUDA条件重放，原始Q非有限门槛停止，未展开训练/正式测试。
> 本次新增离线`analyze_adaptive_round6_results.py`及中文结果分析，完整token复核与两题首次分叉全P/RNG对比通过；没有更改运行算法或新增GPU请求。
> 首次分叉完整Target P相等；1231首个MTP树计算捕获的B/Anchor输入与prefix KV相等，但输出有限性不同，具体内部算子仍待定位。
> [完整结果分析](../../RESULT-ADAPTIVE-RAVEN-ROUND6-ANALYSIS.md)。下方启动状态为历史记录。

> 执行更新：初版真实服务通过health/预热并保存首题8轮张量，发现原始Q全NaN后主动停止。
> 观测位置校验改为prompt+history-2、NaN mask单独比较；新增非有限Q/P gate和有界MTP隐藏/prefix KV捕获。
> 109项CPU测试通过（12.05秒）。新版最多15自然诊断，先前1条中断单列；发现未解决Q故障时阻止后续训练扩展。
> 详见[故障证据与后续行动](../../FAULT-MTP-Q-NONFINITE-2026-10-06.md)。原始采样算法未用均匀替换强行“修好”。

新版v2实际模型加载、health=200、预热及首题生成已确认；前8轮观测边界全部通过。
首轮有限输入/有限prefix KV经第一次MTP树计算输出4096个NaN，已记录中间状态传播证据；具体算子修复尚未完成。

## 对代码的改动

- `remtp/dynamic_mtp_tree.py`：新增默认关闭的`lazy_residual`，仅已接受节点不构造残差；拒绝路径仍使用原函数。新增可选`decision_trace`，复制实际P/Q/H、均匀验收及纠正/bonus输入概率与原generator状态，不抽取额外随机数。
- `remtp/dynamic_tree_vllm.py`：传递`REMTP_DYNAMIC_TREE_LAZY_RESIDUAL`和仅诊断时活跃的trace；worker报告懒残差有效设置。
- `remtp/decision_trace.py`、`generation_guard.py`：诊断请求前8轮保存完整验收与proposal根状态；忽略短预热，不复制整片KV。proposal发生在CPU bookkeeping之前，显式保存`history_before_current_sampling`，由下一验收的真实提交前缀、根位置、Q/ID及RNG边界核对，避免误将早一步历史当完整proposal前缀。
- `scripts/run_adaptive_round2.py`：默认关闭上述开关；D与N仍独立设置，松弛权重显式按D写出，全1与历史最后一层延用1相同；原默认数值环境保持。
- 新`run_adaptive_round6.py`、`launch_adaptive_round6.py`、`status_adaptive_round6.sh`：明确Round4→Round2继承启动链，冻结源码，先验收再扩大，逐请求fsync，72小时上限，Restart=no，拒绝占用端口，失败分类留档。
- 新`analyze_adaptive_round6_trace.py`、`tests/test_round6_execution.py`：条件概率诊断、真实启动委派/报告schema/有限捕获测试，跨验收模式与EOS/保护/零分布的等价测试。

## 执行范围与验收

先4条失败复现，32真实帧×2=64次CUDA条件调用，通过后补齐16条诊断，再做B/懒残差各72条新训练请求。
代码变更后**不继承旧数值gate**；历史记录只作只读输出参照。新门槛仅绑定新源码的自然复现和条件验收范围。
诊断耗时不计性能，训练时间只作工程效果；未满足独立留出、三基准与统计区间时不能称正式成功。

107项相关CPU测试通过（11.80秒）；加严有效开关日志验收后，新Round6的21项测试再次通过（6.82秒）；py_compile及git diff --check通过。
真实冻结代码check-only已通过，后台单元于2026-10-06 13:37 UTC启动，结果根为`results/adaptive_round6_20261006_trace_lazy`。
GPU实际门槛由本轮中文报告追加。当前截断风险不由懒残差修复，需新条件证据支持质量候选。
