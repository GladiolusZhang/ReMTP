# MTP首个非有限算子定位与修复记录

状态：首坏算子定位及16次条件GPU对照完成，代码修复已验证CPU；新的模型加载/完整请求与训练正准备启动。

- 新`remtp/mtp_operator_audit.py`：临时pre/post hooks只覆盖真实请求首轮首个MTP树计算；输入在原地norm改写前复制，记录实际attention metadata、forward slots和可见KV，512位置上限。
- `decision_trace.py`在显式`REMTP_MTP_OPERATOR_AUDIT=1`时启用；默认关闭，不采样、不修改隐藏/缓存，hooks在finally移除。
- `run_adaptive_round2.py`显式传递该诊断开关，现有调用默认关闭。
- 新有界`run_mtp_probability_repair.py`与专门计划：B/Anchor同题同seed各32token，独立服务，0自动重试；不计质量/性能，不改旧记录。
- 接下来依据保存的首坏算子决定修复，修复后再登记新数值重验；旧门槛不能替代新代码验证。

## 实际故障与修复

1. vLLM `TreeAttentionMetadataBuilder.build_for_drafting`切出1×1 bias后`.contiguous()`仍stride(0)=2；kernel把stride当有效列数，读取逻辑bias外的数据。
2. 新`fixed6_vllm._compact_draft_tree_bias`在真正draft metadata边界检查row stride等于列宽；仅不匹配时`clone(memory_format=torch.contiguous_format)`。逻辑树掩码不变。
3. `target_attention_fp32.py`的QQ bias load同时mask有效query行/头，处理query tile的padding；不改变合法行的算术。
4. `dynamic_tree_vllm.py`primary/branch/fallback在随机数前检查有限/非负/正质量；非法Q报错，禁止生成固定0后让验收假定均匀Q。合法Q RNG保持原语与顺序。
5. 新`replay_mtp_attention_fault.py`用真实Q/KV做owned Inf padding反事实；16调用分修复前后，compact输出均与无bias参考逐位一致。仅query mask不能修stride2列宽，二者联合才保护范围。
6. 新冻结/后台控制器`run_mtp_repaired_regression.py`、`launch_mtp_repaired_regression.py`、`status_mtp_repaired_regression.sh`，门槛失败终止、不自动重试。正式质量仍2048，诊断探针32，训练不启用trace。

验证：120项CPU（11.60秒）、22项CPU（2.83秒）；py_compile/bash -n/diff检查通过。两条GPU算子探针以及两组8次attention对照真实完成。完整模型修复验证与质量/性能统计仍等待新单元。
限制：guard引入同步，速度影响待测；旧故障的相邻内存值未捕获；该局部Q故障不能直接解释Anchor两道错答和截断。懒残差不减少截断，下一质量候选仍需针对过强松弛。

实际执行更新：15:51:41 UTC前B/Anchor各1条修复后真实smoke均通过，16帧×2=32次新CUDA验收通过；后台开始4条完整2048诊断，训练尚未开始。

## 2026-10-07完成复核（UTC+8）

- 11:03:28后台正常完成150请求+32CUDA调用，144训练=两组各72；质量与输出等价通过，速度未通过，未自动采用。
- 两组均67/72判对、61/72正常结束判对、11/72截断；72输出完全相同。吞吐2.624640→2.623474token/s，-0.0444%，按题95%区间跨1。
- 本次仅新增CPU离线分析器和中文文档/历史notice，原评分/汇总/bootstrap、48捕获帧、四条完整token重建、有效lazy开关复核；无新GPU调用、运行源码算法未更改、冻结不修改。
- 新结果文档doc/RESULT-MTP-REPAIRED-REGRESSION-2026-10-07.md、另存postrun_audit_20261007.json；11条截断耗时占44.8%，9条命中重复启发式；Anchor高H/P局部接受仍存在，不以此证明最终错答因果。
- 现阶段保留数值修复，懒残差默认关闭；下一方向为结束/重复和局部松弛约束、成本定位，正式三基准目标未达成。
