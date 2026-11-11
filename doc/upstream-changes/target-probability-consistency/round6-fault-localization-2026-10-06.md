# 相对原版改动：离线故障定位与计划细化

日期：2026-10-06。状态：离线诊断完成，新GPU/算法优化未执行。

- 新scripts/analyze_adaptive_round5_faults.py：只读日志，按实际提交长度重建精确token，
  校验API与prefix，定位分叉角色与缺失概率，统计实际松弛接受及冗余残差调用。
- 新tests/test_adaptive_fault_forensics.py：5项CPU实际通过（0.04秒），覆盖提交裁剪、
  错prefix/位置/文本拒绝与分叉角色。真实6条诊断重建全部通过。
- 新中文FAULT-LOCALIZATION报告；细化已有round6-path-quality-speed计划，
  将完整条件证据与决策保持的懒残差加速置于优先位置。总调用预算保持2684主路径上限。
- 证据位于ignored results/adaptive_round5_fault_localization_20261006/，输入hash可追溯。
  原始日志/gate/冻结代码未改变，Target/MTP模型、验收公式、RNG、服务和评分均未修改。

局限：首次分叉处完整bonus/纠正P/Q与CUDA RNG状态缺失；最终错误唯一因果未证明。
懒残差、相对TV上限、重复回退与成本选树均为下一阶段候选，不能当成已修复或加速。
