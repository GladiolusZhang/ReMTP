# 相对原版改动：第五轮质量定位

> 完成补记（2026-10-06）：v2实际完成104请求，重放通过、old_eos01通过训练质量门槛，Anchor未通过；质量保守候选吞吐未提高。新增中文结果分析、更新计划/报告入口与历史状态，原始冻结代码/报告/gate保持。没有新增算法修改或启动GPU任务。

日期：2026-10-06。状态：配置与控制器实现；GPU结果待运行。

- `scripts/run_adaptive_round2.py`：允许候选覆盖既有EOS threshold与repeat guard。
  未提供覆盖时仍为0.5、repeat跟随protect；默认行为不变。
- `scripts/run_adaptive_round5.py`：冻结两个单因素候选、校验旧数值/训练证据、失败题重放、
  8条诊断与96条回归，区分门槛拒绝/运行故障。无强制EOS、无修改评测或论文数学公式。
- `scripts/launch_adaptive_round5.py`、`scripts/status_adaptive_round5.sh`：独立用户systemd、
  新目录和代码快照、104请求上限、无重试、72小时上限；第四轮状态与结果入口保持。
- `tests/test_round5_controller.py`、`tests/test_generation_guard.py`：拒绝不完整/错种子/
  错预算/观测计时/配置/数据/评分参照，验证中等EOS概率保护等同严格验收与RNG行为。
- 新中文失败分析与执行计划；旧第四轮文档加历史状态，报告入口增加第五轮。

已实际验证：37项相关CPU测试通过。新GPU质量和速度结论尚未产生。

首个启动暴露Round4.start_server的零参super与兄弟类不兼容，0请求退出。
已改Round5为Round4子类、显式调用Round2初始化与报告，并增加启动包装/MRO路径测试。
失败冻结代码与记录保留，使用新目录v2启动；不把首次CPU测试误写成启动成功。
修复后38项CPU测试通过（3.64秒），静态检查和只读证据预检查通过，v2独立systemd已启动。
保护阈值0.01是消融假设；训练结果不能代替独立留出、三个基线正式对照或真实性能区间。
所有旧失败gate与原始记录保留，不改成通过，不覆盖原冻结代码。
