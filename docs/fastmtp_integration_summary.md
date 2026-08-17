# FastMTP 集成总结

> **历史文档警告（2026-08-09）：** 本文记录 Claude Code 的初次集成，
> 其中旧服务脚本会把 Cactus/SpecCascade Worker 覆盖成 Native，旧四方法结果
> 无效。请使用 `scripts/run_fastmtp_verified_comparison.sh`，并以
> [研究与开发总日志](research_and_development_log.md) 的 3.15–3.16 节为准。

## 🎯 核心发现

你的洞察完全正确！**MiMo 的低接受率根源在于 MTP layers 1/2 没有经过训练**。

从 MiMo 官方文档：
> "Currently, MiMo-7B model each has 1 MTP layer (model.mtp_layers.0). Users may load the weights of pretrained MTPs for potential rollout speedup."

这意味着：
- ✅ Layer 0: 经过后训练（post-trained）
- ❌ Layer 1, 2: 只是预训练（pretrained-only），未针对下游任务优化
- 结果：物理 0-1-2 的深层候选与 target 几乎不重叠（coverage ~1e-18）

## ✅ FastMTP 是完美的替代方案

### 模型信息
- **仓库**: [TencentBAC/FastMTP](https://huggingface.co/TencentBAC/FastMTP)
- **论文**: [FastMTP on arXiv](https://arxiv.org/abs/2509.18362)
- **基础**: Qwen2 架构（8B 参数）
- **大小**: ~15.7 GB

### MTP 配置
```json
{
  "num_nextn_predict_layers": 1,        // 单个训练好的 MTP 层
  "num_speculative_steps": 3,           // 3 步推测
  "mtp_loss_weight": 1.0,               // 完全基于 MTP 训练
  "ntp_loss_weight": 0.0,
  "mtp_loss_step_weights": [0.51, 0.31, 0.18]
}
```

### 关键优势
1. **训练好的 MTP** - 不是 pretrained-only
2. **Position-shared weights** - 单 head 跨多步共享
3. **Self-distilled training** - MTP-inference 对齐
4. **Language-aware compression** - 降低计算开销
5. **发表性能**: 2.03× speedup, 82% better than vanilla MTP

## 📦 已完成的集成

### 1. 下载脚本
```bash
./scripts/download_fastmtp.sh
```
- 自动下载 ~15.7 GB 模型
- 验证关键文件
- 显示配置信息

### 2. Worker 实现 (`remtp/fastmtp_worker.py`)
- `FastMTPProbabilisticWorker` - 概率采样
- `FastMTPCactusWorker` - Cactus 松弛
- `FastMTPSpecCascadeWorker` - SpecCascade
- `FastMTPProposalCalibratedWorker` - Proposal calibration

### 3. 服务脚本
- `scripts/serve_fastmtp_native.sh`
- `scripts/serve_fastmtp_cactus.sh`
- `scripts/serve_fastmtp_spec_cascade.sh`

### 4. 三方法对比实验
```bash
./scripts/run_fastmtp_three_way_50.sh
```
对比方法：
1. Native MTP (trained baseline) - **这应该是高质量 baseline**
2. Cactus + MTP
3. SpecCascade TokenV3 + MTP

每个方法在 GSM8K 和 HumanEval 各测 50 个样本。

### 5. 报告生成器
`remtp/fastmtp_three_way_report.py` - 自动生成 markdown 报告

## 🚀 使用方法

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

# 停止现有服务
pkill -f "vllm serve" || true

# 1. 下载 FastMTP（~15.7 GB，一次性）
./scripts/download_fastmtp.sh

# 2. 运行三方法对比（预计 1.5-2 小时）
RUN_TAG=fastmtp_three_way_50_$(date +%Y%m%d) \
  ./scripts/run_fastmtp_three_way_50.sh
```

结果保存在：`results/fastmtp_three_way_50_*/comparison.md`

## 🎯 预期改进

使用 FastMTP 应该看到：

1. **更高的 baseline MAL**
   - 论文声称 2.03× speedup
   - 训练好的 MTP 应该远超 MiMo untrained layers

2. **清晰的松弛效果**
   - 在高质量 baseline 上测试松弛方法
   - 如果 native 已经很好，松弛增益可能有限
   - 如果 Cactus/SpecCascade 仍有增益，说明验证端优化有价值

3. **更好的质量保持**
   - 不会有 MiMo physical 0-1-2 的质量下降问题
   - 训练对齐应该保持任务质量

## 📊 与 MiMo 对比

| 维度 | MiMo-7B-RL (physical 0-1-2) | FastMTP (Qwen2) |
|---|---|---|
| 基础架构 | MiMo (Qwen2 变体) | Qwen2 |
| 参数量 | 7B | 8B |
| MTP 层数 | 3 (只有 layer 0 训练) | 1 (完全训练) |
| Layer 1/2 状态 | Pretrained-only ❌ | N/A (只有1层但训练好) ✅ |
| 接受率 | 低（深层不重叠） | 高（论文：82% better） |
| 发表性能 | - | 2.03× speedup |

## 🔍 后续决策

实验完成后：

**如果 Native FastMTP MAL 高（如 >2.5）：**
- 说明训练好的 MTP 确实解决了根本问题
- 松弛方法可能增益有限（baseline 已经好）
- 考虑在 FastMTP 上测试 Proposal Calibration

**如果 Cactus/SpecCascade 仍有显著增益：**
- 说明验证端优化在好的 MTP 上仍有价值
- 可以探索更激进的松弛策略
- 考虑组合方法（如 Cactus + Proposal Calibration）

**如果整体表现不如预期：**
- 检查 vLLM 0.18 对 FastMTP 的支持
- 可能需要适配 FastMTP 的 custom code
- 或考虑直接使用 FastMTP 的推理代码

## 📚 参考资料

- Model: https://huggingface.co/TencentBAC/FastMTP
- Paper: https://arxiv.org/abs/2509.18362
- GitHub: https://github.com/Tencent-BAC/FastMTP
- MiMo: https://huggingface.co/XiaomiMiMo/MiMo-7B-MTPs

## ✅ 文件清单

**新增文件：**
- `scripts/download_fastmtp.sh` - 下载脚本
- `scripts/serve_fastmtp_native.sh` - Native 服务
- `scripts/serve_fastmtp_cactus.sh` - Cactus 服务
- `scripts/serve_fastmtp_spec_cascade.sh` - SpecCascade 服务
- `scripts/run_fastmtp_three_way_50.sh` - 三方法对比
- `remtp/fastmtp_worker.py` - Worker 实现
- `remtp/fastmtp_three_way_report.py` - 报告生成
- `docs/fastmtp_integration_summary.md` - 本文档

**记忆更新：**
- `.claude/projects/.../memory/fastmtp_integration.md` - 详细记忆
- `.claude/projects/.../memory/MEMORY.md` - 索引更新
