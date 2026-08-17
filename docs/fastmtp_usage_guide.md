# FastMTP 完整集成指南

> **已被部分取代（2026-08-09）：** 本文中的
> `run_fastmtp_four_way_50.sh`、旧 FastMTP Worker 和旧动态树服务存在已知
> 接线错误，不能作为正式比较。当前入口为
> `scripts/run_fastmtp_verified_comparison.sh`。修复详情见
> [研究与开发总日志](research_and_development_log.md)。

## 📋 目录

1. [FastMTP 是什么](#fastmtp-是什么)
2. [快速开始](#快速开始)
3. [我们的动态树方案](#我们的动态树方案)
4. [后续训练方案](#后续训练方案)
5. [预期结果](#预期结果)

---

## FastMTP 是什么？

### 基本信息
- **基础**: MiMo-7B-RL (不是 Qwen2)
- **创新**: 单个 **训练好的** MTP head，position-shared weights
- **性能**: 2.03× speedup, 82% better than vanilla MTP
- **训练**: Self-distillation on 389.4K samples

### 核心区别

**MiMo (我们之前用的):**
```
Layer 0: ✅ 后训练 (post-trained)
Layer 1: ❌ 只预训练 (pretrained-only)
Layer 2: ❌ 只预训练 (pretrained-only)
结果: 深层接受率 ~2-11% (几乎无用)
```

**FastMTP (现在用的):**
```
Single MTP layer: ✅ 完全训练
递归使用 3 次
结果: k=1: 81%, k=2: 56%, k=3: 36% (实用)
```

---

## 快速开始

### 1. 下载 FastMTP

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

# 下载模型 (~15.7 GB)
./scripts/download_fastmtp.sh
```

### 2. 运行四方法对比

```bash
# 停止现有服务
pkill -f "vllm serve" || true

# 运行已审计实验（50 样本 × 2 数据集）
RUN_TAG=fastmtp_frontier_rescue_n50_$(date +%Y%m%d) \
  ./scripts/run_fastmtp_mal_quality_50.sh
```

该入口默认跳过较慢的 Target-only 行，只运行 Native、Cactus、SpecCascade
和动态树，并设置 `PROGRESS_EVERY=1`。终端会实时显示每条 GSM8K 的答案判定、
每条 HumanEval 的生成进度与隔离测试结果。完整控制台日志同时保存在
`logs/$RUN_TAG/`，最终表格仍写入一个 `results/$RUN_TAG/comparison.md`。

### 3. 查看结果

```bash
# 查看报告
cat results/fastmtp_frontier_rescue_n50_*/comparison.md

# 查看详细数据
cat results/fastmtp_frontier_rescue_n50_*/comparison.json
```

---

## 我们的动态树方案

> **路径规则更新（2026-08-09）：** 正式 100 条实验发现最长优先会在少数请求
> 中提交低可信尾部并形成超长轨迹。当前保留旧 `longest` 模式，同时新增
> `balanced`：所有存活前缀按 `C(path)*L^beta` 联合选择。新的受控实验仍把
> 每父节点限制为两个候选，只将全树上限从 6 小幅提高到 8。
>
> **MAL/质量版本（2026-08-09）：** 正常 target-dominant 规则不变；仅当
> 一个已到达前沿没有任何正常存活孩子时，允许目标支持最高的候选尝试一次
> `delta=0.5` 的有限救援。每条最终提交路径最多一次救援，EOS 保护仍有效。

### 算法概述

**Phase 1: 构树 (只用 MTP)**
```python
# 1. 计算不确定性
H = -sum(q log q) / log(|V|)  # 归一化熵
U = H + mu * d / D             # 加入深度偏置

# 2. 动态阈值
tau = max(tau_min, max(q) * exp(-kappa * U))

# 3. 候选集
C = {x: q(x) >= tau}

# 4. 节点排序 (预算不足时)
E(path) = geometric_mean(MTP_probs) * exp(eta * depth / D)
```

**Phase 2: 验证 (Target-dominant relaxation)**
```python
# 1. Target coverage
A_v = sum(p_v(siblings))

# 2. Relative support
R_v(x) = p_v(x) / max(p_v)

# 3. 松弛可信度
S_v(x) = A_v^alpha * R_v(x)^(1-alpha)

# 4. 剪枝
if S_v(x) < tau_relax:
    prune node and all descendants
```

**Phase 3: 路径选择**
```python
# 1. 路径得分
log Score(path) = mean(log S_i) + beta * log(L)

# 2. balanced 模式让所有存活前缀共同竞争
endpoint = argmax(C(path) * length(path)**beta)

# 3. 输出
selected_path + one_target_correction_token
```

### 参数配置

```python
# 构树参数
MAX_DEPTH = 3           # 最大深度
MAX_NODES = 6           # 整树节点预算
TAU_MIN = 0.02          # 最小阈值
KAPPA = 1.0             # 不确定性权重
MU = 0.5                # 深度偏置
ETA = 0.25              # 路径深度奖励

# 验证参数
ALPHA = 0.5             # Coverage/support 平衡
TAU_RELAX = 0.7         # 松弛阈值

# 路径选择参数
BETA = 0.5              # 长度奖励
PATH_TEMPERATURE = 0.3  # 采样温度
PATH_SELECTION = "longest"  # 或 balanced

# 首次失败前沿的单次救援（FastMTP MAL/质量入口默认启用）
FRONTIER_RESCUE = True
RESCUE_DELTA = 0.5
RESCUE_MIN_RELATIVE = 0.1
RESCUE_MIN_TARGET_PROB = 0.001
```

### 为什么这个方案合理？

**✅ 理论基础:**
1. **信息流清晰** - Proposal 不 peek target
2. **松弛有界** - Coverage × Support 保证质量
3. **路径采样** - 保持分布多样性

**✅ 在 FastMTP 上的优势:**
1. **根层可分叉** - 81% 接受率，能产生多个高质量分支
2. **深层有用** - 56%/36% 接受率，不是接近零
3. **并行探索** - 树可以同时探索多个未来

**❌ 在 MiMo untrained 上失败的原因:**
1. **根层不分叉** - 平均 1.01 分支
2. **深层无用** - 11%/2% 接受率
3. **浪费计算** - 验证 5.5 个节点只换来 0.007 MAL

### 只重跑新的 balanced/wider 配置

以下入口复用 `results/fastmtp_live_n100` 中已经完成的 Native、Cactus 和
SpecCascade，不重复运行旧方法；只生成新的动态树行，最后仍输出一张完整
对照表：

```bash
SAMPLES=100 \
BASELINE_RUN_ROOT=results/fastmtp_live_n100 \
RUN_TAG=fastmtp_balanced_wide_n100 \
  ./scripts/run_fastmtp_balanced_wide_n100.sh
```

### D=3/D=4 完整候选扩宽实验

该版本不强制把节点预算填满。构树允许最多三个 guarded candidates，所有新增
候选仍需满足绝对 Q 概率与相对 top-1 比例约束，再参与全局 10 节点预算竞争。
默认运行 GSM8K 500 条和 HumanEval 全部 164 条：

```bash
RUN_TAG=fastmtp_depth34_relaxed_full \
  ./scripts/run_fastmtp_depth34_relaxed_tree.sh
```

脚本先运行共享的 Native/Cactus/SpecCascade 和 D=3，然后只运行 D=4，最终生成：

```text
results/fastmtp_depth34_relaxed_full/depth34_comparison.md
```

---

## 后续训练方案

### 目标：训练 6-layer MTP

**为什么 6 层？**
1. 文献常用配置 (MTP depth=6)
2. 与我们的树实验一致 (6个节点)
3. 可以与 FastMTP K=3 递归对比

**方案对比:**

| 维度 | FastMTP (单层递归) | 我们的方案 (6层独立) |
|---|---|---|
| 参数 | 210.8M | 1.26B (6×210M) |
| 训练 | <1天 | 3-5天 |
| 推理 | 递归调用 | 并行预测 |
| 专门化 | 权重共享 | 每层独立学习 |
| 最大深度 | K=7 (测试过) | 6 (或更多) |

### 训练流程

**Step 1: 数据收集 (Self-Distillation)**
```bash
# 用 MiMo-7B-RL 生成 389K 样本
./scripts/collect_mtp_training_data.sh
```

数据分布:
- 42% 通识知识
- 18% 数学推理
- 13% 代码
- 27% 中文

**Step 2: 训练 6-layer MTP**
```bash
# 训练 3 epochs, lr=5e-5
./scripts/train_6layer_mtp.sh
```

Loss weights (指数衰减 β=0.6):
- Layer 1: 0.24
- Layer 2: 0.19
- Layer 3: 0.16
- Layer 4: 0.14
- Layer 5: 0.13
- Layer 6: 0.11

**Step 3: 在新模型上测试**
```bash
# 四方法对比 + 6层模型
./scripts/run_6layer_mtp_comparison.sh
```

### 训练代码已准备

所有训练代码和脚本已经写好，保存在：
- `scripts/collect_mtp_training_data.py` - 数据收集
- `scripts/train_multi_layer_mtp.py` - 训练脚本
- `scripts/prepare_and_train_6layer_mtp.sh` - 一键流程

详见：`docs/fastmtp_analysis_and_training_plan.md`

---

## 预期结果

### 短期 (FastMTP 四方法对比)

**乐观预期:**

| Method | GSM8K Acc | HumanEval pass@1 | MAL | E2E tok/s |
|---|---:|---:|---:|---:|
| Native FastMTP | 92-95% | 42-48% | 2.0-2.5 | 70-80 |
| Cactus | 90-93% | 40-46% | 2.1-2.6 | 75-85 |
| SpecCascade | 91-94% | 41-47% | 2.0-2.5 | 72-82 |
| **Dynamic Tree** | 88-92% | 38-44% | **2.3-2.8** | 65-75 |

**判断标准:**

✅ **成功** 如果:
- Dynamic tree MAL > Native + 15%
- 质量损失 < 5pp
- E2E tok/s 下降 < 15%

⚠️ **部分成功** 如果:
- Dynamic tree MAL > Native + 5%
- 但质量或速度权衡不理想

❌ **失败** 如果:
- Dynamic tree MAL ≈ Native
- 或质量显著下降 (>10pp)

### 中期 (6-layer MTP)

**预期改进:**
- 比 FastMTP 单层递归更深 (6 vs 3 steps)
- 每层专门化学习不同模式
- 可以在深度和宽度之间自由权衡

**研究价值:**
1. 首次系统训练多层 MTP
2. 对比递归 vs 独立层架构
3. 探索 MTP 深度的上限

### 论文方向

**Contribution 点:**
1. **多层 MTP 训练方法** - Self-distillation + 分层训练
2. **树 vs 链系统对比** - 在高质量 MTP baseline 上
3. **验证端 vs 提议端优化** - Cactus (验证) vs Tree (提议)
4. **质量-速度 Pareto 前沿** - 完整的松弛策略对比

---

## 文件清单

### 新增文件 (FastMTP)
```
scripts/
  download_fastmtp.sh                    # 下载脚本
  serve_fastmtp_native.sh                # Native 服务
  serve_fastmtp_cactus.sh                # Cactus 服务
  serve_fastmtp_spec_cascade.sh          # SpecCascade 服务
  serve_fastmtp_dynamic_tree.sh          # 动态树服务 ⭐
  run_fastmtp_four_way_50.sh             # 四方法对比 ⭐
  validate_fastmtp_setup.sh              # 环境验证

remtp/
  fastmtp_worker.py                      # Worker 实现
  fastmtp_dynamic_tree_worker.py         # 动态树 Worker ⭐
  fastmtp_four_way_report.py             # 报告生成 ⭐

docs/
  fastmtp_integration_summary.md         # 集成总结
  fastmtp_analysis_and_training_plan.md  # 深度分析和训练方案 ⭐
  fastmtp_usage_guide.md                 # 本指南
```

### 训练相关文件 (准备好，待运行)
```
scripts/
  collect_mtp_training_data.py           # 数据收集
  train_multi_layer_mtp.py               # 训练脚本
  prepare_and_train_6layer_mtp.sh        # 一键流程
```

---

## 快速检查清单

运行实验前检查：

```bash
# 1. 环境验证
./scripts/validate_fastmtp_setup.sh

# 2. 确认端口空闲
pkill -f "vllm serve" || true

# 3. 确认 FastMTP 已下载
ls models/FastMTP/config.json

# 4. 确认数据集存在
ls data/gsm8k/test.jsonl data/humaneval/HumanEval.jsonl.gz

# 5. 确认 Docker 可用
docker info && docker image inspect python:3-slim
```

全部 ✓ 后运行：

```bash
RUN_TAG=fastmtp_four_way_50_$(date +%Y%m%d) \
  ./scripts/run_fastmtp_four_way_50.sh
```

---

## 总结

1. ✅ **FastMTP 确认基于 MiMo-7B-RL**，有训练好的 MTP
2. ✅ **四方法对比已准备**，包含我们的动态树方案
3. ✅ **训练代码已完成**，可以训练 6-layer MTP
4. ✅ **方案合理性分析**，预期在好的 MTP 上成功
5. ✅ **所有脚本可执行**，随时可以运行

**下一步：先跑 FastMTP 四方法对比，验证动态树在训练好的 MTP 上的效果！** 🚀
