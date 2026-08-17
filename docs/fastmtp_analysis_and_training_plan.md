# FastMTP 深度分析与后续训练方案

> **仅为历史方案，尚未实现（2026-08-09 核查）：** 旧交接摘要声称已有
> `collect_mtp_training_data.py`、`train_multi_layer_mtp.py` 和一键训练脚本，
> 但当前仓库中这些文件不存在。本文不得作为“训练代码已就绪”的依据。当前
> 有效 FastMTP 集成和状态见
> [研究与开发总日志](research_and_development_log.md)。

## 1. FastMTP 是基于 MiMo 的吗？

**是的！** FastMTP 明确基于 **MiMo-7B-RL** checkpoint。

论文原文：
> "FastMTP is implemented using the pre-trained MiMo-7B-RL checkpoint (Xiaomi et al., 2025), a dense 7B parameter model with 36 decoder layers, a single-layer MTP module, and a vocabulary size of 152K tokens (based on the Qwen2 tokenizer)."

### 关键信息
- **基础模型**: MiMo-7B-RL (不是 Qwen2)
- **训练内容**: 只训练 MiMo 自带的单个 MTP 模块（210.8M 参数，<3%）
- **主模型冻结**: Transformer layers、embeddings、output head 全部冻结
- **训练策略**: Self-distillation - 用模型自己生成的数据来训练 MTP

## 2. FastMTP 的核心创新

### 2.1 单 MTP Head + Position-Shared Weights

**不是训练多个 MTP 层，而是让单个 MTP head 递归使用：**

```
k=1: hidden_state_main + embedding(shifted_token) → predict next
k=2: hidden_state_from_k1 + embedding(prev_shifted) → predict next
k=3: hidden_state_from_k2 + embedding(prev_shifted) → predict next
```

**对比：**
- **Vanilla MTP/MiMo**: 3 个独立 MTP 模块（layer 0, 1, 2），每个有自己的权重
- **FastMTP**: 1 个 MTP 模块，递归调用 3 次，权重共享

### 2.2 训练数据和方法

**Self-Distillation 数据集：**
- **389.4K 样本**（英文+中文）
- **领域分布**:
  - 42% 通识知识
  - 18% 数学推理
  - 13% 代码
  - 27% 中文
- **生成配置**: temperature=0.6, top_k=20, top_p=0.95, max_len=4096

**训练设置：**
- 3 epochs
- AdamW: lr=5e-5 (cosine schedule), β=(0.9, 0.95)
- Batch size: 64
- **Loss weights**: α_k = β^(k-1) / Σβ^(j-1), where β=0.6
  - Step 1: 0.51
  - Step 2: 0.31
  - Step 3: 0.18
- **训练时间**: <1 天 on single H20

**训练后接受率提升：**
- k=1: 70% → 81%
- k=2: 11% → 56%
- k=3: 2% → 36%

## 3. 我们能否训练更多 MTP 层？

**绝对可以！这是一个很有价值的研究方向。**

### 3.1 两种方案对比

**方案 A: FastMTP 方式 (单 head 递归)**
- ✅ 参数效率高（只 210.8M）
- ✅ 内存友好（不需要多个 KV cache）
- ✅ 训练快（<1 天）
- ❌ 递归深度受限（论文测到 K=7，最优 K=3）

**方案 B: 多层 MTP (我们的方案)**
- ✅ 每层专门化（不同层学不同模式）
- ✅ 可以更深（6 层甚至更多）
- ✅ 不需要递归（并行预测）
- ❌ 参数更多（6×210M = 1.26B）
- ❌ 训练时间更长

### 3.2 我们的训练方案

基于 MiMo-7B-RL，训练 6 个 MTP 层：

**目标：**
- Layer 0: 已有（MiMo 预训练）
- **Layer 1-2: 后训练**（替换 MiMo-7B-MTPs 的 pretrained-only 层）
- **Layer 3-5: 新增训练**（扩展到 6 层）

**为什么 6 层？**
1. 文献中 MTP depth=6 是常见配置
2. 我们的 Tree attention 实验用的就是 6 个节点
3. 可以与 FastMTP K=3 递归对比（FastMTP 3 步 vs 我们 6 独立层）

## 4. 训练代码和数据准备

### 4.1 数据收集（模仿 FastMTP）

```python
# scripts/collect_mtp_training_data.py
"""
Collect self-distilled data for MTP training.
Following FastMTP methodology.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import requests

# 数据源配置
DOMAINS = {
    "general_knowledge": 0.42,  # 42%
    "math_reasoning": 0.18,      # 18%
    "code": 0.13,                # 13%
    "chinese": 0.27,             # 27%
}

TARGET_SAMPLES = 389400  # FastMTP 用了 389.4K

# 数据集映射
DATASET_SOURCES = {
    "general_knowledge": [
        "Open-Orca/OpenOrca",  # 通识 QA
        "teknium/GPT4-LLM-Cleaned",
    ],
    "math_reasoning": [
        "meta-math/MetaMathQA",
        "TIGER-Lab/MathInstruct",
    ],
    "code": [
        "bigcode/starcoderdata",
        "Vezora/Tested-188k-Python-Alpaca",
    ],
    "chinese": [
        "shibing624/alpaca-zh",
        "Chinese-Vicuna/guanaco_belle_merge_v1.0",
    ],
}

def generate_self_distilled_sample(
    prompt: str,
    model_url: str = "http://127.0.0.1:8000/v1/completions",
    temperature: float = 0.6,
    top_k: int = 20,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """用 MiMo-7B-RL 自己生成训练数据"""
    response = requests.post(
        model_url,
        json={
            "model": "XiaomiMiMo/MiMo-7B-RL",
            "prompt": prompt,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    )
    return response.json()["choices"][0]["text"]

def collect_training_data():
    """收集训练数据"""
    output_file = Path("data/mtp_training/self_distilled_389k.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    samples_per_domain = {
        domain: int(TARGET_SAMPLES * ratio)
        for domain, ratio in DOMAINS.items()
    }

    collected = []

    for domain, target_count in samples_per_domain.items():
        print(f"\nCollecting {domain}: {target_count} samples")

        # 从各数据源加载 prompts
        prompts = load_prompts_from_sources(DATASET_SOURCES[domain])
        random.shuffle(prompts)

        for prompt in tqdm(prompts[:target_count]):
            try:
                response = generate_self_distilled_sample(prompt)
                collected.append({
                    "domain": domain,
                    "prompt": prompt,
                    "response": response,
                })
            except Exception as e:
                print(f"Error: {e}")
                continue

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in collected:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nCollected {len(collected)} samples")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    collect_training_data()
```

### 4.2 训练脚本（扩展到 6 层）

```python
# scripts/train_multi_layer_mtp.py
"""
Train 6-layer MTP on MiMo-7B-RL.
Layers 0: Use existing (from MiMo)
Layers 1-2: Replace pretrained-only with trained versions
Layers 3-5: New layers
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

class MultiLayerMTPTrainer:
    def __init__(
        self,
        base_model_path: str = "models/MiMo-7B-Base",
        num_mtp_layers: int = 6,
        prediction_depth: int = 1,  # 每层预测1个token
        loss_decay_beta: float = 0.6,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # 冻结主模型
        for param in self.model.parameters():
            param.requires_grad = False

        # 只训练 MTP layers
        self.num_mtp_layers = num_mtp_layers
        self._init_mtp_layers()

        # Loss weights (exponential decay)
        self.loss_weights = self._compute_loss_weights(
            num_mtp_layers, loss_decay_beta
        )

    def _init_mtp_layers(self):
        """初始化 MTP 层"""
        # Layer 0: 使用 MiMo 自带的（已预训练）
        # Layers 1-2: 如果存在，加载 MiMo-7B-MTPs；否则随机初始化
        # Layers 3-5: 随机初始化

        for i in range(1, self.num_mtp_layers):
            if i < 3:
                # 尝试加载 MiMo-7B-MTPs
                try:
                    mtp_layer = load_mimo_mtp_layer(i)
                except:
                    mtp_layer = init_new_mtp_layer()
            else:
                # 新层：随机初始化
                mtp_layer = init_new_mtp_layer()

            # 设置为可训练
            for param in mtp_layer.parameters():
                param.requires_grad = True

    def _compute_loss_weights(self, K: int, beta: float) -> torch.Tensor:
        """计算 loss weights: α_k = β^(k-1) / Σβ^(j-1)"""
        weights = torch.tensor([beta ** (k - 1) for k in range(1, K + 1)])
        return weights / weights.sum()

    def train_step(self, batch):
        """训练一个 batch"""
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Forward pass: 主模型
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        total_loss = 0.0

        # 逐层训练 MTP
        for layer_idx in range(self.num_mtp_layers):
            # 每层预测下一个 token
            mtp_logits = self.model.mtp_layers[layer_idx](
                hidden_states, input_ids
            )

            # 计算该层的 loss
            shift_logits = mtp_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., layer_idx + 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

            # 加权累加
            total_loss += self.loss_weights[layer_idx] * loss

            # 更新 hidden states（用于下一层）
            hidden_states = outputs_from_mtp_layer

        return total_loss

    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int = 3,
        lr: float = 5e-5,
    ):
        """完整训练流程"""
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            betas=(0.9, 0.95),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_dataloader) * epochs,
        )

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()
                loss = self.train_step(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # 保存 checkpoint
            self.save_checkpoint(f"checkpoints/mtp_6layer_epoch{epoch+1}")

    def save_checkpoint(self, path: str):
        """只保存 MTP layers"""
        torch.save({
            f"mtp_layer_{i}": layer.state_dict()
            for i, layer in enumerate(self.model.mtp_layers)
        }, path)


if __name__ == "__main__":
    # 准备数据
    from torch.utils.data import Dataset

    class MTPDataset(Dataset):
        def __init__(self, jsonl_path: str, tokenizer):
            self.data = []
            with open(jsonl_path) as f:
                for line in f:
                    self.data.append(json.loads(line))
            self.tokenizer = tokenizer

        def __getitem__(self, idx):
            sample = self.data[idx]
            text = sample["prompt"] + sample["response"]
            tokens = self.tokenizer(
                text,
                max_length=4096,
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": tokens["input_ids"].squeeze(0),
                "labels": tokens["input_ids"].squeeze(0),
            }

        def __len__(self):
            return len(self.data)

    # 训练
    dataset = MTPDataset(
        "data/mtp_training/self_distilled_389k.jsonl",
        AutoTokenizer.from_pretrained("models/MiMo-7B-Base"),
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    trainer = MultiLayerMTPTrainer(
        base_model_path="models/MiMo-7B-Base",
        num_mtp_layers=6,
    )

    trainer.train(dataloader, epochs=3, lr=5e-5)
```

### 4.3 完整流程脚本

```bash
#!/usr/bin/env bash
# scripts/prepare_and_train_6layer_mtp.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== 6-Layer MTP Training Pipeline ==="
echo

# Step 1: 准备 MiMo-7B-RL 模型
echo "Step 1: Preparing MiMo-7B-RL base model..."
if [[ ! -d "models/MiMo-7B-Base" ]]; then
  echo "Downloading MiMo-7B-RL..."
  huggingface-cli download XiaomiMiMo/MiMo-7B-RL \
    --local-dir models/MiMo-7B-Base \
    --local-dir-use-symlinks False
fi

# Step 2: 启动 MiMo 服务用于 self-distillation
echo
echo "Step 2: Starting MiMo-7B-RL server for data generation..."
MODEL_PATH="models/MiMo-7B-Base" \
  PORT=8000 \
  ./scripts/serve_mimo_native.sh &
SERVER_PID=$!

sleep 60  # 等待服务启动

# Step 3: 收集 self-distilled 数据
echo
echo "Step 3: Collecting self-distilled training data (389K samples)..."
python scripts/collect_mtp_training_data.py \
  --output data/mtp_training/self_distilled_389k.jsonl \
  --target-samples 389400 \
  --model-url http://127.0.0.1:8000/v1/completions

# Step 4: 停止服务
kill $SERVER_PID

# Step 5: 训练 6-layer MTP
echo
echo "Step 4: Training 6-layer MTP..."
python scripts/train_multi_layer_mtp.py \
  --base-model models/MiMo-7B-Base \
  --data data/mtp_training/self_distilled_389k.jsonl \
  --num-layers 6 \
  --epochs 3 \
  --lr 5e-5 \
  --batch-size 8 \
  --output checkpoints/mtp_6layer

echo
echo "Training complete! Checkpoints saved to checkpoints/mtp_6layer"
```

## 5. 关于我们的树注意力松弛方案

### 5.1 当前方案回顾

从之前的记忆看，我们的动态树方案是：

**构树阶段（只用 MTP）：**
```python
H = -sum(q log q) / log(|V|)  # 归一化熵
U = H + mu * d / D             # 不确定性
tau = max(tau_min, max(q) * exp(-kappa * U))
C = {x: q(x) >= tau}          # 候选集
```

**验证阶段（Target-dominant relaxation）：**
```python
A_v = sum(target prob of retained siblings)  # coverage
R_v(x) = p_v(x) / max(p_v)                   # relative support
S_v(x) = A_v^alpha * R_v(x)^(1-alpha)        # 松弛可信度
```

节点 `S < tau_relax` 及其后代被剪枝。

**路径选择：**
```python
log Score(path) = mean(log S_i) + beta * log(L)
endpoint = softmax(log Score / T_path) sample
```

### 5.2 问题分析

**❌ 在 MiMo untrained layers 上失败的原因：**
1. MTP 深层质量太差（coverage ~10⁻¹⁸）
2. 树主要在深层扩展宽度，但这些候选几乎都被 target 拒绝
3. 根层平均只有 1.01 分支（几乎不分叉）
4. 验证了 5.5 个节点，但只增加了 0.007 MAL

**✅ 方案本身是否合理？我认为是合理的，原因：**

1. **信息流边界清晰**
   - Proposal 只用 MTP（不 peek target）
   - Target 只用于验证和剪枝
   - 这是正确的 speculative decoding 范式

2. **松弛策略合理**
   - `A_v^alpha * R_v^(1-alpha)` 平衡了 coverage 和 support
   - 允许目标 top-1 但非 argmax 的候选通过
   - 这是 Cactus 思想的树版本

3. **路径选择有理论基础**
   - Geometric mean + length bonus
   - Softmax 采样（不是 greedy）
   - 保持一定随机性

**❓ 为什么在 FastMTP 上可能成功？**

FastMTP 的训练好的 MTP 应该：
1. **根层有效分叉** - k=1 接受率 81%，可以产生多个高质量分支
2. **深层仍然有用** - k=2: 56%, k=3: 36%，不是接近零
3. **树扩展有价值** - 如果每层都有 50%+ 接受率，树的并行探索才有意义

### 5.3 预期效果

**乐观预期：**
- Native FastMTP: MAL ~2.0-2.5（基于论文 2.03× speedup）
- 动态树松弛: MAL ~2.3-2.8（+15-20%）
- 质量：在 Native 基础上略降 (0-5pp)

**为什么认为会有平衡？**
1. **FastMTP baseline 高** - 有提升空间不大，但也不会严重损害
2. **树验证节点增加** - 5-6 个节点，如果 MTP 好，应该能找到更长路径
3. **松弛适度** - `tau_relax=0.7` 不算激进，只救援高置信度候选

### 5.4 与三方法对比的位置

**当前三方法对比：**
1. Native MTP (FastMTP trained)
2. Cactus + MTP
3. SpecCascade + MTP

**建议：保留动态树作为第四方法！**

理由：
1. 完全不同的方向（树 vs 链）
2. 在好的 MTP 上，树的价值才能体现
3. 可以对比"验证端松弛（Cactus）"vs"提议端扩展（树）"

### 5.5 优化建议

如果树方案仍然不够好，可以尝试：

**优化1: 早期分支策略**
```python
# 让根层和第1层更激进地分支
if depth <= 1:
    tau_early = tau * 0.5  # 更低的阈值
    max_children_early = 4  # 允许更多子节点
```

**优化2: 渐进松弛**
```python
# 深度越大，松弛越保守
tau_relax_depth = tau_relax * exp(depth * 0.1)
```

**优化3: Quality-aware pruning**
```python
# 只保留最有希望的路径
if len(surviving_paths) > 3:
    keep_top_k_paths(3, by=geometric_mean_score)
```

## 6. 完整实验计划

### 6.1 短期（FastMTP）

```bash
# 1. 下载 FastMTP
./scripts/download_fastmtp.sh

# 2. 四方法对比（包含树）
RUN_TAG=fastmtp_four_way_50_$(date +%Y%m%d) \
  ./scripts/run_fastmtp_four_way_50.sh
```

四方法：
1. Native FastMTP (trained baseline)
2. Cactus + FastMTP
3. SpecCascade + FastMTP
4. **Dynamic tree + FastMTP** (我们的方案)

### 6.2 中期（训练 6-layer MTP）

```bash
# 1. 收集 self-distilled 数据
./scripts/collect_mtp_training_data.sh

# 2. 训练 6-layer MTP
./scripts/train_6layer_mtp.sh

# 3. 在 6-layer MTP 上重复四方法对比
```

### 6.3 长期（论文方向）

**Contribution 点：**
1. **多层 MTP 训练** - 首次系统训练 6 层 MTP
2. **树 vs 链对比** - 在好的 MTP 上对比两种方向
3. **质量-速度 Pareto** - 完整的松弛策略对比

## 总结

1. ✅ FastMTP 确实基于 MiMo-7B-RL
2. ✅ 我们可以训练更多层（建议 6 层）
3. ✅ 训练方案清晰：self-distillation + 分层训练
4. ✅ 动态树方案合理，在好的 MTP 上应该有效
5. ✅ 建议保留动态树作为第四方法对比

下一步：先跑 FastMTP 四方法对比，看动态树在训练好的 MTP 上的表现！
