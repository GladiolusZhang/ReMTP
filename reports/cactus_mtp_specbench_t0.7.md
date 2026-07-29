# Cactus + 概率 MTP：Spec-Bench 子集结果

## 配置

- 模型：Qwen3.5-4B，单张 RTX 4090，vLLM 0.18.0
- 基线：完整 `q_mtp` 的标准非松弛概率 MTP
- 方法：同一个概率 MTP + Cactus，`delta=1.0`
- 采样：`temperature=0.7`，generation seed `42`
- 草稿：`MTP_TOKENS=2`
- 数据：translation、summarization、math_reasoning、rag 各固定抽样 20
  条，共 80 条
- 每条最多生成 128 tokens；请求串行执行
- 两个服务在正式测速前执行相同的 32-token 排除式预热
- 两次运行的 `sample_manifest.json` 完全相同，SHA-256：
  `f478523825376381ad84a08ae29fd9a590c9bbec5fcbc1ad7b3ab8028ea6ec51`

## 原始结果

| 任务 | 方法 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 | 输出 tokens |
|---|---|---:|---:|---:|---:|---:|
| translation | 标准概率 MTP | 146.841 | 118.882 | 2.440 | 71.99% | 591 |
| translation | Cactus + 概率 MTP | 161.864 | 130.801 | 2.706 | 85.32% | 640 |
| summarization | 标准概率 MTP | 136.586 | 126.488 | 2.317 | 65.86% | 2560 |
| summarization | Cactus + 概率 MTP | 157.071 | 143.845 | 2.677 | 83.87% | 2560 |
| math_reasoning | 标准概率 MTP | 166.788 | 157.096 | 2.766 | 88.30% | 2560 |
| math_reasoning | Cactus + 概率 MTP | 172.033 | 161.775 | 2.878 | 93.88% | 2560 |
| rag | 标准概率 MTP | 147.703 | 135.169 | 2.496 | 74.79% | 2344 |
| rag | Cactus + 概率 MTP | 164.502 | 149.241 | 2.813 | 90.66% | 2377 |
| overall | 标准概率 MTP | 149.205 | 136.879 | 2.508 | 75.42% | 8055 |
| overall | Cactus + 概率 MTP | 164.110 | 149.463 | 2.780 | 89.00% | 8137 |

## Cactus 相对标准非松弛概率 MTP 的变化

| 任务 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 |
|---|---:|---:|---:|---:|
| translation | +10.23% | +10.03% | +10.92% | +13.33 pp |
| summarization | +15.00% | +13.72% | +15.54% | +18.01 pp |
| math_reasoning | +3.14% | +2.98% | +4.04% | +5.59 pp |
| rag | +11.37% | +10.41% | +12.72% | +15.88 pp |
| overall | **+9.99%** | **+9.19%** | **+10.83%** | **+13.58 pp** |

## 结论和限制

1. `delta=1.0` 下，Cactus 将整体平均接受长度从 2.508 提高到
   2.780，草稿接受率提高 13.58 个百分点，对应 decode 吞吐提高
   9.99%。
2. 四个任务的 decode 都提高；标准 MTP 本身接受率最高的
   math_reasoning 增幅最小，为 3.14%。
3. `delta=0` 时 `h=p`，实现严格退化到标准概率 MTP；`delta=1`
   是论文 Spec-Bench 使用的免调参设置，不代表当前模型的最优质量—速度点。
4. Cactus 在每个草稿位置改变目标分布，因而输出文本和长度可以变化。本实验
   没有计算 BLEU、ROUGE、数学正确率或 RAG F1，不能据此声称质量不变。
5. 这是单个 generation seed、每任务 20 条的效率实验，没有误差条。正式
   实验应至少增加三个 seed，并扫描 `delta=0, 0.1, 0.5, 1.0`。
6. Cactus 只作用于两个 MTP 草稿位置；all-accepted bonus 位置仍由原目标
   模型分布采样。
7. 当前实现复用 vLLM 的 PyTorch 残差采样，没有移植作者实验性的 fused
   Triton kernel。因此结果反映当前 vLLM 0.18 适配，而不是作者 vLLM 0.7.3
   实现的逐算子复现。
