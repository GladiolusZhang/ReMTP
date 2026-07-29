# Speculative Cascade + MTP：Spec-Bench 子集结果

## 配置

- 模型：Qwen3.5-4B，单张 RTX 4090，vLLM 0.18.0
- 方法：SpecCascade TokenV3，确定性 MTP argmax 草稿，`alpha=0.5`
- 采样：`temperature=0.7`，generation seed `42`
- 草稿：`MTP_TOKENS=2`
- 数据：translation、summarization、math_reasoning、rag 各固定抽样 20
  条，共 80 条
- 每条最多生成 128 tokens；请求串行执行
- 两次运行的 `sample_manifest.json` 完全相同，SHA-256：
  `f478523825376381ad84a08ae29fd9a590c9bbec5fcbc1ad7b3ab8028ea6ec51`

## 原始结果

| 任务 | 方法 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 | 输出 tokens |
|---|---|---:|---:|---:|---:|---:|
| translation | 原生 MTP | 151.186 | 121.096 | 2.357 | 67.84% | 602 |
| translation | TokenV3 + MTP | 163.442 | 132.448 | 2.570 | 78.48% | 593 |
| summarization | 原生 MTP | 137.785 | 128.077 | 2.187 | 59.34% | 2505 |
| summarization | TokenV3 + MTP | 154.043 | 142.100 | 2.473 | 73.63% | 2526 |
| math_reasoning | 原生 MTP | 172.203 | 162.792 | 2.674 | 83.68% | 2560 |
| math_reasoning | TokenV3 + MTP | 179.156 | 169.081 | 2.812 | 90.61% | 2560 |
| rag | 原生 MTP | 151.062 | 138.337 | 2.402 | 70.11% | 2237 |
| rag | TokenV3 + MTP | 165.454 | 150.687 | 2.664 | 83.22% | 2291 |
| overall | 原生 MTP | 152.478 | 140.077 | 2.403 | 70.14% | 7904 |
| overall | TokenV3 + MTP | 165.482 | 151.527 | 2.637 | 81.85% | 7970 |

## 相对原生 MTP 的变化

| 任务 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 |
|---|---:|---:|---:|---:|
| translation | +8.11% | +9.37% | +9.02% | +10.64 pp |
| summarization | +11.80% | +10.95% | +13.07% | +14.29 pp |
| math_reasoning | +4.04% | +3.86% | +5.19% | +6.93 pp |
| rag | +9.53% | +8.93% | +10.91% | +13.11 pp |
| overall | **+8.53%** | **+8.17%** | **+9.75%** | **+11.71 pp** |

## 结论和限制

1. TokenV3 在四个任务上都提高了接受长度和吞吐；summarization 的
   decode 增幅最大，为 11.80%。
2. math_reasoning 的原生草稿接受率已经较高，因此进一步提升空间最小，
   decode 增幅为 4.04%。
3. 这是单个 generation seed 的效率实验，没有误差条；正式论文实验应增加
   多 seed 重复。
4. TokenV3 改变目标分布，输出文本和长度会随之变化。本实验没有计算 BLEU、
   ROUGE、数学正确率或 RAG F1，因此只能说明效率提升，不能据此声称质量不变。
5. vLLM 当前接口没有给 all-accepted bonus 位置提供 MTP 提议分布，所以该位置
   仍采样目标模型 `p`。论文机制仅应用于两个 MTP 草稿位置。

建议下一步对 `alpha` 做质量—速度扫描，例如
`0.1, 0.3, 0.5, 0.7, 0.9`，并为每个任务加入对应质量指标和至少三个 seed。
