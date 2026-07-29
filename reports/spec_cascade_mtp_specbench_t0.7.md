# Speculative Cascade + MTP：Spec-Bench 子集结果

## 配置

- 模型：Qwen3.5-4B，单张 RTX 4090，vLLM 0.18.0
- 基线：完整 `q` 的概率 MTP，`D ~ q`
- 方法：同一个概率 MTP + SpecCascade TokenV3，`alpha=0.5`
- 采样：`temperature=0.7`，generation seed `42`
- 草稿：`MTP_TOKENS=2`
- 数据：translation、summarization、math_reasoning、rag 各固定抽样 20
  条，共 80 条
- 每条最多生成 128 tokens；请求串行执行
- 两个服务在正式测速前执行相同的 32-token 排除式预热
- 三方法统一运行的 `sample_manifest.json` 完全相同，SHA-256：
  `f478523825376381ad84a08ae29fd9a590c9bbec5fcbc1ad7b3ab8028ea6ec51`

## 原始结果

| 任务 | 方法 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 | 输出 tokens |
|---|---|---:|---:|---:|---:|---:|
| translation | 概率 MTP | 145.563 | 67.796 | 2.439 | 71.93% | 594 |
| translation | TokenV3 + 概率 MTP | 147.545 | 68.808 | 2.504 | 75.21% | 605 |
| summarization | 概率 MTP | 138.492 | 128.164 | 2.356 | 67.79% | 2559 |
| summarization | TokenV3 + 概率 MTP | 139.483 | 128.811 | 2.392 | 69.58% | 2521 |
| math_reasoning | 概率 MTP | 166.950 | 157.292 | 2.772 | 88.58% | 2560 |
| math_reasoning | TokenV3 + 概率 MTP | 163.447 | 154.101 | 2.756 | 87.78% | 2560 |
| rag | 概率 MTP | 146.128 | 134.040 | 2.482 | 74.09% | 2388 |
| rag | TokenV3 + 概率 MTP | 149.427 | 136.440 | 2.582 | 79.11% | 2324 |
| overall | 概率 MTP | 149.371 | 128.957 | 2.519 | 75.97% | 8101 |
| overall | TokenV3 + 概率 MTP | 150.029 | 129.174 | 2.564 | 78.18% | 8010 |

## TokenV3 相对标准概率 MTP 的变化

| 任务 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 |
|---|---:|---:|---:|---:|
| translation | +1.36% | +1.49% | +2.69% | +3.28 pp |
| summarization | +0.72% | +0.50% | +1.52% | +1.79 pp |
| math_reasoning | -2.10% | -2.03% | -0.58% | -0.80 pp |
| rag | +2.26% | +1.79% | +4.05% | +5.02 pp |
| overall | **+0.44%** | **+0.17%** | **+1.75%** | **+2.20 pp** |

## 结论和限制

1. 完整 `q` 解决了级联规则退化问题。运行诊断确认 `q(D)<1`，并且
   TokenV3 实际构造的 `pi` 与 `q`、`p` 均可不同。
2. TokenV3 的整体平均接受长度提高 1.75%，草稿接受率提高 2.20
   个百分点，但 decode 吞吐仅提高 0.44%，e2e 吞吐提高 0.17%；在单次
   运行误差下应视为吞吐基本持平。
3. 任务差异明显：RAG 的 decode 提高 2.26%，math_reasoning 则下降
   2.10%。固定 `alpha=0.5` 不是所有任务的统一最优点。
4. 相比旧的 argmax/点质量 MTP，完整 `q` 的标准 MTP 接受长度提高
   4.86%，但完整 softmax、随机采样和概率传递带来开销，使 decode
   吞吐下降 2.09%。
5. 这是单个 generation seed 的效率实验，没有误差条；正式论文实验应增加
   多 seed 重复。
6. TokenV3 改变目标分布，输出文本和长度会随之变化。本实验没有计算 BLEU、
   ROUGE、数学正确率或 RAG F1，因此只能说明效率提升，不能据此声称质量不变。
7. MTP 只为两个草稿位置生成 `q`；all-accepted bonus 位置没有草稿分布，所以
   仍采样目标模型 `p`。论文机制仅应用于两个 MTP 草稿位置。

建议下一步对 `alpha` 做质量—速度扫描，例如
`0.1, 0.3, 0.5, 0.7, 0.9`，并为每个任务加入对应质量指标和至少三个 seed。
