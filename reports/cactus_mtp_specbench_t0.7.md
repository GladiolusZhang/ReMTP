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
| translation | 标准概率 MTP | 145.563 | 67.796 | 2.439 | 71.93% | 594 |
| translation | Cactus + 概率 MTP | 162.269 | 74.278 | 2.711 | 85.53% | 642 |
| summarization | 标准概率 MTP | 138.492 | 128.164 | 2.356 | 67.79% | 2559 |
| summarization | Cactus + 概率 MTP | 155.559 | 142.443 | 2.667 | 83.37% | 2530 |
| math_reasoning | 标准概率 MTP | 166.950 | 157.292 | 2.772 | 88.58% | 2560 |
| math_reasoning | Cactus + 概率 MTP | 173.325 | 162.904 | 2.889 | 94.43% | 2560 |
| rag | 标准概率 MTP | 146.128 | 134.040 | 2.482 | 74.09% | 2388 |
| rag | Cactus + 概率 MTP | 164.436 | 149.317 | 2.824 | 91.19% | 2404 |
| overall | 标准概率 MTP | 149.371 | 128.957 | 2.519 | 75.97% | 8101 |
| overall | Cactus + 概率 MTP | 163.999 | 139.747 | 2.784 | 89.18% | 8136 |

## Cactus 相对标准非松弛概率 MTP 的变化

| 任务 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 |
|---|---:|---:|---:|---:|
| translation | +11.48% | +9.56% | +11.16% | +13.61 pp |
| summarization | +12.32% | +11.14% | +13.22% | +15.58 pp |
| math_reasoning | +3.82% | +3.57% | +4.22% | +5.85 pp |
| rag | +12.53% | +11.40% | +13.78% | +17.10 pp |
| overall | **+9.79%** | **+8.37%** | **+10.48%** | **+13.21 pp** |

## 结论和限制

1. `delta=1.0` 下，Cactus 将整体平均接受长度从 2.519 提高到
   2.784，草稿接受率提高 13.21 个百分点，对应 decode 吞吐提高
   9.79%。
2. 四个任务的 decode 都提高；标准 MTP 本身接受率最高的
   math_reasoning 增幅最小，为 3.82%。
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
