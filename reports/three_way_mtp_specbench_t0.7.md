# 三种 MTP 验证方法：统一 Spec-Bench 对比

## 实验协议

- 模型：Qwen3.5-4B，单张 RTX 4090，vLLM 0.18.0
- 方法：
  1. 标准非松弛概率 MTP
  2. SpecCascade [TokenV3] + 概率 MTP，`alpha=0.5`
  3. Cactus + 概率 MTP，`delta=1.0`
- 采样：`temperature=0.7`，generation seed `42`
- 草稿：`MTP_TOKENS=2`
- 数据：translation、summarization、math_reasoning、rag 各固定抽样 20 条，
  共 80 条
- 数据抽样 seed：`20260729`
- 每条最多生成 128 tokens；请求串行执行
- 每个服务在正式测速前执行相同的 32-token 排除式预热
- 三次运行的 `sample_manifest.json` 完全相同，SHA-256：
  `f478523825376381ad84a08ae29fd9a590c9bbec5fcbc1ad7b3ab8028ea6ec51`

## 总体结果

括号内是相对标准概率 MTP 的变化。草稿接受率的变化使用百分点（pp）。

| 方法 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 | 输出 tokens |
|---|---:|---:|---:|---:|---:|
| 标准概率 MTP | 149.371 | 128.957 | 2.519 | 75.97% | 8101 |
| SpecCascade [TokenV3] | 150.029（+0.44%） | 129.174（+0.17%） | 2.564（+1.75%） | 78.18%（+2.20 pp） | 8010 |
| Cactus + MTP | **163.999（+9.79%）** | **139.747（+8.37%）** | **2.784（+10.48%）** | **89.18%（+13.21 pp）** | 8136 |

## 逐任务结果

| 任务 | 方法 | decode tok/s | e2e tok/s | 平均接受长度 | 草稿接受率 |
|---|---|---:|---:|---:|---:|
| translation | 标准概率 MTP | 145.563 | 67.796 | 2.439 | 71.93% |
| translation | SpecCascade [TokenV3] | 147.545 | 68.808 | 2.504 | 75.21% |
| translation | Cactus + MTP | **162.269** | **74.278** | **2.711** | **85.53%** |
| summarization | 标准概率 MTP | 138.492 | 128.164 | 2.356 | 67.79% |
| summarization | SpecCascade [TokenV3] | 139.483 | 128.811 | 2.392 | 69.58% |
| summarization | Cactus + MTP | **155.559** | **142.443** | **2.667** | **83.37%** |
| math_reasoning | 标准概率 MTP | 166.950 | 157.292 | 2.772 | 88.58% |
| math_reasoning | SpecCascade [TokenV3] | 163.447 | 154.101 | 2.756 | 87.78% |
| math_reasoning | Cactus + MTP | **173.325** | **162.904** | **2.889** | **94.43%** |
| rag | 标准概率 MTP | 146.128 | 134.040 | 2.482 | 74.09% |
| rag | SpecCascade [TokenV3] | 149.427 | 136.440 | 2.582 | 79.11% |
| rag | Cactus + MTP | **164.436** | **149.317** | **2.824** | **91.19%** |

## 相对标准概率 MTP 的变化

| 任务 | 方法 | decode | e2e | 平均接受长度 | 草稿接受率 |
|---|---|---:|---:|---:|---:|
| translation | SpecCascade [TokenV3] | +1.36% | +1.49% | +2.69% | +3.28 pp |
| translation | Cactus + MTP | +11.48% | +9.56% | +11.16% | +13.61 pp |
| summarization | SpecCascade [TokenV3] | +0.72% | +0.50% | +1.52% | +1.79 pp |
| summarization | Cactus + MTP | +12.32% | +11.14% | +13.22% | +15.58 pp |
| math_reasoning | SpecCascade [TokenV3] | -2.10% | -2.03% | -0.58% | -0.80 pp |
| math_reasoning | Cactus + MTP | +3.82% | +3.57% | +4.22% | +5.85 pp |
| rag | SpecCascade [TokenV3] | +2.26% | +1.79% | +4.05% | +5.02 pp |
| rag | Cactus + MTP | +12.53% | +11.40% | +13.78% | +17.10 pp |
| overall | SpecCascade [TokenV3] | **+0.44%** | **+0.17%** | **+1.75%** | **+2.20 pp** |
| overall | Cactus + MTP | **+9.79%** | **+8.37%** | **+10.48%** | **+13.21 pp** |

## 解读与限制

1. 在这次统一单次运行中，SpecCascade [TokenV3] 的整体吞吐与标准概率
   MTP 基本持平；Cactus 的整体 decode 吞吐提高 9.79%。
2. Cactus 在四个任务上均提高 decode 和 e2e 吞吐。标准 MTP 接受率已经较高
   的 math_reasoning 提升最小。
3. `mean acceptance length = 1 + accepted_draft_tokens / draft_rounds`。
   `MTP_TOKENS=2` 时范围是 `[1, 3]`，其中额外的 `1` 是每轮目标模型保证
   提交的恢复或 bonus token。
4. SpecCascade 和 Cactus 都松弛了目标分布，所以输出文本和输出 token 数可以
   与标准 MTP 不同。当前表只能说明效率和接受率，不能说明质量不变。
5. 当前实验只有一个 generation seed、每任务 20 条，没有误差条。正式论文
   结论需要增加多个 seed，并分别测量 BLEU、ROUGE、数学正确率和 RAG 指标。
6. 三组 translation 的 e2e 数值都包含约 4.6 秒 prefill，而其余任务的
   prefill 较短；由于三种方法采用相同协议，该任务仍可横向比较，但总体
   decode tok/s 更直接反映解码阶段差异。

## 原始结果目录

- 标准概率 MTP：
  `results/spec_bench_probabilistic_mtp_t0.7_20260729_224556`
- SpecCascade [TokenV3]：
  `results/spec_bench_spec_cascade_token_v3_a0.5_t0.7_20260729_224806`
- Cactus + MTP：
  `results/spec_bench_cactus_mtp_d1.0_t0.7_20260729_225022`

这些目录默认由 `.gitignore` 排除；本报告记录了可提交的汇总结果。
