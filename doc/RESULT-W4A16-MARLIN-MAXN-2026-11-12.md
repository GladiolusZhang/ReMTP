# W4A16 + Marlin 目标模型与 MAXN 功耗模式（Jetson Orin NX，2026-11-12）

状态：15W 与 MAXN 两部分均已实测完成（16:14）。机器当前仍处于 MAXN，切回 15W 需要用户执行 `sudo nvpmodel -m 2`（可能需重启）。

## 1. 为什么试 W4A16 + Marlin

15W 分段计时（`results/third_layer_ab_20261111_profile/`）显示，每轮约 1.02 s 的目标 forward 里，bitsandbytes 的 4 位反量化 CUDA 跨度约 0.70 s/轮（每轮 252 次 = 36 层 × 7 个线性层），BF16 线性层约 0.41 s/轮（两者跨度有重叠，不能相加），注意力只有 5.5 ms。原因是公共数值设置把 bnb 在 sm87 上固定为“每层先反量化成 BF16 再做 GEMM”（为了让链式 4 行和树 10 行走同一条数值路径），每次 forward 都要把约 14 GB 的 BF16 权重写一遍、读一遍。

## 2. 做了什么

- 新增 `scripts/quantize_fastmtp_w4a16_rtn.py`：逐张量流式读取 `models/FastMTP`，把 36 层目标线性层（q/k/v/o、gate/up/down，共 252 个）量化为 GPTQ v1 格式（4 位、对称、group 128、无 act-order），四舍五入（RTN）、无校准数据。词嵌入、目标 LM head、norm、bias 和物理 MTP 层（`model.mtp_layers.*`）原样保留 BF16。输出 `models/FastMTP-W4A16-RTN`（5.9 GB，74 秒完成）。
- `scripts/ab_canonical_tree_attention.py` 新增两个臂标记：`+w4a16`（`MODEL_PATH` 指向上述目录、`QUANTIZATION=gptq_marlin`、`LOAD_FORMAT=auto`、`REMTP_JETSON_BNB_DRAFT=1` 保持 MTP 层 BF16，只去掉 `REMTP_TARGET_BNB_ROW_BLOCK` 与 `REMTP_TARGET_BNB_UNIFORM_GEMM` 两个 bnb 专用开关，其余公共数值设置不变）；`+cactus`（改用冻结的 Cactus 服务环境 `servers/cactus/attempt_1`）。

## 3. 正确性检查

- 打包/反量化往返与直接 RTN 逐位相同（CPU 小矩阵）。
- vLLM 0.18 的 Marlin 内核在 sm87 上可用（`check_marlin_supported(uint4b8, 128)` 为真）。4 种层形状 × M=1/4/10/64，Marlin 输出与反量化参考的相对差 2.7e-3~3.5e-3（BF16 舍入量级）。
- 行一致性：同一行在 M=1、4、10 的批中计算结果逐位相同，因此链式与树验证天然走同一数值，不再需要“统一反量化”路径。
- 权重相对误差（抽查 5 层）：RTN int4 g128 0.122~0.140，bnb FP4 block64 0.124~0.134，bnb NF4 block64 0.093~0.096。更正（2026-11-13）：vLLM 对未量化 checkpoint 的 bitsandbytes 现场量化实际是 `quantize_4bit(..., quant_type="nf4", compress_statistics=True)`，部署的是 NF4 而不是 FP4（`target_numerics` 里的 "FP4" 只是历史命名）。所以本 RTN 模型的权重误差明显大于实际部署的 bnb NF4。留出文本上的输出 KL 也证实了这一点：RTN 0.260、NF4 0.136，见 `doc/RESULT-RAVEN-V4-TARGET-GATE-2026-11-13.md`。
- 服务启动日志：`Using MarlinLinearKernel for GPTQMarlinLinearMethod`，模型占 5.83 GiB（bnb 为 5.72 GiB），MTP 层 BF16，与目标共享词嵌入和 LM head。RAVEN 的 3 题全部答对、正常结束。

## 4. 15W 实测（GSM8K 第 17、353、856 题，与基准相同的 prompt/seed/2048 预算）

| 目标内核 | 方法 | 输出 token | 轮数 | 每轮秒 | 每轮 token | decode tok/s | 目录 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| bnb NF4 | Cactus | 1285 | 400 | 1.1421 | 3.213 | 2.813 | `results/w4a16_marlin_ab_20261112_cactus/canonical_0+cactus` |
| bnb NF4 | RAVEN v3 | 1723 | 513 | 1.2155 | 3.359 | 2.763 | `results/third_layer_ab_20261111/`（11-11 测） |
| W4A16 Marlin | Cactus | 2904 | 889 | 0.2498 | 3.267 | 13.075 | `results/w4a16_marlin_ab_20261112_cactus/canonical_0+cactus+w4a16` |
| W4A16 Marlin | RAVEN v3 | 1099 | 334 | 0.3225 | 3.290 | 10.202 | `results/w4a16_marlin_ab_20261112/canonical_0+fast+inplace+meta+skip+w4a16` |

- 换内核后每轮时间：Cactus 1.142 → 0.250 s（4.6 倍），RAVEN v3 1.216 → 0.323 s（3.8 倍）。
- 换内核后 RAVEN 每轮比 Cactus 多约 73 ms（+29%），decode 低 22%。在 bnb NF4 路径下这部分开销被约 1 s 的目标 forward 掩盖；目标 forward 变快后它成为主要瓶颈。
- 限制：只有 3 道题、各方法生成文本不同（Cactus 的第 856 题写满 2048 token），每轮 token 不能用来比较 MAL；RTN 模型的质量未评测。

## 5. MAXN 实测（同样 3 道题，2026-11-12 16:03–16:14，`results/maxn_ab_20261112/`）

`sudo nvpmodel -m 0` 提示需重启，重启后 `nvpmodel -q` 为 MAXN：8 核 CPU 在线，GPU 上限 918 MHz。四组顺序运行，同一 A/B 脚本与冻结服务环境。温度采样（每 30 s）最高 70°C，负载时 GPU 频率 918 MHz，无降频迹象。

| 目标内核 | 方法 | 输出 token | 轮数 | 每轮秒 | decode tok/s | 对 15W 同组每轮时间 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bnb NF4 | Cactus | 1178 | 361 | 0.5404 | 6.038 | 2.11 倍快 |
| bnb NF4 | RAVEN v3 | 502 | 155 | 0.5963 | 5.432 | 2.04 倍快 |
| W4A16 Marlin | Cactus | 481 | 159 | 0.1493 | 20.261 | 1.67 倍快 |
| W4A16 Marlin | RAVEN v3 | 1355 | 416 | 0.1796 | 18.132 | 1.80 倍快 |

- 相对 15W + bnb 的 Cactus（2.813 tok/s），MAXN + W4A16 的 Cactus decode 为 7.2 倍，RAVEN v3 为 6.4 倍。
- RAVEN 每轮比 Cactus 多出的时间：15W bnb 约 73 ms、15W W4A16 约 73 ms、MAXN bnb 约 56 ms、MAXN W4A16 约 30 ms（+20%）。CPU 变快明显缩小了 RAVEN 的调度开销，但在 W4A16 下它占 Cactus 每轮的比例仍约 20%，高于 Spec-Bench 上约 10% 的 MAL 优势。
- 限制：每组只有 3 道题，各组生成文本与长度不同（Cactus W4A16 只生成 481 token，上下文更短），每轮时间受上下文长度影响；未做分段计时。

## 6. 待做

1. 切回 15W 或决定后续实验统一使用 MAXN（与已有 15W 结果不可直接比较）。
2. 若采用 W4A16：在 4090 上做校准的 GPTQ/AWQ 量化以评测质量；W4A16 + MAXN 下对 RAVEN 做分段计时（`AB_PROFILE=1`），重新定位每轮约 30 ms 的额外开销。
