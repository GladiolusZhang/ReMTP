# Target概率一致性改动记录

> 当前状态（2026-10-06）：第四轮已通过152项长状态与344项fresh数值检查；第五轮104请求训练消融已完成，EOS保护有局部收益，整体质量/速度目标尚未达成。下一步见[完整优化计划](../../../plan/adaptive-raven-round6-path-quality-speed-2026-10-06.md)与[本次规划记录](round6-plan-2026-10-06.md)。下文第三轮失败/执行中说明为历史记录。

> 2026-10-04 最新深入复查：60条新鲜对照（含原old/head1）通过；原持续生成误差尚未定位。
> 本次诊断/文档改动见 [专门记录](deep-audit-2026-10-04.md)，
> 后续以 [长状态修复计划](../../../plan/target-consistency-deep-repair-2026-10-04.md) 为准。

> 最新状态：round3在真实生成的新287-token前缀上再次失败，top误差0.25，后台主动exit2。
> 下文固定案例验收及启动结果仍保留；完整生成修复尚未完成，见
> [第三轮停止核查](../../STATUS-ADAPTIVE-RAVEN-ROUND3-2026-10-04.md)。本次核查仅文档更新，未重启。

日期：2026-10-04。状态：固定条件前缀 GPU 验收通过；第三轮无人值守优化已启动。

完整中文执行记录：[Target 概率一致性修复与优化恢复](../../FIX-TARGET-PROBABILITY-CONSISTENCY-2026-10-04.md)。

## 修改范围

| 文件 | 与原实现的差异 |
| --- | --- |
| `remtp/target_diagnostics.py` | 新增默认关闭的实际Target input/position/hidden/raw logits记录、同hidden投影形状探针；树模式保存root及节点行和拓扑 |
| `remtp/target_numerics.py` | 新增可选sm87统一FP4解量化计算路径、关闭BF16低精度归约、固定16/64行计算块；保留FP4存储与BF16激活，逐层临时反量化 |
| `remtp/target_attention_fp32.py` | Apache-2.0本地vLLM 0.18.0 2D kernel副本；只将P×V改为FP32 / IEEE，不改mask/KV/位置；默认关闭 |
| `remtp/canonical_tree_attention.py` | 真实Target树query按实际祖先顺序索引原KV；token-level table / shared view，无整份KV复制和写入；每step清空索引缓存 |
| `remtp/fastmtp_verified_worker.py` | 全部适用worker显式安装公共数值适配，新增无MTP Target诊断worker |
| `scripts/serve_fastmtp_verified.sh` | 显式诊断/数值开关启用Target worker；独立Target选择attention backend，链基线可使用TRITON_ATTN，树仍TREE_ATTN |
| `scripts/run_adaptive_round2.py` | 新开关纳入协议与服务环境；前缀选择使用稳定组顺序；保存prefix哈希/来源、原始请求/响应，恢复失败检查不能跳过 |
| `scripts/investigate_target_consistency.py` | 新增固定4个实际前缀的48请求对照：echo、单行、真实decode，缓存开/关，各两次 |
| `scripts/validate_target_tree_consistency.py` | 新增真实树D3/D4、备选seed、缓存对照；核对root token/position、回放主/备选祖先；保持0.15门槛，首组失败停止 |
| `tests/test_target_probability_diagnostics.py` | 固定案例选择/歧义、固定形状线性层输入隔离和bias、共同logit偏移/相对/EOS误差门槛 |
| `tests/test_canonical_tree_attention.py` | 物理block置换下精确前缀/祖先、兄弟和拒绝槽不可见、32边界补齐，无原表写入 |
| `deploy/remtp-adaptive-round3.service` | 新冻结来源/结果目录；启动前核验全部概率gate与公共数值配置，协议/质量失败不重启 |
| `scripts/status_adaptive_round3.sh` | 查询本轮服务、逐阶段JSON进度、中文报告和最近日志 |

没有修改已安装torch/vLLM/bitsandbytes源码或模型权重。数值适配在进程内默认关闭、显式启用，
private BNB接口对应当前0.50.2版本；软件升级后需要重新验证。
若采用公共数值/attention配置，将对Native、Cactus、SpecCascade和Adaptive统一应用、重新比较。

## 实际实验

1. 原生Target默认配置48请求完成：prefill/decode自身log-odds误差最高0.625，
   完整词表TV最高0.05144；同hidden投影形状只解释部分差异。
2. CUDA量化矩阵隔离：仅统一BNB路由仍差1.0；仅关闭归约仍差1.0；两者结合M1/M13差0。
   直接BF16 GEMM显示低精度归约影响大批量准确性，关闭后接近FP32参考。
3. 两项组合全模型48请求仍部分失败，不能采用为完成修复。
4. 再加row64固定形状，全模型48请求完成：缓存关闭四例hidden/top log-odds差0；
   缓存开启前三例差0，286例top差0.125。全部满足原0.15的top误差条件，TV约≤8.17e-13；
   此时仍不能代替树及EOS验收。
5. D3/D4、缓存关闭实际16条树请求：root位置/token对齐，变更备选seed root logits一致，
   与默认FLASH_ATTN参考仍失败（top最大0.375、EOS最大0.328125）。已停止该组后续诊断。
6. TRITON原生48请求显示单token默认3D仍有0.3125差异；统一2D后另48请求cache-off四例差0。
7. 新鲜D3/D4 cache-off共16root请求（24项检查）全通过；D3 cache-on8请求286例差0.25，失败。
   本轮全部方法采用cache-off，仍保留普通解码KV缓存，旧cache-on失败证据不覆盖。
8. 新服务独立12条祖先评分发现D3 prefix160差0.1875，尚不能验收；另有D4 TV≈0.001034。
9. FP32 P×V候选24原生+16root+16祖先全部实际完成，D3路径通过，但D4 prefix161差0.25 / EOS0.15234375，
   TV0.06682，整体失败，旧结果保留。新增canonical祖先读取，在 `canonical_tree/` 重新验证。
10. canonical真实16root请求 / 24检查 + 16祖先评分全部通过，top / EOS差均0，
    root TV≤8.31e-12，祖先TV≤1.88e-10。160/161两条前后失败/修复评分prompt逐token完全一致。
    新冻结round3已通过gate和启动预检，user unit已启用并启动，NRestarts=0，完整质量尚未结束。

当前代码完整CPU回归实际379 passed、2 skipped（46.88秒）；git diff和bash语法检查通过。
CUDA隔离kernel能够编译运行，但随机用例两版均相同，不能代替真实前缀修复验收。

## 证据与限制

新证据均在 `results/target_probability_consistency_20261004/`，各候选使用独立冻结代码。
旧 `adaptive_optimization_20261002_fixed` 来源、输出和失败门槛未覆盖。
强制末token只用于构造精确诊断前缀，记录采样约束前raw logits，不作为质量结果。
数值门槛没有放宽；没有通过概率和完整重放门槛前不进入正式选参。
新增逐层临时权重和固定行计算有额外资源成本，最终性能必须在公共协议下重测。
