# 第四轮执行改动记录

状态：两次数值来源已定位；RMSNorm完整模型通过、LMHead微测通过；统一候选后台验收和优化进行中。
尚无新正式质量/速度结论。

| 文件 | 本轮改动 |
| --- | --- |
| `remtp/target_state_audit.py` | 新增默认关闭的Target逐层操作前后/逻辑KV快照、实际源目的复制检查；只保存CPU数据；模型warmup校验本机metadata接口 |
| `remtp/target_diagnostics.py` | 按已生成token范围和帧数上限采集，能覆盖晚于前32调用的真实现场 |
| `scripts/run_adaptive_round2.py` | 显式记录/传递诊断开关、范围、帧数和reference模式，不改变算法默认参数 |
| `scripts/run_target_state_audit.py` | 原2950/29542自然2048请求，核对历史287hash后独立重算每个窗口前缀，输出逐层/逻辑KV/raw概率比较 |
| `tests/test_target_state_audit.py` | 原地融合norm前输入快照独立性、置换block逻辑KV、越界拒绝 |
| `scripts/status_adaptive_round4.sh` | 用户级unit/已提交token/快照/逐请求状态与中文报告入口 |
| 第四轮计划、结果入口、总研究日志 | 阶段放行条件、实际运行证据和失败记录 |

## 已执行

- 相关CPU实际9 passed（3.00秒），编译和diff检查通过。
- 首次GPU诊断在新增探针读取层名时失败：本机Attention使用`layer_name`，误用`prefix`。
  失败保留于 `results/target_state_round4_20261004/`。修正探针，并在实际模型warmup预检metadata/KV接口。
- 新冻结候选 `results/target_state_round4_20261004_v2/natural_code`，原算法/数值关键执行文件与第三轮一致。
  新unit `remtp-target-state-round4-v2.service` 有限运行，关闭对话仍继续。
- 长轨迹、首次算子差异及修复结果待实际完成后追加；上轮60新鲜对照不用于放行本轮长状态。

计划：[第四轮详细执行](../../../plan/adaptive-raven-round4-execution-2026-10-04.md)。

## 详细定位与修复证据

1. v2真实自然483token与旧输出相同；v3完成6参考，但CPU配对误把sampling/prompt-logprob两次
   compute当作多现场。v4按精确调用只读配对：KV首次在第6层原提示位置129分叉，228复制均相同。
   两个探针接口错误（layer_name、TRITON KV layout）及配对错误已修正，原失败保留。
2. 156/288×TREE/TRITON四请求：首次`6:input_norm`输入差0、输出差0.0078125；同长度跨后端逐位相同。
   [安装版vLLM 0.18 CUDA源码](https://github.com/vllm-project/vllm/blob/v0.18.0/csrc/layernorm_kernels.cu)
   在256token切换线程块。新增`remtp/target_norm.py`，用≤255行view调用原plain/fused CUDA函数，
   保留公式、BF16和原地残差；开关`REMTP_TARGET_RMSNORM_ROW_LIMIT=255`，默认0。
3. 真实输入/权重独立GPU微测：256～2048复现旧差，修复后一致。完整模型五组比较全部逐位一致。
   新原自然轨迹六窗口36层/KV一致、原287 top/EOS差0、228复制全相同；旧预算/轨迹未改变。
4. 280窗口hidden相同、top差0.125、TV0.007642。真实最终norm/LMHead权重分离测试证实最终norm一致，
   LMHead小M与大M不同。fixed64未保留小M，首微测退出2并保留；第二测16/32/64各自跨M稳定，
   只有16保持观测现场概率。新增`remtp/target_head.py`，仅实际BF16 ParallelLMHead用原F.linear固定16行，
   复用已有固定GEMM helper；开关`REMTP_TARGET_LM_HEAD_ROW_BLOCK=16`，默认0，所有方法共同采用。
   公共舍入路线改变可能改变自然输出，须重新质量/计时；不宣称数学上的完全无损推理。

## 新增或扩展的执行文件

- `scripts/investigate_prefill_history.py`：位置129的真实长度×后端对照。
- `scripts/investigate_norm_dispatch.py`、`scripts/investigate_lm_head_shape.py`：真实张量/权重算子重放。
- `scripts/run_adaptive_round4.py`：完整冻结来源、晚期/边界/祖先、严格参考、质量优先、三个基线、多种子测试。
- `scripts/launch_adaptive_round4.py`：先CPU预检再冻结候选后台运行，无自动重启。
- `target_diagnostics.py`：有界检查点跨点捕获；只控制记录，不控制生成。
- `run_adaptive_round2.py`：新数值参数、检查点协议记录与显式传递，不改变原算法默认。
- `run_target_state_audit.py`：只有显式修复候选允许新自然轨迹，实际独立对照不省略，旧来源拒绝复用。
- `tests/test_target_norm.py`、`tests/test_round4_controller.py`及state测试：融合alias/阈值边界、最后chunk配对、draft覆盖、旧协议拒绝。

CPU相关19项已实际通过（3.71秒）；统一候选4个prefill+原自然483token+6参考已实际完成，
六晚期top/EOS/完整TV全部0、36层与KV一致、228复制相同；原输出和287hash仍相同。
多配置长验收与正式质量速度待实时中文报告更新。

安装版源码复核发现2048 root在上一完整chunk、最后EOS-only chunk没有prompt-logprobs的特殊评分情况。
新增明确配对和测试，在长测前主动迁移冻结v2控制器；原首版数值文件与GPU结果保持原样，
只读复用完整通过的11个前置请求，核验公共数值/模型/版本/服务脚本及所有非诊断remtp文件。
运行unit改为`remtp-adaptive-round4-v2.service`；源状态/迁移原因保留，最新进度脚本动态读取manifest。
普通候选gate拒绝用exit3及中文说明，与runtime/protocol错误分开；不构成候选达标或目标完成。

## 长评分内存审计与当前v3

- 原完整prompt-logprobs会在安装版sampler生成float32全位置logsoftmax；2048词表矩阵与logits
  合计约2.31GiB，超过Tree运行available1.9GiB，纯Target少0.39GiB仍没有可靠临时余量。
- 长诊断改为精确prefix独立重算、仅最后raw词表；max_tokens1仅触发末分布，丢弃单额外token，
  不计质量/速度，不强制EOS。完整EOS/draft/双方top8、全词表TV照常检查；主评测2048不改。
- 新增chunked single-last选择测试，20项实际3.30s通过；安装版源码/内存实测是改动依据，未等OOM再重复长测。
- 2629/seed26332自然请求完整2048token已落盘；按题/seed/prompt/数据/数值源/精确probe UUID复制，
  旧目录不追加，未完成请求不复用。新冻结v3unit已实际active并完成下一2950请求，四题组2/4。
- 当前unit `remtp-adaptive-round4-v3.service`；进入多配置长验收。已完成的11个前置请求只读复用。
旧冻结代码、旧失败gate和旧结果未修改。微测和带探针请求不计正式质量或速度。
