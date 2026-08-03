# HumanEval 质量—速度评测

HumanEval 不比较答案文本是否与参考答案逐字一致。每道题只生成一个完整
Python 函数；该函数通过官方测试即计为正确，因此主质量指标是
`pass@1`。生成速度和判题时间严格分离。

## 1. 准备隔离执行环境

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate
./scripts/setup_humaneval.sh
```

默认使用本机已有的 `python:3-slim` Docker 镜像。评测时每道题使用一个
全新容器，并启用：无网络、只读根文件系统、删除 Linux capabilities、
非 root 用户、CPU/内存/PID/文件数限制和宿主超时。脚本不会在宿主 Python
进程中直接执行模型生成代码。

## 2. 下载数据

```bash
./scripts/download_humaneval.sh
```

脚本从 OpenAI 官方仓库下载 `HumanEval.jsonl.gz`，验证 164 条任务和必需
字段后保存到 `data/humaneval/`。数据目录已被 Git 忽略。

若需手动下载：

```bash
mkdir -p data/humaneval
curl -L \
  https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o data/humaneval/HumanEval.jsonl.gz
```

## 3. 先跑 5 条冒烟测试

完整对比前建议先验证服务切换、生成、Docker 判题和汇总链路：

```bash
SAMPLES=5 \
PROFILES="native_mtp cactus regret_calibrated_block" \
./scripts/run_humaneval_comparison.sh
```

## 4. 跑完整对比

```bash
SAMPLES=164 \
TEMPERATURE=0.7 \
SEED=42 \
MTP_TOKENS=6 \
./scripts/run_humaneval_comparison.sh
```

默认顺序测试：

1. Native probabilistic MTP；
2. Cactus + MTP；
3. Cactus + residual regret feedback；
4. SpecCascade TokenV3 + MTP；
5. Native MTP + Block Verification；
6. Cactus + Block Verification；
7. Current-block debt balanced；
8. Cactus-dominant target surplus；
9. Exact-TV + head calibration；
10. Regret-Calibrated Block Relaxation。
11. Target-Mode Rescue + within-block regret。
12. Fused strict-MTP identity control（归因对照）。

可用 `PROFILES` 选择子集，但正式对比强制包含 `native_mtp` 和 `cactus`。

### Target-Mode Rescue + 块内遗憾

当前大改版本只松弛目标概率至少为 `0.5` 的候选，并使用当前块已经分配
的 TV 作为后续位置的负反馈。它不使用跨块 hidden、旧 token 残差或
learned Router。完整说明和固定对比入口见
[target_mode_regret.md](target_mode_regret.md)：

```bash
SAMPLES=164 ./scripts/run_humaneval_target_mode_regret.sh
```

### Exact-TV + 期望遗憾 Router

Router 的无标签 trace 收集、训练和完整 HumanEval 对比命令见
[exact_tv_regret_router.md](exact_tv_regret_router.md)。训练完成后可直接运行：

```bash
SAMPLES=164 ./scripts/run_humaneval_regret_router.sh
```

### Cactus 不变、只反馈后续 MTP 的实验

该入口保持 Cactus verifier 完全不变，只把 causal Cactus acceptance 中
被 Cactus 临时分布压低的目标偏好残差用于下一块第一颗 MTP 草稿：

```bash
# 默认固定抽 40 条，用于小规模机制筛查
./scripts/run_humaneval_cactus_regret.sh

# 参数锁定后跑全部 164 条
SAMPLES=164 CACTUS_REGRET_ALPHA=0.03 \
./scripts/run_humaneval_cactus_regret.sh
```

The regret profile keeps Cactus verification unchanged. By default it uses
the continuous posterior responsibility
`(A_cactus - A_strict) / A_cactus` for each committed draft token, then
projects the lexical `P-H` regret through the shared output head and steers
only the next block's first MTP proposal. Set
`CACTUS_REGRET_RESPONSIBILITY=realized` only for the higher-variance binary
event ablation. The experimental native-feature variants use
`CACTUS_REGRET_INJECTION_SITE=root` with
`CACTUS_REGRET_RESIDUAL_SPACE=boundary_hidden` or `hidden`; they are retained
for ablation but are not the evidence-backed default.

固定包含纯概率 MTP、Cactus、SpecCascade 和 Cactus + residual regret。
反馈不会修改 Cactus 的 `H`、接受公式、纠正分布或 bonus token。

## 5. 输出

本地结果位于：

```text
results/humaneval_comparison_<timestamp>/comparison.md
results/humaneval_comparison_<timestamp>/comparison.csv
results/humaneval_comparison_<timestamp>/comparison.json
```

每个方法还保留逐题生成和判题状态。主表字段为：

- `pass@1`；
- decode tok/s；
- E2E generation tok/s（不含 Docker 判题）；
- mean acceptance length；
- draft acceptance；
- 截断率和判题超时数。

结果与日志目录均被 Git 忽略，不会上传到 GitHub。
