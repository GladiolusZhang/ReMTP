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
3. SpecCascade TokenV3 + MTP；
4. Native MTP + Block Verification；
5. Cactus + Block Verification；
6. Current-block debt balanced；
7. Cactus-dominant target surplus；
8. Exact-TV + head calibration；
9. Regret-Calibrated Block Relaxation（当前方法）。

可用 `PROFILES` 选择子集，但正式对比强制包含 `native_mtp` 和 `cactus`。

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
