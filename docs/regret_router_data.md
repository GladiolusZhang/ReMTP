# Regret Router 数据准备与去污染

Router 不读取标准答案，但测试 prompt 本身仍会暴露 benchmark 分布。因此
训练 trace 必须由独立的普通指令语料收集，并在收集前完成去污染。

## 1. 安装本地工具依赖

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate
uv pip install -U datasets pyarrow datasketch 'huggingface-hub>=0.34,<1.0'
```

`huggingface-hub<1.0` 是当前仓库固定的 Transformers/vLLM 兼容约束；不要让
`datasets` 安装过程把它升级到 1.x。

## 2. 构建混合 prompt

先跑 4,000 条 pilot：

```bash
ROUTER_CORPUS_PROFILE=pilot ./scripts/build_regret_router_corpus.sh
```

配额为 UltraChat 1,500、No Robots 1,000、Magicoder 1,000、OASST1
500。确认训练与推理链路后，再构建 8,000 条正式版本：

```bash
ROUTER_CORPUS_PROFILE=full ./scripts/build_regret_router_corpus.sh
```

正式配额为 3,000 / 2,000 / 2,000 / 1,000。脚本只读取：

- `HuggingFaceH4/ultrachat_200k`：`train_sft.prompt`；
- `HuggingFaceH4/no_robots`：`train.prompt`；
- `ise-uiuc/Magicoder-OSS-Instruct-75K`：`train.problem`；
- `OpenAssistant/oasst1`：通过质量检查、未删除的英文根 prompter 消息。

原始混合写入 `data/regret_router/router_corpus_raw.jsonl`，并生成来源、
许可证、配额、随机种子和 SHA256 清单。该目录被 Git 忽略。

No Robots 是 `CC-BY-NC-4.0`。如果许可证要求不能使用非商业数据，可用
Dolly 15k 替换同一配额：

```bash
ROUTER_CORPUS_PROFILE=full \
ROUTER_INSTRUCTION_SOURCE=dolly \
./scripts/build_regret_router_corpus.sh
```

脚本此时读取 `databricks/databricks-dolly-15k` 的 `instruction`，并在
manifest 中记录 `CC-BY-SA-3.0`。第一版不要加入完整 Tulu 3、FLAN、
Persona GSM/MATH/Python/Algebra/IF 或任何 benchmark 派生训练集。

## 3. 准备 benchmark prompt 文件

统一去污染入口需要：

- HumanEval；
- HumanEval+；
- MBPP；
- MBPP+；
- GSM8K test；
- IFEval prompts。

HumanEval 和 GSM8K 默认读取仓库已有路径。其余文件由你下载后，通过绝对
路径传入。任何 JSON、JSONL 或 JSONL.GZ 均可；脚本识别 `prompt`、
`input_prompt`、`question`、`instruction`、`problem`、`text` 和
`messages`。

## 4. 执行统一去污染

```bash
HUMANEVAL_PLUS_DATA=/path/HumanEvalPlus.jsonl.gz \
MBPP_DATA=/path/mbpp_test.jsonl \
MBPP_PLUS_DATA=/path/MbppPlus.jsonl.gz \
IFEVAL_DATA=/path/ifeval_input_data.jsonl \
./scripts/decontaminate_regret_router_corpus.sh
```

正式模式下缺少任一 benchmark 会直接终止。仅调试脚本时可以设置
`ALLOW_PARTIAL_DECONTAMINATION=1`，但这种输出不能用于正式实验。

过滤规则包括：

1. 规范化精确匹配；
2. 连续 13-token gram 重合；
3. 5-token shingle 的 128-permutation MinHash 候选检索，再执行真实
   Jaccard ≥ 0.8 检查；
4. HumanEval 函数名、签名和 docstring 检查。

过滤后语料位于 `data/regret_router/router_corpus.jsonl`，详细本地审计
报告位于 `data/regret_router/decontamination_report.json`。

## 5. 收集与训练

4k pilot：

```bash
ROUTER_CORPUS=data/regret_router/router_corpus.jsonl \
REGRET_ROUTER_COLLECT_DIR=data/regret_router/traces_pilot \
SAMPLES=4000 \
TEMPERATURE=0.7 \
MAX_TOKENS=256 \
./scripts/collect_regret_router.sh

REGRET_ROUTER_TRACE_DIR=data/regret_router/traces_pilot \
EPOCHS=10 BATCH_SIZE=64 ./scripts/train_regret_router.sh
```

每个 trace block 都携带全局唯一的 `collection_id:request_id`。训练器先按
请求划分 90%/10% train/validation，再展开 block；同一请求的相邻 block
不会跨集合。checkpoint 元数据会记录 `split_unit=request_id` 以及两侧的
请求数和 block 数。旧版没有 `request_id` 的 trace 会被明确拒绝，必须
重新收集。

收集脚本默认拒绝向已有 trace 目录追加，防止把不同数据版本无意混合。只有
明确希望合并同协议的多次收集时，才设置 `ALLOW_APPEND_ROUTER_TRACES=1`；
每次运行的 collection ID 不同，因此请求键仍保持全局唯一。

第一轮保持收集和评测温度均为 0.7。验证收益稳定后，再单独收集 0.3 和
1.0 温度的小规模 trace 做泛化实验，不与主训练集的结论混写。
