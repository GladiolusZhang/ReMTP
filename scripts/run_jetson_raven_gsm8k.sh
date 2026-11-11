#!/usr/bin/env bash
# Device-sized GSM8K comparison for FastMTP/RAVEN when Docker is unavailable.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

SAMPLES="${SAMPLES:-100}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1024M}"
HTTP_TIMEOUT="${HTTP_TIMEOUT:-1800}"
SAMPLE_SEED="${SAMPLE_SEED:-20260730}"
GENERATION_SEED="${GENERATION_SEED:-42}"
RUN_TAG="${RUN_TAG:-raven_jetson_gsm8k_n${SAMPLES}}"
RUN_ROOT="$PROJECT_DIR/results/$RUN_TAG"
LOG_ROOT="$RUN_ROOT/logs"
MODEL_NAME="TencentBAC/FastMTP"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:$PORT"
server_pid=""

cleanup_server() {
  [[ -n "$server_pid" ]] || return 0
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -INT -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
trap cleanup_server EXIT
trap 'exit 130' INT TERM

[[ "$SAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo "SAMPLES must be positive" >&2; exit 2; }
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_TOKENS must be positive" >&2; exit 2; }
[[ -f data/gsm8k/test.jsonl && -f models/FastMTP/config.json ]] || {
  echo "FastMTP model or GSM8K test data is missing" >&2; exit 2;
}
if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "$BASE_URL is already occupied" >&2; exit 2
fi

mkdir -p "$LOG_ROOT"
exec 9>"$RUN_ROOT/.controller.lock"
flock -n 9 || { echo "Comparison already running: $RUN_ROOT" >&2; exit 2; }
python - "$RUN_ROOT/protocol.json" "$SAMPLES" "$MAX_TOKENS" "$MAX_MODEL_LEN" \
  "$SAMPLE_SEED" "$GENERATION_SEED" \
  "$KV_CACHE_MEMORY_BYTES" <<'PY'
import hashlib, json, sys
from pathlib import Path
from transformers import AutoTokenizer
from remtp.gsm8k_benchmark import (
    PROMPT_TEMPLATE, benchmark_messages, load_gsm8k, sample_gsm8k,
)
path = Path(sys.argv[1])
tokenizer = AutoTokenizer.from_pretrained("models/FastMTP", trust_remote_code=True)
template = Path("configs/fastmtp_no_think_chat_template.jinja").read_text()
selected = sample_gsm8k(
    load_gsm8k(Path("data/gsm8k/test.jsonl")), int(sys.argv[2]), int(sys.argv[5])
)
max_prompt_tokens = max(
    len(tokenizer.apply_chat_template(
        benchmark_messages(
            PROMPT_TEMPLATE.format(question=row["question"]),
            empty_system_prompt=True,
        ),
        tokenize=True, add_generation_prompt=True,
        chat_template=template, enable_thinking=False,
    ))
    for row in selected
)
if max_prompt_tokens + int(sys.argv[3]) > int(sys.argv[4]):
    raise SystemExit(
        f"Input/output budget exceeds MAX_MODEL_LEN: "
        f"{max_prompt_tokens}+{sys.argv[3]}>{sys.argv[4]}"
    )
protocol = {
    "model": "TencentBAC/FastMTP",
    "target_load": "bitsandbytes 4-bit",
    "draft_load": "BF16",
    "dataset": "GSM8K test",
    "data_sha256": hashlib.sha256(Path("data/gsm8k/test.jsonl").read_bytes()).hexdigest(),
    "samples": int(sys.argv[2]),
    "sample_seed": int(sys.argv[5]),
    "generation_seed": int(sys.argv[6]),
    "temperature": 0.6,
    "max_tokens": int(sys.argv[3]),
    "max_model_len": int(sys.argv[4]),
    "kv_cache_memory_bytes": sys.argv[7],
    "max_prompt_tokens": max_prompt_tokens,
    "methods": ["native", "cactus", "spec_cascade", "uniform", "adaptive"],
    "system_prompt": "",
    "thinking": False,
}
if path.exists():
    existing = json.loads(path.read_text())
    legacy = {k: v for k, v in protocol.items()
              if k not in ("kv_cache_memory_bytes", "max_prompt_tokens")}
    if existing != protocol and not (
        existing == legacy and protocol["kv_cache_memory_bytes"] == "256M"
    ):
        raise SystemExit(f"Protocol mismatch; choose another RUN_TAG: {path}")
path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
PY

for method in native cactus spec_cascade uniform adaptive; do
  run_dir="$RUN_ROOT/$method"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[$method] complete, skipping"
    continue
  fi
  service_method="$method"
  adaptive=0
  mtp_tokens=3
  audit_file=""
  if [[ "$method" == uniform || "$method" == adaptive ]]; then
    service_method=dynamic_tree
    mtp_tokens=9
    [[ "$method" == adaptive ]] && adaptive=1
    audit_file="$LOG_ROOT/${method}_tree_rounds.jsonl"
    if [[ -s "$run_dir/requests.checkpoint.jsonl" ]]; then
      [[ ! -s "$audit_file" ]] || mv "$audit_file" \
        "$LOG_ROOT/${method}_tree_rounds.interrupted_$(date +%Y%m%d_%H%M%S).jsonl"
      : > "$LOG_ROOT/${method}_audit_partial"
    fi
  fi
  log_file="$LOG_ROOT/${method}_server.log"
  if [[ -s "$log_file" ]]; then
    mv "$log_file" "$LOG_ROOT/${method}_server.interrupted_$(date +%Y%m%d_%H%M%S).log"
  fi
  echo "[$method] starting service"
  setsid env METHOD="$service_method" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    PORT="$PORT" KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES" \
    REMTP_DYNAMIC_TREE_ADAPTIVE_RELAXATION="$adaptive" \
    REMTP_TREE_AUDIT_JSONL="$audit_file" \
    "$PROJECT_DIR/scripts/serve_fastmtp_jetson.sh" >"$log_file" 2>&1 &
  server_pid=$!
  healthy=0
  for _ in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    if grep -Fq "method=$service_method worker=active" "$log_file" &&
       grep -Fq "Application startup complete." "$log_file" &&
       curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      healthy=1
      break
    fi
    sleep 2
  done
  if [[ "$healthy" != 1 ]]; then
    tail -n 70 "$log_file" >&2
    echo "[$method] service did not become healthy" >&2
    exit 1
  fi
  echo "[$method] generating $SAMPLES GSM8K samples"
  resume=()
  [[ -d "$run_dir" ]] && resume=(--resume)
  python -m remtp.gsm8k_benchmark \
    --data data/gsm8k/test.jsonl --samples "$SAMPLES" \
    --sample-seed "$SAMPLE_SEED" --temperature 0.6 \
    --generation-seed "$GENERATION_SEED" --max-tokens "$MAX_TOKENS" \
    --mtp-tokens "$mtp_tokens" --base-url "$BASE_URL" --model "$MODEL_NAME" \
    --run-name "$method" --output-dir "$run_dir" \
    --timeout "$HTTP_TIMEOUT" \
    --progress-every 1 --skip-warmup --empty-system-prompt \
    "${resume[@]}" 2>&1 | tee "$LOG_ROOT/${method}_benchmark.log"
  if [[ -n "$audit_file" ]]; then
    if [[ -f "$LOG_ROOT/${method}_audit_partial" ]]; then
      cp "$audit_file" "$run_dir/tree_rounds.last_session.jsonl"
      echo "[$method] resumed run: full tree-round metrics unavailable"
    else
      cp "$audit_file" "$run_dir/tree_rounds.jsonl"
      python -m remtp.tree_benchmark_metrics \
        --run-dir "$run_dir" --audit "$run_dir/tree_rounds.jsonl"
    fi
  fi
  cleanup_server
done

python - "$RUN_ROOT" <<'PY'
import json, statistics, sys
from pathlib import Path
from remtp.gsm8k_benchmark import HASH_ANSWER_PATTERN, normalize_number
root = Path(sys.argv[1])
p = json.loads((root / "protocol.json").read_text())
lines = [
    "# Jetson FastMTP/RAVEN GSM8K 五方法对照报告", "",
    "## 协议与口径", "",
    f"GSM8K test 固定抽取 {p['samples']} 题；temperature 0.6；每题最多输出 "
    f"{p['max_tokens']} token；抽样种子 {p['sample_seed']}，生成种子 "
    f"{p['generation_seed']}。Target 采用 {p['target_load']}，MTP 草稿层 BF16；"
    f"最大上下文 {p['max_model_len']} token，KV 预算 {p['kv_cache_memory_bytes']}；"
    "空 system prompt，关闭可见思考。五组使用相同题目与答案。", "",
    "“原评测正确”沿用仓库答案解析：优先 `####`，其次 boxed，最后回退到输出末尾数字。"
    "“正常结束且判对”进一步要求 `finish_reason=stop`；“严格最终行正确”还要求"
    "最后一行完整匹配正确的 `#### <数字>`。严格口径也可能排除有效的 boxed 等答案格式，"
    "应作为最终答案可靠性与格式指标阅读。", "",
    "吞吐为全组输出 token 数除以请求总耗时；MAL 为平均每次验证轮的接受长度。"
    "不同方法生成的 token 数不同，吞吐不能直接等同于每题完成速度。", "",
    "## 结果", "",
    "| 方法 | 原评测正确 | 正常结束且判对 | 严格最终行正确 | 达输出上限 | "
    "解码 tok/s | 端到端 tok/s | 平均输出 token/题 | 请求耗时中位数/s | MAL |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
stats = {}
for method in p["methods"]:
    path = root / method / "summary.json"
    if not path.exists():
        lines.append(f"| {method} | 未完成 | - | - | - | - | - | - | - | - |")
        continue
    x = json.loads(path.read_text())["results"][0]
    records = [json.loads(line) for line in
               (root / method / "requests.jsonl").read_text().splitlines() if line]
    strict_correct = 0
    stopped_correct = sum(record["correct"] and record["finish_reason"] == "stop"
                          for record in records)
    for record in records:
        if record["finish_reason"] != "stop":
            continue
        output_lines = record["output"].rstrip().splitlines()
        if not output_lines:
            continue
        last_line = output_lines[-1]
        match = HASH_ANSWER_PATTERN.fullmatch(last_line)
        if match and normalize_number(match.group(1)) == record["gold_answer"]:
            strict_correct += 1
    stats[method] = {
        "summary": x, "records": records, "strict": strict_correct,
        "stopped_correct": stopped_correct,
    }
    lines.append(
        f"| {method} | {x['correct']}/{x['samples']} "
        f"({x['accuracy']:.1%}) | {stopped_correct}/{x['samples']} | "
        f"{strict_correct}/{x['samples']} | "
        f"{x['truncated_outputs']}/{x['samples']} | "
        f"{x['decode_tok_s']:.3f} | {x['e2e_output_tok_s']:.3f} | "
        f"{x['output_tokens'] / x['samples']:.1f} | "
        f"{statistics.median(record['client_seconds'] for record in records):.1f} | "
        f"{x['mean_acceptance_length']:.3f} |"
    )
if all(method in stats for method in p["methods"]):
    n = stats["native"]["summary"]
    u = stats["uniform"]["summary"]
    a = stats["adaptive"]["summary"]
    c = stats["cactus"]["summary"]
    native_median = statistics.median(r["client_seconds"] for r in stats["native"]["records"])
    cactus_median = statistics.median(r["client_seconds"] for r in stats["cactus"]["records"])
    uniform_by_id = {r["question_id"]: r for r in stats["uniform"]["records"]}
    adaptive_by_id = {r["question_id"]: r for r in stats["adaptive"]["records"]}
    uniform_only = sum(r["correct"] and not adaptive_by_id[qid]["correct"]
                       for qid, r in uniform_by_id.items())
    adaptive_only = sum(r["correct"] and not uniform_by_id[qid]["correct"]
                        for qid, r in adaptive_by_id.items())
    lines += [
        "", "## 结果分析", "",
        f"- 原评测口径下，uniform 为 {u['correct']}/{p['samples']}，adaptive 为 "
        f"{a['correct']}/{p['samples']}。同题配对后，uniform 独有判对 "
        f"{uniform_only} 题，adaptive 独有判对 {adaptive_only} 题。"
        "这次数据没有显示 adaptive 的质量收益。",
        f"- 排除截断后，Native、Cactus、SpecCascade、uniform、adaptive 分别判对 "
        + "、".join(str(stats[m]["stopped_correct"]) for m in p["methods"])
        + " 题；严格最终行正确分别为 "
        + "、".join(str(stats[m]["strict"]) for m in p["methods"])
        + " 题。原评测口径会把部分截断答案判对，而严格口径也排除了某些有效的非标准格式。",
        f"- Cactus 的端到端吞吐为 {c['e2e_output_tok_s']:.3f} tok/s，"
        f"比 Native 高 {(c['e2e_output_tok_s'] / n['e2e_output_tok_s'] - 1):.1%}；"
        f"uniform 和 adaptive 分别高 "
        f"{(u['e2e_output_tok_s'] / n['e2e_output_tok_s'] - 1):.1%}、"
        f"{(a['e2e_output_tok_s'] / n['e2e_output_tok_s'] - 1):.1%}。"
        f"但 Cactus 的单题耗时中位数为 {cactus_median:.1f} 秒，高于 Native 的 "
        f"{native_median:.1f} 秒；各组输出长度、耗时分布不同，"
        "吞吐增幅不是同等幅度的任务耗时缩短。",
        f"- uniform 与 adaptive 的 MAL 为 {u['mean_acceptance_length']:.3f}、"
        f"{a['mean_acceptance_length']:.3f}，差值仅 "
        f"{a['mean_acceptance_length'] - u['mean_acceptance_length']:.3f}；"
        f"adaptive 的端到端吞吐为 {a['e2e_output_tok_s']:.3f} tok/s，"
        f"略低于 uniform 的 {u['e2e_output_tok_s']:.3f} tok/s。"
        "更长的接受长度没有在这次运行中转化为可见的吞吐优势。",
        "", "## 解释边界", "",
        "- 本报告是一次固定抽样和生成种子的设备内对照；没有重复种子、置信区间或 HumanEval，"
        "不能据此断言细小差异具有稳定性。",
        "- Target 为 4 位权重，原始 BF16 Target 的质量与速度尚未在本机测量；"
        "HumanEval 执行评分仍受 Docker 访问权限限制。",
    ]
    tree_info = []
    changed_relaxation_nodes = {}
    for method in ("uniform", "adaptive"):
        tree_path = root / method / "tree_metrics.json"
        if tree_path.exists():
            tree = json.loads(tree_path.read_text())["tree"]
            tree_info.append(
                f"{method} {tree['rounds']:,} 轮、每轮 "
                f"{tree['target_forward_calls_per_round']:.1f} 次 Target forward"
            )
        audit_path = root / method / "tree_rounds.jsonl"
        if audit_path.exists():
            changed_relaxation_nodes[method] = sum(
                node.get("relaxation_depth_weight") !=
                node.get("base_relaxation_depth_weight")
                for line in audit_path.open() for node in json.loads(line).get("nodes", [])
                if "relaxation_depth_weight" in node
            )
    if len(tree_info) == 2:
        lines.insert(lines.index("## 解释边界") - 1,
                     "- 树审计记录 " + "；".join(tree_info) + "，确认运行了树验证路径。")
    if len(changed_relaxation_nodes) == 2:
        lines.insert(lines.index("## 解释边界") - 1,
                     f"- 审计中 adaptive 有 {changed_relaxation_nodes['adaptive']:,} 个节点的"
                     "实际松弛权重不同于基础权重，uniform 为 "
                     f"{changed_relaxation_nodes['uniform']:,} 个；"
                     "adaptive 开关确实生效，但本次未体现明确的质量或吞吐增益。")
(root / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
[[ -f "$RUN_ROOT/COMPLETE" ]] || date -u '+%Y-%m-%d %H:%M:%S UTC' > "$RUN_ROOT/COMPLETE"
