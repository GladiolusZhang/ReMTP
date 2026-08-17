# MiMo-7B + multi-layer MTP + binary tree

## 1. Scope and model choice

The implementation targets `XiaomiMiMo/MiMo-7B-Base`, not a Qwen3.5
checkpoint. MiMo uses dense Qwen2-style attention, so a six-node tree can be
represented by ordinary TreeAttention masks and branch KV slots; it does not
need Qwen3.5's branch-specific GDN/linear-attention state surgery.

The official checkpoints are split:

- `MiMo-7B-Base` contains target weights and MTP layer 0.
- `MiMo-7B-MTPs` contains the additional pretrained MTP layers 1 and 2.

The download script pins the two repositories to known revisions and creates
`models/MiMo-7B-Base-MTP3`. The output is a lightweight overlay: it contains a
combined safetensors index and symlinks to the downloaded shards, so it does
not duplicate the 7B weights.

Important limitation: Xiaomi says the two extra MTP layers are pretrained-only
and have not been tested with post-trained MiMo checkpoints. Therefore the
three-layer implementation uses the Base checkpoint, is marked experimental,
and must be compared against the `layer0` control before any research claim.

Official references:

- <https://huggingface.co/XiaomiMiMo/MiMo-7B-Base>
- <https://huggingface.co/XiaomiMiMo/MiMo-7B-MTPs>
- <https://github.com/XiaomiMiMo/MiMo>
- <https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md>

## 2. Download and prepare

The following command downloads both repositories from the official Hugging
Face endpoint and builds the overlay. The script deliberately overrides any
stale `HF_ENDPOINT=hf-mirror.com` value inherited from the shell:

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

./scripts/download_mimo_7b_mtp.sh
```

If both repositories have already been downloaded manually:

```bash
BASE_DIR=/path/to/MiMo-7B-Base \
MTP_DIR=/path/to/MiMo-7B-MTPs \
OUTPUT_DIR=/home/llminference/zsy/ReMTP/models/MiMo-7B-Base-MTP3 \
./scripts/prepare_mimo_7b_mtp.sh
```

Validate the result:

```bash
python -m remtp.mimo_checkpoint check models/MiMo-7B-Base-MTP3
```

The validator checks architecture/dimension compatibility, the presence of
physical MTP layers 0/1/2, duplicate weight names and missing shards. It never
modifies downloaded weights.

## 3. Multi-layer MTP runtime

vLLM 0.18 supports MiMo MTP only through physical layer 0. Its model adapter
asserts `spec_step_idx == 0`, and the proposer does not pass a depth index.
Merely setting `num_speculative_tokens=3` therefore repeats layer 0 three
times.

`remtp.mimo_mtp` adds two explicit modes:

| mode | physical layers for D0,D1,D2 | purpose |
|---|---|---|
| `layer0` | `0,0,0` | stock-compatible control |
| `physical` | `0,1,2` | experimental three-layer MiMo |

Physical routing also requires one accepted-prefix KV prefill for every MTP
layer. Xiaomi's vLLM 0.7.3 fork performs `_num_spec_prefill_steps` proposer
passes with `spec_step_idx=0,1,2`; vLLM 0.18's integrated Eagle path performs
only the layer-0 pass. `remtp.mimo_mtp` mirrors the official invariant by
feeding the real-prefix token/target-hidden rows through layers 1/2 for their
cache side effects before recursive drafting. Setting
`REMTP_MIMO_PREFILL_ALL_LAYERS=0` is retained only as an invalid-state
diagnostic: D1/D2 then run with missing prefix history and their target support
collapses.

For more than three linear draft tokens, `physical` cycles `0,1,2`. This is
provided for ablation only; the most defensible first experiment uses exactly
three draft tokens.

The launch scripts pass `--trust-remote-code` because the official MiMo
checkpoint ships its model/config implementation as repository code. Only the
locally downloaded, revision-pinned Xiaomi checkpoint is trusted.

## 4. Chain baselines adapted to MiMo

The paper verification implementations consume proposal distribution `Q`,
target distribution `P` and drafted token IDs; they are not tied to Qwen's
transformer internals. MiMo workers install the physical-layer router first,
then the unchanged verifier:

| `METHOD` | implementation |
|---|---|
| `native` | strict probabilistic MTP |
| `cactus` | Cactus constrained-acceptance distribution |
| `spec_cascade` | Faster Cascades / TokenV3 |
| `block_verification` | lossless joint block verification |
| `cactus_block` | Cactus distribution + block verification |
| `proposal_calibrated` | repository proposal-calibration method |
| `remtp` | frozen chain ReMTP schemes 1+2 |

Start any chain method:

```bash
MTP_TOKENS=3 \
MIMO_MTP_LAYER_MODE=physical \
METHOD=native \
./scripts/serve_mimo_method.sh
```

Equivalent convenience scripts are available for Native, Cactus,
SpecCascade, Block Verification and ReMTP. In another terminal:

```bash
./scripts/request_mimo.sh
```

Use the stock-compatible control to isolate the effect of physical layers:

```bash
MTP_TOKENS=3 \
MIMO_MTP_LAYER_MODE=layer0 \
./scripts/serve_mimo_native.sh
```

All methods must use identical prompts, sampling parameters and MTP depth.
Changing from Qwen3.5 to MiMo invalidates old speed/quality numbers; those
numbers remain historical results, not MiMo baselines.

## 5. Three-level binary-tree algorithm

The primary topology retains two candidates at every parent:

```text
depth 1: 2 root candidates
depth 2: 4 candidates (2 children per root)
depth 3: 8 candidates (2 children per depth-2 node)
total: 14 target validation nodes
```

The maximum accepted draft depth remains three. The root, second level and
third level use physical MTP layers 0, 1 and 2. The target evaluates all 14
nodes in one TreeAttention forward.

vLLM uses `num_speculative_tokens` both as the scheduler slot count and as the
MTP checkpoint's logical `num_nextn_predict_layers`.  The tree needs 14
scheduler slots, while the Xiaomi checkpoint contains only three physical MTP
modules.  `serve_mimo_tree.sh` therefore creates a no-copy configuration view
named `MiMo-7B-Base-MTP3-Tree14`: its logical width is 14, its explicit
`remtp_physical_mtp_layers` value remains 3, and all weights are symlinked from
the MTP3 overlay.  At model construction time `remtp.mimo_mtp` instantiates
only the three physical modules.  This keeps all three quantities consistent:

```text
vLLM scheduler slots       = 14
TreeAttention target nodes = 14
physical MiMo MTP modules  = 3
```

Using 15 speculative slots for the 14-node tree is invalid: the scheduler
will append a real fifteenth draft slot and the target-tree input will no
longer match the topology.

The proposal and verification flow is:

1. Sample two root alternatives without replacement from the root MTP
   distribution. Each candidate stores its exact conditional proposal row.
2. For every retained parent, run the next physical MTP layer and sample two
   children without replacement. Repeat once more to obtain 2/4/8 nodes.
3. Keep nodes in breadth-first order with explicit parent IDs and positions.
4. Target input is `[anchor, node0, ..., node13]`. TreeAttention allows a node
   to attend only to the common prefix and its ancestors.
5. One target forward produces 15 logit rows. Each root is scored by the
   anchor row; every descendant is scored by its parent's row.
6. At every parent, the two children use recursive residual rejection
   sampling. This makes root, depth-2 and depth-3 candidate selection strict.
7. Only selected-path KV slots are compacted into the canonical chain; sibling
   KV entries are discarded. The round terminates with exactly one correction
   or target bonus token.

The code never flattens siblings into a token chain. The audit records 14
target nodes and one target forward per round.

The generic vLLM `Per-position acceptance rate` treats returned tokens as
linear speculative slots, so it is not a tree-node statistic. The MiMo tree
adapter therefore reports its own two quantities:

- `mean accepted depth`: accepted draft nodes per tree round;
- `tree MAL`: accepted draft nodes plus the correction/bonus token committed
  at the end of every round.

### Strict tree

```bash
TREE_TOPOLOGY=2x2x2 \
TREE_VERIFY_MODE=strict \
REMTP_TREE_PROFILE_TARGET=1 \
./scripts/serve_mimo_tree.sh
```

While the server is running it prints one compact block per round. Each of the
14 nodes includes branch/depth, token text, proposal `q`, target `p`, strict
acceptance probability and one of `ACCEPT`, `REJECT`, `NOT_VISITED`.

Every launch creates a separate timestamped directory:

```text
results/mimo_tree_YYYYMMDD_HHMMSS/
  rounds.jsonl      # machine-readable records
  tree_trace.log    # readable per-round decisions
  summary.md        # generated when the server stops
```

To inspect the live trace:

```bash
tail -f results/mimo_tree_*/tree_trace.log
```

To regenerate a summary manually:

```bash
python -m remtp.mimo_tree_report \
  --audit results/mimo_tree_YYYYMMDD_HHMMSS/rounds.jsonl \
  --output results/mimo_tree_YYYYMMDD_HHMMSS/summary.md
```

The old `6-chain`, `4+2` and `3+2+1` layouts remain available only as
structural controls.

### Relaxed verification boundary

The 2/4/8 tree currently uses strict verification only. Candidate-specific
Cactus distributions cannot be independently assigned to two competing
siblings without redefining their shared recursive residual. Relaxation will
be added only after a coherent sibling-group distribution is specified.

The separate dynamic-tree relaxed extension is documented in
[`docs/mimo_dynamic_mtp_tree.md`](mimo_dynamic_mtp_tree.md). It does not change
the strict 2/4/8 method or its statistical semantics.

## 6. Files

| file | role |
|---|---|
| `remtp/mimo_checkpoint.py` | validate downloads, build MTP3 overlay and create the no-copy logical Tree14 view |
| `remtp/mimo_mtp.py` | proposal call/depth to physical MTP layer routing |
| `remtp/mimo_worker.py` | MiMo workers for all chain baselines and the tree |
| `remtp/mimo_tree.py` | strict/path-Cactus tree verification semantics |
| `remtp/mimo_tree_vllm.py` | vLLM TreeAttention, positions, KV compaction and audit adapter |
| `remtp/binary_mtp_tree.py` | complete 2/4/8 topology and strict sibling-group verification |
| `remtp/fixed6_microtree.py` | tested six-node topology/state primitives reused by MiMo |
| `scripts/download_mimo_7b_mtp.sh` | pinned official Hugging Face download |
| `scripts/serve_mimo_method.sh` | unified chain server |
| `scripts/serve_mimo_tree.sh` | 2/4/8 binary-tree server |

## 7. Validation order

Run CPU/structural tests first:

```bash
./scripts/test_mimo_mtp.sh
```

After weights are available:

1. `layer0`, `MTP_TOKENS=1` startup and one completion;
2. `layer0`, `MTP_TOKENS=3` chain control;
3. `physical`, `MTP_TOKENS=3` chain;
4. `2x2x2`, strict tree with a short completion;
5. inspect the timestamped `rounds.jsonl`, `tree_trace.log` and target-forward latency;
6. compare selected-path result/cache state with independent token-by-token
   execution;
7. only after state equivalence passes, run a small strict-tree benchmark;
8. only if strict tree is useful, design a coherent tree relaxation rule.

The local structural suite verifies checkpoint assembly, logical/physical
layer separation, layer routing, topology masks/parents, strict tree
probability semantics and the path-Cactus boundary. A real MiMo GPU smoke has
also completed with 14 scheduled slots, 14 target nodes and one target forward
per round. Treat research conclusions as experimental until the selected-path
KV equivalence test and representative benchmark both pass on the actual
checkpoint.
