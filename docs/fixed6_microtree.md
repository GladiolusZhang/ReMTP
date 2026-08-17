# Fixed-6 Micro-Tree

## Scope

Fixed-6 Micro-Tree is an independent strict speculative-decoding extension.
It does not change the frozen chain ReMTP definition or its reported results.
The first version does not apply Cactus or ReMTP relaxation to a tree.

The invariant is:

```text
target forward calls per round = 1
target candidate nodes per round = 6
```

Supported static topologies are:

| name | branch lengths | breadth-first node paths |
|---|---|---|
| `6-chain` | `6` | `(0), (0,0), ..., (0,0,0,0,0,0)` |
| `4+2` | `4,2` | `(0), (1), (0,0), (1,0), (0,0,0), (0,0,0,0)` |
| `3+2+1` | `3,2,1` | `(0), (1), (2), (0,0), (1,0), (0,0,0)` |

Nodes are breadth-first only at the TreeAttention boundary. They are never
treated as a flat autoregressive chain.

## Proposal and strict verification

At temperature zero, root alternatives are the MTP-logit top-k candidates.
Every child is the MTP top-1 conditioned on its own parent path.

With stochastic sampling, roots are sampled without replacement. The exact
conditional proposal distribution is retained for every sampled root. Root
alternatives are checked recursively against the target residual distribution.
After a root is accepted, that branch uses standard sequential speculative
acceptance `min(1, p(y)/q(y))`. The first rejection emits a residual correction;
an entirely accepted branch emits one target bonus token.

Target logits contain seven rows: the common anchor plus six candidate inputs.
A root is scored by the anchor row; a non-root node is scored by its parent's
row. A leaf row supplies its correction/bonus distribution.

## State layout

Qwen3.5 mixes full attention with recurrent GDN layers. Correct tree execution
therefore maintains four independent state families:

- attention KV: unique physical slots per BFS node, with a causal tree mask;
- GDN convolution state: a compact history copied from the logical parent;
- GDN SSM state: a distinct row per node;
- MTP recursive state/KV: unique tree slots and parent-conditioned hidden input.

Logical RoPE positions are branch depths, while physical cache slots remain
unique BFS offsets. After verification, only the selected path is compacted
into the normal contiguous request layout. Discarded branches never alias or
modify the committed state.

The current non-chain implementation is a correctness reference: every GDN
node is advanced with the native one-token kernel from a cloned parent state.
This is intentionally conservative and expensive. A future fused kernel must
match this reference before it can replace it.

## Tests and instrumentation

Run CPU/unit invariants with:

```bash
./scripts/test_fixed6_microtree.sh
```

The tests cover:

- exact node count, BFS ordering, parent rows, positions and masks;
- branch-state non-aliasing for KV, GDN conv, GDN SSM and MTP state;
- selected leaf state equal to independent branch execution;
- probabilistic root residual preserving the target first-token distribution;
- audit rejection of a path that crosses branches.

Every GPU round audit records topology, selected branch/path, output IDs,
target top-1 IDs, target node count and target forward count. Validate it with:

```bash
python -m remtp.fixed6_validation \
  --audit results/fixed6_rounds.jsonl \
  --topology 4+2 \
  --reference-response /tmp/target.json \
  --candidate-response /tmp/tree.json \
  --output results/fixed6_validation.json
```

Non-chain startup remains explicitly guarded because the complete three-
topology gate has not passed:

```bash
REMTP_ALLOW_UNVERIFIED_GDN_TREE=1 \
TREE_TOPOLOGY=4+2 \
./scripts/serve_fixed6_microtree.sh
```

The override is for validation only, not reported benchmarks.

## Current gate status

The `4+2` correctness-reference run matched a 96-token greedy target reference
and exercised both branches plus zero-accept rounds. `3+2+1` exercised all
three branches but diverged at a low-margin greedy position, so the full state
equivalence gate remains open.

An eight-sample GSM8K sanity run (temperature 0.7, 128 output-token cap) gave:

| topology | MAL | decode tok/s | E2E tok/s | MAL vs chain |
|---|---:|---:|---:|---:|
| `6-chain` | 3.764 | 106.501 | 102.094 | — |
| `4+2` | 2.725 | 36.387 | 35.915 | -1.039 |
| `3+2+1` | 2.494 | 33.288 | 32.916 | -1.270 |

This pilot is a go/no-go diagnostic, not a quality result: the short output cap
truncated many answers. Both trees fail the required `MAL >= chain + 0.10` gate
and their E2E slowdown is much larger than 5%. Consequently no full GSM8K or
HumanEval run is authorized, and `tree + ReMTP` is not implemented yet.

With CUDA synchronization around only the target model forward, the same
96-token probe measured 14.675 ms/round for `6-chain` and 39.392 ms/round for
the per-node `4+2` reference (+168.4%). This confirms that the correctness
reference is not an acceptable production kernel even before E2E effects.

## Attribution table (future, only after strict tree passes)

The chain main table remains Native / Cactus / SpecCascade / ReMTP. A separate
tree table will contain:

```text
6-chain + strict
6-chain + ReMTP
6-tree  + strict
6-tree  + ReMTP
```

Attribution is fixed as:

```text
tree coverage = tree-strict - chain-strict
ReMTP relaxation = tree-ReMTP - tree-strict
complete system = tree-ReMTP - chain-strict
```

Tree relaxation is deferred until strict-tree correctness, MAL and latency all
pass their pre-registered gates.
