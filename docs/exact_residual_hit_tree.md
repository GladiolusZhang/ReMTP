# Exact-residual-hit tree exploration

> **Superseded diagnostic notice (2026-08-12):** the earlier failed `shadow`
> control was traced to the next MTP round reading the chain-derived target
> hidden row instead of the actually selected BFS tree leaf. Results produced
> before the selected-leaf hidden-state fix are retained only as negative
> diagnostics and must not be compared as valid tree results. After the fix,
> the 20-task `shadow` MAL rose from `2.978/2.982` to `3.430/3.324` on
> GSM8K/HumanEval. The selected `residual_hit_anchor` pilot reached
> `3.569/3.540`, versus Cactus `3.509/3.361`, with quality `90%/80%` versus
> Cactus `90%/75%`. These are single-seed 20-task pilot results, not a formal
> benchmark conclusion.

## Why this version exists

The previous `sampled_primary_reopen` verifier first inspected tree siblings,
selected the best one, and then applied a second relaxed acceptance test. In
the latest 20-task run its selected rescue acceptance was about 97%. This
raised MAL on some trajectories but also changed four previously correct
GSM8K requests to incorrect ones and reduced HumanEval quality. The tree was
deciding which correction to make instead of merely accelerating an already
valid correction.

This version reverses that causal order:

1. sample one primary token from the full FastMTP proposal distribution `Q`;
2. verify it with the same Cactus distribution `H` as Chain Cactus;
3. on rejection, first sample the correction token from the exact residual
   `normalize(max(H - Q, 0))`;
4. only then ask whether a verified tree sibling has that exact token ID;
5. if no sibling matches, commit the correction exactly as Chain Cactus does;
6. if a sibling matches, reuse its target/tree state to generate more tokens
   without changing the correction that was already sampled.

The tree therefore changes *how much verified work can be reused*, not the
identity of the first correction.

## Routes

The unified experiment contains six controlled routes.

| route | behavior after primary rejection | purpose |
|---|---|---|
| `shadow` | always commit exact correction and stop | Cactus-equivalence/control route |
| `anchor` | on an exact tree hit, append one token from the original target distribution | lowest-risk reuse |
| `strict` | on a hit, follow independently sampled branch `Q` with strict `min(1,P/Q)` verification | distribution-faithful continuation |
| `cactus` | on a hit, follow branch `Q` with the same Cactus relaxation | higher-MAL continuation |
| `strict_wide` | strict continuation with a larger four-candidate/20-node cap | measure correction coverage |
| `cactus_wide` | Cactus continuation with the wider tree | coverage plus relaxed continuation |

All routes use a depth-3 sampled primary trunk. Non-primary branch
continuations are sampled with an independent deterministic branch RNG so they
are genuine samples from their recorded `Q` but do not advance the request RNG
used by the primary Cactus path.

## Residual-coverage allocation

An exact correction can hit the tree only when a sibling exists under the
same rejected parent prefix. Generic reach-first allocation tends to give one
child to many root alternatives, leaving no siblings at later primary depths.
The new `residual_coverage` allocator therefore orders a level as follows:

1. meaningful candidates under the sampled primary parent;
2. local-rank-0 sampled continuations of already retained recovery branches;
3. extra backup ranks under non-primary parents.

It remains a maximum-node budget rather than forced filling. Candidate
probability floors and sibling-ratio guards still apply. The narrow anchor
route uses at most 9 nodes; continuations use at most 15; wide routes use at
most 20.

## Reading the result

The combined report includes:

- quality and MAL versus the shared Native/Cactus/SpecCascade baselines;
- actual target nodes per round;
- `correction hit`: fraction of primary-rejection rounds whose already sampled
  correction appears among sibling candidates;
- `reused rounds`: fraction of all speculative rounds that reuse a hit;
- `unlocked/hit`: mean number of verified tree tokens reused per hit.

The important causal checks are:

1. `shadow` should be quality/MAL-equivalent to Chain Cactus up to finite-seed
   execution details; a material difference indicates an implementation bug;
2. if `correction hit` is low, more relaxed verification cannot fix the
   proposal-coverage bottleneck;
3. if hit rate is adequate but `unlocked/hit` is close to one, branch
   continuation is the bottleneck;
4. compare `strict` and `cactus` to isolate continuation relaxation;
5. compare narrow and wide variants to determine whether extra nodes are worth
   validating.

## Run

Quick exploration:

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate
./scripts/run_fastmtp_residual_hit_suite.sh
```

Larger run with live per-sample progress:

```bash
SAMPLES=100 \
PROGRESS_EVERY=1 \
RUN_TAG=fastmtp_residual_hit_n100 \
./scripts/run_fastmtp_residual_hit_suite.sh
```

Run only the currently selected low-risk route:

```bash
SAMPLES=100 \
PROGRESS_EVERY=1 \
RUN_TAG=fastmtp_residual_anchor_n100 \
./scripts/run_fastmtp_selected_residual_anchor.sh
```

The final table is written to:

```text
results/<RUN_TAG>/comparison.md
```

An interrupted run is resumed by repeating the same command and `RUN_TAG`.
Each completed request is now durably appended to
`requests.checkpoint.jsonl`; completed methods and datasets are skipped. The
first 2026-08-12 `fastmtp_residual_hit_n100` attempt predates this checkpoint
write and stopped at HumanEval `75/100`, so only that one partial HumanEval
dataset must restart from sample 1. All earlier routes and its GSM8K result are
reused.

The first route runs Native, Cactus and SpecCascade once. Later routes reuse
those exact result directories through symlinks, so unchanged baselines are
not rerun. HumanEval is still evaluated in restricted Docker containers.

## Current validation and limitation

CPU unit tests cover exact correction, shadow non-reuse, target anchoring,
strict branch continuation, Cactus branch continuation and the allocation
order. A real RTX 4090 FastMTP/TREE_ATTN smoke completed with depth 3,
`N_max=15`, one target forward per round, and an observed exact correction hit
that continued through two branch nodes.

After fixing selected-leaf hidden-state routing, an RTX 4090 pilot with 20
GSM8K and 20 HumanEval tasks completed for `residual_hit_anchor`. It used one
target tree forward per speculative round, averaged `6.359/6.085` target nodes
per round, and obtained `90% / MAL 3.569` and `80% / MAL 3.540`. The same
manifest's Cactus rows were `90% / 3.509` and `75% / 3.361`.

The selected method is not chosen from generic relaxed siblings. Cactus first
samples its residual correction; only an identical already-verified sibling is
reused, after which one original target-distribution anchor is appended. This
explains the measured rescue MAL of about `0.21` without the quality collapse
seen in direct sibling selection.

The result remains a 20-task, single-seed pilot selected during exploration.
Quality granularity is five percentage points, tree verification currently
uses roughly twice the target nodes of the three-token chain, and its measured
throughput remains lower. A fresh 100-task run is required before claiming
general quality or MAL superiority.
