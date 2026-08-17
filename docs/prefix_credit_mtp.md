# Prefix-Credit native MTP relaxation

## Why this is different

The previous probability, target-band, and Exact-TV variants decide whether
to relax each draft token independently. Prefix-Credit instead uses the fact
that native MTP produces one ordered chain of six full distributions
`Q1,...,Q6`.

For draft token `y_i`, let `p_i=P_i(y_i)`, `q_i=Q_i(y_i)`, and let `a_i` be
the survival probability of the whole draft prefix through position `i`:

```text
a_0 = 1
a_i = min(a_(i-1) * h_i / q_i, 1)
```

An independent verifier considers a token saturated when `h_i=q_i`.
Prefix-Credit considers it saturated only when the **whole prefix** is
repaired:

```text
h_i = min(q_i / a_(i-1), probability cap)
```

Therefore, when an earlier head creates `a_(i-1)<1`, a later confident target
top-1 token may receive `h_i>q_i`. Before saturation, its exact immediate
prefix gain per unit TV is `a_(i-1)/q_i`. The main method permits over-`Q`
credit only when this gain is at least `0.5`, preventing the verifier from
spending large TV mass on an almost irreparable prefix. An `atomic_credit`
ablation is stricter and pays only when full repair is affordable.

## Quality and compute constraints

- TV destinations use the same accurate rule: the MTP candidate must be the
  target top-1 and the target top1-vs-runner-up logit margin must be at least
  `0.10`.
- Per-position and block TV remain explicitly capped.
- The bonus token comes from the original target distribution.
- There is no learned router, cross-block hidden state, task-specific rule,
  extra model forward, or CPU synchronization.
- The six-step recurrence is compiled and runs entirely on GPU.

## Run

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

./scripts/run_humaneval_prefix_credit.sh
```

The wrapper reuses the protocol-compatible full HumanEval results for Native
MTP, Cactus, SpecCascade, and the target-top1 baseline from
`results/humaneval_comparison_target_band_full_01`. It runs only the new
Prefix-Credit profiles. Reused rows are marked `historical` in the combined
table; newly measured rows are marked `new`. A protocol mismatch stops the run
instead of silently rerunning a baseline or mixing incomparable results.

To run only selected new variants:

```bash
PREFIX_CREDIT_NEW_PROFILES="prefix_credit_g025 prefix_credit_g050" \
  ./scripts/run_humaneval_prefix_credit.sh
```

The suite enables a 100-round mechanism audit by default. This adds only an
occasional reporting synchronization and writes the TV/credit statistics to
the local server logs; it does not add a model forward.

To inspect mechanism statistics in a single server run:

```bash
PREFIX_CREDIT_AUDIT_INTERVAL=100 \
  ./scripts/serve_prefix_credit_mtp.sh
```

The main comparison includes:

- native probabilistic MTP;
- Cactus;
- SpecCascade TokenV3;
- the previous target-top1+margin baseline;
- joint verification with the old token cap;
- atomic full-repair Prefix-Credit;
- efficiency-aware Prefix-Credit with gain/TV floors `0.25`, `0.50`, and
  `1.00`.

The `token_cap` row is essential: it keeps the target mask, total TV, and
joint verifier fixed, so the difference isolates ordered prefix credit.
