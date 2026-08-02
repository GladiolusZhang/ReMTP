# Current-block verification-debt control

This branch keeps the saturation-aware Exact-TV verifier, block budget
recycling, head calibration, and prefix-value allocation. It replaces
future-only hidden steering with a risk controller that acts before a relaxed
draft token is committed.

For draft position `i`, let

```text
A_strict(i)  = min(1, P_i(y_i) / Q_i(y_i))
A_relaxed(i) = min(1, H_i(y_i) / Q_i(y_i))
debt(i)      = A_relaxed(i) - A_strict(i)
```

`debt(i)` is the exact increase in acceptance probability bought by the
temporary verification distribution. It is computed before sampling and is
therefore a risk signal, not a post-hoc correctness label.

The controller first computes the original Exact-TV allocation, then caps and
recycles it using:

- a per-position acceptance-debt limit;
- a soft-to-hard target log-probability gap multiplier;
- a maximum TV increment per position;
- a maximum multiple of that position's capped Cactus increment;
- a prefix-ordered cumulative debt limit.

When the hard target veto or cumulative limit is reached, later relaxation is
stopped. The main profile falls back to strict verification; an explicit
ablation falls back to independently capped Cactus. Strict rejection sampling
and the residual correction distribution remain unchanged.

## Run the two-seed gate

```bash
SAMPLES=100 \
TEMPERATURE=0.7 \
SEED=42 \
MTP_TOKENS=6 \
./scripts/run_gsm8k_current_block_debt_gate.sh
```

The first seed screens three fixed profiles. A winner advances only when all
three conditions hold against the same-seed Cactus baseline:

```text
Accuracy > Cactus
mean acceptance length > Cactus
E2E tok/s >= Cactus
```

Only that winner is repeated on a second sample seed. The comparison code
also checks that sample manifests and decoding protocol fields are identical.

Low-cost future feedback is auxiliary rather than part of the main method:

```bash
INCLUDE_WEAK_FEEDBACK=1 \
SAMPLES=100 \
./scripts/run_gsm8k_current_block_debt_gate.sh
```

This adds posterior-scale and nonnegative top-1-bias ablations. It does not
restore cross-block hidden steering.

## Run one server

```bash
MTP_TOKENS=6 \
DEBT_POSITION_LIMIT=0.35 \
DEBT_BLOCK_LIMIT=1.20 \
DEBT_SOFT_LOG_GAP=2.0 \
DEBT_HARD_LOG_GAP=6.0 \
DEBT_MAX_POSITION_TV=0.15 \
DEBT_MAX_CACTUS_RATIO=2.0 \
DEBT_FALLBACK=strict \
./scripts/serve_current_block_debt_mtp.sh
```

Set `TARGET_ANCHORED_AUDIT_INTERVAL` to print mean raw/controlled TV,
acceptance debt, fallback positions, stopped positions, and risk-pruned TV.
Benchmark outputs and logs remain local under ignored `results/` and `logs/`
directories.
