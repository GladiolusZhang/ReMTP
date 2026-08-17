# Small-sample decision tracing and tuning

This guide is for debugging a few requests. Trace collection synchronizes GPU
tensors and writes JSONL every speculative round, so **never use a traced run
to report tok/s**.

## One-command tests

Stop the service currently using port 8000, then run relaxed ReMTP on three
GSM8K questions:

```bash
cd /home/llminference/zsy/ReMTP
source .venv/bin/activate

SAMPLES=3 MAX_TOKENS=96 MTP_TOKENS=6 \
  ./scripts/run_remtp_decision_trace.sh gsm8k
```

For two HumanEval prompts (generation only; no Docker execution is needed):

```bash
SAMPLES=2 MAX_TOKENS=128 MTP_TOKENS=6 \
  ./scripts/run_remtp_decision_trace.sh humaneval
```

The command prints the path to `decision_trace.md`. It contains:

- every MTP draft token and the target/MTP top candidates;
- target probability `P(y)`, proposal probability `Q(y)`, strict acceptance
  probability and relaxed acceptance probability;
- TV assigned to each candidate;
- `accepted/strict`, `accepted/relaxed-only`, first `rejected`, and later
  `skipped` positions;
- the committed recovery or bonus token;
- the final generated answer for each request.

To inspect strict Proposal-Calibrated MTP instead:

```bash
SAMPLES=3 ./scripts/run_proposal_decision_trace.sh gsm8k
```

That method calibrates `Q` but does not relax target verification.

## Adjusting actual ReMTP relaxation

The frozen defaults live in `scripts/serve_remtp_mtp.sh`. The following
environment variables can override them without editing code:

| variable | default | larger value generally does what |
|---|---:|---|
| `RISK_BASE_GAP` | 2.15 | admits candidates farther from target top-1 in ambiguous contexts |
| `REMTP_FINAL_GAP_FLOOR` | 1.15 | admits farther candidates even when the target is decisive |
| `RISK_BLOCK_TV` | 2.10 | permits more total probability transfer per six-token block |
| `RISK_PER_TOKEN_TV` | 0.65 | permits more probability transfer to one candidate |
| `CACTUS_DELTA` | 1.25 | raises the upstream Cactus-derived desired TV |
| `RISK_BUDGET` | 5.50 | raises the block risk-mass ceiling |
| `RISK_MAX_RANK` / `RISK_SCHEME2_RANK` | 8 / 8 | admits lower-ranked target candidates |

Change one dimension first. For example, compare the default with a slightly
larger block budget:

```bash
RUN_TAG=default SAMPLES=3 \
  ./scripts/run_remtp_decision_trace.sh gsm8k

RUN_TAG=block_tv_240 SAMPLES=3 RISK_BLOCK_TV=2.40 \
  ./scripts/run_remtp_decision_trace.sh gsm8k
```

If `allocated TV` does not rise, another constraint is binding. Read the trace:

- candidate gap outside the threshold: increase `RISK_BASE_GAP` cautiously;
- candidate is eligible but TV saturates at 0.65: increase
  `RISK_PER_TOKEN_TV`;
- all positions receive some TV but the block total is 2.10: increase
  `RISK_BLOCK_TV`;
- desired TV itself is small: increase `CACTUS_DELTA`;
- candidate rank is over 8: increase both rank variables, with high quality
  risk;
- sentinel rejects a position: inspect `remtp/risk_entropy_mtp.py` before
  weakening the future-support checks.

Do not change all knobs simultaneously: the trace can then show that MAL rose,
but cannot identify which rule caused a quality regression.

## Files to edit

Start from the narrowest file:

1. `scripts/serve_remtp_mtp.sh`: frozen method defaults and the safest place
   for parameter experiments.
2. `remtp/risk_entropy_mtp.py`: actual ReMTP eligibility, target margin,
   sentinel, TV allocation, temporary verification distribution, and debt.
3. `remtp/block_oracle.py`: debug-only round records; edits here must not
   affect verification.
4. `remtp/proposal_calibration.py`: strict proposal calibration `Q -> Q_tilde`
   and its separate trace; this is not target-verification relaxation.
5. `remtp/probabilistic_mtp.py`: MTP sampling and transporting the complete Q
   into vLLM. Change this only when modifying proposal generation semantics.
6. `remtp/worker.py`: selects and installs the vLLM worker hooks.

Inside `remtp/risk_entropy_mtp.py`, the main flow is:

```text
risk_entropy_distribution
  -> compute P(y), Q(y), target rank/gap/margin
  -> decide local eligibility and sentinel support
  -> compute desired TV and allocate one block budget
  -> construct temporary verifier H
  -> compute relaxed acceptance min(1, H(y)/Q(y))

_risk_entropy_rejection_sample
  -> use vLLM's rejection/residual kernels with H
  -> commit accepted prefix plus recovery/bonus token
```

After a small trace looks sensible, restart without audit variables and use
the normal benchmark scripts for valid accuracy and speed measurements.
