# Cactus + probabilistic MTP

This branch adapts:

> Yongchang Hao and Lili Mou, “Cactus: Accelerating Auto-Regressive
> Decoding with Constrained Acceptance Speculative Sampling,” ICLR 2026.

The paper and its authors' implementation are available at:

- <https://openreview.net/forum?id=lpUIkCAy9p>
- <https://github.com/MANGA-UOFA/Cactus>

The authors' code patches vLLM 0.7.3 with a separate draft model. This
repository ports the same target-distribution rule to the vLLM 0.18 Qwen3.5
hybrid-model path and uses the model's own MTP head as the drafter.

## Notation and algorithm

This repository uses:

- `q_mtp`: the complete temperature-scaled MTP proposal distribution;
- `p`: the temperature-scaled target-model distribution;
- `D ~ q_mtp`: the sampled MTP draft token;
- `h_D`: the Cactus target distribution associated with the sampled `D`.

The paper reverses the letters `p` and `q`, using `p` for the drafter and `q`
for the verifier.

For each MTP draft position, Cactus computes:

```text
gamma = min(p(D) + sqrt(2 delta p(D) (1 - p(D))), 1)

h_D(D) = gamma
h_D(v) = p(v) * (1 - gamma) / (1 - p(D)),  v != D
```

Verification remains ordinary probabilistic rejection sampling, but its target
is changed from `p` to `h_D`:

```text
accept D with probability min(1, h_D(D) / q_mtp(D))

after rejection:
    sample RECOVER from normalize(max(h_D - q_mtp, 0))
```

Thus the implementation changes both acceptance and recovery. It does not
merely lower a scalar rejection threshold. If both MTP drafts are accepted,
vLLM's bonus token still comes from the original target distribution `p`,
because no MTP proposal distribution exists for that position.

`delta=0` gives `h_D=p` exactly and recovers standard, non-relaxed
probabilistic MTP. Larger `delta` increases the probability of accepting the
current draft while allowing a controlled departure from the verifier
distribution. The paper uses `delta=1` for its Spec-Bench experiment without
task-specific tuning, so this repository uses `1.0` by default.

## Runtime checks

The first real request prints a diagnostic such as:

```text
[ReMTP][Cactus][diagnostic] invoked=1 delta=1 full_q=1 ...
q_mtp(D)=[...]
p_target(D)=[...]
h_cactus(D)=[...]
KL(h||p)=[...]
```

`full_q=1` confirms that verification received the complete MTP distribution.
The probabilistic-MTP diagnostic also reports `ids_match=True`, which checks
that the cached probability rows match the draft token IDs seen by the
verifier, and `seeded_rows=[0]`, which confirms that the single active request
uses its request-local random generator.

## Run

Standard non-relaxed probabilistic MTP:

```bash
MTP_TOKENS=2 ./scripts/serve_probabilistic_mtp.sh
```

In a second terminal:

```bash
./scripts/benchmark_probabilistic_mtp.sh
```

Cactus with the paper's Spec-Bench default:

```bash
CACTUS_DELTA=1.0 MTP_TOKENS=2 ./scripts/serve_cactus_mtp.sh
```

In a second terminal:

```bash
CACTUS_DELTA=1.0 ./scripts/benchmark_cactus_mtp.sh
```

Both benchmark paths use temperature 0.7, generation seed 42, four fixed
Spec-Bench tasks, 20 samples per task, and at most 128 output tokens.

## Current limitations

- The adapter targets vLLM 0.18 and `--max-num-seqs 1`.
- MTP applies temperature sampling but not top-k, top-p, or penalties.
- The PyTorch adapter materializes complete `h_D` rows to reuse vLLM's
  standard residual sampler; it does not port the authors' experimental
  fused Triton kernel.
- Cactus changes the generated distribution when `delta>0`. Throughput and
  acceptance measurements alone do not establish unchanged task quality.
