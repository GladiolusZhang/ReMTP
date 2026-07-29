# MTP Speculative Cascade

This branch adapts the target-distribution mechanism from:

> Harikrishna Narasimhan et al., “Faster Cascades via Speculative
> Decoding,” ICLR 2025.

The authors provide an illustrative Gemma/JAX Colab at
<https://github.com/google-research/google-research/tree/master/speculative_cascades>.
It evaluates the target distributions one token at a time and does not contain
a vLLM, MTP, or block-verification implementation. This repository therefore
implements the paper’s distributions directly in vLLM 0.18.0.

## Mapping to MTP

For each draft position:

- `q` is the MTP proposal distribution;
- `p` is the temperature-scaled distribution emitted by the target
  Qwen3.5 model during parallel verification;
- `pi` is the speculative-cascade target distribution;
- a draft `D` is accepted with probability `min(1, pi(D) / q(D))`;
- after rejection, recovery samples from `normalize(max(pi - q, 0))`.

The default rule is TokenV3, equation (15) in the paper. It keeps MTP
probability mass on tokens that the unscaled target model regards as
high-confidence:

```text
T_alpha = {v: p_unscaled(v) >= (1 - alpha) max(p_unscaled)}
pi(v) = q(v) 1[v in T_alpha] + p(v) sum_{u not in T_alpha} q(u)
```

`chow`, `diff`, `opt`, and `token_v3` are configurable. Following Appendix
C.1, confidence decisions use unscaled distributions, while sampling and the
OPT TV distance use temperature-scaled distributions.

### Full MTP distribution adapter

The stock Qwen3.5 hybrid-model runner in vLLM 0.18 uses MTP `argmax` and passes
`draft_probs=None` into verification. That makes the proposal look like the
point mass `q(D)=1`, causing several cascade rules to degenerate.

`remtp.probabilistic_mtp` replaces this narrow path:

```text
MTP logits z
    -> q = softmax(z / temperature)
    -> sample D ~ q
    -> cache the complete q vector for D
    -> pass D and q together to verification
```

Standard probabilistic MTP then verifies with
`min(1, p(D) / q(D))`. SpecCascade constructs `pi` from the same complete `q`
and verifies with `min(1, pi(D) / q(D))`. Startup diagnostics print `q(D)`,
`max(q)`, entropy `H(q)`, and whether the cached token IDs exactly match the
drafts seen by verification.

The MTP sampler currently applies temperature but intentionally ignores
top-k, top-p, and penalties, matching vLLM's own unused probabilistic MTP
helper. The benchmark therefore uses temperature sampling without those
additional transformations.

## vLLM bonus-token difference

The paper’s generic target distribution is applied to generated positions.
The adapter exposes complete `q` distributions for two MTP draft tokens when
`MTP_TOKENS=2`, but there is no MTP proposal for the all-accepted bonus
position. This implementation therefore applies `pi` to both MTP draft
positions and retains vLLM's native target-model `p` bonus.

## Run

Standard probabilistic MTP, terminal 1:

```bash
MTP_TOKENS=2 ./scripts/serve_probabilistic_mtp.sh
```

Terminal 2:

```bash
./scripts/benchmark_probabilistic_mtp.sh
```

SpecCascade with the same probabilistic MTP, terminal 1:

```bash
CASCADE_RULE=token_v3 CASCADE_ALPHA=0.5 \
MTP_TOKENS=2 ./scripts/serve_spec_cascade.sh
```

Terminal 2:

```bash
CASCADE_RULE=token_v3 CASCADE_ALPHA=0.5 \
./scripts/benchmark_spec_cascade.sh
```

Defaults match the native-MTP benchmark: temperature 0.7, seed 42, four
Spec-Bench tasks, 20 fixed samples per task, and at most 128 output tokens.
