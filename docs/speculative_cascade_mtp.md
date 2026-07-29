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

### Deterministic MTP adaptation in vLLM 0.18

The Qwen3.5 hybrid-model runner in vLLM 0.18 generates each MTP draft with
`argmax` and does not return the full MTP logits to the rejection sampler.
Accordingly, this implementation uses the same deterministic proposal
convention as native vLLM:

```text
q(D) = 1, q(v != D) = 0
```

With this `q`, TokenV3 has a particularly direct interpretation:

```text
if p_unscaled(D) >= (1 - alpha) max(p_unscaled):
    accept the MTP draft D
else:
    verify/recover using p
```

This is an explicit adaptation of the paper to the available MTP interface.
The generic probabilistic path is also implemented for vLLM runners that
provide complete MTP proposal logits. For deterministic `q`, the global
`chow`, `diff`, and `opt` rules normally collapse to always choosing `q`, so
`token_v3` is the useful default.

## vLLM bonus-token difference

The paper’s generic target distribution is applied to generated positions.
vLLM’s MTP proposer exposes two draft tokens when `MTP_TOKENS=2`, but no third
proposal for the all-accepted bonus position. This implementation therefore
applies `pi` to both MTP draft positions and retains vLLM’s native target-model
`p` bonus. The draft verification and residual formulas are unchanged.

## Run

Terminal 1:

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
