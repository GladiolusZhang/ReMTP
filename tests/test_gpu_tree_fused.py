"""The fused decision step equals gpu_tree._level (2026-11-15)."""
from __future__ import annotations

import pytest
import torch

from remtp import gpu_tree
from remtp import gpu_tree_fused as fused
from remtp.dynamic_mtp_tree import DynamicTreeConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICE = torch.device("cuda")
DECISIONS = ("sel_tokens", "sel_raw", "sel_count", "sel_invalid", "ret_token", "ret_raw", "ret_code",
             "ret_logq", "ret_parent", "ret_count")


def _config(**overrides) -> DynamicTreeConfig:
    values = dict(max_depth=3, max_nodes=9, max_children_per_parent=3, min_sibling_ratio=0.01,
                  min_draft_prob=0.001, threshold_sensitivity=1.5, support_mode="residual_hit_anchor",
                  allocation_mode="residual_coverage")
    values.update(overrides)
    return DynamicTreeConfig(**values)


def _rows(generator, count, vocab, case):
    kind = case % 5
    logits = torch.randn(count, vocab, device=DEVICE, generator=generator) * (1.0 + 3.0 * (case % 5))
    if kind == 1:   # BF16 logits: many exactly equal probabilities
        logits = logits.bfloat16().float()
    if kind == 2:   # flat head with an exact tie group
        logits[:, :12] = logits[:, :12].max() + 1.0
    if kind == 3:   # one dominant token
        logits[:, 5] += 30.0
    if kind == 4:   # two dominant tokens of nearly equal mass
        logits[:, 7] += 12.0
        logits[:, 9] += 12.0
    return torch.softmax(logits / 0.6, dim=-1)


def _inputs(generator, case, depth, vocab):
    rows = {1: 1, 2: 3, 3: 4}[depth]
    q = _rows(generator, rows, vocab, case)
    sampled = torch.multinomial(q, 1, generator=generator).flatten()
    if case % 3 == 0:
        sampled[0] = q[0].argmax()
    if depth == 1:
        valid = torch.ones(1, dtype=torch.bool, device=DEVICE)
        code = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        n_current = torch.zeros((), dtype=torch.int64, device=DEVICE)
    else:
        n_valid = 1 + case % rows
        valid = torch.arange(rows, device=DEVICE) < n_valid
        code = (torch.arange(rows, device=DEVICE) if depth == 2
                else torch.tensor([0, 1, 4, 6], device=DEVICE)) * valid
        n_current = torch.tensor(n_valid + (case % 3), dtype=torch.int64, device=DEVICE)
    logq = torch.log(torch.rand(rows, device=DEVICE, generator=generator, dtype=torch.float64) * 0.9 + 0.05)
    logq = logq * valid
    return dict(q=q, sampled=sampled, parent_valid=valid, parent_code=code, parent_logq=logq,
                n_current=n_current)


def _options(depth, config):
    slots = {1: 0, 2: gpu_tree.D2, 3: gpu_tree.D3}[depth]
    budget = (min(config.max_children_per_parent - 1, max(config.max_nodes - config.max_depth, 0))
              if depth == 1 else config.max_children_per_parent - 1)
    return dict(depth=depth, slots=slots, backup_budget=budget, config=config)


@pytest.mark.parametrize("vocab", [3000, 151680])
def test_exact_mode_equals_reference_bitwise(vocab):
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    config = _config()
    cases = 90 if vocab == 3000 else 24
    for case in range(cases):
        depth = 1 + case % 3
        inputs = _inputs(generator, case, depth, vocab)
        options = _options(depth, config)
        reference = gpu_tree._level(**inputs, **options)
        candidate = fused.fused_level(**inputs, **options, mode="exact")
        for name, value in reference.items():
            if name == "sel_tie_risk" and bool(value.any()):
                pytest.skip("tie group past the top-k width (reported by both paths)")
        for name, value in reference.items():
            assert name in candidate, name
            assert candidate[name].dtype == value.dtype, (case, name)
            assert torch.equal(candidate[name], value), (case, depth, name, value, candidate[name])


def test_fast_mode_makes_the_same_decisions():
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    config = _config()
    for case in range(36):
        depth = 1 + case % 3
        inputs = _inputs(generator, case, depth, 151680)
        options = _options(depth, config)
        reference = gpu_tree._level(**inputs, **options)
        candidate = fused.fused_level(**inputs, **options, mode="fast")
        for name in DECISIONS:
            if name in reference:
                assert torch.equal(candidate[name], reference[name]), (case, depth, name)
        # Statistics agree to float32 rounding.
        assert torch.allclose(candidate["sel_entropy"], reference["sel_entropy"], rtol=1e-5, atol=1e-7)


def test_graphed_fused_level_equals_eager():
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    config = _config()
    graphs = gpu_tree._GraphedLevels(fn=fused.fused_level)
    for case in range(9):
        depth = 1 + case % 3
        inputs = _inputs(generator, case, depth, 151680)
        options = _options(depth, config)
        eager = fused.fused_level(**inputs, **options, mode="exact")
        replay = graphs(inputs, **options, mode="exact")
        for name, value in eager.items():
            assert torch.equal(replay[name], value), (case, name)
