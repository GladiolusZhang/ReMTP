"""Device-side tree construction must make the host path's decisions (2026-11-15)."""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest
import torch

from remtp import gpu_tree
from remtp import raven_fast_path as fast
from remtp.dynamic_mtp_tree import DynamicTreeConfig, path_expansion_value

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device tree construction runs on CUDA")
DEVICE = torch.device("cuda")


def _config(**overrides) -> DynamicTreeConfig:
    values = dict(max_depth=3, max_nodes=9, max_children_per_parent=3, min_sibling_ratio=0.01,
                  min_draft_prob=0.001, threshold_sensitivity=1.5, support_mode="residual_hit_anchor",
                  allocation_mode="residual_coverage")
    values.update(overrides)
    return DynamicTreeConfig(**values)


def _rows(generator: torch.Generator, count: int, vocab: int, case: int) -> torch.Tensor:
    kind = case % 4
    logits = torch.randn(count, vocab, device=DEVICE, generator=generator) * (1.0 + 3.0 * (case % 5))
    if kind == 1:   # BF16 logits: many exactly equal probabilities
        logits = logits.bfloat16().float()
    if kind == 2:   # flat head with an exact tie group at the cut
        logits[:, :12] = logits[:, :12].max() + 1.0
    if kind == 3:   # one dominant token
        logits[:, 5] += 30.0
    return torch.softmax(logits / 0.6, dim=-1)


def _host_selection(rows, sampled, depth, budget, config):
    items = fast.batched_trunk_children(list(rows), [s for s in sampled], depth=depth, backup_budget=budget,
                                        config=config)
    return [([int(x) for x in item.candidate_ids.tolist()], item.candidate_probs, item.entropy) for item in items]


@pytest.mark.parametrize("vocab", [3000, 151680])
def test_selection_matches_batched_host_path(vocab: int) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    config = _config()
    cases = 60 if vocab == 3000 else 12
    for case in range(cases):
        rows = _rows(generator, 3, vocab, case)
        sampled = torch.multinomial(rows, 1, generator=generator).flatten()
        if case % 3 == 0:
            sampled[0] = rows[0].argmax()
        depth = 1 + case % 3
        budget = 2
        device = gpu_tree.select_children(rows, sampled, depth=depth, backup_budget=budget, config=config)
        host = _host_selection(rows, sampled, depth, budget, config)
        tokens, raws, counts = device["tokens"].tolist(), device["raw"].tolist(), device["count"].tolist()
        entropies = device["entropy"].tolist()
        for r, (ids, probs, entropy) in enumerate(host):
            assert ids == tokens[r][:counts[r]], (case, r)
            assert probs == raws[r][:counts[r]], (case, r)
            assert entropy == entropies[r], (case, r)
        assert not device["invalid"].any()


def test_branch_noise_matches_fast_branch_draw() -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    request = torch.Generator(device=DEVICE).manual_seed(123456789)
    metadata = SimpleNamespace(temperature=torch.tensor([0.6], device=DEVICE), generators={0: request})
    rows = _rows(generator, len(gpu_tree.DEPTH3_BRANCH_PATHS), 151680, 1)
    noise = gpu_tree.branch_noise(gpu_tree.DEPTH3_BRANCH_PATHS, request.initial_seed(), 17, rows.shape[1], DEVICE)
    device = (rows / noise).argmax(dim=-1).tolist()
    for row, path in enumerate(gpu_tree.DEPTH3_BRANCH_PATHS):
        reference = fast.sample_branch_q_token_fast(rows[row], metadata, parent_path=path, proposal_round=17)
        assert int(reference[0]) == device[row], path


@pytest.mark.parametrize("depth,n_current", [(2, 1), (2, 2), (2, 3), (3, 2), (3, 4), (3, 7)])
def test_level_budget_matches_host(depth: int, n_current: int) -> None:
    config = _config()
    device = int(gpu_tree.level_budget(torch.tensor(n_current, device=DEVICE), depth, config))
    from remtp.dynamic_mtp_tree import level_expansion_budget
    budget = level_expansion_budget(current_nodes=n_current, next_depth=depth, config=config)
    remaining_levels = config.max_depth - depth + 1
    budget = min(budget, max((config.max_nodes - n_current) // remaining_levels, 1))
    assert device == max(budget, 0)


def test_retention_matches_host_loop() -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    rng = random.Random(5)
    config = _config()
    for case in range(80):
        depth = 2 if case % 2 == 0 else 3
        parents = 3 if depth == 2 else 4
        n_parents = rng.randint(1, parents)
        if depth == 2:
            parent_codes = list(range(parents))
            n_current = n_parents
        else:
            choices = [4 * a + b for a in range(3) for b in range(3) if (a, b) != (0, 0)]
            parent_codes = [0] + sorted(rng.sample(choices, parents - 1))
            n_current = rng.randint(1, 3) + n_parents
        rows = _rows(generator, parents, 3000, case)
        sampled = torch.multinomial(rows, 1, generator=generator).flatten()
        selection = gpu_tree.select_children(rows, sampled, depth=depth, backup_budget=2, config=config)
        parent_logq = [math.log(rng.uniform(1e-4, 1.0)) for _ in range(parents)]
        retained = gpu_tree.retain_level(
            parent_valid=torch.arange(parents, device=DEVICE) < n_parents,
            parent_code=torch.tensor(parent_codes, device=DEVICE),
            parent_logq=torch.tensor(parent_logq, dtype=torch.float64, device=DEVICE),
            selection=selection, n_current=torch.tensor(n_current, device=DEVICE), depth=depth,
            slots=gpu_tree.D2 if depth == 2 else gpu_tree.D3, config=config)
        host_items = _host_selection(rows[:n_parents], sampled[:n_parents], depth, 2, config)
        proposals = []
        for r, (ids, probs, _) in enumerate(host_items):
            parent_path = tuple(parent_codes[r] // 4 ** (depth - 2 - i) % 4 for i in range(depth - 1))
            for rank, probability in enumerate(probs):
                value = path_expansion_value(parent_logq[r] + math.log(max(probability, 1e-30)), depth, config)
                proposals.append((value, parent_path + (rank,)))
        expected = sorted(gpu_tree.host_retained_paths(proposals, n_current, depth, [0.0], config))
        count = int(retained["count"])
        codes = retained["code"].tolist()[:count]
        got = [tuple(c // 4 ** (depth - 1 - i) % 4 for i in range(depth)) for c in codes]
        assert got == expected, case


def test_graphed_levels_equal_eager_across_replays() -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(21)
    config = _config()
    graphs = gpu_tree._GraphedLevels()
    for case in range(12):
        rows = _rows(generator, 4, 151680, case)
        inputs = dict(q=rows, sampled=torch.multinomial(rows, 1, generator=generator).flatten(),
                      parent_valid=torch.arange(4, device=DEVICE) < 1 + case % 4,
                      parent_code=torch.tensor([0, 1, 6, 9], device=DEVICE),
                      parent_logq=torch.rand(4, dtype=torch.float64, device=DEVICE, generator=generator).log(),
                      n_current=torch.tensor(2 + case % 5, device=DEVICE))
        options = dict(depth=3, slots=gpu_tree.D3, backup_budget=2, config=config)
        eager = gpu_tree._level(**inputs, **options)
        graphed = graphs(inputs, **options)
        assert eager.keys() == graphed.keys()
        for name in eager:
            assert torch.equal(eager[name], graphed[name]), (case, name)
    assert len(graphs.entries) == 1
