"""Batched per-level candidate selection must equal the per-parent fast path."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

from remtp.raven_fast_path import batched_trunk_children, cactus_trunk_children_fast, resolve_batched_counts

_spec = importlib.util.spec_from_file_location(
    "_tree_tests", pathlib.Path(__file__).with_name("test_dynamic_mtp_tree.py"))
_tree_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tree_tests)
_residual_hit_config = _tree_tests._residual_hit_config


def _rows(generator: torch.Generator, vocab: int) -> list[torch.Tensor]:
    rows = []
    for sharp in (0.3, 1.0, 3.0, 8.0):
        rows.append(torch.softmax(torch.randn(vocab, generator=generator) * sharp, dim=-1))
    flat = torch.full((vocab,), 1.0 / vocab) + torch.rand(vocab, generator=generator) * 1e-6
    rows.append(flat / flat.sum())                       # nothing reaches min_draft_prob
    rows.append(torch.nn.functional.one_hot(torch.tensor(3), vocab).float())
    tied = torch.softmax(torch.randn(vocab, generator=generator), dim=-1)
    tied[7] = tied[11] = tied.max() + 0.05               # tie at the top -> reference fallback
    rows.append(tied / tied.sum())
    return rows


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("budget", [0, 1, 2])
def test_batched_equals_per_parent(depth, budget):
    config = _residual_hit_config("residual_hit_anchor", max_nodes=9, max_children_per_parent=3)
    data = torch.Generator().manual_seed(100 * depth + budget)
    for _ in range(25):
        rows = _rows(data, 600)
        trunks = [torch.multinomial(row, 1, generator=data)[0] for row in rows]
        items = batched_trunk_children(rows, trunks, depth=depth, backup_budget=budget, config=config)
        resolve_batched_counts(items)
        for row, trunk, item in zip(rows, trunks, items):
            ids, selection = cactus_trunk_children_fast(row, trunk, depth=depth, backup_budget=budget, config=config)
            assert torch.equal(item.candidate_ids, ids)
            assert item.candidate_probs == row[ids].tolist()
            assert (item.entropy, item.threshold, item.q_max, item.sibling_added) == (
                selection.entropy, selection.threshold, selection.q_max, selection.sibling_added)
            assert item.threshold_candidate_count == selection.threshold_candidate_count
            assert item.candidate_count_before_budget == selection.candidate_count_before_budget


def test_threshold_comparison_matches_device_semantics():
    """A threshold between float32 neighbours must classify like ``q >= threshold``."""
    config = _residual_hit_config("residual_hit_anchor", max_nodes=9, max_children_per_parent=3,
                                  min_draft_prob=0.2)
    row = torch.zeros(16)
    row[:5] = torch.tensor([0.4, 0.2, 0.2 - 1e-9, 0.1, 0.1 - 1e-7])
    row = row / row.sum()
    trunk = torch.tensor(0)
    item = batched_trunk_children([row], [trunk], depth=1, backup_budget=2, config=config)[0]
    resolve_batched_counts([item])
    ids, selection = cactus_trunk_children_fast(row, trunk, depth=1, backup_budget=2, config=config)
    assert torch.equal(item.candidate_ids, ids)
    assert item.threshold_candidate_count == selection.threshold_candidate_count


def test_invalid_proposal_raises():
    config = _residual_hit_config("residual_hit_anchor", max_nodes=9, max_children_per_parent=3)
    row = torch.full((8,), float("nan"))
    with pytest.raises(ValueError, match="Invalid MTP proposal"):
        batched_trunk_children([row], [torch.tensor(0)], depth=1, backup_budget=1, config=config)


def test_ties_below_min_draft_prob_do_not_defer_and_match_reference():
    """Peaked rows whose tail underflows to equal values decide like the reference."""
    from remtp.dynamic_mtp_tree import select_dynamic_candidates
    from remtp.raven_fast_path import select_dynamic_candidates_fast
    config = _residual_hit_config("residual_hit_anchor", max_nodes=9, max_children_per_parent=3)
    data = torch.Generator().manual_seed(9)
    for case in range(60):
        row = torch.zeros(500)
        heads = torch.rand(1 + case % 3, generator=data) + 0.05
        row[torch.randperm(500, generator=data)[: heads.numel()]] = heads
        row = row / row.sum()                                   # zeros tie in the top-k tail
        fast = select_dynamic_candidates_fast(row, depth=1 + case % 3, config=config)
        reference = select_dynamic_candidates(row, depth=1 + case % 3, config=config)
        assert not fast.used_reference
        assert torch.equal(fast.selection.token_ids, reference.token_ids)
        for field in ("entropy", "threshold", "q_max", "candidate_count_before_budget",
                      "threshold_candidate_count", "sibling_added"):
            assert getattr(fast.selection, field) == getattr(reference, field), field
