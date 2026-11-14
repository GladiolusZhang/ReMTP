"""The sync-reduced RAVEN helpers must reproduce the reference bit for bit."""
from __future__ import annotations

import math

import pytest
import torch

from remtp.dynamic_mtp_tree import (
    DynamicTreeConfig,
    DynamicTreeTopology,
    residual_hit_tree_verify,
    select_dynamic_candidates,
)
from remtp.fixed6_microtree import normalize_distribution, residual_distribution
from remtp.raven_fast_path import (
    cactus_trunk_children_fast,
    normalize_rows_nosync,
    residual_distribution_nosync,
    residual_hit_tree_verify_fast,
    select_dynamic_candidates_fast,
)
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "_tree_tests", pathlib.Path(__file__).with_name("test_dynamic_mtp_tree.py"))
_tree_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tree_tests)
_exact_residual_hit_fixture = _tree_tests._exact_residual_hit_fixture
_residual_hit_config = _tree_tests._residual_hit_config


def _assert_same_selection(actual, expected) -> None:
    assert torch.equal(actual.token_ids, expected.token_ids)
    for field in ("entropy", "threshold", "q_max", "candidate_count_before_budget",
                  "threshold_candidate_count", "sibling_added"):
        assert getattr(actual, field) == getattr(expected, field), field


def _random_rows(generator: torch.Generator, rows: int, vocab: int, sharp: float) -> torch.Tensor:
    logits = torch.randn(rows, vocab, generator=generator) * sharp
    return torch.softmax(logits, dim=-1)


def test_row_normalisation_matches_reference_bitwise() -> None:
    generator = torch.Generator().manual_seed(1)
    rows = _random_rows(generator, 6, 4096, 3.0)
    rows[1] = 0.0                     # degenerate row -> uniform
    rows[2, :10] = -1.0               # negatives are clamped
    rows[3] *= 1e-32                  # below the mass floor -> uniform
    expected = torch.stack([normalize_distribution(row) for row in rows])
    assert torch.equal(normalize_rows_nosync(rows), expected)
    assert torch.equal(normalize_rows_nosync(rows[0]), normalize_distribution(rows[0]))


def test_residual_matches_reference_bitwise() -> None:
    generator = torch.Generator().manual_seed(2)
    target, draft = _random_rows(generator, 2, 2048, 4.0)
    assert torch.equal(residual_distribution_nosync(target, draft), residual_distribution(target, draft))
    # Identical rows -> zero residual -> falls back to the normalised target.
    assert torch.equal(residual_distribution_nosync(target, target), residual_distribution(target, target))


@pytest.mark.parametrize("sharp", [0.5, 2.0, 6.0, 12.0])
@pytest.mark.parametrize("ratio", [0.01, 0.25])
def test_candidate_selection_matches_reference(sharp: float, ratio: float) -> None:
    generator = torch.Generator().manual_seed(int(sharp * 10) + int(ratio * 100))
    config = DynamicTreeConfig(
        max_depth=3, max_nodes=9, max_children_per_parent=3, min_sibling_ratio=ratio,
        min_draft_prob=0.02, support_mode="residual_hit_anchor",
    )
    for case in range(40):
        probs = _random_rows(generator, 1, 3000, sharp)[0]
        depth = 1 + case % 3
        budget = None if case % 4 else 2
        reference = select_dynamic_candidates(probs, depth=depth, config=config, budget=budget)
        fast = select_dynamic_candidates_fast(probs, depth=depth, config=config, budget=budget)
        assert torch.equal(fast.selection.token_ids, reference.token_ids), case
        for field in ("entropy", "threshold", "q_max", "candidate_count_before_budget",
                      "threshold_candidate_count", "sibling_added"):
            assert getattr(fast.selection, field) == getattr(reference, field), (case, field)
        assert fast.token_ids == reference.token_ids.tolist()
        assert fast.raw_probs == [float(probs[t]) for t in reference.token_ids.tolist()]
        assert fast.valid


def test_candidate_selection_defers_to_reference_on_ties() -> None:
    config = DynamicTreeConfig(max_depth=3, max_nodes=9, max_children_per_parent=3,
                               min_sibling_ratio=0.01, support_mode="residual_hit_anchor")
    # Ties below min_draft_prob cannot change the decision (2026-11-13): no deferral,
    # identical selection.
    probs = torch.full((500,), 1.0 / 500)
    probs[7] = 0.3
    probs = probs / probs.sum()
    reference = select_dynamic_candidates(probs, depth=1, config=config)
    fast = select_dynamic_candidates_fast(probs, depth=1, config=config)
    assert not fast.used_reference
    _assert_same_selection(fast.selection, reference)
    # A tie among choosable values still defers to the stable full sort.
    probs = torch.full((500,), 1.0 / 500)
    probs[7] = 0.3
    probs[11] = probs[13] = 0.1
    probs = probs / probs.sum()
    reference = select_dynamic_candidates(probs, depth=1, config=config)
    fast = select_dynamic_candidates_fast(probs, depth=1, config=config)
    assert fast.used_reference
    _assert_same_selection(fast.selection, reference)


def test_trunk_children_match_reference() -> None:
    from remtp.dynamic_tree_vllm import _cactus_trunk_children

    generator = torch.Generator().manual_seed(5)
    config = DynamicTreeConfig(max_depth=3, max_nodes=9, max_children_per_parent=3,
                               min_sibling_ratio=0.01, support_mode="residual_hit_anchor")
    for case in range(30):
        probs = _random_rows(generator, 1, 2000, 4.0)[0]
        trunk = (probs / torch.empty_like(probs).exponential_(generator=generator)).argmax().reshape(1)[0]
        for budget in (0, 1, 2):
            expected_ids, expected_selection = _cactus_trunk_children(
                probs, trunk, depth=1 + case % 3, backup_budget=budget, config=config)
            ids, selection = cactus_trunk_children_fast(
                probs, trunk, depth=1 + case % 3, backup_budget=budget, config=config)
            assert torch.equal(ids, expected_ids), (case, budget)
            _assert_same_selection(selection, expected_selection)


def test_trunk_children_reject_invalid_proposal() -> None:
    config = DynamicTreeConfig(max_depth=3, max_nodes=9, support_mode="residual_hit_anchor")
    probs = torch.zeros(100)
    with pytest.raises(ValueError):
        cactus_trunk_children_fast(probs, torch.tensor(3), depth=1, backup_budget=2, config=config)


def _random_tree_case(generator: torch.Generator, vocab: int):
    paths = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)]
    topology = DynamicTreeTopology.from_paths(paths)
    count = topology.node_count
    draft = _random_rows(generator, count, vocab, 3.0)
    target = _random_rows(generator, count + 1, vocab, 3.0)
    # Draft tokens are sampled from the draft rows so acceptance is non-trivial;
    # make some siblings coincide with likely corrections.
    ids = torch.multinomial(draft, 1, generator=generator).flatten()
    for sibling, source in ((1, 1), (2, 0), (4, 0), (5, 1), (7, 0), (8, 0)):
        if int(torch.rand((), generator=generator)) == 0:
            ids[sibling] = torch.multinomial(target[source], 1, generator=generator)[0]
    return topology, ids, draft, target


@pytest.mark.parametrize("mode", ["sampled_primary_shadow", "residual_hit_anchor",
                                  "residual_hit_strict", "residual_hit_cactus"])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_fast_verifier_matches_reference(mode: str, adaptive: bool, delta: float) -> None:
    config = _residual_hit_config(
        mode, max_nodes=9, max_children_per_parent=3, cactus_delta=delta,
        adaptive_relaxation=adaptive, adaptive_relaxation_mix=0.5,
    )
    data = torch.Generator().manual_seed(11)
    terminals = set()
    for case in range(60):
        topology, ids, draft, target = _random_tree_case(data, 512)
        protected = frozenset({0}) if case % 7 == 0 else frozenset()
        seed = 1000 + case
        reference = residual_hit_tree_verify(
            topology, ids, draft, target, torch.Generator().manual_seed(seed),
            config=config, protected_node_indices=protected,
        )
        fast_generator = torch.Generator().manual_seed(seed)
        reference_generator = torch.Generator().manual_seed(seed)
        residual_hit_tree_verify(topology, ids, draft, target, reference_generator,
                                 config=config, protected_node_indices=protected)
        fast = residual_hit_tree_verify_fast(
            topology, ids, draft, target, fast_generator,
            config=config, protected_node_indices=protected,
        )
        assert fast is not None
        assert fast == reference, case
        assert torch.equal(fast_generator.get_state(), reference_generator.get_state()), case
        terminals.add(fast.terminal)
    # The random cases must exercise more than one outcome in the reuse modes;
    # the shadow control almost always ends in its single correction terminal.
    assert len(terminals) >= (1 if mode == "sampled_primary_shadow" else 2), terminals


def test_fast_verifier_fixture_and_fallbacks() -> None:
    topology, ids, draft, target = _exact_residual_hit_fixture()
    for mode in ("sampled_primary_shadow", "residual_hit_anchor", "residual_hit_strict", "residual_hit_cactus"):
        config = _residual_hit_config(mode)
        reference = residual_hit_tree_verify(topology, ids, draft, target, torch.Generator().manual_seed(7), config=config)
        fast = residual_hit_tree_verify_fast(topology, ids, draft, target, torch.Generator().manual_seed(7), config=config)
        assert fast == reference
    unsupported = _residual_hit_config("residual_hit_anchor", protect_eos_relaxation=True, eos_token_ids=(4,))
    assert residual_hit_tree_verify_fast(topology, ids, draft, target, torch.Generator().manual_seed(7), config=unsupported) is None
