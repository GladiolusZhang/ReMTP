"""REMTP_DYNAMIC_TREE_ASYNC_HOST must not change any value or RNG draw (2026-11-14)."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

from remtp import raven_fast_path as fast
from remtp.dynamic_mtp_tree import DynamicTreeConfig, DynamicTreeTopology

_spec = importlib.util.spec_from_file_location(
    "_tree_tests", pathlib.Path(__file__).with_name("test_dynamic_mtp_tree.py"))
_tree_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tree_tests)
_residual_hit_config = _tree_tests._residual_hit_config

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _both(monkeypatch, fn):
    monkeypatch.setattr(fast, "ASYNC_HOST", False)
    reference = fn()
    monkeypatch.setattr(fast, "ASYNC_HOST", True)
    return reference, fn()


@pytest.mark.parametrize("device", DEVICES)
def test_to_device_and_index_one(device: str, monkeypatch) -> None:
    values = torch.randn(1000, device=device)
    index = torch.tensor(417, device=device)
    for flag in (False, True):
        monkeypatch.setattr(fast, "ASYNC_HOST", flag)
        moved = fast.to_device([3, 1, 4, 1, 5], torch.int64, device)
        assert moved.device.type == torch.device(device).type
        assert moved.tolist() == [3, 1, 4, 1, 5]
        assert torch.equal(fast.index_one(values, index), values[417].reshape(1))


@pytest.mark.parametrize("device", DEVICES)
def test_boost_known_identical(device: str, monkeypatch) -> None:
    generator = torch.Generator(device=device).manual_seed(4)
    row = torch.softmax(torch.randn(151680, device=device, generator=generator) * 4, -1)
    original = float(row[777])
    reference, fast_row = _both(monkeypatch, lambda: fast._boost_known(row.clone(), 777, original + 0.13, original))
    assert torch.equal(reference, fast_row)


def _rows(generator, device, count, vocab=4096, sharp=4.0):
    return torch.softmax(torch.randn(count, vocab, device=device, generator=generator) * sharp, -1)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_children_identical(device: str, monkeypatch) -> None:
    config = DynamicTreeConfig(max_depth=3, max_nodes=9, max_children_per_parent=3, min_sibling_ratio=0.01,
                               min_draft_prob=0.001, support_mode="residual_hit_anchor")
    generator = torch.Generator(device=device).manual_seed(9)
    for case in range(20):
        rows = list(_rows(generator, device, 3, sharp=2.0 + case % 5))
        trunks = [torch.multinomial(r, 1, generator=generator)[0] for r in rows]

        def run():
            out = fast.batched_trunk_children(rows, trunks, depth=1 + case % 3, backup_budget=2, config=config)
            fast.resolve_batched_counts(out)
            return [(item.candidate_ids.tolist(), item.candidate_probs, item.entropy, item.threshold, item.q_max,
                     item.sibling_added, item.threshold_candidate_count, item.candidate_count_before_budget)
                    for item in out]

        reference, fast_out = _both(monkeypatch, run)
        assert reference == fast_out, case


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("adaptive", [False, True])
def test_fast_verify_identical(device: str, adaptive: bool, monkeypatch) -> None:
    config = _residual_hit_config("residual_hit_anchor", max_nodes=9, max_children_per_parent=3, cactus_delta=1.0,
                                  adaptive_relaxation=adaptive, adaptive_relaxation_mix=0.5)
    paths = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)]
    topology = DynamicTreeTopology.from_paths(paths)
    data = torch.Generator(device=device).manual_seed(11)
    for case in range(25):
        draft = _rows(data, device, 9, 2048, 3.0)
        target = _rows(data, device, 10, 2048, 3.0)
        ids = torch.multinomial(draft, 1, generator=data).flatten()

        def run():
            generator = torch.Generator(device=device).manual_seed(500 + case)
            result = fast.residual_hit_tree_verify_fast(topology, ids, draft, target, generator, config=config)
            return result, generator.get_state()

        (reference, state_a), (fast_result, state_b) = _both(monkeypatch, run)
        assert reference == fast_result, case
        assert torch.equal(state_a, state_b), case


@pytest.mark.skipif(not torch.cuda.is_available(), reason="prelaunch is a CUDA path")
@pytest.mark.parametrize("gate", [0.0, 0.01])
@pytest.mark.parametrize("mode", ["residual_hit_anchor", "residual_hit_strict", "residual_hit_cactus"])
def test_trunk_prelaunch_identical_results_and_rng(gate: float, mode: str, monkeypatch) -> None:
    config = _residual_hit_config(mode, max_nodes=9, max_children_per_parent=3, cactus_delta=1.0,
                                  adaptive_relaxation=True, adaptive_relaxation_mix=0.5,
                                  adaptive_relaxation_max_ratio=1.0, relaxation_min_target_probability=gate)
    paths = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)]
    topology = DynamicTreeTopology.from_paths(paths)
    data = torch.Generator(device="cuda").manual_seed(13)
    terminals = set()
    for case in range(60):
        sharp = 1.0 + (case % 6)
        draft = _rows(data, "cuda", 9, 2048, sharp)
        target = 0.6 * _rows(data, "cuda", 10, 2048, sharp)
        target[1:] += 0.4 * draft                      # correlated, so trunk nodes are often accepted
        target = target / target.sum(-1, keepdim=True)
        ids = torch.multinomial(draft, 1, generator=data).flatten()

        def run(flag: bool):
            monkeypatch.setattr(fast, "VERIFY_PRELAUNCH", flag)
            generator = torch.Generator(device="cuda").manual_seed(900 + case)
            generator.set_offset(4 * (case % 7))
            result = fast.residual_hit_tree_verify_fast(topology, ids, draft, target, generator, config=config)
            return result, generator.get_state()

        before = dict(fast.PRELAUNCH_STATS)
        (reference, state_a), (pre, state_b) = run(False), run(True)
        assert fast.PRELAUNCH_STATS["used"] == before["used"] + 1, case
        assert fast.PRELAUNCH_STATS["fallback"] == before["fallback"], case
        assert pre == reference, case
        assert torch.equal(state_a, state_b), case
        terminals.add(reference.terminal)
    assert len(terminals) >= 2, terminals
