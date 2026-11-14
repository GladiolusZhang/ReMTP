"""RAVEN v4: target-anchored relaxation (2026-11-13).

The target model decides where relaxation is allowed (p(y) above a support
floor); the static per-position allowance is Cactus's sqrt(2 delta p (1-p)),
itself a function of the target probability only.  Two variants:

* safe  (max_ratio = 1, nominal pool): every acceptance probability is at most
  Cactus's on the same drafted token, and exact below the floor;
* gated (gated pool): the static control's realised TV at gated positions may
  move to supported positions, so realised block TV never exceeds Cactus's.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest
import torch

from remtp.dynamic_mtp_tree import (
    DynamicTreeConfig,
    DynamicTreeTopology,
    adaptive_trunk_relaxation_plan,
    residual_hit_tree_verify,
)
from remtp.raven_fast_path import residual_hit_tree_verify_fast

_spec = importlib.util.spec_from_file_location(
    "_tree_tests", pathlib.Path(__file__).with_name("test_dynamic_mtp_tree.py"))
_tree_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tree_tests)
_residual_hit_config = _tree_tests._residual_hit_config

FLOOR = 0.01
PATHS = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)]
TRUNK = {(0,): 0, (0, 0): 3, (0, 0, 0): 6}


def _cactus_accept(p: float, q: float, delta: float = 1.0) -> float:
    gamma = min(p + math.sqrt(max(2.0 * delta * p * (1.0 - p), 0.0)), 1.0)
    return min(1.0, gamma / max(q, 1e-30))


def _cactus_realised_tv(p: float, q: float, delta: float = 1.0) -> float:
    nominal = min(math.sqrt(max(2.0 * delta * p * (1.0 - p), 0.0)), 1.0 - p)
    return min(nominal, max(q - p, 0.0))


def _config(variant: str, **extra) -> DynamicTreeConfig:
    common = dict(max_nodes=9, max_children_per_parent=3, cactus_delta=1.0, adaptive_relaxation=True)
    if variant == "safe":
        common.update(adaptive_relaxation_mix=0.5, adaptive_relaxation_max_ratio=1.0,
                      relaxation_min_target_probability=FLOOR)
    elif variant == "gated":
        common.update(adaptive_relaxation_mix=1.0, adaptive_relaxation_max_ratio=1.5,
                      relaxation_min_target_probability=FLOOR, adaptive_relaxation_pool="gated")
    elif variant == "v3":
        common.update(adaptive_relaxation_mix=0.5, adaptive_relaxation_max_ratio=1.5)
    common.update(extra)
    return _residual_hit_config("residual_hit_anchor", **common)


def _case(generator: torch.Generator, vocab: int = 64):
    """Random tree whose trunk tokens span supported and unsupported targets."""
    topology = DynamicTreeTopology.from_paths(PATHS)
    count = topology.node_count
    target = torch.softmax(torch.randn(count + 1, vocab, generator=generator) * 4.0, dim=-1)
    noise = torch.softmax(torch.randn(count, vocab, generator=generator) * 2.5, dim=-1)
    parent_rows = torch.tensor([0 if node.parent is None else node.parent + 1 for node in topology.nodes])
    # An imperfect draft: part target, part unrelated noise.
    draft = 0.6 * target[parent_rows] + 0.4 * noise
    ids = torch.multinomial(draft, 1, generator=generator).flatten()
    return topology, ids, draft, target


def _trunk_pq(topology, ids, draft, target):
    rows = []
    for path, index in TRUNK.items():
        node = topology.nodes[index]
        parent_row = 0 if node.parent is None else node.parent + 1
        token = int(ids[index])
        rows.append((index, float(target[parent_row, token]), float(draft[index, token])))
    return rows


@pytest.mark.parametrize("field", ["relaxation_min_target_probability", "relaxation_min_target_ratio"])
@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan")])
def test_invalid_gate_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        DynamicTreeConfig(adaptive_relaxation=True, **{field: value}).validate()


def test_gate_and_pool_require_adaptive_planner():
    with pytest.raises(ValueError, match="require adaptive relaxation"):
        DynamicTreeConfig(relaxation_min_target_probability=0.01).validate()
    with pytest.raises(ValueError, match="require adaptive relaxation"):
        DynamicTreeConfig(adaptive_relaxation_pool="gated").validate()
    with pytest.raises(ValueError, match="adaptive_relaxation_pool"):
        DynamicTreeConfig(adaptive_relaxation=True, adaptive_relaxation_pool="other").validate()


def test_defaults_leave_the_historical_plan_unchanged():
    data = torch.Generator().manual_seed(5)
    for _ in range(40):
        topology, ids, draft, target = _case(data)
        old = adaptive_trunk_relaxation_plan(topology, ids, draft, target, config=_config("v3"))
        new = adaptive_trunk_relaxation_plan(
            topology, ids, draft, target,
            config=_config("v3", relaxation_min_target_probability=0.0, adaptive_relaxation_pool="nominal"))
        assert old == new


@pytest.mark.parametrize("variant", ["safe", "gated"])
def test_gate_is_decided_by_the_target_alone(variant):
    """Changing the draft distribution never changes which positions are gated."""
    data = torch.Generator().manual_seed(17)
    config = _config(variant)
    for _ in range(40):
        topology, ids, draft, target = _case(data)
        other = torch.softmax(torch.randn_like(draft) * 3.0, dim=-1)
        first = adaptive_trunk_relaxation_plan(topology, ids, draft, target, config=config)
        second = adaptive_trunk_relaxation_plan(topology, ids, other, target, config=config)
        for index, p, _q in _trunk_pq(topology, ids, draft, target):
            assert first[index]["adaptive_target_gated"] == second[index]["adaptive_target_gated"] == (p < FLOOR)


def test_safe_variant_never_accepts_more_than_cactus_and_is_exact_below_floor():
    data = torch.Generator().manual_seed(23)
    config = _config("safe")
    gated_seen = supported_seen = 0
    for case in range(300):
        topology, ids, draft, target = _case(data)
        result = residual_hit_tree_verify(topology, ids, draft, target,
                                          torch.Generator().manual_seed(case), config=config)
        for row in result.node_diagnostics:
            if row.get("verification_rule") != "cactus":
                continue
            p, q = row["target_probability"], row["draft_probability"]
            assert row["accept_probability"] <= _cactus_accept(p, q) + 1e-6
            if p < FLOOR:
                gated_seen += 1
                assert row["adaptive_relaxation_tv"] == 0.0
                assert row["accept_probability"] == pytest.approx(min(1.0, p / q), rel=1e-5, abs=1e-9)
            else:
                supported_seen += 1
    assert gated_seen > 50 and supported_seen > 50


def test_safe_variant_plan_bounded_by_cactus_on_every_trunk_position():
    data = torch.Generator().manual_seed(29)
    config = _config("safe")
    for _ in range(300):
        topology, ids, draft, target = _case(data)
        plan = adaptive_trunk_relaxation_plan(topology, ids, draft, target, config=config)
        for index, p, q in _trunk_pq(topology, ids, draft, target):
            tv = plan[index]["adaptive_relaxation_tv"]
            assert tv <= _cactus_realised_tv(p, q) + 1e-6
            if p < FLOOR:
                assert tv == 0.0


def test_gated_pool_never_exceeds_cactus_realised_block_tv():
    data = torch.Generator().manual_seed(31)
    config = _config("gated")
    moved = 0
    for _ in range(400):
        topology, ids, draft, target = _case(data)
        plan = adaptive_trunk_relaxation_plan(topology, ids, draft, target, config=config)
        trunk = _trunk_pq(topology, ids, draft, target)
        realised = sum(plan[index]["adaptive_relaxation_tv"] for index, _p, _q in trunk)
        cactus = sum(_cactus_realised_tv(p, q) for _index, p, q in trunk)
        assert realised <= cactus + 1e-6
        for index, p, q in trunk:
            tv = plan[index]["adaptive_relaxation_tv"]
            nominal = min(math.sqrt(2 * p * (1 - p)), 1 - p)
            assert tv <= min(1.5 * nominal, max(q - p, 0.0)) + 1e-6
            if p < FLOOR:
                assert tv == 0.0
            elif tv > _cactus_realised_tv(p, q) + 1e-9:
                moved += 1
    assert moved > 0            # the pool really reaches supported positions


@pytest.mark.parametrize("variant", ["safe", "gated"])
def test_fast_verifier_matches_reference_with_gate(variant):
    config = _config(variant)
    data = torch.Generator().manual_seed(37)
    for case in range(80):
        topology, ids, draft, target = _case(data, vocab=256)
        reference_generator = torch.Generator().manual_seed(3000 + case)
        fast_generator = torch.Generator().manual_seed(3000 + case)
        reference = residual_hit_tree_verify(topology, ids, draft, target, reference_generator, config=config)
        fast = residual_hit_tree_verify_fast(topology, ids, draft, target, fast_generator, config=config)
        assert fast is not None
        assert fast == reference, case
        assert torch.equal(fast_generator.get_state(), reference_generator.get_state()), case
