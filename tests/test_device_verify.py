"""Device residual-hit verification: same rule as the fast path, fixed randomness (2026-11-15)."""
from __future__ import annotations

import collections
import math

import pytest
import torch

from remtp import device_verify
from remtp import raven_fast_path as fast
from remtp.dynamic_mtp_tree import DynamicTreeConfig, DynamicTreeTopology, adaptive_trunk_relaxation_plan_host

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICE = torch.device("cuda")
PATHS = ((0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0))


def _config(**overrides):
    values = dict(max_depth=3, max_nodes=9, max_children_per_parent=3, min_sibling_ratio=0.01,
                  min_draft_prob=0.001, threshold_sensitivity=1.5, support_mode="residual_hit_anchor",
                  allocation_mode="residual_coverage", cactus_delta=1.0, relaxation_depth_weights=(1.0, 1.0, 1.0),
                  adaptive_relaxation=True, adaptive_relaxation_mix=0.5, adaptive_relaxation_max_ratio=1.0,
                  relaxation_min_target_probability=0.01)
    values.update(overrides)
    return DynamicTreeConfig(**values)


def _case(seed, vocab, paths=PATHS, sharp=2.0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    topology = DynamicTreeTopology.from_paths(paths)
    count = topology.node_count
    target = torch.softmax(torch.randn(count + 1, vocab, device=DEVICE, generator=generator) * sharp, dim=-1)
    draft = torch.softmax(torch.randn(count, vocab, device=DEVICE, generator=generator) * sharp, dim=-1)
    # Draft tokens: sampled from q at each node, siblings distinct.
    ids = torch.multinomial(draft, 1, generator=generator).flatten()
    for node in topology.nodes:
        siblings = [s for s in topology.children_of(node.parent) if s < node.index]
        while int(ids[node.index]) in [int(ids[s]) for s in siblings]:
            ids[node.index] = (ids[node.index] + 1) % vocab
    return topology, ids, draft, target


def test_deterministic_quantities_equal_the_fast_path():
    config = _config()
    for seed in range(40):
        topology, ids, draft, target = _case(seed, 3000, sharp=1.0 + seed % 4)
        generator = torch.Generator(device=DEVICE).manual_seed(1000 + seed)
        result = device_verify.residual_hit_tree_verify_device(topology, ids, draft, target, generator, config=config)
        assert result is not None
        normalized_t = fast.normalize_rows_nosync(target)
        normalized_q = fast.normalize_rows_nosync(draft)
        trunk = [n.index for n in topology.nodes if n.path == (0,) * n.depth]
        parent_rows = [0 if topology.nodes[n].parent is None else topology.nodes[n].parent + 1 for n in range(9)]
        p_list = normalized_t[parent_rows, ids].tolist()
        q_list = normalized_q[torch.arange(9, device=DEVICE), ids].tolist()
        maxes = normalized_t.max(dim=-1).values.tolist()
        plan = adaptive_trunk_relaxation_plan_host(topology, p_list, q_list, maxes, config=config)
        rows = {row["node"]: row for row in result.node_diagnostics}
        for node in trunk:
            if node not in rows:
                continue
            row = rows[node]
            assert row["target_probability"] == p_list[node]
            assert row["draft_probability"] == q_list[node]
            assert row["adaptive_relaxation_tv"] == float(plan[node]["adaptive_relaxation_tv"])
            parent = 0 if topology.nodes[node].parent is None else topology.nodes[node].parent + 1
            reference = fast._boost_known(normalized_t[parent], int(ids[node]),
                                          p_list[node] + float(plan[node]["adaptive_relaxation_tv"]), p_list[node])
            assert row["verification_probability"] == float(reference[int(ids[node])])


def test_decisions_follow_the_rule_given_the_draws():
    config = _config()
    terminals = collections.Counter()
    for seed in range(120):
        topology, ids, draft, target = _case(seed, 40, sharp=1.0 + seed % 3)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        result = device_verify.residual_hit_tree_verify_device(topology, ids, draft, target, generator, config=config)
        terminals[result.terminal] += 1
        rows = {row["node"]: row for row in result.node_diagnostics}
        trunk = [n.index for n in topology.nodes if n.path == (0,) * n.depth]
        visited = [n for n in trunk if n in rows]
        for node in visited[:-1]:
            assert rows[node]["accept_random"] <= rows[node]["accept_probability"]
        last = rows[visited[-1]]
        last_accepted = last["accept_random"] <= last["accept_probability"]
        assert last["trunk_accepted"] == last_accepted
        accepted_trunk = [n for n in visited if rows[n]["trunk_accepted"]]
        out = result.output_token_ids
        assert list(out[:len(accepted_trunk)]) == [int(ids[n]) for n in accepted_trunk]
        if result.terminal == "sampled-primary-full-cactus-bonus":
            assert len(accepted_trunk) == 3 and len(out) == 4
        else:
            correction = out[len(accepted_trunk)]
            parent = topology.nodes[visited[-1]].parent
            siblings = [s for s in topology.children_of(parent) if s != visited[-1]]
            hits = [s for s in siblings if int(ids[s]) == correction]
            if result.terminal == "residual-hit-target-anchor":
                assert hits and result.accepted_node_indices[-1] == hits[0] and len(out) == len(accepted_trunk) + 2
            else:
                assert result.terminal == "residual-correction-tree-miss" and not hits
                assert len(out) == len(accepted_trunk) + 1
            # The correction has positive mass under the exact residual (h - q)+.
            p_row = fast.normalize_rows_nosync(target)[0 if parent is None else parent + 1]
            assert correction != int(ids[visited[-1]]) or float(p_row[correction]) > 0.0
        assert result.accepted_node_indices == tuple(accepted_trunk) + (
            (result.accepted_node_indices[-1],) if result.terminal == "residual-hit-target-anchor" else ())
    assert set(terminals) >= {"sampled-primary-full-cactus-bonus", "residual-correction-tree-miss"}


def test_output_distribution_matches_the_fast_path():
    """First two output tokens: device rule vs per-node fast path over many seeds (small vocabulary)."""
    config = _config()
    topology, ids, draft, target = _case(3, 6, sharp=0.7)
    trials = 3000
    device_counts, host_counts = collections.Counter(), collections.Counter()
    for trial in range(trials):
        gen_d = torch.Generator(device=DEVICE).manual_seed(10_000 + trial)
        gen_h = torch.Generator(device=DEVICE).manual_seed(90_000 + trial)
        d = device_verify.residual_hit_tree_verify_device(topology, ids, draft, target, gen_d, config=config)
        h = fast.residual_hit_tree_verify_fast(topology, ids, draft, target, gen_h, config=config)
        device_counts[d.output_token_ids[:2]] += 1
        host_counts[h.output_token_ids[:2]] += 1
    keys = set(device_counts) | set(host_counts)
    tv = 0.5 * sum(abs(device_counts[k] - host_counts[k]) for k in keys) / trials
    # Two independent samples of the same distribution: expected TV ~ sqrt(K / trials).
    assert tv < 3.0 * math.sqrt(len(keys) / trials), (tv, device_counts, host_counts)


def test_unsupported_configurations_fall_back():
    topology, ids, draft, target = _case(0, 50)
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    for config in (_config(adaptive_relaxation_max_ratio=1.5), _config(support_mode="residual_hit_strict"),
                   _config(relaxation_risk_policy="target_odds")):
        assert device_verify.residual_hit_tree_verify_device(topology, ids, draft, target, generator,
                                                             config=config) is None
