"""residual_hit_anchor never reads nodes outside the trunk and the trunk's siblings (2026-11-15).

For random trees, everything about the other nodes (draft token, draft row,
target row) is replaced by random values; the verification result and the
generator state afterwards must not change.  Checked for the reference
verifier and the fast path, with and without the v4-safe gate.
"""
from __future__ import annotations

import random

import pytest
import torch

from remtp import raven_fast_path as fast
from remtp.dynamic_mtp_tree import DynamicTreeConfig, DynamicTreeTopology, residual_hit_tree_verify

TOPOLOGIES = (
    ((0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0,), (1,), (0, 0), (1, 0), (1, 1), (0, 0, 0), (1, 0, 0), (1, 1, 0)),
    ((0,), (1,), (2,), (0, 0), (1, 0), (2, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0)),
    ((0,), (1,), (2,), (0, 0), (0, 1), (0, 2), (2, 0), (0, 0, 0), (0, 1, 0)),
)


def _config(gate: bool):
    return DynamicTreeConfig(max_depth=3, max_nodes=9, max_children_per_parent=3, min_sibling_ratio=0.01,
                             min_draft_prob=0.001, threshold_sensitivity=1.5, support_mode="residual_hit_anchor",
                             allocation_mode="residual_coverage", cactus_delta=1.0,
                             relaxation_depth_weights=(1.0, 1.0, 1.0), adaptive_relaxation=True,
                             adaptive_relaxation_max_ratio=1.0,
                             relaxation_min_target_probability=0.01 if gate else 0.0)


def _useful(topology):
    """Trunk nodes and their siblings: paths (0,)*j + (r,)."""
    return {n.index for n in topology.nodes if all(v == 0 for v in n.path[:-1])}


def _case(rng, topology, vocab):
    count = topology.node_count
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    target = torch.softmax(torch.randn(count + 1, vocab, generator=g) * rng.uniform(0.5, 3.0), dim=-1)
    draft = torch.softmax(torch.randn(count, vocab, generator=g) * rng.uniform(0.5, 3.0), dim=-1)
    ids = torch.multinomial(draft, 1, generator=g).flatten()
    # Make hits likely: copy some target-favoured tokens into trunk siblings.
    for node in topology.nodes:
        if node.path[-1] != 0 and all(v == 0 for v in node.path[:-1]) and rng.random() < 0.6:
            parent_row = 0 if node.parent is None else node.parent + 1
            ids[node.index] = int(torch.multinomial(target[parent_row], 1, generator=g))
    for node in topology.nodes:          # siblings carry distinct tokens
        while int(ids[node.index]) in [int(ids[s]) for s in topology.children_of(node.parent) if s < node.index]:
            ids[node.index] = (ids[node.index] + 1) % vocab
    return ids, draft, target


def _scramble(rng, topology, ids, draft, target, vocab):
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    ids, draft, target = ids.clone(), draft.clone(), target.clone()
    for node in topology.nodes:
        if node.index in _useful(topology):
            continue
        ids[node.index] = rng.randrange(vocab)
        draft[node.index] = torch.softmax(torch.randn(vocab, generator=g) * 3, dim=-1)
        target[node.index + 1] = torch.softmax(torch.randn(vocab, generator=g) * 3, dim=-1)
    return ids, draft, target


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("verifier", ["reference", "fast"])
def test_nodes_outside_trunk_and_siblings_do_not_matter(gate, verifier):
    rng = random.Random(5 + gate)
    config = _config(gate)
    terminals = set()
    for case in range(160):
        topology = DynamicTreeTopology.from_paths(TOPOLOGIES[case % len(TOPOLOGIES)])
        vocab = 12 if case % 2 else 300
        ids, draft, target = _case(rng, topology, vocab)
        scrambled = _scramble(rng, topology, ids, draft, target, vocab)
        results, states = [], []
        for inputs in ((ids, draft, target), scrambled):
            generator = torch.Generator().manual_seed(1000 + case)
            if verifier == "reference":
                result = residual_hit_tree_verify(topology, *inputs, generator, config=config)
            else:
                result = fast.residual_hit_tree_verify_fast(topology, *inputs, generator, config=config)
            results.append(result)
            states.append(generator.get_state())
        a, b = results
        assert a.output_token_ids == b.output_token_ids, case
        assert a.accepted_node_indices == b.accepted_node_indices, case
        assert a.terminal == b.terminal, case
        assert torch.equal(states[0], states[1]), case
        useless = set(range(topology.node_count)) - _useful(topology)
        visited = {n for n, status in a.evaluated_node_statuses if status not in ("NOT_VISITED",)}
        assert not (visited & useless), (case, visited & useless)
        terminals.add(a.terminal)
    assert {"residual-hit-target-anchor", "residual-correction-tree-miss"} <= terminals
