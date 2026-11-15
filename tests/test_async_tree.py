"""Async RAVEN pieces: device tree state, target mask and host-free verification (2026-11-15)."""
from __future__ import annotations

import pytest
import torch

from remtp import async_tree, device_verify
from remtp.dynamic_mtp_tree import DynamicTreeConfig, DynamicTreeTopology

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICE = torch.device("cuda")
TOPOLOGIES = (
    ((0,), (0, 0), (0, 0, 0)),
    ((0,), (1,), (0, 0), (0, 1), (0, 0, 0)),
    ((0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0,), (1,), (0, 0), (1, 0), (1, 1), (0, 0, 0), (0, 0, 1), (0, 0, 2)),
    ((0,), (1,), (2,), (0, 0), (0, 1), (0, 2), (2, 0), (0, 0, 0), (0, 1, 0)),
)


def _config():
    return DynamicTreeConfig(max_depth=3, max_nodes=9, max_children_per_parent=3, min_sibling_ratio=0.01,
                             min_draft_prob=0.001, threshold_sensitivity=1.5, support_mode="residual_hit_anchor",
                             allocation_mode="residual_coverage", cactus_delta=1.0,
                             relaxation_depth_weights=(1.0, 1.0, 1.0), adaptive_relaxation=True,
                             adaptive_relaxation_max_ratio=1.0, relaxation_min_target_probability=0.01)


def _code(path):
    code = 0
    for value in path:
        code = code * 4 + value
    return code


def _publish(topology, vocab, generator):
    """Feed ``publish_tree`` what gpu_tree.propose_tree computes for this topology."""
    count = topology.node_count
    n1 = sum(1 for n in topology.nodes if n.depth == 1)
    n2 = sum(1 for n in topology.nodes if n.depth == 2)
    n3 = count - n1 - n2
    pad = async_tree.CAPACITY - count
    tokens = torch.randint(0, vocab, (count,), device=DEVICE, generator=generator)
    for node in topology.nodes:             # siblings carry distinct tokens, as proposals do
        while int(tokens[node.index]) in [int(tokens[s]) for s in topology.children_of(node.parent) if s < node.index]:
            tokens[node.index] = (tokens[node.index] + 1) % vocab
    probs = torch.softmax(torch.randn(count, vocab, device=DEVICE, generator=generator) * 2.0, dim=-1)
    codes = torch.tensor([_code(n.path) for n in topology.nodes] + [0] * pad, device=DEVICE)
    parents = torch.tensor([-1 if n.parent is None else n.parent for n in topology.nodes] + [0] * pad,
                           device=DEVICE)
    tokens_bfs = torch.cat((tokens, torch.zeros(pad, dtype=torch.int64, device=DEVICE)))
    probs_bfs = torch.cat((probs, torch.zeros(pad, vocab, device=DEVICE)))
    t = lambda x: torch.tensor(x, dtype=torch.int64, device=DEVICE)
    ids = async_tree.publish_tree(n1=t(n1), n2=t(n2), n3=t(n3), tokens_bfs=tokens_bfs, codes_bfs=codes,
                                  parent_all=parents, target_for_pack=lambda x: x, probs_bfs=probs_bfs)
    return tokens, probs, ids


def test_device_tree_state_and_mask_match_the_host_topology():
    from vllm.v1.attention.backends.tree_attn import _get_depth_counts, _prepare_tree_attn_bias

    generator = torch.Generator(device=DEVICE).manual_seed(1)
    for paths in TOPOLOGIES:
        topology = DynamicTreeTopology.from_paths(paths)
        count = topology.node_count
        tokens, _, ids = _publish(topology, 500, generator)
        state = async_tree.STATE
        assert ids.shape == (1, async_tree.CAPACITY)
        assert ids[0, :count].tolist() == tokens.tolist()
        assert state.valid.tolist() == [i < count for i in range(async_tree.CAPACITY)]
        assert state.depth[:count].tolist() == [n.depth for n in topology.nodes]
        assert state.parent[:count].tolist() == [-1 if n.parent is None else n.parent for n in topology.nodes]
        trunk = [n.index for n in topology.nodes if n.path == (0,) * n.depth]
        assert state.trunk.tolist() == trunk
        for k, node in enumerate(trunk):
            expected = [s for s in topology.children_of(topology.nodes[node].parent) if s != node]
            got = [s for s in state.siblings[k].tolist() if s >= 0]
            assert got == expected, (paths, k)
        mask = async_tree.tree_mask(state.valid, state.parent)
        choices = [tuple(p) for p in paths]
        reference = _prepare_tree_attn_bias(choices, _get_depth_counts(choices), dtype=torch.float32,
                                            device=DEVICE)
        assert torch.equal(mask[:count + 1, :count + 1], reference), paths
        for row in range(count + 1, async_tree.CAPACITY + 1):     # padded rows: anchor and self only
            allowed = (mask[row] == 0).nonzero().flatten().tolist()
            assert allowed == [0, row]
        assert torch.isinf(mask[:count + 1, count + 1:]).all()      # no real row sees a padded row


def test_async_verification_equals_device_verification():
    config = _config()
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    for case in range(60):
        paths = TOPOLOGIES[case % len(TOPOLOGIES)]
        topology = DynamicTreeTopology.from_paths(paths)
        count = topology.node_count
        vocab = 40 if case % 2 else 3000
        tokens, probs, _ = _publish(topology, vocab, generator)
        target = torch.softmax(torch.randn(count + 1, vocab, device=DEVICE, generator=generator) * 2.0, dim=-1)
        target_padded = torch.cat((target, torch.softmax(torch.randn(async_tree.CAPACITY - count, vocab,
                                                                      device=DEVICE, generator=generator), -1)))
        seed = 77 + case
        reference = device_verify.residual_hit_tree_verify_device(
            topology, tokens, probs, target, torch.Generator(device=DEVICE).manual_seed(seed), config=config,
            noise_rows=async_tree.CAPACITY)
        out = device_verify.verify_async(async_tree.STATE, target_padded, 10,
                                         torch.Generator(device=DEVICE).manual_seed(seed), config=config)
        row = out[0].tolist()
        produced = [x for x in row if x >= 0]
        assert produced == list(reference.output_token_ids), (case, row, reference)
        assert row[len(produced):] == [-1] * (10 - len(produced))
        moves = int(async_tree.STATE.moves)
        assert moves == len(reference.accepted_node_indices)
        assert async_tree.STATE.accepted[:moves].tolist() == list(reference.accepted_node_indices)
        leaf = 0 if not moves else reference.accepted_node_indices[-1] + 1
        assert int(async_tree.STATE.leaf_row) == leaf


def test_async_commit_moves_the_first_count_pairs():
    from types import SimpleNamespace

    from remtp.raven_commit_kernel import LayerTable, attention_caches
    from remtp.raven_fast_commit import copy_cache_slots_inplace

    def modules(seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        return [SimpleNamespace(kv_cache=[torch.randn(2, 4, 16, 2, 64, device=DEVICE, generator=g).bfloat16()])
                for _ in range(3)]

    for moves in range(4):
        reference, candidate = modules(9), modules(9)
        sources = torch.tensor([11, 13, 17], device=DEVICE)
        destinations = torch.tensor([10, 11, 12], device=DEVICE)
        LayerTable(attention_caches(candidate)).move_counted(sources, destinations,
                                                             torch.tensor(moves, device=DEVICE))
        copy_cache_slots_inplace(reference, list(zip(sources.tolist(), destinations.tolist()))[:moves])
        for a, b in zip(reference, candidate):
            assert torch.equal(a.kv_cache[0], b.kv_cache[0]), moves


def test_fused_async_verification_equals_async_verification():
    from remtp import device_verify_fused

    config = _config()
    generator = torch.Generator(device=DEVICE).manual_seed(8)
    same = 0
    cases = 80
    for case in range(cases):
        paths = TOPOLOGIES[case % len(TOPOLOGIES)]
        topology = DynamicTreeTopology.from_paths(paths)
        count = topology.node_count
        vocab = 40 if case % 3 == 0 else (3000 if case % 3 == 1 else 151680)
        _publish(topology, vocab, generator)
        target = torch.softmax(torch.randn(async_tree.CAPACITY + 1, vocab, device=DEVICE, generator=generator)
                               * (1.0 + case % 4), dim=-1)
        seed = 300 + case
        reference = device_verify.verify_async(async_tree.STATE, target, 10,
                                               torch.Generator(device=DEVICE).manual_seed(seed), config=config)
        ref_state = (async_tree.STATE.accepted.clone(), async_tree.STATE.moves.clone(),
                     async_tree.STATE.leaf_row.clone())
        fused = device_verify_fused.verify_async_fused(async_tree.STATE, target, 10,
                                                       torch.Generator(device=DEVICE).manual_seed(seed), config=config)
        match = (torch.equal(fused, reference) and torch.equal(async_tree.STATE.accepted, ref_state[0])
                 and torch.equal(async_tree.STATE.moves, ref_state[1])
                 and torch.equal(async_tree.STATE.leaf_row, ref_state[2]))
        same += int(match)
        produced = [x for x in fused[0].tolist() if x >= 0]
        assert 1 <= len(produced) <= 4 and fused[0, len(produced):].eq(-1).all()
    # Row sums are reduced in another order (last-bit differences); decisions agree except at a cut.
    print(f"fused == torch async verification on {same}/{cases} cases")
    assert same >= cases - 2, same


def test_publish_kernel_equals_reference_publish():
    generator = torch.Generator(device=DEVICE).manual_seed(12)
    vocab = 700
    for paths in TOPOLOGIES:
        topology = DynamicTreeTopology.from_paths(paths)
        nodes = topology.nodes
        d1 = [n for n in nodes if n.depth == 1]
        d2 = [n for n in nodes if n.depth == 2]
        d3 = [n for n in nodes if n.depth == 3]
        tokens = torch.randint(0, vocab, (len(nodes),), device=DEVICE, generator=generator)
        q_bank = torch.softmax(torch.randn(8, vocab, device=DEVICE, generator=generator), dim=-1)
        pad = lambda values, size: torch.tensor(values + [0] * (size - len(values)), dtype=torch.int64, device=DEVICE)
        d2_slot = {n.path: k for k, n in enumerate(d2)}
        ret2 = dict(token=pad([int(tokens[n.index]) for n in d2], 4), code=pad([_code(n.path) for n in d2], 4),
                    parent=pad([n.path[0] for n in d2], 4))
        ret3 = dict(token=pad([int(tokens[n.index]) for n in d3], 7), code=pad([_code(n.path) for n in d3], 7),
                    parent=pad([d2_slot[n.path[:2]] for n in d3], 7))
        tok1 = pad([int(tokens[n.index]) for n in d1], 3)
        t = lambda x: torch.tensor(x, dtype=torch.int64, device=DEVICE)
        ids = async_tree.publish_levels(n1=t(len(d1)), n2=t(len(d2)), n3=t(len(d3)), tok1=tok1, ret2=ret2, ret3=ret3,
                                        q_bank=q_bank)
        state = async_tree.STATE
        count = len(nodes)
        assert ids[0, :count].tolist() == tokens.tolist()
        assert state.valid.tolist() == [i < count for i in range(9)]
        assert state.depth[:count].tolist() == [n.depth for n in nodes]
        assert state.parent[:count].tolist() == [-1 if n.parent is None else n.parent for n in nodes]
        assert state.trunk.tolist() == [n.index for n in nodes if n.path == (0,) * n.depth]
        for k, node in enumerate(state.trunk.tolist()):
            expected = [s for s in topology.children_of(nodes[node].parent) if s != node]
            assert [s for s in state.siblings[k].tolist() if s >= 0] == expected, (paths, k)
        bank = [0 if n.depth == 1 else (1 + n.path[0] if n.depth == 2 else 4 + d2_slot[n.path[:2]]) for n in nodes]
        assert torch.equal(state.probs[:count], q_bank[bank])
        assert torch.equal(state.mask, async_tree.tree_mask(state.valid, state.parent)), paths
