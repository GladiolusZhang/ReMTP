"""Caterpillar proposal pieces (2026-11-15): sibling selection, device state, wide verification."""
from __future__ import annotations

import pytest
import torch

from remtp import async_tree, caterpillar, device_verify, device_verify_fused
from remtp.dynamic_mtp_tree import DynamicTreeConfig, DynamicTreeTopology

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICE = torch.device("cuda")


def _config():
    return DynamicTreeConfig(max_depth=3, max_nodes=15, max_children_per_parent=3, min_sibling_ratio=0.01,
                             min_draft_prob=1e-6, threshold_sensitivity=1.5, support_mode="residual_hit_anchor",
                             allocation_mode="residual_coverage", cactus_delta=1.0,
                             relaxation_depth_weights=(1.0, 1.0, 1.0), adaptive_relaxation=True,
                             adaptive_relaxation_max_ratio=1.0, relaxation_min_target_probability=0.01)


def _reference_siblings(q, trunk, j, tau):
    order = sorted(range(q.numel()), key=lambda i: (-float(q[i]), i))
    picked = [i for i in order if i != trunk][:j]
    return picked, [float(q[i]) >= tau for i in picked]


@pytest.mark.parametrize("j", [1, 2, 4, 6])
def test_sibling_selection_matches_stable_reference(j):
    generator = torch.Generator(device=DEVICE).manual_seed(j)
    for case in range(12):
        vocab = 3000 if case % 2 else 151680
        logits = torch.randn(vocab, device=DEVICE, generator=generator) * (1 + case % 4)
        if case % 3 == 0:
            logits = logits.bfloat16().float()          # exact ties
        q = torch.softmax(logits, dim=-1)
        trunk = torch.multinomial(q, 1, generator=generator)
        tau = [0.0, 1e-6, 1e-3, 0.05][case % 4]
        sib, ok = caterpillar.siblings(q, trunk, j, tau)
        ids, flags = _reference_siblings(q.cpu(), int(trunk), j, tau)
        assert sib.tolist() == ids, (case, sib.tolist(), ids)
        assert ok.tolist() == flags


def _publish_case(generator, j, vocab):
    trunk = torch.randint(0, vocab, (3,), device=DEVICE, generator=generator)
    sib = torch.randint(0, vocab, (3, j), device=DEVICE, generator=generator)
    counts = torch.randint(0, j + 1, (3,), generator=torch.Generator().manual_seed(int(trunk[0])))
    valid = torch.arange(j, device=DEVICE)[None, :] < counts.to(DEVICE)[:, None]
    q_bank = torch.softmax(torch.randn(3, vocab, device=DEVICE, generator=generator), dim=-1)
    return trunk, sib, valid, q_bank, counts.tolist()


@pytest.mark.parametrize("j", [2, 4])
def test_publish_builds_the_caterpillar(j):
    generator = torch.Generator(device=DEVICE).manual_seed(30 + j)
    for _ in range(10):
        trunk, sib, valid, q_bank, counts = _publish_case(generator, j, 500)
        ids = caterpillar.publish(async_tree.STATE, trunk, sib, valid, q_bank, j)
        state = async_tree.STATE
        paths = ([(0,)] + [(r,) for r in range(1, counts[0] + 1)] + [(0, 0)]
                 + [(0, r) for r in range(1, counts[1] + 1)] + [(0, 0, 0)] + [(0, 0, r) for r in range(1, counts[2] + 1)])
        topology = DynamicTreeTopology.from_paths(tuple(paths))
        count = topology.node_count
        tokens = ([int(trunk[0])] + sib[0, :counts[0]].tolist() + [int(trunk[1])] + sib[1, :counts[1]].tolist()
                  + [int(trunk[2])] + sib[2, :counts[2]].tolist())
        assert ids[0, :count].tolist() == tokens
        assert state.valid.tolist() == [i < count for i in range(caterpillar.capacity(j))]
        assert state.depth[:count].tolist() == [n.depth for n in topology.nodes]
        assert state.parent[:count].tolist() == [-1 if n.parent is None else n.parent for n in topology.nodes]
        assert state.trunk.tolist() == [n.index for n in topology.nodes if n.path == (0,) * n.depth]
        for k, node in enumerate(state.trunk.tolist()):
            expected = [s for s in topology.children_of(topology.nodes[node].parent) if s != node]
            assert [s for s in state.siblings[k].tolist() if s >= 0] == expected
        assert torch.equal(state.probs[:count], q_bank[[n.depth - 1 for n in topology.nodes]])
        assert torch.equal(state.mask, async_tree.tree_mask(state.valid, state.parent))


def test_wide_fused_verification_equals_torch_async():
    config = _config()
    j = 4
    generator = torch.Generator(device=DEVICE).manual_seed(77)
    same, cases = 0, 60
    for case in range(cases):
        vocab = 40 if case % 2 else 3000
        trunk, sib, valid, q_bank, _ = _publish_case(generator, j, vocab)
        # make siblings plausible correction tokens
        caterpillar.publish(async_tree.STATE, trunk, sib, valid, q_bank, j)
        target = torch.softmax(torch.randn(caterpillar.capacity(j) + 1, vocab, device=DEVICE, generator=generator)
                               * (1 + case % 3), dim=-1)
        width = caterpillar.capacity(j) + 1
        seed = 900 + case
        reference = device_verify.verify_async(async_tree.STATE, target, width,
                                               torch.Generator(device=DEVICE).manual_seed(seed), config=config)
        ref = (async_tree.STATE.accepted.clone(), async_tree.STATE.moves.clone(), async_tree.STATE.leaf_row.clone())
        fused = device_verify_fused.verify_async_fused(async_tree.STATE, target, width,
                                                       torch.Generator(device=DEVICE).manual_seed(seed), config=config)
        same += int(torch.equal(fused, reference) and torch.equal(async_tree.STATE.accepted, ref[0])
                    and torch.equal(async_tree.STATE.moves, ref[1]) and torch.equal(async_tree.STATE.leaf_row, ref[2]))
    print(f"wide fused == torch async on {same}/{cases}")
    assert same >= cases - 2
