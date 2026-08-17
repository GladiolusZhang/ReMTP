from __future__ import annotations

import torch

from remtp.binary_mtp_tree import FULL_BINARY_2X2X2, strict_general_tree_verify


def _one_hot(token: int, vocab: int = 32) -> torch.Tensor:
    row = torch.zeros(vocab)
    row[token] = 1.0
    return row


def test_binary_topology_has_two_four_eight_nodes() -> None:
    topology = FULL_BINARY_2X2X2
    assert topology.max_depth == 3
    assert topology.node_count == 14
    assert [len(topology.level_nodes(depth)) for depth in (1, 2, 3)] == [2, 4, 8]
    assert topology.choices[:6] == (
        (0,),
        (1,),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )


def test_binary_mask_isolates_siblings() -> None:
    topology = FULL_BINARY_2X2X2
    mask = topology.causal_tree_mask()
    node_000 = next(node.index for node in topology.nodes if node.path == (0, 0, 0))
    node_001 = next(node.index for node in topology.nodes if node.path == (0, 0, 1))
    node_010 = next(node.index for node in topology.nodes if node.path == (0, 1, 0))
    assert mask[node_000 + 1, node_000 + 1]
    assert not mask[node_000 + 1, node_001 + 1]
    assert not mask[node_000 + 1, node_010 + 1]


def test_strict_binary_verifier_selects_one_candidate_per_depth() -> None:
    topology = FULL_BINARY_2X2X2
    ids = torch.arange(1, 15)
    q = torch.stack([_one_hot(int(token)) for token in ids])
    target = torch.stack([_one_hot(0) for _ in range(15)])
    # Select root node 1, then its second child node 5, then first child node 12.
    target[0] = _one_hot(int(ids[1]))
    target[1 + 1] = _one_hot(int(ids[5]))
    target[5 + 1] = _one_hot(int(ids[12]))
    target[12 + 1] = _one_hot(31)
    result = strict_general_tree_verify(topology, ids, q, target)
    assert result.accepted_node_indices == (1, 5, 12)
    assert result.output_token_ids == (2, 6, 13, 31)
    assert dict(result.evaluated_node_statuses) == {
        0: "REJECT",
        1: "ACCEPT",
        4: "REJECT",
        5: "ACCEPT",
        12: "ACCEPT",
    }
    assert result.target_validation_nodes == 14
