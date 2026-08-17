from __future__ import annotations

import torch

from remtp.fixed6_microtree import get_topology
from remtp.mimo_tree import verify_mimo_tree


def _one_hot(token: int, vocab: int = 8) -> torch.Tensor:
    value = torch.zeros(vocab)
    value[token] = 1.0
    return value


def test_strict_mimo_tree_is_existing_strict_verifier() -> None:
    topology = get_topology("3+2+1")
    ids = torch.tensor([1, 2, 3, 4, 5, 6])
    q = torch.stack([_one_hot(int(token)) for token in ids])
    p = torch.stack(
        [
            _one_hot(1),  # anchor accepts first root
            _one_hot(4),  # after root 1
            _one_hot(0),
            _one_hot(0),
            _one_hot(6),  # after node 3
            _one_hot(7),
            _one_hot(7),
        ]
    )
    result = verify_mimo_tree(topology, ids, q, p, mode="strict")
    assert result.accepted_node_indices == topology.branch_nodes[0]
    assert result.output_token_ids[:3] == (1, 4, 6)
    assert result.target_forward_calls == 1
    assert result.target_validation_nodes == 6


def test_cactus_path_never_relaxes_root_selection() -> None:
    topology = get_topology("3+2+1")
    ids = torch.tensor([1, 2, 3, 4, 5, 6])
    q = torch.stack([_one_hot(int(token)) for token in ids])
    p = torch.stack([_one_hot(0) for _ in range(7)])
    result = verify_mimo_tree(
        topology, ids, q, p, mode="cactus_path", cactus_delta=10.0
    )
    assert result.accepted_drafts == 0
    assert result.selected_branch is None


def test_cactus_path_can_rescue_a_selected_branch_descendant() -> None:
    topology = get_topology("3+2+1")
    ids = torch.tensor([1, 2, 3, 4, 5, 6])
    q = torch.stack([_one_hot(int(token)) for token in ids])
    p = torch.stack(
        [
            _one_hot(1),
            torch.tensor([0.5, 0, 0, 0, 0.5, 0, 0, 0], dtype=torch.float32),
            _one_hot(0),
            _one_hot(0),
            _one_hot(6),
            _one_hot(7),
            _one_hot(7),
        ]
    )
    generator = torch.Generator().manual_seed(0)
    result = verify_mimo_tree(
        topology,
        ids,
        q,
        p,
        generator,
        mode="cactus_path",
        cactus_delta=2.0,
    )
    assert result.accepted_node_indices == topology.branch_nodes[0]
