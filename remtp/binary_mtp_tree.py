"""Complete binary MTP tree and exact multi-candidate verification.

The default topology keeps two candidates at every parent for three draft
depths: 2 roots, 4 children and 8 grandchildren (14 target validation nodes).
Each sibling group is sampled without replacement and verified with recursive
residual distributions, so strict sampling still recovers target distribution
P despite having multiple candidates at every depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import torch

from remtp.fixed6_microtree import (
    TreeNode,
    TreeVerifyResult,
    _accept,
    normalize_distribution,
    residual_distribution,
    sample_categorical,
)


@dataclass(frozen=True)
class BinaryTreeTopology:
    name: str
    width: int
    depth: int
    nodes: tuple[TreeNode, ...]

    @classmethod
    def complete(cls, *, width: int = 2, depth: int = 3) -> "BinaryTreeTopology":
        if width < 2:
            raise ValueError("tree width must be at least two")
        if depth < 1:
            raise ValueError("tree depth must be positive")
        paths = [
            path
            for current_depth in range(1, depth + 1)
            for path in product(range(width), repeat=current_depth)
        ]
        path_to_index = {path: index for index, path in enumerate(paths)}
        nodes = tuple(
            TreeNode(
                index=index,
                branch=path[0],
                depth=len(path),
                path=tuple(path),
                parent=path_to_index.get(tuple(path[:-1])) if len(path) > 1 else None,
            )
            for index, path in enumerate(paths)
        )
        result = cls(name=f"{width}x{width}x{width}", width=width, depth=depth, nodes=nodes)
        result.validate()
        return result

    def validate(self) -> None:
        expected = sum(self.width**level for level in range(1, self.depth + 1))
        if len(self.nodes) != expected:
            raise ValueError(f"expected {expected} nodes, got {len(self.nodes)}")
        for node in self.nodes:
            if node.depth == 1 and node.parent is not None:
                raise ValueError("root cannot have a parent")
            if node.depth > 1:
                if node.parent is None:
                    raise ValueError("non-root node has no parent")
                if self.nodes[node.parent].path != node.path[:-1]:
                    raise ValueError("invalid parent/path relation")

    @property
    def choices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(node.path for node in self.nodes)

    @property
    def roots(self) -> tuple[int, ...]:
        return self.level_nodes(1)

    @property
    def leaves(self) -> tuple[int, ...]:
        return self.level_nodes(self.depth)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def max_depth(self) -> int:
        return self.depth

    @property
    def node_position_offsets(self) -> tuple[int, ...]:
        return tuple(node.depth for node in self.nodes)

    @property
    def parent_logit_rows(self) -> tuple[int, ...]:
        return tuple(0 if node.parent is None else node.parent + 1 for node in self.nodes)

    @property
    def leaf_logit_rows(self) -> tuple[int, ...]:
        return tuple(index + 1 for index in self.leaves)

    def level_nodes(self, depth: int) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes if node.depth == depth)

    def children_of(self, parent: int | None) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes if node.parent == parent)

    def path_to(self, node_index: int) -> tuple[int, ...]:
        path: list[int] = []
        current: int | None = int(node_index)
        while current is not None:
            path.append(current)
            current = self.nodes[current].parent
        return tuple(reversed(path))

    def causal_tree_mask(self, *, include_anchor: bool = True) -> torch.Tensor:
        offset = 1 if include_anchor else 0
        size = self.node_count + offset
        mask = torch.zeros((size, size), dtype=torch.bool)
        if include_anchor:
            mask[:, 0] = True
        for node in self.nodes:
            row = node.index + offset
            mask[row, row] = True
            parent = node.parent
            while parent is not None:
                mask[row, parent + offset] = True
                parent = self.nodes[parent].parent
        return mask


FULL_BINARY_2X2X2 = BinaryTreeTopology.complete(width=2, depth=3)


def get_mimo_tree_topology(name: str):
    if name in {"2x2x2", "binary-3", "2+4+8"}:
        return FULL_BINARY_2X2X2
    from remtp.fixed6_microtree import get_topology

    return get_topology(name)


def _children(topology, parent: int | None) -> tuple[int, ...]:
    if hasattr(topology, "children_of"):
        return tuple(topology.children_of(parent))
    if parent is None:
        return tuple(topology.roots)
    return tuple(node.index for node in topology.nodes if node.parent == parent)


def strict_general_tree_verify(
    topology,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    generator: torch.Generator | None = None,
) -> TreeVerifyResult:
    """Lossless strict verification for branching at arbitrary depths."""
    count = len(topology.nodes)
    if draft_token_ids.shape != (count,):
        raise ValueError(f"draft_token_ids must have shape [{count}]")
    if draft_probs.ndim != 2 or draft_probs.shape[0] != count:
        raise ValueError(f"draft_probs must have shape [{count}, vocab]")
    if target_probs_by_input.shape != (count + 1, draft_probs.shape[1]):
        raise ValueError(f"target_probs_by_input must have shape [{count + 1}, vocab]")

    accepted: list[int] = []
    output: list[int] = []
    statuses: list[tuple[int, str]] = []
    parent: int | None = None
    verifier = normalize_distribution(target_probs_by_input[0])

    while True:
        siblings = _children(topology, parent)
        accepted_child: int | None = None
        for node_index in siblings:
            token = int(draft_token_ids[node_index])
            proposal = normalize_distribution(draft_probs[node_index])
            if _accept(verifier, proposal, token, generator):
                statuses.append((node_index, "ACCEPT"))
                accepted.append(node_index)
                output.append(token)
                accepted_child = node_index
                break
            statuses.append((node_index, "REJECT"))
            verifier = residual_distribution(verifier, proposal)

        if accepted_child is None:
            correction = sample_categorical(verifier, generator)
            output.append(correction)
            return TreeVerifyResult(
                output_token_ids=tuple(output),
                accepted_node_indices=tuple(accepted),
                selected_branch=(topology.nodes[accepted[0]].branch if accepted else None),
                terminal="correction",
                target_validation_nodes=count,
                evaluated_node_statuses=tuple(statuses),
            )

        children = _children(topology, accepted_child)
        if not children:
            bonus = sample_categorical(
                target_probs_by_input[accepted_child + 1], generator
            )
            output.append(bonus)
            return TreeVerifyResult(
                output_token_ids=tuple(output),
                accepted_node_indices=tuple(accepted),
                selected_branch=topology.nodes[accepted[0]].branch,
                terminal="bonus",
                target_validation_nodes=count,
                evaluated_node_statuses=tuple(statuses),
            )
        parent = accepted_child
        verifier = normalize_distribution(target_probs_by_input[parent + 1])
