"""Correctness primitives for the Fixed-6 native-MTP micro-tree.

The three supported layouts are forests of root alternatives whose branch
lengths sum to six.  Nodes are stored breadth-first for target TreeAttention,
while recurrent target/MTP state is advanced branch-major.  The explicit
permutations in :class:`Fixed6Topology` are the boundary between those two
layouts; treating breadth-first nodes as an ordinary token chain is invalid.

This module deliberately keeps the probability verifier and state-copy
invariants independent from vLLM.  The runtime adapter lives in
``remtp.fixed6_vllm`` and is allowed to call these tested primitives, but not
to redefine their semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch


FIXED_NODE_BUDGET = 6


@dataclass(frozen=True)
class TreeNode:
    """One draft node in breadth-first order."""

    index: int
    branch: int
    depth: int
    path: tuple[int, ...]
    parent: int | None


@dataclass(frozen=True)
class Fixed6Topology:
    """A static six-node root-branching topology."""

    name: str
    branch_lengths: tuple[int, ...]
    nodes: tuple[TreeNode, ...]

    @classmethod
    def from_branch_lengths(
        cls,
        name: str,
        branch_lengths: Sequence[int],
    ) -> "Fixed6Topology":
        lengths = tuple(int(value) for value in branch_lengths)
        if not lengths or any(value <= 0 for value in lengths):
            raise ValueError("all micro-tree branch lengths must be positive")
        if sum(lengths) != FIXED_NODE_BUDGET:
            raise ValueError(
                f"Fixed-6 topology must contain exactly 6 nodes, got {sum(lengths)}"
            )

        raw_paths: list[tuple[int, tuple[int, ...]]] = []
        for branch, length in enumerate(lengths):
            for depth in range(1, length + 1):
                raw_paths.append((branch, (branch,) + (0,) * (depth - 1)))
        raw_paths.sort(key=lambda item: (len(item[1]), item[1]))
        path_to_index = {path: index for index, (_, path) in enumerate(raw_paths)}
        nodes = []
        for index, (branch, path) in enumerate(raw_paths):
            parent_path = path[:-1]
            parent = path_to_index.get(parent_path) if parent_path else None
            nodes.append(
                TreeNode(
                    index=index,
                    branch=branch,
                    depth=len(path),
                    path=path,
                    parent=parent,
                )
            )
        topology = cls(name=name, branch_lengths=lengths, nodes=tuple(nodes))
        topology.validate()
        return topology

    def validate(self) -> None:
        if len(self.nodes) != FIXED_NODE_BUDGET:
            raise ValueError("Fixed-6 topology must have exactly six nodes")
        if tuple(node.index for node in self.nodes) != tuple(range(6)):
            raise ValueError("nodes must be contiguous breadth-first indices")
        if tuple(node.path for node in self.nodes) != tuple(
            sorted((node.path for node in self.nodes), key=lambda path: (len(path), path))
        ):
            raise ValueError("nodes must be breadth-first")
        for node in self.nodes:
            if node.depth == 1:
                if node.parent is not None:
                    raise ValueError("root alternatives cannot have a parent")
            else:
                if node.parent is None:
                    raise ValueError("non-root node is missing its parent")
                parent = self.nodes[node.parent]
                if parent.path != node.path[:-1]:
                    raise ValueError("node parent does not match its path")

    @property
    def choices(self) -> tuple[tuple[int, ...], ...]:
        """vLLM TreeAttention choices in required breadth-first order."""
        return tuple(node.path for node in self.nodes)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def max_depth(self) -> int:
        return max(node.depth for node in self.nodes)

    def level_nodes(self, depth: int) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes if node.depth == depth)

    def children_of(self, parent: int | None) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes if node.parent == parent)

    @property
    def roots(self) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes if node.parent is None)

    @property
    def leaves(self) -> tuple[int, ...]:
        parents = {node.parent for node in self.nodes if node.parent is not None}
        return tuple(node.index for node in self.nodes if node.index not in parents)

    @property
    def branch_nodes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                node.index
                for node in sorted(
                    (item for item in self.nodes if item.branch == branch),
                    key=lambda item: item.depth,
                )
            )
            for branch in range(len(self.branch_lengths))
        )

    @property
    def branch_major(self) -> tuple[int, ...]:
        return tuple(index for branch in self.branch_nodes for index in branch)

    @property
    def bfs_to_branch_major(self) -> tuple[int, ...]:
        """Indices that gather a BFS node tensor into branch-major order."""
        return self.branch_major

    @property
    def branch_major_to_bfs(self) -> tuple[int, ...]:
        inverse = [0] * FIXED_NODE_BUDGET
        for branch_major_index, bfs_index in enumerate(self.branch_major):
            inverse[bfs_index] = branch_major_index
        return tuple(inverse)

    @property
    def node_position_offsets(self) -> tuple[int, ...]:
        """RoPE offsets of BFS draft nodes relative to the common root."""
        return tuple(node.depth for node in self.nodes)

    @property
    def parent_logit_rows(self) -> tuple[int, ...]:
        """Target-logit row that scores each candidate.

        Target input row 0 is the common anchor. Draft node ``i`` is input row
        ``i + 1``. Root alternatives are scored by row 0 and every other node
        by its parent's input row.
        """
        return tuple(0 if node.parent is None else node.parent + 1 for node in self.nodes)

    @property
    def leaf_logit_rows(self) -> tuple[int, ...]:
        return tuple(index + 1 for index in self.leaves)

    def path_to(self, node_index: int) -> tuple[int, ...]:
        node = self.nodes[node_index]
        return self.branch_nodes[node.branch][: node.depth]

    def causal_tree_mask(self, *, include_anchor: bool = True) -> torch.Tensor:
        """Boolean attention mask; siblings and other branches stay isolated."""
        offset = 1 if include_anchor else 0
        size = FIXED_NODE_BUDGET + offset
        mask = torch.zeros((size, size), dtype=torch.bool)
        if include_anchor:
            mask[:, 0] = True
            mask[0, 0] = True
        for node in self.nodes:
            row = node.index + offset
            mask[row, row] = True
            parent = node.parent
            while parent is not None:
                mask[row, parent + offset] = True
                parent = self.nodes[parent].parent
        return mask

    def state_index_matrix(
        self,
        node_state_indices: Sequence[int],
        *,
        pad: int = -1,
    ) -> tuple[tuple[int, ...], ...]:
        """Map six BFS state slots to padded branch-major recurrent rows."""
        if len(node_state_indices) != FIXED_NODE_BUDGET:
            raise ValueError("expected one recurrent state index per tree node")
        width = max(self.branch_lengths)
        return tuple(
            tuple(node_state_indices[index] for index in branch)
            + (pad,) * (width - len(branch))
            for branch in self.branch_nodes
        )


TOPOLOGIES: Mapping[str, Fixed6Topology] = {
    "6-chain": Fixed6Topology.from_branch_lengths("6-chain", (6,)),
    "4+2": Fixed6Topology.from_branch_lengths("4+2", (4, 2)),
    "3+2+1": Fixed6Topology.from_branch_lengths("3+2+1", (3, 2, 1)),
}


def get_topology(name: str) -> Fixed6Topology:
    try:
        return TOPOLOGIES[name]
    except KeyError as exc:
        choices = ", ".join(TOPOLOGIES)
        raise ValueError(f"unknown Fixed-6 topology {name!r}; choose {choices}") from exc


def normalize_distribution(values: torch.Tensor) -> torch.Tensor:
    probs = values.to(torch.float32).clamp_min(0.0)
    total = probs.sum()
    if not torch.isfinite(total) or float(total) <= 1e-30:
        return torch.full_like(probs, 1.0 / probs.numel())
    return probs / total


def residual_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """Standard lossless speculative-decoding residual ``(p-q)+``."""
    residual = (target_probs.to(torch.float32) - draft_probs.to(torch.float32)).clamp_min(0.0)
    if float(residual.sum()) <= 1e-30:
        return normalize_distribution(target_probs)
    return residual / residual.sum()


def sample_categorical(
    probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> int:
    probs = normalize_distribution(probs)
    return int(torch.multinomial(probs, 1, generator=generator).item())


def sample_without_replacement(
    probs: torch.Tensor,
    count: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential categorical roots and their exact conditional Q rows."""
    if count <= 0 or count > probs.numel():
        raise ValueError("invalid without-replacement sample count")
    remaining = normalize_distribution(probs).clone()
    token_ids: list[int] = []
    proposal_rows: list[torch.Tensor] = []
    for _ in range(count):
        conditional = normalize_distribution(remaining)
        token_id = sample_categorical(conditional, generator)
        token_ids.append(token_id)
        proposal_rows.append(conditional)
        remaining[token_id] = 0.0
    return (
        torch.tensor(token_ids, dtype=torch.int64, device=probs.device),
        torch.stack(proposal_rows, dim=0),
    )


@dataclass(frozen=True)
class TreeVerifyResult:
    output_token_ids: tuple[int, ...]
    accepted_node_indices: tuple[int, ...]
    selected_branch: int | None
    terminal: str
    target_forward_calls: int = 1
    target_validation_nodes: int = FIXED_NODE_BUDGET
    evaluated_node_statuses: tuple[tuple[int, str], ...] = ()
    node_diagnostics: tuple[dict, ...] = ()
    surviving_paths: int = 0
    selected_path_score: float | None = None

    @property
    def accepted_drafts(self) -> int:
        return len(self.accepted_node_indices)

    @property
    def mean_acceptance_length_contribution(self) -> int:
        return len(self.output_token_ids)


def _accept(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    token_id: int,
    generator: torch.Generator | None,
) -> bool:
    p = float(target_probs[token_id])
    q = max(float(draft_probs[token_id]), 1e-30)
    alpha = min(1.0, p / q)
    draw = float(
        torch.rand((), device=target_probs.device, generator=generator)
    )
    return draw <= alpha


def strict_tree_verify(
    topology: Fixed6Topology,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    generator: torch.Generator | None = None,
) -> TreeVerifyResult:
    """Lossless multi-candidate verification for a root-branching tree.

    ``target_probs_by_input`` has seven rows: row 0 is the common target
    anchor, and row ``i+1`` is the distribution after draft node ``i``. Root
    alternatives are tried with recursive residual distributions. Once a root
    is accepted, its branch is verified with ordinary sequential rejection
    sampling. Exactly one correction or bonus token terminates the round.
    """
    if draft_token_ids.shape != (FIXED_NODE_BUDGET,):
        raise ValueError("draft_token_ids must have shape [6]")
    if draft_probs.ndim != 2 or draft_probs.shape[0] != FIXED_NODE_BUDGET:
        raise ValueError("draft_probs must have shape [6, vocab]")
    if target_probs_by_input.shape != (FIXED_NODE_BUDGET + 1, draft_probs.shape[1]):
        raise ValueError("target_probs_by_input must have shape [7, vocab]")

    root_target = normalize_distribution(target_probs_by_input[0])
    for root_index in topology.roots:
        root_token = int(draft_token_ids[root_index])
        root_q = normalize_distribution(draft_probs[root_index])
        if not _accept(root_target, root_q, root_token, generator):
            root_target = residual_distribution(root_target, root_q)
            continue

        branch = topology.nodes[root_index].branch
        accepted = [root_index]
        output = [root_token]
        for node_index in topology.branch_nodes[branch][1:]:
            node = topology.nodes[node_index]
            assert node.parent is not None
            target = normalize_distribution(target_probs_by_input[node.parent + 1])
            proposal = normalize_distribution(draft_probs[node_index])
            token = int(draft_token_ids[node_index])
            if not _accept(target, proposal, token, generator):
                correction = sample_categorical(
                    residual_distribution(target, proposal), generator
                )
                output.append(correction)
                return TreeVerifyResult(
                    output_token_ids=tuple(output),
                    accepted_node_indices=tuple(accepted),
                    selected_branch=branch,
                    terminal="correction",
                )
            accepted.append(node_index)
            output.append(token)

        leaf = accepted[-1]
        bonus = sample_categorical(target_probs_by_input[leaf + 1], generator)
        output.append(bonus)
        return TreeVerifyResult(
            output_token_ids=tuple(output),
            accepted_node_indices=tuple(accepted),
            selected_branch=branch,
            terminal="bonus",
        )

    correction = sample_categorical(root_target, generator)
    return TreeVerifyResult(
        output_token_ids=(correction,),
        accepted_node_indices=(),
        selected_branch=None,
        terminal="correction",
    )


@dataclass
class BranchState:
    """Minimal branch state used by state-isolation/reference tests."""

    kv: torch.Tensor
    gdn_conv: torch.Tensor
    gdn_ssm: torch.Tensor
    mtp: torch.Tensor
    position: int

    def clone(self) -> "BranchState":
        return BranchState(
            kv=self.kv.clone(),
            gdn_conv=self.gdn_conv.clone(),
            gdn_ssm=self.gdn_ssm.clone(),
            mtp=self.mtp.clone(),
            position=int(self.position),
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.kv, self.gdn_conv, self.gdn_ssm, self.mtp


StepState = Callable[[BranchState, int, TreeNode], BranchState]


def materialize_branch_states(
    topology: Fixed6Topology,
    root_state: BranchState,
    token_ids: Sequence[int],
    step: StepState,
) -> dict[int, BranchState]:
    """Execute every branch from an isolated root-state clone."""
    if len(token_ids) != FIXED_NODE_BUDGET:
        raise ValueError("expected exactly six node tokens")
    states: dict[int, BranchState] = {}
    for branch in topology.branch_nodes:
        current = root_state.clone()
        for node_index in branch:
            current = step(current, int(token_ids[node_index]), topology.nodes[node_index])
            states[node_index] = current.clone()
    assert_branch_isolation(states)
    return states


def assert_branch_isolation(states: Mapping[int, BranchState]) -> None:
    """Raise if any mutable state tensor aliases another node's storage."""
    seen: dict[int, tuple[int, str]] = {}
    for node_index, state in states.items():
        for name, tensor in zip(("kv", "gdn_conv", "gdn_ssm", "mtp"), state.tensors()):
            pointer = tensor.untyped_storage().data_ptr()
            if pointer in seen:
                other_node, other_name = seen[pointer]
                raise AssertionError(
                    f"state alias: node {node_index}.{name} shares storage with "
                    f"node {other_node}.{other_name}"
                )
            seen[pointer] = (node_index, name)


def commit_selected_state(
    root_state: BranchState,
    states: Mapping[int, BranchState],
    accepted_node_indices: Sequence[int],
) -> BranchState:
    """Commit only the selected prefix; discarded branches remain unreachable."""
    if not accepted_node_indices:
        return root_state.clone()
    return states[int(accepted_node_indices[-1])].clone()


def expected_mal_from_acceptance(
    branch_acceptance: Iterable[Sequence[float]],
    root_selection_probability: Sequence[float],
) -> float:
    """Expected MAL for disjoint branches plus one correction/bonus token."""
    total = 1.0
    for root_probability, acceptance in zip(
        root_selection_probability, branch_acceptance, strict=True
    ):
        reach = float(root_probability)
        for probability in acceptance:
            reach *= float(probability)
            total += reach
    return total
