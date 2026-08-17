"""Verification rules for dense-attention MiMo trees.

Root alternatives are always selected with exact multi-candidate rejection
sampling.  Optional relaxation is deliberately restricted to descendants of
the selected root: using a different candidate-specific relaxed distribution
for every root would no longer define one coherent recursive residual.
"""

from __future__ import annotations

import torch

from remtp.cactus_mtp import cactus_target_distribution
from remtp.binary_mtp_tree import strict_general_tree_verify
from remtp.fixed6_microtree import (
    Fixed6Topology,
    TreeVerifyResult,
    _accept,
    normalize_distribution,
    residual_distribution,
    sample_categorical,
    strict_tree_verify,
)


def path_cactus_tree_verify(
    topology: Fixed6Topology,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    delta: float = 1.0,
) -> TreeVerifyResult:
    """Strict root coverage followed by Cactus on the selected branch.

    The bonus token is always drawn from the original target distribution.
    A rejection is corrected from ``(h-q)+`` at the same descendant position.
    """
    if draft_token_ids.shape != (6,):
        raise ValueError("draft_token_ids must have shape [6]")
    if draft_probs.ndim != 2 or draft_probs.shape[0] != 6:
        raise ValueError("draft_probs must have shape [6, vocab]")
    if target_probs_by_input.shape != (7, draft_probs.shape[1]):
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
            relaxed = cactus_target_distribution(
                target.unsqueeze(0),
                torch.tensor([token], device=target.device),
                delta,
            )[0]
            if not _accept(relaxed, proposal, token, generator):
                correction = sample_categorical(
                    residual_distribution(relaxed, proposal), generator
                )
                output.append(correction)
                return TreeVerifyResult(
                    output_token_ids=tuple(output),
                    accepted_node_indices=tuple(accepted),
                    selected_branch=branch,
                    terminal="cactus-correction",
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


def verify_mimo_tree(
    topology: Fixed6Topology,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    mode: str = "strict",
    cactus_delta: float = 1.0,
) -> TreeVerifyResult:
    if mode == "strict":
        return strict_general_tree_verify(
            topology,
            draft_token_ids,
            draft_probs,
            target_probs_by_input,
            generator,
        )
    if mode == "cactus_path":
        if len(topology.nodes) != 6:
            raise ValueError(
                "cactus_path is not defined for multi-candidate sibling groups; "
                "use strict for the 2/4/8 binary tree"
            )
        return path_cactus_tree_verify(
            topology,
            draft_token_ids,
            draft_probs,
            target_probs_by_input,
            generator,
            delta=cactus_delta,
        )
    raise ValueError("REMTP_TREE_VERIFY_MODE must be strict or cactus_path")
