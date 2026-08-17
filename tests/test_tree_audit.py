from __future__ import annotations

import torch

from remtp.fixed6_microtree import TreeVerifyResult, get_topology
from remtp.tree_audit import build_node_records, node_statuses


def test_statuses_mark_only_evaluated_selected_path() -> None:
    topology = get_topology("3+2+1")
    result = TreeVerifyResult(
        output_token_ids=(10, 20),
        accepted_node_indices=(1,),
        selected_branch=1,
        terminal="correction",
    )
    assert node_statuses(topology, result) == {
        0: "REJECT",
        1: "ACCEPT",
        2: "NOT_VISITED",
        3: "NOT_VISITED",
        4: "REJECT",
        5: "NOT_VISITED",
    }


def test_root_record_uses_recursive_residual_probability() -> None:
    topology = get_topology("3+2+1")
    result = TreeVerifyResult(
        output_token_ids=(2,),
        accepted_node_indices=(1,),
        selected_branch=1,
        terminal="bonus",
    )
    ids = torch.tensor([1, 2, 3, 4, 5, 6])
    q = torch.zeros(6, 8)
    q[0, 1] = 0.5
    q[0, 7] = 0.5
    q[1:, :] = 1 / 8
    p = torch.full((7, 8), 1 / 8)
    records = build_node_records(topology, result, ids, q, p, lambda token: str(token))
    assert records[0]["status"] == "REJECT"
    assert records[1]["status"] == "ACCEPT"
    # Rejecting root 0 removes positive q mass at token 1 from root P.
    assert records[1]["target_source"] == "sibling-residual"
