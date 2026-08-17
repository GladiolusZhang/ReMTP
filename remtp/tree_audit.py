"""Human-readable and JSON audit helpers for MTP tree verification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from remtp.fixed6_microtree import (
    Fixed6Topology,
    TreeVerifyResult,
    normalize_distribution,
    residual_distribution,
)


DecodeToken = Callable[[int], str]


def node_statuses(
    topology: Fixed6Topology,
    result: TreeVerifyResult,
) -> dict[int, str]:
    """Return ACCEPT/REJECT/NOT_VISITED for all BFS nodes."""
    statuses = {node.index: "NOT_VISITED" for node in topology.nodes}
    if result.evaluated_node_statuses:
        statuses.update(dict(result.evaluated_node_statuses))
        return statuses
    selected = set(result.accepted_node_indices)

    selected_root = result.accepted_node_indices[0] if result.accepted_node_indices else None
    for root in topology.roots:
        if selected_root is None:
            statuses[root] = "REJECT"
        elif root == selected_root:
            statuses[root] = "ACCEPT"
            break
        else:
            statuses[root] = "REJECT"

    for index in selected:
        statuses[index] = "ACCEPT"
    if result.accepted_node_indices and result.terminal.endswith("correction"):
        last = result.accepted_node_indices[-1]
        children = tuple(
            topology.children_of(last)
            if hasattr(topology, "children_of")
            else (node.index for node in topology.nodes if node.parent == last)
        )
        if len(children) == 1:
            statuses[children[0]] = "REJECT"
    return statuses


def build_node_records(
    topology: Fixed6Topology,
    result: TreeVerifyResult,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs_by_input: torch.Tensor,
    decode_token: DecodeToken,
) -> list[dict[str, Any]]:
    """Build six compact records without serializing full P/Q tensors."""
    ids = draft_token_ids.to(device=target_probs_by_input.device, dtype=torch.int64)
    draft = draft_probs.to(device=target_probs_by_input.device, dtype=torch.float32)
    target = target_probs_by_input.to(torch.float32)
    statuses = node_statuses(topology, result)

    # Every sibling group is tested against a recursive residual. Record the
    # actual verifier distribution at each visited candidate.
    verifier_distributions: dict[int, torch.Tensor] = {}
    parent: int | None = None
    running = normalize_distribution(target[0])
    while True:
        siblings = tuple(
            topology.children_of(parent)
            if hasattr(topology, "children_of")
            else (
                topology.roots
                if parent is None
                else tuple(node.index for node in topology.nodes if node.parent == parent)
            )
        )
        accepted_child: int | None = None
        for sibling in siblings:
            if statuses[sibling] == "NOT_VISITED":
                break
            verifier_distributions[sibling] = running
            if statuses[sibling] == "REJECT":
                running = residual_distribution(running, draft[sibling])
            elif statuses[sibling] == "ACCEPT":
                accepted_child = sibling
                break
        if accepted_child is None:
            break
        parent = accepted_child
        children = tuple(
            topology.children_of(parent)
            if hasattr(topology, "children_of")
            else (node.index for node in topology.nodes if node.parent == parent)
        )
        if not children:
            break
        running = normalize_distribution(target[parent + 1])

    rows: list[dict[str, Any]] = []
    for node in topology.nodes:
        token_id = int(ids[node.index].item())
        proposal = normalize_distribution(draft[node.index])
        if node.index in verifier_distributions:
            verifier = verifier_distributions[node.index]
            target_source = "sibling-residual"
        else:
            verifier_row = 0 if node.parent is None else node.parent + 1
            verifier = normalize_distribution(target[verifier_row])
            target_source = "anchor" if node.parent is None else f"parent-row-{verifier_row}"
        p_value = float(verifier[token_id])
        q_value = float(proposal[token_id])
        alpha = min(1.0, p_value / max(q_value, 1e-30))
        top1_id = int(verifier.argmax().item())
        rows.append(
            {
                "node": node.index,
                "branch": node.branch,
                "depth": node.depth,
                "parent": node.parent,
                "token_id": token_id,
                "token": decode_token(token_id),
                "q": q_value,
                "target_p": p_value,
                "strict_accept_probability": alpha,
                "target_top1_id": top1_id,
                "target_top1": decode_token(top1_id),
                "target_source": target_source,
                "status": statuses[node.index],
            }
        )
    return rows


def format_round_trace(record: dict[str, Any]) -> str:
    branch = record.get("selected_branch")
    branch_text = "none" if branch is None else str(branch)
    lines = [
        "[ReMTP][MiMoTree] "
        f"round={record['round']:03d} topology={record['topology']} "
        f"branch={branch_text} accepted_depth={record['accepted_drafts']} "
        f"terminal={record['terminal']} "
        f"rolling_mean_depth={record['rolling_mean_accepted_depth']:.3f} "
        f"rolling_tree_MAL={record['rolling_mean_acceptance_length']:.3f}"
    ]
    for node in record.get("nodes", []):
        parent = "root" if node["parent"] is None else f"N{node['parent']}"
        relaxed = ""
        if "relaxed_confidence" in node:
            relaxed = (
                f" coverage={node['coverage']:.4f} "
                f"relative={node['relative_target_support']:.4f} "
                f"S={node['relaxed_confidence']:.4f}"
            )
        rescue = ""
        if node.get("rescue_eligible"):
            decision = "ACCEPT" if node.get("rescued") else "REJECT"
            if node.get("rescue_decision_mode") == "score-threshold":
                rescue = (
                    f" rescue={decision} "
                    f"score={float(node['target_path_rescue_score']):.4f} "
                    f"tau={float(node['target_path_rescue_threshold']):.4f}"
                )
            else:
                rescue = (
                    f" rescue={decision} "
                    f"a={float(node['rescue_accept_probability']):.4f} "
                    f"u={float(node['rescue_random']):.4f}"
                )
        lines.append(
            "  "
            f"N{node['node']} B{node['branch']}/D{node['depth']} parent={parent} "
            f"draft={node['token_id']}({node['token']!r}) "
            f"q={node['q']:.4f} target_p={node['target_p']:.4f} "
            f"alpha={node['strict_accept_probability']:.4f} "
            f"target_top1={node['target_top1_id']}({node['target_top1']!r})"
            f"{relaxed}{rescue} "
            f"-> {node['status']}"
        )
    construction = record.get("dynamic_construction")
    if construction:
        lines.append(
            "  TREE   : "
            f"nodes={construction.get('node_count')} "
            f"depth={construction.get('max_depth')} "
            f"surviving_paths={record.get('surviving_paths', 0)}"
        )
    accepted = record.get("accepted_tokens", [])
    committed = record.get("output_tokens", [])
    lines.append(
        "  KEEP   : "
        + (" ".join(f"{item['id']}({item['text']!r})" for item in accepted) or "<none>")
    )
    lines.append(
        "  COMMIT : "
        + " ".join(f"{item['id']}({item['text']!r})" for item in committed)
    )
    return "\n".join(lines)
