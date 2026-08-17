"""Dense-attention MiMo adapter for fixed-topology MTP trees.

This reuses the already unit-tested topology, parent-logit mapping, target
TreeAttention input preparation, selected-path KV compaction and audit path
from ``remtp.fixed6_vllm``. It supports the legacy six-node controls and the
three-level complete binary tree with 2/4/8 nodes. It intentionally omits every Qwen3.5 GDN patch:
MiMo uses dense Qwen2-style attention, so each branch is represented by the
tree attention mask and ordinary branch KV slots.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from remtp.binary_mtp_tree import get_mimo_tree_topology
from remtp.mimo_mtp import force_mimo_mtp_layer
from remtp.mimo_tree import verify_mimo_tree


_INSTALLED = False
_ORIGINAL_TREE_RUN: Any | None = None


def _run_mimo_tree_depth(self: Any, **kwargs: Any) -> Any:
    assert _ORIGINAL_TREE_RUN is not None
    node_ids = kwargs["node_ids"]
    from remtp import fixed6_vllm as fixed

    depth = max(fixed._RUNTIME.topology.nodes[index].depth for index in node_ids)
    # Root candidates are produced by physical layer 0 before propose_tree.
    # A cumulative pass over depth d produces depth d+1 and therefore uses
    # layer d, capped at the deepest published MiMo layer.
    layer = min(depth, 2)
    with force_mimo_mtp_layer(layer):
        return _ORIGINAL_TREE_RUN(self, **kwargs)


def _tree_verify_dispatch(*args: Any, **kwargs: Any) -> Any:
    return verify_mimo_tree(
        *args,
        **kwargs,
        mode=os.getenv("REMTP_TREE_VERIFY_MODE", "strict"),
        cactus_delta=float(os.getenv("REMTP_CACTUS_DELTA", "1.0")),
    )


def install_mimo_microtree() -> None:
    global _INSTALLED, _ORIGINAL_TREE_RUN
    if _INSTALLED:
        return
    from remtp import fixed6_vllm as fixed

    topology = get_mimo_tree_topology(os.getenv("REMTP_TREE_TOPOLOGY", "2x2x2"))
    mode = os.getenv("REMTP_TREE_VERIFY_MODE", "strict")
    if mode not in {"strict", "cactus_path"}:
        raise RuntimeError("MiMo tree verification must be strict or cactus_path")
    if len(topology.nodes) != 6 and mode != "strict":
        raise RuntimeError(
            "multi-candidate branching at every depth currently requires strict "
            "verification; candidate-specific Cactus sibling residuals are undefined"
        )
    fixed._RUNTIME.topology = topology

    eagle = importlib.import_module("vllm.v1.spec_decode.eagle")
    runner_module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    rejection = importlib.import_module("vllm.v1.sample.rejection_sampler")
    mimo = importlib.import_module("vllm.model_executor.models.mimo")

    proposer_cls = eagle.EagleProposer
    fixed._propose_with_sampling_metadata._fixed6_original = proposer_cls.propose
    proposer_cls.propose = fixed._propose_with_sampling_metadata
    proposer_cls.propose_tree = fixed._fixed6_propose_tree

    _ORIGINAL_TREE_RUN = fixed._run_draft_tree_nodes
    fixed._run_draft_tree_nodes = _run_mimo_tree_depth
    fixed.strict_tree_verify = _tree_verify_dispatch

    runner_cls = runner_module.GPUModelRunner
    fixed._prepare_target_tree_inputs._fixed6_original = runner_cls._prepare_inputs
    runner_cls._prepare_inputs = fixed._prepare_target_tree_inputs
    fixed._capture_target_slots._fixed6_original = runner_cls._get_slot_mappings
    runner_cls._get_slot_mappings = fixed._capture_target_slots
    fixed._compact_selected_path._fixed6_original = (
        runner_cls._update_states_after_model_execute
    )
    runner_cls._update_states_after_model_execute = fixed._compact_selected_path

    sampler_cls = rejection.RejectionSampler
    fixed._strict_tree_sampler._fixed6_original = sampler_cls.forward
    sampler_cls.forward = fixed._strict_tree_sampler

    target_cls = mimo.MiMoForCausalLM
    fixed._profile_target_forward._fixed6_original = target_cls.forward
    target_cls.forward = fixed._profile_target_forward

    _INSTALLED = True
    print(
        "[ReMTP][MiMo tree] installed "
        f"topology={topology.name} choices={fixed.topology_config_string(topology)} "
        f"verify={mode} nodes={len(topology.nodes)} target_forwards=1",
        flush=True,
    )
