"""vLLM 0.18 runtime adapter for dynamic MiMo MTP candidate trees."""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import replace
from typing import Any

import torch

from remtp.dynamic_mtp_tree import (
    adaptive_node_limit,
    cactus_trunk_rescue_verify,
    DynamicTreeConfig,
    DynamicTreeTopology,
    level_expansion_order,
    level_expansion_budget,
    mtp_layer_for_tree_depth,
    path_expansion_value,
    residual_hit_tree_verify,
    select_dynamic_candidates,
    target_dominant_relaxed_verify,
)
from remtp.mimo_mtp import force_mimo_mtp_layer


_INSTALLED = False
_ORIGINAL_TREE_RUN: Any | None = None
_BRANCH_PROPOSAL_ROUND = 0


def _copy_variable_drafts_to_cpu(
    self: Any,
    scheduler_output: Any,
    zeros_only: bool = False,
) -> None:
    """Copy the real dynamic width into vLLM's maximum-width CPU buffer."""
    draft = self._draft_token_ids
    if not torch.is_tensor(draft):
        original = getattr(_copy_variable_drafts_to_cpu, "_dynamic_original")
        return original(self, scheduler_output, zeros_only=zeros_only)
    if zeros_only:
        self._dynamic_draft_width = 0
        self._draft_token_req_ids = self.input_batch.req_ids.copy()
        return
    self._draft_token_req_ids = self.input_batch.req_ids.copy()
    width = int(draft.shape[1])
    self._dynamic_draft_width = width
    assert self.draft_token_ids_event is not None
    assert self.draft_token_ids_copy_stream is not None
    assert self.draft_token_ids_cpu is not None
    default_stream = torch.cuda.current_stream()
    with torch.cuda.stream(self.draft_token_ids_copy_stream):
        self.draft_token_ids_copy_stream.wait_stream(default_stream)
        self.draft_token_ids_cpu[: draft.shape[0], :width].copy_(
            draft, non_blocking=True
        )
        self.draft_token_ids_event.record()


def _get_variable_drafts_cpu(self: Any) -> tuple[list[list[int]], list[str]]:
    width = getattr(self, "_dynamic_draft_width", None)
    if width is None:
        original = getattr(_get_variable_drafts_cpu, "_dynamic_original")
        return original(self)
    req_ids = self._draft_token_req_ids or []
    assert self.draft_token_ids_event is not None
    assert self.draft_token_ids_cpu is not None
    self.draft_token_ids_event.synchronize()
    values = self.draft_token_ids_cpu[: len(req_ids), : int(width)].tolist()
    return values, req_ids


def dynamic_tree_config_from_env() -> DynamicTreeConfig:
    eos_ids_raw = os.getenv("REMTP_DYNAMIC_TREE_EOS_TOKEN_IDS", "").strip()
    eos_token_ids = tuple(
        int(part.strip()) for part in eos_ids_raw.split(",") if part.strip()
    )
    config = DynamicTreeConfig(
        max_depth=int(os.getenv("REMTP_DYNAMIC_TREE_MAX_DEPTH", "3")),
        max_nodes=int(os.getenv("REMTP_DYNAMIC_TREE_MAX_NODES", "6")),
        adaptive_base_nodes=int(
            os.getenv("REMTP_DYNAMIC_TREE_ADAPTIVE_BASE_NODES", "0")
        ),
        adaptive_entropy_threshold=float(
            os.getenv(
                "REMTP_DYNAMIC_TREE_ADAPTIVE_ENTROPY_THRESHOLD", "1.0"
            )
        ),
        max_children_per_parent=int(
            os.getenv("REMTP_DYNAMIC_TREE_MAX_CHILDREN", "2")
        ),
        min_sibling_ratio=float(
            os.getenv("REMTP_DYNAMIC_TREE_MIN_SIBLING_RATIO", "0.25")
        ),
        min_draft_prob=float(os.getenv("REMTP_DYNAMIC_TREE_TAU_MIN", "0.02")),
        threshold_sensitivity=float(os.getenv("REMTP_DYNAMIC_TREE_KAPPA", "1.0")),
        depth_exploration=float(os.getenv("REMTP_DYNAMIC_TREE_MU", "0.5")),
        depth_reward=float(os.getenv("REMTP_DYNAMIC_TREE_ETA", "0.25")),
        allocation_mode=os.getenv(
            "REMTP_DYNAMIC_TREE_ALLOCATION", "geometric"
        ).strip().lower(),
        rank_penalty=float(
            os.getenv("REMTP_DYNAMIC_TREE_RANK_PENALTY", "2.0")
        ),
        coverage_weight=float(os.getenv("REMTP_DYNAMIC_TREE_ALPHA", "0.5")),
        coverage_mode=os.getenv(
            "REMTP_DYNAMIC_TREE_COVERAGE_MODE", "geometric"
        ).strip().lower(),
        min_target_coverage=float(
            os.getenv("REMTP_DYNAMIC_TREE_MIN_TARGET_COVERAGE", "0.0")
        ),
        relax_threshold=float(os.getenv("REMTP_DYNAMIC_TREE_TAU_RELAX", "0.7")),
        support_mode=os.getenv(
            "REMTP_DYNAMIC_TREE_SUPPORT_MODE", "relative"
        ).strip().lower(),
        proposal_support_ratio=float(
            os.getenv(
                "REMTP_DYNAMIC_TREE_PROPOSAL_SUPPORT_RATIO", "1.0"
            )
        ),
        confirmation_min_relative=float(
            os.getenv(
                "REMTP_DYNAMIC_TREE_CONFIRMATION_MIN_RELATIVE", "0.05"
            )
        ),
        cactus_delta=float(
            os.getenv("REMTP_DYNAMIC_TREE_CACTUS_DELTA", "1.0")
        ),
        cactus_target_weight=float(
            os.getenv("REMTP_DYNAMIC_TREE_CACTUS_TARGET_WEIGHT", "0.25")
        ),
        max_guided_rescues_per_path=int(
            os.getenv("REMTP_DYNAMIC_TREE_MAX_GUIDED_RESCUES", "1")
        ),
        rescue_score_threshold=float(
            os.getenv(
                "REMTP_DYNAMIC_TREE_RESCUE_SCORE_THRESHOLD", "0.35"
            )
        ),
        rescue_depth_penalty=float(
            os.getenv("REMTP_DYNAMIC_TREE_RESCUE_DEPTH_PENALTY", "0.0")
        ),
        rescue_continuation_discount=float(
            os.getenv(
                "REMTP_DYNAMIC_TREE_RESCUE_CONTINUATION_DISCOUNT", "0.0"
            )
        ),
        rescue_continuation_min_depth=int(
            os.getenv(
                "REMTP_DYNAMIC_TREE_RESCUE_CONTINUATION_MIN_DEPTH", "2"
            )
        ),
        rescue_margin_reference=float(
            os.getenv("REMTP_DYNAMIC_TREE_RESCUE_MARGIN_REFERENCE", "0.0")
        ),
        rescue_margin_penalty=float(
            os.getenv("REMTP_DYNAMIC_TREE_RESCUE_MARGIN_PENALTY", "0.0")
        ),
        length_reward=float(os.getenv("REMTP_DYNAMIC_TREE_BETA", "0.5")),
        path_temperature=float(os.getenv("REMTP_DYNAMIC_TREE_PATH_TEMPERATURE", "0.3")),
        path_selection_mode=os.getenv(
            "REMTP_DYNAMIC_TREE_PATH_SELECTION", "longest"
        ).strip().lower(),
        frontier_rescue=os.getenv(
            "REMTP_DYNAMIC_TREE_FRONTIER_RESCUE", "0"
        ) == "1",
        rescue_delta=float(
            os.getenv("REMTP_DYNAMIC_TREE_RESCUE_DELTA", "0.5")
        ),
        rescue_min_relative=float(
            os.getenv("REMTP_DYNAMIC_TREE_RESCUE_MIN_RELATIVE", "0.1")
        ),
        rescue_min_target_prob=float(
            os.getenv("REMTP_DYNAMIC_TREE_RESCUE_MIN_TARGET_PROB", "0.001")
        ),
        append_target_anchor=os.getenv("REMTP_DYNAMIC_TREE_APPEND_ANCHOR", "1") == "1",
        eos_token_ids=eos_token_ids,
        eos_protection_threshold=float(
            os.getenv("REMTP_DYNAMIC_TREE_EOS_THRESHOLD", "0.5")
        ),
    )
    config.validate()
    return config


def _set_builder_topology(builder: Any, topology: DynamicTreeTopology) -> None:
    from vllm.v1.attention.backends.tree_attn import (
        TreeAttentionMetadataBuilder,
        _get_depth_counts,
        _prepare_tree_attn_bias,
    )

    if not isinstance(builder, TreeAttentionMetadataBuilder):
        raise RuntimeError("dynamic MTP tree requires TREE_ATTN metadata")
    choices = list(topology.choices)
    current = builder.tree_attn_bias
    device = current.device
    dtype = current.dtype if current.dtype.is_floating_point else torch.float32
    builder.tree_attn_bias = _prepare_tree_attn_bias(
        choices,
        _get_depth_counts(choices),
        dtype=dtype,
        device=device,
    )
    builder.reorder_batch_threshold = builder.tree_attn_bias.shape[0]


def _set_target_tree_topology(runner: Any) -> None:
    from remtp import fixed6_vllm as fixed

    topology = fixed._RUNTIME.topology
    for groups in runner.attn_groups:
        for group in groups:
            _set_builder_topology(group.get_metadata_builder(), topology)


def _prepare_dynamic_target_inputs(self: Any, *args: Any, **kwargs: Any) -> Any:
    from remtp import fixed6_vllm as fixed

    _set_target_tree_topology(self)
    return fixed._prepare_target_tree_inputs(self, *args, **kwargs)


def _run_dynamic_tree_nodes(self: Any, **kwargs: Any) -> Any:
    assert _ORIGINAL_TREE_RUN is not None
    from remtp import fixed6_vllm as fixed

    _set_builder_topology(
        self.draft_attn_groups[0].get_metadata_builder(), fixed._RUNTIME.topology
    )
    node_ids = kwargs["node_ids"]
    depth = max(fixed._RUNTIME.topology.nodes[index].depth for index in node_ids)
    route = os.getenv("REMTP_DYNAMIC_TREE_MTP_ROUTE", "physical_012")
    physical_layer = mtp_layer_for_tree_depth(depth, route)
    with force_mimo_mtp_layer(physical_layer):
        return _ORIGINAL_TREE_RUN(self, **kwargs)


def _fallback_root(
    probs: torch.Tensor,
) -> torch.Tensor:
    return probs.argmax().reshape(1)


def _sample_q_token(
    probs: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Use the same exponential-race categorical primitive as chain MTP."""
    temperatures = getattr(sampling_metadata, "temperature", None)
    if temperatures is None or float(temperatures[0]) < 1e-5:
        return probs.argmax().reshape(1)
    noise = torch.empty_like(probs)
    generator = (getattr(sampling_metadata, "generators", None) or {}).get(0)
    if generator is not None:
        noise.exponential_(generator=generator)
    else:
        noise.exponential_()
    return (probs / noise).argmax().reshape(1)


def _sample_branch_q_token(
    probs: torch.Tensor,
    sampling_metadata: Any,
    *,
    parent_path: tuple[int, ...],
    proposal_round: int,
) -> torch.Tensor:
    """Sample branch-local Q without advancing the request RNG stream.

    The primary path must consume exactly the request's proposal RNG. Backup
    descendants need genuine Q samples for strict rejection sampling, but
    consuming the same generator would perturb the later primary-verification
    draws. A deterministic branch-local generator derived from request seed,
    round and path keeps those two random streams independent and reproducible.
    """
    temperatures = getattr(sampling_metadata, "temperature", None)
    if temperatures is None or float(temperatures[0]) < 1e-5:
        return probs.argmax().reshape(1)
    request_generator = (
        (getattr(sampling_metadata, "generators", None) or {}).get(0)
    )
    initial_seed = (
        int(request_generator.initial_seed())
        if request_generator is not None
        else 0
    )
    path_hash = 1469598103934665603
    for value in parent_path:
        path_hash ^= int(value) + 1
        path_hash = (path_hash * 1099511628211) & ((1 << 63) - 1)
    seed = (
        initial_seed
        ^ (int(proposal_round) * 0x9E3779B185EBCA87)
        ^ path_hash
    ) & ((1 << 63) - 1)
    branch_generator = torch.Generator(device=probs.device)
    branch_generator.manual_seed(seed)
    noise = torch.empty_like(probs)
    noise.exponential_(generator=branch_generator)
    return (probs / noise).argmax().reshape(1)


def _cactus_trunk_children(
    probs: torch.Tensor,
    trunk_token: torch.Tensor,
    *,
    depth: int,
    backup_budget: int,
    config: DynamicTreeConfig,
) -> tuple[torch.Tensor, Any]:
    """Return sampled trunk first, then deterministic high-Q backups."""
    selection = select_dynamic_candidates(probs, depth=depth, config=config)
    trunk_id = int(trunk_token)
    backups = [
        token
        for token in selection.token_ids
        if int(token) != trunk_id
    ][: max(backup_budget, 0)]
    values = [trunk_token.reshape(())] + [token.reshape(()) for token in backups]
    return torch.stack(values), selection


def _propose_cactus_trunk_rescue(
    self: Any,
    *,
    logits: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    common_attn_metadata: Any,
    sampling_metadata: Any,
    config: DynamicTreeConfig,
) -> list[torch.Tensor]:
    """Build a sampled-Q trunk with rejection-only siblings at each depth."""
    from remtp import fixed6_vllm as fixed

    if config.max_nodes < config.max_depth:
        raise ValueError(
            "cactus_trunk_rescue needs at least one node per trunk depth"
        )
    token_by_path: dict[tuple[int, ...], torch.Tensor] = {}
    q_by_path: dict[tuple[int, ...], torch.Tensor] = {}
    hidden_by_path: dict[tuple[int, ...], torch.Tensor] = {}
    construction: list[dict[str, Any]] = []
    root_hidden = hidden_states.reshape(1, -1)[0]
    parent_q = fixed._temperature_probs(logits[:1], sampling_metadata)[0]
    parent_hidden = root_hidden

    for depth in range(1, config.max_depth + 1):
        nodes_before = len(token_by_path)
        future_trunk_nodes = config.max_depth - depth
        available_after_trunk = (
            config.max_nodes
            - nodes_before
            - 1
            - future_trunk_nodes
        )
        backup_budget = min(
            config.max_children_per_parent - 1,
            max(available_after_trunk, 0),
        )
        trunk_token = _sample_q_token(parent_q, sampling_metadata)[0]
        children, selection = _cactus_trunk_children(
            parent_q,
            trunk_token,
            depth=depth,
            backup_budget=backup_budget,
            config=config,
        )
        parent_path = (0,) * (depth - 1)
        for local_rank, token in enumerate(children):
            path = parent_path + (local_rank,)
            token_by_path[path] = token
            q_by_path[path] = parent_q
            hidden_by_path[path] = parent_hidden
        construction.append(
            {
                "parent_path": list(parent_path) if parent_path else None,
                "depth": depth,
                "proposal_role": "sampled-cactus-trunk-with-backups",
                "sampled_trunk_token": int(trunk_token),
                "entropy": selection.entropy,
                "threshold": selection.threshold,
                "q_max": selection.q_max,
                "eligible": selection.candidate_count_before_budget,
                "threshold_eligible": selection.threshold_candidate_count,
                "retained": int(children.numel()),
                "retained_backups": int(children.numel() - 1),
                "forced_top1": False,
            }
        )
        topology = DynamicTreeTopology.from_paths(
            tuple(token_by_path), name="cactus-trunk-rescue"
        )
        fixed._RUNTIME.topology = topology

        if depth == config.max_depth:
            break
        current_paths = list(topology.choices)
        node_tokens = {
            index: token_by_path[path]
            for index, path in enumerate(current_paths)
        }
        node_hidden = {
            index: hidden_by_path[path]
            for index, path in enumerate(current_paths)
        }
        last_hidden, hidden, _ = _run_dynamic_tree_nodes(
            self,
            node_ids=list(range(len(current_paths))),
            node_tokens=node_tokens,
            node_hidden_inputs=node_hidden,
            root_positions=positions,
            base_common_metadata=common_attn_metadata,
        )
        trunk_path = (0,) * depth
        path_to_row = {path: index for index, path in enumerate(current_paths)}
        trunk_row = path_to_row[trunk_path]
        parent_q = fixed._temperature_probs(
            self.model.compute_logits(last_hidden[trunk_row : trunk_row + 1]),
            sampling_metadata,
        )[0]
        parent_hidden = hidden[trunk_row]

    # Populate branch-local MTP KV for exactly the final caterpillar topology.
    current_paths = list(topology.choices)
    node_tokens = {
        index: token_by_path[path] for index, path in enumerate(current_paths)
    }
    node_hidden = {
        index: hidden_by_path[path] for index, path in enumerate(current_paths)
    }
    _run_dynamic_tree_nodes(
        self,
        node_ids=list(range(len(current_paths))),
        node_tokens=node_tokens,
        node_hidden_inputs=node_hidden,
        root_positions=positions,
        base_common_metadata=common_attn_metadata,
    )

    ids = torch.stack([token_by_path[path] for path in current_paths]).to(
        torch.int32
    )
    probs = torch.stack([q_by_path[path] for path in current_paths]).contiguous()
    fixed._RUNTIME.draft_token_ids = ids.to(torch.int64)
    fixed._RUNTIME.draft_probs = probs
    fixed._RUNTIME.dynamic_construction = {
        "config": config.__dict__,
        "proposal": "sampled-q-cactus-trunk-rejection-only-rescue",
        "mtp_route": os.getenv(
            "REMTP_DYNAMIC_TREE_MTP_ROUTE", "physical_012"
        ),
        "node_count": topology.node_count,
        "max_depth": topology.max_depth,
        "branching": construction,
    }
    return [
        ids[list(topology.level_nodes(depth))].view(1, -1)
        for depth in range(1, topology.max_depth + 1)
        if topology.level_nodes(depth)
    ]


def _dynamic_propose_tree(
    self: Any,
    batch_size: int,
    logits: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    common_attn_metadata: Any,
    slot_mappings: Any = None,
) -> list[torch.Tensor]:
    del slot_mappings
    if batch_size != 1:
        raise RuntimeError("dynamic MTP tree currently requires --max-num-seqs 1")
    from remtp import fixed6_vllm as fixed

    sampling_metadata = getattr(self, "_fixed6_sampling_metadata", None)
    if sampling_metadata is None:
        raise RuntimeError("dynamic MTP tree lost sampling metadata")
    config = dynamic_tree_config_from_env()
    if config.support_mode == "cactus_trunk_rescue":
        return _propose_cactus_trunk_rescue(
            self,
            logits=logits,
            positions=positions,
            hidden_states=hidden_states,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            config=config,
        )
    global _BRANCH_PROPOSAL_ROUND
    sampled_primary_modes = {
        "sampled_primary_reopen",
        "sampled_primary_shadow",
        "residual_hit_anchor",
        "residual_hit_strict",
        "residual_hit_cactus",
    }
    sampled_primary_mode = config.support_mode in sampled_primary_modes
    proposal_round = _BRANCH_PROPOSAL_ROUND
    _BRANCH_PROPOSAL_ROUND += 1
    if sampled_primary_mode and config.max_nodes < config.max_depth:
        raise ValueError(
            "sampled-primary tree modes need at least one node per trunk depth"
        )
    root_q = fixed._temperature_probs(logits[:1], sampling_metadata)[0]
    if sampled_primary_mode:
        sampled_root = _sample_q_token(root_q, sampling_metadata)[0]
        root_backup_budget = min(
            config.max_children_per_parent - 1,
            max(config.max_nodes - config.max_depth, 0),
        )
        root_ids, root_selection = _cactus_trunk_children(
            root_q,
            sampled_root,
            depth=1,
            backup_budget=root_backup_budget,
            config=config,
        )
        forced_root = False
    else:
        root_selection = select_dynamic_candidates(
            root_q, depth=1, config=config, budget=config.max_nodes
        )
        forced_root = root_selection.token_ids.numel() == 0
        root_ids = _fallback_root(root_q) if forced_root else root_selection.token_ids

    token_by_path: dict[tuple[int, ...], torch.Tensor] = {}
    q_by_path: dict[tuple[int, ...], torch.Tensor] = {}
    hidden_by_path: dict[tuple[int, ...], torch.Tensor] = {}
    log_q_sum: dict[tuple[int, ...], float] = {}
    root_hidden = hidden_states.reshape(batch_size, -1)[0]
    construction: list[dict[str, Any]] = [
        {
            "parent": None,
            "depth": 1,
            "entropy": root_selection.entropy,
            "threshold": root_selection.threshold,
            "q_max": root_selection.q_max,
            "eligible": root_selection.candidate_count_before_budget,
            "threshold_eligible": root_selection.threshold_candidate_count,
            "sibling_added": root_selection.sibling_added,
            "retained": int(root_ids.numel()),
            "forced_top1": forced_root,
            "proposal_role": (
                "sampled-primary-with-backups"
                if sampled_primary_mode
                else "dynamic-q-candidates"
            ),
        }
    ]
    for rank, token in enumerate(root_ids):
        path = (rank,)
        token_by_path[path] = token
        q_by_path[path] = root_q
        hidden_by_path[path] = root_hidden
        log_q_sum[path] = math.log(max(float(root_q[int(token)]), 1e-30))

    topology = DynamicTreeTopology.from_paths(tuple(token_by_path))
    fixed._RUNTIME.topology = topology

    for depth in range(2, config.max_depth + 1):
        remaining = config.max_nodes - len(token_by_path)
        if remaining <= 0:
            break
        current_paths = list(topology.choices)
        node_tokens = {
            index: token_by_path[path] for index, path in enumerate(current_paths)
        }
        node_hidden = {
            index: hidden_by_path[path] for index, path in enumerate(current_paths)
        }
        last_hidden, hidden, _ = _run_dynamic_tree_nodes(
            self,
            node_ids=list(range(len(current_paths))),
            node_tokens=node_tokens,
            node_hidden_inputs=node_hidden,
            root_positions=positions,
            base_common_metadata=common_attn_metadata,
        )
        path_to_row = {path: index for index, path in enumerate(current_paths)}
        parent_paths = [
            path for path in current_paths if len(path) == depth - 1
        ]
        if not parent_paths:
            break
        parent_rows = torch.stack([last_hidden[path_to_row[path]] for path in parent_paths])
        parent_q = fixed._temperature_probs(
            self.model.compute_logits(parent_rows), sampling_metadata
        )
        proposed: list[tuple[float, tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        level_entropies: list[float] = []
        trunk_proposal_path = (0,) * depth
        for row, parent_path in enumerate(parent_paths):
            if sampled_primary_mode:
                is_primary_parent = parent_path == (0,) * (depth - 1)
                sampled_token = (
                    _sample_q_token(parent_q[row], sampling_metadata)[0]
                    if is_primary_parent
                    else _sample_branch_q_token(
                        parent_q[row],
                        sampling_metadata,
                        parent_path=parent_path,
                        proposal_round=proposal_round,
                    )[0]
                )
                candidate_ids, selection = _cactus_trunk_children(
                    parent_q[row],
                    sampled_token,
                    depth=depth,
                    backup_budget=config.max_children_per_parent - 1,
                    config=config,
                )
                proposal_role = (
                    "sampled-primary-with-backups"
                    if is_primary_parent
                    else "sampled-recovery-branch-with-backups"
                )
            else:
                selection = select_dynamic_candidates(
                    parent_q[row], depth=depth, config=config
                )
                candidate_ids = selection.token_ids
                proposal_role = "dynamic-q-candidates"
            level_entropies.append(selection.entropy)
            construction.append(
                {
                    "parent_path": list(parent_path),
                    "parent": path_to_row[parent_path],
                    "depth": depth,
                    "entropy": selection.entropy,
                    "threshold": selection.threshold,
                    "q_max": selection.q_max,
                    "eligible": selection.candidate_count_before_budget,
                    "threshold_eligible": selection.threshold_candidate_count,
                    "sibling_added": selection.sibling_added,
                    "retained": 0,
                    "forced_top1": False,
                    "proposal_role": proposal_role,
                }
            )
            for local_rank, token in enumerate(candidate_ids):
                path = parent_path + (local_rank,)
                token_probability = max(float(parent_q[row, int(token)]), 1e-30)
                path_log = log_q_sum[parent_path] + math.log(token_probability)
                value = path_expansion_value(path_log, depth, config)
                proposed.append(
                    (
                        value,
                        path,
                        token,
                        parent_q[row],
                        hidden[path_to_row[parent_path]],
                        path_log,
                    )
                )
        if not proposed:
            break
        active_node_limit = adaptive_node_limit(level_entropies, config)
        active_config = replace(config, max_nodes=active_node_limit)
        retention_budget = level_expansion_budget(
            current_nodes=len(token_by_path),
            next_depth=depth,
            config=active_config,
        )
        if sampled_primary_mode:
            # Split the active budget across remaining levels so retained
            # recovery branches can still receive a continuation.
            remaining_levels = config.max_depth - depth + 1
            active_remaining = active_node_limit - len(token_by_path)
            retention_budget = min(
                retention_budget,
                max(active_remaining // remaining_levels, 1),
            )
        if retention_budget <= 0:
            break
        proposal_order = level_expansion_order(
            tuple((item[0], item[1]) for item in proposed),
            config=config,
        )
        if sampled_primary_mode:
            trunk_index = next(
                index
                for index, item in enumerate(proposed)
                if item[1] == trunk_proposal_path
            )
            retained_indices = [trunk_index]
            retained_indices.extend(
                index
                for index in proposal_order
                if index != trunk_index
            )
            retained_indices = retained_indices[:retention_budget]
        else:
            retained_indices = list(proposal_order[:retention_budget])
        retained = [proposed[index] for index in retained_indices]
        retained_per_parent: dict[tuple[int, ...], int] = {}
        for _, path, token, q_row, parent_hidden, path_log in retained:
            token_by_path[path] = token
            q_by_path[path] = q_row
            hidden_by_path[path] = parent_hidden
            log_q_sum[path] = path_log
            retained_per_parent[path[:-1]] = retained_per_parent.get(path[:-1], 0) + 1
        for row in construction:
            parent_path = tuple(row.get("parent_path", ()))
            if row.get("depth") == depth and parent_path in retained_per_parent:
                row["retained"] = retained_per_parent[parent_path]
                row["active_node_limit"] = active_node_limit
        topology = DynamicTreeTopology.from_paths(tuple(token_by_path))
        fixed._RUNTIME.topology = topology

    # Materialize the final dynamic topology and all branch-local draft KV.
    current_paths = list(topology.choices)
    node_tokens = {
        index: token_by_path[path] for index, path in enumerate(current_paths)
    }
    node_hidden = {
        index: hidden_by_path[path] for index, path in enumerate(current_paths)
    }
    _run_dynamic_tree_nodes(
        self,
        node_ids=list(range(len(current_paths))),
        node_tokens=node_tokens,
        node_hidden_inputs=node_hidden,
        root_positions=positions,
        base_common_metadata=common_attn_metadata,
    )

    ids = torch.stack([token_by_path[path] for path in current_paths]).to(torch.int32)
    probs = torch.stack([q_by_path[path] for path in current_paths]).contiguous()
    fixed._RUNTIME.draft_token_ids = ids.to(torch.int64)
    fixed._RUNTIME.draft_probs = probs
    fixed._RUNTIME.dynamic_construction = {
        "config": config.__dict__,
        "mtp_route": os.getenv(
            "REMTP_DYNAMIC_TREE_MTP_ROUTE", "physical_012"
        ),
        "node_count": topology.node_count,
        "max_depth": topology.max_depth,
        "branching": construction,
    }

    return [
        ids[list(topology.level_nodes(depth))].view(1, -1)
        for depth in range(1, topology.max_depth + 1)
        if topology.level_nodes(depth)
    ]


def _dynamic_verify_dispatch(*args: Any, **kwargs: Any) -> Any:
    from remtp import fixed6_vllm as fixed
    from remtp.fixed6_microtree import TreeVerifyResult, sample_categorical

    construction = getattr(fixed._RUNTIME, "dynamic_construction", None) or {}
    config = dynamic_tree_config_from_env()
    if config.support_mode in {
        "cactus_trunk_rescue",
        "sampled_primary_reopen",
    }:
        return cactus_trunk_rescue_verify(
            *args,
            **kwargs,
            config=config,
        )
    if config.support_mode in {
        "sampled_primary_shadow",
        "residual_hit_anchor",
        "residual_hit_strict",
        "residual_hit_cactus",
    }:
        return residual_hit_tree_verify(
            *args,
            **kwargs,
            config=config,
        )
    branching = construction.get("branching", [])
    if branching and bool(branching[0].get("forced_top1")):
        topology, _, _, target_probs = args[:4]
        generator = args[4] if len(args) > 4 else kwargs.get("generator")
        correction = sample_categorical(target_probs[0], generator)
        return TreeVerifyResult(
            output_token_ids=(correction,),
            accepted_node_indices=(),
            selected_branch=None,
            terminal="draft-truncated-target-fallback",
            target_validation_nodes=len(topology.nodes),
            evaluated_node_statuses=tuple(
                (node.index, "PRUNE") for node in topology.nodes
            ),
        )
    return target_dominant_relaxed_verify(
        *args,
        **kwargs,
        config=config,
    )


def install_mimo_dynamic_tree() -> None:
    global _INSTALLED, _ORIGINAL_TREE_RUN
    if _INSTALLED:
        return
    from remtp import fixed6_vllm as fixed

    config = dynamic_tree_config_from_env()
    route = os.getenv("REMTP_DYNAMIC_TREE_MTP_ROUTE", "physical_012")
    mtp_layer_for_tree_depth(1, route)
    # A one-node placeholder is replaced by the proposal before every target
    # forward. It only gives shared runtime code a valid initial topology.
    fixed._RUNTIME.topology = DynamicTreeTopology.from_paths(((0,),))
    fixed._RUNTIME.dynamic_construction = None

    eagle = importlib.import_module("vllm.v1.spec_decode.eagle")
    runner_module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    rejection = importlib.import_module("vllm.v1.sample.rejection_sampler")
    mimo = importlib.import_module("vllm.model_executor.models.mimo")

    proposer_cls = eagle.EagleProposer
    fixed._propose_with_sampling_metadata._fixed6_original = proposer_cls.propose
    proposer_cls.propose = fixed._propose_with_sampling_metadata
    proposer_cls.propose_tree = _dynamic_propose_tree
    fixed._prepare_selected_path_inputs._fixed6_original = (
        proposer_cls.prepare_inputs_padded
    )
    proposer_cls.prepare_inputs_padded = fixed._prepare_selected_path_inputs

    _ORIGINAL_TREE_RUN = fixed._run_draft_tree_nodes
    fixed._run_draft_tree_nodes = _run_dynamic_tree_nodes
    fixed.strict_tree_verify = _dynamic_verify_dispatch

    runner_cls = runner_module.GPUModelRunner
    fixed._prepare_target_tree_inputs._fixed6_original = runner_cls._prepare_inputs
    runner_cls._prepare_inputs = _prepare_dynamic_target_inputs
    fixed._capture_target_slots._fixed6_original = runner_cls._get_slot_mappings
    runner_cls._get_slot_mappings = fixed._capture_target_slots
    fixed._compact_selected_path._fixed6_original = runner_cls._update_states_after_model_execute
    runner_cls._update_states_after_model_execute = fixed._compact_selected_path
    _copy_variable_drafts_to_cpu._dynamic_original = (
        runner_cls._copy_draft_token_ids_to_cpu
    )
    runner_cls._copy_draft_token_ids_to_cpu = _copy_variable_drafts_to_cpu
    _get_variable_drafts_cpu._dynamic_original = runner_cls._get_draft_token_ids_cpu
    runner_cls._get_draft_token_ids_cpu = _get_variable_drafts_cpu

    sampler_cls = rejection.RejectionSampler
    fixed._strict_tree_sampler._fixed6_original = sampler_cls.forward
    sampler_cls.forward = fixed._strict_tree_sampler

    target_cls = mimo.MiMoForCausalLM
    fixed._profile_target_forward._fixed6_original = target_cls.forward
    target_cls.forward = fixed._profile_target_forward

    _INSTALLED = True
    proposal_name = (
        "sampled-q-trunk+rejection-rescue"
        if config.support_mode == "cactus_trunk_rescue"
        else (
            "sampled-q-primary+full-tree-recovery"
            if config.support_mode == "sampled_primary_reopen"
            else (
                "sampled-q-primary+exact-residual-hit-tree"
                if config.support_mode
                in {
                    "sampled_primary_shadow",
                    "residual_hit_anchor",
                    "residual_hit_strict",
                    "residual_hit_cactus",
                }
                else "dynamic-q-tree"
            )
        )
    )
    print(
        "[ReMTP][DynamicTree] installed target-dominant relaxed tree "
        f"D={config.max_depth} N_max={config.max_nodes} "
        f"children_max={config.max_children_per_parent} "
        f"sibling_ratio={config.min_sibling_ratio:g} "
        f"tau_min={config.min_draft_prob:g} kappa={config.threshold_sensitivity:g} "
        f"coverage_mode={config.coverage_mode} "
        f"alpha={config.coverage_weight:g} "
        f"min_coverage={config.min_target_coverage:g} "
        f"tau_relax={config.relax_threshold:g} "
        f"support_mode={config.support_mode} "
        f"proposal={proposal_name} "
        f"cactus_delta={config.cactus_delta:g} "
        f"cactus_target_weight={config.cactus_target_weight:g} "
        f"max_guided_rescues={config.max_guided_rescues_per_path} "
        f"rescue_score_threshold={config.rescue_score_threshold:g} "
        f"rescue_depth_penalty={config.rescue_depth_penalty:g} "
        f"rescue_continuation_discount={config.rescue_continuation_discount:g} "
        f"rescue_continuation_min_depth={config.rescue_continuation_min_depth} "
        f"rescue_margin_reference={config.rescue_margin_reference:g} "
        f"rescue_margin_penalty={config.rescue_margin_penalty:g} "
        f"path_selection={config.path_selection_mode} "
        f"beta={config.length_reward:g} "
        f"frontier_rescue={int(config.frontier_rescue)} "
        f"rescue_delta={config.rescue_delta:g} "
        f"eos_ids={list(config.eos_token_ids)} "
        f"eos_threshold={config.eos_protection_threshold:g} "
        f"mtp_route={route} "
        "target_forwards=1",
        flush=True,
    )
