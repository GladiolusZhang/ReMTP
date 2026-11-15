"""vLLM 0.18 runtime adapter for dynamic MiMo MTP candidate trees."""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import replace
from typing import Any

import torch

from remtp.dynamic_mtp_tree import (
    anchor_reusable_topology,
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


def _parse_relaxation_depth_weights(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated, 1-based native-MTP depth schedule."""
    value = raw.strip()
    if not value:
        return ()
    try:
        return tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(
            "REMTP_DYNAMIC_TREE_RELAX_DEPTH_WEIGHTS must be a "
            "comma-separated float list"
        ) from exc


def _copy_variable_drafts_to_cpu(
    self: Any,
    scheduler_output: Any,
    zeros_only: bool = False,
) -> None:
    """Copy the real dynamic width into vLLM's maximum-width CPU buffer."""
    draft = self._draft_token_ids
    from remtp import async_tree
    if not torch.is_tensor(draft) or async_tree.enabled():
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
    from remtp import async_tree
    if width is None or async_tree.enabled():
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
        relaxation_depth_weights=_parse_relaxation_depth_weights(
            os.getenv("REMTP_DYNAMIC_TREE_RELAX_DEPTH_WEIGHTS", "")
        ),
        adaptive_relaxation=os.getenv(
            "REMTP_DYNAMIC_TREE_ADAPTIVE_RELAXATION", "0"
        ) == "1",
        adaptive_relaxation_mix=float(
            os.getenv("REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MIX", "0.5")
        ),
        adaptive_relaxation_max_ratio=float(
            os.getenv("REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_MAX_RATIO", "1.5")
        ),
        relaxation_max_probability_ratio=float(
            os.getenv("REMTP_DYNAMIC_TREE_RELAX_MAX_PROBABILITY_RATIO", "0")
        ),
        adaptive_relaxation_target_power=float(
            os.getenv("REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_TARGET_POWER", "0.25")
        ),
        adaptive_relaxation_steps=int(
            os.getenv("REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_STEPS", "16")
        ),
        relaxation_risk_policy=os.getenv("REMTP_DYNAMIC_TREE_RELAX_RISK_POLICY", "none"),
        relaxation_min_target_probability=float(
            os.getenv("REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_PROB", "0")
        ),
        relaxation_min_target_ratio=float(
            os.getenv("REMTP_DYNAMIC_TREE_RELAX_MIN_TARGET_RATIO", "0")
        ),
        adaptive_relaxation_pool=os.getenv("REMTP_DYNAMIC_TREE_ADAPTIVE_RELAX_POOL", "nominal"),
        prune_anchor_descendants=os.getenv(
            "REMTP_DYNAMIC_TREE_PRUNE_ANCHOR_DESCENDANTS", "0"
        ) == "1",
        exact_topk_candidates=os.getenv(
            "REMTP_DYNAMIC_TREE_EXACT_TOPK", "0"
        ) == "1",
        anchor_only_proposal=os.getenv(
            "REMTP_DYNAMIC_TREE_ANCHOR_ONLY_PROPOSAL", "0"
        ) == "1",
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
        protect_eos_relaxation=os.getenv("REMTP_DYNAMIC_TREE_PROTECT_EOS", "0") == "1",
    )
    config.validate()
    return config


def _set_builder_topology(builder: Any, topology: DynamicTreeTopology, *,
                          persistent: bool = False) -> None:
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
    if os.getenv("REMTP_DYNAMIC_TREE_FAST_METADATA", "0") == "1":
        # Same 0/-inf mask, built once per topology on the host and copied in
        # one transfer instead of ~20 one-element device writes per call.
        builder.tree_attn_bias = _cached_tree_attn_bias(
            tuple(choices), dtype, device, _prepare_tree_attn_bias, _get_depth_counts
        )
    else:
        builder.tree_attn_bias = _prepare_tree_attn_bias(
            choices,
            _get_depth_counts(choices),
            dtype=dtype,
            device=device,
        )
    builder.reorder_batch_threshold = builder.tree_attn_bias.shape[0]
    if persistent:
        # CUDA graphs (2026-11-13): same mask values, in the captured buffer.
        from remtp.tree_cudagraph import write_persistent_mask

        write_persistent_mask(builder, builder.tree_attn_bias)


_TREE_BIAS_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_TREE_BIAS_CACHE_LIMIT = 4096


def _cached_tree_attn_bias(
    choices: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
    device: torch.device,
    prepare: Any,
    depth_counts: Any,
) -> torch.Tensor:
    """Return the read-only tree mask for ``choices``; values equal ``prepare``."""
    key = (choices, dtype, str(device))
    bias = _TREE_BIAS_CACHE.get(key)
    if bias is None:
        ordered = list(choices)
        host = prepare(ordered, depth_counts(ordered), dtype=dtype, device=torch.device("cpu"))
        bias = host.to(device)
        if len(_TREE_BIAS_CACHE) >= _TREE_BIAS_CACHE_LIMIT:
            _TREE_BIAS_CACHE.clear()
        _TREE_BIAS_CACHE[key] = bias
    return bias


def _set_target_tree_topology(runner: Any) -> None:
    from remtp import fixed6_vllm as fixed

    from remtp.tree_cudagraph import tree_cudagraph_enabled

    topology = fixed._RUNTIME.topology
    persistent = tree_cudagraph_enabled()
    for groups in runner.attn_groups:
        for group in groups:
            _set_builder_topology(group.get_metadata_builder(), topology, persistent=persistent)


def _prepare_dynamic_target_inputs(self: Any, *args: Any, **kwargs: Any) -> Any:
    from remtp import async_tree
    from remtp import fixed6_vllm as fixed

    if async_tree.enabled():
        # The mask of the device-built tree, written on the device (2026-11-15).
        if async_tree.STATE.ready:
            async_tree.write_target_mask(self)
    else:
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
    _validate_proposal_probs(probs)
    return probs.argmax().reshape(1)


def _validate_proposal_probs(probs: torch.Tensor) -> None:
    """Reject invalid Q before any draw; do not silently invent a proposal."""
    valid = torch.isfinite(probs).all() & (probs >= 0).all() & (probs.sum() > 0)
    if not bool(valid):
        raise ValueError('Invalid MTP proposal Q: nonfinite, negative, or zero mass; sampling aborted before RNG')


def _sample_q_token(
    probs: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Use the same exponential-race categorical primitive as chain MTP."""
    _validate_proposal_probs(probs)
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
    _validate_proposal_probs(probs)
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
            tuple(token_by_path), name=("anchor-primary-siblings"
                                       if config.anchor_only_proposal else "cactus-trunk-rescue")
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
        logit_hidden = last_hidden[trunk_row : trunk_row + 1]
        if config.anchor_only_proposal and "REMTP_HEAD_BATCH_ROWS" not in os.environ:
            # On Orin the BF16 head's 3-row GEMM is faster than its 1-row
            # path. Duplicate only head inputs; no extra MTP/KV nodes or RNG.
            logit_hidden = logit_hidden.expand(3, -1).contiguous()
        parent_q = fixed._temperature_probs(
            self.model.compute_logits(logit_hidden)[:1],
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
        "proposal": ("sampled-q-anchor-primary-siblings"
                     if config.anchor_only_proposal else "sampled-q-cactus-trunk-rejection-only-rescue"),
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
    if config.support_mode == "cactus_trunk_rescue" or config.anchor_only_proposal:
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
    if _GPU_BUILD and sampled_primary_mode:
        from remtp import raven_fast_path as fast
        temperature = fast.cached_temperature(sampling_metadata)
        if temperature is not None and temperature >= 1e-5:
            # Device-side construction, one host read per proposal (2026-11-15).
            from remtp import gpu_tree
            return gpu_tree.propose_tree(
                self, logits=logits, positions=positions, hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata, sampling_metadata=sampling_metadata,
                config=config, proposal_round=proposal_round,
            )
    root_q = fixed._temperature_probs(logits[:1], sampling_metadata)[0]
    # One host transfer per tree level instead of three per parent (2026-11-13).
    batch_select = sampled_primary_mode and _BATCH_SELECT
    batched_pending: list[tuple[dict[str, Any], Any]] = []
    if sampled_primary_mode:
        sampled_root = _sample_q_token(root_q, sampling_metadata)[0]
        root_backup_budget = min(
            config.max_children_per_parent - 1,
            max(config.max_nodes - config.max_depth, 0),
        )
        if batch_select:
            from remtp.raven_fast_path import batched_trunk_children
            root_item = batched_trunk_children(
                [root_q], [sampled_root], depth=1, backup_budget=root_backup_budget, config=config,
            )[0]
            root_ids, root_selection = root_item.candidate_ids, root_item
        else:
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
    # One device transfer for all root candidates; the values are the same
    # float32 entries that float(root_q[int(token)]) read one by one.
    if batch_select:
        batched_pending.append((construction[0], root_item))
        root_probabilities = root_item.candidate_probs
    else:
        root_probabilities = root_q[root_ids].tolist()
    for rank, (token, probability) in enumerate(zip(root_ids, root_probabilities)):
        path = (rank,)
        token_by_path[path] = token
        q_by_path[path] = root_q
        hidden_by_path[path] = root_hidden
        log_q_sum[path] = math.log(max(probability, 1e-30))

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
        if batch_select:
            from remtp.raven_fast_path import batched_trunk_children
            # Same draws in the same per-generator order as the per-parent loop.
            sampled_tokens = [
                _sample_q_token(parent_q[row], sampling_metadata)[0]
                if parent_path == (0,) * (depth - 1)
                else _sample_branch_q_token(
                    parent_q[row], sampling_metadata,
                    parent_path=parent_path, proposal_round=proposal_round,
                )[0]
                for row, parent_path in enumerate(parent_paths)
            ]
            level_items = batched_trunk_children(
                [parent_q[row] for row in range(len(parent_paths))], sampled_tokens,
                depth=depth, backup_budget=config.max_children_per_parent - 1, config=config,
            )
        for row, parent_path in enumerate(parent_paths):
            if batch_select:
                is_primary_parent = parent_path == (0,) * (depth - 1)
                selection = level_items[row]
                candidate_ids = selection.candidate_ids
                proposal_role = (
                    "sampled-primary-with-backups"
                    if is_primary_parent
                    else "sampled-recovery-branch-with-backups"
                )
            elif sampled_primary_mode:
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
            if batch_select:
                batched_pending.append((construction[-1], selection))
                candidate_probabilities = selection.candidate_probs
            else:
                candidate_probabilities = parent_q[row, candidate_ids].tolist()
            for local_rank, (token, probability) in enumerate(
                zip(candidate_ids, candidate_probabilities)
            ):
                path = parent_path + (local_rank,)
                token_probability = max(probability, 1e-30)
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

    if batched_pending:
        from remtp.raven_fast_path import resolve_batched_counts
        resolve_batched_counts([item for _, item in batched_pending])
        for row, item in batched_pending:
            row["eligible"] = item.candidate_count_before_budget
            row["threshold_eligible"] = item.threshold_candidate_count

    # Keep original proposal/budget decisions; rematerialize KV with new indices.
    unpruned_node_count = topology.node_count
    if config.prune_anchor_descendants:
        topology, _ = anchor_reusable_topology(topology)
        fixed._RUNTIME.topology = topology

    # Materialize the final dynamic topology and all branch-local draft KV.
    current_paths = list(topology.choices)
    if (
        os.getenv("REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV", "0") == "1"
        and not config.prune_anchor_descendants
    ):
        # The outputs of this pass are discarded and its only side effect is
        # draft-layer KV for the tree nodes.  Every cumulative MTP pass
        # rewrites its own node slots before attending, and the next round's
        # first MTP pass rewrites slots root+1..root+nodes+1 from the target
        # hidden states, so nothing ever reads that KV.  Skip the forward and
        # tell the path commit there is no draft KV to move.
        fixed._RUNTIME.draft_slots = None
    else:
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
        "unpruned_node_count": unpruned_node_count,
        "pruned_descendants": unpruned_node_count - topology.node_count,
        "max_depth": topology.max_depth,
        "branching": construction,
    }

    from remtp import raven_fast_path as fast

    levels = [topology.level_nodes(depth) for depth in range(1, topology.max_depth + 1)]
    if fast.ASYNC_HOST and all(
        nodes == tuple(range(nodes[0], nodes[-1] + 1)) for nodes in levels if nodes
    ):
        # BFS levels are contiguous: a slice copy instead of a host-list index
        # (which copies the index to the device synchronously).
        return [ids[nodes[0]:nodes[-1] + 1].clone().view(1, -1) for nodes in levels if nodes]
    return [ids[list(nodes)].view(1, -1) for nodes in levels if nodes]


_BATCH_SELECT = False
_GPU_BUILD = False


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
        from remtp.generation_guard import repeated_suffix
        history = getattr(fixed._RUNTIME, "generated_history", ())
        repetition = os.getenv("REMTP_DYNAMIC_TREE_REPEAT_GUARD", "0") == "1" and repeated_suffix(history)
        protected = frozenset(n.index for n in args[0].nodes
                              if n.path == (0,) * n.depth) if repetition else frozenset()
        from remtp.decision_trace import active_trace
        trace = active_trace()
        from remtp import device_verify
        if device_verify.enabled() and trace is None:
            # Same rule, one host read per round, fixed randomness (2026-11-15).
            device_result = device_verify.residual_hit_tree_verify_device(
                *args, **kwargs, config=config, protected_node_indices=protected)
            if device_result is not None:
                return device_result
        if os.getenv("REMTP_DYNAMIC_TREE_FAST_VERIFY", "0") == "1" and trace is None:
            from remtp.raven_fast_path import residual_hit_tree_verify_fast
            fast_result = residual_hit_tree_verify_fast(
                *args,
                **kwargs,
                config=config,
                protected_node_indices=protected,
            )
            if fast_result is not None:
                return fast_result
        return residual_hit_tree_verify(
            *args,
            **kwargs,
            config=config,
            protected_node_indices=protected,
            lazy_residual=os.getenv("REMTP_DYNAMIC_TREE_LAZY_RESIDUAL", "0") == "1",
            decision_trace=trace,
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


def _anchor_batched_logits(self: Any, hidden_states: torch.Tensor) -> Any:
    original = getattr(_anchor_batched_logits, "_dynamic_original")
    if hidden_states.shape[0] != 1:
        return original(self, hidden_states)
    logits = original(self, hidden_states.expand(3, -1).contiguous())
    return None if logits is None else logits[:1]


def _install_fast_select(config: Any) -> None:
    """Rebind the sampled-primary proposal helpers to the two-transfer versions."""
    global _sample_q_token, _sample_branch_q_token, _cactus_trunk_children
    from remtp import fixed6_vllm as fixed
    from remtp import raven_fast_path as fast

    if config.support_mode not in fast.SAMPLED_PRIMARY_MODES or config.anchor_only_proposal:
        raise ValueError(
            "REMTP_DYNAMIC_TREE_FAST_SELECT requires a sampled-primary tree mode "
            "without anchor-only proposal"
        )
    _sample_q_token = fast.sample_q_token_fast
    _sample_branch_q_token = fast.sample_branch_q_token_fast
    _cactus_trunk_children = fast.cactus_trunk_children_fast
    fixed._temperature_probs = fast.temperature_probs_fast
    print(
        "[ReMTP][DynamicTree] fast candidate selection: two device transfers per parent",
        flush=True,
    )


def install_mimo_dynamic_tree() -> None:
    global _INSTALLED, _ORIGINAL_TREE_RUN, _BATCH_SELECT, _GPU_BUILD
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
    if os.getenv("REMTP_DYNAMIC_TREE_FAST_SELECT", "0") == "1":
        _install_fast_select(config)
    if os.getenv("REMTP_DYNAMIC_TREE_BATCH_SELECT", "0") == "1":
        if os.getenv("REMTP_DYNAMIC_TREE_FAST_SELECT", "0") != "1":
            raise ValueError("REMTP_DYNAMIC_TREE_BATCH_SELECT requires REMTP_DYNAMIC_TREE_FAST_SELECT=1")
        _BATCH_SELECT = True
        print("[ReMTP][DynamicTree] batched candidate selection: one transfer per tree level", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_FAST_VERIFY", "0") == "1":
        print("[ReMTP][DynamicTree] fast residual-hit verification: batched transfers", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_FAST_COMMIT", "0") == "1":
        print("[ReMTP][DynamicTree] fused KV path commit across layers", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_FAST_COMMIT", "0") == "2":
        print("[ReMTP][DynamicTree] in-place ordered KV path commit, identity moves skipped", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_FAST_METADATA", "0") == "1":
        print("[ReMTP][DynamicTree] cached host-built tree masks and position offsets", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV", "0") == "1":
        if config.prune_anchor_descendants:
            raise RuntimeError("REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV is only validated without anchor pruning")
        print("[ReMTP][DynamicTree] dead draft KV skipped: no final tree rematerialization, no draft KV commit", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_PATH_ORDERED_FIRST_PASS", "0") == "1":
        print("[ReMTP][DynamicTree] first MTP pass reordered to the accepted path (changes proposals)", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_GPU_BUILD", "0") == "1":
        from remtp import gpu_tree
        ok, reason = gpu_tree.supported(config)
        if not ok:
            raise ValueError(f"REMTP_DYNAMIC_TREE_GPU_BUILD does not support this configuration ({reason})")
        _GPU_BUILD = True
        check = os.getenv("REMTP_DYNAMIC_TREE_GPU_BUILD_CHECK", "")
        print("[ReMTP][DynamicTree] device-side tree construction: one host read per proposal"
              + (f" + host check -> {check}" if check else ""), flush=True)
    from remtp import raven_fast_path as _fast
    if _fast.VERIFY_PRELAUNCH:
        print("[ReMTP][DynamicTree] trunk verification prelaunched: one host read, RNG offsets realigned", flush=True)
    if _fast.ASYNC_HOST:
        print("[ReMTP][DynamicTree] async host reads: pinned non-blocking copies, no scalar read-backs", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_GPU_BUILD_FUSED", "0") in {"exact", "fast"}:
        print("[ReMTP][DynamicTree] fused device tree decisions: "
              f"{os.getenv('REMTP_DYNAMIC_TREE_GPU_BUILD_FUSED')} row statistics, one Triton kernel per level", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_FAST_COMMIT", "0") == "3":
        print("[ReMTP][DynamicTree] one-launch Triton KV path commit, device slot indices", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_DEVICE_VERIFY", "0") == "1":
        print("[ReMTP][DynamicTree] device residual-hit verification: one host read, fixed randomness per round "
              "(changes samples, not the distribution)", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_VERIFY_FUSED", "0") == "1":
        print("[ReMTP][DynamicTree] fused async verification: 4 Triton kernels per round", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_BATCHED_BRANCH_NOISE", "0") == "1":
        print("[ReMTP][DynamicTree] batched branch noise: one generator per round (changes branch samples)", flush=True)
    if os.getenv("REMTP_DYNAMIC_TREE_ASYNC", "0") == "1":
        print("[ReMTP][DynamicTree] async scheduling: rounds stay on the device, fixed 9-slot capacity, "
              "padded slots masked", flush=True)
    if os.getenv("REMTP_TREE_CUDAGRAPH", "0") == "1":
        from remtp.tree_cudagraph import install_tree_cudagraph

        install_tree_cudagraph()
    elif os.getenv("REMTP_TREE_HOST_METADATA", "0") == "1":
        from remtp.tree_cudagraph import install_tree_host_metadata

        install_tree_host_metadata()

    eagle = importlib.import_module("vllm.v1.spec_decode.eagle")
    runner_module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    rejection = importlib.import_module("vllm.v1.sample.rejection_sampler")
    mimo = importlib.import_module("vllm.model_executor.models.mimo")
    if config.anchor_only_proposal and "REMTP_HEAD_BATCH_ROWS" not in os.environ:
        mtp = importlib.import_module("vllm.model_executor.models.mimo_mtp")
        _anchor_batched_logits._dynamic_original = mtp.MiMoMTP.compute_logits
        mtp.MiMoMTP.compute_logits = _anchor_batched_logits

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
    if config.anchor_only_proposal:
        proposal_name = "sampled-q-primary+direct-siblings+batched-head"
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
        "relax_depth_weights="
        f"{list(config.relaxation_depth_weights) or ['uniform']} "
        f"adaptive_relaxation={int(config.adaptive_relaxation)} "
        f"lazy_residual={int(os.getenv('REMTP_DYNAMIC_TREE_LAZY_RESIDUAL', '0') == '1')} "
        f"adaptive_mix={config.adaptive_relaxation_mix:g} "
        f"adaptive_max_ratio={config.adaptive_relaxation_max_ratio:g} "
        f"relax_max_probability_ratio={config.relaxation_max_probability_ratio:g} "
        f"adaptive_target_power={config.adaptive_relaxation_target_power:g} "
        f"adaptive_steps={config.adaptive_relaxation_steps} "
        f"risk_policy={config.relaxation_risk_policy} "
        f"target_gate_min_p={config.relaxation_min_target_probability:g} "
        f"target_gate_min_ratio={config.relaxation_min_target_ratio:g} "
        f"adaptive_pool={config.adaptive_relaxation_pool} "
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
