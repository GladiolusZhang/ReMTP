"""Experimental vLLM 0.18 adapter for the Fixed-6 micro-tree.

This adapter is intentionally narrow: Qwen3.5 MTP, TP=1, one request, eager
execution, and static root-branching topologies.  It patches four explicit
boundaries that stock vLLM 0.18 does not connect:

* uneven native-MTP tree proposal with full conditional Q rows;
* target TreeAttention positions and parent-logit mapping;
* Qwen3.5 GDN common-root update followed by branch-major recurrent updates;
* strict multi-candidate verification and selected-path cache compaction.

Merely setting ``speculative_token_tree`` is not sufficient in this vLLM
version: the stock sampler still interprets the flattened nodes as a chain and
the stock GDN metadata advances them recurrently as one sequence.  The
``6-chain`` control uses vLLM's native GDN path. Non-chain Qwen3.5 topologies
remain opt-in until selected-branch GDN state matches independent execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from remtp.fixed6_microtree import (
    FIXED_NODE_BUDGET,
    Fixed6Topology,
    TreeVerifyResult,
    get_topology,
    normalize_distribution,
    sample_categorical,
    sample_without_replacement,
    strict_tree_verify,
)


@dataclass
class _Runtime:
    topology: Fixed6Topology
    draft_probs: torch.Tensor | None = None
    draft_token_ids: torch.Tensor | None = None
    selected_nodes: tuple[int, ...] = ()
    selected_branch: int | None = None
    terminal: str | None = None
    target_tree_active: bool = False
    target_slots: torch.Tensor | None = None
    draft_slots: torch.Tensor | None = None
    target_gdn_state_rows: dict[str, torch.Tensor] | None = None
    target_top1_ids: tuple[int, ...] = ()
    target_probs_by_input: torch.Tensor | None = None
    rounds: int = 0
    accepted_drafts_total: int = 0
    target_calls: int = 0
    target_nodes: int = 0
    target_forward_seconds: float = 0.0
    last_target_forward_seconds: float = 0.0
    gdn_debug_done: bool = False
    dynamic_construction: dict[str, Any] | None = None


def _initial_topology() -> Any:
    name = os.getenv("REMTP_TREE_TOPOLOGY", "6-chain")
    try:
        return get_topology(name)
    except ValueError:
        from remtp.binary_mtp_tree import get_mimo_tree_topology

        return get_mimo_tree_topology(name)


_RUNTIME = _Runtime(_initial_topology())
_INSTALLED = False
_TOKENIZER: Any | None = None
_TOKENIZER_FAILED = False


def _decode_tree_token(token_id: int) -> str:
    global _TOKENIZER, _TOKENIZER_FAILED
    if _TOKENIZER_FAILED:
        return f"<id:{token_id}>"
    if _TOKENIZER is None:
        model_path = os.getenv("REMTP_TREE_TOKENIZER")
        if not model_path:
            return f"<id:{token_id}>"
        try:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:
            _TOKENIZER_FAILED = True
            print(f"[ReMTP][MiMoTree] tokenizer trace disabled: {exc}", flush=True)
            return f"<id:{token_id}>"
    return str(
        _TOKENIZER.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _profile_target_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    original = getattr(_profile_target_forward, "_fixed6_original")
    profile = (
        _RUNTIME.target_tree_active
        and os.getenv("REMTP_TREE_PROFILE_TARGET") == "1"
    )
    if not profile:
        return original(self, *args, **kwargs)
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = original(self, *args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    _RUNTIME.last_target_forward_seconds = elapsed
    _RUNTIME.target_forward_seconds += elapsed
    return result


def topology_config_string(topology: Fixed6Topology) -> str:
    return str(list(topology.choices))


def _generator(sampling_metadata: Any, row: int = 0) -> torch.Generator | None:
    generators = getattr(sampling_metadata, "generators", None) or {}
    return generators.get(row)


def _temperature_probs(
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Convert raw model logits to the request distribution."""
    temperatures = sampling_metadata.temperature
    if temperatures is None:
        ids = logits.argmax(dim=-1)
        return torch.nn.functional.one_hot(
            ids, num_classes=logits.shape[-1]
        ).to(torch.float32)
    temperature = temperatures[0].to(device=logits.device, dtype=torch.float32)
    raw = logits.to(torch.float32)
    if float(temperature) < 1e-5:
        ids = raw.argmax(dim=-1)
        return torch.nn.functional.one_hot(
            ids, num_classes=raw.shape[-1]
        ).to(torch.float32)
    return torch.softmax(raw / temperature, dim=-1)


def _processed_target_probs(
    processed_logits: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    """Convert vLLM-processed target logits without scaling twice.

    ``apply_sampling_constraints`` has already divided stochastic rows by the
    request temperature and applied top-k/top-p.  Only a softmax remains.
    Greedy rows still need the deterministic one-hot representation expected
    by the tree verifier.
    """
    raw = processed_logits.to(torch.float32)
    if _is_greedy(sampling_metadata):
        ids = raw.argmax(dim=-1)
        return torch.nn.functional.one_hot(
            ids, num_classes=raw.shape[-1]
        ).to(torch.float32)
    return torch.softmax(raw, dim=-1)


def _is_greedy(sampling_metadata: Any) -> bool:
    temperatures = sampling_metadata.temperature
    if temperatures is None:
        return True
    return float(temperatures[0].to(dtype=torch.float32)) < 1e-5


def _sample_rows(
    probs: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    generator = _generator(sampling_metadata)
    return torch.tensor(
        [sample_categorical(row, generator) for row in probs],
        dtype=torch.int64,
        device=probs.device,
    )


def _tree_positions(
    root_positions: torch.Tensor,
    topology: Fixed6Topology,
    node_indices: list[int],
) -> torch.Tensor:
    offsets = torch.tensor(
        [topology.nodes[index].depth for index in node_indices],
        device=root_positions.device,
        dtype=root_positions.dtype,
    )
    if root_positions.ndim == 1:
        root = root_positions.reshape(-1)[0]
        return root + offsets
    root = root_positions[:, :1]
    return root + offsets.unsqueeze(0)


def _run_draft_tree_nodes(
    self: Any,
    *,
    node_ids: list[int],
    node_tokens: dict[int, torch.Tensor],
    node_hidden_inputs: dict[int, torch.Tensor],
    root_positions: torch.Tensor,
    base_common_metadata: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one cumulative MTP tree pass and return both hidden outputs."""
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backends.tree_attn import (
        TreeAttentionMetadataBuilder,
    )

    builder = self.draft_attn_groups[0].get_metadata_builder()
    if not isinstance(builder, TreeAttentionMetadataBuilder):
        raise RuntimeError(
            "Fixed-6 requires --attention-backend TREE_ATTN for the MTP layer"
        )
    count = len(node_ids)
    node_budget = len(_RUNTIME.topology.nodes)
    if count <= 0 or count > node_budget:
        raise RuntimeError("invalid cumulative MTP tree size")

    # The physical cache slots are unique BFS offsets. Logical RoPE positions
    # are branch depths; these two notions must not be conflated.
    common = replace(
        base_common_metadata,
        query_start_loc=count * self.arange[:2],
        seq_lens=base_common_metadata.seq_lens + count,
        num_actual_tokens=count,
        max_query_len=count,
    )
    attn_metadata = builder.build_for_drafting(
        common_attn_metadata=common,
        draft_index=max(_RUNTIME.topology.nodes[index].depth for index in node_ids),
    )
    attn_metadata.max_seq_len = min(attn_metadata.max_seq_len, self.max_model_len)

    flat_root = int(root_positions.reshape(-1)[0].item())
    physical_positions = torch.arange(
        flat_root + 1,
        flat_root + count + 1,
        device=attn_metadata.block_table.device,
        dtype=torch.int64,
    ).view(1, -1)
    block_size = builder.kv_cache_spec.block_size
    block_numbers = physical_positions // block_size
    block_ids = attn_metadata.block_table.gather(dim=1, index=block_numbers)
    slot_mapping = block_ids * block_size + physical_positions % block_size
    attn_metadata.slot_mapping = slot_mapping.reshape(-1)

    token_tensor = torch.stack([node_tokens[index] for index in node_ids]).to(
        device=self.input_ids.device, dtype=self.input_ids.dtype
    )
    hidden_tensor = torch.stack(
        [node_hidden_inputs[index] for index in node_ids], dim=0
    ).to(device=self.hidden_states.device, dtype=self.hidden_states.dtype)
    position_tensor = _tree_positions(root_positions, _RUNTIME.topology, node_ids)

    self.input_ids[:count] = token_tensor
    self.hidden_states[:count] = hidden_tensor
    self._set_positions(count, position_tensor)

    per_layer = {}
    for group in self.draft_attn_groups:
        for layer_name in group.layer_names:
            per_layer[layer_name] = attn_metadata
    cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(count)
    num_input_tokens = batch_desc.num_tokens
    with set_forward_context(
        per_layer,
        self.vllm_config,
        num_tokens=num_input_tokens,
        cudagraph_runtime_mode=cudagraph_mode,
        slot_mapping=self._get_slot_mapping(
            num_input_tokens, attn_metadata.slot_mapping
        ),
    ):
        model_output = self.model(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self._get_positions(num_input_tokens),
            hidden_states=self.hidden_states[:num_input_tokens],
            inputs_embeds=None,
        )
        if self.model_returns_tuple():
            last_hidden, hidden = model_output
        else:
            last_hidden = model_output
            hidden = model_output
    _RUNTIME.draft_slots = attn_metadata.slot_mapping[:count].clone()
    return last_hidden[:count], hidden[:count], attn_metadata.slot_mapping[:count]


def _fixed6_propose_tree(
    self: Any,
    batch_size: int,
    logits: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    common_attn_metadata: Any,
    slot_mappings: Any = None,
) -> list[torch.Tensor]:
    if batch_size != 1:
        raise RuntimeError("Fixed-6 currently requires --max-num-seqs 1")
    sampling_metadata = getattr(self, "_fixed6_sampling_metadata", None)
    if sampling_metadata is None:
        raise RuntimeError("Fixed-6 proposer lost sampling metadata")
    topology = _RUNTIME.topology
    root_count = len(topology.roots)
    root_distribution = _temperature_probs(logits[:1], sampling_metadata)[0]
    generator = _generator(sampling_metadata)
    if _is_greedy(sampling_metadata):
        root_tokens = torch.topk(
            logits[0].to(torch.float32), root_count, sorted=True
        ).indices
        root_q_rows = torch.nn.functional.one_hot(
            root_tokens, num_classes=logits.shape[-1]
        ).to(torch.float32)
    else:
        root_tokens, root_q_rows = sample_without_replacement(
            root_distribution, root_count, generator
        )

    node_tokens: dict[int, torch.Tensor] = {}
    node_q: dict[int, torch.Tensor] = {}
    node_hidden_inputs: dict[int, torch.Tensor] = {}
    root_hidden = hidden_states.reshape(batch_size, -1)[0]
    for offset, node_index in enumerate(topology.roots):
        node_tokens[node_index] = root_tokens[offset]
        node_q[node_index] = root_q_rows[offset]
        node_hidden_inputs[node_index] = root_hidden

    max_depth = topology.max_depth
    for depth in range(1, max_depth):
        current_nodes = [node.index for node in topology.nodes if node.depth <= depth]
        last_hidden, hidden, _ = _run_draft_tree_nodes(
            self,
            node_ids=current_nodes,
            node_tokens=node_tokens,
            node_hidden_inputs=node_hidden_inputs,
            root_positions=positions,
            base_common_metadata=common_attn_metadata,
        )
        position_by_node = {node: pos for pos, node in enumerate(current_nodes)}
        parents = [
            node
            for node in topology.nodes
            if node.depth == depth and node.index in position_by_node
            and any(child.parent == node.index for child in topology.nodes)
        ]
        if not parents:
            continue
        parent_rows = torch.stack(
            [last_hidden[position_by_node[parent.index]] for parent in parents]
        )
        probs = _temperature_probs(self.model.compute_logits(parent_rows), sampling_metadata)
        for row, parent in enumerate(parents):
            children = [node for node in topology.nodes if node.parent == parent.index]
            child_count = len(children)
            if child_count == 0:
                continue
            if _is_greedy(sampling_metadata):
                child_tokens = torch.topk(
                    probs[row], child_count, sorted=True
                ).indices
                child_q_rows = torch.nn.functional.one_hot(
                    child_tokens, num_classes=probs.shape[-1]
                ).to(torch.float32)
            else:
                child_tokens, child_q_rows = sample_without_replacement(
                    probs[row], child_count, generator
                )
            for child_offset, child in enumerate(children):
                node_tokens[child.index] = child_tokens[child_offset]
                node_q[child.index] = child_q_rows[child_offset]
                node_hidden_inputs[child.index] = hidden[
                    position_by_node[parent.index]
                ]

    node_budget = len(topology.nodes)
    if set(node_tokens) != set(range(node_budget)):
        raise RuntimeError(
            f"MTP tree proposal did not materialize all {node_budget} nodes"
        )

    # A final pass materializes every leaf's MTP KV state. It is cheap (one
    # MTP layer), makes selected-path cache compaction explicit, and is included
    # in the reported tree drafting latency.
    _run_draft_tree_nodes(
        self,
        node_ids=list(range(node_budget)),
        node_tokens=node_tokens,
        node_hidden_inputs=node_hidden_inputs,
        root_positions=positions,
        base_common_metadata=common_attn_metadata,
    )

    ids = torch.stack([node_tokens[index] for index in range(node_budget)]).to(
        torch.int32
    )
    probs = torch.stack([node_q[index] for index in range(node_budget)]).contiguous()
    _RUNTIME.draft_token_ids = ids.to(torch.int64)
    _RUNTIME.draft_probs = probs

    by_level: list[torch.Tensor] = []
    for depth in range(1, max_depth + 1):
        level = [node.index for node in topology.nodes if node.depth == depth]
        if level:
            by_level.append(ids[level].view(1, -1))
    return by_level


def _propose_with_sampling_metadata(self: Any, *args: Any, **kwargs: Any) -> Any:
    original = getattr(_propose_with_sampling_metadata, "_fixed6_original")
    sampling_metadata = kwargs.get("sampling_metadata")
    if sampling_metadata is None:
        raise RuntimeError("Fixed-6 requires keyword sampling_metadata")
    self._fixed6_sampling_metadata = sampling_metadata
    try:
        return original(self, *args, **kwargs)
    finally:
        del self._fixed6_sampling_metadata


def _selected_target_hidden_row() -> int:
    """Return the target-forward row preceding the next proposal token.

    Target row zero is the common anchor and draft node ``i`` is input row
    ``i + 1``.  vLLM's chain preparation derives this row from the number of
    accepted drafts, which is wrong for a BFS tree path such as ``0, 2, 5``.
    """
    selected = _RUNTIME.selected_nodes
    return 0 if not selected else int(selected[-1]) + 1


def _prepare_selected_path_inputs(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Make the next MTP root consume the selected tree leaf hidden state."""
    original = getattr(_prepare_selected_path_inputs, "_fixed6_original")
    common, token_indices, rejected = original(self, *args, **kwargs)
    if _RUNTIME.target_tree_active:
        if token_indices.numel() != 1:
            raise RuntimeError("dynamic MTP tree requires one request")
        row = _selected_target_hidden_row()
        if row >= int(common.num_actual_tokens):
            raise RuntimeError(
                f"selected target hidden row {row} exceeds "
                f"{common.num_actual_tokens} inputs"
            )
        token_indices[0] = row
    return common, token_indices, rejected


def _prepare_target_tree_inputs(self: Any, *args: Any, **kwargs: Any) -> Any:
    original = getattr(_prepare_target_tree_inputs, "_fixed6_original")
    result = original(self, *args, **kwargs)
    _, metadata = result
    _RUNTIME.target_tree_active = metadata is not None
    if metadata is None:
        return result
    node_budget = len(_RUNTIME.topology.nodes)
    if (
        len(metadata.num_draft_tokens) != 1
        or metadata.num_draft_tokens[0] != node_budget
    ):
        raise RuntimeError(
            f"tree target input must contain exactly {node_budget} nodes"
        )

    input_count = node_budget + 1
    positions = self._get_positions(input_count)
    offsets = torch.tensor(
        (0,) + _RUNTIME.topology.node_position_offsets,
        dtype=positions.dtype,
        device=positions.device,
    )
    if positions.ndim == 1:
        positions[:input_count] = positions[0].clone() + offsets
    else:
        positions[:, :input_count] = positions[:, :1].clone() + offsets.unsqueeze(0)
    metadata.fixed6_topology = _RUNTIME.topology.name
    metadata.fixed6_parent_logit_rows = _RUNTIME.topology.parent_logit_rows
    return result


def _capture_target_slots(self: Any, *args: Any, **kwargs: Any) -> Any:
    original = getattr(_capture_target_slots, "_fixed6_original")
    result = original(self, *args, **kwargs)
    if _RUNTIME.target_tree_active:
        by_group = result[0]
        attention_gid = self._get_attention_kv_cache_gid()
        slots = by_group[attention_gid]
        input_count = len(_RUNTIME.topology.nodes) + 1
        if slots.numel() >= input_count:
            _RUNTIME.target_slots = slots[:input_count].clone()
    return result


def _fixed6_gdn_forward_core(
    self: Any,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> Any:
    original = getattr(_fixed6_gdn_forward_core, "_fixed6_original")
    if not _RUNTIME.target_tree_active or mixed_qkv.shape[0] < 7:
        return original(self, mixed_qkv, b, a, core_attn_out)

    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    context = get_forward_context()
    metadata_by_layer = context.attn_metadata
    if not isinstance(metadata_by_layer, dict):
        return original(self, mixed_qkv, b, a, core_attn_out)
    metadata = metadata_by_layer.get(self.prefix)
    if not isinstance(metadata, GDNAttentionMetadata):
        return original(self, mixed_qkv, b, a, core_attn_out)
    if metadata.spec_state_indices_tensor is None:
        return original(self, mixed_qkv, b, a, core_attn_out)
    state_row = metadata.spec_state_indices_tensor[0, :7]
    if state_row.numel() != 7:
        raise RuntimeError("Fixed-6 GDN requires one root plus six state blocks")
    if _RUNTIME.target_gdn_state_rows is None:
        _RUNTIME.target_gdn_state_rows = {}
    _RUNTIME.target_gdn_state_rows[self.prefix] = state_row.clone()

    # The chain is the exact control, so keep vLLM's native recurrent kernel
    # and metadata untouched.  Any difference here would invalidate both the
    # state reference and the tree-coverage attribution.
    if _RUNTIME.topology.name == "6-chain":
        return original(self, mixed_qkv, b, a, core_attn_out)

    debug_reference: tuple[torch.Tensor, list[torch.Tensor]] | None = None
    debug_enabled = (
        os.getenv("REMTP_TREE_DEBUG_COMPARE_GDN") == "1"
        and not _RUNTIME.gdn_debug_done
    )
    if debug_enabled:
        # Compare the split root/branch execution against vLLM's native chain
        # kernel on exactly the same inputs and state rows.  Only seven rows are
        # cloned, so this diagnostic does not duplicate the full cache.
        cache = self.kv_cache[context.virtual_engine]
        rows = state_row.to(torch.int64)
        before = [part.index_select(0, rows).clone() for part in cache]
        reference_out = torch.empty_like(core_attn_out)
        original(self, mixed_qkv, b, a, reference_out)
        reference_cache = [part.index_select(0, rows).clone() for part in cache]
        for part, saved in zip(cache, before, strict=True):
            part.index_copy_(0, rows, saved)
        debug_reference = (reference_out.clone(), reference_cache)

    device = mixed_qkv.device
    original_metadata = metadata_by_layer[self.prefix]
    cache = self.kv_cache[context.virtual_engine]
    canonical = state_row[0].to(torch.int64)
    accepted_before = metadata.num_accepted_tokens
    if accepted_before is None:
        raise RuntimeError("Fixed-6 GDN lost previous accepted-token metadata")
    previous_offset = max(int(accepted_before[0].item()) - 1, 0)
    if previous_offset >= state_row.numel():
        raise RuntimeError(
            f"invalid carried GDN offset {previous_offset} for seven state rows"
        )

    # vLLM's speculative convolution cache uses one expanded state row. The
    # accepted count selects a width-1 history window inside that row. Its SSM
    # cache instead stores one state row per speculative position. Keep these
    # layouts distinct while cloning the common prefix for every branch.
    initial_temporal = state_row[previous_offset].to(torch.int64)
    conv_history = int(self.conv_kernel_size) - 1
    initial_conv = cache[0][
        canonical, previous_offset : previous_offset + conv_history
    ].clone()
    if initial_conv.shape[0] != conv_history:
        raise RuntimeError("Fixed-6 GDN convolution history window is truncated")
    initial_ssm = cache[1][initial_temporal].clone()

    # Advance the common anchor exactly once and retain it in the canonical
    # row. This is the state that must survive a zero-draft-accept round. Each
    # branch then starts from an isolated clone of that anchor state.
    cache[0][canonical].zero_()
    cache[0][canonical, :conv_history].copy_(initial_conv)
    cache[1][canonical].copy_(initial_ssm)
    one_query = torch.tensor([0, 1], dtype=torch.int32, device=device)
    anchor_metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
        non_spec_query_start_loc=one_query,
        non_spec_state_indices_tensor=canonical.view(1),
    )
    try:
        metadata_by_layer[self.prefix] = anchor_metadata
        anchor_buffer = torch.empty_like(core_attn_out[:1])
        original(
            self,
            mixed_qkv[:1],
            b[:1],
            a[:1],
            anchor_buffer,
        )
        core_attn_out[0] = anchor_buffer[0]

        for branch in _RUNTIME.topology.branch_nodes:
            parent_state = canonical
            for node_index in branch:
                node_state = state_row[node_index + 1].to(torch.int64)
                cache[0][node_state].zero_()
                cache[0][node_state, :conv_history].copy_(
                    cache[0][parent_state, :conv_history]
                )
                cache[1][node_state].copy_(cache[1][parent_state])

                node_metadata = GDNAttentionMetadata(
                    num_prefills=0,
                    num_prefill_tokens=0,
                    num_decodes=1,
                    num_decode_tokens=1,
                    num_spec_decodes=0,
                    num_spec_decode_tokens=0,
                    num_actual_tokens=1,
                    non_spec_query_start_loc=one_query,
                    non_spec_state_indices_tensor=node_state.view(1),
                )
                metadata_by_layer[self.prefix] = node_metadata
                node_out = torch.empty_like(core_attn_out[:1])
                input_row = node_index + 1
                original(
                    self,
                    mixed_qkv[input_row : input_row + 1],
                    b[input_row : input_row + 1],
                    a[input_row : input_row + 1],
                    node_out,
                )
                core_attn_out[input_row] = node_out[0]
                parent_state = node_state
    finally:
        metadata_by_layer[self.prefix] = original_metadata
    if debug_reference is not None:
        reference_out, reference_cache = debug_reference
        cache = self.kv_cache[context.virtual_engine]
        rows = state_row.to(torch.int64)
        output_diff = float(
            (core_attn_out[:7].float() - reference_out[:7].float()).abs().max()
        )
        cache_diffs = [
            float(
                (
                    part.index_select(0, rows).float() - expected.float()
                ).abs().max()
            )
            for part, expected in zip(cache, reference_cache, strict=True)
        ]
        first_root = _RUNTIME.topology.roots[0]
        root_state = int(state_row[first_root + 1])
        canonical_state = int(state_row[0])
        prefix_width = min(int(self.conv_kernel_size), cache[0].shape[1] - 1)
        conv_prefix_diff = float(
            (
                cache[0][root_state, 1 : 1 + prefix_width].float()
                - reference_cache[0][0, 1 : 1 + prefix_width].float()
            )
            .abs()
            .max()
        )
        temporal_root_diff = float(
            (
                cache[1][root_state].float()
                - reference_cache[1][1].float()
            )
            .abs()
            .max()
        )
        prefix_output_diff = float(
            (
                core_attn_out[:2].float() - reference_out[:2].float()
            ).abs().max()
        )
        print(
            "[Fixed-6][GDN reference] "
            f"layer={self.prefix} output_max_abs={output_diff:.6g} "
            f"cache_max_abs={cache_diffs} "
            f"root_output_max_abs={prefix_output_diff:.6g} "
            f"root_conv_max_abs={conv_prefix_diff:.6g} "
            f"root_temporal_max_abs={temporal_root_diff:.6g} "
            f"previous_accepted={accepted_before[:1].tolist()} "
            f"previous_offset={previous_offset} "
            f"cache_shapes={[tuple(part.shape) for part in cache]}",
            flush=True,
        )
        _RUNTIME.gdn_debug_done = True
    return None


def _strict_tree_sampler(self: Any, metadata: Any, draft_probs: Any, logits: torch.Tensor, sampling_metadata: Any) -> Any:
    if not hasattr(metadata, "fixed6_topology"):
        original = getattr(_strict_tree_sampler, "_fixed6_original")
        return original(self, metadata, draft_probs, logits, sampling_metadata)
    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.rejection_sampler import apply_sampling_constraints

    if sampling_metadata.max_num_logprobs is not None:
        raise RuntimeError("Fixed-6 smoke/bench does not support logprobs")
    if not sampling_metadata.no_penalties or sampling_metadata.bad_words_token_ids:
        raise RuntimeError("Fixed-6 first version requires no penalties/bad words")
    if _RUNTIME.draft_probs is None or _RUNTIME.draft_token_ids is None:
        raise RuntimeError("Fixed-6 target verification has no cached MTP tree")
    node_budget = len(_RUNTIME.topology.nodes)
    input_count = node_budget + 1
    if logits.shape[0] != input_count:
        raise RuntimeError(
            f"tree expects {input_count} target logit rows, got {logits.shape[0]}"
        )

    target_logits = logits.to(torch.float32).clone()
    cu = torch.tensor([input_count], dtype=torch.int32, device=logits.device)
    target_logits = apply_sampling_constraints(target_logits, cu, sampling_metadata)
    # ``apply_sampling_constraints`` already applied temperature/top-k/top-p.
    # Applying ``_temperature_probs`` here used to divide by temperature a
    # second time (T=0.6 became an effective T=0.36), making tree verification
    # spuriously sharp and incomparable with vLLM's chain rejection sampler.
    target_probs = _processed_target_probs(target_logits, sampling_metadata)
    _RUNTIME.target_probs_by_input = target_probs.detach()
    _RUNTIME.target_top1_ids = tuple(
        int(value) for value in target_probs.argmax(dim=-1).tolist()
    )
    if os.getenv("REMTP_TREE_FORCE_ROOT_REJECT") == "1":
        correction = sample_categorical(target_probs[0], _generator(sampling_metadata))
        result = TreeVerifyResult(
            output_token_ids=(correction,),
            accepted_node_indices=(),
            selected_branch=None,
            terminal="correction",
        )
    else:
        result = strict_tree_verify(
            _RUNTIME.topology,
            _RUNTIME.draft_token_ids.to(device=logits.device),
            _RUNTIME.draft_probs.to(device=logits.device),
            target_probs,
            _generator(sampling_metadata),
        )
    debug_limit = int(os.getenv("REMTP_TREE_DEBUG_MAX_ACCEPTED", "0"))
    if debug_limit > 0 and result.accepted_drafts > debug_limit:
        selected = result.accepted_node_indices[:debug_limit]
        row = selected[-1] + 1
        correction = sample_categorical(
            target_probs[row], _generator(sampling_metadata)
        )
        result = TreeVerifyResult(
            output_token_ids=tuple(
                int(_RUNTIME.draft_token_ids[index]) for index in selected
            )
            + (correction,),
            accepted_node_indices=selected,
            selected_branch=result.selected_branch,
            terminal="correction",
        )
    _RUNTIME.selected_nodes = result.accepted_node_indices
    _RUNTIME.selected_branch = result.selected_branch
    _RUNTIME.terminal = result.terminal
    _RUNTIME.rounds += 1
    _RUNTIME.accepted_drafts_total += result.accepted_drafts
    _RUNTIME.target_calls += 1
    _RUNTIME.target_nodes += node_budget

    output = torch.full(
        (1, metadata.max_spec_len + 1),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    output[0, : len(result.output_token_ids)] = torch.tensor(
        result.output_token_ids, dtype=torch.int32, device=logits.device
    )
    _append_round_audit(result)
    return SamplerOutput(sampled_token_ids=output, logprobs_tensors=None)


def _copy_cache_slots(module: Any, sources: torch.Tensor, destinations: torch.Tensor) -> None:
    cache = module.kv_cache[0]
    if cache.numel() == 0 or sources.numel() == 0:
        return
    if cache.ndim < 3 or cache.shape[0] != 2:
        raise RuntimeError(f"unexpected attention KV cache shape {tuple(cache.shape)}")
    flattened = cache.flatten(1, 2)
    temp = flattened[:, sources.to(torch.int64)].clone()
    flattened[:, destinations.to(torch.int64)] = temp


def _compact_selected_path(self: Any, output_token_ids: torch.Tensor, scheduler_output: Any) -> Any:
    original = getattr(_compact_selected_path, "_fixed6_original")
    if not _RUNTIME.target_tree_active:
        return original(self, output_token_ids, scheduler_output)
    if _RUNTIME.topology.name == "6-chain":
        _RUNTIME.target_gdn_state_rows = None
        return original(self, output_token_ids, scheduler_output)
    selected = _RUNTIME.selected_nodes
    if selected and _RUNTIME.target_slots is not None:
        from vllm.model_executor.layers.attention.attention import Attention

        source = torch.tensor(
            [int(_RUNTIME.target_slots[index + 1]) for index in selected],
            device=_RUNTIME.target_slots.device,
        )
        destination = _RUNTIME.target_slots[1 : 1 + len(selected)]
        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, Attention):
                _copy_cache_slots(module, source, destination)

        if _RUNTIME.draft_slots is not None and hasattr(self, "drafter"):
            draft_source = torch.tensor(
                [int(_RUNTIME.draft_slots[index]) for index in selected],
                device=_RUNTIME.draft_slots.device,
            )
            draft_destination = _RUNTIME.draft_slots[: len(selected)]
            for module in self.drafter.model.modules():
                if isinstance(module, Attention) and module.kv_cache:
                    _copy_cache_slots(module, draft_source, draft_destination)

    # The correctness-reference path stores a complete compact conv/SSM state
    # at every node. Commit only the selected leaf into row zero.
    state_rows = _RUNTIME.target_gdn_state_rows or {}
    if selected:
        assert _RUNTIME.selected_branch is not None
        for module in self.compilation_config.static_forward_context.values():
            prefix = getattr(module, "prefix", None)
            state_row = state_rows.get(prefix)
            if state_row is None or not getattr(module, "kv_cache", None):
                continue
            cache = module.kv_cache[0]
            if not isinstance(cache, (list, tuple)) or len(cache) != 2:
                continue
            canonical = int(state_row[0])
            history = int(module.conv_kernel_size) - 1
            leaf_state = int(state_row[selected[-1] + 1])
            final_conv = cache[0][
                leaf_state, :history
            ].clone()
            if final_conv.shape[0] != history:
                raise RuntimeError("selected GDN conv window is truncated")
            cache[0][canonical].zero_()
            cache[0][canonical, :history].copy_(final_conv)
            cache[1][canonical].copy_(cache[1][leaf_state])

    _RUNTIME.target_gdn_state_rows = None
    # The target scheduler still sees every committed output token. Only the
    # hybrid-state postprocessor receives a single-token view, because row zero
    # already contains the exact compact state after the selected path (or the
    # common anchor when no draft was accepted).
    return original(self, output_token_ids[:, :1], scheduler_output)


def _append_round_audit(result: Any) -> None:
    path = os.getenv("REMTP_TREE_AUDIT_JSONL")
    trace_enabled = os.getenv("REMTP_TREE_TRACE", "0") == "1"
    if not path and not trace_enabled:
        return
    from remtp.tree_audit import build_node_records, format_round_trace

    if (
        _RUNTIME.draft_token_ids is None
        or _RUNTIME.draft_probs is None
        or _RUNTIME.target_probs_by_input is None
    ):
        return
    diagnostics = getattr(result, "node_diagnostics", ())
    by_node = {int(row["node"]): row for row in diagnostics}
    # The readable audit intentionally gathers full P/Q values and decodes
    # tokens.  Formal throughput runs use the lightweight mode below so that
    # this observability work is not charged to the decoding method.
    detailed_audit = trace_enabled or os.getenv("REMTP_TREE_AUDIT_DETAIL", "1") == "1"
    if detailed_audit:
        nodes = build_node_records(
            _RUNTIME.topology,
            result,
            _RUNTIME.draft_token_ids,
            _RUNTIME.draft_probs,
            _RUNTIME.target_probs_by_input,
            _decode_tree_token,
        )
        accepted_tokens = [
            {
                "id": int(_RUNTIME.draft_token_ids[index]),
                "text": _decode_tree_token(int(_RUNTIME.draft_token_ids[index])),
            }
            for index in result.accepted_node_indices
        ]
        output_tokens = [
            {"id": int(token), "text": _decode_tree_token(int(token))}
            for token in result.output_token_ids
        ]
        draft_token_ids = [int(value) for value in _RUNTIME.draft_token_ids.tolist()]
    else:
        status_by_node = dict(result.evaluated_node_statuses)
        nodes = [
            {
                "node": node.index,
                "parent": node.parent,
                "branch": node.branch,
                "depth": node.depth,
                "path": list(node.path),
                "status": status_by_node.get(node.index, "NOT_VISITED"),
                **by_node.get(node.index, {}),
            }
            for node in _RUNTIME.topology.nodes
        ]
        accepted_tokens = []
        output_tokens = []
        draft_token_ids = []
    mean_depth = _RUNTIME.accepted_drafts_total / max(_RUNTIME.rounds, 1)
    record = {
        "version": 2,
        "topology": _RUNTIME.topology.name,
        "round": _RUNTIME.rounds,
        "verification_mode": os.getenv("REMTP_TREE_VERIFY_MODE", "strict"),
        "target_validation_nodes": len(_RUNTIME.topology.nodes),
        "target_forward_calls": 1,
        "target_forward_seconds": _RUNTIME.last_target_forward_seconds,
        "accepted_drafts": result.accepted_drafts,
        "rolling_mean_accepted_depth": mean_depth,
        "rolling_mean_acceptance_length": mean_depth + 1.0,
        "selected_branch": result.selected_branch,
        "selected_nodes": list(result.accepted_node_indices),
        "terminal": result.terminal,
        "draft_token_ids": draft_token_ids,
        "target_top1_ids": list(_RUNTIME.target_top1_ids),
        "output_token_ids": list(result.output_token_ids),
        "accepted_tokens": accepted_tokens,
        "output_tokens": output_tokens,
        "nodes": nodes,
    }
    construction = getattr(_RUNTIME, "dynamic_construction", None)
    if construction is not None:
        record["dynamic_construction"] = construction
    if diagnostics:
        for node in record["nodes"]:
            node.update(by_node.get(int(node["node"]), {}))
        record["surviving_paths"] = int(getattr(result, "surviving_paths", 0))
        record["selected_path_score"] = getattr(result, "selected_path_score", None)
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    trace_limit = int(os.getenv("REMTP_TREE_TRACE_MAX_ROUNDS", "128"))
    if trace_enabled and _RUNTIME.rounds <= trace_limit:
        trace = format_round_trace(record)
        print(trace, flush=True)
        trace_path = os.getenv("REMTP_TREE_TRACE_LOG")
        if trace_path:
            trace_output = Path(trace_path)
            trace_output.parent.mkdir(parents=True, exist_ok=True)
            with trace_output.open("a", encoding="utf-8") as handle:
                handle.write(trace + "\n")


def runtime_summary() -> dict[str, Any]:
    return {
        "topology": _RUNTIME.topology.name,
        "rounds": _RUNTIME.rounds,
        "mean_accepted_depth": (
            _RUNTIME.accepted_drafts_total / _RUNTIME.rounds
            if _RUNTIME.rounds
            else 0.0
        ),
        "mean_acceptance_length": (
            1.0 + _RUNTIME.accepted_drafts_total / _RUNTIME.rounds
            if _RUNTIME.rounds
            else 0.0
        ),
        "target_forward_calls": _RUNTIME.target_calls,
        "target_validation_nodes": _RUNTIME.target_nodes,
        "target_forward_seconds": _RUNTIME.target_forward_seconds,
        "mean_target_forward_seconds": (
            _RUNTIME.target_forward_seconds / _RUNTIME.target_calls
            if _RUNTIME.target_calls
            else 0.0
        ),
        "nodes_per_round": (
            _RUNTIME.target_nodes / _RUNTIME.rounds if _RUNTIME.rounds else 0.0
        ),
    }


def install_fixed6_microtree() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    topology = get_topology(os.getenv("REMTP_TREE_TOPOLOGY", "6-chain"))
    if (
        topology.name != "6-chain"
        and os.getenv("REMTP_ALLOW_UNVERIFIED_GDN_TREE") != "1"
    ):
        raise RuntimeError(
            "The Qwen3.5 non-chain correctness/performance gate has not passed. "
            "Use 6-chain, or set REMTP_ALLOW_UNVERIFIED_GDN_TREE=1 only "
            "for state-validation/debugging; never for benchmark claims."
        )
    _RUNTIME.topology = topology

    eagle = importlib.import_module("vllm.v1.spec_decode.eagle")
    runner_module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    rejection = importlib.import_module("vllm.v1.sample.rejection_sampler")
    qwen = importlib.import_module("vllm.model_executor.models.qwen3_next")
    qwen35 = importlib.import_module("vllm.model_executor.models.qwen3_5")

    proposer_cls = eagle.EagleProposer
    _propose_with_sampling_metadata._fixed6_original = proposer_cls.propose
    proposer_cls.propose = _propose_with_sampling_metadata
    proposer_cls.propose_tree = _fixed6_propose_tree
    _prepare_selected_path_inputs._fixed6_original = (
        proposer_cls.prepare_inputs_padded
    )
    proposer_cls.prepare_inputs_padded = _prepare_selected_path_inputs

    runner_cls = runner_module.GPUModelRunner
    _prepare_target_tree_inputs._fixed6_original = runner_cls._prepare_inputs
    runner_cls._prepare_inputs = _prepare_target_tree_inputs
    _capture_target_slots._fixed6_original = runner_cls._get_slot_mappings
    runner_cls._get_slot_mappings = _capture_target_slots
    _compact_selected_path._fixed6_original = (
        runner_cls._update_states_after_model_execute
    )
    runner_cls._update_states_after_model_execute = _compact_selected_path

    sampler_cls = rejection.RejectionSampler
    _strict_tree_sampler._fixed6_original = sampler_cls.forward
    sampler_cls.forward = _strict_tree_sampler

    gdn_cls = qwen.Qwen3NextGatedDeltaNet
    _fixed6_gdn_forward_core._fixed6_original = gdn_cls._forward_core
    gdn_cls._forward_core = _fixed6_gdn_forward_core

    target_cls = qwen35.Qwen3_5ForConditionalGeneration
    _profile_target_forward._fixed6_original = target_cls.forward
    target_cls.forward = _profile_target_forward

    _INSTALLED = True
    print(
        "[Fixed-6] installed experimental strict micro-tree "
        f"topology={topology.name} choices={topology_config_string(topology)}",
        flush=True,
    )
