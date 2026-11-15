"""Device-side dynamic tree construction (engineering only, 2026-11-15).

``REMTP_DYNAMIC_TREE_GPU_BUILD=1`` builds the RAVEN proposal tree without a
host round trip between levels.  The proposal rule is unchanged: the same
sampled trunk tokens (request generator, same order), the same
path-seeded branch draws, the same candidate selection (entropy threshold,
sibling rule, per-parent cap) and the same ``residual_coverage`` retention
under the global budget.  Only where the decisions are computed changes:

* every level is computed with fixed shapes (at most 3 depth-1 nodes, 4
  depth-2 nodes, 9 nodes in all for the supported D3/N9 configuration);
  unused rows are masked out of attention and never write draft KV;
* selection replicates ``raven_fast_path.batched_trunk_children`` op for op
  (per-row 1-D reductions, float32 threshold comparison) and resolves
  probability ties in the reference's stable order (value desc, index asc);
* the host reads the finished tree once, at the end of the proposal.

Numerics: decisions use the same float32 tensors as the host path; the
threshold and path values use float64 ``exp``/``log`` on the device instead
of glibc, which can differ in the last bit (a decision can only change if a
value lies within one ulp of a cut).  The MTP layer runs on 3 or 7 rows and
the LM head on 3 or 4 parent rows; cuBLAS gives different bits for a
1-row GEMM than for >= 2 rows, so rounds whose host path ran a 1-row pass can
differ at the ulp level.  ``REMTP_DYNAMIC_TREE_GPU_BUILD_CHECK=<jsonl>``
recomputes every level's selection, branch draw and retention with the host
reference on the same tensors and logs any difference.

Supported configuration (anything else keeps the host path): sampled-primary
``residual_hit_*`` mode, D=3, N=9, 3 children, ``residual_coverage``
allocation, ``adaptive_base_nodes=0``, no anchor pruning, stochastic sampling,
skipped dead draft KV.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from typing import Any

import torch

K_STABLE = 32          # top-k width used to order ties stably
D1, D2, D3 = 3, 4, 7   # depth-1 slots, depth-2 slots, depth-3 slots
PASS2_ROWS, PASS3_ROWS = 3, 7
NEG_INF = float("-inf")


def supported(config: Any) -> tuple[bool, str]:
    checks = [
        (config.support_mode in {"residual_hit_anchor", "residual_hit_strict", "residual_hit_cactus",
                                 "sampled_primary_shadow"}, "support_mode"),
        (config.max_depth == 3 and config.max_nodes == 9 and config.max_children_per_parent == 3, "D3/N9/C3"),
        (config.allocation_mode == "residual_coverage", "allocation_mode"),
        (config.adaptive_base_nodes == 0, "adaptive_base_nodes"),
        (not config.prune_anchor_descendants, "prune_anchor_descendants"),
        (not config.anchor_only_proposal, "anchor_only_proposal"),
        (not config.exact_topk_candidates, "exact_topk_candidates"),
        (os.getenv("REMTP_DYNAMIC_TREE_SKIP_DEAD_DRAFT_KV", "0") == "1", "SKIP_DEAD_DRAFT_KV"),
        (os.getenv("REMTP_DYNAMIC_TREE_FAST_SELECT", "0") == "1", "FAST_SELECT"),
    ]
    for ok, name in checks:
        if not ok:
            return False, name
    return True, ""


# --------------------------------------------------------------------------
# Selection: batched_trunk_children on the device.
# --------------------------------------------------------------------------
def select_children(q_rows: torch.Tensor, sampled: torch.Tensor, *, depth: int, backup_budget: int,
                    config: Any) -> dict[str, torch.Tensor]:
    """Per parent row: sampled trunk first, then up to ``backup_budget`` backups.

    Returns device tensors: ``tokens``/``raw`` [P, 3] (slot 0 = trunk),
    ``count`` [P], ``entropy`` [P] (float32), ``invalid`` [P] and
    ``tie_risk`` [P] (a tie group at the top-k cut may extend past K_STABLE).
    """
    rows, vocabulary = q_rows.shape
    guarded_limit = min(config.max_children_per_parent, vocabulary)
    k = min(guarded_limit + 1, vocabulary)
    width = min(K_STABLE, vocabulary)
    stats, tops, tidx, traw, trunk_raw = [], [], [], [], []
    for row in range(rows):
        p32 = q_rows[row] if q_rows.dtype == torch.float32 else q_rows[row].to(torch.float32)
        # Same op sequence as batched_trunk_children (bitwise-equal scalars).
        clamped = p32.clamp_min(0.0)
        total = clamped.sum()
        q = clamped / total
        q2 = q / q.sum()
        entropy_t = -(q2 * q2.clamp_min(1e-30).log()).sum() / math.log(vocabulary)
        values, indices = torch.topk(q, width, sorted=True)
        # Stable order of the reference: value descending, index ascending.
        by_index, perm = torch.sort(indices, stable=True)
        values_by_index = values[perm]
        order = torch.sort(values_by_index, descending=True, stable=True).indices
        ordered_values = values_by_index[order]
        ordered_indices = by_index[order]
        stats.append(torch.stack((entropy_t, total, p32.min(), ordered_values[k - 1], values[width - 1])))
        tops.append(ordered_values[:k])
        tidx.append(ordered_indices[:k])
        traw.append(p32[ordered_indices[:k]])
        trunk_raw.append(p32[sampled[row].reshape(1)])
    stats = torch.stack(stats)
    top_values = torch.stack(tops)                  # [P, k] float32
    top_ids = torch.stack(tidx)                     # [P, k] int64
    top_raw = torch.stack(traw)                     # [P, k] float32
    trunk_raw = torch.cat(trunk_raw)                # [P]
    entropy32, total, minimum = stats[:, 0], stats[:, 1], stats[:, 2]
    invalid = ~(torch.isfinite(total) & (total > 0.0) & (minimum >= 0.0))
    tie_risk = (stats[:, 3] == stats[:, 4]) & (stats[:, 3] >= float(torch.tensor(config.min_draft_prob,
                                                                                    dtype=torch.float32)))

    entropy = entropy32.double()
    if vocabulary <= 1:
        entropy = torch.zeros_like(entropy)
    q_max = top_values[:, 0].double()
    uncertainty = entropy + config.depth_exploration * (depth / config.max_depth)
    threshold = torch.clamp_min(q_max * torch.exp(-config.threshold_sensitivity * uncertainty),
                                config.min_draft_prob)
    threshold32 = threshold.to(torch.float32)
    values64 = top_values.double()
    eligible = top_values[:, :guarded_limit] >= threshold32[:, None]
    sibling = torch.zeros_like(eligible)
    if guarded_limit > 1:
        rest = values64[:, 1:guarded_limit]
        sibling[:, 1:] = (~eligible[:, 1:]) & (rest >= config.min_draft_prob) & (
            rest >= (q_max * config.min_sibling_ratio)[:, None])
    any_eligible = eligible[:, :1]
    chosen = (eligible | sibling) & any_eligible
    trunk = sampled.reshape(-1)
    backup_ok = chosen & (top_ids[:, :guarded_limit] != trunk[:, None])
    rank = torch.cumsum(backup_ok.to(torch.int64), dim=1)
    take = backup_ok & (rank <= max(backup_budget, 0))
    count = 1 + take.sum(dim=1)
    tokens = torch.zeros(rows, 3, dtype=torch.int64, device=q_rows.device)
    raw = torch.zeros(rows, 3, dtype=torch.float32, device=q_rows.device)
    tokens[:, 0] = trunk
    raw[:, 0] = trunk_raw
    for slot in (1, 2):
        pick = take & (rank == slot)                       # at most one j per row
        has = pick.any(dim=1)
        j = pick.to(torch.int64).argmax(dim=1)
        tokens[:, slot] = torch.where(has, top_ids.gather(1, j[:, None])[:, 0], 0)
        raw[:, slot] = torch.where(has, top_raw.gather(1, j[:, None])[:, 0], 0.0)
    return dict(tokens=tokens, raw=raw, count=count, entropy=entropy32, invalid=invalid, tie_risk=tie_risk,
                threshold=threshold, q_max=q_max)


# --------------------------------------------------------------------------
# Retention: residual_coverage ordering under the global budget.
# --------------------------------------------------------------------------
def level_budget(n_current: torch.Tensor, depth: int, config: Any) -> torch.Tensor:
    """Device version of the host budget (adaptive_base_nodes == 0)."""
    remaining = config.max_nodes - n_current
    level = torch.clamp_min(remaining - (config.max_depth - depth), 0)
    levels_left = config.max_depth - depth + 1
    spread = torch.clamp_min(torch.div(remaining, levels_left, rounding_mode="floor"), 1)
    return torch.where(remaining <= 0, torch.zeros_like(level), torch.minimum(level, spread))


def retain_level(*, parent_valid: torch.Tensor, parent_code: torch.Tensor, parent_logq: torch.Tensor,
                 selection: dict[str, torch.Tensor], n_current: torch.Tensor, depth: int, slots: int,
                 config: Any) -> dict[str, torch.Tensor]:
    """Keep the level's candidates in the host order; outputs sorted by path code."""
    device = parent_code.device
    parents = parent_code.shape[0]
    rank = torch.arange(3, device=device)
    valid = parent_valid[:, None] & (rank[None, :] < selection["count"][:, None])
    code = parent_code[:, None] * 4 + rank[None, :]
    logq = parent_logq[:, None] + torch.log(torch.clamp_min(selection["raw"].double(), 1e-30))
    value = torch.exp(logq) * math.exp(config.depth_reward * depth / config.max_depth)
    trunk_parent = (parent_code == 0)[:, None].expand(parents, 3)
    klass = torch.where(trunk_parent, 0, torch.where(rank[None, :] == 0, 1, 2))
    order_rank = torch.where(trunk_parent, rank[None, :].expand(parents, 3), 0)
    v, c, r, s = (t.reshape(-1) for t in (valid, klass, order_rank, code))
    val = value.reshape(-1)
    # before[i, j]: candidate j precedes candidate i in the host sort key
    # (class, rank for trunk-parent children, -value, path).
    ci, cj = c[:, None], c[None, :]
    ri, rj = r[:, None], r[None, :]
    vi, vj = val[:, None], val[None, :]
    si, sj = s[:, None], s[None, :]
    before = (cj < ci) | ((cj == ci) & ((rj < ri) | ((rj == ri) & ((vj > vi) | ((vj == vi) & (sj < si))))))
    position = (before & v[None, :]).sum(dim=1)
    budget = level_budget(n_current, depth, config)
    keep = v & (position < budget)
    out_slot = (keep[None, :] & (sj < si)).sum(dim=1)          # path order among the kept
    flat = dict(token=selection["tokens"].reshape(-1), raw=selection["raw"].reshape(-1), code=s,
                logq=logq.reshape(-1), parent=torch.arange(parents, device=device)[:, None].expand(parents, 3).reshape(-1))
    target = torch.where(keep, out_slot, slots)                 # dropped candidates go to a spill slot
    result = {}
    for name, tensor in flat.items():
        buffer = torch.zeros(slots + 1, dtype=tensor.dtype, device=device)
        buffer.scatter_(0, target, tensor)
        result[name] = buffer[:slots]
    result["count"] = keep.sum()
    return result


# --------------------------------------------------------------------------
# CUDA graphs for the fixed-shape decision steps (hundreds of tiny ops each;
# eager launches would make the proposal host-bound on Jetson).
# --------------------------------------------------------------------------
def _level(q: torch.Tensor, sampled: torch.Tensor, parent_valid: torch.Tensor, parent_code: torch.Tensor,
           parent_logq: torch.Tensor, n_current: torch.Tensor, *, depth: int, slots: int, backup_budget: int,
           config: Any) -> dict[str, torch.Tensor]:
    selection = select_children(q, sampled, depth=depth, backup_budget=backup_budget, config=config)
    out = {f"sel_{k}": v for k, v in selection.items()}
    if slots:
        retained = retain_level(parent_valid=parent_valid, parent_code=parent_code, parent_logq=parent_logq,
                                selection=selection, n_current=n_current, depth=depth, slots=slots, config=config)
        out.update({f"ret_{k}": v for k, v in retained.items()})
    return out


class _GraphedLevels:
    """Capture ``_level`` once per (depth, rows); replay with inputs copied into static buffers.

    Pure tensor function, fixed shapes, no host reads: replaying the graph runs the
    same kernels on the same values as the eager call.
    """

    def __init__(self, fn: Any = None) -> None:
        self.fn = _level if fn is None else fn
        self.entries: dict[tuple[Any, ...], tuple[Any, dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {}

    def __call__(self, inputs: dict[str, torch.Tensor], **options: Any) -> dict[str, torch.Tensor]:
        key = (options["depth"], options["slots"], options["backup_budget"], options.get("mode"),
               tuple(inputs["q"].shape))
        entry = self.entries.get(key)
        if entry is None:
            static = {name: value.clone() for name, value in inputs.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.fn(**static, **options)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self.fn(**static, **options)
            entry = (graph, static, outputs)
            self.entries[key] = entry
        graph, static, outputs = entry
        for name, value in inputs.items():
            static[name].copy_(value)
        graph.replay()
        return outputs


_GRAPHS = _GraphedLevels()
_FUSED_GRAPHS: _GraphedLevels | None = None


def run_level(inputs: dict[str, torch.Tensor], **options: Any) -> dict[str, torch.Tensor]:
    global _FUSED_GRAPHS
    from remtp import gpu_tree_fused as fused

    graphs = os.getenv("REMTP_DYNAMIC_TREE_GPU_BUILD_GRAPHS", "1") == "1"
    mode = fused.stats_mode()
    if mode:
        # REMTP_DYNAMIC_TREE_GPU_BUILD_FUSED=exact|fast: same outputs, a few kernels.
        if not graphs:
            return fused.fused_level(**inputs, **options, mode=mode)
        if _FUSED_GRAPHS is None:
            _FUSED_GRAPHS = _GraphedLevels(fn=fused.fused_level)
        return _FUSED_GRAPHS(inputs, **options, mode=mode)
    if graphs:
        return _GRAPHS(inputs, **options)
    return _level(**inputs, **options)


def _split(outputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    selection = {k[4:]: v for k, v in outputs.items() if k.startswith("sel_")}
    retained = {k[4:]: v for k, v in outputs.items() if k.startswith("ret_")}
    return selection, retained


# --------------------------------------------------------------------------
# Branch draws: path-seeded exponential races, pre-generated per possible path.
# --------------------------------------------------------------------------
def branch_seed(initial_seed: int, proposal_round: int, parent_path: tuple[int, ...]) -> int:
    path_hash = 1469598103934665603
    for value in parent_path:
        path_hash ^= int(value) + 1
        path_hash = (path_hash * 1099511628211) & ((1 << 63) - 1)
    return (initial_seed ^ (int(proposal_round) * 0x9E3779B185EBCA87) ^ path_hash) & ((1 << 63) - 1)


def branch_noise(paths: list[tuple[int, ...]], initial_seed: int, proposal_round: int, vocabulary: int,
                 device: torch.device) -> torch.Tensor:
    """Noise rows equal to ``sample_branch_q_token_fast``'s for each parent path."""
    noise = torch.empty(len(paths), vocabulary, dtype=torch.float32, device=device)
    for row, path in enumerate(paths):
        generator = torch.Generator(device=device)
        generator.manual_seed(branch_seed(initial_seed, proposal_round, path))
        noise[row].exponential_(generator=generator)
    return noise


DEPTH2_BRANCH_PATHS = [(1,), (2,)]
DEPTH3_BRANCH_PATHS = [(a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)]
# depth-2 path code (4a + b) -> row of the depth-3 branch noise bank
_CODE_TO_BANK3 = [0] * 12
for _row, (_a, _b) in enumerate(DEPTH3_BRANCH_PATHS):
    _CODE_TO_BANK3[4 * _a + _b] = _row


_TABLES: dict[str, torch.Tensor] = {}


def _device_table(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _TABLES:
        _TABLES[key] = torch.tensor(_CODE_TO_BANK3, dtype=torch.int64, device=device)
    return _TABLES[key]


# --------------------------------------------------------------------------
# Fixed-shape cumulative MTP pass.
# --------------------------------------------------------------------------
def run_pass(proposer: Any, *, tokens: torch.Tensor, hidden: torch.Tensor, depths: torch.Tensor,
             parent_row: torch.Tensor, n_real: torch.Tensor, root_positions: torch.Tensor,
             base_common_metadata: Any, draft_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``fixed6_vllm._run_draft_tree_nodes`` for a fixed row count ``M``.

    Real rows (``< n_real``) occupy the same rows, positions and KV slots as in
    the host path; padded rows write no KV (slot -1) and are masked out of the
    real rows' attention.  ``parent_row`` is the in-pass row of each row's
    parent (-1 for none).
    """
    from vllm.forward_context import set_forward_context
    from remtp import fixed6_vllm as fixed

    builder = proposer.draft_attn_groups[0].get_metadata_builder()
    rows = tokens.shape[0]
    device = tokens.device
    index = torch.arange(rows, device=device)
    real = index < n_real
    # Node mask: self plus in-pass ancestor (depth <= 2 here); root/prefix is context.
    allowed = (index[:, None] == index[None, :]) | (parent_row[:, None] == index[None, :])
    full = torch.full((rows + 1, rows + 1), NEG_INF, dtype=torch.float32, device=device)
    full[1:, 1:] = torch.where(allowed, 0.0, NEG_INF)
    full[:, 0].fill_(0.0)              # fill_, not a Python-scalar write (which syncs)
    builder.tree_attn_bias = full
    builder.reorder_batch_threshold = rows + 1

    root = root_positions.reshape(-1)[:1]
    base = base_common_metadata
    common = replace(
        base, query_start_loc=rows * proposer.arange[:2],
        query_start_loc_cpu=torch.tensor([0, rows], dtype=torch.int32),
        seq_lens=(root + 1 + rows).to(base.seq_lens.dtype).expand_as(base.seq_lens),
        max_seq_len=proposer.max_model_len, num_actual_tokens=rows, max_query_len=rows,
        _seq_lens_cpu=None, _num_computed_tokens_cpu=None,
    )
    attn_metadata = builder.build_for_drafting(common_attn_metadata=common, draft_index=draft_index)
    attn_metadata.tree_attn_bias = fixed._compact_draft_tree_bias(attn_metadata.tree_attn_bias)
    attn_metadata.max_seq_len = min(attn_metadata.max_seq_len, proposer.max_model_len)
    physical = (root + 1 + index).view(1, -1).to(torch.int64)
    block_size = builder.kv_cache_spec.block_size
    block_ids = attn_metadata.block_table.gather(dim=1, index=physical // block_size)
    slots = (block_ids * block_size + physical % block_size).reshape(-1)
    attn_metadata.slot_mapping = torch.where(real, slots, torch.full_like(slots, -1))

    proposer.input_ids[:rows] = tokens.to(proposer.input_ids.dtype)
    proposer.hidden_states[:rows] = hidden.to(proposer.hidden_states.dtype)
    positions = root.to(torch.int64) + depths
    proposer._set_positions(rows, positions.to(proposer._get_positions(rows).dtype))
    per_layer = {name: attn_metadata for group in proposer.draft_attn_groups for name in group.layer_names}
    cudagraph_mode, batch_desc = proposer.cudagraph_dispatcher.dispatch(rows)
    num_input_tokens = batch_desc.num_tokens
    with set_forward_context(per_layer, proposer.vllm_config, num_tokens=num_input_tokens,
                             cudagraph_runtime_mode=cudagraph_mode,
                             slot_mapping=proposer._get_slot_mapping(num_input_tokens, attn_metadata.slot_mapping)):
        output = proposer.model(input_ids=proposer.input_ids[:num_input_tokens],
                                positions=proposer._get_positions(num_input_tokens),
                                hidden_states=proposer.hidden_states[:num_input_tokens], inputs_embeds=None)
        if proposer.model_returns_tuple():
            last_hidden, hidden_out = output
        else:
            last_hidden = hidden_out = output
    return last_hidden[:rows], hidden_out[:rows]


# --------------------------------------------------------------------------
# Host reference for the check mode.
# --------------------------------------------------------------------------
def host_retained_paths(proposals: list[tuple[float, tuple[int, ...]]], n_current: int, depth: int,
                        entropies: list[float], config: Any) -> list[tuple[int, ...]]:
    """The host loop's retention step, verbatim, for given (value, path) proposals."""
    from remtp.dynamic_mtp_tree import adaptive_node_limit, level_expansion_budget, level_expansion_order

    if config.max_nodes - n_current <= 0 or not proposals:
        return []
    active_node_limit = adaptive_node_limit(entropies, config)
    active_config = replace(config, max_nodes=active_node_limit)
    budget = level_expansion_budget(current_nodes=n_current, next_depth=depth, config=active_config)
    remaining_levels = config.max_depth - depth + 1
    active_remaining = active_node_limit - n_current
    budget = min(budget, max(active_remaining // remaining_levels, 1))
    if budget <= 0:
        return []
    order = level_expansion_order(tuple(proposals), config=config)
    trunk_path = (0,) * depth
    trunk_index = next(i for i, item in enumerate(proposals) if item[1] == trunk_path)
    indices = [trunk_index] + [i for i in order if i != trunk_index]
    return [proposals[i][1] for i in indices[:budget]]


class Checker:
    """Recompute each level on the host from the same tensors and log differences."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.levels = 0
        self.mismatches = 0

    def log(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def level(self, *, depth: int, q_rows: torch.Tensor, sampled: torch.Tensor, n_parents: int,
              parent_paths: list[tuple[int, ...]], parent_logq: list[float], selection: dict[str, torch.Tensor],
              retained: dict[str, torch.Tensor] | None, n_current: int, backup_budget: int, config: Any,
              sampling_metadata: Any, proposal_round: int, extra: dict[str, Any] | None = None) -> None:
        from remtp import raven_fast_path as fast
        from remtp.dynamic_mtp_tree import path_expansion_value

        self.levels += 1
        issues = []
        rows = [q_rows[r] for r in range(n_parents)]
        for r in range(1, n_parents):
            reference = fast.sample_branch_q_token_fast(rows[r], sampling_metadata, parent_path=parent_paths[r],
                                                        proposal_round=proposal_round)
            if int(reference[0]) != int(sampled[r]):
                issues.append(f"branch draw row {r}")
        host = fast.batched_trunk_children(rows, [sampled[r] for r in range(n_parents)], depth=depth,
                                           backup_budget=backup_budget, config=config)
        tokens = selection["tokens"].tolist()
        raws = selection["raw"].tolist()
        counts = selection["count"].tolist()
        proposals, entropies = [], []
        for r, item in enumerate(host):
            ids = [int(x) for x in item.candidate_ids.tolist()]
            if ids != tokens[r][:counts[r]] or item.candidate_probs != raws[r][:counts[r]]:
                issues.append(f"selection row {r}: host {ids} {item.candidate_probs} device "
                              f"{tokens[r][:counts[r]]} {raws[r][:counts[r]]}")
            entropies.append(item.entropy)
            for rank, probability in enumerate(item.candidate_probs):
                path_log = parent_logq[r] + math.log(max(probability, 1e-30))
                proposals.append((path_expansion_value(path_log, depth, config), parent_paths[r] + (rank,)))
        if retained is not None:
            host_paths = sorted(host_retained_paths(proposals, n_current, depth, entropies, config))
            count = int(retained["count"])
            codes = retained["code"].tolist()[:count]
            device_paths = [tuple(int(c) // 4 ** (depth - 1 - i) % 4 for i in range(depth)) for c in codes]
            if host_paths != device_paths:
                issues.append(f"retention: host {host_paths} device {device_paths}")
        if issues:
            self.mismatches += 1
            self.log(dict(depth=depth, issues=issues, **(extra or {})))


_CHECKER: Checker | None = None


def checker() -> Checker | None:
    global _CHECKER
    path = os.getenv("REMTP_DYNAMIC_TREE_GPU_BUILD_CHECK", "")
    if path and _CHECKER is None:
        _CHECKER = Checker(path)
    return _CHECKER


# --------------------------------------------------------------------------
# The proposal.
# --------------------------------------------------------------------------
def propose_tree(proposer: Any, *, logits: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor,
                 common_attn_metadata: Any, sampling_metadata: Any, config: Any, proposal_round: int) -> list[torch.Tensor]:
    from remtp import fixed6_vllm as fixed
    from remtp import raven_fast_path as fast
    from remtp.dynamic_mtp_tree import DynamicTreeTopology, mtp_layer_for_tree_depth
    from remtp.mimo_mtp import force_mimo_mtp_layer

    check = checker()
    device = logits.device
    route = os.getenv("REMTP_DYNAMIC_TREE_MTP_ROUTE", "physical_012")
    request_generator = (getattr(sampling_metadata, "generators", None) or {}).get(0)
    initial_seed = int(request_generator.initial_seed()) if request_generator is not None else 0

    # ---- depth 1: root candidates ------------------------------------------------
    root_q = fixed._temperature_probs(logits[:1], sampling_metadata)
    vocabulary = root_q.shape[-1]
    sampled_root = fast.sample_q_token_fast(root_q[0], sampling_metadata)
    root_budget = min(config.max_children_per_parent - 1, max(config.max_nodes - config.max_depth, 0))
    zero = torch.zeros((), dtype=torch.int64, device=device)
    sel1, _ = _split(run_level(
        dict(q=root_q, sampled=sampled_root.reshape(1), parent_valid=torch.ones(1, dtype=torch.bool, device=device),
             parent_code=zero.reshape(1), parent_logq=torch.zeros(1, dtype=torch.float64, device=device),
             n_current=zero),
        depth=1, slots=0, backup_budget=root_budget, config=config))
    n1 = sel1["count"][0]
    tok1 = sel1["tokens"][0]                                   # [3]
    logq1 = torch.log(torch.clamp_min(sel1["raw"][0].double(), 1e-30))
    valid1 = torch.arange(D1, device=device) < n1
    if check is not None:
        check.level(depth=1, q_rows=root_q, sampled=sampled_root.reshape(1), n_parents=1, parent_paths=[()],
                    parent_logq=[0.0], selection=sel1, retained=None, n_current=0, backup_budget=root_budget,
                    config=config, sampling_metadata=sampling_metadata, proposal_round=proposal_round)
    root_hidden = hidden_states.reshape(1, -1)[0]
    # Branch noise for both levels (independent generators; no request-RNG use).
    noise2 = branch_noise(DEPTH2_BRANCH_PATHS, initial_seed, proposal_round, vocabulary, device)
    noise3 = branch_noise(DEPTH3_BRANCH_PATHS, initial_seed, proposal_round, vocabulary, device)

    # ---- depth 2 --------------------------------------------------------------------
    with force_mimo_mtp_layer(mtp_layer_for_tree_depth(2, route)):
        last2, hid2 = run_pass(
            proposer, tokens=tok1, hidden=root_hidden.expand(PASS2_ROWS, -1),
            depths=torch.ones(PASS2_ROWS, dtype=torch.int64, device=device),
            parent_row=torch.full((PASS2_ROWS,), -1, dtype=torch.int64, device=device), n_real=n1,
            root_positions=positions, base_common_metadata=common_attn_metadata, draft_index=1)
    q2 = fixed._temperature_probs(proposer.model.compute_logits(last2), sampling_metadata)   # [3, V]
    trunk2 = fast.sample_q_token_fast(q2[0], sampling_metadata)
    branch2 = (q2[1:] / noise2).argmax(dim=-1)
    sampled2 = torch.cat((trunk2.reshape(1), branch2))
    sel2, ret2 = _split(run_level(
        dict(q=q2, sampled=sampled2, parent_valid=valid1, parent_code=torch.arange(D1, device=device),
             parent_logq=logq1, n_current=n1),
        depth=2, slots=D2, backup_budget=config.max_children_per_parent - 1, config=config))
    n2 = ret2["count"]
    if check is not None:
        # Host path logs from the transferred float32 probabilities, as the host loop computes them.
        n1_host = int(n1)
        host_logq = {(r,): math.log(max(p, 1e-30)) for r, p in enumerate(sel1["raw"][0].tolist()[:n1_host])}
        check.level(depth=2, q_rows=q2, sampled=sampled2, n_parents=n1_host,
                    parent_paths=[(r,) for r in range(n1_host)],
                    parent_logq=[host_logq[(r,)] for r in range(n1_host)],
                    selection=sel2, retained=ret2, n_current=n1_host,
                    backup_budget=config.max_children_per_parent - 1, config=config,
                    sampling_metadata=sampling_metadata, proposal_round=proposal_round)

    # ---- depth 3: pass over depth-1 rows then depth-2 rows ---------------------------
    rows3 = torch.arange(PASS3_ROWS, device=device)
    is_d1 = rows3 < n1
    d2_slot = torch.clamp(rows3 - n1, 0, D2 - 1)
    tok3_in = torch.where(is_d1, tok1[torch.clamp(rows3, max=D1 - 1)], ret2["token"][d2_slot])
    hid_d2 = hid2.index_select(0, ret2["parent"][d2_slot])
    hid3_in = torch.where(is_d1[:, None], root_hidden[None, :].expand(PASS3_ROWS, -1), hid_d2)
    depths3 = torch.where(is_d1, 1, 2).to(torch.int64)
    parent3 = torch.where(is_d1, -1, ret2["parent"][d2_slot])
    n12 = n1 + n2
    with force_mimo_mtp_layer(mtp_layer_for_tree_depth(3, route)):
        last3, _ = run_pass(
            proposer, tokens=tok3_in, hidden=hid3_in, depths=depths3, parent_row=parent3, n_real=n12,
            root_positions=positions, base_common_metadata=common_attn_metadata, draft_index=2)
    parent_rows3 = torch.clamp(n1 + torch.arange(D2, device=device), max=PASS3_ROWS - 1)
    q3 = fixed._temperature_probs(proposer.model.compute_logits(last3.index_select(0, parent_rows3)),
                                  sampling_metadata)                                         # [4, V]
    trunk3 = fast.sample_q_token_fast(q3[0], sampling_metadata)
    bank_rows = _device_table(device)[ret2["code"][1:]]
    branch3 = (q3[1:] / noise3.index_select(0, bank_rows)).argmax(dim=-1)
    sampled3 = torch.cat((trunk3.reshape(1), branch3))
    valid2 = torch.arange(D2, device=device) < n2
    sel3, ret3 = _split(run_level(
        dict(q=q3, sampled=sampled3, parent_valid=valid2, parent_code=ret2["code"], parent_logq=ret2["logq"],
             n_current=n12),
        depth=3, slots=D3, backup_budget=config.max_children_per_parent - 1, config=config))
    n3 = ret3["count"]
    if check is not None:
        n2_host = int(n2)
        codes2 = ret2["code"].tolist()[:n2_host]
        paths2 = [(c // 4, c % 4) for c in codes2]
        raws2 = ret2["raw"].tolist()[:n2_host]
        logq2 = [host_logq[(path[0],)] + math.log(max(raw, 1e-30)) for path, raw in zip(paths2, raws2)]
        check.level(depth=3, q_rows=q3, sampled=sampled3, n_parents=n2_host,
                    parent_paths=paths2, parent_logq=logq2,
                    selection=sel3, retained=ret3, n_current=int(n12),
                    backup_budget=config.max_children_per_parent - 1, config=config,
                    sampling_metadata=sampling_metadata, proposal_round=proposal_round)

    # ---- assemble in BFS order and read the tree once --------------------------------
    level_valid = torch.cat((valid1, valid2, torch.arange(D3, device=device) < n3))
    tokens_all = torch.cat((tok1, ret2["token"], ret3["token"]))
    codes_all = torch.cat((torch.arange(D1, device=device), ret2["code"], ret3["code"]))
    bank_all = torch.cat((torch.zeros(D1, dtype=torch.int64, device=device), 1 + ret2["parent"],
                          1 + D1 + ret3["parent"]))
    position = torch.cumsum(level_valid.to(torch.int64), 0) - 1
    target = torch.where(level_valid, position, config.max_nodes)
    def pack(values: torch.Tensor) -> torch.Tensor:
        buffer = torch.zeros(config.max_nodes + 1, dtype=values.dtype, device=device)
        buffer.scatter_(0, target, values)
        return buffer[:config.max_nodes]
    tokens_bfs = pack(tokens_all)
    bank_bfs = pack(bank_all)
    q_bank = torch.cat((root_q, q2, q3))                       # [1 + 3 + 4, V]
    probs_bfs = q_bank.index_select(0, bank_bfs)               # [9, V]
    from remtp import async_tree
    if async_tree.enabled():
        # No host read: the tree stays in fixed-capacity device tensors (2026-11-15).
        parent_all = torch.cat((torch.full((D1,), -1, dtype=torch.int64, device=device), ret2["parent"],
                                n1 + ret3["parent"]))
        ids = async_tree.publish_tree(n1=n1, n2=n2, n3=n3, tokens_bfs=tokens_bfs, codes_bfs=pack(codes_all),
                                      parent_all=parent_all, target_for_pack=pack, probs_bfs=probs_bfs)
        fixed._RUNTIME.topology = async_tree.capacity_topology()
        fixed._RUNTIME.draft_slots = None
        fixed._RUNTIME.draft_token_ids = async_tree.STATE.tokens
        fixed._RUNTIME.draft_probs = async_tree.STATE.probs
        fixed._RUNTIME.dynamic_construction = None
        return [ids]
    invalid = torch.cat((sel1["invalid"][:1], sel2["invalid"] & valid1, sel3["invalid"] & valid2)).any()
    tie_risk = torch.cat((sel1["tie_risk"][:1], sel2["tie_risk"] & valid1, sel3["tie_risk"] & valid2)).any()
    header = torch.stack((n1, n2, n3, invalid.to(torch.int64), tie_risk.to(torch.int64)))
    host = torch.cat((header, pack(codes_all))).tolist()      # the one host read of the proposal
    n1_h, n2_h, n3_h, invalid_h, tie_h = (int(x) for x in host[:5])
    if invalid_h:
        raise ValueError("Invalid MTP proposal Q: nonfinite, negative, or zero mass")
    if tie_h and check is not None:
        check.log(dict(tie_risk=True, proposal_round=proposal_round))
    codes = [int(x) for x in host[5:]]
    count = n1_h + n2_h + n3_h
    depths_bfs = [1] * n1_h + [2] * n2_h + [3] * n3_h
    paths = [tuple(code // 4 ** (depth - 1 - i) % 4 for i in range(depth))
             for code, depth in zip(codes[:count], depths_bfs)]
    topology = DynamicTreeTopology.from_paths(tuple(paths))
    if tuple(node.path for node in topology.nodes) != tuple(paths):
        raise RuntimeError(f"device tree is not in BFS order: {paths}")
    fixed._RUNTIME.topology = topology
    fixed._RUNTIME.draft_slots = None
    ids = tokens_bfs[:count].to(torch.int32)
    fixed._RUNTIME.draft_token_ids = ids.to(torch.int64)
    fixed._RUNTIME.draft_probs = probs_bfs[:count].contiguous()
    fixed._RUNTIME.dynamic_construction = {
        "config": config.__dict__, "mtp_route": route, "node_count": count, "unpruned_node_count": count,
        "pruned_descendants": 0, "max_depth": topology.max_depth, "device_built": True,
        "branching": [{"parent": None, "depth": 1, "forced_top1": False}],
    }
    bounds = [(0, n1_h), (n1_h, n1_h + n2_h), (n1_h + n2_h, count)]
    return [ids[a:b].clone().view(1, -1) for a, b in bounds if b > a]
