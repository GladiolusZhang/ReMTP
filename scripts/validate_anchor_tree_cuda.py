#!/usr/bin/env python3
"""Validate real TreeAttention and KV compaction against independent tensors.

Synthetic BF16 Q/K/V; this checks the backend and slot mapping, not full-model
logit equivalence or answer quality. Run with no model server for memory safety.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from remtp.dynamic_mtp_tree import DynamicTreeTopology, anchor_reusable_topology
from remtp.fixed6_vllm import _copy_cache_slots
from vllm.v1.attention.backends.tree_attn import _get_depth_counts, _prepare_tree_attn_bias
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda"
    rng = torch.Generator(device=device).manual_seed(42)
    topology = DynamicTreeTopology.from_paths([
        (0,), (1,), (2,), (0, 0), (0, 1), (1, 0),
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
    ])
    reduced, kept = anchor_reusable_topology(topology)
    prefix = 19
    count = topology.node_count + 1
    query = torch.randn(count, 8, 128, device=device, dtype=torch.bfloat16, generator=rng)
    cache = torch.randn(2, 4, 16, 2, 128, device=device, dtype=torch.bfloat16, generator=rng)

    def execute(tree, q, kv):
        n = tree.node_count + 1
        output = torch.empty_like(q)
        bias = _prepare_tree_attn_bias(list(tree.choices), _get_depth_counts(list(tree.choices)),
                                       dtype=torch.float32, device=device)
        unified_attention(
            q=q, k=kv[0], v=kv[1], out=output,
            cu_seqlens_q=torch.tensor([0, n], device=device, dtype=torch.int32),
            max_seqlen_q=n, seqused_k=torch.tensor([prefix + n], device=device, dtype=torch.int32),
            max_seqlen_k=prefix + n, softmax_scale=128 ** -0.5, causal=True,
            alibi_slopes=None, qq_bias=bias, window_size=(-1, -1),
            block_table=torch.tensor([[0, 1, 2, 3]], device=device, dtype=torch.int32),
            softcap=0, q_descale=None,
            k_descale=torch.ones(1, 2, device=device), v_descale=torch.ones(1, 2, device=device),
        )
        return output

    full = execute(topology, query, cache)
    keys = cache[0].flatten(0, 1)[:prefix + count].repeat_interleave(4, dim=1).float()
    values = cache[1].flatten(0, 1)[:prefix + count].repeat_interleave(4, dim=1).float()
    allowed = torch.cat((torch.ones(count, prefix, dtype=torch.bool), topology.causal_tree_mask()), dim=1).to(device)
    scores = torch.einsum("qhd,khd->hqk", query.float(), keys) * 128 ** -0.5
    reference = torch.einsum("hqk,khd->qhd", scores.masked_fill(~allowed, -torch.inf).softmax(-1), values)
    torch.testing.assert_close(full.float(), reference, rtol=0.02, atol=0.02)
    rows = [0] + [i + 1 for i in kept]
    pruned_cache = cache.clone()
    pruned_cache.flatten(1, 2)[:, prefix:prefix + len(rows)] = cache.flatten(1, 2)[:, [prefix + i for i in rows]]
    pruned = execute(reduced, query[rows].contiguous(), pruned_cache)
    torch.testing.assert_close(pruned, full[rows], rtol=0.02, atol=0.02)
    # Commit primary and sibling-hit paths using the actual compaction helper.
    cases = 0
    for tree, kv in ((topology, cache), (reduced, pruned_cache)):
        for path in ((1,), (0, 1), (0, 0, 0), (0, 0, 1)):
            node = next(n for n in tree.nodes if n.path == path)
            selected = tree.path_to(node.index)
            sources = torch.tensor([prefix + i + 1 for i in selected], device=device)
            destinations = torch.arange(prefix + 1, prefix + 1 + len(selected), device=device)
            copied = kv.clone()
            expected = kv.flatten(1, 2)[:, sources].clone()
            _copy_cache_slots(SimpleNamespace(kv_cache=[copied]), sources, destinations)
            torch.testing.assert_close(copied.flatten(1, 2)[:, destinations], expected, rtol=0, atol=0)
            cases += 1
    result = {"status": "passed", "dtype": "BF16", "cache_cases": cases,
              "max_abs_backend_vs_reference": float((full.float() - reference).abs().max()),
              "max_abs_pruned_vs_full": float((pruned.float() - full[rows].float()).abs().max()),
              "scope": "synthetic backend/KV mapping, not full-model probability or quality equivalence"}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
