"""Opt-in, read-only probe: native qq_bias tree attention vs the canonical layout.

Enabled only when ``REMTP_TREE_ATTENTION_PROBE_JSONL`` names an output file.
For every ``tree_attn.unified_attention`` call that carries a tree bias it
records shapes.  For calls the canonical layout would intercept (one request,
``rows == nodes + 1``) it additionally recomputes the same attention through
``canonical_tree_table`` into a scratch buffer and logs the max abs difference
and whether the bias equals the topology ancestor mask.  The served output is
never modified; this is a diagnostic, not a performance configuration.
"""
import json
import os


def install_tree_attention_probe(path: str, limit: int = 400) -> None:
    import torch
    from vllm.v1.attention.backends import tree_attn

    original = tree_attn.unified_attention
    if getattr(original, "_remtp_probe", False):
        return
    state = {"written": 0, "calls": 0}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(record):
        if state["written"] >= limit:
            return
        state["written"] += 1
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def probe(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            q = kwargs.get("q")
            bias = kwargs.get("qq_bias")
            state["calls"] += 1
            if q is None or bias is None or args:
                return result
            from remtp.fixed6_vllm import _RUNTIME
            from remtp.canonical_tree_attention import canonical_tree_table

            nodes = len(_RUNTIME.topology.nodes)
            record = dict(
                call=state["calls"], pid=os.getpid(), rows=int(q.shape[0]), nodes=nodes,
                target_tree_active=bool(_RUNTIME.target_tree_active), bias_shape=list(bias.shape),
                max_seqlen_q=int(kwargs.get("max_seqlen_q", -1)), max_seqlen_k=int(kwargs["max_seqlen_k"]),
                block_size=int(kwargs["k"].shape[1]), seqused=kwargs["seqused_k"].tolist()[:3],
            )
            intercepted = (
                _RUNTIME.target_tree_active and q.shape[0] == nodes + 1
                and kwargs["block_table"].shape[0] == 1 and bias.numel() > 0
            )
            if intercepted:
                k, v, table = kwargs["k"], kwargs["v"], kwargs["block_table"]
                context = int(kwargs["max_seqlen_k"]) - int(q.shape[0])
                token_table, lengths, starts, maximum = canonical_tree_table(
                    table, context, _RUNTIME.topology.choices, k.shape[1])
                scratch = torch.empty_like(kwargs["out"])
                mapped = dict(
                    kwargs, k=k.view(-1, 1, k.shape[2], k.shape[3]), v=v.view(-1, 1, v.shape[2], v.shape[3]),
                    block_table=token_table, seqused_k=lengths, cu_seqlens_q=starts, max_seqlen_q=1,
                    max_seqlen_k=maximum, qq_bias=None, seq_threshold_3D=None,
                    num_par_softmax_segments=None, out=scratch,
                )
                original(**mapped)
                native = kwargs["out"].float()
                difference = (native - scratch.float()).abs()
                mask = _RUNTIME.topology.causal_tree_mask(include_anchor=True).to(bias.device)
                allowed = bias == 0
                record.update(
                    context=context, max_abs_diff=difference.max().item(),
                    mean_abs_out=native.abs().mean().item(),
                    rows_with_diff=int((difference.flatten(1).max(1).values > 0).sum().item()),
                    bias_equals_topology_mask=bool((allowed == mask).all().item()),
                    bias_allowed=int(allowed.sum().item()), mask_allowed=int(mask.sum().item()),
                    cu_seqlens_q=kwargs["cu_seqlens_q"].tolist()[:4],
                )
            write(record)
        except Exception as exc:  # the probe must never break serving
            write(dict(error=repr(exc)))
        return result

    probe._remtp_probe = True
    tree_attn.unified_attention = probe
    print(f"[ReMTP][TreeAttentionProbe] recording up to {limit} calls to {path}", flush=True)
