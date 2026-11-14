"""Opt-in, unbounded per-round JSONL log for every speculative method.

Set ``REMTP_ROUND_LOG_JSONL=<path>`` to enable.  One line per verification
round, appended with line buffering.  The logger never draws random numbers
and never changes a verifier input or output; it only reads tensors the
verifier already received or returned.

Chain methods (Native, Cactus, SpecCascade) all reach vLLM's
``rejection_sample``; the logger wraps the outermost installed version, so it
sees the processed (temperature-applied) target logits before any method's
own transform.  It records, per drafted position, the drafted token, target
probability ``p(y)``, MTP probability ``q(y)``, the target and draft top-1
token and probability, and the emitted tokens.  Method-specific relaxed
probabilities are not recorded for chain methods (Cactus's ``h(y)`` is a
deterministic function of ``p(y)`` and delta and can be recomputed offline).

The dynamic tree (RAVEN) wraps ``fixed6_vllm.strict_tree_verify`` after the
dynamic dispatch is installed and records the verifier's own per-node
diagnostics (``p``, ``q``, relaxed ``h``, the uniform draw, the adaptive plan,
correction-hit flags), node tokens and paths, and the target top-1 per row.

Rows carry a wall-clock ``t`` so they can be joined to requests by time when
the server runs one request at a time.  Observed runs pay one small extra
device-to-host copy per round; quantify that before comparing throughput.
"""
from __future__ import annotations

import functools
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

_INSTALLED = False
_STREAM = None
_COUNTERS = {"chain": 0, "tree": 0}
_FAILED = False


def _stream():
    global _STREAM
    if _STREAM is None:
        path = Path(os.environ["REMTP_ROUND_LOG_JSONL"])
        path.parent.mkdir(parents=True, exist_ok=True)
        _STREAM = path.open("a", buffering=1)
    return _STREAM


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _write(kind: str, record: dict, t: float | None = None) -> None:
    record = dict(kind=kind, round=_COUNTERS[kind], t=time.time() if t is None else t, **record)
    _COUNTERS[kind] += 1
    _stream().write(json.dumps(_plain(record), separators=(",", ":")) + "\n")


def _guard(function):
    """Logging must never break generation: disable itself after one failure."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        global _FAILED
        if _FAILED:
            return None
        try:
            return function(*args, **kwargs)
        except Exception as error:  # pragma: no cover - defensive
            _FAILED = True
            print(f"[ReMTP][RoundLog] disabled after error: {type(error).__name__}: {error}", flush=True)
            return None
    return wrapper


def chain_inputs(draft_token_ids: torch.Tensor, draft_probs: torch.Tensor | None,
                 target_logits: torch.Tensor) -> torch.Tensor:
    """Per drafted token: p(y), p_top, p_top_id, q(y), q_top, q_top_id (device tensor)."""
    ids = draft_token_ids.to(torch.int64)
    rows = torch.arange(ids.numel(), device=target_logits.device)
    target = torch.softmax(target_logits.to(torch.float32), dim=-1)
    p_top, p_id = target.max(dim=-1)
    columns = [target[rows, ids], p_top, p_id.to(torch.float32)]
    if draft_probs is not None:
        draft = draft_probs.to(torch.float32)
        q_top, q_id = draft.max(dim=-1)
        columns += [draft[rows, ids], q_top, q_id.to(torch.float32)]
    return torch.stack(columns, dim=-1)


_PENDING = None


def flush_pending() -> None:
    """Write the previous chain round once its asynchronous copy has completed."""
    global _PENDING
    pending, _PENDING = _PENDING, None
    if pending is None:
        return
    host, event, writer, meta = pending
    event.synchronize()
    writer(host.tolist(), **meta)


def _queue(flat: torch.Tensor, writer, meta: dict) -> None:
    """Non-blocking copy + CUDA event; CPU tensors are written immediately."""
    global _PENDING
    flush_pending()
    if not flat.is_cuda:
        writer(flat.tolist(), **meta)
        return
    host = torch.empty(flat.shape, dtype=flat.dtype, pin_memory=True)
    host.copy_(flat, non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    _PENDING = (host, event, writer, meta)


@_guard
def log_chain_round(draft_token_ids, num_draft_tokens, cu_num_draft_tokens, packed, output_token_ids) -> None:
    """Queue one non-blocking device-to-host copy; never insert a sync point.

    vLLM overlaps the next step's launches with the rejection sampler; a
    blocking ``.cpu()`` here would serialise them.  The record is written at
    the next round (or at exit) after the copy's CUDA event has completed.
    Token ids < 2**24 are exact in float32.
    """
    flat = torch.cat((packed.reshape(-1), draft_token_ids.to(packed.dtype).reshape(-1),
                      output_token_ids.to(packed.dtype).reshape(-1)))
    meta = dict(width=packed.shape[1], count=draft_token_ids.numel(), num_draft_tokens=list(num_draft_tokens),
                columns=output_token_ids.shape[1], batch=output_token_ids.shape[0], t=time.time())
    _queue(flat, _write_chain, meta)


def _write_chain(flat, *, width, count, num_draft_tokens, columns, batch, t) -> None:
    table = [flat[i * width:(i + 1) * width] for i in range(count)]
    drafts = [int(v) for v in flat[count * width:count * width + count]]
    tail = [int(v) for v in flat[count * width + count:]]
    outputs = [tail[i * columns:(i + 1) * columns] for i in range(batch)]
    start = 0
    for request_index, per_request in enumerate(num_draft_tokens):
        rows = table[start:start + per_request]
        emitted = [token for token in outputs[request_index] if token != -1]
        record = dict(batch_index=request_index, k=per_request, draft=drafts[start:start + per_request],
                      p=[row[0] for row in rows], p_top=[row[1] for row in rows],
                      p_top_id=[int(row[2]) for row in rows], emitted=emitted,
                      accepted=max(len(emitted) - 1, 0))
        if rows and len(rows[0]) > 3:
            record.update(q=[row[3] for row in rows], q_top=[row[4] for row in rows],
                          q_top_id=[int(row[5]) for row in rows])
        _write("chain", record, t=t)
        start += per_request


@_guard
def log_tree_round(args, result) -> None:
    """Same asynchronous pattern: device values are copied without a sync point."""
    from remtp import fixed6_vllm as fixed
    topology, ids, _draft, target = args[:4]
    runtime = fixed._RUNTIME
    top = target.max(dim=-1)
    rows = top.values.numel()
    flat = torch.cat((top.values.to(torch.float32), top.indices.to(torch.float32),
                      ids.to(device=top.values.device, dtype=torch.float32).reshape(-1)))
    history = getattr(runtime, "generated_history", ())
    meta = dict(rows=rows, t=time.time(), fields=dict(
        request=str(getattr(runtime, "diagnostic_request", "")),
        generated_length=len(history) if history is not None else None,
        node_paths=[list(node.path) for node in topology.nodes],
        emitted=list(result.output_token_ids),
        accepted_nodes=list(result.accepted_node_indices),
        terminal=result.terminal,
        statuses=[list(item) for item in result.evaluated_node_statuses],
        nodes=list(result.node_diagnostics),
    ))
    _queue(flat, _write_tree, meta)


def _write_tree(flat, *, rows, t, fields) -> None:
    record = dict(fields)
    record.update(target_top_prob=flat[:rows], target_top_id=[int(v) for v in flat[rows:2 * rows]],
                  node_tokens=[int(v) for v in flat[2 * rows:]])
    _write("tree", record, t=t)


def install_round_logger() -> None:
    """Install after every method hook so the wrappers are outermost."""
    global _INSTALLED
    if _INSTALLED or not os.environ.get("REMTP_ROUND_LOG_JSONL"):
        return
    import vllm.v1.sample.rejection_sampler as module

    original = module.rejection_sample

    @functools.wraps(original)
    def logged_rejection_sample(draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
                                draft_probs, target_logits, bonus_token_ids, sampling_metadata):
        packed = None
        if not _FAILED:
            try:
                packed = chain_inputs(draft_token_ids, draft_probs, target_logits)
            except Exception:  # pragma: no cover - defensive
                packed = None
        output = original(draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
                          draft_probs, target_logits, bonus_token_ids, sampling_metadata)
        if packed is not None:
            log_chain_round(draft_token_ids, num_draft_tokens, cu_num_draft_tokens, packed, output)
        return output

    logged_rejection_sample._remtp_round_logger = True
    module.rejection_sample = logged_rejection_sample

    from remtp import fixed6_vllm as fixed
    verify = fixed.strict_tree_verify

    @functools.wraps(verify)
    def logged_tree_verify(*args, **kwargs):
        result = verify(*args, **kwargs)
        log_tree_round(args, result)
        return result

    logged_tree_verify._remtp_round_logger = True
    fixed.strict_tree_verify = logged_tree_verify
    import atexit
    atexit.register(_guard(flush_pending))
    _INSTALLED = True
    print(f"[ReMTP][RoundLog] per-round JSONL enabled: {os.environ['REMTP_ROUND_LOG_JSONL']}", flush=True)
